from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DriftSnapshot:
    ready: bool = False
    reference_mode: str = "missing"
    reference_samples: int = 0
    current_samples: int = 0
    features_total: int = 0
    features_alerting: int = 0
    overall_score: float = 0.0
    drift_ratio: float = 0.0
    reason: str = "not_initialized"
    feature_scores: dict[str, float] = field(default_factory=dict)


class DriftMonitor:
    """Rolling-window drift tracker for live inference traffic.

    Preferred reference data source is ``data/processed/train.parquet``.
    If that file is absent, the monitor falls back to using an initial warm-up
    window from live requests as the baseline so monitoring can still run in
    lightweight environments such as Codespaces.
    """

    def __init__(
        self,
        processed_dir: str,
        window_size: int | None = None,
        reference_sample_size: int | None = None,
        warmup_min_samples: int | None = None,
        alert_threshold: float | None = None,
    ) -> None:
        self._processed_dir = Path(processed_dir)
        self._window_size = window_size or int(os.getenv("DRIFT_WINDOW_SIZE", "500"))
        self._reference_sample_size = reference_sample_size or int(
            os.getenv("DRIFT_REFERENCE_SAMPLE_SIZE", "2000")
        )
        self._warmup_min_samples = warmup_min_samples or int(
            os.getenv("DRIFT_WARMUP_MIN_SAMPLES", "100")
        )
        self._alert_threshold = alert_threshold or float(
            os.getenv("DRIFT_ALERT_THRESHOLD", "0.2")
        )

        self._lock = Lock()
        self._current_records: deque[dict[str, float]] = deque(maxlen=self._window_size)
        self._warmup_records: deque[dict[str, float]] = deque(
            maxlen=self._reference_sample_size
        )
        self._reference_df: pd.DataFrame | None = None
        self._feature_cols: list[str] = []
        self._snapshot = DriftSnapshot(reason="waiting_for_reference")

    @property
    def alert_threshold(self) -> float:
        return self._alert_threshold

    def bootstrap(self, detector: Any) -> DriftSnapshot:
        """Load reference data when model artifacts are available."""
        with self._lock:
            if self._reference_df is not None:
                return self._snapshot

            feature_cols = self._extract_feature_cols(detector)
            train_path = self._processed_dir / "train.parquet"

            if train_path.exists():
                try:
                    reference_df = pd.read_parquet(train_path)
                    reference_df = self._select_feature_frame(
                        reference_df,
                        feature_cols=feature_cols,
                    )
                    if not reference_df.empty:
                        self._reference_df = self._sample_reference(reference_df)
                        self._feature_cols = self._reference_df.columns.tolist()
                        self._snapshot = self._build_snapshot(
                            reason="reference_loaded_from_train_parquet"
                        )
                        return self._snapshot
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    logger.warning("Failed to load train.parquet for drift monitoring: %s", exc)

            self._snapshot = self._build_snapshot(reason="waiting_for_warmup_window")
            return self._snapshot

    def observe(self, raw_df: pd.DataFrame, detector: Any) -> DriftSnapshot:
        """Update the current traffic window from raw prediction requests."""
        try:
            feature_frame = self._transform_live_frame(raw_df=raw_df, detector=detector)
        except Exception as exc:
            logger.warning("Skipping drift update for current batch: %s", exc)
            with self._lock:
                self._snapshot = self._build_snapshot(reason="transform_unavailable")
                return self._snapshot

        records = feature_frame.to_dict(orient="records")
        with self._lock:
            for record in records:
                self._current_records.append(record)
                if self._reference_df is None:
                    self._warmup_records.append(record)

            if self._reference_df is None and len(self._warmup_records) >= self._warmup_min_samples:
                warmup_df = pd.DataFrame(list(self._warmup_records))
                warmup_df = self._select_feature_frame(
                    warmup_df,
                    feature_cols=feature_frame.columns.tolist(),
                )
                if not warmup_df.empty:
                    self._reference_df = warmup_df.copy()
                    self._feature_cols = self._reference_df.columns.tolist()

            self._snapshot = self._build_snapshot(reason="updated_from_live_traffic")
            return self._snapshot

    def snapshot(self) -> DriftSnapshot:
        with self._lock:
            return self._snapshot

    def _transform_live_frame(self, raw_df: pd.DataFrame, detector: Any) -> pd.DataFrame:
        transform_fn = getattr(detector, "_transform", None)
        if not callable(transform_fn):
            raise RuntimeError("Detector does not expose a callable _transform().")

        transformed_df = transform_fn(raw_df.copy())
        if not isinstance(transformed_df, pd.DataFrame):
            raise RuntimeError("Detector transform did not return a DataFrame.")

        feature_cols = self._extract_feature_cols(detector)
        feature_frame = self._select_feature_frame(
            transformed_df,
            feature_cols=feature_cols,
        )
        if feature_frame.empty:
            raise RuntimeError("No transformed features available for drift monitoring.")
        return feature_frame

    def _select_feature_frame(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        selected_cols = [col for col in (feature_cols or []) if col in df.columns]
        if not selected_cols:
            numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
            selected_cols = [col for col in numeric_cols if col != "is_fraud"]

        feature_frame = df[selected_cols].copy()
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
        feature_frame = feature_frame.dropna(axis=1, how="all")
        return feature_frame

    def _extract_feature_cols(self, detector: Any) -> list[str]:
        feature_cols = getattr(detector, "_feature_cols", None)
        if isinstance(feature_cols, list):
            return feature_cols
        if isinstance(feature_cols, tuple):
            return list(feature_cols)
        return []

    def _sample_reference(self, reference_df: pd.DataFrame) -> pd.DataFrame:
        if len(reference_df) <= self._reference_sample_size:
            return reference_df.copy()
        return reference_df.sample(
            n=self._reference_sample_size,
            random_state=42,
        ).reset_index(drop=True)

    def _build_snapshot(self, reason: str) -> DriftSnapshot:
        reference_df = self._reference_df
        current_df = pd.DataFrame(list(self._current_records))

        if reference_df is None:
            return DriftSnapshot(
                ready=False,
                reference_mode="missing",
                reference_samples=0,
                current_samples=len(current_df),
                reason=reason,
            )

        if current_df.empty:
            feature_scores = {
                col: 0.0
                for col in reference_df.columns.tolist()
            }
            return DriftSnapshot(
                ready=False,
                reference_mode=self._reference_mode(),
                reference_samples=len(reference_df),
                current_samples=0,
                features_total=len(reference_df.columns),
                reason="waiting_for_current_window",
                feature_scores=feature_scores,
            )

        feature_cols = [col for col in reference_df.columns if col in current_df.columns]
        if not feature_cols:
            feature_scores = {
                col: 0.0
                for col in reference_df.columns.tolist()
            }
            return DriftSnapshot(
                ready=False,
                reference_mode=self._reference_mode(),
                reference_samples=len(reference_df),
                current_samples=len(current_df),
                features_total=len(reference_df.columns),
                reason="feature_overlap_missing",
                feature_scores=feature_scores,
            )

        reference_numeric = reference_df[feature_cols].copy()
        current_numeric = current_df[feature_cols].copy()

        for col in feature_cols:
            fallback = float(reference_numeric[col].median()) if reference_numeric[col].notna().any() else 0.0
            reference_numeric[col] = reference_numeric[col].fillna(fallback)
            current_numeric[col] = current_numeric[col].fillna(fallback)

        feature_scores: dict[str, float] = {}
        for col in feature_cols:
            ref_values = reference_numeric[col].to_numpy(dtype=float)
            cur_values = current_numeric[col].to_numpy(dtype=float)
            if len(ref_values) < 2 or len(cur_values) < 2:
                continue
            feature_scores[col] = round(self._drift_score(ref_values, cur_values), 6)

        features_total = len(feature_scores)
        features_alerting = sum(
            1 for score in feature_scores.values() if score >= self._alert_threshold
        )
        overall_score = float(np.mean(list(feature_scores.values()))) if feature_scores else 0.0
        drift_ratio = (features_alerting / features_total) if features_total else 0.0

        return DriftSnapshot(
            ready=features_total > 0,
            reference_mode=self._reference_mode(),
            reference_samples=len(reference_df),
            current_samples=len(current_df),
            features_total=features_total,
            features_alerting=features_alerting,
            overall_score=round(overall_score, 6),
            drift_ratio=round(drift_ratio, 6),
            reason=reason,
            feature_scores=feature_scores,
        )

    def _reference_mode(self) -> str:
        if self._reference_df is None:
            return "missing"

        train_path = self._processed_dir / "train.parquet"
        if train_path.exists():
            return "train_parquet"
        return "warmup_window"

    def _drift_score(self, reference: np.ndarray, current: np.ndarray) -> float:
        if self._looks_categorical(reference, current):
            return self._categorical_distance(reference, current)
        return self._population_stability_index(reference, current)

    @staticmethod
    def _looks_categorical(reference: np.ndarray, current: np.ndarray) -> bool:
        combined = np.concatenate([reference, current])
        unique_count = len(np.unique(combined))
        looks_integer_like = np.allclose(combined, np.round(combined), atol=1e-9)
        return unique_count <= 12 and looks_integer_like

    @staticmethod
    def _categorical_distance(reference: np.ndarray, current: np.ndarray) -> float:
        ref_series = pd.Series(reference).astype(str)
        cur_series = pd.Series(current).astype(str)
        all_labels = sorted(set(ref_series.unique()).union(cur_series.unique()))
        ref_dist = ref_series.value_counts(normalize=True).reindex(all_labels, fill_value=0.0)
        cur_dist = cur_series.value_counts(normalize=True).reindex(all_labels, fill_value=0.0)
        score = float(0.5 * np.abs(ref_dist - cur_dist).sum())
        return float(min(max(score, 0.0), 1.0))

    @staticmethod
    def _population_stability_index(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(reference, quantiles))
        if len(edges) < 3:
            scale = float(np.std(reference)) or 1.0
            score = float(abs(np.mean(current) - np.mean(reference)) / scale)
            return float(min(max(score, 0.0), 1.0))

        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_hist, _ = np.histogram(reference, bins=edges)
        cur_hist, _ = np.histogram(current, bins=edges)

        ref_dist = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
        cur_dist = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)

        score = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
        return float(min(max(score, 0.0), 1.0))
