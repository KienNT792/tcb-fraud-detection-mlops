from __future__ import annotations

import hashlib
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from infrastructure.pipeline import PIPELINE_CONFIG

_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _APP_DIR.parent.parent
_ML_PIPELINE_SRC = _PROJECT_ROOT / "ml_pipeline" / "src"

if str(_ML_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE_SRC))

from inference import FraudDetector  # noqa: E402

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class CanaryRolloutManager:
    """Manages stable/candidate detectors and weighted canary traffic routing."""

    def __init__(
        self,
        stable_models_dir: str,
        stable_processed_dir: str,
        config_path: str | None = None,
    ) -> None:
        self._stable_models_dir = str(Path(stable_models_dir))
        self._stable_processed_dir = str(Path(stable_processed_dir))
        self._candidate_models_dir: str | None = None
        self._candidate_processed_dir: str | None = None
        self._config_path = Path(config_path or PIPELINE_CONFIG.serving.rollout_config_path)
        self._lock = RLock()

        self._stable_detector = FraudDetector(self._stable_models_dir, self._stable_processed_dir)
        self._candidate_detector: FraudDetector | None = None
        self._state = self._load_state()
        self._load_candidate_if_needed()

    def get_stable_detector(self) -> FraudDetector:
        return self._stable_detector

    def health_check(self) -> dict[str, Any]:
        stable_health = self._stable_detector.health_check()
        candidate_health = self._candidate_detector.health_check() if self._candidate_detector else None
        return {
            "stable": stable_health,
            "candidate": candidate_health,
            "rollout": self.get_rollout_status(),
        }

    def get_rollout_status(self) -> dict[str, Any]:
        with self._lock:
            state = deepcopy(self._state)

        stable_info = {
            "models_dir": self._stable_models_dir,
            "processed_dir": self._stable_processed_dir,
            "model_version": self._stable_detector.health_check()["model_version"],
            "loaded": True,
        }
        candidate_info: dict[str, Any] | None = None
        if self._candidate_detector is not None:
            candidate_health = self._candidate_detector.health_check()
            candidate_info = {
                "models_dir": self._candidate_models_dir,
                "processed_dir": self._candidate_processed_dir,
                "model_version": candidate_health["model_version"],
                "loaded": True,
            }
        elif state["candidate_models_dir"]:
            candidate_info = {
                "models_dir": state["candidate_models_dir"],
                "processed_dir": state["candidate_processed_dir"],
                "model_version": None,
                "loaded": False,
            }

        return {
            "config_path": str(self._config_path),
            "enabled": bool(state["enabled"]),
            "status": state["status"],
            "traffic_percent_candidate": int(state["traffic_percent_candidate"]),
            "rollout_steps": list(state["rollout_steps"]),
            "current_step_index": int(state["current_step_index"]),
            "step_interval_minutes": int(state["step_interval_minutes"]),
            "auto_advance": bool(state["auto_advance"]),
            "auto_promote_when_complete": bool(state["auto_promote_when_complete"]),
            "last_transition_at": state["last_transition_at"],
            "next_step_at": state["next_step_at"],
            "stable": stable_info,
            "candidate": candidate_info,
        }

    def update_rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            updated = deepcopy(self._state)
            for field in (
                "enabled",
                "traffic_percent_candidate",
                "rollout_steps",
                "current_step_index",
                "step_interval_minutes",
                "auto_advance",
                "auto_promote_when_complete",
                "candidate_models_dir",
                "candidate_processed_dir",
                "status",
            ):
                if field in payload and payload[field] is not None:
                    updated[field] = payload[field]

            updated["rollout_steps"] = self._normalize_steps(updated["rollout_steps"])
            updated["traffic_percent_candidate"] = self._normalize_percent(
                updated["traffic_percent_candidate"]
            )

            if updated["enabled"] and updated["traffic_percent_candidate"] == 0:
                updated["traffic_percent_candidate"] = updated["rollout_steps"][0]

            updated["current_step_index"] = self._resolve_step_index(
                updated["rollout_steps"],
                updated["traffic_percent_candidate"],
            )

            now = utcnow().isoformat()
            updated["last_transition_at"] = payload.get("last_transition_at") or now
            updated["next_step_at"] = self._compute_next_step_at(updated)
            self._state = updated
            self._load_candidate_if_needed()
            self._save_state()
        return self.get_rollout_status()

    def advance_rollout(self) -> dict[str, Any]:
        with self._lock:
            self._advance_locked()
            self._save_state()
        return self.get_rollout_status()

    def pause_rollout(self) -> dict[str, Any]:
        with self._lock:
            self._state["auto_advance"] = False
            self._state["status"] = "paused"
            self._state["enabled"] = (
                self._candidate_detector is not None
                and int(self._state["traffic_percent_candidate"]) > 0
            )
            self._state["last_transition_at"] = utcnow().isoformat()
            self._state["next_step_at"] = None
            self._save_state()
        return self.get_rollout_status()

    def rollback_rollout(self) -> dict[str, Any]:
        with self._lock:
            self._candidate_detector = None
            self._state["enabled"] = False
            self._state["auto_advance"] = False
            self._state["traffic_percent_candidate"] = 0
            self._state["current_step_index"] = -1
            self._state["status"] = "rolled_back"
            self._state["last_transition_at"] = utcnow().isoformat()
            self._state["next_step_at"] = None
            self._save_state()
        return self.get_rollout_status()

    def promote_candidate(self) -> dict[str, Any]:
        with self._lock:
            if self._candidate_detector is None:
                raise RuntimeError("No candidate model is loaded for promotion.")

            self._stable_detector = self._candidate_detector
            self._stable_models_dir = self._state["candidate_models_dir"]
            self._stable_processed_dir = self._state["candidate_processed_dir"]
            self._candidate_detector = None

            self._state["enabled"] = False
            self._state["traffic_percent_candidate"] = 0
            self._state["current_step_index"] = -1
            self._state["status"] = "promoted"
            self._state["last_transition_at"] = utcnow().isoformat()
            self._state["next_step_at"] = None
            self._state["candidate_models_dir"] = str(PIPELINE_CONFIG.serving.candidate_models_dir)
            self._state["candidate_processed_dir"] = str(PIPELINE_CONFIG.serving.candidate_processed_dir)
            self._save_state()
        return self.get_rollout_status()

    def route_single(self, transaction: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        with self._lock:
            self._maybe_advance_locked()
            lane = self._select_lane(str(transaction.get("transaction_id", "unknown")))
            detector = self._candidate_detector if lane == "candidate" else self._stable_detector

        result = detector.predict_single(transaction)
        return lane, result

    def route_batch(self, df: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        with self._lock:
            self._maybe_advance_locked()
            routes = {
                index: self._select_lane(str(row.get("transaction_id", index)))
                for index, row in df.iterrows()
            }
            stable_detector = self._stable_detector
            candidate_detector = self._candidate_detector

        stable_indices = [idx for idx, lane in routes.items() if lane == "stable"]
        candidate_indices = [idx for idx, lane in routes.items() if lane == "candidate"]
        frames: list[pd.DataFrame] = []

        if stable_indices:
            stable_frame = stable_detector.predict_batch(df.loc[stable_indices].copy())
            stable_frame["threshold"] = float(stable_detector._threshold)
            stable_frame["served_by"] = "stable"
            frames.append(stable_frame)
        if candidate_indices and candidate_detector is not None:
            candidate_frame = candidate_detector.predict_batch(df.loc[candidate_indices].copy())
            candidate_frame["threshold"] = float(candidate_detector._threshold)
            candidate_frame["served_by"] = "candidate"
            frames.append(candidate_frame)

        if frames:
            result = pd.concat(frames).sort_index()
        else:
            result = stable_detector.predict_batch(df.copy())
            result["threshold"] = float(stable_detector._threshold)
            result["served_by"] = "stable"
        return {
            "stable": len(stable_indices),
            "candidate": len(candidate_indices),
        }, result

    def _load_state(self) -> dict[str, Any]:
        default_state = {
            "enabled": False,
            "status": "idle",
            "traffic_percent_candidate": 0,
            "rollout_steps": list(PIPELINE_CONFIG.serving.rollout_steps),
            "current_step_index": -1,
            "step_interval_minutes": PIPELINE_CONFIG.serving.rollout_step_interval_minutes,
            "auto_advance": PIPELINE_CONFIG.serving.rollout_auto_advance,
            "auto_promote_when_complete": PIPELINE_CONFIG.serving.rollout_auto_promote,
            "last_transition_at": None,
            "next_step_at": None,
            "candidate_models_dir": str(PIPELINE_CONFIG.serving.candidate_models_dir),
            "candidate_processed_dir": str(PIPELINE_CONFIG.serving.candidate_processed_dir),
        }
        if not self._config_path.exists():
            return default_state

        with open(self._config_path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        default_state.update(loaded)
        default_state["rollout_steps"] = self._normalize_steps(default_state["rollout_steps"])
        default_state["traffic_percent_candidate"] = self._normalize_percent(
            default_state["traffic_percent_candidate"]
        )
        default_state["current_step_index"] = self._resolve_step_index(
            default_state["rollout_steps"],
            default_state["traffic_percent_candidate"],
        )
        return default_state

    def _save_state(self) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2)

    def _load_candidate_if_needed(self) -> None:
        if not self._state["enabled"]:
            self._candidate_detector = None
            self._candidate_models_dir = None
            self._candidate_processed_dir = None
            return

        candidate_models_dir = Path(self._state["candidate_models_dir"])
        candidate_processed_dir = Path(self._state["candidate_processed_dir"])
        if not (candidate_models_dir / "xgb_fraud_model.joblib").exists():
            raise FileNotFoundError(
                f"Candidate model artifact not found in {candidate_models_dir}. "
                "Upload the candidate model before enabling rollout."
            )

        if self._candidate_detector is not None:
            current_models_dir = self._candidate_models_dir
            current_processed_dir = self._candidate_processed_dir
            if current_models_dir == str(candidate_models_dir) and current_processed_dir == str(candidate_processed_dir):
                return

        logger.info(
            "Loading candidate detector — models_dir=%s | processed_dir=%s",
            candidate_models_dir,
            candidate_processed_dir,
        )
        self._candidate_detector = FraudDetector(
            str(candidate_models_dir),
            str(candidate_processed_dir),
        )
        self._candidate_models_dir = str(candidate_models_dir)
        self._candidate_processed_dir = str(candidate_processed_dir)

    def _normalize_steps(self, steps: list[int]) -> list[int]:
        normalized = sorted({self._normalize_percent(int(step)) for step in steps if int(step) > 0})
        return normalized or [10, 25, 50, 100]

    def _normalize_percent(self, value: int) -> int:
        return max(0, min(int(value), 100))

    def _resolve_step_index(self, steps: list[int], traffic_percent: int) -> int:
        for idx, step in enumerate(steps):
            if step == traffic_percent:
                return idx
        return -1

    def _compute_next_step_at(self, state: dict[str, Any]) -> str | None:
        if not state["enabled"] or not state["auto_advance"]:
            return None

        current_step_index = state["current_step_index"]
        if current_step_index >= len(state["rollout_steps"]) - 1:
            return None

        last_transition_at = _parse_dt(state["last_transition_at"]) or utcnow()
        next_dt = last_transition_at + timedelta(minutes=int(state["step_interval_minutes"]))
        return next_dt.isoformat()

    def _advance_locked(self) -> None:
        if self._candidate_detector is None:
            raise RuntimeError("Cannot advance rollout without a loaded candidate model.")

        next_idx = self._state["current_step_index"] + 1
        if next_idx >= len(self._state["rollout_steps"]):
            if self._state["auto_promote_when_complete"]:
                self.promote_candidate()
            return

        self._state["enabled"] = True
        self._state["traffic_percent_candidate"] = self._state["rollout_steps"][next_idx]
        self._state["current_step_index"] = next_idx
        self._state["status"] = "running"
        self._state["last_transition_at"] = utcnow().isoformat()
        self._state["next_step_at"] = self._compute_next_step_at(self._state)

        if (
            self._state["traffic_percent_candidate"] == 100
            and self._state["auto_promote_when_complete"]
        ):
            self.promote_candidate()

    def _maybe_advance_locked(self) -> None:
        next_step_at = _parse_dt(self._state["next_step_at"])
        if next_step_at is None:
            return
        if utcnow() >= next_step_at:
            self._advance_locked()
            self._save_state()

    def _select_lane(self, key: str) -> str:
        if (
            not self._state["enabled"]
            or self._candidate_detector is None
            or self._state["traffic_percent_candidate"] <= 0
        ):
            return "stable"

        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
        if bucket < int(self._state["traffic_percent_candidate"]):
            return "candidate"
        return "stable"
