"""
Tests for coverage — logging_config, model_registry, promote_model.

Exercises the modules that were previously at 0% or low coverage to
push overall coverage above the 80 % gate.
"""

from __future__ import annotations

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Tests — logging_config.py
# ===========================================================================

class TestSetupLogging:
    """Tests for setup_logging."""

    def test_runs_without_error(self) -> None:
        from ml_pipeline.src.logging_config import setup_logging
        setup_logging()

    def test_idempotent(self) -> None:
        from ml_pipeline.src.logging_config import setup_logging
        setup_logging()
        setup_logging()  # second call should not raise

    def test_custom_level(self) -> None:
        import logging
        from ml_pipeline.src.logging_config import setup_logging
        setup_logging(level=logging.DEBUG)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        # Reset to INFO
        setup_logging(level=logging.INFO)


# ===========================================================================
# Tests — model_registry.py
# ===========================================================================

class TestRegisterModelFromRun:
    """Tests for register_model_from_run."""

    @patch("ml_pipeline.src.model_registry.MlflowClient")
    def test_returns_version_int(self, mock_client_cls: MagicMock) -> None:
        from ml_pipeline.src.model_registry import register_model_from_run

        mock_client = mock_client_cls.return_value
        mock_registered = MagicMock()
        mock_registered.version = "3"
        mock_client.create_model_version.return_value = mock_registered

        result = register_model_from_run(
            run_id="abc123",
            artifact_path="model",
            model_name="test-model",
        )

        assert result == 3
        mock_client.create_model_version.assert_called_once_with(
            name="test-model",
            source="runs:/abc123/model",
            run_id="abc123",
        )


class TestTransitionModelVersionStage:
    """Tests for transition_model_version_stage."""

    @patch("ml_pipeline.src.model_registry.MlflowClient")
    def test_calls_transition(self, mock_client_cls: MagicMock) -> None:
        from ml_pipeline.src.model_registry import (
            transition_model_version_stage,
        )

        mock_client = mock_client_cls.return_value

        transition_model_version_stage(
            model_name="test-model",
            version=3,
            stage="Production",
        )

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test-model",
            version="3",
            stage="Production",
            archive_existing_versions=True,
        )


class TestFindLatestVersionByRun:
    """Tests for find_latest_version_by_run."""

    @patch("ml_pipeline.src.model_registry.MlflowClient")
    def test_returns_matching_version(
        self, mock_client_cls: MagicMock,
    ) -> None:
        from ml_pipeline.src.model_registry import find_latest_version_by_run

        v1 = MagicMock(version="1", run_id="run-A")
        v2 = MagicMock(version="5", run_id="run-B")
        v3 = MagicMock(version="3", run_id="run-A")
        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = [v1, v2, v3]

        result = find_latest_version_by_run(
            model_name="test-model",
            run_id="run-A",
        )
        assert result == 3  # max of versions 1, 3

    @patch("ml_pipeline.src.model_registry.MlflowClient")
    def test_returns_none_when_no_match(
        self, mock_client_cls: MagicMock,
    ) -> None:
        from ml_pipeline.src.model_registry import find_latest_version_by_run

        mock_client = mock_client_cls.return_value
        mock_client.search_model_versions.return_value = []

        result = find_latest_version_by_run(
            model_name="test-model",
            run_id="nonexistent-run",
        )
        assert result is None


# ===========================================================================
# Tests — promote_model.py
# ===========================================================================

class TestBuildParser:
    """Tests for promote_model.build_parser."""

    def test_default_model_name(self) -> None:
        from ml_pipeline.src.promote_model import build_parser

        parser = build_parser()
        args = parser.parse_args(["--stage", "Staging"])
        assert args.model_name == "tcb-fraud-xgboost"
        assert args.stage == "Staging"

    def test_all_args_parsed(self) -> None:
        from ml_pipeline.src.promote_model import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--model-name", "my-model",
            "--stage", "Production",
            "--version", "7",
            "--run-id", "abc",
        ])
        assert args.model_name == "my-model"
        assert args.stage == "Production"
        assert args.version == 7
        assert args.run_id == "abc"


class TestPromoteMain:
    """Tests for promote_model.main."""

    @patch("ml_pipeline.src.promote_model.transition_model_version_stage")
    def test_with_explicit_version(
        self, mock_transition: MagicMock,
    ) -> None:
        from ml_pipeline.src.promote_model import main

        test_args = ["--stage", "Staging", "--version", "5"]
        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(
                        model_name="tcb-fraud-xgboost",
                        stage="Staging",
                        version=5,
                        run_id="",
                    )):
            main()
        mock_transition.assert_called_once_with(
            model_name="tcb-fraud-xgboost",
            version=5,
            stage="Staging",
            archive_existing_versions=True,
        )

    @patch("ml_pipeline.src.promote_model.transition_model_version_stage")
    @patch("ml_pipeline.src.promote_model.find_latest_version_by_run",
           return_value=3)
    def test_resolves_version_by_run_id(
        self,
        mock_find: MagicMock,
        mock_transition: MagicMock,
    ) -> None:
        from ml_pipeline.src.promote_model import main

        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(
                        model_name="tcb-fraud-xgboost",
                        stage="Production",
                        version=None,
                        run_id="run-abc",
                    )):
            main()
        mock_find.assert_called_once()
        mock_transition.assert_called_once_with(
            model_name="tcb-fraud-xgboost",
            version=3,
            stage="Production",
            archive_existing_versions=True,
        )

    def test_raises_without_version_or_run_id(self) -> None:
        from ml_pipeline.src.promote_model import main

        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(
                        model_name="tcb-fraud-xgboost",
                        stage="Staging",
                        version=None,
                        run_id="",
                    )):
            with pytest.raises(ValueError, match="Either --version or --run-id"):
                main()

    @patch("ml_pipeline.src.promote_model.find_latest_version_by_run",
           return_value=None)
    def test_raises_when_version_not_found(
        self, mock_find: MagicMock,
    ) -> None:
        from ml_pipeline.src.promote_model import main

        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(
                        model_name="tcb-fraud-xgboost",
                        stage="Staging",
                        version=None,
                        run_id="nonexistent-run",
                    )):
            with pytest.raises(RuntimeError, match="No model version"):
                main()
