"""
Tests for model_loader.py — singleton lifecycle and path resolution.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from serving_api.app import model_loader


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton is cleared before and after each test."""
    model_loader.unload_model()
    yield
    model_loader.unload_model()


class TestGetDetector:
    def test_raises_if_not_loaded(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            model_loader.get_detector()

    def test_returns_instance_after_load(self):
        mock_detector = MagicMock()
        with patch("serving_api.app.model_loader.FraudDetector", return_value=mock_detector):
            model_loader.load_model()
            result = model_loader.get_detector()
        assert result is mock_detector


class TestLoadModel:
    def test_loads_once(self):
        mock_detector = MagicMock()
        with patch("serving_api.app.model_loader.FraudDetector", return_value=mock_detector) as mock_cls:
            model_loader.load_model()
            model_loader.load_model()   # second call — should NOT re-instantiate
        assert mock_cls.call_count == 1

    def test_returns_cached_on_second_call(self):
        mock_detector = MagicMock()
        with patch("serving_api.app.model_loader.FraudDetector", return_value=mock_detector):
            first  = model_loader.load_model()
            second = model_loader.load_model()
        assert first is second

    def test_propagates_file_not_found(self):
        with patch(
            "serving_api.app.model_loader.FraudDetector",
            side_effect=FileNotFoundError("artifact missing"),
        ):
            with pytest.raises(FileNotFoundError, match="artifact missing"):
                model_loader.load_model()


class TestUnloadModel:
    def test_unload_clears_singleton(self):
        mock_detector = MagicMock()
        with patch("serving_api.app.model_loader.FraudDetector", return_value=mock_detector):
            model_loader.load_model()
        model_loader.unload_model()
        with pytest.raises(RuntimeError):
            model_loader.get_detector()

    def test_unload_idempotent(self):
        """Calling unload when already unloaded should not raise."""
        model_loader.unload_model()
        model_loader.unload_model()  # should not raise
