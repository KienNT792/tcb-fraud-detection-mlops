"""
TCB Fraud Detection — Centralised Logging Configuration.

Call ``setup_logging()`` **once** at the application entry point (e.g. the
``if __name__ == "__main__"`` block or the FastAPI lifespan handler).
Individual modules should only call ``logging.getLogger(__name__)``.

Why centralise?
    ``logging.basicConfig()`` is designed to be called once; subsequent
    calls are silently ignored by the Python runtime.  Scattering it across
    every module creates the illusion of configuration without any effect.
"""

from __future__ import annotations

import logging
import sys


_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_DATEFMT: str = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a unified format.

    Safe to call multiple times — idempotent by design.  Uses
    ``force=True`` (Python 3.8+) so the handler is always applied even
    if another library already called ``basicConfig`` before us.

    Args:
        level: Logging level (default ``INFO``).
    """
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
        stream=sys.stderr,
        force=True,
    )
