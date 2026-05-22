"""Centralised logging setup for DALAL TERMINAL."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)
    # third-party noise reduction
    for noisy in ("httpx", "urllib3", "uvicorn.access", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (configures logging on first use)."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
