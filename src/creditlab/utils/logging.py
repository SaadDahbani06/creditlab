"""Standard-library logging setup shared by the CLI and modules."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(level)
    # Third-party libraries are noisy at INFO.
    for name in ("matplotlib", "urllib3", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
