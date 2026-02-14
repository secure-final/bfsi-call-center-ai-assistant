"""Centralized logging setup. No sensitive data in logs."""
import logging
import os
import sys
from pathlib import Path

from src.config import get_logging_config, load_config, PROJECT_ROOT


def setup_logging(
    level: str | None = None,
    log_file: Path | None = None,
    log_sensitive: bool = False,
) -> None:
    """Configure root logger. Never log PII or sensitive customer data."""
    cfg = load_config()
    log_cfg = get_logging_config(cfg)
    lvl = level or os.getenv("LOG_LEVEL") or log_cfg.get("level", "INFO")
    fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(
        level=getattr(logging, lvl.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module. Ensure no PII in messages."""
    return logging.getLogger(name)
