"""Separate rotating logs for ETL, model training and application events."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import psutil

CHANNELS = {
    "etl": "pipeline",
    "training": "model_training",
    "application": "application",
}
LOG = logging.getLogger("greenforge.etl")
_SETUP_LOCK = threading.RLock()
_RUN_ID = None
_RUN_PID = None


def setup_logging(directory: Path | None = None, *, level=logging.INFO,
                  max_bytes=10 * 1024 * 1024, backup_count=5, new_run=False):
    """Configure all channels once per call; call before starting worker threads.

    Each process gets timestamped files; repeated setup reuses its run ID.
    Pass new_run=True for another run in a notebook or long-lived process.
    Files are opened only when their channel emits its first message.
    Rotation limits each run; earlier runs are retained.
    """
    global _RUN_ID, _RUN_PID
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parents[3] / "logs"
    if max_bytes <= 0 or backup_count < 1:
        raise ValueError("Log rotation requires positive max_bytes and backup_count")
    directory.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(name)s | pid=%(process)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    with _SETUP_LOCK:
        if new_run or _RUN_ID is None or _RUN_PID != os.getpid():
            _RUN_PID = os.getpid()
            _RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ") + f"_{_RUN_PID}"
        # Build handlers first so a file-open failure leaves existing logging intact.
        pending = []
        try:
            for channel, folder in CHANNELS.items():
                filename = Path(folder) / f"{_RUN_ID}.log"
                (directory / filename).parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    directory / filename, maxBytes=max_bytes, backupCount=backup_count,
                    encoding="utf-8", errors="backslashreplace", delay=True,
                )
                pending.append((channel, file_handler, logging.StreamHandler()))
        except Exception:
            for _, file_handler, console in pending:
                file_handler.close()
                console.close()
            raise
        for channel, file_handler, console in pending:
            logger = logging.getLogger(f"greenforge.{channel}")
            logger.setLevel(level)
            logger.propagate = False
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
            for handler in (file_handler, console):
                handler.setFormatter(formatter)
                logger.addHandler(handler)


def get_logger(name: str, channel: str | None = None) -> logging.Logger:
    """Route module names automatically, or explicitly select a channel.

    Existing ETL modules can keep using LOG. Training and application entry
    points should call setup_logging() before obtaining their module logger.
    """
    if channel is None:
        parts = name.lower().split(".")
        if any(part in {"training", "train", "training_pipeline", "model_training", "mlflow"}
               or part.startswith("train_") for part in parts):
            channel = "training"
        elif any(part in {"etl", "etl_pipeline", "feature_pipeline", "ingestion", "transformations", "features"}
                 for part in parts):
            channel = "etl"
        else:
            channel = "application"
    if channel not in CHANNELS:
        raise ValueError(f"Unknown logging channel: {channel!r}")
    return logging.getLogger(f"greenforge.{channel}.{name}")


def rss_mib():
    try:
        return psutil.Process().memory_info().rss / 2**20
    except (psutil.Error, OSError):
        return float("nan")


@contextlib.contextmanager
def stage(name, logger: logging.Logger | None = None):
    logger = logger if logger is not None else LOG
    start = time.monotonic()
    logger.info("START %s | RSS=%.1f MiB", name, rss_mib())
    try:
        yield
    except Exception:
        logger.exception("FAILED %s", name)
        raise
    else:
        logger.info("DONE %s | seconds=%.2f | RSS=%.1f MiB", name, time.monotonic() - start, rss_mib())
