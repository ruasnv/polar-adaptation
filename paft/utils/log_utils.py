from __future__ import annotations
import logging
import sys
from pathlib import Path


def setup_logging(method_name: str, log_dir: str = "logs"):
    """
    Configures logging to both the console and a file.
    Each experiment gets its own .log file for post-run analysis.
    """
    # Create the logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Format: 2026-05-05 19:30:00 - INFO - [pure_paft] Message
    log_format = "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers (prevents duplicate logs in a sweep)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # 2. File Handler (Persistent record for your thesis data)
    file_handler = logging.FileHandler(log_path / f"{method_name}.log")
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized for experiment: {method_name}")
    logging.info(f"Log file: {log_path / f'{method_name}.log'}")