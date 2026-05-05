"""
Structured logging setup.

Named log_utils (not logging) to avoid shadowing Python's stdlib logging module.

Each experiment gets its own .log file so sweep runs can be analysed
individually after the fact.  The root logger is configured so that all
module-level `logging.getLogger(__name__)` calls in the codebase are
automatically captured.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(experiment_name: str, log_dir: str | Path = "logs") -> None:
    """
    Configure the root logger with a console handler and a per-experiment
    file handler.  Call once per experiment run before any other code runs.

    Clearing existing handlers prevents duplicate log lines when multiple
    experiments are run in the same Python process (sweep mode).

    Args:
        experiment_name: Used as the log filename stem.  Typically the full
                         experiment ID: "{model}_{domain}_{method}".
        log_dir:         Directory for .log files.  Created if absent.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    fmt      = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
    datefmt  = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers — prevents duplicate lines in sweep mode
    if root.hasHandlers():
        root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Per-experiment file handler
    log_file = log_path / f"{experiment_name}.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logging.info(f"Logging initialised — experiment: {experiment_name}")
    logging.info(f"Log file: {log_file}")