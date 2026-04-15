"""
utils/logging.py — Structured, human-readable logger for trajectory_store_svc.

Every log line follows:
  [TIMESTAMP] [LEVEL] [MODULE] message  {key=value ...}

This makes logs trivially grep-able and easy to tail in prod.
"""

import logging
import sys
from typing import Any


# ── Formatter ─────────────────────────────────────────────────────────────────


class _PrettyFormatter(logging.Formatter):
    """Emit one line per record with optional key=value pairs appended."""

    LEVEL_COLOURS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S")
        colour = self.LEVEL_COLOURS.get(record.levelname, "")
        level_tag = f"{colour}[{record.levelname:<8}]{self.RESET}"
        module_tag = f"[{record.name}]"

        msg = record.getMessage()

        # Extra key=value context appended by callers via `extra`
        ctx_items = {
            k: v
            for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__
            and not k.startswith("_")
            and k
            not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "message",
            )
        }
        ctx_str = (
            "  " + "  ".join(f"{k}={v!r}" for k, v in ctx_items.items())
            if ctx_items
            else ""
        )
        line = f"[{ts}] {level_tag} {module_tag}  {msg}{ctx_str}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with pretty console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_PrettyFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


def log_separator(logger: logging.Logger, label: str = "") -> None:
    """Print a visible separator line — useful between pipeline stages."""
    bar = "─" * 60
    logger.info(f"{bar} {label} {bar}" if label else bar)
