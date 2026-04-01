"""Structured logging for dataset_svc.

Provides a pre-configured logger used by all modules in the service.
"""

import logging
import sys

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)

def get_logger(name: str = "dataset_svc") -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger name, typically the module's __name__.

    Returns:
        A logging.Logger configured with structured formatting.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
