"""
main.py – Entry point for reward_metrics_svc.

Usage
-----
  # Development / standalone (recommended)
  python -m reward_metrics_svc.main

  # Production (ASGI server directly)
  uvicorn reward_metrics_svc.app:app --host 0.0.0.0 --port 8002 --log-level debug
"""

import logging

from .config import setup_logging

# Configure logging before importing any other project module so that
# all eager-load log messages (tree build etc.) are captured.
setup_logging(level=logging.DEBUG)

from .app import app  # noqa: E402  (must come after logging setup)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reward_metrics_svc.app:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="debug",
    )
