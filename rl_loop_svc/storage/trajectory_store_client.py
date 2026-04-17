"""
storage/trajectory_store_client.py — HTTP client for trajectory_store_svc.

Provides a thin wrapper around the trajectory_store_svc REST API so that
rl_loop_svc can query service health and surface it on its own status endpoint.

Rollout data is consumed via the shared filesystem (RolloutLoader reads JSONL
files written by trajectory_store_svc) rather than via HTTP, because the full
prompt text needed for model tokenisation is only available in the JSONL files,
not in the prepared-batch JSON files served by the trajectory store API.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TrajectoryStoreClient:
    """
    Minimal HTTP client for trajectory_store_svc.

    Args:
        base_url: Base URL, e.g. ``http://localhost:8200``.
        timeout:  Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8200",
        timeout: float = 0.25,
        cache_ttl_seconds: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._session = requests.Session()
        self._last_status: Optional[dict] = None
        self._last_status_ts: float = 0.0

    def get_status(self) -> Optional[dict]:
        """
        Fetch ``GET /status`` from trajectory_store_svc.

        Returns the parsed JSON dict on success, or ``None`` if the service
        is unreachable or returns a non-2xx response.
        """
        now = time.monotonic()
        if (
            self._last_status_ts > 0
            and (now - self._last_status_ts) < self.cache_ttl_seconds
        ):
            return self._last_status

        try:
            resp = self._session.get(f"{self.base_url}/status", timeout=self.timeout)
            resp.raise_for_status()
            status = resp.json()
            self._last_status = status
            self._last_status_ts = now
            return status
        except requests.RequestException as exc:
            self._last_status = None
            self._last_status_ts = now
            logger.debug("trajectory_store_svc unreachable: %s", exc)
            return None

    def is_healthy(self) -> bool:
        """Return True when trajectory_store_svc responds with status='ok'."""
        status = self.get_status()
        return status is not None and status.get("status") == "ok"
