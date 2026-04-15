"""
pipeline_logger/hash_utils.py

Deterministic, short prompt hashes used throughout the pretty logs.
Reuses the same SHA-256 approach as trajectory_store_svc/processing/preprocessing.py
but exposes a configurable digest length (default 12 hex chars).
"""

from __future__ import annotations

import hashlib


def prompt_hash(text: str, length: int = 12) -> str:
    """Return the first `length` hex chars of SHA-256(text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
