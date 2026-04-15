"""
processing/preprocessing.py — Enrich rollouts with group_id and sample_weight.

_resolve_group_id   : deterministic hash of the prompt (or rollout_id fallback)
_resolve_sample_weight : always 1.0 unless future policy overrides it
"""

from __future__ import annotations

import hashlib
from typing import List

from schemas.rollout_schema import Rollout
from utils.logging import get_logger

log = get_logger("processing.preprocess")


def _resolve_group_id(rollout: Rollout) -> str:
    """
    Deterministic group identifier — SHA-256 of the prompt text, truncated
    to 8 hex chars.  Rollouts sharing the same prompt fall in the same GRPO
    group.  Falls back to the rollout_id hash if prompt is empty.
    """
    source = rollout.prompt.strip() or rollout.rollout_id
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return digest[:8]


def _resolve_sample_weight(rollout: Rollout) -> float:  # noqa: ARG001
    """Default sample weight — 1.0 for every rollout."""
    return 1.0


def preprocess(rollouts: List[Rollout]) -> List[Rollout]:
    """Attach group_id and sample_weight to each rollout in-place."""
    for r in rollouts:
        r.group_id = _resolve_group_id(r)
        r.sample_weight = _resolve_sample_weight(r)
        log.debug(
            "Preprocessed rollout",
            extra={"rollout_id": r.rollout_id, "group_id": r.group_id},
        )
    log.info("Preprocessing complete", extra={"count": len(rollouts)})
    return rollouts
