"""
processing/reward_processing.py — Guard against double-normalisation.

The spec is explicit:
  "If reward is already normalised → DO NOT normalise again."
  "If config.normalize_rewards = True AND reward is raw → normalise."

This module applies that rule to an entire batch's reward list.
"""

from __future__ import annotations

import math
from typing import List

from config import cfg
from utils.logging import get_logger

log = get_logger("processing.reward")


def process_rewards(rewards: List[float]) -> List[float]:
    """
    Return rewards ready for advantage computation.

    • If reward_already_normalized is True  → pass through unchanged.
    • If normalize_rewards is True          → z-score normalise.
    • Otherwise                             → pass through unchanged.
    """
    if cfg.reward_already_normalized:
        log.info(
            "Rewards already normalised — skipping normalisation",
            extra={"sample_reward": f"{rewards[0]:.4f}" if rewards else "n/a"},
        )
        return rewards

    if cfg.normalize_rewards:
        normalised = _z_score(rewards)
        log.info(
            "Rewards z-score normalised",
            extra={
                "mean_before": f"{_mean(rewards):.4f}",
                "std_before": f"{_std(rewards):.4f}",
            },
        )
        return normalised

    log.info("Rewards passed through without normalisation")
    return rewards


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _z_score(values: List[float]) -> List[float]:
    mean = _mean(values)
    std = _std(values)
    if std < 1e-8:
        log.warning("Reward std ≈ 0 — returning zeros to avoid division by zero")
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]
