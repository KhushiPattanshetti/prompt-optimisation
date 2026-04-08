"""
log_utils.py – Explainable logging helpers and aggregate stats tracker.

Per-request debug helpers  (spec §15.3)
  log_coverage_debug(gt, pred, label) – GT codes with lowest coverage
  log_extra_debug(pred, gt, label)    – Pred codes with lowest match score

Aggregate tracker  (spec §15.4)
  update_aggregate_stats(reward, delta_d) – append + emit every 10 samples
"""

import logging
import statistics
from typing import List

from similarity import depth, sim

logger = logging.getLogger("reward_metrics_svc.log_utils")

# Module-level history – accumulates across the process lifetime
_rewards_history: List[float] = []
_delta_d_history: List[float] = []


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats  (spec §15.4)
# ─────────────────────────────────────────────────────────────────────────────


def update_aggregate_stats(reward: float, delta_d: float = 0.0) -> None:
    """Append *reward* / *delta_d* and emit aggregate stats every 10 samples."""
    _rewards_history.append(reward)
    _delta_d_history.append(delta_d)
    if len(_rewards_history) % 10 == 0:
        _log_aggregate()


def _log_aggregate() -> None:
    n = len(_rewards_history)
    if n == 0:
        return
    avg_r = statistics.mean(_rewards_history)
    pct_pos = 100.0 * sum(1 for r in _rewards_history if r > 0) / n
    pct_neg = 100.0 * sum(1 for r in _rewards_history if r < 0) / n
    pct_zer = 100.0 - pct_pos - pct_neg
    avg_dD = statistics.mean(_delta_d_history) if _delta_d_history else 0.0
    logger.info(
        "AGGREGATE | n=%-4d  avg_reward=%.4f  pos=%.1f%%  neg=%.1f%%  zero=%.1f%%"
        "  avg_ΔD=%.4f",
        n,
        avg_r,
        pct_pos,
        pct_neg,
        pct_zer,
        avg_dD,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-request debug helpers  (spec §15.3)
# ─────────────────────────────────────────────────────────────────────────────


def log_coverage_debug(gt: List[str], pred: List[str], label: str = "") -> None:
    """Log the 3 GT codes with the lowest coverage score (worst covered first)."""
    if not gt or not pred:
        return
    pairs = [(g, max(sim(g, p) for p in pred), depth(g)) for g in gt]
    pairs.sort(key=lambda x: x[1])  # ascending = worst first
    for g, score, d in pairs[:3]:
        logger.debug(
            "COVERAGE_DEBUG [%s] | gt=%-8s  depth=%-2d  coverage_score=%.4f",
            label,
            g,
            d,
            score,
        )


def log_extra_debug(pred: List[str], gt: List[str], label: str = "") -> None:
    """Log the 3 predicted codes with the lowest match score (most spurious first)."""
    if not pred or not gt:
        return
    pairs = [(p, max(sim(p, g) for g in gt), depth(p)) for p in pred]
    pairs.sort(key=lambda x: x[1])  # ascending = worst first
    for p, score, d in pairs[:3]:
        logger.debug(
            "EXTRA_DEBUG [%s] | pred=%-8s  depth=%-2d  match_score=%.4f",
            label,
            p,
            d,
            score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-request debug helpers  (spec §14.3)
# ─────────────────────────────────────────────────────────────────────────────


def log_coverage_debug(gt: List[str], pred: List[str], label: str = "") -> None:
    """Log the 3 GT codes with the lowest coverage score (worst covered first)."""
    if not gt or not pred:
        return
    pairs = [(g, max(sim(g, p) for p in pred), depth(g)) for g in gt]
    pairs.sort(key=lambda x: x[1])  # ascending = worst first
    for g, score, d in pairs[:3]:
        logger.debug(
            "COVERAGE_DEBUG [%s] | gt=%-8s  depth=%-2d  coverage_score=%.4f",
            label,
            g,
            d,
            score,
        )


def log_extra_debug(pred: List[str], gt: List[str], label: str = "") -> None:
    """Log the 3 predicted codes with the lowest match score (most spurious first)."""
    if not pred or not gt:
        return
    pairs = [(p, max(sim(p, g) for g in gt), depth(p)) for p in pred]
    pairs.sort(key=lambda x: x[1])  # ascending = worst first
    for p, score, d in pairs[:3]:
        logger.debug(
            "EXTRA_DEBUG [%s] | pred=%-8s  depth=%-2d  match_score=%.4f",
            label,
            p,
            d,
            score,
        )
