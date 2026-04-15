"""
processing/advantage.py — Advantage computation: standard, GRPO, and hybrid.

Public entry-point
──────────────────
compute_advantages(batch) → (advantages: List[float], returns: List[float])

Modes (driven by cfg.advantage_mode)
─────────────────────────────────────
standard  : advantage_i = reward_i - value_estimate_i
grpo      : within-group relative rewards (Group Relative Policy Optimisation)
hybrid    : λ * grpo + (1 - λ) * standard
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from config import cfg
from buffer.rollout_buffer import RolloutBatch
from processing.reward_processing import process_rewards
from utils.logging import get_logger

log = get_logger("processing.advantage")


# ── Public API ────────────────────────────────────────────────────────────────


def compute_advantages(batch: RolloutBatch) -> Tuple[List[float], List[float]]:
    """
    Return (advantages, returns) for the batch.

    Returns are always:  returns_i = reward_i   (no bootstrapping — reward is terminal).
    Advantages depend on cfg.advantage_mode.
    """
    rewards = process_rewards(batch.rewards)
    returns = rewards[:]  # returns == rewards for terminal-step rollouts

    mode = cfg.advantage_mode
    log.info("Computing advantages", extra={"mode": mode, "batch_size": batch.size()})

    if mode == "standard":
        advantages = _compute_standard(rewards, batch)
    elif mode == "grpo":
        advantages = _compute_grpo_relative_rewards(rewards, batch)
    elif mode == "hybrid":
        adv_grpo = _compute_grpo_relative_rewards(rewards, batch)
        adv_std = _compute_standard(rewards, batch)
        lam = cfg.hybrid_advantage_lambda
        advantages = [lam * g + (1 - lam) * s for g, s in zip(adv_grpo, adv_std)]
        log.info("Hybrid advantages merged", extra={"lambda": lam})
    else:
        log.warning(f"Unknown advantage_mode '{mode}' — falling back to standard")
        advantages = _compute_standard(rewards, batch)

    _log_advantage_stats(advantages)
    return advantages, returns


# ── Standard ──────────────────────────────────────────────────────────────────


def _compute_standard(rewards: List[float], batch: RolloutBatch) -> List[float]:
    """advantage_i = reward_i - value_estimate_i"""
    return [r - v.value_estimate for r, v in zip(rewards, batch.rollouts)]


# ── GRPO helpers ──────────────────────────────────────────────────────────────


def _resolve_grpo_group_ids(batch: RolloutBatch) -> List[str]:
    """Return per-rollout group IDs, using rollout_id as fallback."""
    return [r.group_id or r.rollout_id for r in batch.rollouts]


def _build_grpo_fallback_mask(group_counts: Dict[str, int]) -> Dict[str, bool]:
    """
    Groups with fewer than grpo_min_group_size members get a fallback mask
    (advantage = 0 instead of computed GRPO advantage).
    """
    min_size = cfg.grpo_min_group_size
    mask = {gid: (count < min_size) for gid, count in group_counts.items()}
    small = [gid for gid, flag in mask.items() if flag]
    if small:
        log.warning(
            "Small GRPO groups found — advantages will be zeroed",
            extra={"small_groups": small, "min_group_size": min_size},
        )
    return mask


def _compute_grpo_relative_rewards(
    rewards: List[float], batch: RolloutBatch
) -> List[float]:
    """
    For each rollout i in group g:
        adv_i = (reward_i - mean(group_rewards_g)) / (std(group_rewards_g) + ε)

    Groups below grpo_min_group_size get advantage = 0.0.
    """
    group_ids = _resolve_grpo_group_ids(batch)

    # accumulate group statistics
    group_rewards: Dict[str, List[float]] = defaultdict(list)
    for gid, rwd in zip(group_ids, rewards):
        group_rewards[gid].append(rwd)

    group_counts = {gid: len(v) for gid, v in group_rewards.items()}
    fallback_mask = _build_grpo_fallback_mask(group_counts)

    group_mean: Dict[str, float] = {}
    group_std: Dict[str, float] = {}
    for gid, rwds in group_rewards.items():
        group_mean[gid] = sum(rwds) / len(rwds)
        group_std[gid] = _safe_std(rwds)

    advantages: List[float] = []
    for gid, rwd in zip(group_ids, rewards):
        if fallback_mask.get(gid, False):
            advantages.append(0.0)
        else:
            adv = (rwd - group_mean[gid]) / (group_std[gid] + 1e-8)
            advantages.append(adv)

    log.info(
        "GRPO advantages computed",
        extra={
            "num_groups": len(group_rewards),
            "group_sizes": dict(group_counts),
        },
    )
    return advantages


# ── Utilities ─────────────────────────────────────────────────────────────────


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _log_advantage_stats(advantages: List[float]) -> None:
    if not advantages:
        return
    mean = sum(advantages) / len(advantages)
    mn = min(advantages)
    mx = max(advantages)
    log.info(
        "Advantage stats",
        extra={
            "mean": f"{mean:.4f}",
            "min": f"{mn:.4f}",
            "max": f"{mx:.4f}",
            "count": len(advantages),
        },
    )
