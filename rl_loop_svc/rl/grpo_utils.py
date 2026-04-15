"""
Shared GRPO (Group Relative Policy Optimization) utilities.

Used by both the local training loop and the distributed training script
to avoid logic duplication.
"""

import hashlib
import math
from collections import Counter
from typing import List

import torch

from ..schemas.rollout_schema import RolloutEntry
from .rollout_buffer import RolloutBatch


def resolve_group_id(entry: RolloutEntry) -> str:
    if entry.group_id:
        return str(entry.group_id)
    base = str(entry.original_prompt or "")
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"g_{digest}"


def resolve_sample_weight(entry: RolloutEntry) -> float:
    if entry.sample_weight is None:
        return 1.0

    try:
        weight = float(entry.sample_weight)
    except (TypeError, ValueError):
        return 1.0

    if not math.isfinite(weight):
        return 1.0

    return float(min(max(weight, 0.0), 1.0))


def compute_grpo_relative_rewards(
    rewards: torch.Tensor,
    group_ids: List[str],
    min_group_size: int,
) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards

    relative = torch.zeros_like(rewards)
    grouped: dict[str, List[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(idx)

    for indices in grouped.values():
        if len(indices) < max(min_group_size, 2):
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        group_rewards = rewards[idx_tensor]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False) + 1e-6
        relative[idx_tensor] = (group_rewards - group_mean) / group_std

    return relative


def resolve_grpo_group_ids(
    group_ids: List[str],
    min_group_size: int,
    fallback_group_size: int,
) -> List[str]:
    """Ensure GRPO has usable groups by backfilling deterministic K-sized groups when needed."""
    normalized = [str(group_id) for group_id in group_ids]
    if not normalized:
        return normalized

    threshold = max(int(min_group_size), 2)
    counts = Counter(normalized)
    if any(size >= threshold for size in counts.values()):
        return normalized

    group_size = max(int(fallback_group_size), threshold)
    return [f"grpo_auto_{idx // group_size}" for idx in range(len(normalized))]


def build_grpo_fallback_mask(
    group_ids: List[str],
    min_group_size: int,
    device: torch.device,
) -> torch.Tensor:
    threshold = max(int(min_group_size), 2)
    counts = Counter(str(group_id) for group_id in group_ids)
    mask = [counts.get(str(group_id), 0) < threshold for group_id in group_ids]
    return torch.tensor(mask, dtype=torch.bool, device=device)


def group_reward_std_mean(
    rewards: torch.Tensor,
    group_ids: List[str],
    min_group_size: int,
) -> float:
    threshold = max(int(min_group_size), 2)
    grouped: dict[str, List[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(idx)

    std_values: List[float] = []
    for indices in grouped.values():
        if len(indices) < threshold:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        group_std = rewards[idx_tensor].std(unbiased=False)
        std_values.append(float(group_std.item()))

    if not std_values:
        return 0.0
    return float(sum(std_values) / len(std_values))


def select_rollout_batch(batch: RolloutBatch, keep_idx: torch.Tensor) -> RolloutBatch:
    idx_list = [int(v) for v in keep_idx.detach().cpu().tolist()]
    return RolloutBatch(
        rewards=batch.rewards[keep_idx],
        log_probs_old=batch.log_probs_old[keep_idx],
        values=batch.values[keep_idx],
        advantages=batch.advantages[keep_idx],
        returns=batch.returns[keep_idx],
        sample_weights=batch.sample_weights[keep_idx],
        original_prompts=[batch.original_prompts[i] for i in idx_list],
        rewritten_prompts=[batch.rewritten_prompts[i] for i in idx_list],
        concept_rewards=batch.concept_rewards[keep_idx],
        group_ids=[batch.group_ids[i] for i in idx_list],
    )


def build_repeated_index(size: int, target_size: int, device: torch.device) -> torch.Tensor:
    if size <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    if size >= target_size:
        return torch.arange(size, dtype=torch.long, device=device)

    repeats = int(math.ceil(target_size / float(size)))
    base = torch.arange(size, dtype=torch.long, device=device)
    return base.repeat(repeats)[:target_size]
