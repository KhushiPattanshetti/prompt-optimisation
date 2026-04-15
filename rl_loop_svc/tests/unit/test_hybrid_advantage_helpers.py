import torch

from ...rl.grpo_utils import (
    compute_grpo_relative_rewards,
    resolve_grpo_group_ids,
)
from ...schemas.rollout_schema import RolloutEntry


def _entry(original_prompt: str, group_id: str | None = None) -> RolloutEntry:
    return RolloutEntry(
        original_prompt=original_prompt,
        rewritten_prompt="rewrite",
        reward=0.1,
        log_prob_old=-0.2,
        value_estimate=0.0,
        group_id=group_id,
    )


def test_grpo_entries_with_preassigned_group_id_are_used_directly():
    """
    group_id is now assigned upstream by trajectory_store_svc preprocessing.
    Entries arriving with a group_id should use it as-is.
    """
    entry = _entry("note A", group_id="group-123")
    assert entry.group_id == "group-123"


def test_grpo_entries_without_group_id_have_none():
    """
    Entries without a group_id (not yet preprocessed) have group_id=None.
    """
    e = _entry("same original note")
    assert e.group_id is None


def test_grpo_relative_rewards_zero_when_no_group_meets_threshold():
    rewards = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    group_ids = ["a", "b", "c"]
    rel = compute_grpo_relative_rewards(rewards, group_ids, min_group_size=2)

    assert torch.allclose(rel, torch.zeros_like(rewards))


def test_grpo_relative_rewards_have_variance_for_valid_groups():
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3], dtype=torch.float32)
    group_ids = ["g1", "g1", "g1", "g2"]
    rel = compute_grpo_relative_rewards(rewards, group_ids, min_group_size=3)

    assert torch.isfinite(rel).all()
    assert rel.shape == rewards.shape
    assert float(torch.std(rel)) > 0.0


def test_resolve_grpo_group_ids_creates_triplets_for_sparse_groups():
    incoming = [f"g{i}" for i in range(7)]
    resolved = resolve_grpo_group_ids(incoming, min_group_size=3, fallback_group_size=3)

    assert resolved[0] == resolved[1] == resolved[2]
    assert resolved[3] == resolved[4] == resolved[5]
    assert resolved[6] != resolved[5]


def test_resolve_grpo_group_ids_preserves_existing_valid_groups():
    incoming = ["g1", "g1", "g1", "g2"]
    resolved = resolve_grpo_group_ids(incoming, min_group_size=3, fallback_group_size=3)

    assert resolved == incoming
