import torch

from rl.training_loop import (
    _compute_grpo_relative_rewards,
    _resolve_group_id,
    _resolve_grpo_group_ids,
)
from schemas.rollout_schema import RolloutEntry


def _entry(original_prompt: str, group_id: str | None = None) -> RolloutEntry:
    return RolloutEntry(
        original_prompt=original_prompt,
        rewritten_prompt="rewrite",
        reward=0.1,
        log_prob_old=-0.2,
        value_estimate=0.0,
        group_id=group_id,
    )


def test_resolve_group_id_prefers_existing_group_id():
    entry = _entry("note A", group_id="group-123")
    assert _resolve_group_id(entry) == "group-123"


def test_resolve_group_id_fallback_is_stable_per_prompt():
    e1 = _entry("same original note")
    e2 = _entry("same original note")
    g1 = _resolve_group_id(e1)
    g2 = _resolve_group_id(e2)

    assert g1 == g2
    assert g1.startswith("g_")


def test_grpo_relative_rewards_zero_when_no_group_meets_threshold():
    rewards = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    group_ids = ["a", "b", "c"]
    rel = _compute_grpo_relative_rewards(rewards, group_ids, min_group_size=2)

    assert torch.allclose(rel, torch.zeros_like(rewards))


def test_grpo_relative_rewards_have_variance_for_valid_groups():
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3], dtype=torch.float32)
    group_ids = ["g1", "g1", "g1", "g2"]
    rel = _compute_grpo_relative_rewards(rewards, group_ids, min_group_size=3)

    assert torch.isfinite(rel).all()
    assert rel.shape == rewards.shape
    assert float(torch.std(rel)) > 0.0


def test_resolve_grpo_group_ids_creates_triplets_for_sparse_groups():
    incoming = [f"g{i}" for i in range(7)]
    resolved = _resolve_grpo_group_ids(incoming, min_group_size=3, fallback_group_size=3)

    assert resolved[0] == resolved[1] == resolved[2]
    assert resolved[3] == resolved[4] == resolved[5]
    assert resolved[6] != resolved[5]


def test_resolve_grpo_group_ids_preserves_existing_valid_groups():
    incoming = ["g1", "g1", "g1", "g2"]
    resolved = _resolve_grpo_group_ids(incoming, min_group_size=3, fallback_group_size=3)

    assert resolved == incoming
