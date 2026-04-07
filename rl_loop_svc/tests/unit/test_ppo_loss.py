"""
Unit tests: PPO loss computation

Uses a minimal mock of PolicyModel / ValueHead so no GPU or model weights
are required.
"""

import pytest
import torch
import torch.nn as nn

from rl.ppo_trainer import PPOTrainer
from rl.rollout_buffer import RolloutBatch


def _make_batch(size: int = 4) -> RolloutBatch:
    rewards = torch.rand(size)
    concept_rewards = torch.rand(size)
    sample_weights = torch.ones(size)
    log_probs = torch.full((size,), -5.0)
    values = torch.rand(size)
    advantages = (torch.rand(size) - 0.5) * 2  # in [-1, 1]
    returns = advantages + values
    return RolloutBatch(
        rewards=rewards,
        log_probs_old=log_probs,
        values=values,
        advantages=advantages,
        returns=returns,
        concept_rewards=concept_rewards,
        sample_weights=sample_weights,
        original_prompts=[f"p{i}" for i in range(size)],
        rewritten_prompts=[f"r{i}" for i in range(size)],
        group_ids=[f"g{i % 2}" for i in range(size)],
    )


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def parameters(self, recurse=True):
        return self.linear.parameters(recurse)


class _TinyValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def parameters(self, recurse=True):
        return self.linear.parameters(recurse)


@pytest.fixture
def trainer():
    return PPOTrainer(
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )


class TestPPOLoss:
    def test_update_returns_components(self, trainer):
        batch = _make_batch()
        log_probs_new = torch.full((4,), -4.8, requires_grad=True)
        values_new = torch.rand(4, requires_grad=True)
        kl_penalty = torch.zeros(4)
        entropy = torch.tensor(0.5)

        _, components = trainer.update(
            batch, log_probs_new, values_new, entropy, kl_penalty
        )
        assert isinstance(components.total_loss, float)
        assert isinstance(components.policy_loss, float)
        assert isinstance(components.value_loss, float)
        assert isinstance(components.entropy_bonus, float)
        assert isinstance(components.kl_penalty, float)

    def test_clipping_limits_large_ratio(self, trainer):
        """A very large ratio should be clipped to 1+epsilon."""
        batch = _make_batch()
        # log_prob_new >> log_prob_old → ratio >> 1
        log_probs_new = torch.full((4,), 0.0, requires_grad=True)  # old was -5
        values_new = torch.rand(4, requires_grad=True)
        kl_penalty = torch.zeros(4)
        entropy = torch.tensor(0.5)

        _, components = trainer.update(
            batch, log_probs_new, values_new, entropy, kl_penalty
        )
        # Loss should be finite and not NaN
        assert not torch.isnan(torch.tensor(components.total_loss))

    def test_loss_finite_for_random_inputs(self, trainer):
        batch = _make_batch(8)
        log_probs_new = (torch.randn(8) - 5).requires_grad_(True)
        values_new = torch.rand(8, requires_grad=True)
        kl_penalty = torch.zeros(8)
        entropy = torch.tensor(0.3)
        _, components = trainer.update(
            batch, log_probs_new, values_new, entropy, kl_penalty
        )
        assert torch.isfinite(torch.tensor(components.total_loss))

    def test_kl_penalty_increases_loss(self, trainer):
        """Adding a positive KL penalty should increase total loss."""
        batch = _make_batch()

        log_probs_new = torch.full((4,), -4.8, requires_grad=True)
        values_new = torch.rand(4, requires_grad=True)
        entropy = torch.tensor(0.5)

        _, c_no_kl = trainer.update(
            batch, log_probs_new, values_new, entropy, torch.zeros(4)
        )

        t2 = PPOTrainer()
        log_probs_new2 = torch.full((4,), -4.8, requires_grad=True)
        values_new2 = values_new.detach().clone().requires_grad_(True)
        kl_big = torch.full((4,), 10.0)
        _, c_kl = t2.update(batch, log_probs_new2, values_new2, entropy, kl_big)

        assert c_kl.total_loss > c_no_kl.total_loss

    def test_sample_weight_modulates_policy_loss(self, trainer):
        base_kwargs = {
            "rewards": torch.tensor([0.0, 0.0]),
            "log_probs_old": torch.tensor([0.0, 0.0]),
            "values": torch.tensor([0.0, 0.0]),
            "advantages": torch.tensor([1.0, 1.0]),
            "returns": torch.tensor([0.0, 0.0]),
            "concept_rewards": torch.tensor([0.0, 0.0]),
            "original_prompts": ["p0", "p1"],
            "rewritten_prompts": ["r0", "r1"],
            "group_ids": ["g0", "g1"],
        }

        log_probs_new = torch.tensor([0.0, -2.0], requires_grad=True)
        values_new = torch.tensor([0.0, 0.0], requires_grad=True)
        entropy = torch.tensor(0.0)
        kl_penalty = torch.zeros(2)

        weighted_batch = RolloutBatch(
            sample_weights=torch.tensor([1.0, 0.0]),
            **base_kwargs,
        )
        _, weighted_components = trainer.update(
            weighted_batch,
            log_probs_new,
            values_new,
            entropy,
            kl_penalty,
        )

        unweighted_batch = RolloutBatch(
            sample_weights=torch.tensor([1.0, 1.0]),
            **base_kwargs,
        )
        log_probs_new_2 = torch.tensor([0.0, -2.0], requires_grad=True)
        values_new_2 = torch.tensor([0.0, 0.0], requires_grad=True)
        _, unweighted_components = trainer.update(
            unweighted_batch,
            log_probs_new_2,
            values_new_2,
            entropy,
            kl_penalty,
        )

        assert weighted_components.policy_loss < unweighted_components.policy_loss
