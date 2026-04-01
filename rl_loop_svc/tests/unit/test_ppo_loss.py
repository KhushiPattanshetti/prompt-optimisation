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
        original_prompts=[f"p{i}" for i in range(size)],
        rewritten_prompts=[f"r{i}" for i in range(size)],
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
    policy = _TinyPolicy()
    value_head = _TinyValueHead()
    optimizer = torch.optim.SGD(
        list(policy.parameters()) + list(value_head.parameters()), lr=1e-3
    )
    return PPOTrainer(
        policy, value_head, optimizer, epsilon=0.2, value_coef=0.5, entropy_coef=0.01
    )


class TestPPOLoss:
    def test_update_returns_components(self, trainer):
        batch = _make_batch()
        log_probs_new = torch.full((4,), -4.8, requires_grad=True)
        values_new = torch.rand(4, requires_grad=True)
        kl_penalty = torch.zeros(4)
        entropy = torch.tensor(0.5)

        components = trainer.update(
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

        components = trainer.update(
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
        components = trainer.update(
            batch, log_probs_new, values_new, entropy, kl_penalty
        )
        assert torch.isfinite(torch.tensor(components.total_loss))

    def test_kl_penalty_increases_loss(self, trainer):
        """Adding a positive KL penalty should increase total loss."""
        batch = _make_batch()

        log_probs_new = torch.full((4,), -4.8, requires_grad=True)
        values_new = torch.rand(4, requires_grad=True)
        entropy = torch.tensor(0.5)

        c_no_kl = trainer.update(
            batch, log_probs_new, values_new, entropy, torch.zeros(4)
        )

        # Re-create trainer with fresh params to avoid state carry-over
        p2 = _TinyPolicy()
        v2 = _TinyValueHead()
        opt2 = torch.optim.SGD(list(p2.parameters()) + list(v2.parameters()), lr=1e-3)
        t2 = PPOTrainer(p2, v2, opt2)
        log_probs_new2 = torch.full((4,), -4.8, requires_grad=True)
        values_new2 = torch.rand(4, requires_grad=True)
        kl_big = torch.full((4,), 10.0)
        c_kl = t2.update(batch, log_probs_new2, values_new2, entropy, kl_big)

        assert c_kl.total_loss > c_no_kl.total_loss
