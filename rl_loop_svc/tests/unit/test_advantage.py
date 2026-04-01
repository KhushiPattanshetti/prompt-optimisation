"""
Unit tests: GAE advantage computation
"""

import pytest
import torch

from rl.advantage import compute_gae


class TestComputeGAE:
    def test_output_shape(self, rewards, values):
        adv = compute_gae(rewards, values)
        assert adv.shape == rewards.shape

    def test_output_dtype(self, rewards, values):
        adv = compute_gae(rewards, values)
        assert adv.dtype == torch.float32

    def test_normalised_mean_near_zero(self, rewards, values):
        adv = compute_gae(rewards, values)
        assert abs(adv.mean().item()) < 1e-5

    def test_normalised_std_near_one(self, rewards, values):
        adv = compute_gae(rewards, values)
        if len(adv) > 1:
            assert abs(adv.std().item() - 1.0) < 0.1

    def test_single_step(self):
        r = torch.tensor([1.0])
        v = torch.tensor([0.5])
        adv = compute_gae(r, v)
        # Single step: no normalisation applied; delta = 1.0 + 0 - 0.5 = 0.5
        assert adv.shape == (1,)
        assert abs(adv[0].item() - 0.5) < 1e-5

    def test_high_gamma_propagates_reward(self):
        """Future rewards should flow back with gamma close to 1 (raw, pre-normalisation)."""
        rewards = torch.tensor([0.0, 0.0, 1.0])
        values = torch.zeros(3)
        adv_high = compute_gae(rewards, values, gamma=0.99, lam=1.0, normalize=False)
        adv_low = compute_gae(rewards, values, gamma=0.01, lam=1.0, normalize=False)
        # With high gamma the first step should receive more of the future reward
        assert adv_high[0].item() > adv_low[0].item()

    def test_zero_rewards_zero_unadjusted(self):
        rewards = torch.zeros(4)
        values = torch.zeros(4)
        adv = compute_gae(rewards, values)
        # After normalisation std is 0 so result is all-zero (std guard: +1e-8)
        assert adv.abs().max().item() < 1e-3

    def test_custom_gamma_lambda(self, rewards, values):
        adv1 = compute_gae(rewards, values, gamma=0.99, lam=0.95)
        adv2 = compute_gae(rewards, values, gamma=0.5, lam=0.5)
        # Different hyper-params should produce different advantages
        assert not torch.allclose(adv1, adv2)
