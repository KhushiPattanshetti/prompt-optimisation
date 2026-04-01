"""
Unit tests: KL divergence controller
"""

import pytest
import torch

from rl.kl_controller import KLController


class TestKLController:
    @pytest.fixture
    def controller(self):
        return KLController(beta=0.01)

    def test_compute_kl_positive_when_policy_higher(self, controller):
        lp_policy = torch.tensor([-2.0, -3.0])
        lp_ref = torch.tensor([-4.0, -5.0])
        kl = controller.compute_kl(lp_policy, lp_ref)
        assert (kl > 0).all()

    def test_compute_kl_negative_when_policy_lower(self, controller):
        lp_policy = torch.tensor([-5.0, -6.0])
        lp_ref = torch.tensor([-3.0, -4.0])
        kl = controller.compute_kl(lp_policy, lp_ref)
        assert (kl < 0).all()

    def test_compute_kl_zero_when_equal(self, controller):
        lp = torch.tensor([-3.0, -4.0])
        kl = controller.compute_kl(lp, lp)
        assert torch.allclose(kl, torch.zeros_like(kl))

    def test_last_kl_updated(self, controller):
        lp_policy = torch.tensor([-2.0])
        lp_ref = torch.tensor([-4.0])
        controller.compute_kl(lp_policy, lp_ref)
        assert abs(controller.last_kl - 2.0) < 1e-5

    def test_adjust_rewards_subtracts_beta_kl(self, controller):
        rewards = torch.tensor([1.0, 1.0])
        lp_policy = torch.tensor([-2.0, -2.0])
        lp_ref = torch.tensor([-4.0, -4.0])
        adjusted = controller.adjust_rewards(rewards, lp_policy, lp_ref)
        # KL = 2.0 per sample; penalty = beta * KL = 0.01 * 2.0 = 0.02
        expected = torch.tensor([1.0 - 0.02, 1.0 - 0.02])
        assert torch.allclose(adjusted, expected, atol=1e-5)

    def test_adjust_rewards_shape_preserved(self, controller):
        rewards = torch.zeros(8)
        lp_policy = torch.zeros(8)
        lp_ref = torch.zeros(8)
        adjusted = controller.adjust_rewards(rewards, lp_policy, lp_ref)
        assert adjusted.shape == rewards.shape

    def test_kl_output_shape(self, controller):
        lp_policy = torch.randn(6)
        lp_ref = torch.randn(6)
        kl = controller.compute_kl(lp_policy, lp_ref)
        assert kl.shape == (6,)

    def test_different_beta_scales_penalty(self):
        c1 = KLController(beta=0.01)
        c2 = KLController(beta=0.1)
        rewards = torch.ones(4)
        lp_policy = torch.full((4,), -2.0)
        lp_ref = torch.full((4,), -4.0)
        adj1 = c1.adjust_rewards(rewards, lp_policy, lp_ref)
        adj2 = c2.adjust_rewards(rewards, lp_policy, lp_ref)
        # Larger beta → larger penalty → lower adjusted reward
        assert (adj2 < adj1).all()
