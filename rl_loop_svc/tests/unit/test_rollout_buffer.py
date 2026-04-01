"""
Unit tests: RolloutBuffer
"""

import pytest
import torch

from rl.rollout_buffer import RolloutBuffer


class TestRolloutBuffer:
    def test_store_and_len(self):
        buf = RolloutBuffer()
        assert len(buf) == 0
        buf.store(0.5, -5.0, 0.3, "orig", "rewr")
        assert len(buf) == 1

    def test_multiple_stores(self):
        buf = RolloutBuffer()
        for i in range(10):
            buf.store(float(i) * 0.1, -float(i), float(i) * 0.05, f"o{i}", f"r{i}")
        assert len(buf) == 10

    def test_build_shapes(self):
        buf = RolloutBuffer()
        for _ in range(4):
            buf.store(0.5, -5.0, 0.3, "orig", "rewr")
        advantages = torch.zeros(4)
        batch = buf.build(advantages)
        assert batch.rewards.shape == (4,)
        assert batch.log_probs_old.shape == (4,)
        assert batch.values.shape == (4,)
        assert batch.advantages.shape == (4,)
        assert batch.returns.shape == (4,)

    def test_returns_equals_advantage_plus_value(self):
        buf = RolloutBuffer()
        buf.store(0.5, -5.0, 0.4, "orig", "rewr")
        advantages = torch.tensor([0.3])
        batch = buf.build(advantages)
        expected_return = 0.4 + 0.3
        assert abs(batch.returns[0].item() - expected_return) < 1e-5

    def test_clear_resets_buffer(self):
        buf = RolloutBuffer()
        buf.store(0.5, -5.0, 0.3, "orig", "rewr")
        buf.clear()
        assert len(buf) == 0

    def test_build_prompts_preserved(self):
        buf = RolloutBuffer()
        buf.store(0.5, -5.0, 0.3, "clinical note A", "enhanced note A")
        buf.store(0.8, -4.0, 0.6, "clinical note B", "enhanced note B")
        advantages = torch.zeros(2)
        batch = buf.build(advantages)
        assert batch.original_prompts == ["clinical note A", "clinical note B"]
        assert batch.rewritten_prompts == ["enhanced note A", "enhanced note B"]

    def test_negative_rewards_stored_correctly(self):
        buf = RolloutBuffer()
        buf.store(-1.0, -10.0, -0.5, "orig", "rewr")
        assert len(buf) == 1

    def test_build_tensor_dtype(self):
        buf = RolloutBuffer()
        buf.store(0.5, -5.0, 0.3, "orig", "rewr")
        advantages = torch.zeros(1)
        batch = buf.build(advantages)
        assert batch.rewards.dtype == torch.float32
        assert batch.log_probs_old.dtype == torch.float32
