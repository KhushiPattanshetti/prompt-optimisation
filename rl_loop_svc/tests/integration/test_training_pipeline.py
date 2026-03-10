"""
Integration tests: rollout loader ↔ buffer ↔ training step pipeline

These tests verify that the three components interact correctly:
1. RolloutLoader reads files and produces RolloutEntry objects
2. RolloutBuffer stores them correctly
3. GAE advantage computation produces valid tensors

Model loading is avoided — we test data flow only.
"""

import json
from pathlib import Path

import pytest
import torch

from rl.advantage import compute_gae
from rl.rollout_buffer import RolloutBuffer
from schemas.rollout_schema import RolloutEntry, RolloutFile
from storage.rollout_loader import RolloutLoader


class TestRolloutLoaderBufferIntegration:
    def test_loader_fills_buffer(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        entries = loader.load_all()
        buf = RolloutBuffer()
        for e in entries:
            buf.store(
                e.reward,
                e.log_prob_old,
                e.value_estimate,
                e.original_prompt,
                e.rewritten_prompt,
            )
        assert len(buf) == len(entries)

    def test_loader_new_entries_only(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        first = loader.load_new()
        second = loader.load_new()  # Should be empty — nothing new
        assert len(first) > 0
        assert len(second) == 0

    def test_incremental_loading_accumulates(self, tmp_path):
        d = tmp_path / "rollouts"
        d.mkdir()
        loader = RolloutLoader(d)

        entry = {
            "original_prompt": "note",
            "rewritten_prompt": "enhanced",
            "reward": 0.5,
            "log_prob_old": -5.0,
            "value_estimate": 0.3,
        }
        (d / "batch_001.json").write_text(json.dumps({"rollouts": [entry]}))
        first = loader.load_new()
        assert len(first) == 1

        (d / "batch_002.json").write_text(json.dumps({"rollouts": [entry, entry]}))
        second = loader.load_new()
        assert len(second) == 2

    def test_advantage_computed_from_loader_data(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        entries = loader.load_all()
        rewards = torch.tensor([e.reward for e in entries])
        values = torch.tensor([e.value_estimate for e in entries])
        adv = compute_gae(rewards, values)
        assert adv.shape == rewards.shape
        assert torch.isfinite(adv).all()

    def test_buffer_build_after_loader(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        entries = loader.load_all()
        buf = RolloutBuffer()
        for e in entries:
            buf.store(
                e.reward,
                e.log_prob_old,
                e.value_estimate,
                e.original_prompt,
                e.rewritten_prompt,
            )
        rewards = torch.tensor([e.reward for e in entries])
        values = torch.tensor([e.value_estimate for e in entries])
        adv = compute_gae(rewards, values)
        batch = buf.build(adv)
        assert batch.rewards.shape[0] == len(entries)
        assert batch.advantages.shape[0] == len(entries)

    def test_malformed_file_skipped(self, tmp_path):
        d = tmp_path / "rollouts"
        d.mkdir()
        (d / "bad.json").write_text("{invalid json}")
        loader = RolloutLoader(d)
        entries = loader.load_all()
        assert entries == []

    def test_loader_reset_allows_reload(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        first = loader.load_new()
        loader.reset()
        second = loader.load_new()
        assert len(first) == len(second)
