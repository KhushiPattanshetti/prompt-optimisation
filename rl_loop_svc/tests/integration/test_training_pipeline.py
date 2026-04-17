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

from ...rl.advantage import compute_gae
from ...rl.rollout_buffer import RolloutBuffer
from ...storage.rollout_loader import RolloutLoader


class TestRolloutLoaderBufferIntegration:
    def test_loader_fills_buffer(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        entries = loader.load_all()
        buf = RolloutBuffer()
        for e in entries:
            value_estimate = (
                float(e.value_estimate) if e.value_estimate is not None else 0.0
            )
            buf.store(
                e.reward,
                e.log_prob_old,
                value_estimate,
                e.group_id or "g_default",
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
        (d / "rollout_batch_001.json").write_text(json.dumps({"rollouts": [entry]}))
        first = loader.load_new()
        assert len(first) == 1

        (d / "rollout_batch_002.json").write_text(
            json.dumps({"rollouts": [entry, entry]})
        )
        second = loader.load_new()
        assert len(second) == 2

    def test_advantage_computed_from_loader_data(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        entries = loader.load_all()
        rewards = torch.tensor([e.reward for e in entries])
        values = torch.tensor(
            [
                float(e.value_estimate) if e.value_estimate is not None else 0.0
                for e in entries
            ]
        )
        adv = compute_gae(rewards, values)
        assert adv.shape == rewards.shape
        assert torch.isfinite(adv).all()

    def test_buffer_build_after_loader(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        entries = loader.load_all()
        buf = RolloutBuffer()
        for e in entries:
            value_estimate = (
                float(e.value_estimate) if e.value_estimate is not None else 0.0
            )
            buf.store(
                e.reward,
                e.log_prob_old,
                value_estimate,
                e.group_id or "g_default",
                e.original_prompt,
                e.rewritten_prompt,
            )
        rewards = torch.tensor([e.reward for e in entries])
        values = torch.tensor(
            [
                float(e.value_estimate) if e.value_estimate is not None else 0.0
                for e in entries
            ]
        )
        adv = compute_gae(rewards, values)
        batch = buf.build(adv)
        assert batch.rewards.shape[0] == len(entries)
        assert batch.advantages.shape[0] == len(entries)

    def test_malformed_file_skipped(self, tmp_path):
        d = tmp_path / "rollouts"
        d.mkdir()
        (d / "rollout_batch_bad.json").write_text("{invalid json}")
        loader = RolloutLoader(d)
        entries = loader.load_all()
        assert entries == []

    def test_loader_reset_allows_reload(self, rollouts_dir):
        loader = RolloutLoader(rollouts_dir)
        first = loader.load_new()
        loader.reset()
        second = loader.load_new()
        assert len(first) == len(second)


class TestTrajectoryStoreIntegration:
    """
    Verify that RolloutLoader can read bare-JSONL files written by
    trajectory_store_svc (one Rollout JSON object per line, with 'prompt'
    field instead of 'original_prompt', and pre-computed group_id /
    sample_weight).
    """

    def _traj_line(
        self, prompt: str = "Patient note", group_id: str = "abc12345"
    ) -> str:
        import json
        import uuid

        row = {
            "rollout_id": str(uuid.uuid4()),
            "prompt": prompt,
            "rewritten_prompt": f"[Rewritten] {prompt}",
            "log_prob_old": -0.5,
            "value_estimate": 0.3,
            "reward": 0.6,
            "group_id": group_id,
            "sample_weight": 1.0,
        }
        return json.dumps(row)

    def test_reads_traj_store_jsonl(self, tmp_path):
        rl_dir = tmp_path / "rl_rollouts"
        rl_dir.mkdir()
        traj_dir = tmp_path / "traj_rollouts"
        traj_dir.mkdir()

        jsonl_file = traj_dir / "rollouts_1234567890.jsonl"
        jsonl_file.write_text(
            "\n".join([self._traj_line("note A"), self._traj_line("note B")]) + "\n",
            encoding="utf-8",
        )

        loader = RolloutLoader(rl_dir, traj_store_dir=traj_dir)
        entries = loader.load_new()

        assert len(entries) == 2
        assert entries[0].original_prompt == "note A"
        assert entries[1].original_prompt == "note B"
        assert entries[0].group_id == "abc12345"
        assert entries[0].sample_weight == 1.0

    def test_traj_store_incremental_offset_tracking(self, tmp_path):
        rl_dir = tmp_path / "rl_rollouts"
        rl_dir.mkdir()
        traj_dir = tmp_path / "traj_rollouts"
        traj_dir.mkdir()

        jsonl_file = traj_dir / "rollouts_1111111111.jsonl"
        jsonl_file.write_text(self._traj_line("note A") + "\n", encoding="utf-8")

        loader = RolloutLoader(rl_dir, traj_store_dir=traj_dir)
        first = loader.load_new()
        assert len(first) == 1

        # Append a second rollout to the same file
        with jsonl_file.open("a", encoding="utf-8") as fh:
            fh.write(self._traj_line("note B") + "\n")

        second = loader.load_new()
        assert len(second) == 1
        assert second[0].original_prompt == "note B"

    def test_traj_store_group_id_passed_through(self, tmp_path):
        rl_dir = tmp_path / "rl_rollouts"
        rl_dir.mkdir()
        traj_dir = tmp_path / "traj_rollouts"
        traj_dir.mkdir()

        import json, uuid

        row = {
            "rollout_id": str(uuid.uuid4()),
            "prompt": "Patient note",
            "rewritten_prompt": "[Rewritten] Patient note",
            "log_prob_old": -0.5,
            "value_estimate": 0.3,
            "reward": 0.6,
            "group_id": "precomputed_by_upstream",
            "sample_weight": 0.75,
        }
        (traj_dir / "rollouts_9999999999.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )

        loader = RolloutLoader(rl_dir, traj_store_dir=traj_dir)
        entries = loader.load_new()

        assert len(entries) == 1
        assert entries[0].group_id == "precomputed_by_upstream"
        assert abs(entries[0].sample_weight - 0.75) < 1e-6
