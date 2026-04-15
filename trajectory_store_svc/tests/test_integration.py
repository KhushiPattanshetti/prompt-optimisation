"""
tests/test_integration.py — Full-pipeline integration tests.

Tests:
  • rollout → storage → load → buffer → advantages → batch file
  • Repeat mode: same batch reused correctly
  • Single-pass mode: sequential consumption, no duplicates
  • No duplicate rollouts across loads
  • Correct batch sizes
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _make_rollout(**kwargs):
    from schemas.rollout_schema import Rollout

    defaults = dict(
        prompt="Integration test prompt",
        rewritten_prompt="Rewritten integration prompt",
        log_prob_old=-0.3,
        value_estimate=0.2,
        reward=0.5,
    )
    defaults.update(kwargs)
    return Rollout(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_pipeline(tmp_path: Path, num_rollouts: int = 8, batch_size: int = 4):
    """Run the full pipeline and return the list of written batch paths."""
    import config as _cfg_mod

    _cfg_mod.cfg.base_dir = tmp_path
    _cfg_mod.cfg.batch_size = batch_size
    _cfg_mod.cfg.reward_already_normalized = True
    _cfg_mod.cfg.normalize_rewards = False
    _cfg_mod.cfg.advantage_mode = "grpo"
    _cfg_mod.cfg.grpo_min_group_size = 2
    _cfg_mod.cfg.batch_reuse_mode = "single_pass"

    from storage.rollout_loader import (
        append_rollout,
        new_rollout_filepath,
        RolloutLoader,
    )
    from processing.preprocessing import preprocess
    from buffer.rollout_buffer import RolloutBuffer
    from processing.advantage import compute_advantages
    from serving.batch_writer import BatchWriter

    rollout_file = tmp_path / "rollouts_store" / "rollouts_int.jsonl"
    prompts = ["Prompt A", "Prompt B", "Prompt C", "Prompt D"]
    for i in range(num_rollouts):
        r = _make_rollout(prompt=prompts[i % len(prompts)], reward=0.4 + 0.1 * (i % 4))
        append_rollout(r, rollout_file)

    loader = RolloutLoader()
    loaded = loader.load_new()
    preprocessed = preprocess(loaded)

    buffer = RolloutBuffer(batch_size=batch_size)
    buffer.add_many(preprocessed)
    batches = buffer.flush()

    writer = BatchWriter()
    written = []
    for batch in batches:
        advs, rets = compute_advantages(batch)
        batch.advantages = advs
        batch.returns = rets
        written.append(writer.write(batch))

    return written, loaded


# ─────────────────────────────────────────────────────────────────────────────
# 1. Full pipeline smoke-test
# ─────────────────────────────────────────────────────────────────────────────


class TestFullPipeline:
    def test_batch_files_created(self, tmp_path):
        paths, _ = _run_pipeline(tmp_path, num_rollouts=8, batch_size=4)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_batch_file_has_required_keys(self, tmp_path):
        paths, _ = _run_pipeline(tmp_path, num_rollouts=4, batch_size=4)
        with paths[0].open() as fh:
            data = json.load(fh)
        for key in (
            "batch_id",
            "rollout_ids",
            "rewards",
            "advantages",
            "returns",
            "group_ids",
            "sample_weights",
        ):
            assert key in data, f"Missing key: {key}"

    def test_advantages_present_and_correct_length(self, tmp_path):
        paths, loaded = _run_pipeline(tmp_path, num_rollouts=8, batch_size=4)
        with paths[0].open() as fh:
            data = json.load(fh)
        assert len(data["advantages"]) == 4
        assert len(data["returns"]) == 4

    def test_rollout_ids_in_batch_are_unique(self, tmp_path):
        paths, _ = _run_pipeline(tmp_path, num_rollouts=4, batch_size=4)
        with paths[0].open() as fh:
            data = json.load(fh)
        ids = data["rollout_ids"]
        assert len(ids) == len(set(ids)), "Duplicate rollout_ids in batch"

    def test_no_duplicate_rollouts_across_batches(self, tmp_path):
        paths, _ = _run_pipeline(tmp_path, num_rollouts=8, batch_size=4)
        all_ids = []
        for p in paths:
            with p.open() as fh:
                data = json.load(fh)
            all_ids.extend(data["rollout_ids"])
        assert len(all_ids) == len(set(all_ids)), "Duplicate rollout_ids across batches"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Batch reuse — repeat mode
# ─────────────────────────────────────────────────────────────────────────────


class TestRepeatMode:
    def test_reuse_counter_increments(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        _cfg_mod.cfg.batch_reuse_mode = "repeat"
        _cfg_mod.cfg.max_batch_reuse_count = 3

        from serving.batch_writer import BatchWriter
        from buffer.rollout_buffer import RolloutBatch

        rollouts = [_make_rollout() for _ in range(4)]
        batch = RolloutBatch(rollouts=rollouts, advantages=[0.1] * 4, returns=[0.5] * 4)

        writer = BatchWriter()
        writer.write(batch)

        count = 0
        while writer.should_reuse():
            writer.record_reuse()
            count += 1

        assert count == 3

    def test_repeat_does_not_write_duplicate_files(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        _cfg_mod.cfg.batch_reuse_mode = "repeat"
        _cfg_mod.cfg.max_batch_reuse_count = 2

        from serving.batch_writer import BatchWriter
        from buffer.rollout_buffer import RolloutBatch

        rollouts = [_make_rollout() for _ in range(4)]
        batch = RolloutBatch(rollouts=rollouts, advantages=[0.2] * 4, returns=[0.4] * 4)

        writer = BatchWriter()
        writer.write(batch)
        while writer.should_reuse():
            writer.record_reuse()

        # Only one file written (reuse is in-memory)
        assert len(writer.written_paths) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Single-pass mode
# ─────────────────────────────────────────────────────────────────────────────


class TestSinglePassMode:
    def test_each_batch_written_once(self, tmp_path):
        paths, _ = _run_pipeline(tmp_path, num_rollouts=12, batch_size=4)
        # 12 rollouts / batch_size 4 = 3 batches
        assert len(paths) == 3

    def test_batch_ids_are_unique(self, tmp_path):
        paths, _ = _run_pipeline(tmp_path, num_rollouts=8, batch_size=4)
        ids = []
        for p in paths:
            with p.open() as fh:
                data = json.load(fh)
            ids.append(data["batch_id"])
        assert len(ids) == len(set(ids))

    def test_no_double_load_after_restart(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        from storage.rollout_loader import append_rollout, RolloutLoader

        filepath = tmp_path / "rollouts_store" / "rollouts_sp.jsonl"
        for _ in range(6):
            append_rollout(_make_rollout(), filepath)

        loader1 = RolloutLoader()
        first_batch = loader1.load_new()

        # Simulate restart by creating new loader
        loader2 = RolloutLoader()
        second_batch = loader2.load_new()

        assert len(first_batch) == 6
        assert (
            len(second_batch) == 0
        ), "Restart should not re-read already processed rollouts"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correct batch sizes
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchSizes:
    def test_each_batch_exactly_batch_size(self, tmp_path):
        paths, _ = _run_pipeline(tmp_path, num_rollouts=8, batch_size=4)
        for p in paths:
            with p.open() as fh:
                data = json.load(fh)
            assert len(data["rollout_ids"]) == 4

    def test_partial_batch_not_written(self, tmp_path):
        """11 rollouts with batch_size=4 → 2 full batches, 3 leftover discarded."""
        paths, _ = _run_pipeline(tmp_path, num_rollouts=11, batch_size=4)
        assert len(paths) == 2
