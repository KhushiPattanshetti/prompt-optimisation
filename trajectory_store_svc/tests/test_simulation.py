"""
tests/test_simulation.py — Failure-mode and simulation end-to-end tests.

Tests:
  • Corrupted JSONL entries
  • Partial writes (truncated line)
  • Empty buffer / insufficient rollouts
  • Full simulator.run_simulation() end-to-end
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _make_rollout(**kwargs):
    from schemas.rollout_schema import Rollout

    defaults = dict(
        prompt="Sim test prompt",
        rewritten_prompt="Rewritten sim prompt",
        log_prob_old=-0.4,
        value_estimate=0.25,
        reward=0.55,
    )
    defaults.update(kwargs)
    return Rollout(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Failure handling
# ─────────────────────────────────────────────────────────────────────────────


class TestFailureHandling:
    def test_corrupted_jsonl_skipped_not_crashed(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        from storage.rollout_loader import RolloutLoader

        filepath = tmp_path / "rollouts_store" / "rollouts_corrupt.jsonl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as fh:
            fh.write("}{INVALID_JSON}\n")
            fh.write('{"also": "broken"\n')  # unclosed
            fh.write(json.dumps(_make_rollout().to_dict()) + "\n")  # valid

        loader = RolloutLoader()
        results = loader.load_new()
        # 2 corrupted lines skipped, 1 valid returned
        assert len(results) == 1

    def test_partial_write_truncated_line_handled(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        from storage.rollout_loader import RolloutLoader

        filepath = tmp_path / "rollouts_store" / "rollouts_partial.jsonl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as fh:
            fh.write(json.dumps(_make_rollout().to_dict()) + "\n")
            # Simulate truncated write (no closing brace)
            fh.write('{"rollout_id": "')

        loader = RolloutLoader()
        results = loader.load_new()
        # Only the complete line should parse
        assert len(results) == 1

    def test_empty_buffer_flush_is_safe(self):
        from buffer.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer(batch_size=4)
        assert buf.flush() == []
        assert buf.pending_count() == 0

    def test_insufficient_rollouts_no_batch_produced(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        from storage.rollout_loader import append_rollout, RolloutLoader
        from buffer.rollout_buffer import RolloutBuffer

        filepath = tmp_path / "rollouts_store" / "rollouts_short.jsonl"
        for _ in range(3):  # less than batch_size=4
            append_rollout(_make_rollout(), filepath)

        loader = RolloutLoader()
        loaded = loader.load_new()

        buf = RolloutBuffer(batch_size=4)
        buf.add_many(loaded)
        batches = buf.flush()

        assert batches == []

    def test_all_corrupted_lines_returns_empty(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        from storage.rollout_loader import RolloutLoader

        filepath = tmp_path / "rollouts_store" / "all_corrupt.jsonl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as fh:
            for _ in range(5):
                fh.write("NOT JSON\n")

        loader = RolloutLoader()
        results = loader.load_new()
        assert results == []

    def test_empty_jsonl_file_handled(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        from storage.rollout_loader import RolloutLoader

        filepath = tmp_path / "rollouts_store" / "rollouts_empty.jsonl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()

        loader = RolloutLoader()
        results = loader.load_new()
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulator end-to-end
# ─────────────────────────────────────────────────────────────────────────────


class TestSimulator:
    def test_run_simulation_creates_batch_files(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        _cfg_mod.cfg.batch_size = 4
        _cfg_mod.cfg.reward_already_normalized = True
        _cfg_mod.cfg.grpo_min_group_size = 2
        _cfg_mod.cfg.advantage_mode = "grpo"
        _cfg_mod.cfg.batch_reuse_mode = "single_pass"

        from simulation.simulator import run_simulation

        paths = run_simulation(num_rollouts=8, batch_size=4)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_batch_file_fields_are_non_empty(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        _cfg_mod.cfg.batch_size = 4
        _cfg_mod.cfg.reward_already_normalized = True
        _cfg_mod.cfg.grpo_min_group_size = 2
        _cfg_mod.cfg.advantage_mode = "grpo"
        _cfg_mod.cfg.batch_reuse_mode = "single_pass"

        from simulation.simulator import run_simulation

        paths = run_simulation(num_rollouts=4, batch_size=4)

        with paths[0].open() as fh:
            data = json.load(fh)

        assert len(data["rollout_ids"]) == 4
        assert len(data["advantages"]) == 4
        assert len(data["returns"]) == 4
        assert len(data["group_ids"]) == 4
        assert all(w == 1.0 for w in data["sample_weights"])

    def test_generate_rollout_returns_valid_rollout(self):
        from simulation.simulator import generate_rollout

        r = generate_rollout(prompt="Hello world")
        assert r.prompt == "Hello world"
        assert r.log_prob_old <= 0
        assert -1.0 <= r.reward <= 1.0

    def test_simulation_repeat_mode(self, tmp_path):
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        _cfg_mod.cfg.batch_size = 4
        _cfg_mod.cfg.reward_already_normalized = True
        _cfg_mod.cfg.grpo_min_group_size = 2
        _cfg_mod.cfg.advantage_mode = "grpo"
        _cfg_mod.cfg.batch_reuse_mode = "repeat"
        _cfg_mod.cfg.max_batch_reuse_count = 2

        from simulation.simulator import run_simulation

        paths = run_simulation(num_rollouts=8, batch_size=4)

        # Files are still written (reuse is additional in-memory behaviour)
        assert len(paths) >= 1

    def test_simulation_no_restart_duplicates(self, tmp_path):
        """Two consecutive simulation runs should not produce duplicate rollout_ids."""
        import config as _cfg_mod

        _cfg_mod.cfg.base_dir = tmp_path
        _cfg_mod.cfg.batch_size = 4
        _cfg_mod.cfg.reward_already_normalized = True
        _cfg_mod.cfg.advantage_mode = "grpo"
        _cfg_mod.cfg.grpo_min_group_size = 2
        _cfg_mod.cfg.batch_reuse_mode = "single_pass"

        from simulation.simulator import run_simulation

        # Run 1
        paths1 = run_simulation(num_rollouts=8, batch_size=4)
        # Run 2 — loader should not re-read run-1 rollouts
        paths2 = run_simulation(num_rollouts=8, batch_size=4)

        all_ids = set()
        for p in paths1 + paths2:
            with p.open() as fh:
                data = json.load(fh)
            for rid in data["rollout_ids"]:
                assert rid not in all_ids, f"Duplicate rollout_id across runs: {rid}"
                all_ids.add(rid)
