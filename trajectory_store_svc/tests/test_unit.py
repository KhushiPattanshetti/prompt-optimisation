"""
tests/test_unit.py — Unit tests for all core modules.

Coverage:
  • Rollout schema — creation, validation
  • JSONL read/write + offset tracking
  • Preprocessing — group_id determinism, sample_weight
  • Buffer — filling, flushing, remainder
  • Reward processing — normalisation guard
  • Advantage computation — standard, grpo, hybrid
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_rollout(**kwargs):
    from schemas.rollout_schema import Rollout

    defaults = dict(
        prompt="Test prompt",
        rewritten_prompt="Rewritten test prompt",
        log_prob_old=-0.5,
        value_estimate=0.3,
        reward=0.6,
    )
    defaults.update(kwargs)
    return Rollout(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rollout schema
# ─────────────────────────────────────────────────────────────────────────────


class TestRolloutSchema:
    def test_default_rollout_id_is_uuid(self):
        r = _make_rollout()
        assert uuid.UUID(r.rollout_id)  # must not raise

    def test_custom_rollout_id_accepted(self):
        rid = str(uuid.uuid4())
        r = _make_rollout(rollout_id=rid)
        assert r.rollout_id == rid

    def test_invalid_rollout_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_rollout(rollout_id="not-a-uuid")

    def test_positive_log_prob_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_rollout(log_prob_old=0.1)  # must be ≤ 0

    def test_roundtrip_dict(self):
        from schemas.rollout_schema import Rollout

        r = _make_rollout()
        r2 = Rollout.from_dict(r.to_dict())
        assert r.rollout_id == r2.rollout_id
        assert r.reward == r2.reward

    def test_default_sample_weight(self):
        r = _make_rollout()
        assert r.sample_weight == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. JSONL read/write + offset tracking
# ─────────────────────────────────────────────────────────────────────────────


class TestJSONLStorage:
    def test_append_and_read_back(self, tmp_path):
        from storage.rollout_loader import append_rollout, RolloutLoader
        import config

        config.cfg.base_dir = tmp_path

        filepath = tmp_path / "rollouts_store" / "rollouts_test.jsonl"
        r = _make_rollout()
        append_rollout(r, filepath)

        assert filepath.exists()
        with filepath.open() as fh:
            data = json.loads(fh.readline())
        assert data["rollout_id"] == r.rollout_id

    def test_no_duplicate_reads(self, tmp_path):
        from storage.rollout_loader import append_rollout, RolloutLoader
        import config

        config.cfg.base_dir = tmp_path

        filepath = tmp_path / "rollouts_store" / "rollouts_test.jsonl"
        for _ in range(5):
            append_rollout(_make_rollout(), filepath)

        loader = RolloutLoader()
        first = loader.load_new()
        second = loader.load_new()  # nothing new since last load

        assert len(first) == 5
        assert len(second) == 0

    def test_incremental_reads(self, tmp_path):
        from storage.rollout_loader import append_rollout, RolloutLoader
        import config

        config.cfg.base_dir = tmp_path

        filepath = tmp_path / "rollouts_store" / "rollouts_inc.jsonl"
        for _ in range(3):
            append_rollout(_make_rollout(), filepath)

        loader = RolloutLoader()
        batch1 = loader.load_new()

        # Append more after the first load
        for _ in range(2):
            append_rollout(_make_rollout(), filepath)

        batch2 = loader.load_new()

        assert len(batch1) == 3
        assert len(batch2) == 2

    def test_corrupted_line_skipped(self, tmp_path):
        from storage.rollout_loader import RolloutLoader
        import config

        config.cfg.base_dir = tmp_path

        filepath = tmp_path / "rollouts_store" / "rollouts_corrupt.jsonl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as fh:
            fh.write("NOT_VALID_JSON\n")
            fh.write(json.dumps(_make_rollout().to_dict()) + "\n")

        loader = RolloutLoader()
        results = loader.load_new()
        # Only the valid line should be returned
        assert len(results) == 1

    def test_offset_persisted_on_restart(self, tmp_path):
        from storage.rollout_loader import append_rollout, RolloutLoader
        import config

        config.cfg.base_dir = tmp_path

        filepath = tmp_path / "rollouts_store" / "rollouts_restart.jsonl"
        for _ in range(4):
            append_rollout(_make_rollout(), filepath)

        loader1 = RolloutLoader()
        loader1.load_new()  # consume all 4

        # New loader instance — simulates restart
        loader2 = RolloutLoader()
        new_reads = loader2.load_new()
        assert len(new_reads) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessing:
    def test_group_id_deterministic(self):
        from processing.preprocessing import _resolve_group_id

        r = _make_rollout(prompt="Fixed prompt")
        gid1 = _resolve_group_id(r)
        gid2 = _resolve_group_id(r)
        assert gid1 == gid2

    def test_same_prompt_same_group_id(self):
        from processing.preprocessing import _resolve_group_id

        r1 = _make_rollout(prompt="Same prompt")
        r2 = _make_rollout(prompt="Same prompt")
        assert _resolve_group_id(r1) == _resolve_group_id(r2)

    def test_different_prompts_different_group_ids(self):
        from processing.preprocessing import _resolve_group_id

        r1 = _make_rollout(prompt="Prompt A")
        r2 = _make_rollout(prompt="Prompt B")
        assert _resolve_group_id(r1) != _resolve_group_id(r2)

    def test_sample_weight_default_one(self):
        from processing.preprocessing import _resolve_sample_weight

        r = _make_rollout()
        assert _resolve_sample_weight(r) == 1.0

    def test_preprocess_attaches_group_id(self):
        from processing.preprocessing import preprocess

        rollouts = [_make_rollout() for _ in range(3)]
        preprocessed = preprocess(rollouts)
        for r in preprocessed:
            assert r.group_id is not None
            assert len(r.group_id) == 8  # 8-char hex digest

    def test_preprocess_returns_same_count(self):
        from processing.preprocessing import preprocess

        rollouts = [_make_rollout() for _ in range(5)]
        result = preprocess(rollouts)
        assert len(result) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 4. RolloutBuffer
# ─────────────────────────────────────────────────────────────────────────────


class TestRolloutBuffer:
    def test_flush_returns_correct_batch_count(self):
        from buffer.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer(batch_size=4)
        for _ in range(10):
            buf.add(_make_rollout())
        batches = buf.flush()
        assert len(batches) == 2  # 10 // 4 = 2 full batches

    def test_remainder_stays_in_buffer(self):
        from buffer.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer(batch_size=4)
        for _ in range(6):
            buf.add(_make_rollout())
        buf.flush()
        assert buf.pending_count() == 2  # 6 - 4 = 2 pending

    def test_empty_buffer_flush_returns_empty(self):
        from buffer.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer(batch_size=4)
        assert buf.flush() == []

    def test_batch_size_exact_multiple(self):
        from buffer.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer(batch_size=4)
        for _ in range(8):
            buf.add(_make_rollout())
        batches = buf.flush()
        assert len(batches) == 2
        assert buf.pending_count() == 0

    def test_batch_contains_correct_rollout_count(self):
        from buffer.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer(batch_size=5)
        for _ in range(5):
            buf.add(_make_rollout())
        batch = buf.flush()[0]
        assert batch.size() == 5

    def test_batch_reward_mean(self):
        from buffer.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer(batch_size=4)
        for rwd in [0.2, 0.4, 0.6, 0.8]:
            buf.add(_make_rollout(reward=rwd))
        batch = buf.flush()[0]
        assert abs(batch.reward_mean() - 0.5) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reward processing — normalisation guard
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardProcessing:
    def test_no_normalisation_when_already_normalised(self):
        import config

        config.cfg.reward_already_normalized = True
        from processing.reward_processing import process_rewards

        rewards = [0.1, 0.5, 0.9]
        result = process_rewards(rewards)
        assert result == rewards  # unchanged

    def test_z_score_applied_when_raw(self):
        import config

        config.cfg.reward_already_normalized = False
        config.cfg.normalize_rewards = True
        from processing import reward_processing
        import importlib

        importlib.reload(reward_processing)
        from processing.reward_processing import process_rewards

        rewards = [1.0, 2.0, 3.0]
        result = process_rewards(rewards)
        mean = sum(result) / len(result)
        assert abs(mean) < 1e-6  # mean ≈ 0 after z-score

    def test_passthrough_when_no_normalisation_configured(self):
        import config

        config.cfg.reward_already_normalized = False
        config.cfg.normalize_rewards = False
        from processing.reward_processing import process_rewards

        rewards = [0.3, 0.7]
        result = process_rewards(rewards)
        assert result == rewards


# ─────────────────────────────────────────────────────────────────────────────
# 6. Advantage computation
# ─────────────────────────────────────────────────────────────────────────────


class TestAdvantageComputation:
    def _make_batch(self, rewards, value_estimates=None, group_ids=None):
        from buffer.rollout_buffer import RolloutBatch

        n = len(rewards)
        rollouts = []
        for i, rwd in enumerate(rewards):
            r = _make_rollout(
                reward=rwd,
                value_estimate=(value_estimates[i] if value_estimates else 0.0),
            )
            if group_ids:
                r.group_id = group_ids[i]
            rollouts.append(r)
        return RolloutBatch(rollouts=rollouts)

    def test_standard_mode(self):
        import config

        config.cfg.advantage_mode = "standard"
        config.cfg.reward_already_normalized = True
        from processing.advantage import compute_advantages

        rewards = [0.4, 0.6]
        values = [0.1, 0.2]
        batch = self._make_batch(rewards, value_estimates=values)
        advs, rets = compute_advantages(batch)
        assert abs(advs[0] - (0.4 - 0.1)) < 1e-6
        assert abs(advs[1] - (0.6 - 0.2)) < 1e-6

    def test_grpo_mode_same_group(self):
        import config

        config.cfg.advantage_mode = "grpo"
        config.cfg.grpo_min_group_size = 2
        config.cfg.reward_already_normalized = True
        from processing.advantage import compute_advantages

        rewards = [0.4, 0.8]
        group_ids = ["grp1", "grp1"]
        batch = self._make_batch(rewards, group_ids=group_ids)
        advs, rets = compute_advantages(batch)
        # Within same group: mean=0.6, std=0.2
        # adv[0] = (0.4 - 0.6) / (0.2 + 1e-8)  ≈ -1
        # adv[1] = (0.8 - 0.6) / (0.2 + 1e-8)  ≈ +1
        assert advs[0] < 0 and advs[1] > 0

    def test_grpo_small_group_zeroed(self):
        import config

        config.cfg.advantage_mode = "grpo"
        config.cfg.grpo_min_group_size = 3
        config.cfg.reward_already_normalized = True
        from processing.advantage import compute_advantages

        rewards = [0.5, 0.7]
        group_ids = ["tiny", "tiny"]  # only 2 members < min_group_size=3
        batch = self._make_batch(rewards, group_ids=group_ids)
        advs, rets = compute_advantages(batch)
        assert advs[0] == 0.0
        assert advs[1] == 0.0

    def test_returns_equal_rewards(self):
        import config

        config.cfg.advantage_mode = "standard"
        config.cfg.reward_already_normalized = True
        from processing.advantage import compute_advantages

        rewards = [0.3, 0.9]
        batch = self._make_batch(rewards)
        _, rets = compute_advantages(batch)
        assert rets == rewards

    def test_hybrid_mode_blends(self):
        import config

        config.cfg.advantage_mode = "hybrid"
        config.cfg.hybrid_advantage_lambda = 0.5
        config.cfg.grpo_min_group_size = 2
        config.cfg.reward_already_normalized = True
        from processing.advantage import compute_advantages

        rewards = [0.4, 0.8]
        group_ids = ["g1", "g1"]
        batch = self._make_batch(
            rewards, value_estimates=[0.2, 0.2], group_ids=group_ids
        )
        advs, _ = compute_advantages(batch)
        assert len(advs) == 2
