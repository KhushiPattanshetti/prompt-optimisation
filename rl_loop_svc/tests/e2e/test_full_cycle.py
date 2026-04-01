"""
End-to-end tests: full RL cycle with mocked model components.

Verifies the entire COLLECT → TRAIN → CHECKPOINT → IDLE state machine
using lightweight stub models (no actual transformer weights needed).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from rl.lifecycle_manager import TrainerState
from rl.rollout_buffer import RolloutBuffer
from rl.training_loop import TrainingLoop
from storage.checkpoint_manager import CheckpointManager
from storage.rollout_loader import RolloutLoader


# ── Stub model helpers ────────────────────────────────────────────────────────


class _StubPolicyModel(nn.Module):
    """Minimal stub that avoids loading real transformer weights."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.model = self  # checkpoint_manager calls self.model.state_dict()

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        # Route through self.linear so output has a grad_fn
        bias = self.linear(torch.ones(B, 1)) * 0.0  # (B, 1) — keeps grad graph
        lp = torch.full((B, max(T - 1, 1)), -5.0) + bias
        hidden = torch.zeros(B, T, 16) + bias.unsqueeze(2) * 0.0
        return lp, hidden

    def get_sequence_log_prob(self, input_ids, attention_mask=None):
        B = input_ids.shape[0]
        bias = self.linear(torch.ones(B, 1)) * 0.0
        return torch.full((B,), -5.0) + bias.squeeze(1)

    def tokenize(self, texts):
        ids = torch.ones(len(texts), 8, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def parameters(self, recurse=True):
        return self.linear.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        return self.linear.state_dict(*args, **kwargs)


class _StubReferenceModel:
    def get_sequence_log_prob(self, input_ids, attention_mask=None):
        B = input_ids.shape[0]
        return torch.full((B,), -5.5)


class _StubValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 1)

    def forward(self, hidden_states):
        # Route through self.linear so output participates in autograd
        pooled = hidden_states.mean(dim=1)  # (B, 16)
        return self.linear(pooled).squeeze(-1)  # (B,)

    def parameters(self, recurse=True):
        return self.linear.parameters(recurse)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rollout_entry():
    return {
        "original_prompt": "Patient with chest pain",
        "rewritten_prompt": "Patient presenting with acute chest pain",
        "reward": 0.7,
        "log_prob_old": -5.0,
        "value_estimate": 0.4,
    }


@pytest.fixture
def populated_rollouts_dir(tmp_path, rollout_entry):
    d = tmp_path / "rollouts"
    d.mkdir()
    batch = {"rollouts": [rollout_entry] * 4}
    (d / "batch_001.json").write_text(json.dumps(batch))
    return d


@pytest.fixture
def ckpt_dir(tmp_path):
    d = tmp_path / "rl_checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def training_loop(populated_rollouts_dir, ckpt_dir):
    policy_model = _StubPolicyModel()
    reference_model = _StubReferenceModel()
    value_head = _StubValueHead()

    return TrainingLoop(
        rollout_loader=RolloutLoader(populated_rollouts_dir),
        checkpoint_manager=CheckpointManager(ckpt_dir, max_checkpoints=5),
        policy_model=policy_model,
        reference_model=reference_model,
        value_head=value_head,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestFullCycle:
    def test_run_once_returns_true_when_rollouts_present(self, training_loop):
        result = training_loop.run_once()
        assert result is True

    def test_run_once_returns_false_when_no_new_rollouts(self, training_loop):
        training_loop.run_once()  # consumes the file
        result = training_loop.run_once()
        assert result is False

    def test_state_returns_to_idle_after_cycle(self, training_loop):
        training_loop.run_once()
        assert training_loop.lifecycle.state == TrainerState.IDLE

    def test_training_step_incremented(self, training_loop):
        from app.config import settings

        training_loop.run_once()
        assert training_loop.training_step == settings.ppo_epochs

    def test_checkpoint_created_after_cycle(self, training_loop, ckpt_dir):
        training_loop.run_once()
        ckpts = list(ckpt_dir.iterdir())
        assert len(ckpts) == 1
        ckpt = ckpts[0]
        assert (ckpt / "policy_model.pt").exists()
        assert (ckpt / "value_head.pt").exists()
        assert (ckpt / "optimizer.pt").exists()
        assert (ckpt / "training_state.json").exists()

    def test_rollouts_loaded_count_accumulates(self, populated_rollouts_dir, ckpt_dir):
        d = populated_rollouts_dir
        loop = TrainingLoop(
            rollout_loader=RolloutLoader(d),
            checkpoint_manager=CheckpointManager(ckpt_dir, max_checkpoints=5),
            policy_model=_StubPolicyModel(),
            reference_model=_StubReferenceModel(),
            value_head=_StubValueHead(),
        )
        loop.run_once()
        assert loop.rollouts_loaded == 4

    def test_multiple_checkpoint_files_increment_index(self, tmp_path):
        d = tmp_path / "rollouts"
        d.mkdir()
        ckpt_d = tmp_path / "rl_checkpoints"
        ckpt_d.mkdir()

        entry = {
            "original_prompt": "note",
            "rewritten_prompt": "better note",
            "reward": 0.5,
            "log_prob_old": -5.0,
            "value_estimate": 0.3,
        }

        loop = TrainingLoop(
            rollout_loader=RolloutLoader(d),
            checkpoint_manager=CheckpointManager(ckpt_d, max_checkpoints=5),
            policy_model=_StubPolicyModel(),
            reference_model=_StubReferenceModel(),
            value_head=_StubValueHead(),
        )

        # Write one file per cycle so each run_once() sees exactly one new file
        for i in range(3):
            (d / f"batch_{i:03d}.json").write_text(
                json.dumps({"rollouts": [entry] * 2})
            )
            loop.run_once()

        ckpts = sorted(ckpt_d.iterdir())
        assert len(ckpts) == 3
        assert ckpts[0].name == "checkpoint_0001"
        assert ckpts[2].name == "checkpoint_0003"
