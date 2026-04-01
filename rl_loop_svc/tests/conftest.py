"""
Shared pytest fixtures for the rl_loop_svc test suite.
"""

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Rollout fixtures ──────────────────────────────────────────────────────────


def make_rollout_entry(
    reward: float = 0.5,
    log_prob_old: float = -5.0,
    value_estimate: float = 0.3,
) -> dict:
    return {
        "original_prompt": "Patient has fever and cough",
        "rewritten_prompt": "Patient presents with pyrexia and productive cough",
        "reward": reward,
        "log_prob_old": log_prob_old,
        "value_estimate": value_estimate,
    }


@pytest.fixture
def sample_rollout_entries() -> List[dict]:
    return [make_rollout_entry(reward=r) for r in [0.5, 0.8, -0.2, 0.1, 0.9]]


@pytest.fixture
def rollout_file(tmp_path, sample_rollout_entries) -> Path:
    """Write a single rollout JSON file to a temp directory."""
    f = tmp_path / "rollout_batch_001.json"
    f.write_text(json.dumps({"rollouts": sample_rollout_entries}))
    return f


@pytest.fixture
def rollouts_dir(tmp_path, sample_rollout_entries) -> Path:
    """Populate a temp directory with two rollout files."""
    d = tmp_path / "rollouts"
    d.mkdir()
    for i, chunk in enumerate([sample_rollout_entries[:3], sample_rollout_entries[3:]]):
        (d / f"rollout_batch_{i+1:03d}.json").write_text(
            json.dumps({"rollouts": chunk})
        )
    return d


@pytest.fixture
def checkpoints_dir(tmp_path) -> Path:
    d = tmp_path / "rl_checkpoints"
    d.mkdir()
    return d


# ── Tensor fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def rewards(batch_size) -> torch.Tensor:
    return torch.tensor([0.5, 0.8, -0.2, 0.1], dtype=torch.float32)


@pytest.fixture
def values(batch_size) -> torch.Tensor:
    return torch.tensor([0.4, 0.7, 0.0, 0.2], dtype=torch.float32)


@pytest.fixture
def log_probs_old(batch_size) -> torch.Tensor:
    return torch.tensor([-5.0, -4.5, -6.0, -5.5], dtype=torch.float32)
