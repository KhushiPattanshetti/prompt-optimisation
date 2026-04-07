"""Unit tests for rewriter model_loader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import rewriter_inference_svc.model_loader as model_loader


def test_find_latest_checkpoint_dir_returns_highest_index(tmp_path, monkeypatch):
    ckpt_root = tmp_path / "rl_checkpoints"
    ckpt_root.mkdir()
    (ckpt_root / "checkpoint_0002").mkdir()
    (ckpt_root / "checkpoint_0010").mkdir()
    (ckpt_root / "not_a_checkpoint").mkdir()

    monkeypatch.setattr(model_loader, "RL_CHECKPOINT_PATH", ckpt_root)

    latest = model_loader._find_latest_checkpoint_dir()
    assert latest == ckpt_root / "checkpoint_0010"


def test_build_value_head_loads_compatible_state_dict(tmp_path):
    ckpt_dir = tmp_path / "checkpoint_0001"
    ckpt_dir.mkdir()
    value_path = ckpt_dir / "value_head.pt"

    source_head = model_loader.ValueHead(hidden_size=model_loader.VALUE_HEAD_HIDDEN_SIZE, dropout=0.1)
    randomized_state = {
        name: torch.randn_like(tensor)
        for name, tensor in source_head.state_dict().items()
    }
    torch.save(randomized_state, value_path)

    head = model_loader._build_value_head(torch.device("cpu"), ckpt_dir)
    loaded_state = head.state_dict()
    for key, tensor in randomized_state.items():
        assert key in loaded_state
        assert tuple(loaded_state[key].shape) == tuple(tensor.shape)


def test_build_value_head_fails_fast_for_legacy_linear_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoint_0001"
    ckpt_dir.mkdir()
    value_path = ckpt_dir / "value_head.pt"

    torch.save({"weight": torch.randn(1, model_loader.VALUE_HEAD_HIDDEN_SIZE)}, value_path)

    try:
        model_loader._build_value_head(torch.device("cpu"), ckpt_dir)
        assert False, "Expected RuntimeError for legacy single-linear value_head checkpoint"
    except RuntimeError as exc:
        assert "legacy single-linear value_head checkpoint" in str(exc)


def test_migrate_value_head_checkpoints_writes_marker(tmp_path, monkeypatch):
    ckpt_root = tmp_path / "rl_checkpoints"
    ckpt_root.mkdir()
    ckpt_dir = ckpt_root / "checkpoint_0001"
    ckpt_dir.mkdir()

    source_head = model_loader.ValueHead(hidden_size=model_loader.VALUE_HEAD_HIDDEN_SIZE, dropout=0.1)
    state = {
        f"module.{name}": tensor
        for name, tensor in source_head.state_dict().items()
    }
    torch.save(state, ckpt_dir / "value_head.pt")

    monkeypatch.setattr(model_loader, "RL_CHECKPOINT_PATH", ckpt_root)
    model_loader._migrate_value_head_checkpoints_once()

    marker = ckpt_root / ".value_head_migration_v2.json"
    assert marker.exists()

    migrated_state = torch.load(ckpt_dir / "value_head.pt", map_location="cpu")
    assert all(not key.startswith("module.") for key in migrated_state.keys())


def test_load_model_caches_instances(monkeypatch):
    model_loader.clear_cache()

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    base_model = MagicMock()
    base_model.config = MagicMock()
    base_model.gradient_checkpointing_disable = MagicMock()
    base_model.disable_input_require_grads = MagicMock()
    base_model.eval = MagicMock()
    base_param = torch.nn.Parameter(torch.zeros(1))
    base_model.parameters.return_value = iter([base_param])

    monkeypatch.setattr(model_loader, "_find_latest_checkpoint_dir", lambda: None)
    monkeypatch.setattr(model_loader, "_migrate_value_head_checkpoints_once", lambda: None)

    with patch("rewriter_inference_svc.model_loader.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("rewriter_inference_svc.model_loader.AutoModelForCausalLM.from_pretrained", return_value=base_model):
        model_1, tok_1, vh_1 = model_loader.load_model()
        model_2, tok_2, vh_2 = model_loader.load_model()

    assert model_1 is base_model
    assert model_1 is model_2
    assert tok_1 is tok_2
    assert vh_1 is vh_2

    model_loader.clear_cache()
