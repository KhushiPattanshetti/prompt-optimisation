"""Unit tests for model_loader module.

Tests:
- SFT checkpoint loads if RL checkpoint absent
- RL checkpoint loads when available
- Tokenizer loads correctly
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the service root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model_loader import _resolve_checkpoint_path, clear_cache, load_model


class TestResolveCheckpointPath:
    """Tests for the checkpoint resolution logic."""

    def test_sft_checkpoint_when_rl_empty(self, tmp_path: Path) -> None:
        """When rl_checkpoints/ is empty, should fall back to sft_checkpoints/."""
        rl_dir = tmp_path / "rl_checkpoints"
        rl_dir.mkdir()

        sft_dir = tmp_path / "sft_checkpoints"
        sft_dir.mkdir()
        sft_ckpt = sft_dir / "checkpoint-100"
        sft_ckpt.mkdir()

        with patch("model_loader.RL_CHECKPOINT_PATH", str(rl_dir)), \
             patch("model_loader.SFT_CHECKPOINT_PATH", str(sft_dir)):
            resolved = _resolve_checkpoint_path()

        assert resolved == sft_ckpt

    def test_rl_checkpoint_loads_when_available(self, tmp_path: Path) -> None:
        """When rl_checkpoints/ has checkpoints, should load the latest one."""
        rl_dir = tmp_path / "rl_checkpoints"
        rl_dir.mkdir()
        ckpt_old = rl_dir / "checkpoint-100"
        ckpt_old.mkdir()
        ckpt_new = rl_dir / "checkpoint-200"
        ckpt_new.mkdir()

        # Make checkpoint-200 newer
        os.utime(ckpt_old, (1000, 1000))
        os.utime(ckpt_new, (2000, 2000))

        sft_dir = tmp_path / "sft_checkpoints"
        sft_dir.mkdir()

        with patch("model_loader.RL_CHECKPOINT_PATH", str(rl_dir)), \
             patch("model_loader.SFT_CHECKPOINT_PATH", str(sft_dir)):
            resolved = _resolve_checkpoint_path()

        assert resolved == ckpt_new

    def test_raises_when_no_checkpoint_exists(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when no checkpoints exist at all."""
        rl_dir = tmp_path / "rl_checkpoints"
        sft_dir = tmp_path / "sft_checkpoints"

        with patch("model_loader.RL_CHECKPOINT_PATH", str(rl_dir)), \
             patch("model_loader.SFT_CHECKPOINT_PATH", str(sft_dir)):
            with pytest.raises(FileNotFoundError):
                _resolve_checkpoint_path()


class TestLoadModel:
    """Tests for the load_model function."""

    def setup_method(self) -> None:
        clear_cache()

    @patch("model_loader.AutoTokenizer")
    @patch("model_loader.AutoModelForCausalLM")
    @patch("model_loader._resolve_checkpoint_path")
    def test_tokenizer_loads_correctly(
        self,
        mock_resolve: MagicMock,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tokenizer should be loaded from the resolved checkpoint path."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        mock_resolve.return_value = ckpt

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        model, tokenizer = load_model()

        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            str(ckpt), trust_remote_code=True
        )
        assert tokenizer is mock_tokenizer

    @patch("model_loader.AutoTokenizer")
    @patch("model_loader.AutoModelForCausalLM")
    def test_model_loads_from_explicit_path(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When an explicit checkpoint_path is given, it should be used directly."""
        ckpt = tmp_path / "my_ckpt"
        ckpt.mkdir()

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model

        load_model(checkpoint_path=ckpt)

        mock_model_cls.from_pretrained.assert_called_once()
        call_args = mock_model_cls.from_pretrained.call_args
        assert call_args[0][0] == str(ckpt)
