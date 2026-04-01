"""Unit tests for the inference engine module.

Tests:
- Generation returns rewritten_prompt
- log_prob_old is float
- value_estimate is float
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from inference_engine import _compute_log_prob, _compute_value_estimate, run_inference


class TestComputeLogProb:
    """Tests for _compute_log_prob."""

    def test_returns_float(self) -> None:
        """log_prob_old must be a float."""
        vocab_size = 10
        seq_len = 5
        input_length = 2

        # Build mock model that returns logits
        logits = torch.randn(1, seq_len, vocab_size)
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model = MagicMock(return_value=mock_output)

        full_ids = torch.randint(0, vocab_size, (1, seq_len))

        result = _compute_log_prob(mock_model, full_ids, input_length)

        assert isinstance(result, float)

    def test_log_prob_is_negative(self) -> None:
        """Sum of log probabilities should typically be negative."""
        vocab_size = 100
        seq_len = 8
        input_length = 3

        logits = torch.randn(1, seq_len, vocab_size)
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model = MagicMock(return_value=mock_output)

        full_ids = torch.randint(0, vocab_size, (1, seq_len))

        result = _compute_log_prob(mock_model, full_ids, input_length)

        # Log probs are ≤ 0; their sum should be ≤ 0
        assert result <= 0.0


class TestComputeValueEstimate:
    """Tests for _compute_value_estimate."""

    def test_returns_float_with_value_head(self) -> None:
        """value_estimate should be a float when a value_head exists."""
        hidden_dim = 16
        seq_len = 4

        hidden_states = [torch.randn(1, seq_len, hidden_dim)]
        mock_output = MagicMock()
        mock_output.hidden_states = hidden_states

        mock_value_head = MagicMock(return_value=torch.tensor([0.42]))

        mock_model = MagicMock(return_value=mock_output)
        mock_model.value_head = mock_value_head
        mock_model.v_head = None
        mock_model.score = None

        result = _compute_value_estimate(mock_model, torch.randint(0, 100, (1, seq_len)))

        assert isinstance(result, float)

    def test_returns_float_fallback(self) -> None:
        """value_estimate should still be a float even without an explicit value head."""
        hidden_dim = 16
        seq_len = 4

        hidden_states = [torch.randn(1, seq_len, hidden_dim)]
        mock_output = MagicMock()
        mock_output.hidden_states = hidden_states

        mock_model = MagicMock(return_value=mock_output)
        # No value_head, v_head, or score attribute
        mock_model.value_head = None
        mock_model.v_head = None
        mock_model.score = None

        result = _compute_value_estimate(mock_model, torch.randint(0, 100, (1, seq_len)))

        assert isinstance(result, float)


class TestRunInference:
    """Tests for the run_inference entry point."""

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    def test_returns_rewritten_prompt(
        self,
        mock_load_model: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """run_inference should return a dict containing 'rewritten_prompt'."""
        vocab_size = 50
        input_len = 3
        gen_len = 5
        hidden_dim = 16

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        input_ids = torch.randint(0, vocab_size, (1, input_len))
        mock_tokenizer.return_value = {"input_ids": input_ids, "attention_mask": torch.ones(1, input_len)}
        mock_tokenizer.return_value = MagicMock(
            **{
                "to.return_value": {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(1, input_len),
                },
            }
        )
        # Make tokenizer callable and subscriptable
        tok_result = MagicMock()
        tok_result.__getitem__ = lambda self, key: {
            "input_ids": input_ids,
            "attention_mask": torch.ones(1, input_len),
        }[key]
        tok_result.to.return_value = tok_result
        tok_result.keys.return_value = ["input_ids", "attention_mask"]
        tok_result.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
        mock_tokenizer.side_effect = lambda *a, **kw: tok_result
        mock_tokenizer.decode.return_value = "Rewritten clinical note."

        # Mock model
        generated_ids = torch.randint(0, vocab_size, (1, input_len + gen_len))
        generated_ids[:, :input_len] = input_ids

        logits = torch.randn(1, input_len + gen_len, vocab_size)
        hidden = torch.randn(1, input_len, hidden_dim)

        mock_model = MagicMock()
        mock_model.generate.return_value = generated_ids

        fwd_output = MagicMock()
        fwd_output.logits = logits
        fwd_output.hidden_states = [hidden]
        mock_model.side_effect = lambda *a, **kw: fwd_output
        mock_model.value_head = MagicMock(return_value=torch.tensor([0.5]))
        mock_model.v_head = None
        mock_model.score = None

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_save.return_value = Path("/tmp/dummy.json")

        result = run_inference("Patient has chest pain.")

        assert "rewritten_prompt" in result
        assert isinstance(result["rewritten_prompt"], str)
        assert isinstance(result["log_prob_old"], float)
        assert isinstance(result["value_estimate"], float)
