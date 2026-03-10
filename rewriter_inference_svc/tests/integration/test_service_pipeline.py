"""Integration test for the full inference pipeline.

Tests the flow: clinical_note → rewritten_prompt + PPO metrics
Uses mocked model/tokenizer to avoid requiring real weights.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from inference_engine import run_inference
from model_loader import clear_cache


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    """Ensure a fresh model cache for every test."""
    clear_cache()
    yield  # type: ignore[misc]
    clear_cache()


def _build_mock_model_and_tokenizer(
    vocab_size: int = 50,
    input_len: int = 4,
    gen_len: int = 6,
    hidden_dim: int = 16,
):
    """Create a pair of mock model & tokenizer that behave like the real ones."""
    input_ids = torch.randint(0, vocab_size, (1, input_len))
    generated_ids = torch.cat(
        [input_ids, torch.randint(0, vocab_size, (1, gen_len))], dim=1
    )

    # --- tokenizer ---
    tok_result = MagicMock()
    tok_result.__getitem__ = lambda self, key: {
        "input_ids": input_ids,
        "attention_mask": torch.ones(1, input_len),
    }[key]
    tok_result.to.return_value = tok_result
    tok_result.keys.return_value = ["input_ids", "attention_mask"]
    tok_result.__iter__ = lambda self: iter(["input_ids", "attention_mask"])

    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = lambda *a, **kw: tok_result
    mock_tokenizer.decode.return_value = "Summarize chest pain presentation with differential diagnosis."

    # --- model ---
    logits = torch.randn(1, input_len + gen_len, vocab_size)
    hidden = torch.randn(1, input_len, hidden_dim)

    fwd_output = MagicMock()
    fwd_output.logits = logits
    fwd_output.hidden_states = [hidden]

    mock_model = MagicMock()
    mock_model.generate.return_value = generated_ids
    mock_model.side_effect = lambda *a, **kw: fwd_output
    mock_model.value_head = MagicMock(return_value=torch.tensor([0.35]))
    mock_model.v_head = None
    mock_model.score = None

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    return mock_model, mock_tokenizer


class TestServicePipeline:
    """Full pipeline integration tests."""

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    def test_pipeline_returns_all_fields(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """The pipeline should return rewritten_prompt, log_prob_old, and value_estimate."""
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_save.return_value = Path("/tmp/dummy.json")

        result = run_inference("Patient presents with chest pain and shortness of breath.")

        assert "rewritten_prompt" in result
        assert "log_prob_old" in result
        assert "value_estimate" in result

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    def test_pipeline_types(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """All returned values should have the correct types."""
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_save.return_value = Path("/tmp/dummy.json")

        result = run_inference("Patient with acute onset headache.")

        assert isinstance(result["rewritten_prompt"], str)
        assert isinstance(result["log_prob_old"], float)
        assert isinstance(result["value_estimate"], float)

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    def test_pipeline_log_prob_is_nonpositive(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """log_prob_old should be ≤ 0 (sum of log probabilities)."""
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_save.return_value = Path("/tmp/dummy.json")

        result = run_inference("Elevated troponin levels detected.")

        assert result["log_prob_old"] <= 0.0

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    def test_save_output_called(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """The pipeline should persist inference output to disk."""
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_save.return_value = Path("/tmp/dummy.json")

        run_inference("Patient reports dizziness.")

        mock_save.assert_called_once()
