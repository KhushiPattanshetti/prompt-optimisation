"""Integration tests for rewriter inference pipeline."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from rewriter_inference_svc.inference_engine import run_inference
from rewriter_inference_svc.model_loader import clear_cache


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    clear_cache()
    yield
    clear_cache()


def _build_mock_bundle(hidden_dim: int = 16):
    class TokenizerStub:
        eos_token_id = 0

        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": torch.randint(0, 20, (1, 4)),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

        def decode(self, _ids, skip_special_tokens=True):
            return (
                "Extract all ICD-10-CM diagnosis codes from the clinical note, "
                "identify all active conditions and relevant history, and output only "
                "a JSON list of code strings with no explanation."
            )

    model = MagicMock()
    prompt_ids = torch.randint(0, 20, (1, 4))
    model.generate.return_value = torch.cat([prompt_ids, torch.randint(0, 20, (1, 4))], dim=1)

    logits = torch.randn(1, 8, 24)
    hidden = torch.randn(1, 8, hidden_dim)
    model.side_effect = lambda *a, **k: SimpleNamespace(logits=logits, hidden_states=[hidden])

    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])

    value_head = torch.nn.Linear(hidden_dim, 1, bias=False)
    return model, TokenizerStub(), value_head


class TestServicePipeline:
    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_pipeline_returns_all_fields(self, mock_load: MagicMock, mock_save: MagicMock) -> None:
        mock_load.return_value = _build_mock_bundle()
        mock_save.return_value = Path("/tmp/dummy.json")

        result = run_inference("Patient presents with chest pain and shortness of breath.")

        assert "rewritten_prompt" in result
        assert "log_prob_old" in result
        assert "value_estimate" in result

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_pipeline_types(self, mock_load: MagicMock, mock_save: MagicMock) -> None:
        mock_load.return_value = _build_mock_bundle()
        mock_save.return_value = Path("/tmp/dummy.json")

        result = run_inference("Patient with acute onset headache.")

        assert isinstance(result["rewritten_prompt"], str)
        assert isinstance(result["log_prob_old"], float)
        assert isinstance(result["value_estimate"], float)

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_save_output_called(self, mock_load: MagicMock, mock_save: MagicMock) -> None:
        mock_load.return_value = _build_mock_bundle()
        mock_save.return_value = Path("/tmp/dummy.json")

        run_inference("Patient reports dizziness.")
        mock_save.assert_called_once()
