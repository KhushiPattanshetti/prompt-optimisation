"""End-to-end test for the rewriter inference service.

Steps:
1. Start the FastAPI service (httpx AsyncClient + ASGITransport).
2. Send an HTTP POST request.
3. Verify response structure contains rewritten_prompt, log_prob_old, value_estimate.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model_loader import clear_cache


def _build_mock_model_and_tokenizer(
    vocab_size: int = 50,
    input_len: int = 4,
    gen_len: int = 6,
    hidden_dim: int = 16,
):
    input_ids = torch.randint(0, vocab_size, (1, input_len))
    generated_ids = torch.cat(
        [input_ids, torch.randint(0, vocab_size, (1, gen_len))], dim=1
    )

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
    mock_tokenizer.decode.return_value = "Rewritten clinical prompt for evaluation."

    logits = torch.randn(1, input_len + gen_len, vocab_size)
    hidden = torch.randn(1, input_len, hidden_dim)

    fwd_output = MagicMock()
    fwd_output.logits = logits
    fwd_output.hidden_states = [hidden]

    mock_model = MagicMock()
    mock_model.generate.return_value = generated_ids
    mock_model.side_effect = lambda *a, **kw: fwd_output
    mock_model.value_head = MagicMock(return_value=torch.tensor([0.42]))
    mock_model.v_head = None
    mock_model.score = None

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    return mock_model, mock_tokenizer


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def _make_async_client():
    """Create an httpx async client bound to the FastAPI app via ASGI transport."""
    from app import app
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


class TestEndToEndInference:
    """End-to-end tests exercising the full HTTP → response cycle."""

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    @pytest.mark.asyncio
    async def test_rewrite_prompt_endpoint(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """POST /rewrite_prompt should return 200 with correct response fields."""
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_save.return_value = Path("/tmp/dummy.json")

        async with _make_async_client() as client:
            response = await client.post(
                "/rewrite_prompt",
                json={"clinical_note": "Patient presents with chest pain and shortness of breath."},
            )

        assert response.status_code == 200
        body = response.json()
        assert "rewritten_prompt" in body
        assert "log_prob_old" in body
        assert "value_estimate" in body
        assert isinstance(body["rewritten_prompt"], str)
        assert isinstance(body["log_prob_old"], float)
        assert isinstance(body["value_estimate"], float)

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    @pytest.mark.asyncio
    async def test_empty_note_returns_422(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """An empty clinical_note should trigger a 422 validation error."""
        async with _make_async_client() as client:
            response = await client.post("/rewrite_prompt", json={"clinical_note": ""})
        assert response.status_code == 422

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    @pytest.mark.asyncio
    async def test_missing_field_returns_422(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """A missing clinical_note field should trigger a 422 validation error."""
        async with _make_async_client() as client:
            response = await client.post("/rewrite_prompt", json={})
        assert response.status_code == 422
