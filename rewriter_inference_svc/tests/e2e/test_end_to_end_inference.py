"""End-to-end tests for rewriter HTTP API using ASGI transport."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
import torch

from rewriter_inference_svc.model_loader import clear_cache

pytestmark = pytest.mark.e2e


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
                "Extract all diagnosis codes from the note and return only a JSON list "
                "of strings without explanations."
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


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def _make_async_client():
    from rewriter_inference_svc.app import app

    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


class TestEndToEndInference:
    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    @pytest.mark.asyncio
    async def test_rewrite_prompt_endpoint(self, mock_load: MagicMock, mock_save: MagicMock) -> None:
        mock_load.return_value = _build_mock_bundle()
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

    @pytest.mark.asyncio
    async def test_empty_note_returns_422(self) -> None:
        async with _make_async_client() as client:
            response = await client.post("/rewrite_prompt", json={"clinical_note": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_field_returns_422(self) -> None:
        async with _make_async_client() as client:
            response = await client.post("/rewrite_prompt", json={})
        assert response.status_code == 422
