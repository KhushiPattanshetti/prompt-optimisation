"""Stress tests for rewriter inference service using ASGI transport."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
import torch

from rewriter_inference_svc.model_loader import clear_cache

pytestmark = pytest.mark.stress

CONCURRENT_REQUESTS = 100


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


class TestLoadInference:
    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_load: MagicMock, mock_save: MagicMock) -> None:
        mock_load.return_value = _build_mock_bundle()
        mock_save.return_value = Path("/tmp/dummy.json")

        from rewriter_inference_svc.app import app

        transport = httpx.ASGITransport(app=app)
        payload = {"clinical_note": "Patient presents with acute abdominal pain."}

        errors = 0
        latencies = []

        async def _send_request(client: httpx.AsyncClient):
            t0 = time.perf_counter()
            resp = await client.post("/rewrite_prompt", json=payload)
            elapsed = time.perf_counter() - t0
            return resp.status_code, elapsed

        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            tasks = [_send_request(client) for _ in range(CONCURRENT_REQUESTS)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                errors += 1
                continue
            status, elapsed = result
            latencies.append(elapsed)
            if status != 200:
                errors += 1

        error_rate = errors / CONCURRENT_REQUESTS
        assert error_rate == 0.0
