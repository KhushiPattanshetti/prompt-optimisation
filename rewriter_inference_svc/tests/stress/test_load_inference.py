"""Stress test for the rewriter inference service.

Simulates 100 concurrent inference requests using asyncio.
Measures latency and error rate.
The service must handle concurrent requests without crashing.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model_loader import clear_cache

CONCURRENT_REQUESTS = 100


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
    mock_tokenizer.decode.return_value = "Stress-test rewritten prompt."

    logits = torch.randn(1, input_len + gen_len, vocab_size)
    hidden = torch.randn(1, input_len, hidden_dim)

    fwd_output = MagicMock()
    fwd_output.logits = logits
    fwd_output.hidden_states = [hidden]

    mock_model = MagicMock()
    mock_model.generate.return_value = generated_ids
    mock_model.side_effect = lambda *a, **kw: fwd_output
    mock_model.value_head = MagicMock(return_value=torch.tensor([0.33]))
    mock_model.v_head = None
    mock_model.score = None

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.side_effect = lambda: iter([mock_param])

    return mock_model, mock_tokenizer


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


class TestLoadInference:
    """Stress / load tests for the inference service."""

    @patch("inference_engine._save_output")
    @patch("inference_engine.load_model")
    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Service must handle 100 concurrent requests without crashing."""
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_save.return_value = Path("/tmp/dummy.json")

        from app import app

        transport = httpx.ASGITransport(app=app)
        payload = {"clinical_note": "Patient presents with acute abdominal pain."}

        errors = 0
        latencies: list = []

        async def _send_request(client: httpx.AsyncClient) -> tuple:
            t0 = time.perf_counter()
            resp = await client.post("/rewrite_prompt", json=payload)
            elapsed = time.perf_counter() - t0
            return resp.status_code, elapsed

        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            tasks = [_send_request(client) for _ in range(CONCURRENT_REQUESTS)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                errors += 1
            else:
                status, elapsed = r
                latencies.append(elapsed)
                if status != 200:
                    errors += 1

        error_rate = errors / CONCURRENT_REQUESTS
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0

        print(f"\n--- Stress Test Results ---")
        print(f"Total requests : {CONCURRENT_REQUESTS}")
        print(f"Errors         : {errors} ({error_rate:.1%})")
        print(f"Avg latency    : {avg_latency:.4f}s")
        print(f"P95 latency    : {p95_latency:.4f}s")

        # Assert no crashes — allow zero errors
        assert error_rate == 0.0, f"Error rate was {error_rate:.1%}, expected 0%"
