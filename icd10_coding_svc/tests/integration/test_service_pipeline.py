import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Patch heavy dependencies before importing app
_mock_model = MagicMock()
_mock_tokenizer = MagicMock()


@pytest.fixture()
def client(monkeypatch):
    """Create a TestClient with startup dependencies mocked."""
    monkeypatch.setattr("model_loader.load_model", lambda: (_mock_model, _mock_tokenizer))
    monkeypatch.setattr("gt_fetcher.init_datasets", lambda: None)

    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app) as c:
        yield c

VALID_REQUEST = {
    "note_id": "12345",
    "original_prompt": "Extract ICD-10 codes from: Patient has chest pain...",
    "rewritten_prompt": "Step 1: Identify symptoms... Step 2: Map to ICD-10...",
}

MOCK_RESULT = {
    "note_id": "12345",
    "enh_codes": ["R07.9", "I20.9"],
    "org_codes": ["R07.9"],
    "gt_codes": ["R07.9", "I20.9", "R06.0"],
    "enh_raw_output": '["R07.9", "I20.9"]',
    "org_raw_output": '["R07.9"]',
    "parsing_success": True,
}


class TestPostGenerateCodesValidRequest:
    @patch("inference_engine.run_inference", return_value=MOCK_RESULT)
    def test_returns_200_with_valid_body(self, mock_infer, client):
        resp = client.post("/generate_codes", json=VALID_REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert body["note_id"] == "12345"
        assert body["enh_codes"] == ["R07.9", "I20.9"]
        assert body["org_codes"] == ["R07.9"]
        assert body["gt_codes"] == ["R07.9", "I20.9", "R06.0"]
        assert body["parsing_success"] is True
        assert "enh_raw_output" in body
        assert "org_raw_output" in body


class TestPostGenerateCodesMissingField:
    def test_returns_422_when_field_missing(self, client):
        bad_request = {
            "note_id": "12345",
            "original_prompt": "Extract ICD-10 codes...",
            # rewritten_prompt intentionally missing
        }
        resp = client.post("/generate_codes", json=bad_request)
        assert resp.status_code == 422


class TestGetHealth:
    def test_returns_health_status(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "m42-health/Llama3-Med42-8B"
        assert body["weights_frozen"] is True


class TestForwardingCalledAfterResponse:
    @patch("inference_engine._forward_to_reward_service")
    @patch("inference_engine.run_inference", return_value=MOCK_RESULT)
    def test_forwarding_invoked(self, mock_infer, mock_fwd, client):
        resp = client.post("/generate_codes", json=VALID_REQUEST)
        assert resp.status_code == 200
        mock_infer.assert_called_once_with(
            note_id="12345",
            original_prompt=VALID_REQUEST["original_prompt"],
            rewritten_prompt=VALID_REQUEST["rewritten_prompt"],
        )


class TestForwardingFailureDoesNotPropagate:
    @patch("inference_engine._forward_to_reward_service", side_effect=Exception("network error"))
    @patch("inference_engine.run_inference", return_value=MOCK_RESULT)
    def test_response_still_200_on_forwarding_error(self, mock_infer, mock_fwd, client):
        resp = client.post("/generate_codes", json=VALID_REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert body["note_id"] == "12345"
