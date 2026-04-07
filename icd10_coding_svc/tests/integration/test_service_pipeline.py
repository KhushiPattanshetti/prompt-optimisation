"""Integration tests for ICD API service pipeline."""

from unittest.mock import patch

import pytest

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


@pytest.fixture()
def client(monkeypatch):
    """Create a TestClient with model loading mocked during startup."""
    monkeypatch.setattr(
        "icd10_coding_svc.model_loader.load_model",
        lambda: (object(), object()),
    )

    from fastapi.testclient import TestClient
    from icd10_coding_svc.app import app

    with TestClient(app) as c:
        yield c


class TestPostGenerateCodesValidRequest:
    @patch("icd10_coding_svc.inference_engine.run_inference", return_value=MOCK_RESULT)
    def test_returns_200_with_valid_body(self, _mock_infer, client):
        resp = client.post("/generate_codes", json=VALID_REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert body["note_id"] == "12345"
        assert body["enh_codes"] == ["R07.9", "I20.9"]
        assert body["org_codes"] == ["R07.9"]
        assert body["gt_codes"] == ["R07.9", "I20.9", "R06.0"]
        assert body["parsing_success"] is True


class TestPostGenerateCodesMissingField:
    def test_returns_422_when_field_missing(self, client):
        bad_request = {
            "note_id": "12345",
            "original_prompt": "Extract ICD-10 codes...",
        }
        resp = client.post("/generate_codes", json=bad_request)
        assert resp.status_code == 422


class TestGetHealth:
    @patch("icd10_coding_svc.model_loader.get_cached_model", return_value=None)
    def test_returns_health_status(self, _mock_cached, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "m42-health/Llama3-Med42-8B"
        assert body["weights_frozen"] is False


class TestGetObservability:
    @patch(
        "icd10_coding_svc.inference_engine.get_observability_snapshot",
        return_value={
            "total_requests": 10,
            "parse_success_rate": 0.8,
            "parse_failure_taxonomy": {
                "joint_failure_modes": {"both_failed": 2},
                "enhanced_failure_reasons": {"empty_output": 1},
                "original_failure_reasons": {"no_valid_icd_pattern": 1},
            },
        },
    )
    def test_returns_observability_payload(self, _mock_obs, client):
        resp = client.get("/observability")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_requests"] == 10
        assert body["parse_success_rate"] == 0.8
        assert "parse_failure_taxonomy" in body
