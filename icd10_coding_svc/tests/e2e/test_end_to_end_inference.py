"""
End-to-end tests for icd10_coding_svc.

Prerequisites:
  - The service must be running locally on port 8001.
  - A GPU must be available (skipped otherwise).
  - data/diagnoses.csv must contain real note_ids.
"""

import glob
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

BASE_URL = "http://localhost:8001"
SAMPLE_NOTE_ID = "12345"

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required for e2e tests")


def _service_is_live() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


@pytest.fixture(autouse=True)
def skip_if_service_down():
    if not _service_is_live():
        pytest.skip("icd10_coding_svc not running on port 8001")


class TestFullInferenceCycle:
    def test_full_inference_cycle(self):
        payload = {
            "note_id": SAMPLE_NOTE_ID,
            "original_prompt": "Extract ICD-10 codes from: Patient presents with chest pain.",
            "rewritten_prompt": "Step 1: Identify chief complaint. Step 2: Map diagnoses to ICD-10.",
        }
        resp = requests.post(f"{BASE_URL}/generate_codes", json=payload, timeout=120)
        assert resp.status_code == 200
        body = resp.json()
        assert body["note_id"] == SAMPLE_NOTE_ID
        assert isinstance(body["enh_codes"], list)
        assert isinstance(body["org_codes"], list)
        assert isinstance(body["gt_codes"], list)
        assert isinstance(body["parsing_success"], bool)


class TestGtCodesPersistedToDisk:
    def test_gt_codes_file_exists(self):
        from config import GT_CODES_PATH
        import json

        path = os.path.join(GT_CODES_PATH, f"{SAMPLE_NOTE_ID}.json")
        assert os.path.isfile(path), f"Expected {path} to exist after inference"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data.get("gt_codes"), list)


class TestInferenceOutputPersistedToDisk:
    def test_output_file_exists(self):
        from config import OUTPUT_PATH
        import json

        pattern = os.path.join(OUTPUT_PATH, f"{SAMPLE_NOTE_ID}_*.json")
        files = glob.glob(pattern)
        assert len(files) >= 1, f"Expected at least one output file matching {pattern}"
        with open(files[0]) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "enh_codes" in data
        assert "org_codes" in data
        assert "gt_codes" in data


class TestHealthEndpointLive:
    def test_health_returns_frozen(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["weights_frozen"] is True
        assert body["model"] == "m42-health/Llama3-Med42-8B"
