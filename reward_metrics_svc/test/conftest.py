"""
conftest.py – Shared pytest fixtures for reward_metrics_svc tests.

Inserts the service root into sys.path so test files can import
config, tree, similarity, metrics, reward, log_utils, schemas, app
without any per-file path manipulation.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Make the reward_metrics_svc/ directory importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A single TestClient instance reused across the entire test session."""
    return TestClient(app)


@pytest.fixture
def post_reward(client):
    """Helper fixture: POSTs to /compute_reward and returns the response body."""

    def _post(
        note_id,
        gt,
        enh,
        org,
        parsing_success=True,
        invalid_codes=None,
        duplicate_codes=None,
        state="",
        action="",
        log_prob_old=0.0,
        value_estimate=0.0,
    ):
        resp = client.post(
            "/compute_reward",
            json={
                "note_id": note_id,
                "gt_codes": gt,
                "enh_codes": enh,
                "org_codes": org,
                "parsing_success": parsing_success,
                "invalid_codes": invalid_codes or [],
                "duplicate_codes": duplicate_codes or [],
                "state": state,
                "action": action,
                "log_prob_old": log_prob_old,
                "value_estimate": value_estimate,
            },
        )
        resp.raise_for_status()
        return resp.json()

    return _post
