import pytest
from fastapi.testclient import TestClient

from reward_metrics_svc.main import app, distance_between, calculate_reward, ICD10_GRAPH, MAX_DEPTH

client = TestClient(app)


def test_distance_between_basic():
    a = ["A01.1"]
    b = ["A01.2"]
    dist = distance_between(a, b, ICD10_GRAPH, MAX_DEPTH)
    assert 0 <= dist <= 1


def test_distance_between_empty():
    a = []
    b = ["A01.2"]
    dist = distance_between(a, b, ICD10_GRAPH, MAX_DEPTH)
    assert dist == 1.0


def test_calculate_reward_better_enh():
    gt = ["A01.1"]
    enh = ["A01.1"]
    org = ["A01.2"]
    reward = calculate_reward(gt, enh, org)
    assert reward > 0


def test_calculate_reward_better_org():
    gt = ["A01.2"]
    enh = ["A01.1"]
    org = ["A01.2"]
    reward = calculate_reward(gt, enh, org)
    assert reward < 0


def test_calculate_reward_equal():
    gt = ["A01.1"]
    enh = ["A01.2"]
    org = ["A01.2"]
    reward = calculate_reward(gt, enh, org)
    # Under fixed-weight shaping, exact mismatch should no longer receive positive reward.
    assert reward <= 0.0


def test_api_reward():
    payload = {"gt_codes": ["A01.1"], "enh_codes": ["A01.1"], "org_codes": ["A01.2"]}
    resp = client.post("/reward", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reward" in data
    assert -1.0 <= data["reward"] <= 1.0


def test_observability_endpoint_has_queue_metrics():
    resp = client.get("/observability")
    assert resp.status_code == 200
    data = resp.json()
    assert "rollout_drop_rate" in data
    assert "queue_lag_seconds" in data
    assert "enqueue_attempts" in data
    assert "acked_rollouts" in data
    assert data["rollout_drop_rate"] >= 0.0


def test_observability_reset_endpoint():
    client.post("/reward", json={"gt_codes": ["A01.1"], "enh_codes": ["A01.1"], "org_codes": ["A01.2"]})

    before = client.get("/observability").json()
    assert before["reward_count"] >= 1

    reset_resp = client.post("/observability/reset")
    assert reset_resp.status_code == 200

    after = client.get("/observability").json()
    assert after["reward_count"] == 0
    assert after["enqueue_attempts"] == 0
    assert after["dropped_rollouts"] == 0
    assert after["transport_consecutive_failures"] == 0
