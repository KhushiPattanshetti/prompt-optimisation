import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import app, distance_between, calculate_reward, ICD10_GRAPH

client = TestClient(app)


def test_distance_between_basic():
    a = ["A01.1"]
    b = ["A01.2"]
    dist = distance_between(a, b, ICD10_GRAPH)
    assert 0 <= dist <= 1


def test_distance_between_empty():
    a = []
    b = ["A01.2"]
    dist = distance_between(a, b, ICD10_GRAPH)
    assert dist == 1.0


def test_calculate_reward_better_enh():
    gt = ["A01.1"]
    enh = ["A01.1"]
    org = ["A01.2"]
    reward = calculate_reward(gt, enh, org, ICD10_GRAPH)
    assert reward > 0


def test_calculate_reward_better_org():
    gt = ["A01.2"]
    enh = ["A01.1"]
    org = ["A01.2"]
    reward = calculate_reward(gt, enh, org, ICD10_GRAPH)
    assert reward < 0


def test_calculate_reward_equal():
    gt = ["A01.1"]
    enh = ["A01.2"]
    org = ["A01.2"]
    reward = calculate_reward(gt, enh, org, ICD10_GRAPH)
    assert reward == 1.0 or reward < 0


def test_api_reward():
    payload = {"gt_codes": ["A01.1"], "enh_codes": ["A01.1"], "org_codes": ["A01.2"]}
    resp = client.post("/reward", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reward" in data
    assert -1.0 <= data["reward"] <= 1.0
