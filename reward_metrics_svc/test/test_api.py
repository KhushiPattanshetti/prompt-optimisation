"""
test_api.py – Integration tests for the FastAPI app  (spec §14-16, v2)

Uses the session-scoped TestClient fixture from conftest.py.
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoints
# ─────────────────────────────────────────────────────────────────────────────


def test_health_get(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["tree_loaded"] is True
    assert body["node_count"] > 0


def test_health_post(client):
    resp = client.post("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Response shape  (spec §14.1)
# ─────────────────────────────────────────────────────────────────────────────


def test_response_contains_top_level_fields(post_reward):
    body = post_reward(note_id="shape_test", gt=["A01.1"], enh=["A01.2"], org=["B01"])
    assert {"note_id", "reward", "metrics"} <= set(body)


def test_metrics_has_required_keys(post_reward):
    body = post_reward("keys_test", ["A01.1"], ["A01.2"], ["B01"])
    m = body["metrics"]
    required = {
        "D_enh",
        "D_org",
        "delta_D",
        "reward_components",
        "components_enh",
        "components_org",
        "diagnostics",
    }
    assert required <= set(m)


def test_reward_components_keys(post_reward):
    body = post_reward("rc_test", ["A01.1"], ["A01.2"], ["B01"])
    rc = body["metrics"]["reward_components"]
    assert {"R_tree", "R_exact", "R_structure"} <= set(rc)


def test_diagnostics_keys(post_reward):
    body = post_reward("diag_test", ["A01.1"], ["A01.2"], ["B01"])
    diag = body["metrics"]["diagnostics"]
    assert {
        "worst_gt_coverage_code",
        "worst_pred_match_code",
        "invalid_codes",
        "duplicate_codes",
    } <= set(diag)


def test_component_metrics_keys(post_reward):
    body = post_reward("comp_test", ["A01.1"], ["A01.2"], ["B01"])
    m = body["metrics"]
    for comp_key in ("components_enh", "components_org"):
        assert {"D_set", "P_cov", "P_extra", "P_card"} <= set(m[comp_key])


def test_note_id_echoed(post_reward):
    body = post_reward("note_echo_42", ["A01.1"], ["A01.1"], ["B01"])
    assert body["note_id"] == "note_echo_42"


def test_reward_in_valid_range(post_reward):
    body = post_reward("range_test", ["A01.1"], ["A01.2"], ["B01"])
    assert -1.0 <= body["reward"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Reward components – spot checks
# ─────────────────────────────────────────────────────────────────────────────


def test_r_exact_is_one_on_exact_match(post_reward):
    body = post_reward("r_exact_test", ["A01.1"], ["A01.1"], ["B01"])
    assert body["metrics"]["reward_components"]["R_exact"] == pytest.approx(1.0)


def test_r_structure_is_one_on_clean_input(post_reward):
    body = post_reward("r_struct_clean", ["A01.1"], ["A01.1"], ["B01"])
    assert body["metrics"]["reward_components"]["R_structure"] == pytest.approx(1.0)


def test_r_structure_minus_one_on_parse_fail(client):
    resp = client.post(
        "/compute_reward",
        json={
            "note_id": "parse_fail_struct",
            "gt_codes": ["A01.1"],
            "enh_codes": ["A01.2"],
            "org_codes": ["B01"],
            "parsing_success": False,
            "invalid_codes": [],
            "duplicate_codes": [],
        },
    )
    assert resp.status_code == 200
    rc = resp.json()["metrics"]["reward_components"]
    assert rc["R_structure"] == pytest.approx(-1.0)


def test_r_structure_reduced_by_invalid_codes(post_reward):
    body_clean = post_reward("inv_clean", ["A01.1"], ["A01.1", "A01.2"], ["B01"])
    body_dirty = post_reward(
        "inv_dirty",
        ["A01.1"],
        ["A01.1", "A01.2"],
        ["B01"],
        invalid_codes=["A01.2"],
    )
    assert (
        body_clean["metrics"]["reward_components"]["R_structure"]
        > body_dirty["metrics"]["reward_components"]["R_structure"]
    )


def test_r_structure_reduced_by_duplicate_codes(post_reward):
    body_clean = post_reward("dup_clean", ["A01.1"], ["A01.1", "A01.2"], ["B01"])
    body_dirty = post_reward(
        "dup_dirty",
        ["A01.1"],
        ["A01.1", "A01.2"],
        ["B01"],
        duplicate_codes=["A01.1"],
    )
    assert (
        body_clean["metrics"]["reward_components"]["R_structure"]
        > body_dirty["metrics"]["reward_components"]["R_structure"]
    )


def test_diagnostics_invalid_codes_echoed(post_reward):
    body = post_reward(
        "inv_echo",
        ["A01.1"],
        ["A01.1"],
        ["B01"],
        invalid_codes=["X99"],
    )
    assert "X99" in body["metrics"]["diagnostics"]["invalid_codes"]


def test_diagnostics_duplicate_codes_echoed(post_reward):
    body = post_reward(
        "dup_echo",
        ["A01.1"],
        ["A01.1"],
        ["B01"],
        duplicate_codes=["A01.1"],
    )
    assert "A01.1" in body["metrics"]["diagnostics"]["duplicate_codes"]


# ─────────────────────────────────────────────────────────────────────────────
# §19 spec scenarios
# ─────────────────────────────────────────────────────────────────────────────


def test_spec_improvement(post_reward):
    """§19.1 enh closer to gt than org → reward > 0"""
    body = post_reward(
        note_id="spec_19_1",
        gt=["A01.1"],
        enh=["A01.2"],
        org=["B01"],
    )
    assert body["reward"] > 0.0
    assert body["metrics"]["delta_D"] > 0.0


def test_spec_perfect_enh(post_reward):
    """§19.2 enh exact match, org far → large positive reward"""
    body = post_reward(
        note_id="spec_19_2",
        gt=["A01.1"],
        enh=["A01.1"],
        org=["B01"],
    )
    assert body["reward"] > 0.5


def test_spec_no_change(post_reward):
    """§19.3 enh == org, both same as gt: R_tree=0, R_exact=1 → reward > 0.
    In v2 this is a perfect-match reward (not neutral like v1)."""
    body = post_reward(
        note_id="spec_19_3",
        gt=["A01.1"],
        enh=["A01.1"],
        org=["A01.1"],
    )
    # R_exact=1 and R_structure=1 dominate; reward is positive
    assert body["reward"] > 0.0


def test_spec_deterioration(post_reward):
    """§19.4 enh far, org close → reward < 0"""
    body = post_reward(
        note_id="spec_19_4",
        gt=["A01.1"],
        enh=["B01"],
        org=["A01.2"],
    )
    assert body["reward"] < 0.0
    assert body["metrics"]["delta_D"] < 0.0


def test_spec_overprediction_penalised(post_reward):
    """§19.5 enh adds irrelevant codes → reward lower than perfect match"""
    body_over = post_reward(
        note_id="spec_19_5_over",
        gt=["A01.1"],
        enh=["A01.1", "A01.2", "A02.1"],
        org=["B01"],
    )
    body_perfect = post_reward(
        note_id="spec_19_5_perfect",
        gt=["A01.1"],
        enh=["A01.1"],
        org=["B01"],
    )
    assert body_perfect["reward"] >= body_over["reward"]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing failure (spec §14.1)
# ─────────────────────────────────────────────────────────────────────────────


def test_parsing_failure_reduces_reward_vs_success(client):
    """In v2, parsing_success=False forces R_structure=-1 but doesn't hard-code
    the total reward to -1.0.  The reward is lower than with parsing_success=True."""
    payload = {
        "note_id": "parse_cmp",
        "enh_codes": [],
        "org_codes": ["B01"],
        "gt_codes": ["A01.1"],
        "invalid_codes": [],
        "duplicate_codes": [],
    }
    resp_fail = client.post(
        "/compute_reward", json={**payload, "parsing_success": False}
    )
    resp_ok = client.post("/compute_reward", json={**payload, "parsing_success": True})
    assert resp_fail.status_code == 200
    assert resp_ok.status_code == 200
    # parse fail must yield a lower (or equal) reward than parse success
    assert resp_fail.json()["reward"] <= resp_ok.json()["reward"]


def test_parsing_failure_metrics_shape(client):
    resp = client.post(
        "/compute_reward",
        json={
            "note_id": "parse_fail_shape",
            "enh_codes": [],
            "org_codes": [],
            "gt_codes": [],
            "parsing_success": False,
            "invalid_codes": [],
            "duplicate_codes": [],
        },
    )
    body = resp.json()
    assert "metrics" in body
    assert "D_enh" in body["metrics"]
    assert "reward_components" in body["metrics"]
    assert "diagnostics" in body["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_codes_deduped(post_reward):
    body_dup = post_reward("dup_test", ["A01.1"], ["A01.2", "A01.2"], ["B01"])
    body_dedup = post_reward("no_dup", ["A01.1"], ["A01.2"], ["B01"])
    assert body_dup["reward"] == pytest.approx(body_dedup["reward"], abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Case normalisation
# ─────────────────────────────────────────────────────────────────────────────


def test_case_insensitive_input(post_reward):
    body_upper = post_reward("upper", ["A01.1"], ["A01.2"], ["B01"])
    body_lower = post_reward("lower", ["a01.1"], ["a01.2"], ["b01"])
    assert body_upper["reward"] == pytest.approx(body_lower["reward"], abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Empty code-list edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_enh_codes(client):
    resp = client.post(
        "/compute_reward",
        json={
            "note_id": "empty_enh",
            "enh_codes": [],
            "org_codes": ["A01.2"],
            "gt_codes": ["A01.1"],
            "invalid_codes": [],
            "duplicate_codes": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reward"] < 0.0


def test_empty_org_codes(client):
    resp = client.post(
        "/compute_reward",
        json={
            "note_id": "empty_org",
            "enh_codes": ["A01.2"],
            "org_codes": [],
            "gt_codes": ["A01.1"],
            "invalid_codes": [],
            "duplicate_codes": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reward"] > 0.0


def test_all_empty_enh_and_org(client):
    """Both enh and org empty → R_tree=0 (equal), but R_exact=-1, so reward < 0."""
    resp = client.post(
        "/compute_reward",
        json={
            "note_id": "all_empty",
            "enh_codes": [],
            "org_codes": [],
            "gt_codes": ["A01.1"],
            "invalid_codes": [],
            "duplicate_codes": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # R_tree=0, R_exact=-1 (no match), R_structure=1 → net negative
    assert body["reward"] < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Multi-code GT scenarios
# ─────────────────────────────────────────────────────────────────────────────


def test_multi_code_improvement(post_reward):
    body = post_reward(
        note_id="multi_improve",
        gt=["A01.1", "A02.1"],
        enh=["A01.2", "A02.1"],
        org=["B01", "B02"],
    )
    assert body["reward"] > 0.0


def test_multi_code_deterioration(post_reward):
    body = post_reward(
        note_id="multi_worsen",
        gt=["A01.1", "A02.1"],
        enh=["B01", "B02"],
        org=["A01.2", "A02.1"],
    )
    assert body["reward"] < 0.0
