"""
test_reward.py – Unit tests for reward.py  (spec §12–13, v2 hybrid formula)

Covers:
  - _r_exact       : Jaccard-based exact reward
  - _r_structure   : structural validity reward
  - compute_reward : return shape, bounds, sign, diagnostics, new components
"""

import math
import pytest
from reward_metrics_svc.reward import compute_reward, _r_exact, _r_structure


# ─────────────────────────────────────────────────────────────────────────────
# _r_exact  (spec §12.1)
# ─────────────────────────────────────────────────────────────────────────────


def test_r_exact_perfect_match():
    assert _r_exact(["A01.1"], ["A01.1"]) == pytest.approx(1.0)


def test_r_exact_both_empty():
    assert _r_exact([], []) == pytest.approx(1.0)


def test_r_exact_disjoint():
    # J = 0 → 2*0 - 1 = -1
    assert _r_exact(["A01.1"], ["B01"]) == pytest.approx(-1.0)


def test_r_exact_partial_overlap():
    # |{A,B} ∩ {A,C}| / |{A,B,C}| = 1/3 → 2*(1/3)-1 = -1/3
    result = _r_exact(["A", "B"], ["A", "C"])
    assert result == pytest.approx(-1 / 3, abs=1e-6)


def test_r_exact_gt_subset_of_enh():
    # gt={A}, enh={A,B}  J=1/2 → 0.0
    assert _r_exact(["A"], ["A", "B"]) == pytest.approx(0.0)


def test_r_exact_returns_float():
    assert isinstance(_r_exact(["A01.1"], ["A01.1"]), float)


# ─────────────────────────────────────────────────────────────────────────────
# _r_structure  (spec §12.2)
# ─────────────────────────────────────────────────────────────────────────────


def test_r_structure_clean():
    assert _r_structure(["A01.1"], [], [], True) == pytest.approx(1.0)


def test_r_structure_parse_fail_is_minus_one():
    assert _r_structure(["A01.1"], [], [], False) == pytest.approx(-1.0)


def test_r_structure_parse_fail_regardless_of_codes():
    assert _r_structure(["A01.1"], ["X"], ["Y"], False) == pytest.approx(-1.0)


def test_r_structure_one_invalid():
    # enh=[A,B] 2 codes, 1 invalid → p_invalid=0.5 → r=0.5
    result = _r_structure(["A", "B"], ["A"], [], True)
    assert result == pytest.approx(0.5)


def test_r_structure_one_duplicate():
    result = _r_structure(["A", "B"], [], ["A"], True)
    assert result == pytest.approx(0.5)


def test_r_structure_clamped_to_minus_one():
    # many invalid+dupes → penalty > 1 → clamped to -1
    enh = ["A"]
    result = _r_structure(enh, ["A"], ["A"], True)
    assert result == pytest.approx(-1.0)


def test_r_structure_returns_float():
    assert isinstance(_r_structure(["A01.1"], [], [], True), float)


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward – return type and structure
# ─────────────────────────────────────────────────────────────────────────────


def test_returns_two_tuple():
    result = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    assert isinstance(result, tuple) and len(result) == 2


def test_reward_is_float():
    reward, _ = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    assert isinstance(reward, float)


def test_metrics_dict_has_required_keys():
    _, metrics = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    required = {
        "D_enh",
        "D_org",
        "delta_D",
        "reward_components",
        "components_enh",
        "components_org",
        "diagnostics",
    }
    assert required <= set(metrics)


def test_reward_components_has_required_keys():
    _, metrics = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    assert {"R_tree", "R_exact", "R_structure"} <= set(metrics["reward_components"])


def test_diagnostics_has_required_keys():
    _, metrics = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    required = {
        "worst_gt_coverage_code",
        "worst_pred_match_code",
        "invalid_codes",
        "duplicate_codes",
    }
    assert required <= set(metrics["diagnostics"])


def test_components_enh_has_required_keys():
    _, metrics = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    assert {"D_set", "P_cov", "P_extra", "P_card"} <= set(metrics["components_enh"])


def test_components_org_has_required_keys():
    _, metrics = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    assert {"D_set", "P_cov", "P_extra", "P_card"} <= set(metrics["components_org"])


def test_invalid_codes_passed_through_to_diagnostics():
    _, metrics = compute_reward(
        ["A01.1"], ["A01.1"], ["B01"], invalid_codes=["X99"], duplicate_codes=[]
    )
    assert "X99" in metrics["diagnostics"]["invalid_codes"]


def test_duplicate_codes_passed_through_to_diagnostics():
    _, metrics = compute_reward(
        ["A01.1"], ["A01.1"], ["B01"], invalid_codes=[], duplicate_codes=["A01.1"]
    )
    assert "A01.1" in metrics["diagnostics"]["duplicate_codes"]


# ─────────────────────────────────────────────────────────────────────────────
# Reward bounds
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "gt,enh,org",
    [
        (["A01.1"], ["A01.2"], ["B01"]),
        (["A01.1"], ["B01"], ["A01.2"]),
        (["A01.1"], ["A01.1"], ["A01.1"]),
        (["A01.1", "A02.1"], ["A01.1", "A02.1"], ["B01", "B02"]),
    ],
)
def test_reward_always_in_minus_one_to_one(gt, enh, org):
    reward, _ = compute_reward(gt, enh, org)
    assert -1.0 <= reward <= 1.0, f"reward={reward} out of bounds"


def test_reward_with_parsing_failure_in_bounds():
    reward, _ = compute_reward(["A01.1"], [], ["B01"], parsing_success=False)
    assert -1.0 <= reward <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Sign semantics
# ─────────────────────────────────────────────────────────────────────────────


def test_improvement_gives_positive_reward():
    reward, _ = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    assert reward > 0.0


def test_deterioration_gives_negative_reward():
    reward, _ = compute_reward(["A01.1"], ["B01"], ["A01.2"])
    assert reward < 0.0


def test_perfect_enh_vs_far_org_large_positive():
    reward, _ = compute_reward(["A01.1"], ["A01.1"], ["B01"])
    assert reward > 0.5


def test_no_change_near_zero():
    # In v2: enh==org==gt → R_tree=0, R_exact=1, R_structure=1
    # raw = 0.6*0 + 0.25*1 + 0.15*1 = 0.4 → tanh(0.4) ≈ 0.38
    # This is positive (perfect exact match), not neutral.
    reward, _ = compute_reward(["A01.1"], ["A01.1"], ["A01.1"])
    # When enh equals gt (perfect match), reward should be > 0.
    assert reward > 0.0


def test_antisymmetry():
    r_forward, _ = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    r_reverse, _ = compute_reward(["A01.1"], ["B01"], ["A01.2"])
    assert r_forward > 0
    assert r_reverse < 0


def test_parsing_failure_reduces_reward():
    reward_ok, _ = compute_reward(["A01.1"], ["A01.1"], ["B01"], parsing_success=True)
    reward_fail, _ = compute_reward(
        ["A01.1"], ["A01.1"], ["B01"], parsing_success=False
    )
    assert reward_ok > reward_fail


def test_invalid_codes_reduce_r_structure():
    _, m_clean = compute_reward(["A01.1"], ["A01.1"], ["B01"], invalid_codes=[])
    _, m_dirty = compute_reward(["A01.1"], ["A01.1"], ["B01"], invalid_codes=["X"])
    assert (
        m_clean["reward_components"]["R_structure"]
        > m_dirty["reward_components"]["R_structure"]
    )


def test_r_structure_is_minus_one_on_parse_fail():
    _, metrics = compute_reward(["A01.1"], [], ["B01"], parsing_success=False)
    assert metrics["reward_components"]["R_structure"] == pytest.approx(-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics consistency
# ─────────────────────────────────────────────────────────────────────────────


def test_delta_d_equals_d_org_minus_d_enh():
    _, metrics = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    expected = metrics["D_org"] - metrics["D_enh"]
    assert abs(metrics["delta_D"] - expected) < 1e-6


def test_r_tree_equals_delta_d():
    _, metrics = compute_reward(["A01.1"], ["A01.2"], ["B01"])
    assert metrics["reward_components"]["R_tree"] == pytest.approx(metrics["delta_D"])


def test_r_exact_for_exact_match():
    _, metrics = compute_reward(["A01.1"], ["A01.1"], ["B01"])
    assert metrics["reward_components"]["R_exact"] == pytest.approx(1.0)


def test_multi_code_improvement():
    gt = ["A01.1", "A02.1"]
    enh = ["A01.2", "A02.1"]
    org = ["B01", "B02"]
    reward, _ = compute_reward(gt, enh, org)
    assert reward > 0.0


def test_determinism():
    args = (["A01.1"], ["A01.2"], ["B01"])
    rewards = [compute_reward(*args)[0] for _ in range(3)]
    assert len(set(rewards)) == 1
