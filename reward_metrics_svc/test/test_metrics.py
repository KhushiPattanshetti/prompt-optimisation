"""
test_metrics.py – Unit tests for metrics.py

Covers: set_distance, coverage_penalty, extra_penalty,
        cardinality_penalty, distance_between, distance_with_components
"""

import pytest
from metrics import (
    cardinality_penalty,
    coverage_penalty,
    distance_between,
    distance_with_components,
    extra_penalty,
    set_distance,
)
from config import LAMBDA_CARD, LAMBDA_EXTRA


# ─────────────────────────────────────────────────────────────────────────────
# set_distance
# ─────────────────────────────────────────────────────────────────────────────


def test_set_distance_both_empty():
    assert set_distance([], []) == 0.0


def test_set_distance_pred_empty():
    assert set_distance(["A01.1"], []) == 1.0


def test_set_distance_gt_empty():
    assert set_distance([], ["A01.1"]) == 0.0


def test_set_distance_perfect_match():
    assert set_distance(["A01.1"], ["A01.1"]) == 0.0


def test_set_distance_in_range():
    d = set_distance(["A01.1"], ["B01"])
    assert 0.0 <= d <= 1.0


def test_set_distance_close_pair_less_than_far():
    d_near = set_distance(["A01.1"], ["A01.2"])
    d_far = set_distance(["A01.1"], ["B01"])
    assert d_near < d_far


# ─────────────────────────────────────────────────────────────────────────────
# coverage_penalty
# ─────────────────────────────────────────────────────────────────────────────


def test_coverage_penalty_empty_gt():
    assert coverage_penalty([], ["A01.1"]) == 0.0


def test_coverage_penalty_perfect_pred():
    # Prediction matches GT exactly → no coverage loss
    assert coverage_penalty(["A01.1"], ["A01.1"]) == pytest.approx(0.0, abs=1e-9)


def test_coverage_penalty_no_pred():
    # Nothing predicted → maximum coverage loss
    p = coverage_penalty(["A01.1"], [])
    assert p > 0.0


def test_coverage_penalty_sibling_lower_than_unrelated():
    p_sibling = coverage_penalty(["A01.1"], ["A01.2"])
    p_unrelated = coverage_penalty(["A01.1"], ["B01"])
    assert p_sibling < p_unrelated


# ─────────────────────────────────────────────────────────────────────────────
# extra_penalty
# ─────────────────────────────────────────────────────────────────────────────


def test_extra_penalty_empty_pred():
    assert extra_penalty(["A01.1"], []) == 0.0


def test_extra_penalty_perfect_pred():
    # Prediction exactly matches GT → no extra penalty
    assert extra_penalty(["A01.1"], ["A01.1"]) == pytest.approx(0.0, abs=1e-9)


def test_extra_penalty_unrelated_code():
    p = extra_penalty(["A01.1"], ["B01"])
    assert p > 0.0


def test_extra_penalty_bounded_by_lambda():
    # Maximum raw penalty is 1.0, so final ≤ LAMBDA_EXTRA
    p = extra_penalty(["A01.1"], ["B01"])
    assert p <= LAMBDA_EXTRA + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# cardinality_penalty
# ─────────────────────────────────────────────────────────────────────────────


def test_cardinality_penalty_empty_gt():
    assert cardinality_penalty([], ["A01.1", "A01.2"]) == 0.0


def test_cardinality_penalty_same_size():
    assert cardinality_penalty(["A01.1"], ["A01.2"]) == 0.0


def test_cardinality_penalty_double_prediction():
    # |pred|=2, |gt|=1 → raw=1.0 → LAMBDA_CARD * 1.0
    p = cardinality_penalty(["A01.1"], ["A01.1", "A01.2"])
    assert p == pytest.approx(LAMBDA_CARD, abs=1e-9)


def test_cardinality_penalty_triple_prediction():
    # |pred|=3, |gt|=1 → raw=2 → capped at 1.0 → LAMBDA_CARD
    p = cardinality_penalty(["A01.1"], ["A01.1", "A01.2", "A02.1"])
    assert p == pytest.approx(LAMBDA_CARD, abs=1e-9)


def test_cardinality_penalty_underprediction():
    # |pred|=1, |gt|=2 → raw=0.5
    p = cardinality_penalty(["A01.1", "A01.2"], ["A01.1"])
    assert p == pytest.approx(LAMBDA_CARD * 0.5, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# distance_between / distance_with_components
# ─────────────────────────────────────────────────────────────────────────────


def test_distance_between_perfect():
    assert distance_between(["A01.1"], ["A01.1"]) == pytest.approx(0.0, abs=1e-9)


def test_distance_between_in_range():
    d = distance_between(["A01.1"], ["B01"])
    assert 0.0 <= d <= 1.0


def test_distance_between_near_less_than_far():
    d_near = distance_between(["A01.1"], ["A01.2"])
    d_far = distance_between(["A01.1"], ["B01"])
    assert d_near < d_far


def test_distance_with_components_returns_dict_keys():
    _, comps = distance_with_components(["A01.1"], ["A01.2"])
    assert set(comps.keys()) == {"D_set", "P_cov", "P_extra", "P_card"}


def test_distance_with_components_all_nonnegative():
    _, comps = distance_with_components(["A01.1"], ["B01"])
    for k, v in comps.items():
        assert v >= 0.0, f"{k} is negative: {v}"


def test_distance_behaves_consistently_with_components():
    d_scalar, comps = distance_with_components(["A01.1"], ["A01.2"])
    assert d_scalar >= 0.0
    # The weighted sum of components is internally normalised;
    # the returned scalar should match what distance_between returns.
    assert d_scalar == pytest.approx(distance_between(["A01.1"], ["A01.2"]), abs=1e-9)


def test_distance_between_multi_code_perfect():
    gt = ["A01.1", "A02.1"]
    pred = ["A01.1", "A02.1"]
    assert distance_between(gt, pred) == pytest.approx(0.0, abs=1e-9)
