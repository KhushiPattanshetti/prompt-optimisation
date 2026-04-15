"""
test_similarity.py – Unit tests for similarity.py

Covers: _infer_depth fallback, depth(), lca(), sim(), distance()
"""

import pytest
from reward_metrics_svc.similarity import depth, distance, lca, sim, _infer_depth
from reward_metrics_svc.config import VIRTUAL_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# _infer_depth  (string heuristic for out-of-tree codes)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code,expected",
    [
        ("X99", 1),  # no dot → category level
        ("X99.1", 2),  # 1 digit after dot
        ("X99.12", 3),  # 2 digits after dot
        ("Z00", 1),
        ("Z00.0", 2),
        ("Z00.00", 3),
    ],
)
def test_infer_depth_heuristic(code, expected):
    assert _infer_depth(code) == expected


# ─────────────────────────────────────────────────────────────────────────────
# depth()  – returns from depth_map or falls back to heuristic
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code,expected",
    [
        # ROOT=0, chapter=1, block=2, category=3, subcategory=4
        ("A00", 3),
        ("A01", 3),
        ("A01.1", 4),
        ("A01.2", 4),
        ("B00", 3),
        ("B01", 3),
    ],
)
def test_depth_known_codes(code, expected):
    assert depth(code) == expected


def test_depth_out_of_tree_no_dot():
    # Not in tree → heuristic → 1
    assert depth("Z99") == 1


def test_depth_out_of_tree_with_dot():
    # Not in tree → heuristic → 2
    assert depth("Z99.1") == 2


# ─────────────────────────────────────────────────────────────────────────────
# lca()
# ─────────────────────────────────────────────────────────────────────────────


def test_lca_siblings_share_parent():
    # A01.1 and A01.2 both have A01 as parent
    assert lca("A01.1", "A01.2") == "A01"


def test_lca_parent_child():
    # LCA of A01 and A01.1 is A01 itself (A01 is an ancestor of A01.1)
    assert lca("A01", "A01.1") == "A01"


def test_lca_cousin_codes():
    # A01.1 and A02.1 share grandparent A00-A09 (both categories under that block)
    assert lca("A01.1", "A02.1") == "A00-A09"


def test_lca_different_top_branches():
    # A01.1 (A branch) and B01 (B branch) only share VIRTUAL_ROOT
    assert lca("A01.1", "B01") == VIRTUAL_ROOT


def test_lca_same_code():
    # LCA of a code with itself is the code
    assert lca("A01.1", "A01.1") == "A01.1"


def test_lca_out_of_tree_codes():
    # Codes not in tree → no shared ancestor found → VIRTUAL_ROOT
    assert lca("Z99.1", "X01.2") == VIRTUAL_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# sim()  –  Wu-Palmer similarity ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────


def test_sim_identical():
    assert sim("A01.1", "A01.1") == 1.0


def test_sim_always_in_range():
    for a, b in [("A01.1", "A01.2"), ("A01.1", "B01"), ("A00", "B00")]:
        assert 0.0 <= sim(a, b) <= 1.0, f"sim({a},{b}) out of range"


def test_sim_siblings_value():
    # sim(A01.1, A01.2) = 2*depth(A01) / (depth(A01.1) + depth(A01.2))
    #                   = 2*3 / (4+4) = 6/8 = 0.75
    s = sim("A01.1", "A01.2")
    assert abs(s - 6 / 8) < 1e-9


def test_sim_different_branches_is_zero():
    # LCA(A01.1, B01) = VIRTUAL_ROOT (depth 0) → sim = 0
    assert sim("A01.1", "B01") == 0.0


def test_sim_siblings_greater_than_cross_branch():
    assert sim("A01.1", "A01.2") > sim("A01.1", "B01")


def test_sim_cousin_greater_than_cross_branch():
    assert sim("A01.1", "A02.1") > sim("A01.1", "B01")


def test_sim_sibling_greater_than_cousin():
    s_sibling = sim("A01.1", "A01.2")  # lca=A01 depth 3 → sim=0.75
    s_cousin = sim("A01.1", "A02.1")  # lca=A00-A09 depth 2 → sim=0.5
    assert s_sibling > s_cousin


# ─────────────────────────────────────────────────────────────────────────────
# distance()  =  1 - sim()
# ─────────────────────────────────────────────────────────────────────────────


def test_distance_identical():
    assert distance("A01.1", "A01.1") == 0.0


def test_distance_always_in_range():
    for a, b in [("A01.1", "A01.2"), ("A01.1", "B01")]:
        assert 0.0 <= distance(a, b) <= 1.0


@pytest.mark.parametrize(
    "a,b",
    [
        ("A01.1", "A01.2"),
        ("A01.1", "B01"),
        ("A02.1", "B02"),
    ],
)
def test_distance_plus_sim_equals_one(a, b):
    assert abs(distance(a, b) + sim(a, b) - 1.0) < 1e-9
