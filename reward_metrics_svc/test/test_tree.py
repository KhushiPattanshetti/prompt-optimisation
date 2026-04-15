"""
test_tree.py – Unit tests for tree.py

Verifies that the ICD-10 adjacency-list JSON is correctly parsed into
parent_map, depth_map, and that the BFS produces accurate depths.
"""

import pytest
from reward_metrics_svc import tree as _tree
from reward_metrics_svc.config import VIRTUAL_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# State after eager load
# ─────────────────────────────────────────────────────────────────────────────


def test_tree_loaded():
    assert _tree.state.loaded is True


def test_parent_map_not_empty():
    assert len(_tree.state.parent_map) > 0


def test_depth_map_not_empty():
    assert len(_tree.state.depth_map) > 0


def test_max_depth_positive():
    assert _tree.state.max_depth > 0


# ─────────────────────────────────────────────────────────────────────────────
# VIRTUAL_ROOT anchoring
# ─────────────────────────────────────────────────────────────────────────────


def test_virtual_root_depth_zero():
    assert _tree.state.depth_map[VIRTUAL_ROOT] == 0


def test_top_level_codes_have_virtual_root_as_parent():
    # Chapter letters (A, B, …) are the direct children of VIRTUAL_ROOT
    assert _tree.state.parent_map.get("A") == VIRTUAL_ROOT
    assert _tree.state.parent_map.get("B") == VIRTUAL_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Known parent relationships  (using test icd10_tree.json)
# Tree: A00 → A01 → A01.1 / A01.2
#           → A02 → A02.1
#       B00 → B01 / B02
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "child,expected_parent",
    [
        # 3-char categories live under their block
        ("A00", "A00-A09"),
        ("A01", "A00-A09"),
        ("A02", "A00-A09"),
        # subcategories live under their category
        ("A00.0", "A00"),
        ("A01.1", "A01"),
        ("A01.2", "A01"),
        ("A02.1", "A02"),
    ],
)
def test_parent_map_known_pairs(child, expected_parent):
    assert _tree.state.parent_map.get(child) == expected_parent


# ─────────────────────────────────────────────────────────────────────────────
# Known depths
# Depths (ROOT=0): A00=1, A01=2, A01.1=3, A01.2=3, A02=2, A02.1=3
#                  B00=1, B01=2, B02=2
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code,expected_depth",
    [
        # ROOT=0, chapter=1, block=2, category=3, subcategory=4
        ("A00", 3),
        ("A01", 3),
        ("A01.1", 4),
        ("A01.2", 4),
        ("A02", 3),
        ("A02.1", 4),
        ("B00", 3),
        ("B01", 3),
        ("B02", 3),
    ],
)
def test_depth_map_known_codes(code, expected_depth):
    assert _tree.state.depth_map[code] == expected_depth


def test_max_depth_value():
    # Full ICD-10: ROOT(0)→chapter(1)→block(2)→category(3)→subcategory(4)
    assert _tree.state.max_depth == 4


# ─────────────────────────────────────────────────────────────────────────────
# Reload idempotency
# ─────────────────────────────────────────────────────────────────────────────


def test_reload_preserves_correct_state():
    """Calling load() again must not corrupt tree state."""
    _tree.load()
    assert _tree.state.loaded is True
    assert _tree.state.depth_map.get("A01.1") == 4
