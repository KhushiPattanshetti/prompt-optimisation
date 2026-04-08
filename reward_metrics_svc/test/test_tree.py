"""
test_tree.py – Unit tests for tree.py

Verifies that the ICD-10 adjacency-list JSON is correctly parsed into
parent_map, depth_map, and that the BFS produces accurate depths.
"""

import pytest
import tree as _tree
from config import VIRTUAL_ROOT


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
    # A00 and B00 are top-level codes in the test tree
    assert _tree.state.parent_map.get("A00") == VIRTUAL_ROOT
    assert _tree.state.parent_map.get("B00") == VIRTUAL_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Known parent relationships  (using test icd10_tree.json)
# Tree: A00 → A01 → A01.1 / A01.2
#           → A02 → A02.1
#       B00 → B01 / B02
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "child,expected_parent",
    [
        ("A01", "A00"),
        ("A02", "A00"),
        ("A01.1", "A01"),
        ("A01.2", "A01"),
        ("A02.1", "A02"),
        ("B01", "B00"),
        ("B02", "B00"),
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
        ("A00", 1),
        ("A01", 2),
        ("A01.1", 3),
        ("A01.2", 3),
        ("A02", 2),
        ("A02.1", 3),
        ("B00", 1),
        ("B01", 2),
        ("B02", 2),
    ],
)
def test_depth_map_known_codes(code, expected_depth):
    assert _tree.state.depth_map[code] == expected_depth


def test_max_depth_value():
    # Deepest nodes in the test tree are at depth 3 (A01.1, A01.2, A02.1)
    assert _tree.state.max_depth == 3


# ─────────────────────────────────────────────────────────────────────────────
# Reload idempotency
# ─────────────────────────────────────────────────────────────────────────────


def test_reload_preserves_correct_state():
    """Calling load() again must not corrupt tree state."""
    _tree.load()
    assert _tree.state.loaded is True
    assert _tree.state.depth_map.get("A01.1") == 3
