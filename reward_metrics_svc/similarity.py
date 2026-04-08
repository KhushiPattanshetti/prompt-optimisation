"""
similarity.py – Wu-Palmer code-level similarity and distance.

Public API
----------
depth(code)         → int    : edges from VIRTUAL_ROOT
lca(code_a, code_b) → str    : lowest common ancestor
sim(code_a, code_b) → float  : Wu-Palmer similarity ∈ [0, 1]
distance(code_a, code_b) → float : 1 - sim ∈ [0, 1]
"""

import logging
from typing import Optional

import tree as _tree
from config import VIRTUAL_ROOT

logger = logging.getLogger("reward_metrics_svc.similarity")


# ─────────────────────────────────────────────────────────────────────────────
# Depth
# ─────────────────────────────────────────────────────────────────────────────


def _infer_depth(code: str) -> int:
    """
    String-heuristic depth for codes absent from depth_map.
    ICD-10 convention:
      "A00"    → 1   (3-char category, no dot)
      "A00.0"  → 2   (one digit after the dot)
      "A00.00" → 3   (two digits after the dot)
    """
    if "." not in code:
        return 1
    dot_pos = code.index(".")
    return 1 + (len(code) - dot_pos - 1)


def depth(code: str) -> int:
    """
    Return the depth of *code* as edges from VIRTUAL_ROOT.
    Falls back to the string heuristic for out-of-tree codes.
    """
    if code in _tree.state.depth_map:
        return _tree.state.depth_map[code]
    inferred = _infer_depth(code)
    logger.debug("depth fallback | code='%s'  inferred=%d", code, inferred)
    return inferred


# ─────────────────────────────────────────────────────────────────────────────
# LCA  (spec §6.2)
# ─────────────────────────────────────────────────────────────────────────────


def _infer_string_lca(code_a: str, code_b: str) -> str:
    """
    String-prefix heuristic LCA for out-of-tree codes.
    ICD-10 convention:
      J45   is parent of J45.0  (J45.0.startswith("J45."))
      J45.0 is parent of J45.00 (J45.00.startswith("J45.0."))  [extended]
      J45.0 and J45.1 share parent J45 (common base before the dot)
    Returns the inferred ancestor, or VIRTUAL_ROOT if none found.
    """
    if code_a == code_b:
        return code_a
    # Direct parent: one code is a dot-prefix of the other
    if code_a.startswith(code_b + "."):
        return code_b
    if code_b.startswith(code_a + "."):
        return code_a
    # Sibling: both have a dot and the base (before the dot) is identical
    if "." in code_a and "." in code_b:
        base_a = code_a[: code_a.index(".")]
        base_b = code_b[: code_b.index(".")]
        if base_a == base_b:
            return base_a
    return VIRTUAL_ROOT


def lca(code_a: str, code_b: str) -> str:
    """
    Lowest Common Ancestor via parent_map traversal.
    Falls back to string-prefix heuristic for out-of-tree codes.
    """
    # Collect all ancestors of code_a (inclusive)
    ancestors_a: set = set()
    curr: Optional[str] = code_a
    while curr is not None:
        ancestors_a.add(curr)
        curr = _tree.state.parent_map.get(curr)

    # Walk up from code_b until we find a shared ancestor
    curr = code_b
    while curr is not None and curr not in ancestors_a:
        curr = _tree.state.parent_map.get(curr)

    if curr is not None:
        result = curr
    else:
        # String-prefix fallback for codes absent from the tree
        result = _infer_string_lca(code_a, code_b)
        if result != VIRTUAL_ROOT:
            logger.debug("LCA prefix fallback | %s ∩ %s → %s", code_a, code_b, result)

    logger.debug("LCA(%s, %s) = %s", code_a, code_b, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Wu-Palmer similarity & distance  (spec §6.3)
# ─────────────────────────────────────────────────────────────────────────────


def sim(code_a: str, code_b: str) -> float:
    """
    Wu-Palmer similarity ∈ [0, 1].

    sim(a, b) = 2 * depth(LCA(a, b)) / (depth(a) + depth(b))
    """
    if code_a == code_b:
        return 1.0
    lca_node = lca(code_a, code_b)
    d_lca = depth(lca_node)
    d_a = depth(code_a)
    d_b = depth(code_b)
    denom = d_a + d_b
    if denom == 0:
        return 1.0
    score = (2.0 * d_lca) / denom
    logger.debug(
        "sim(%s,%s) | lca=%s  d_lca=%d  d_a=%d  d_b=%d  →  %.4f",
        code_a,
        code_b,
        lca_node,
        d_lca,
        d_a,
        d_b,
        score,
    )
    return score


def distance(code_a: str, code_b: str) -> float:
    """Wu-Palmer distance ∈ [0, 1].  distance = 1 - sim."""
    return 1.0 - sim(code_a, code_b)
