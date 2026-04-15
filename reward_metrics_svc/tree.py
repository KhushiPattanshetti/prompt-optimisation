"""
tree.py – Loads the ICD-10 adjacency-list JSON and builds the two data
structures required by Wu-Palmer similarity:

  parent_map  : Dict[code → parent_code]
  depth_map   : Dict[code → int]   (edges from VIRTUAL_ROOT)
  max_depth   : int

A synthetic VIRTUAL_ROOT node is inserted so that all top-level category
codes share a common ancestor.  The tree is loaded eagerly at import time
so every other module (and test) always operates on a live tree.
"""

import json
import logging
from collections import deque
from typing import Any, Dict, List, Tuple

from .config import ICD10_TREE_PATH, VIRTUAL_ROOT

logger = logging.getLogger("reward_metrics_svc.tree")


# ─────────────────────────────────────────────────────────────────────────────
# Tree state container  (mutable singleton – modified in place on reload)
# ─────────────────────────────────────────────────────────────────────────────


class _TreeState:
    def __init__(self) -> None:
        self.parent_map: Dict[str, str] = {}
        self.depth_map: Dict[str, int] = {}
        self.max_depth: int = 1
        self.loaded: bool = False


# Module-level singleton – imported by similarity.py and metrics.py
state = _TreeState()


# ─────────────────────────────────────────────────────────────────────────────
# Internal build logic
# ─────────────────────────────────────────────────────────────────────────────


def _build(
    tree_dict: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Dict[str, int], int]:
    """
    Convert the adjacency-list dict (parent → [children]) into
    parent_map + depth_map, inserting VIRTUAL_ROOT at the apex.
    """
    # Normalise to uppercase
    norm: Dict[str, List[str]] = {
        k.upper(): [c.upper() for c in v] for k, v in tree_dict.items()
    }

    # ── parent_map ────────────────────────────────────────────────────────────
    parent_map: Dict[str, str] = {}
    all_children: set = set()
    for parent, children in norm.items():
        for child in children:
            parent_map[child] = parent
            all_children.add(child)

    # Top-level nodes (never appear as a child) become children of VIRTUAL_ROOT
    roots = [code for code in norm if code not in all_children]
    for root in roots:
        parent_map[root] = VIRTUAL_ROOT
    logger.info("ICD-10 tree roots linked to %s: %s", VIRTUAL_ROOT, roots)

    # ── BFS depth computation from VIRTUAL_ROOT ───────────────────────────────
    child_map: Dict[str, List[str]] = {VIRTUAL_ROOT: roots, **norm}
    depth_map: Dict[str, int] = {VIRTUAL_ROOT: 0}
    queue: deque = deque([VIRTUAL_ROOT])
    while queue:
        node = queue.popleft()
        for child in child_map.get(node, []):
            if child not in depth_map:
                depth_map[child] = depth_map[node] + 1
                queue.append(child)

    max_depth = max(v for k, v in depth_map.items() if k != VIRTUAL_ROOT)
    logger.info(
        "Tree built | nodes=%d  max_depth=%d  sample=%s",
        len(depth_map) - 1,
        max_depth,
        dict(list(depth_map.items())[1:6]),
    )
    return parent_map, depth_map, max_depth


# ─────────────────────────────────────────────────────────────────────────────
# Public load function  (called at import time + on ASGI startup)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_indexed_format(
    index: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, int], int]:
    """
    Parse the new ICD-10 indexed format where each entry has 'parent' and
    'depth' fields.  Codes are normalised to uppercase; the sentinel value
    'ROOT' is mapped to VIRTUAL_ROOT.
    """
    parent_map: Dict[str, str] = {}
    depth_map: Dict[str, int] = {VIRTUAL_ROOT: 0}

    for code_raw, entry in index.items():
        code = code_raw.upper()
        raw_parent = entry.get("parent")
        depth_val = int(entry.get("depth", 1))
        depth_map[code] = depth_val
        if raw_parent:
            parent = raw_parent.upper()
            parent_map[code] = VIRTUAL_ROOT if parent == "ROOT" else parent

    max_d = max((v for k, v in depth_map.items() if k != VIRTUAL_ROOT), default=1)
    roots = [c for c, p in parent_map.items() if p == VIRTUAL_ROOT]
    logger.info("ICD-10 tree roots linked to %s: %s", VIRTUAL_ROOT, roots[:10])
    logger.info(
        "Tree built | nodes=%d  max_depth=%d",
        len(depth_map) - 1,
        max_d,
    )
    return parent_map, depth_map, max_d


def load() -> None:
    """Parse ICD10_TREE_PATH and populate the module-level `state`."""
    with open(ICD10_TREE_PATH, "r") as fh:
        data = json.load(fh)
    if "index" in data and isinstance(data["index"], dict):
        # New indexed format: {"version": ..., "index": {"A00.0": {parent, depth, ...}}}
        state.parent_map, state.depth_map, state.max_depth = _parse_indexed_format(
            data["index"]
        )
    else:
        # Legacy adjacency-list format: {"A00": ["A00.0", "A00.1"], ...}
        state.parent_map, state.depth_map, state.max_depth = _build(data)
    state.loaded = True


# ─────────────────────────────────────────────────────────────────────────────
# Eager initialisation – every importer gets a ready-to-use tree
# ─────────────────────────────────────────────────────────────────────────────

try:
    load()
except Exception as _err:
    logger.error("Eager tree load failed: %s", _err)
