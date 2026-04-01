"""Reward Metrics Microservice.

Computes a scalar reward in [-1.0, 1.0] from ICD-10 code sets using
the ICD-10 tree hierarchy, then forwards the full rollout payload to
the RL loop service.

FIX SUMMARY (from review):
  - Added note_id field to RewardRequest so rollout can be traced.
  - post_to_rl_block now forwards the FULL rollout payload
    (original_prompt, rewritten_prompt, reward, log_prob_old,
     value_estimate) not just {"reward": value}.
    The rl_loop_svc rollout schema requires all five fields.
  - Service must be started on port 8002 explicitly (was clashing
    with rewriter_inference_svc on default 8000).
  - icd10_tree.json is a toy mock with 9 codes. The tree is loaded
    from file so you can replace icd10_tree.json with a full
    ICD-10-CM hierarchy without changing code.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import networkx as nx
import requests
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reward_metrics_svc")

# ── Configuration ──────────────────────────────────────────────────────────────

TREE_PATH          = Path(__file__).parent / "icd10_tree.json"
GT_CODES_DIR       = Path(__file__).parent / "gt_codes"
RL_BLOCK_ENDPOINT  = os.environ.get("RL_BLOCK_ENDPOINT", "http://localhost:8004/rollout")

# ── ICD-10 Tree Loading ────────────────────────────────────────────────────────

def load_icd10_tree() -> nx.DiGraph:
    """Load ICD-10 hierarchy from JSON into a directed graph.

    The JSON format is an adjacency dict:
        { "parent_code": ["child_code1", "child_code2"], ... }
    A virtual ROOT node is added connecting all top-level nodes.
    """
    with open(TREE_PATH, "r", encoding="utf-8") as f:
        tree_dict = json.load(f)

    graph = nx.DiGraph()
    for parent, children in tree_dict.items():
        for child in children:
            graph.add_edge(parent, child)

    # Add virtual root connecting all nodes with no incoming edges
    graph.add_node("ROOT")
    for node in list(graph.nodes):
        if node != "ROOT" and graph.in_degree(node) == 0:
            graph.add_edge("ROOT", node)

    log.info("ICD-10 tree loaded | nodes=%d | edges=%d",
             graph.number_of_nodes(), graph.number_of_edges())
    return graph


ICD10_GRAPH = load_icd10_tree()


def get_max_path_length(graph: nx.DiGraph) -> int:
    """Return the longest shortest path from ROOT to any leaf."""
    max_len = 0
    for node in graph.nodes:
        if node == "ROOT":
            continue
        try:
            length = nx.shortest_path_length(graph, "ROOT", node)
            max_len = max(max_len, length)
        except nx.NetworkXNoPath:
            pass
    return max(max_len, 1)


MAX_PATH_LENGTH = get_max_path_length(ICD10_GRAPH)


# ── Pydantic Schemas ───────────────────────────────────────────────────────────

class RewardRequest(BaseModel):
    # FIX: added note_id and full rollout fields so they can be
    # forwarded to rl_loop_svc with the complete payload
    note_id:          Optional[str]        = None
    gt_codes:         Optional[List[str]]  = None
    gt_file:          Optional[str]        = None
    enh_codes:        List[str]
    org_codes:        List[str]
    # Rollout fields — required to forward complete trajectory to RL loop
    original_prompt:  Optional[str]        = None
    rewritten_prompt: Optional[str]        = None
    log_prob_old:     Optional[float]      = None
    value_estimate:   Optional[float]      = None


class RewardResponse(BaseModel):
    reward: float


# ── Distance Metric ────────────────────────────────────────────────────────────

def normalize_codes(codes: List[str]) -> List[str]:
    """Upper-case and strip whitespace from all codes."""
    return [c.strip().upper() for c in codes]


def _lca_depth(graph: nx.DiGraph, code_a: str, code_b: str) -> int:
    """Return the depth of the Lowest Common Ancestor of two codes."""
    try:
        path_a = nx.shortest_path(graph, "ROOT", code_a)
        path_b = nx.shortest_path(graph, "ROOT", code_b)
    except nx.NetworkXNoPath:
        return 0
    set_a = set(path_a)
    common = [n for n in path_b if n in set_a]
    return len(common)


def _single_code_distance(
    graph: nx.DiGraph, code: str, code_set: List[str], max_depth: int
) -> float:
    """Normalised tree distance from one code to the nearest code in a set."""
    if not code_set:
        return 1.0
    if code not in graph:
        return 1.0

    best = 0
    for target in code_set:
        if target not in graph:
            continue
        depth = _lca_depth(graph, code, target)
        best  = max(best, depth)

    return 1.0 - (best / max_depth)


def distance_between(
    codes_a: List[str],
    codes_b: List[str],
    graph: nx.DiGraph = ICD10_GRAPH,
    max_depth: int    = MAX_PATH_LENGTH,
) -> float:
    """Compute the mean normalised tree distance between two ICD-10 code sets.

    Uses LCA (Lowest Common Ancestor) depth as the similarity measure.
    Higher LCA depth = closer in hierarchy = lower distance.

    Returns:
        Float in [0.0, 1.0]. 0.0 = identical sets. 1.0 = completely unrelated.
    """
    if not codes_a or not codes_b:
        return 1.0

    codes_a = normalize_codes(codes_a)
    codes_b = normalize_codes(codes_b)

    distances = [
        _single_code_distance(graph, c, codes_b, max_depth)
        for c in codes_a
    ]
    return float(sum(distances) / len(distances))


# ── Reward Calculation ─────────────────────────────────────────────────────────

def calculate_reward(
    gt_codes:  List[str],
    enh_codes: List[str],
    org_codes: List[str],
) -> float:
    """Compute scalar reward in [-1.0, 1.0].

    Rules:
        d_gt_enh < d_gt_org  →  reward ∈ (0, 1.0]   (enhanced is better)
        d_gt_enh > d_gt_org  →  reward ∈ [-1.0, 0)  (original is better)
        d_gt_enh == d_gt_org →
            if d_enh_org == 0: reward = 1.0           (identical, good)
            else:              reward ∈ [-1.0, 0)     (tied but different)
    """
    d_gt_enh  = distance_between(gt_codes, enh_codes)
    d_gt_org  = distance_between(gt_codes, org_codes)
    d_enh_org = distance_between(enh_codes, org_codes)

    if d_gt_enh < d_gt_org:
        # Enhanced is closer to GT — positive reward proportional to improvement
        improvement = d_gt_org - d_gt_enh  # in (0, 1]
        reward      = min(improvement * 2.0, 1.0)
        return round(max(reward, 1e-4), 4)

    elif d_gt_enh > d_gt_org:
        # Enhanced is farther from GT — negative reward
        degradation = d_gt_enh - d_gt_org
        reward      = -min(degradation * 2.0, 1.0)
        return round(min(reward, -1e-4), 4)

    else:
        # Tie
        if d_enh_org == 0:
            return 1.0
        else:
            return round(-d_gt_enh, 4)


# ── Ground Truth Loading (file-based fallback) ─────────────────────────────────

def load_gt_codes(gt_file: str) -> List[str]:
    """Load gt_codes from a JSON file in the gt_codes/ directory."""
    filepath = GT_CODES_DIR / gt_file
    if not filepath.exists():
        raise FileNotFoundError(f"GT codes file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("gt_codes", [])


# ── RL Block Forwarding ────────────────────────────────────────────────────────

def post_to_rl_block(reward: float, request: RewardRequest) -> None:
    """Forward the complete rollout payload to rl_loop_svc POST /rollout.

    FIX: Previously only sent {"reward": value}.
    rl_loop_svc RolloutEntry schema requires all five fields:
        original_prompt, rewritten_prompt, reward,
        log_prob_old, value_estimate.
    Missing fields are skipped with a warning.
    """
    if not RL_BLOCK_ENDPOINT:
        return

    payload = {"reward": reward}

    for field in ("original_prompt", "rewritten_prompt",
                  "log_prob_old", "value_estimate"):
        value = getattr(request, field, None)
        if value is not None:
            payload[field] = value
        else:
            log.warning(
                "post_to_rl_block: field '%s' is None — "
                "rl_loop_svc may reject this rollout",
                field,
            )

    try:
        resp = requests.post(RL_BLOCK_ENDPOINT, json=payload, timeout=10)
        log.info("Posted to RL block | status=%d", resp.status_code)
    except requests.RequestException as exc:
        log.error("Failed to post to RL block | error=%s", exc)


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="reward_metrics_svc")


@app.post("/reward", response_model=RewardResponse)
def reward_endpoint(req: RewardRequest) -> RewardResponse:
    """Compute reward and forward to RL loop.

    Accepts gt_codes directly in the request body (sent by icd10_coding_svc)
    or loads them from a gt_file path as a fallback.
    """
    if req.gt_codes is not None:
        gt_codes = req.gt_codes
    elif req.gt_file is not None:
        try:
            gt_codes = load_gt_codes(req.gt_file)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="gt_codes or gt_file required",
        )

    reward = calculate_reward(gt_codes, req.enh_codes, req.org_codes)
    log.info(
        "reward computed | reward=%.4f | note_id=%s",
        reward, req.note_id,
    )

    post_to_rl_block(reward, req)

    return RewardResponse(reward=reward)


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "tree_nodes": ICD10_GRAPH.number_of_nodes(),
        "tree_edges": ICD10_GRAPH.number_of_edges(),
    }


# ── Run hint ─────────────────────────────────────────────────────────────────
# uvicorn reward_metrics_svc.main:app --host 0.0.0.0 --port 8002

# import os
# import json
# from typing import List, Optional
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import networkx as nx
# import requests

# # --- Config ---
# ICD10_TREE_PATH = os.path.join(os.path.dirname(__file__), "icd10_tree.json")
# GT_CODES_DIR = os.path.join(os.path.dirname(__file__), "gt_codes")
# RL_BLOCK_ENDPOINT = os.environ.get(
#     "RL_BLOCK_ENDPOINT", None
# )  # Set this in your environment


# # --- Data Models ---
# class RewardRequest(BaseModel):
#     enh_codes: List[str]
#     org_codes: List[str]
#     gt_codes: Optional[List[str]] = None
#     gt_file: Optional[str] = None  # Optionally specify a file in gt_codes/


# class RewardResponse(BaseModel):
#     reward: float


# # --- Load ICD-10 Tree ---
# def load_icd10_tree():
#     with open(ICD10_TREE_PATH, "r") as f:
#         tree = json.load(f)
#     G = nx.DiGraph()
#     for parent, children in tree.items():
#         for child in children:
#             G.add_edge(parent.upper(), child.upper())
#         if not children:
#             G.add_node(parent.upper())
#     return G


# ICD10_GRAPH = load_icd10_tree()


# # --- Utility Functions ---
# def get_max_path_length(graph):
#     # Longest shortest path in the graph
#     lengths = dict(nx.all_pairs_shortest_path_length(graph))
#     max_len = 0
#     for d in lengths.values():
#         if d:
#             max_len = max(max_len, max(d.values()))
#     return max_len or 1


# MAX_PATH_LENGTH = get_max_path_length(ICD10_GRAPH)


# def load_gt_codes(gt_file: str) -> List[str]:
#     path = os.path.join(GT_CODES_DIR, gt_file)
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
#     with open(path, "r") as f:
#         codes = json.load(f)
#     return [c.upper() for c in codes]


# def normalize_codes(codes: List[str]) -> List[str]:
#     return [c.upper() for c in codes]


# def distance_between(a: List[str], b: List[str], graph: nx.DiGraph) -> float:
#     # Symmetric average of min shortest path distances
#     a = normalize_codes(a)
#     b = normalize_codes(b)
#     if not a or not b:
#         return 1.0  # Max distance if any set is empty
#     dists = []
#     for code_a in a:
#         min_dist = min(
#             (
#                 nx.shortest_path_length(graph, code_a, code_b)
#                 if nx.has_path(graph, code_a, code_b)
#                 else MAX_PATH_LENGTH
#             )
#             for code_b in b
#         )
#         dists.append(min_dist)
#     for code_b in b:
#         min_dist = min(
#             (
#                 nx.shortest_path_length(graph, code_b, code_a)
#                 if nx.has_path(graph, code_b, code_a)
#                 else MAX_PATH_LENGTH
#             )
#             for code_a in a
#         )
#         dists.append(min_dist)
#     avg_dist = sum(dists) / len(dists)
#     return min(avg_dist / MAX_PATH_LENGTH, 1.0)


# def calculate_reward(gt_codes, enh_codes, org_codes, graph):
#     d_gt_enh = distance_between(gt_codes, enh_codes, graph)
#     d_gt_org = distance_between(gt_codes, org_codes, graph)
#     d_enh_org = distance_between(enh_codes, org_codes, graph)
#     if d_gt_enh < d_gt_org:
#         return round(1.0 - d_gt_enh, 4)
#     elif d_gt_enh > d_gt_org:
#         return round(-d_gt_enh, 4)
#     else:
#         if d_enh_org == 0:
#             return 1.0
#         else:
#             return round(-d_gt_enh, 4)


# def post_to_rl_block(reward: float):
#     if RL_BLOCK_ENDPOINT:
#         try:
#             requests.post(RL_BLOCK_ENDPOINT, json={"reward": reward})
#         except Exception as e:
#             print(f"Failed to POST reward to RL block: {e}")


# # --- FastAPI App ---
# app = FastAPI()


# @app.post("/reward", response_model=RewardResponse)
# def reward_endpoint(req: RewardRequest):
#     if req.gt_codes:
#         gt_codes = req.gt_codes
#     elif req.gt_file:
#         gt_codes = load_gt_codes(req.gt_file)
#     else:
#         raise HTTPException(status_code=400, detail="gt_codes or gt_file required")
#     enh_codes = req.enh_codes
#     org_codes = req.org_codes
#     reward = calculate_reward(gt_codes, enh_codes, org_codes, ICD10_GRAPH)
#     post_to_rl_block(reward)
#     return RewardResponse(reward=reward)
