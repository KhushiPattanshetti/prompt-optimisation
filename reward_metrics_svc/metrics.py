"""
metrics.py – Composite ICD-10 set distance (spec §7-11).

Public API
----------
set_distance(gt, pred)           → float   spec §7
coverage_penalty(gt, pred)       → float   spec §8
extra_penalty(gt, pred)          → float   spec §9
cardinality_penalty(gt, pred)    → float   spec §10
distance_between(gt, pred)       → float   spec §11  (scalar, public)
distance_with_components(gt, pred) → (float, dict)   (scalar + breakdown)
"""

import logging
from typing import Dict, List, Tuple

from . import tree as _tree
from .config import ALPHA, BETA, GAMMA, DELTA, WEIGHT_SUM, LAMBDA_EXTRA, LAMBDA_CARD
from .similarity import depth, sim, distance

logger = logging.getLogger("reward_metrics_svc.metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Set-to-Set Distance  (spec §7)
# ─────────────────────────────────────────────────────────────────────────────


def set_distance(gt: List[str], pred: List[str]) -> float:
    """
    Bidirectional average of directed min-distances.

    Edge cases (spec §7):
      both empty  → 0.0
      pred empty  → 1.0
      gt   empty  → 0.0
    """
    if not gt and not pred:
        return 0.0
    if not pred:
        return 1.0
    if not gt:
        return 0.0
    d_gt_pred = sum(min(distance(g, p) for p in pred) for g in gt) / len(gt)
    d_pred_gt = sum(min(distance(p, g) for g in gt) for p in pred) / len(pred)
    return (d_gt_pred + d_pred_gt) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Penalty Functions  (spec §8-10)
# ─────────────────────────────────────────────────────────────────────────────


def coverage_penalty(gt: List[str], pred: List[str]) -> float:
    """
    P_cov: depth-weighted penalty for GT codes not covered by pred.  spec §8

    For each gt_i:
      coverage_score = max sim(gt_i, pred_j)
      weight         = depth(gt_i) / max_depth
      penalty(gt_i)  = weight * (1 - coverage_score)
    P_cov = mean over all gt_i
    """
    if not gt:
        return 0.0
    penalties: List[float] = []
    for g in gt:
        cov_score = max(sim(g, p) for p in pred) if pred else 0.0
        w = depth(g) / _tree.state.max_depth
        pen = w * (1.0 - cov_score)
        penalties.append(pen)
        logger.debug(
            "coverage | gt=%-8s  depth=%-2d  weight=%.3f  cov_score=%.4f  penalty=%.4f",
            g,
            depth(g),
            w,
            cov_score,
            pen,
        )
    p_cov = sum(penalties) / len(gt)
    logger.debug("P_cov = %.4f", p_cov)
    return p_cov


def extra_penalty(gt: List[str], pred: List[str]) -> float:
    """
    P_extra_final: λ_extra × depth-inverse-weighted penalty for spurious
    predicted codes.  spec §9

    For each pred_j:
      match_score   = max sim(pred_j, gt_i)
      weight        = 1 - depth(pred_j)/max_depth
      penalty(pred_j) = weight * (1 - match_score)
    P_extra_final = λ_extra * mean over all pred_j
    """
    if not pred:
        return 0.0
    penalties: List[float] = []
    for p in pred:
        match_score = max(sim(p, g) for g in gt) if gt else 0.0
        w = 1.0 - (depth(p) / _tree.state.max_depth)
        pen = w * (1.0 - match_score)
        penalties.append(pen)
        logger.debug(
            "extra    | pred=%-8s  depth=%-2d  weight=%.3f  match_score=%.4f  penalty=%.4f",
            p,
            depth(p),
            w,
            match_score,
            pen,
        )
    raw = sum(penalties) / len(pred)
    p_extra_final = LAMBDA_EXTRA * raw
    logger.debug(
        "P_extra_final = %.4f  (λ=%.2f × raw=%.4f)",
        p_extra_final,
        LAMBDA_EXTRA,
        raw,
    )
    return p_extra_final


def cardinality_penalty(gt: List[str], pred: List[str]) -> float:
    """
    P_card_final: λ_card × normalised |n_pred - n_gt| / n_gt.  spec §10
    """
    n_gt = len(gt)
    n_pred = len(pred)
    if n_gt == 0:
        return 0.0
    p_card = abs(n_pred - n_gt) / n_gt
    p_card_norm = min(1.0, p_card)
    p_card_final = LAMBDA_CARD * p_card_norm
    logger.debug(
        "P_card | n_gt=%d  n_pred=%d  p_card=%.4f  p_card_final=%.4f",
        n_gt,
        n_pred,
        p_card,
        p_card_final,
    )
    return p_card_final


# ─────────────────────────────────────────────────────────────────────────────
# Composite Distance  (spec §11)
# ─────────────────────────────────────────────────────────────────────────────


def distance_with_components(
    gt: List[str], pred: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute D_final and return both the scalar and its component breakdown.

    D = α·D_set + β·P_cov + γ·P_extra_final + δ·P_card_final
    D_final = D / (α+β+γ+δ)   ∈ [0, 1]
    """
    d_set = set_distance(gt, pred)
    p_cov = coverage_penalty(gt, pred)
    p_ext = extra_penalty(gt, pred)
    p_car = cardinality_penalty(gt, pred)

    D = ALPHA * d_set + BETA * p_cov + GAMMA * p_ext + DELTA * p_car
    D_final = D / WEIGHT_SUM
    logger.debug(
        "distance_between | D_set=%.4f  P_cov=%.4f  P_extra=%.4f  P_card=%.4f  →  D_final=%.4f",
        d_set,
        p_cov,
        p_ext,
        p_car,
        D_final,
    )
    components = {"D_set": d_set, "P_cov": p_cov, "P_extra": p_ext, "P_card": p_car}
    return D_final, components


def distance_between(gt: List[str], pred: List[str]) -> float:
    """Public scalar interface: composite distance ∈ [0, 1].  spec §11"""
    d_final, _ = distance_with_components(gt, pred)
    return d_final
