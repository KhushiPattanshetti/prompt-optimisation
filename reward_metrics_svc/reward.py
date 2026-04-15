"""
reward.py – Computes the hybrid scalar RL reward from three sets of ICD-10 codes.

Public API
----------
compute_reward(gt, enh, org,
               invalid_codes=[], duplicate_codes=[], parsing_success=True)
    → (reward: float, metrics: dict)

Hybrid formula (spec §13):
    R_tree      = D_org - D_enh                (improvement signal)
    R_exact     = 2 * J - 1                    (Jaccard ∈ [-1, 1])
    R_structure = 1 - p_invalid - p_dupes      (validity signal, clamped)
    reward      = tanh(w_tree*R_tree + w_exact*R_exact + w_structure*R_structure)
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

from .config import W_TREE, W_EXACT, W_STRUCTURE
from .metrics import distance_with_components
from .similarity import sim

logger = logging.getLogger("reward_metrics_svc.reward")


# ─────────────────────────────────────────────────────────────────────────────
# Exact reward  (spec §12.1)
# ─────────────────────────────────────────────────────────────────────────────


def _r_exact(gt: List[str], enh: List[str]) -> float:
    """Jaccard-based exact reward ∈ [-1, 1]."""
    gt_s, enh_s = set(gt), set(enh)
    union = gt_s | enh_s
    if not union:
        return 1.0  # both empty → perfect
    j = len(gt_s & enh_s) / len(union)
    return 2.0 * j - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Structure reward  (spec §12.2)
# ─────────────────────────────────────────────────────────────────────────────


def _r_structure(
    enh: List[str],
    invalid_codes: List[str],
    duplicate_codes: List[str],
    parsing_success: bool,
) -> float:
    """Structural validity reward ∈ [-1, 1]."""
    if not parsing_success:
        return -1.0
    n = max(1, len(enh))
    p_invalid = len(invalid_codes) / n
    p_dupes = len(duplicate_codes) / n
    r = 1.0 - p_invalid - p_dupes
    return max(-1.0, min(1.0, r))


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward  (spec §13)
# ─────────────────────────────────────────────────────────────────────────────


def compute_reward(
    gt: List[str],
    enh: List[str],
    org: List[str],
    invalid_codes: Optional[List[str]] = None,
    duplicate_codes: Optional[List[str]] = None,
    parsing_success: bool = True,
) -> Tuple[float, Dict]:
    """
    Compute hybrid improvement-based reward for PPO.

    Parameters
    ----------
    gt               : deduplicated ground-truth ICD-10 codes
    enh              : deduplicated enhanced codes (from LLM rewrite)
    org              : deduplicated original codes (baseline)
    invalid_codes    : enh codes that failed ICD validation (from upstream)
    duplicate_codes  : raw duplicates detected in enh (from upstream)
    parsing_success  : whether the LLM produced parseable output

    Returns
    -------
    reward   : float ∈ [-1, 1]
    metrics  : dict with D_enh, D_org, delta_D, reward_components, components_enh,
               components_org, diagnostics
    """
    invalid_codes = invalid_codes or []
    duplicate_codes = duplicate_codes or []

    D_enh, comps_enh = distance_with_components(gt, enh)
    D_org, comps_org = distance_with_components(gt, org)

    delta_D = D_org - D_enh
    R_tree = delta_D
    R_exact = _r_exact(gt, enh)
    R_structure = _r_structure(enh, invalid_codes, duplicate_codes, parsing_success)

    raw = W_TREE * R_tree + W_EXACT * R_exact + W_STRUCTURE * R_structure
    reward = math.tanh(raw)

    logger.debug(
        "compute_reward | D_enh=%.4f  D_org=%.4f  ΔD=%.4f"
        "  R_tree=%.4f  R_exact=%.4f  R_structure=%.4f  raw=%.4f  reward=%.4f",
        D_enh,
        D_org,
        delta_D,
        R_tree,
        R_exact,
        R_structure,
        raw,
        reward,
    )
    logger.debug(
        "compute_reward | w_tree=%.2f  w_exact=%.2f  w_structure=%.2f"
        "  invalid=%s  dupes=%s  parsing_ok=%s",
        W_TREE,
        W_EXACT,
        W_STRUCTURE,
        invalid_codes,
        duplicate_codes,
        parsing_success,
    )

    # Diagnostics for response (spec §14.1 response shape)
    worst_gt = _worst_gt_coverage(gt, enh)
    worst_pred = _worst_pred_match(enh, gt)

    metrics: Dict = {
        "D_enh": round(D_enh, 6),
        "D_org": round(D_org, 6),
        "delta_D": round(delta_D, 6),
        "reward_components": {
            "R_tree": round(R_tree, 6),
            "R_exact": round(R_exact, 6),
            "R_structure": round(R_structure, 6),
        },
        "components_enh": {k: round(v, 6) for k, v in comps_enh.items()},
        "components_org": {k: round(v, 6) for k, v in comps_org.items()},
        "diagnostics": {
            "worst_gt_coverage_code": worst_gt,
            "worst_pred_match_code": worst_pred,
            "invalid_codes": list(invalid_codes),
            "duplicate_codes": list(duplicate_codes),
        },
    }
    return round(reward, 6), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers for diagnostics  (spec §14.3)
# ─────────────────────────────────────────────────────────────────────────────


def _worst_gt_coverage(gt: List[str], pred: List[str]) -> str:
    """GT code with the lowest coverage score (spec §14.3)."""
    if not gt or not pred:
        return gt[0] if gt else ""
    pairs = [(g, max(sim(g, p) for p in pred)) for g in gt]
    return min(pairs, key=lambda x: x[1])[0]


def _worst_pred_match(pred: List[str], gt: List[str]) -> str:
    """Predicted code with the lowest match score to any GT (spec §14.3)."""
    if not pred or not gt:
        return pred[0] if pred else ""
    pairs = [(p, max(sim(p, g) for g in gt)) for p in pred]
    return min(pairs, key=lambda x: x[1])[0]
