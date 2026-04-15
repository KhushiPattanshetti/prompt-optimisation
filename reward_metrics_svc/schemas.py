"""
schemas.py – Pydantic request/response models for reward_metrics_svc.  spec §14
"""

from typing import List, Optional
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Request  (mirrors CodeResponse from ICD-10 coding block  spec §3.2)
# ─────────────────────────────────────────────────────────────────────────────


class RewardRequest(BaseModel):
    """Full payload emitted by the ICD-10 coding block."""

    note_id: str
    enh_codes: List[str]
    org_codes: List[str]
    gt_codes: List[str]
    # Parsing / validation metadata (spec §3.2, §12.2)
    parsing_success: bool = True
    invalid_codes: List[str] = []
    duplicate_codes: List[str] = []
    # Raw LLM outputs (optional, for debug logging)
    enh_raw_output: Optional[str] = None
    org_raw_codes: Optional[str] = None
    # PPO rollout fields (spec §16)
    state: str = ""
    action: str = ""
    log_prob_old: float = 0.0
    value_estimate: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Response  (spec §14.1)
# ─────────────────────────────────────────────────────────────────────────────


class RewardComponents(BaseModel):
    """Hybrid reward sub-scores (spec §12)."""

    R_tree: float
    R_exact: float
    R_structure: float


class ComponentMetrics(BaseModel):
    """Distance component breakdown for one side (enh or org)."""

    D_set: float
    P_cov: float
    P_extra: float
    P_card: float


class Diagnostics(BaseModel):
    """Critical debug fields (spec §14.3)."""

    worst_gt_coverage_code: str
    worst_pred_match_code: str
    invalid_codes: List[str]
    duplicate_codes: List[str]


class RewardMetrics(BaseModel):
    D_enh: float
    D_org: float
    delta_D: float
    reward_components: RewardComponents
    components_enh: ComponentMetrics
    components_org: ComponentMetrics
    diagnostics: Diagnostics


class RewardResponse(BaseModel):
    note_id: str
    reward: float
    metrics: RewardMetrics
    # Code lists echoed back so callers can build the pretty-log payload
    og_codes: List[str] = []
    enh_codes: List[str] = []
    gt_codes: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    tree_loaded: bool
    node_count: int
