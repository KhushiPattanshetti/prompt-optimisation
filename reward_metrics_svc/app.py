"""
app.py – FastAPI application wiring routes, lifespan, and request handling.

Endpoints
---------
POST /compute_reward   spec §14.1
GET  /health           spec §14.2
POST /health           spec §14.2
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import tree as _tree
from .config import ALPHA, BETA, GAMMA, DELTA, LAMBDA_EXTRA, LAMBDA_CARD
from .config import W_TREE, W_EXACT, W_STRUCTURE
from .log_utils import log_coverage_debug, log_extra_debug, update_aggregate_stats
from .reward import compute_reward
from .rollout import enqueue_rollout
from .schemas import (
    ComponentMetrics,
    Diagnostics,
    HealthResponse,
    RewardComponents,
    RewardMetrics,
    RewardRequest,
    RewardResponse,
)

logger = logging.getLogger("reward_metrics_svc.app")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan  (ASGI startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Reload ICD-10 tree on ASGI boot (tree is also eagerly loaded at import)."""
    _tree.load()
    logger.info(
        "reward_metrics_svc started | tree_nodes=%d  max_depth=%d  "
        "config=α=%.1f β=%.1f γ=%.1f δ=%.1f λe=%.2f λc=%.2f "
        "w_tree=%.2f w_exact=%.2f w_struct=%.2f",
        len(_tree.state.depth_map) - 1,
        _tree.state.max_depth,
        ALPHA,
        BETA,
        GAMMA,
        DELTA,
        LAMBDA_EXTRA,
        LAMBDA_CARD,
        W_TREE,
        W_EXACT,
        W_STRUCTURE,
    )
    yield
    logger.info("reward_metrics_svc shutting down.")


app = FastAPI(title="reward_metrics_svc", version="3.0.0", lifespan=_lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# POST /compute_reward  (spec §14.1)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/compute_reward", response_model=RewardResponse)
def compute_reward_endpoint(req: RewardRequest) -> RewardResponse:
    # ── 1. Deduplicate (spec §4) ───────────────────────────────────────────────
    gt = list({c.upper() for c in req.gt_codes})
    enh = list({c.upper() for c in req.enh_codes})
    org = list({c.upper() for c in req.org_codes})
    invalid = [c.upper() for c in req.invalid_codes]
    duplicates = [c.upper() for c in req.duplicate_codes]

    # ── 2. Per-request input log (spec §15.1) ─────────────────────────────────
    logger.info(
        "REQUEST | note_id=%-20s | gt=%s | enh=%s | org=%s"
        " | parsing_ok=%s | invalid=%s | dupes=%s",
        req.note_id,
        gt,
        enh,
        org,
        req.parsing_success,
        invalid,
        duplicates,
    )

    # ── 3. Parsing failure short-circuit (spec §4) ────────────────────────────
    if not req.parsing_success:
        logger.warning(
            "note_id=%-20s | parsing_success=False → R_structure=-1", req.note_id
        )

    # ── 4. Core computation ───────────────────────────────────────────────────
    reward, metrics = compute_reward(
        gt,
        enh,
        org,
        invalid_codes=invalid,
        duplicate_codes=duplicates,
        parsing_success=req.parsing_success,
    )

    # ── 5. Per-request result log (spec §15.1) ────────────────────────────────
    rc = metrics["reward_components"]
    logger.info(
        "RESULT  | note_id=%-20s | D_enh=%.4f  D_org=%.4f  ΔD=%.4f  reward=%.4f"
        "  R_tree=%.4f  R_exact=%.4f  R_structure=%.4f",
        req.note_id,
        metrics["D_enh"],
        metrics["D_org"],
        metrics["delta_D"],
        reward,
        rc["R_tree"],
        rc["R_exact"],
        rc["R_structure"],
    )

    # ── 6. Critical debug logs (spec §15.3) ───────────────────────────────────
    log_coverage_debug(gt, enh, label="enh")
    log_coverage_debug(gt, org, label="org")
    log_extra_debug(enh, gt, label="enh")
    log_extra_debug(org, gt, label="org")

    # ── 7. Aggregate stats tracker (spec §15.4) ───────────────────────────────
    update_aggregate_stats(reward, metrics["delta_D"])

    # ── 8. Rollout queue (spec §16) ───────────────────────────────────────────
    enqueue_rollout(
        note_id=req.note_id,
        state=req.state,
        action=req.action,
        reward=reward,
        log_prob_old=req.log_prob_old,
        value_estimate=req.value_estimate,
    )

    # ── 9. Build response ─────────────────────────────────────────────────────
    diag = metrics["diagnostics"]
    return RewardResponse(
        note_id=req.note_id,
        reward=reward,
        metrics=RewardMetrics(
            D_enh=metrics["D_enh"],
            D_org=metrics["D_org"],
            delta_D=metrics["delta_D"],
            reward_components=RewardComponents(**rc),
            components_enh=ComponentMetrics(**metrics["components_enh"]),
            components_org=ComponentMetrics(**metrics["components_org"]),
            diagnostics=Diagnostics(**diag),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET|POST /health  (spec §14.2)
# ─────────────────────────────────────────────────────────────────────────────


def _health_response() -> HealthResponse:
    loaded = _tree.state.loaded
    node_count = max(0, len(_tree.state.depth_map) - 1)  # exclude VIRTUAL_ROOT
    logger.debug("HEALTH | tree_loaded=%s  node_count=%d", loaded, node_count)
    return HealthResponse(status="ok", tree_loaded=loaded, node_count=node_count)


@app.get("/health", response_model=HealthResponse)
def health_get() -> HealthResponse:
    return _health_response()


@app.post("/health", response_model=HealthResponse)
def health_post() -> HealthResponse:
    return _health_response()
