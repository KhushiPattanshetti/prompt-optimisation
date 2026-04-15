"""
api/routes.py — Minimal FastAPI router.

GET /status  → service health + counters
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from config import cfg

router = APIRouter()


@router.get("/status")
def get_status() -> dict:
    rollouts_dir: Path = cfg.rollouts_dir
    batches_dir: Path = cfg.prepared_batches_dir

    total_rollouts = sum(
        1
        for f in rollouts_dir.glob("rollouts_*.jsonl")
        for _ in f.open("r", encoding="utf-8")
        if _.strip()
    )

    total_batches = len(list(batches_dir.glob("prepared_batch_*.json")))

    return {
        "service": "trajectory_store_svc",
        "status": "ok",
        "total_rollouts": total_rollouts,
        "total_batches": total_batches,
        "current_mode": cfg.advantage_mode,
        "batch_reuse_mode": cfg.batch_reuse_mode,
    }
