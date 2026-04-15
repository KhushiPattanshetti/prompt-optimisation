"""
main.py — FastAPI application entry-point for trajectory_store_svc.

Start the server:
    uvicorn main:app --host 0.0.0.0 --port 8200 --reload

Run simulation standalone (no HTTP server needed):
    python -m simulation.simulator [num_rollouts]
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import router
from utils.logging import get_logger

log = get_logger("main")

app = FastAPI(
    title="trajectory_store_svc",
    description=(
        "Generates rollouts, stores them in JSONL, computes advantages, "
        "and outputs prepared batch files for RL training."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    log.info("trajectory_store_svc starting up")


@app.on_event("shutdown")
async def _shutdown() -> None:
    log.info("trajectory_store_svc shutting down")


if __name__ == "__main__":
    import uvicorn
    from config import cfg

    uvicorn.run("main:app", host=cfg.host, port=cfg.port, reload=False)
