"""
FastAPI application entry point for the RL training microservice.

Start with:
    uvicorn app.main:app --reload

All heavy model loading happens once at startup via the lifespan context.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make project root importable when launched with uvicorn from rl_loop_svc/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI

from app.api_routes import router, set_training_loop
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initialises all RL components on startup and tears down cleanly on
    shutdown.  Heavy model loading (GPU / CPU) happens here so the API
    is ready before accepting requests.
    """
    logger.info("Starting RL training microservice …")

    # Ensure required directories exist
    settings.rollouts_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy import to keep startup fast during testing ───────────────────
    from models.policy_model import PolicyModel
    from models.reference_model import ReferenceModel
    from models.value_head import ValueHead
    from rl.training_loop import TrainingLoop
    from storage.checkpoint_manager import CheckpointManager
    from storage.rollout_loader import RolloutLoader

    policy_model = PolicyModel(model_name=settings.model_name)
    reference_model = ReferenceModel(model_name=settings.model_name)
    value_head = ValueHead(hidden_size=settings.hidden_size).to(policy_model.device)

    rollout_loader = RolloutLoader(rollouts_dir=settings.rollouts_dir)
    checkpoint_manager = CheckpointManager(
        checkpoints_dir=settings.checkpoints_dir,
        max_checkpoints=settings.max_checkpoints,
    )

    training_loop = TrainingLoop(
        rollout_loader=rollout_loader,
        checkpoint_manager=checkpoint_manager,
        policy_model=policy_model,
        reference_model=reference_model,
        value_head=value_head,
    )

    set_training_loop(training_loop)
    logger.info("RL components initialised. Service is ready.")

    yield  # ── application runs here ──────────────────────────────────────

    logger.info("Shutting down RL training microservice.")


app = FastAPI(
    title="RL Training Microservice",
    description=(
        "PPO-based RL trainer that fine-tunes the Prompt Rewriter language model "
        "using trajectories collected from the ICD coding pipeline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
