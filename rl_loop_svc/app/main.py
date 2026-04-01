"""
FastAPI application entry point for the RL training microservice.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8004

FIX SUMMARY (from review):
  - set_training_loop() now also receives rollouts_dir so the
    POST /rollout endpoint knows where to write incoming rollouts.
  - Policy and reference models now use 4-bit quantization
    (delegated to PolicyModel and ReferenceModel constructors).
  - RL checkpoint loading wired into PolicyModel constructor.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI

from app.api_routes import router, set_training_loop
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _find_latest_rl_checkpoint() -> Path | None:
    """Return the most recent checkpoint directory, or None."""
    ckpt_dir = settings.checkpoints_dir
    if not ckpt_dir.exists():
        return None
    subdirs = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint_")
    ]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise all RL components at startup.

    Loading order:
        1. Ensure directories exist.
        2. Resolve latest RL checkpoint (if any).
        3. Load PolicyModel (4-bit + LoRA, from checkpoint or HuggingFace).
        4. Load ReferenceModel (4-bit, frozen, always from HuggingFace).
        5. Build ValueHead with correct hidden_size (3072 for Phi-3).
        6. Wire up TrainingLoop and expose via API.
    """
    logger.info("Starting RL training microservice …")

    settings.rollouts_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    from models.policy_model import PolicyModel
    from models.reference_model import ReferenceModel
    from models.value_head import ValueHead
    from rl.training_loop import TrainingLoop
    from storage.checkpoint_manager import CheckpointManager
    from storage.rollout_loader import RolloutLoader

    # Resolve latest RL checkpoint for policy model
    latest_ckpt = _find_latest_rl_checkpoint()
    lora_path   = None
    if latest_ckpt is not None:
        lora_adapter = latest_ckpt / "lora_adapter"
        if lora_adapter.exists():
            lora_path = str(lora_adapter)
            logger.info("Resuming from RL checkpoint: %s", latest_ckpt)
        else:
            logger.warning(
                "Checkpoint dir found but no lora_adapter/ inside: %s",
                latest_ckpt,
            )

    # Load policy model (4-bit + LoRA)
    policy_model = PolicyModel(
        model_name=settings.model_name,
        checkpoint_path=lora_path,   # None → fresh LoRA on first run
    )

    # Load reference model (4-bit, frozen, always base weights)
    reference_model = ReferenceModel(model_name=settings.model_name)

    # Build value head — hidden_size must match Phi-3 (3072)
    value_head = ValueHead(hidden_size=settings.hidden_size).to(
        policy_model.device
    )
    value_head = value_head.to(next(policy_model.model.parameters()).dtype)

    # Load value head weights if checkpoint exists
    if latest_ckpt is not None:
        vh_path = latest_ckpt / "value_head.pt"
        if vh_path.exists():
            import torch
            state = torch.load(vh_path, map_location=str(policy_model.device))
            value_head.load_state_dict(state)
            logger.info("Value head weights loaded from checkpoint")

    rollout_loader     = RolloutLoader(rollouts_dir=settings.rollouts_dir)
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

    # FIX: pass rollouts_dir so POST /rollout knows where to write files
    set_training_loop(training_loop, settings.rollouts_dir)
    logger.info("RL components initialised. Service is ready.")

    yield  # ── application runs here ──────────────────────────────────────

    logger.info("Shutting down RL training microservice.")


app = FastAPI(
    title="RL Training Microservice",
    description=(
        "PPO-based RL trainer that fine-tunes the Prompt Rewriter "
        "using trajectories collected from the ICD coding pipeline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
# """
# FastAPI application entry point for the RL training microservice.

# Start with:
#     uvicorn app.main:app --reload

# All heavy model loading happens once at startup via the lifespan context.
# """

# import logging
# import sys
# from contextlib import asynccontextmanager
# from pathlib import Path

# # Make project root importable when launched with uvicorn from rl_loop_svc/
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from fastapi import FastAPI

# from app.api_routes import router, set_training_loop
# from app.config import settings

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
# )
# logger = logging.getLogger(__name__)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Application lifespan handler.

#     Initialises all RL components on startup and tears down cleanly on
#     shutdown.  Heavy model loading (GPU / CPU) happens here so the API
#     is ready before accepting requests.
#     """
#     logger.info("Starting RL training microservice …")

#     # Ensure required directories exist
#     settings.rollouts_dir.mkdir(parents=True, exist_ok=True)
#     settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)

#     # ── Lazy import to keep startup fast during testing ───────────────────
#     from models.policy_model import PolicyModel
#     from models.reference_model import ReferenceModel
#     from models.value_head import ValueHead
#     from rl.training_loop import TrainingLoop
#     from storage.checkpoint_manager import CheckpointManager
#     from storage.rollout_loader import RolloutLoader

#     policy_model = PolicyModel(model_name=settings.model_name)
#     reference_model = ReferenceModel(model_name=settings.model_name)
#     value_head = ValueHead(hidden_size=settings.hidden_size).to(policy_model.device)

#     rollout_loader = RolloutLoader(rollouts_dir=settings.rollouts_dir)
#     checkpoint_manager = CheckpointManager(
#         checkpoints_dir=settings.checkpoints_dir,
#         max_checkpoints=settings.max_checkpoints,
#     )

#     training_loop = TrainingLoop(
#         rollout_loader=rollout_loader,
#         checkpoint_manager=checkpoint_manager,
#         policy_model=policy_model,
#         reference_model=reference_model,
#         value_head=value_head,
#     )

#     set_training_loop(training_loop)
#     logger.info("RL components initialised. Service is ready.")

#     yield  # ── application runs here ──────────────────────────────────────

#     logger.info("Shutting down RL training microservice.")


# app = FastAPI(
#     title="RL Training Microservice",
#     description=(
#         "PPO-based RL trainer that fine-tunes the Prompt Rewriter language model "
#         "using trajectories collected from the ICD coding pipeline."
#     ),
#     version="0.1.0",
#     lifespan=lifespan,
# )

# app.include_router(router)
