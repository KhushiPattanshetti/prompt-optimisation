import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ..rl.lifecycle_manager import TrainerState
from ..storage.trajectory_store_client import TrajectoryStoreClient
from .config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

_training_loop = None
_traj_client: Optional[TrajectoryStoreClient] = None


def set_training_loop(
    loop, rollouts_dir=None
) -> None:  # rollouts_dir kept for backward compat
    global _training_loop, _traj_client
    _training_loop = loop
    _traj_client = TrajectoryStoreClient(base_url=settings.trajectory_store_url)


class StatusResponse(BaseModel):
    trainer_state: str
    rollouts_loaded: int
    training_step: int
    last_loss: float
    kl_divergence: float
    last_train_success: Optional[bool] = None
    last_train_error: Optional[str] = None
    last_train_started_at: Optional[str] = None
    last_train_finished_at: Optional[str] = None
    trajectory_store_healthy: Optional[bool] = None


class TrainResponse(BaseModel):
    triggered: bool
    message: str


class CheckpointResponse(BaseModel):
    available: bool
    metadata: Optional[Dict[str, Any]] = None


def _background_train() -> None:
    if _training_loop is None:
        return
    _training_loop.last_train_started_at = datetime.now(timezone.utc).isoformat()
    _training_loop.last_train_finished_at = None
    _training_loop.last_train_success = None
    _training_loop.last_train_error = None
    try:
        _training_loop.run_once()
        _training_loop.last_train_success = True
        _training_loop.last_train_error = None
    except Exception as exc:
        _training_loop.last_train_success = False
        _training_loop.last_train_error = str(exc)
        logger.error("Background training failed: %s", exc, exc_info=True)
        _training_loop.lifecycle.reset()
    finally:
        _training_loop.last_train_finished_at = datetime.now(timezone.utc).isoformat()


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    if _training_loop is None:
        raise HTTPException(status_code=503, detail="Training loop not initialised")

    traj_healthy = _traj_client.is_healthy() if _traj_client is not None else None

    return StatusResponse(
        trainer_state=_training_loop.lifecycle.state.value,
        rollouts_loaded=_training_loop.rollouts_loaded,
        training_step=_training_loop.training_step,
        last_loss=_training_loop.last_loss,
        kl_divergence=_training_loop.kl_controller.last_kl,
        last_train_success=_training_loop.last_train_success,
        last_train_error=_training_loop.last_train_error,
        trajectory_store_healthy=traj_healthy,
        last_train_started_at=_training_loop.last_train_started_at,
        last_train_finished_at=_training_loop.last_train_finished_at,
    )


@router.post("/train", response_model=TrainResponse)
def trigger_train(background_tasks: BackgroundTasks) -> TrainResponse:
    if _training_loop is None:
        raise HTTPException(status_code=503, detail="Training loop not initialised")

    if _training_loop.lifecycle.state != TrainerState.IDLE:
        return TrainResponse(
            triggered=False,
            message=f"Trainer is busy: {_training_loop.lifecycle.state.value}",
        )

    background_tasks.add_task(_background_train)
    return TrainResponse(triggered=True, message="Training cycle started in background")


@router.get("/checkpoint", response_model=CheckpointResponse)
def get_checkpoint() -> CheckpointResponse:
    if _training_loop is None:
        raise HTTPException(status_code=503, detail="Training loop not initialised")

    meta = _training_loop.checkpoint_manager.load_latest_meta()
    if meta is None:
        return CheckpointResponse(available=False)
    return CheckpointResponse(available=True, metadata=meta)
