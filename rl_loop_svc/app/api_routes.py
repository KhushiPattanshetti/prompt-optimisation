import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from rl.lifecycle_manager import TrainerState

logger = logging.getLogger(__name__)

router = APIRouter()

_training_loop = None
_rollouts_dir: Optional[Path] = None
_seen_rollout_ids: set[str] = set()
_seen_rollout_ids_index: Optional[Path] = None
_rollout_lock = threading.Lock()


def set_training_loop(loop, rollouts_dir: Path) -> None:
    global _training_loop, _rollouts_dir, _seen_rollout_ids, _seen_rollout_ids_index
    _training_loop = loop
    _rollouts_dir = rollouts_dir
    _seen_rollout_ids_index = _rollouts_dir / ".seen_rollout_ids"
    _seen_rollout_ids = _load_seen_rollout_ids()


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


class TrainResponse(BaseModel):
    triggered: bool
    message: str


class CheckpointResponse(BaseModel):
    available: bool
    metadata: Optional[Dict[str, Any]] = None


class RolloutSubmission(BaseModel):
    rollout_id: Optional[str] = None
    run_id: Optional[str] = None
    group_id: Optional[str] = None
    original_prompt: str
    rewritten_prompt: str
    reward: float = Field(..., ge=-1.0, le=1.0)
    concept_reward: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    sample_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    log_prob_old: float
    value_estimate: Optional[float] = None


class RolloutBatchSubmission(BaseModel):
    run_id: Optional[str] = None
    rollouts: List[RolloutSubmission] = Field(..., min_length=1)


class RolloutAck(BaseModel):
    accepted: bool
    file_path: str
    accepted_count: int = 0
    duplicate_count: int = 0
    run_id: Optional[str] = None


def _load_seen_rollout_ids() -> set[str]:
    if _seen_rollout_ids_index is None or not _seen_rollout_ids_index.exists():
        return set()

    try:
        lines = _seen_rollout_ids_index.read_text(encoding="utf-8").splitlines()
        return {line.strip() for line in lines if line.strip()}
    except Exception:
        return set()


def _persist_seen_rollout_ids() -> None:
    if _seen_rollout_ids_index is None:
        return
    _seen_rollout_ids_index.write_text(
        "\n".join(sorted(_seen_rollout_ids)),
        encoding="utf-8",
    )


def _sanitize_run_id(run_id: Optional[str]) -> str:
    candidate = str(run_id or "default").strip()
    if not candidate:
        candidate = "default"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", candidate)


def _persist_rollout_batch(submissions: List[RolloutSubmission], run_id: Optional[str]) -> RolloutAck:
    if _rollouts_dir is None:
        raise HTTPException(status_code=503, detail="Rollout directory not configured")

    _rollouts_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_id = _sanitize_run_id(run_id or submissions[0].run_id)
    accepted_rollouts: List[Dict[str, Any]] = []
    duplicate_count = 0

    with _rollout_lock:
        for submission in submissions:
            rollout_id = (submission.rollout_id or "").strip()
            if rollout_id:
                if rollout_id in _seen_rollout_ids:
                    duplicate_count += 1
                    continue
                _seen_rollout_ids.add(rollout_id)

            row = submission.model_dump(exclude_none=True)
            row["run_id"] = resolved_run_id
            accepted_rollouts.append(row)

        if accepted_rollouts:
            _persist_seen_rollout_ids()

    if not accepted_rollouts:
        return RolloutAck(
            accepted=True,
            file_path="",
            accepted_count=0,
            duplicate_count=duplicate_count,
            run_id=resolved_run_id,
        )

    payload = {
        "run_id": resolved_run_id,
        "rollouts": accepted_rollouts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": uuid4().hex,
    }

    segment_path = _rollouts_dir / f"rollout_segment_{resolved_run_id}.jsonl"
    with segment_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")

    return RolloutAck(
        accepted=True,
        file_path=str(segment_path),
        accepted_count=len(accepted_rollouts),
        duplicate_count=duplicate_count,
        run_id=resolved_run_id,
    )


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

    return StatusResponse(
        trainer_state=_training_loop.lifecycle.state.value,
        rollouts_loaded=_training_loop.rollouts_loaded,
        training_step=_training_loop.training_step,
        last_loss=_training_loop.last_loss,
        kl_divergence=_training_loop.kl_controller.last_kl,
        last_train_success=_training_loop.last_train_success,
        last_train_error=_training_loop.last_train_error,
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


@router.post("/rollout", response_model=RolloutAck)
def submit_rollout(submission: RolloutSubmission) -> RolloutAck:
    return _persist_rollout_batch([submission], submission.run_id)


@router.post("/rollout_batch", response_model=RolloutAck)
def submit_rollout_batch(submission: RolloutBatchSubmission) -> RolloutAck:
    return _persist_rollout_batch(submission.rollouts, submission.run_id)
