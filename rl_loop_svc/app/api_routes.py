"""
Value head: MLP mapping pooled hidden state → scalar V(s).

FIX SUMMARY (from review):
  - Default hidden_size was 768 (GPT-2).
    Phi-3-mini hidden size is 3072.
    ValueHead(768→1) would throw a shape mismatch when fed
    Phi-3 hidden states of shape (B, T, 3072).
  - Default changed to 3072. Settings.hidden_size also fixed to 3072.
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Two-layer MLP value head.

    Args:
        hidden_size: Dimensionality of the input hidden state.
                     Must match the policy model hidden size.
                     Phi-3-mini = 3072.
        dropout:     Dropout probability applied between layers.
    """

    def __init__(self, hidden_size: int = 3072, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape (B, T, H).

        Returns:
            values: Tensor of shape (B,) — mean-pooled value estimate.
        """
        pooled = hidden_states.mean(dim=1)      # (B, H)
        return self.net(pooled).squeeze(-1)     # (B,)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=device))

# """
# FastAPI route definitions for the RL training microservice.

# Endpoints:
#     GET  /status      → current training state + metrics
#     POST /train       → manually trigger one training cycle
#     GET  /checkpoint  → latest checkpoint metadata
# """

# import logging
# from typing import Any, Dict, Optional

# from fastapi import APIRouter, BackgroundTasks, HTTPException
# from pydantic import BaseModel

# logger = logging.getLogger(__name__)

# router = APIRouter()

# # The TrainingLoop instance is injected at app startup (set by main.py)
# _training_loop = None  # type: ignore


# def set_training_loop(loop: Any) -> None:  # pragma: no cover
#     global _training_loop
#     _training_loop = loop


# # ── Response schemas ──────────────────────────────────────────────────────────


# class StatusResponse(BaseModel):
#     trainer_state: str
#     rollouts_loaded: int
#     training_step: int
#     last_loss: float
#     kl_divergence: float


# class TrainResponse(BaseModel):
#     triggered: bool
#     message: str


# class CheckpointResponse(BaseModel):
#     available: bool
#     metadata: Optional[Dict[str, Any]] = None


# # ── Routes ─────────────────────────────────────────────────────────────────────


# @router.get("/status", response_model=StatusResponse, summary="Current training status")
# def get_status() -> StatusResponse:
#     """Return current lifecycle state and live training metrics."""
#     if _training_loop is None:
#         raise HTTPException(status_code=503, detail="Training loop not initialised")

#     return StatusResponse(
#         trainer_state=_training_loop.lifecycle.state.value,
#         rollouts_loaded=_training_loop.rollouts_loaded,
#         training_step=_training_loop.training_step,
#         last_loss=_training_loop.last_loss,
#         kl_divergence=_training_loop.kl_controller.last_kl,
#     )


# def _background_train() -> None:
#     """Run one RL cycle in a background thread."""
#     if _training_loop is None:
#         return
#     try:
#         _training_loop.run_once()
#     except Exception as exc:
#         logger.error("Background training failed: %s", exc, exc_info=True)
#         _training_loop.lifecycle.reset()


# @router.post(
#     "/train", response_model=TrainResponse, summary="Trigger manual training cycle"
# )
# def trigger_train(background_tasks: BackgroundTasks) -> TrainResponse:
#     """
#     Manually kick off one collect → train → checkpoint cycle.
#     Returns immediately; training runs in the background.
#     """
#     if _training_loop is None:
#         raise HTTPException(status_code=503, detail="Training loop not initialised")

#     from rl.lifecycle_manager import TrainerState

#     if _training_loop.lifecycle.state != TrainerState.IDLE:
#         return TrainResponse(
#             triggered=False,
#             message=f"Trainer is busy: {_training_loop.lifecycle.state.value}",
#         )

#     background_tasks.add_task(_background_train)
#     return TrainResponse(triggered=True, message="Training cycle started in background")


# @router.get(
#     "/checkpoint",
#     response_model=CheckpointResponse,
#     summary="Latest checkpoint metadata",
# )
# def get_checkpoint() -> CheckpointResponse:
#     """Return metadata of the most recently saved checkpoint."""
#     if _training_loop is None:
#         raise HTTPException(status_code=503, detail="Training loop not initialised")

#     meta = _training_loop.checkpoint_manager.load_latest_meta()
#     if meta is None:
#         return CheckpointResponse(available=False)
#     return CheckpointResponse(available=True, metadata=meta)
