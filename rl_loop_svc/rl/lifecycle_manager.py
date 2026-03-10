"""
Lifecycle state machine for the RL trainer.

States:
    COLLECT    → read rollout files
    TRAIN      → run PPO update
    CHECKPOINT → persist model weights
    IDLE       → wait for new rollouts
"""

import logging
from enum import Enum, auto
from threading import Lock

logger = logging.getLogger(__name__)


class TrainerState(str, Enum):
    COLLECT = "COLLECT"
    TRAIN = "TRAIN"
    CHECKPOINT = "CHECKPOINT"
    IDLE = "IDLE"


class LifecycleManager:
    """
    Thread-safe state machine that manages trainer lifecycle transitions.
    """

    # Valid transitions: current_state → set of allowed next states
    _VALID_TRANSITIONS: dict[TrainerState, set[TrainerState]] = {
        TrainerState.IDLE: {TrainerState.COLLECT},
        TrainerState.COLLECT: {TrainerState.TRAIN, TrainerState.IDLE},
        TrainerState.TRAIN: {TrainerState.CHECKPOINT, TrainerState.IDLE},
        TrainerState.CHECKPOINT: {TrainerState.IDLE},
    }

    def __init__(self) -> None:
        self._state = TrainerState.IDLE
        self._lock = Lock()

    @property
    def state(self) -> TrainerState:
        return self._state

    def transition(self, new_state: TrainerState) -> None:
        """
        Attempt a state transition.

        Raises:
            ValueError: If the transition is not allowed.
        """
        with self._lock:
            allowed = self._VALID_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                raise ValueError(
                    f"Invalid transition: {self._state} → {new_state}. "
                    f"Allowed: {allowed}"
                )
            logger.info("Lifecycle: %s → %s", self._state.value, new_state.value)
            self._state = new_state

    def reset(self) -> None:
        """Force-reset to IDLE (use only for error recovery)."""
        with self._lock:
            logger.warning("Lifecycle force-reset to IDLE from %s", self._state.value)
            self._state = TrainerState.IDLE
