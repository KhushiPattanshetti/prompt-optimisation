"""
schemas/rollout_schema.py — Pydantic model for a single rollout record.

All inter-module data exchange uses this schema so that serialisation,
validation, and documentation stay in one place.
"""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Rollout(BaseModel):
    rollout_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    rewritten_prompt: str
    log_prob_old: float
    value_estimate: float
    reward: float

    # Optional metadata added during preprocessing
    group_id: Optional[str] = None
    sample_weight: float = 1.0

    @field_validator("rollout_id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError(f"rollout_id must be a valid UUID, got: {v!r}") from exc
        return v

    @field_validator("log_prob_old")
    @classmethod
    def _validate_log_prob(cls, v: float) -> float:
        if v > 0:
            raise ValueError(
                f"log_prob_old must be ≤ 0 (it is a log-probability), got {v}"
            )
        return v

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Rollout":
        return cls(**data)
