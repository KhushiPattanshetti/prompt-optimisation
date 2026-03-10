"""
Pydantic schemas for rollout trajectory data.

Each rollout file produced by the Trajectory Store must conform to
the RolloutFile schema.
"""

from typing import List
from pydantic import BaseModel, Field


class RolloutEntry(BaseModel):
    """Single trajectory step produced by the Trajectory Store."""

    original_prompt: str = Field(
        ..., description="Clinical note / original prompt (state s)"
    )
    rewritten_prompt: str = Field(
        ..., description="Enhanced prompt produced by the Prompt Rewriter (action a)"
    )
    reward: float = Field(
        ..., ge=-1.0, le=1.0, description="Scalar reward from Reward Metrics Service"
    )
    log_prob_old: float = Field(
        ..., description="Log-probability of the action under the behaviour policy"
    )
    value_estimate: float = Field(
        ..., description="Value estimate V(s) at collection time"
    )


class RolloutFile(BaseModel):
    """Top-level wrapper matching the JSON files written to rollouts/."""

    rollouts: List[RolloutEntry] = Field(..., min_length=1)
