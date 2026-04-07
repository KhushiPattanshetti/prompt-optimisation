"""
Pydantic schemas for rollout trajectory data.

Each rollout file produced by the Trajectory Store must conform to
the RolloutFile schema.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class RolloutEntry(BaseModel):
    """Single trajectory step produced by the Trajectory Store."""

    rollout_id: Optional[str] = Field(
        default=None,
        description="Optional idempotency key for deduplicating retried rollouts",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional pipeline run identifier for isolation and diagnostics",
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Optional grouping key for relative (GRPO-style) reward normalization",
    )

    original_prompt: str = Field(
        ..., description="Clinical note / original prompt (state s)"
    )
    rewritten_prompt: str = Field(
        ..., description="Enhanced prompt produced by the Prompt Rewriter (action a)"
    )
    reward: float = Field(
        ..., ge=-1.0, le=1.0, description="Scalar reward from Reward Metrics Service"
    )
    concept_reward: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional concept-level dense reward signal",
    )
    sample_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional per-sample weight used during PPO optimization",
    )
    log_prob_old: float = Field(
        ..., description="Log-probability of the action under the behaviour policy"
    )
    value_estimate: Optional[float] = Field(
        default=None,
        description="Optional value estimate V(s) at collection time",
    )


class RolloutFile(BaseModel):
    """Top-level wrapper matching the JSON files written to rollouts/."""

    run_id: Optional[str] = None

    rollouts: List[RolloutEntry] = Field(..., min_length=1)
