"""
simulation/simulator.py — End-to-end pipeline runner with synthetic data.

This module is the "main loop" of the service.  It can be invoked:
  • directly:  python -m simulation.simulator
  • as a library:  from simulation.simulator import run_simulation

Pipeline executed per call
──────────────────────────
1. generate_rollout()          → Rollout (with stub reward)
2. append_rollout()            → JSONL on disk
3. RolloutLoader.load_new()    → List[Rollout]
4. preprocess()                → group_id, sample_weight
5. RolloutBuffer.add_many()
6. RolloutBuffer.flush()       → List[RolloutBatch]
7. compute_advantages()        → advantages, returns
8. BatchWriter.write()         → prepared_batch_*.json
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import List, Optional

from config import cfg
from schemas.rollout_schema import Rollout
from storage.rollout_loader import RolloutLoader, append_rollout, new_rollout_filepath
from buffer.rollout_buffer import RolloutBuffer, RolloutBatch
from processing.preprocessing import preprocess
from processing.advantage import compute_advantages
from serving.batch_writer import BatchWriter
from utils.logging import get_logger, log_separator

log = get_logger("simulation.simulator")


# ── Reward stub ───────────────────────────────────────────────────────────────


def get_reward(prompt: str, rewritten_prompt: str) -> float:  # noqa: ARG001
    """
    External reward service stub.

    In production this would call the reward_metrics_svc HTTP endpoint.
    Here we sample from a configurable Gaussian distribution.

    The value is treated as already normalised (cfg.reward_already_normalized=True).
    """
    reward = random.gauss(cfg.sim_reward_mean, cfg.sim_reward_std)
    # Clip to [-1, 1] — sensible range for a normalised reward
    return max(-1.0, min(1.0, reward))


# ── Rollout generator ─────────────────────────────────────────────────────────

_SAMPLE_PROMPTS = [
    "Explain the water cycle in simple terms.",
    "What is the capital of France?",
    "Summarise the French Revolution.",
    "How does a neural network learn?",
    "Describe photosynthesis.",
    "What causes thunder?",
    "Explain supply and demand.",
    "What is the Turing test?",
]


def generate_rollout(
    prompt: Optional[str] = None,
    reward_mean: float | None = None,
    reward_std: float | None = None,
) -> Rollout:
    """
    Produce a single synthetic rollout.

    Args:
        prompt:      Override fixed prompt (random choice if None).
        reward_mean: Override cfg.sim_reward_mean for this call.
        reward_std:  Override cfg.sim_reward_std for this call.
    """
    if prompt is None:
        prompt = random.choice(_SAMPLE_PROMPTS)
    rewritten = f"[Rewritten] {prompt}"

    # Temporarily override if caller specifies custom distribution
    _orig_mean, _orig_std = cfg.sim_reward_mean, cfg.sim_reward_std
    if reward_mean is not None:
        cfg.sim_reward_mean = reward_mean
    if reward_std is not None:
        cfg.sim_reward_std = reward_std

    reward = get_reward(prompt, rewritten)

    cfg.sim_reward_mean = _orig_mean
    cfg.sim_reward_std = _orig_std

    rollout = Rollout(
        rollout_id=str(uuid.uuid4()),
        prompt=prompt,
        rewritten_prompt=rewritten,
        log_prob_old=-abs(random.gauss(0.5, 0.2)),  # always ≤ 0
        value_estimate=random.gauss(0.3, 0.1),
        reward=reward,
    )
    log.info(
        "Rollout generated",
        extra={"rollout_id": rollout.rollout_id, "reward": f"{reward:.4f}"},
    )
    return rollout


# ── Full pipeline ─────────────────────────────────────────────────────────────


def run_simulation(
    num_rollouts: int | None = None,
    batch_size: int | None = None,
) -> List[Path]:
    """
    Run the full trajectory pipeline end-to-end.

    Returns:
        List of Paths to written batch files.
    """
    num_rollouts = num_rollouts or cfg.sim_num_rollouts
    if batch_size:
        cfg.batch_size = batch_size

    log_separator(log, "SIMULATION START")
    log.info(
        "Simulation parameters",
        extra={
            "num_rollouts": num_rollouts,
            "batch_size": cfg.batch_size,
            "advantage_mode": cfg.advantage_mode,
            "batch_reuse_mode": cfg.batch_reuse_mode,
        },
    )

    # ── Step 1 & 2 : generate + store ────────────────────────────────────────
    log_separator(log, "STAGE 1: Generate & Store")
    rollout_file = new_rollout_filepath()
    for i in range(num_rollouts):
        rollout = generate_rollout()
        append_rollout(rollout, rollout_file)
    log.info(f"Wrote {num_rollouts} rollouts", extra={"file": str(rollout_file)})

    # ── Step 3 : load ─────────────────────────────────────────────────────────
    log_separator(log, "STAGE 2: Load from JSONL")
    loader = RolloutLoader()
    loaded = loader.load_new()
    log.info(f"Loaded {len(loaded)} rollouts from disk")

    # ── Step 4 : preprocess ───────────────────────────────────────────────────
    log_separator(log, "STAGE 3: Preprocess")
    preprocessed = preprocess(loaded)

    # ── Step 5 & 6 : buffer + flush ───────────────────────────────────────────
    log_separator(log, "STAGE 4: Buffer & Flush")
    buffer = RolloutBuffer(batch_size=cfg.batch_size)
    buffer.add_many(preprocessed)
    batches: List[RolloutBatch] = buffer.flush()
    log.info(
        "Buffer flushed",
        extra={"batches_ready": len(batches), "pending": buffer.pending_count()},
    )

    # ── Step 7 & 8 : advantages + write ──────────────────────────────────────
    log_separator(log, "STAGE 5: Advantages & Batch Writing")
    writer = BatchWriter()
    written_paths: List[Path] = []

    for batch in batches:
        advantages, returns = compute_advantages(batch)
        batch.advantages = advantages
        batch.returns = returns
        path = writer.write(batch)
        written_paths.append(path)

        # Demonstrate reuse behaviour
        if cfg.batch_reuse_mode == "repeat":
            while writer.should_reuse():
                writer.record_reuse()

    log_separator(log, "SIMULATION COMPLETE")
    log.info(
        "Pipeline finished",
        extra={
            "batches_written": len(written_paths),
            "rollouts_dropped": buffer.pending_count(),
        },
    )
    return written_paths


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    num = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.sim_num_rollouts
    paths = run_simulation(num_rollouts=num)
    print("\nBatch files created:")
    for p in paths:
        print(f"  {p}")
