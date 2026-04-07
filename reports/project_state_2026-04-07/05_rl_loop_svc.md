# rl_loop_svc Report

## Service role

rl_loop_svc consumes rollout batches and performs PPO/GRPO-style policy updates with checkpointing.

## Major changes since clone

- RL service architecture, lifecycle manager, rollout buffer, and trainer loop were added.
- Distributed training path and safety guards were introduced.
- Rollout ingestion matured to support batching/idempotency/segment loading.
- Latest iteration added hard diagnostics and train-failure status visibility.

Representative files:
- rl_loop_svc/rl/training_loop.py
- rl_loop_svc/scripts/distributed_train_once.py
- rl_loop_svc/app/api_routes.py
- rl_loop_svc/storage/rollout_loader.py

## Current strengths

- Clear trainer lifecycle and API status model.
- Distributed launch guardrails for RAM/VRAM checks.
- Per-rank failure artifact capture now available for root-cause analysis.
- Pipeline-integrated train failure signaling via status fields.

## Current limitations

- Distributed train cycles can fail with CUDA OOM under memory pressure.
- Throughput and stability are sensitive to world size and effective batch shape.
- Barrier errors on non-fault ranks mask the primary rank fault unless per-rank artifacts are inspected.

## Future prospects

- Implement automatic OOM fallback (world size/effective batch downshift + retry).
- Add rank-aware memory profiling before launch.
- Add checkpoint quality gates and health checks before rewriter reload.
