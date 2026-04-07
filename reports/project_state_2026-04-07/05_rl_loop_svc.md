# rl_loop_svc Detailed Report

## Navigation

- [Overview](00_overview.md)
- [Dataset Service](01_dataset_svc.md)
- [Rewriter Service](02_rewriter_inference_svc.md)
- [ICD10 Service](03_icd10_coding_svc.md)
- [Reward Service](04_reward_metrics_svc.md)
- [RL Loop Service](05_rl_loop_svc.md)
- [Theory Meta-Doc](07_theoretical_concepts_flow_methods_schemas.md)

## 1. What this service does

rl_loop_svc is the trainer and policy lifecycle manager.

It is responsible for:
- receiving rollout data from reward service
- maintaining policy/value training state
- running one training cycle on demand
- exposing status and diagnostics

Beginner mental model:
- If reward service is the teacher giving scores,
- rl_loop_svc is the student optimizer that updates policy weights.

## 2. Core RL concepts used here

## 2.1 Policy-gradient objective

The model is updated to increase probability of higher-reward outputs.
The service uses PPO-style clipped updates for stability.

Core idea:
- old policy produced sampled actions during rollout
- new policy is optimized but prevented from moving too far in one step

## 2.2 Advantage estimation

Advantage estimates whether an action did better than expected baseline value.

High-level relation:

A_t = R_t - V_t

where:
- R_t is reward-like return signal
- V_t is value estimate baseline

This service includes grouped relative advantage logic to normalize within groups.

## 2.3 KL control

KL divergence between new and reference/old policy is monitored and controlled.
Purpose:
- avoid policy collapse
- keep updates stable

## 2.4 Grouped relative rewards (GRPO-like)

Rollouts can be grouped by group_id.
Within group, rewards are normalized relatively so learning focuses on ranking and comparative quality, not only absolute scale.

## 2.5 Lifecycle state machine

Key lifecycle state:
- idle
- training

Train endpoints enforce lifecycle guards to avoid overlapping cycles and stale transitions.

## 3. API endpoints and contracts

## 3.1 POST /rollouts/ingest

Receives rollout batch from reward service.
Validates schema and stores buffer entries for later training.

## 3.2 POST /train/run-once

Runs a single training cycle.

Key semantics:
- can enforce min rollout requirement
- can return explicit no-op if insufficient data (depending on strictness)
- updates last-train success/error metadata

## 3.3 GET /status

Returns lifecycle and training health fields, including:
- lifecycle_state
- pending_rollouts
- last_train_success
- last_train_error
- last_train_started_at / last_train_finished_at

These fields are critical for orchestration scripts that poll until idle.

## 3.4 Adapter/model configuration endpoints

Service includes routes for adapter/model config refresh and reload behavior used during runtime operations.

## 4. Schemas explained

## 4.1 Rollout schema

Important fields typically include:
- note_id
- rewritten_note
- reward and reward_components
- log_prob_old
- value_estimate
- group_id
- generation_source metadata

This schema bridges inference-time metadata with RL-time optimization inputs.

## 4.2 Train request schema

Common controls:
- num_epochs / batch sizes
- min_rollouts thresholds
- distributed mode toggles
- strict/no-op behavior flags

## 4.3 Status response schema

Provides operational state for safe orchestration and monitoring.
The strict guard logic in orchestration depends on these fields being accurate.

## 5. Method-level walkthrough

## 5.1 API routing layer

File focus:
- app/api_routes.py

Responsibilities:
- request validation
- lifecycle guard checks
- delegation into training loop/lifecycle manager

## 5.2 Lifecycle manager

File focus:
- rl/lifecycle_manager.py

Responsibilities:
- acquire/release train lock semantics
- maintain lifecycle transitions
- track timestamps and last error state

## 5.3 Training loop

File focus:
- rl/training_loop.py

Responsibilities:
- load buffered rollouts
- compute advantages and normalized group signals
- call PPO trainer steps
- checkpoint/save and produce train summary

## 5.4 PPO trainer

File focus:
- rl/ppo_trainer.py

Responsibilities:
- ratio computation between new and old policy probabilities
- clipping objective
- policy/value loss composition
- KL-aware stabilization

## 5.5 Advantage and KL helpers

File focus:
- rl/advantage.py
- rl/kl_controller.py

Responsibilities:
- transform raw rewards into train-ready advantages
- adapt KL behavior by current divergence regime

## 5.6 Buffer and storage

File focus:
- rl/rollout_buffer.py
- storage/rollout_loader.py
- storage/checkpoint_manager.py

Responsibilities:
- ingest and persist rollouts
- load slices for train cycles
- maintain checkpoint continuity and recovery

## 5.7 Distributed single-cycle entrypoint

File focus:
- scripts/distributed_train_once.py

Responsibilities:
- initialize torch.distributed runtime
- run coordinated single training cycle
- collect rank-level diagnostics
- propagate clear failure context back to API caller

## 6. Orchestration behavior with full_dataset_train_eval

The orchestration script relies on strict RL semantics:
1. request train cycle
2. poll RL status until idle
3. fail hard if train endpoint/status indicates error

This prevents silent training skips and stale checkpoints from being misinterpreted as success.

## 7. Current limitations

- End-to-end learning quality remains sensitive to upstream reward signal quality.
- Distributed training increases operational complexity (environment, ranks, synchronization).
- Checkpoint and I/O overhead may limit cycle throughput.

## 8. Future improvements

1. Curriculum and phase-aware schedules for safer long runs.
2. Stronger automatic fallback/recovery playbooks for distributed partial failures.
3. More granular evaluation hooks per train cycle.
4. Better multi-run isolation and queue partitioning.
