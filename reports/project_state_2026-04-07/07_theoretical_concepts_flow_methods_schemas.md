# Theoretical Concepts, Flow, Methods, and Schemas

## Navigation

- [Overview](00_overview.md)
- [Dataset Service](01_dataset_svc.md)
- [Rewriter Service](02_rewriter_inference_svc.md)
- [ICD10 Service](03_icd10_coding_svc.md)
- [Reward Service](04_reward_metrics_svc.md)
- [RL Loop Service](05_rl_loop_svc.md)
- [Theory Meta-Doc](07_theoretical_concepts_flow_methods_schemas.md)

Date: 2026-04-07
Scope: Integrated pipeline implemented in dataset_svc, rewriter_inference_svc, icd10_coding_svc, reward_metrics_svc, rl_loop_svc, and scripts/full_dataset_train_eval.py

## 1. End-to-end pipeline flow

The implemented runtime flow is a staged decision pipeline with asynchronous training feedback:

1. dataset_svc provides clinical note batches and GT ICD labels.
2. rewriter_inference_svc rewrites each note into a coding-optimized prompt and emits policy metadata.
3. icd10_coding_svc runs two inference branches:
   - enhanced branch using rewritten prompt
   - original branch using baseline prompt
4. reward_metrics_svc compares enhanced vs original vs GT codes and computes a scalar reward and component breakdown.
5. reward service forwards rollout payloads to rl_loop_svc through a durable queue.
6. rl_loop_svc trains policy using PPO/GRPO style updates, checkpoints, and serves status.
7. Orchestration script triggers periodic train cycles and enforces hard run guards.

### Operational flow in orchestration script

In scripts/full_dataset_train_eval.py:
- Notes are split into train/val/test using deterministic stratified logic.
- Train split:
  - sends full rollout metadata and contributes to RL updates.
- Val/test split:
  - computes reward only, without training payload fields, preventing leakage.
- For every train interval and at final flush:
  - queue is drained,
  - RL train cycle is triggered,
  - status is polled until IDLE,
  - cycle is marked failed if:
    - training endpoint was not triggered,
    - cycle reported failure,
    - training_step did not advance.

This converts previously silent RL failures into explicit pipeline guard failures.

## 2. Theoretical concepts implemented

## 2.1 Multi-objective reward shaping

Reward is a weighted blend of five components:
- exact reward
- semantic reward
- concept reward
- structure reward
- delta reward

Conceptually:

R_total = clamp(
  w_exact * R_exact +
  w_sem * R_sem +
  w_concept * R_concept +
  w_struct * R_struct +
  w_delta * R_delta + bonus
)

Why this matters:
- Exact matching alone is sparse and brittle.
- Semantic and concept channels densify learning signal.
- Structure and delta channels reduce degenerate or non-improving behavior.

## 2.2 Concept precision/recall/F1 shaping

For concept descriptors extracted from rewritten context:
- precision = overlap / predicted_codes
- recall = overlap / descriptors
- F1 = 2PR / (P + R)

Then concept reward is mapped to [-1, 1]:
- R_concept = clamp(2 * F1 - 1)

Additional implemented behavior:
- If no predicted codes, concept branch returns strong negative.
- If descriptors are missing, concept branch returns soft negative, not neutral.

## 2.3 PPO objective with clipping

In PPO mode, policy objective follows clipped surrogate optimization:

r_t(theta) = exp(logpi_new - logpi_old)
L_clip = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)

The trainer also includes:
- value loss with value clipping
- entropy bonus
- KL penalty against reference model

## 2.4 GRPO-style grouped relative rewards

In grouped mode, rewards are standardized within group_id cohorts:

A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

Design effect:
- Converts absolute reward scale into relative ranking within comparable prompts.
- Reduces inter-note scale variance.

Fallback behavior is implemented when group quality is poor (small/non-comparable groups).

## 2.5 KL regularization

Policy drift is controlled by positive KL penalty:

KL_i = max(logpi_policy - logpi_ref, 0)
R_adjusted = R - beta * KL

Implemented in KLController and merged into training loss.

## 2.6 GAE for variance-bias tradeoff

Generalized Advantage Estimation is implemented as:

delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}

Then normalized and clipped for stability.

## 2.7 Reliability engineering in distributed RL

The implementation now treats distributed failures as first-class signals:
- torchrun per-rank log redirection enabled
- torchelastic rank error recording enabled
- worker error files copied into persistent failure artifacts
- pipeline guard fails run if train cycle fails or no step progress

This changed failure mode from hidden degradation to explicit fault visibility.

## 3. Schema contracts and method working by service

## 3.1 dataset_svc

Primary schemas:
- NoteResponse: note_id, text
- GTCodesResponse: note_id, gt_codes
- BatchRecord: note_id, text, gt_codes
- BatchResponse: batch, offset, size, total
- NoteIdsResponse: note_ids, offset, size, total
- HealthResponse: status, total_notes, total_coded_notes, loading_time_sec

Key runtime behavior:
- Lifespan validates source files and loads indexed store.
- Batch endpoint supports bounded paging and total count.
- Health endpoint reflects loading and runtime dataset stats.

## 3.2 rewriter_inference_svc

Primary schemas:
- RewriteRequest:
  - note_id optional (cache identity)
  - clinical_note required
- RewriteResponse:
  - rewritten_prompt
  - log_prob_old
  - value_estimate
  - generation_source

Key methods:
- run_inference: executes rewrite generation and fallback logic.
- extract_semantic_diagnosis_descriptors: multi-stage descriptor extraction:
  - high-priority section terms
  - contextual noun phrases
  - dictionary condition terms
  - strict cleaning and medical-term filtering

Design intention:
- Provide both optimized prompt text and RL metadata for downstream learning.

## 3.3 icd10_coding_svc

Primary schemas:
- CodeRequest:
  - note_id, run_id, group_id
  - original_prompt, rewritten_prompt
  - generation_source, log_prob_old, value_estimate
- CodeResponse:
  - enhanced/original code outputs
  - raw outputs
  - parsing_success flag

Key methods:
- run_inference:
  - executes enhanced and original passes
  - parses ICD codes
  - applies parse-recovery fallback when enabled
  - records observability taxonomy
  - forwards payload to reward service

Observability model:
- tracks enhanced-only failures, original-only failures, both failures, and reason categories.

## 3.4 reward_metrics_svc

Primary schemas:
- RewardRequest:
  - GT source (gt_codes or gt_file)
  - enhanced and original codes
  - parse comparability flags
  - prompt fields for rollout forwarding
  - RL metadata fields (group_id, generation_source, log_prob_old, value_estimate, sample_weight)
- QueueStatusResponse: pending_count, acked_count, failed_count
- QueueFlushResponse: posted_batches, posted_rollouts, duplicate_count, failed_batches, pending_count

Key methods:
- calculate_reward_components: computes exact/semantic/concept/structure/delta and bonuses.
- reward_endpoint:
  - resolves GT,
  - computes reward,
  - records observability,
  - enqueues/flushes rollout payloads to RL service.

Queue design:
- pending/acked/failed directories,
- retry and failover behavior,
- transport degradation detection.

## 3.5 rl_loop_svc

Primary API schemas:
- StatusResponse:
  - trainer_state, rollouts_loaded, training_step, last_loss, kl_divergence
  - last_train_success, last_train_error, last_train_started_at, last_train_finished_at
- TrainResponse: triggered, message
- CheckpointResponse: available, metadata
- RolloutSubmission and RolloutBatchSubmission
- RolloutAck: accepted/file_path/accepted_count/duplicate_count/run_id

Primary rollout schemas:
- RolloutEntry:
  - identity: rollout_id, run_id, group_id
  - state/action text: original_prompt, rewritten_prompt
  - learning targets: reward, sample_weight
  - behavior/value: log_prob_old, value_estimate
- RolloutFile: run_id + rollouts[]

Key methods:
- TrainingLoop.run_once:
  - COLLECT new rollouts,
  - TRAIN (distributed or local),
  - CHECKPOINT,
  - notify rewriter reload,
  - return IDLE.
- TrainingLoop._run_distributed_ppo:
  - validates memory headroom,
  - launches torch.distributed.run,
  - captures per-rank failure artifacts on non-zero exit,
  - returns training_step/loss/kl on success.
- api_routes._background_train:
  - writes train start/end timestamps,
  - marks success/failure and error text for status consumers.

## 3.6 Orchestration and guards

In full_dataset_train_eval.py, run_train_cycle now enforces strict semantics:
- train trigger must be accepted,
- cycle must complete with updated finish timestamp,
- status must report success,
- training_step must advance.

If any condition fails:
- cycle failure is recorded,
- summary includes train_cycle_failure_count and failure payload,
- run_guard_failures includes train_cycle_failures_detected,
- process exits non-zero.

## 4. Current limitations (theory-to-implementation gap)

- Reward alignment channels still show weak concept precision/recall in observed probes.
- Distributed RL currently fails hard on OOM instead of adaptive retry.
- Non-fault ranks surface barrier errors secondary to primary fault rank.
- End-to-end reward variance can remain low in small-sample runs.

## 5. Future prospects and research-grade upgrades

1. OOM-aware adaptive distributed scheduler:
- auto-reduce world size or effective batch and retry once.

2. Curriculum-aware dynamic reward weights:
- early exploration-heavy shaping,
- later exactness/concept emphasis.

3. Confidence-aware sample weighting:
- down-weight uncertain parse or fallback-generated samples.

4. Stronger schema evolution contracts:
- versioned request/response contracts with compatibility checks.

5. Better run observability payloads:
- write machine-readable guard diagnostics per run for dashboards.

## 6. Practical interpretation

The implementation now combines:
- dense reward shaping,
- policy-regularized optimization,
- grouped relative learning,
- durable queue transport,
- strict train-cycle guardrails,
- per-rank distributed failure diagnostics.

This is a robust foundation for iterative RL-based prompt optimization, with the main blocker now being memory-adaptive distributed scheduling rather than missing observability or silent failures.
