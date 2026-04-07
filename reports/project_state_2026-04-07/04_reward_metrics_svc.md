# reward_metrics_svc Detailed Report

## Navigation

- [Overview](00_overview.md)
- [Dataset Service](01_dataset_svc.md)
- [Rewriter Service](02_rewriter_inference_svc.md)
- [ICD10 Service](03_icd10_coding_svc.md)
- [Reward Service](04_reward_metrics_svc.md)
- [RL Loop Service](05_rl_loop_svc.md)
- [Theory Meta-Doc](07_theoretical_concepts_flow_methods_schemas.md)

## 1. What this service does

reward_metrics_svc is the scoring and rollout transport service.

It computes:
- scalar total reward
- component rewards and diagnostics

It also handles:
- rollout queueing and retries
- forwarding to RL service
- reward and transport observability

Beginner mental model:
- This service is the teacher signal generator for RL.
- It translates prediction quality into numeric learning feedback.

## 2. Core concepts

## 2.1 Multi-objective reward shaping

Reward is not a single metric. It blends multiple objectives:
- exact match behavior
- semantic closeness
- concept alignment with descriptors
- structural parse validity
- relative improvement over baseline

Conceptual formula:

R_total = clamp(
	w_exact * R_exact +
	w_semantic * R_semantic +
	w_concept * R_concept +
	w_structure * R_structure +
	w_delta * R_delta + bonuses
)

## 2.2 Why component rewards are needed

- Exact reward alone is sparse and brittle.
- Semantic reward captures approximate correctness.
- Concept reward links rewritten clinical intent to code descriptions.
- Structure reward discourages parse-invalid behavior.
- Delta reward captures improvement versus original prompt branch.

## 2.3 Concept precision/recall/F1 branch

Concept branch compares predicted code descriptions to semantic descriptors.

Derived metrics:
- concept_precision
- concept_recall
- concept_f1

Then mapped to reward range [-1, 1].

Implemented edge behavior:
- no predicted codes: strong negative signal
- missing descriptor guidance: soft negative signal

## 2.4 Semantic similarity branch

Uses text embeddings and similarity matching to produce a semantic reward term.
This provides denser signal than strict exact code overlap.

## 2.5 Durable rollout queue transport

Forwarding to RL is decoupled through pending/acked/failed queue directories.

Benefits:
- retry support
- reduced data loss risk
- observability of transport health

## 3. API endpoints

## 3.1 POST /reward

Main scoring endpoint.

Flow:
1. resolve GT source (gt_codes or gt_file)
2. compute reward components
3. record reward observability
4. enqueue rollout if rollout metadata exists
5. flush queue and optionally fail fast if transport degraded
6. return scalar reward and component breakdown

## 3.2 GET /queue/status

Returns pending, acked, failed queue counts.

## 3.3 POST /queue/flush

Attempts posting pending batches to RL endpoint and updates ack/failure state.

## 3.4 GET /observability

Returns transport and reward distribution metrics.

## 3.5 POST /observability/reset

Resets cumulative counters for clean run evaluation.

## 3.6 GET /health

Reports tree integrity, queue health, and transport degradation state.

## 4. Schemas explained

## 4.1 RewardRequest

Important fields:
- gt_codes or gt_file
- enh_codes
- org_codes
- parse comparability flags
- prompt fields for rollout forwarding
- generation_source
- sample_weight
- log_prob_old
- value_estimate
- group_id and run_id

Design detail:
- If rollout-specific fields are absent, request is treated as eval-only and not forwarded to RL.

## 4.2 QueueStatusResponse

Fields:
- pending_count
- acked_count
- failed_count

## 4.3 QueueFlushResponse

Fields:
- posted_batches
- posted_rollouts
- duplicate_count
- failed_batches
- pending_count

## 5. Method-level walkthrough

## 5.1 calculate_reward_components

Inputs:
- GT codes
- enhanced/original predicted codes
- optional semantic descriptors and parse flags

Outputs:
- exact_reward
- semantic_reward and semantic_score
- concept_reward, precision, recall, f1
- structure_reward
- delta_reward
- bonuses/penalty
- total_reward

## 5.2 reward_endpoint

Coordinates the full path from request to response:
1. validate GT source
2. compute components
3. round reward
4. update metrics
5. forward/queue rollout as needed

## 5.3 _enqueue_rollout and flush_rollout_queue

Queue logic:
- build deterministic rollout_id
- write pending envelope
- batch post to RL endpoint
- move to acked or failed with retry budget

## 5.4 Transport degradation guard

Transport health uses:
- consecutive failure streak
- drop rate
- contract mismatch signal

If degraded and fail-fast is enabled, endpoint can reject training acknowledgements.

## 5.5 Description and embedding helpers

Modules:
- icd_descriptions.py: loads/normalizes ICD description map
- embedding_model.py: embedding generation and similarity helpers

These modules support semantic and concept reward branches.

## 6. How this service connects upstream/downstream

Upstream dependencies:
- GT and predictions from ICD service
- rewrite descriptors/metadata indirectly through payload fields

Downstream dependencies:
- RL rollout submission endpoint
- RL status endpoint for curriculum state caching

## 7. Current limitations

- Concept branch quality still depends on descriptor extraction quality upstream.
- Reward variance can be low in small run windows.
- Embedding computation adds latency and dependency complexity.

## 8. Future improvements

1. Dynamic component weighting by confidence and training phase.
2. Stronger descriptor-to-ontology alignment for concept reward quality.
3. Better run-profile presets for faster calibration and diagnostics.
4. Automatic alerting on prolonged transport degradation.
