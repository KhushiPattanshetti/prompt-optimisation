# reward_metrics_svc Report

## Service role

reward_metrics_svc computes reward signals for evaluation and RL training, and forwards rollout payloads.

## Major changes since clone

- Reward microservice was introduced with structured reward computation.
- Reward design evolved into a multi-component blend (exact/semantic/concept/structure/delta).
- Queue/observability and rollout forwarding behavior were expanded.
- Concept reward fallback and schema alignment fixes were applied in latest iteration.

Representative files:
- reward_metrics_svc/main.py
- reward_metrics_svc/embedding_model.py
- reward_metrics_svc/icd_descriptions.py
- reward_metrics_svc/test/test_concept_reward.py

## Current strengths

- Rich observability for reward distribution and forwarding outcomes.
- Supports eval-only reward calls and rollout-forwarding train calls.
- Better handling of missing descriptor context with non-neutral penalties.

## Current limitations

- Recent probe metrics show low concept precision/recall and weak semantic-concept alignment.
- Reward variance can remain below desired exploration thresholds on small batches.
- Embedding/model dependencies can increase startup/runtime overhead.

## Future prospects

- Improve concept extraction and ontology alignment.
- Add adaptive reward weighting by confidence/comparability.
- Add run-profile-specific reward diagnostics and threshold gates.
