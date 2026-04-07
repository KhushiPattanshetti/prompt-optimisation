# rewriter_inference_svc Detailed Report

## Navigation

- [Overview](00_overview.md)
- [Dataset Service](01_dataset_svc.md)
- [Rewriter Service](02_rewriter_inference_svc.md)
- [ICD10 Service](03_icd10_coding_svc.md)
- [Reward Service](04_reward_metrics_svc.md)
- [RL Loop Service](05_rl_loop_svc.md)
- [Theory Meta-Doc](07_theoretical_concepts_flow_methods_schemas.md)

## 1. What this service does

rewriter_inference_svc transforms a raw clinical note into a coding-oriented rewritten prompt.

It also returns two RL-critical metadata signals:
- log_prob_old: how probable the generated rewrite was under current policy
- value_estimate: scalar baseline estimate used for advantage computation

Beginner mental model:
- Input is noisy clinical narrative.
- Output is a structured prompt that makes ICD extraction easier.
- Service acts as the policy generator in the RL loop.

## 2. Core concepts

## 2.1 Prompt optimization as policy action

In RL framing:
- state s = original clinical note
- action a = rewritten prompt
- reward comes later from reward_metrics_svc

So this service is not just text rewriting; it is action generation for policy learning.

## 2.2 Multi-stage note preprocessing

Before model generation, the service builds a high-signal representation:
1. section extraction
2. clinical compression
3. coding-priority reordering
4. semantic descriptor extraction and rendering

This improves signal-to-noise before LLM generation.

## 2.3 Semantic descriptor extraction

The service extracts likely diagnosis descriptors using:
- section-priority cues
- context phrase patterns
- dictionary condition terms

Then applies strict normalization and medical-term filters.
If descriptor quality is too low, it returns empty descriptor list by design.

## 2.4 Controlled fallback strategy

If model generation fails or produces low-quality output, fallback strategy applies:
- cache rewrite
- first model sample
- second model sample
- base adapter disabled sample
- guided fallback
- rule fallback
- model-load-error fallbacks

The final source is surfaced through generation_source for observability and weighting downstream.

## 3. API endpoints

## 3.1 POST /rewrite_prompt

Consumes RewriteRequest and returns RewriteResponse.

## 3.2 POST /reload_checkpoint

Triggers model reload from latest RL checkpoint for closed-loop training refresh.

## 3.3 POST /update_best_prompt

Updates best prompt cache for note_id when reward is above threshold.

## 3.4 GET /health

Returns model readiness and configured model identity.

## 4. Schemas explained

All schemas are in rewriter_inference_svc/schemas.py.

## 4.1 RewriteRequest

Fields:
- note_id (optional): used for cache lookup and tracking
- clinical_note (required): raw text to rewrite

## 4.2 RewriteResponse

Fields:
- rewritten_prompt: final prompt selected by generation/fallback logic
- log_prob_old: sequence log probability for PPO behavior policy term
- value_estimate: value head prediction
- generation_source: provenance label (model/cache/fallback family)

Why generation_source matters:
- downstream reward service can adjust sample_weight for fallback-generated outputs

## 5. Method-level runtime flow

## 5.1 run_inference (main entry)

High-level sequence:
1. Build rule-based pipeline artifacts and semantic descriptors.
2. Choose model input mode/source.
3. Special-case healthcheck payload fast path.
4. Load model/tokenizer/value head.
5. Attempt generation candidates in priority order.
6. Validate candidate rewrite quality.
7. Compute log_prob_old and value_estimate.
8. Save output record.
9. Return RewriteResponse payload.

## 5.2 Descriptor pipeline methods

Key methods:
- extract_high_signal_sections
- compress_clinical_text
- reorder_for_coding_priority
- extract_semantic_diagnosis_descriptors
- build_optimized_prompt

Together they implement deterministic prompt shaping before LLM decoding.

## 5.3 Validation methods

Methods like _is_valid_rewrite and related helpers enforce:
- clinical relevance
- non-generic output
- descriptor/context alignment
- safe fallback when quality is low

## 5.4 Model loading and checkpoint compatibility

model_loader.py ensures:
- 4-bit quantized model load
- optional LoRA adapter load from latest checkpoint
- value head architecture compatibility checks
- migration guardrails for value_head checkpoints

## 6. Beginner example request/response

Request shape:
- note_id: optional identifier
- clinical_note: raw note text

Response shape:
- rewritten_prompt: coding-optimized prompt
- log_prob_old: float
- value_estimate: float
- generation_source: one of model/cache/fallback variants

## 7. How this service connects downstream

Output from rewriter goes directly to icd10_coding_svc /generate_codes.
The same output fields are forwarded:
- rewritten_prompt
- generation_source
- log_prob_old
- value_estimate

These are later consumed by reward and RL training paths.

## 8. Current limitations

- Quality still depends on note style diversity and token budget.
- Checkpoint compatibility remains strict by design and can fail fast.
- GPU pressure can affect model load and inference latency.

## 9. Future improvements

1. Adaptive rewrite depth based on note complexity estimates.
2. Clinical ontology-backed descriptor expansion and normalization.
3. Confidence scoring for rewrite quality to improve downstream sample weighting.
4. More explicit rejection reasons in response metadata for debugging and analytics.
