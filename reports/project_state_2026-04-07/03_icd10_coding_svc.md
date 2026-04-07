# icd10_coding_svc Detailed Report

## Navigation

- [Overview](00_overview.md)
- [Dataset Service](01_dataset_svc.md)
- [Rewriter Service](02_rewriter_inference_svc.md)
- [ICD10 Service](03_icd10_coding_svc.md)
- [Reward Service](04_reward_metrics_svc.md)
- [RL Loop Service](05_rl_loop_svc.md)
- [Theory Meta-Doc](07_theoretical_concepts_flow_methods_schemas.md)

## 1. What this service does

icd10_coding_svc converts prompts into ICD-10 code predictions and parse diagnostics.

It runs two branches per note:
- enhanced branch (rewritten prompt)
- original branch (baseline prompt)

Beginner mental model:
- This service is the evaluator of rewrite usefulness.
- If enhanced branch improves coding quality over original branch, reward can be positive.

## 2. Core concepts

## 2.1 Dual-branch inference

Why two branches:
- enhanced branch measures rewrite-driven coding output
- original branch gives baseline output for relative comparison

This is required for delta-style reward shaping.

## 2.2 Parse-first reliability

Raw model text is not trusted directly.
The parser extracts structured ICD labels using layered strategies.

## 2.3 Parse recovery

When parsing fails, recovery prompting can be attempted with tighter output instructions.
This increases recoverable parse rate on malformed generations.

## 2.4 GT code retrieval and canonicalization

Service retrieves GT codes from dataset_svc and caches per note_id.
Codes are canonicalized (dot normalization etc.) to reduce mismatch noise.

## 2.5 Observability taxonomy

Service records parse outcomes by branch and reason classes, enabling targeted diagnostics:
- enhanced_failed_only
- original_failed_only
- both_failed
- reason categories such as no_valid_icd_pattern

## 3. API endpoints

## 3.1 POST /generate_codes

Input: CodeRequest
Output: CodeResponse

Performs:
1. dual branch generation
2. parsing and optional parse recovery
3. GT retrieval
4. observability update
5. forwarding payload to reward service

## 3.2 GET /health

Returns service status, model identity, and weight-freeze status.

## 3.3 GET /observability

Returns parse success/failure metrics and taxonomy breakdown.

## 4. Schemas explained

All schemas are in icd10_coding_svc/schemas.py.

## 4.1 CodeRequest

Fields:
- note_id
- run_id optional
- group_id optional
- original_prompt
- rewritten_prompt
- generation_source optional
- log_prob_old optional
- value_estimate optional

Why run_id and group_id matter:
- run_id isolates audit/training runs
- group_id supports grouped reward normalization in RL

## 4.2 CodeResponse

Fields:
- note_id
- enh_codes
- org_codes
- gt_codes
- enh_raw_output
- org_raw_output
- parsing_success

Additional parse flags are produced by inference engine and forwarded to reward.

## 5. Method-level walkthrough

## 5.1 run_inference (inference_engine.py)

Main steps:
1. fetch GT codes via gt_fetcher
2. load frozen Med42 model/tokenizer
3. run rewritten prompt pass
4. run original prompt pass
5. parse both outputs
6. run parse recovery if enabled and needed
7. compute parse flags and observability updates
8. save per-note output artifact
9. forward full payload to reward service

## 5.2 parse_icd10_codes (code_parser.py)

Strategy cascade:
1. JSON parse attempt (full output)
2. JSON array extraction inside noisy text
3. Regex extraction
4. Empty fallback with warning

Normalization and deduplication are applied before return.

## 5.3 get_gt_codes (gt_fetcher.py)

Behavior:
- read local cache if present
- else call dataset_svc /gt_codes/{note_id}
- canonicalize and cache result

## 5.4 model_loader.py

Load behavior:
- 4-bit NF4 quantized Med42 load
- cache local model assets
- freeze all model weights
- reuse cached model in-process

## 6. Beginner interpretation of comparability flags

- enh_parse_ok: enhanced branch parsed at least one valid code
- org_parse_ok: original branch parsed at least one valid code
- both_parse_success: both are parse-valid, best case for fair relative reward
- parsing_success: at least one branch parsed successfully

These flags are critical in reward weighting and run guards.

## 7. How this service connects downstream

After inference, this service posts to reward_metrics_svc /reward with:
- GT, enhanced, original codes
- parse status flags
- prompt and policy metadata fields

That payload becomes reward computation input and potentially RL rollout input.

## 8. Current limitations

- Hard notes can still produce parse failures after recovery.
- Output quality depends strongly on prompt quality from rewriter.
- Regex fallback can recover syntax but not semantic correctness.

## 9. Future improvements

1. Add constrained decoding to reduce malformed outputs at generation time.
2. Add confidence estimates (parse confidence and semantic confidence).
3. Expand observability with note-type or specialty-level failure slices.
4. Add optional consensus decoding across multiple samples for robustness.
