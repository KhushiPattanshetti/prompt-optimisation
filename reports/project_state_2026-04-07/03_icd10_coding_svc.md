# icd10_coding_svc Report

## Service role

icd10_coding_svc converts prompts into ICD code candidates and parse diagnostics.

## Major changes since clone

- ICD generation microservice, parser, model loader, and schemas were added.
- Parser/canonicalization robustness improvements were applied.
- Observability fields for parse outcomes and comparability were expanded.
- Tests were extended across unit/integration/e2e layers.

Representative files:
- icd10_coding_svc/inference_engine.py
- icd10_coding_svc/code_parser.py
- icd10_coding_svc/gt_fetcher.py
- icd10_coding_svc/tests/

## Current strengths

- Better normalization for code forms and parser resilience.
- Supplies parse success indicators consumed by reward and pipeline guards.
- Works consistently with split-based train/val/test orchestration.

## Current limitations

- Parse failures still occur for difficult outputs and malformed generations.
- Service behavior can degrade if prompt quality drops upstream.
- Canonical mapping quality is tied to maintained code-source integrity.

## Future prospects

- Add constrained decoding or post-parse repair for invalid code patterns.
- Add stronger confidence outputs for downstream weighting.
- Add detailed parser failure class telemetry per note family.
