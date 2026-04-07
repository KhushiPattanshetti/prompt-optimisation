# dataset_svc Report

## Service role

dataset_svc is the ground-truth note provider and batch source for the rest of the pipeline.

## Major changes since clone

- Service scaffolding, schemas, store layer, and health/batch APIs were added.
- Unit, integration, and e2e tests were introduced.
- Data plumbing was aligned to the full-dataset orchestration flow.

Representative files:
- dataset_svc/app.py
- dataset_svc/store.py
- dataset_svc/schemas.py
- dataset_svc/tests/

## Current strengths

- Stable health endpoint and batch retrieval.
- Works with deterministic split logic in orchestration script.
- Test coverage exists for schema/store and service-path behavior.

## Current limitations

- Data dependency management is external (requires local dataset provisioning).
- Service currently assumes local file availability and expected formatting.
- No advanced dataset versioning contract is enforced at runtime.

## Future prospects

- Add dataset manifest/version endpoint for reproducibility checks.
- Add startup validation for schema drift and missing fields.
- Add optional lightweight cache/index for faster large-batch scans.
