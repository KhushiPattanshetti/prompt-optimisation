# Streaming Refactor Plan (Phased)

Date: 2026-04-17

## Goal
Move pipeline execution toward note-wise streaming for rewriter + ICD + reward while preserving RL rollout validity and trajectory-store correctness.

## Constraints
- Do not break group-level rollout validity checks.
- Keep existing batched-overlap path as default for safety.
- Keep trajectory submission to RL store batchwise.

## Phase 0 (Baseline retained)
- Existing mode remains default.
- Existing overlapped batch execution is unchanged unless flags are enabled.

## Phase 1 (Implemented)
- New flag: `--stream-note-wise`.
- Within each fetched dataset batch, process each note end-to-end:
  - Rewriter -> ICD -> reward/train-group handling -> persist note record.
- This removes the per-batch rewriter barrier for downstream stages.

## Phase 2 (Implemented)
- New flag: `--trajectory-flush-size`.
- Valid train rollouts are buffered in-memory and flushed to RL trajectory store batchwise.
- Group validity checks are still performed per note-group before buffering.
- Flush occurs when buffer reaches threshold and once at final drain.

## Rollout Validity Guarantees (preserved)
A train group is accepted for buffering/submission only if:
- all rollouts share the same state
- all rollouts share one group_id
- at least 2 unique actions exist
- at least 2 rollouts survive filtering

## Observability
- Existing per-service IO logs remain in place.
- Added flush log line: `trajectory_buffer_flushed requested=... accepted=... duplicate=...`.

## Iterations Per Batch
Current pipeline summary logging still uses:
- `iter_num=1`
- `max_iters=1`
per emitted pipeline batch summary.

## Next Phase (not yet implemented)
- Replace in-process trajectory buffer with durable queue-backed buffering.
- Add adaptive backpressure from rewriter latency and RL backlog.
- Add explicit note-level retry budget and dead-letter handling.
