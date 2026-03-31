# End-to-End Pipeline Statistical Report (20 notes)

**Run window (artifact timestamps):** 2026-03-26 08:42:41 → 09:09:40

This report is computed from persisted run artifacts written by services during execution:
- Rewriter outputs: `rewriter_inference_repo/inference_outputs/2026-03-26T*.json` (n=20)
- ICD-10 coding outputs: `icd10_coding_repo/inference_outputs/icd10/*_2026-03-26T*.json` (n=20)
- RL rollouts recorded: `rl_loop_repo/rl_loop_svc/rollouts/rollout_batch_20260326T*.json` (n=20)

## 1) Run Integrity & Coverage

- Notes processed end-to-end (paired across rewriter → icd10 → rollout): **20 / 20 (100%)**
- Artifact pairing method:
  - For each ICD10 output, the nearest *preceding* rewriter output was selected.
  - For each ICD10 output, the nearest rollout batch file was selected (microseconds in filename mean exact second-level match is not possible).
- Rewriter→ICD10 handoff delay (from artifact timestamps):
  - min: **32s**
  - median: **34s**
  - max: **62s**

## 2) Service-by-Service Statistical Summary

### 2.1 Rewriter Inference Service (`rewriter_inference_svc`)
Computed from `rewriter_inference_repo/inference_outputs/2026-03-26T*.json`.

**Input / output size**
- Clinical note length (characters):
  - min: **4,639**
  - median: **10,412.5**
  - max: **16,489**
- Rewritten prompt length (characters):
  - min: **419**
  - median: **826**
  - max: **1,499**

**Model-side scalars (as stored)**
- `log_prob_old`:
  - mean: **-198.92**
  - median: **-215.16**
  - n: **20**
- `value_estimate`:
  - mean: **-0.3854**
  - median: **-0.3799**
  - n: **20**

**Observations (implementation quality signals)**
- The rewritten prompts are dramatically shorter than the clinical notes (median 826 chars vs median 10.4k chars), suggesting the “rewrite” is likely collapsing content rather than improving a prompt while preserving salient information.

### 2.2 ICD-10 Coding Service (`icd10_coding_svc`)
Computed from `icd10_coding_repo/inference_outputs/icd10/*_2026-03-26T*.json`.

**Parsing success**
- `parsing_success` rate: **55%** (11/20)

**Code yield**
- `org_codes` count per note:
  - min / median / max: **0 / 0 / 11**
- `enh_codes` count per note:
  - min / median / max: **0 / 0 / 7**

**Observations (implementation quality signals)**
- Median code count is **0** for both original and enhanced code lists, which usually implies a systemic extraction/parsing issue (prompt formatting, model output format drift, brittle parser, or non-robust retry/error handling).

### 2.3 Reward / Rollout Recording (`reward_metrics_svc` → `rl_loop_svc`)
Computed from `rl_loop_repo/rl_loop_svc/rollouts/rollout_batch_20260326T*.json`.

**Rollout persistence**
- Rollout batches: **20**
- Rollouts recorded: **20**
- Rollouts per batch: **1** consistently

**Reward distribution**
- Reward values observed: **all -1.0**
  - min / mean / median / max: **-1.0 / -1.0 / -1.0 / -1.0**
  - counts: `{-1.0: 20}`

**Reward vs ICD10 parsing_success**
- Mean reward when `parsing_success=true`: **-1.0** (n=11)
- Mean reward when `parsing_success=false`: **-1.0** (n=9)

**Observations (implementation quality signals)**
- The reward signal is completely saturated at the minimum value for every note. This prevents learning (no gradient signal diversity) and indicates one of:
  - Reward function/normalization bug (always returns -1),
  - Missing/empty code sets causing a hard failure path,
  - ICD-10 hierarchy file is too small/toy to score realistically,
  - Request payload mismatch between services (e.g., fields not forwarded as expected).

### 2.4 Dataset Service (`dataset_svc`)
No persisted per-run artifacts were found under `dataset_svc_repo/` for this run window.

What can be inferred indirectly:
- The note IDs appearing in ICD10 outputs indicate dataset yielded 20 records successfully.

### 2.5 RL Training Cycles / Checkpoints (`rl_loop_svc`)
This workspace contains rollout files but no explicit per-cycle training metrics or checkpoint metadata tied to this specific 20-note run were found in a persisted, timestamped form.

Artifacts present:
- Rollouts: `rl_loop_repo/rl_loop_svc/rollouts/rollout_batch_*.json`
- Code for checkpoint management: `rl_loop_repo/rl_loop_svc/storage/checkpoint_manager.py`

If you want checkpoint-level statistics (loss curves, KL, entropy, advantage stats, etc.), the RL service needs to persist structured training logs per train invocation.

## 3) Cross-Step / End-to-End Statistics

### 3.1 End-to-end success
- Notes with ICD10 parsing success: **11/20 (55%)**
- Notes with any `enh_codes` produced (>0): **(see note-level table)**

### 3.2 Latency proxy (artifact timestamps)
- Rewriter→ICD10 delay (sec): min **32**, median **34**, max **62**

This is not pure model latency; it includes service queueing/overhead.

## 4) Note-Level Breakdown (20 notes)

Columns:
- `note_id`
- `rewrite_to_icd10_sec` (timestamp gap)
- `parsing_success`
- `org_codes_n`, `enh_codes_n`
- `reward`

| note_id | rewrite_to_icd10_sec | parsing_success | org_codes_n | enh_codes_n | reward |
|---|---:|---:|---:|---:|---:|
| 10000084-DS-17 | 33 | false | 0 | 0 | -1.0 |
| 10000117-DS-21 | 34 | true | 0 | 1 | -1.0 |
| 10000117-DS-22 | 33 | true | 11 | 0 | -1.0 |
| 10000980-DS-26 | 36 | false | 0 | 0 | -1.0 |
| 10000980-DS-24 | 46 | true | 5 | 4 | -1.0 |
| 10000980-DS-25 | 32 | false | 0 | 0 | -1.0 |
| 10001401-DS-17 | 40 | true | 1 | 0 | -1.0 |
| 10001401-DS-19 | 33 | true | 0 | 2 | -1.0 |
| 10001401-DS-18 | 33 | true | 0 | 2 | -1.0 |
| 10001401-DS-22 | 35 | true | 0 | 3 | -1.0 |
| 10001401-DS-20 | 37 | true | 0 | 7 | -1.0 |
| 10001401-DS-21 | 32 | false | 0 | 0 | -1.0 |
| 10001667-DS-10 | 62 | false | 0 | 0 | -1.0 |
| 10001884-DS-37 | 33 | false | 0 | 0 | -1.0 |
| 10001884-DS-35 | 38 | false | 0 | 0 | -1.0 |
| 10001884-DS-38 | 36 | true | 0 | 2 | -1.0 |
| 10001884-DS-31 | 34 | false | 0 | 0 | -1.0 |
| 10001884-DS-36 | 33 | true | 0 | 3 | -1.0 |
| 10001884-DS-33 | 32 | false | 0 | 0 | -1.0 |
| 10001884-DS-34 | 34 | true | 0 | 4 | -1.0 |
