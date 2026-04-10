# Testing Summary — rewriter_sft_svc

> Generated: 2026-03-15 13:55:44
> Overall Status: **PASS**

---

## Overview

| Metric | Count |
|--------|-------|
| Total test categories | 8 |
| Assertions passed | 83 |
| Assertions failed | 0 |
| Errors | 0 |

---

## Unit Tests

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 10  |  **Failed:** 0

### Passed
- DATASET_KEYS contains required keys
- SYSTEM_INSTRUCTION is non-empty string
- REQUIRED_FIELDS contains 12 fields in correct order
- record_to_messages produces correct 3-turn message structure
- convert_dataset processes list correctly
- validate_schema correctly filters malformed records
- validate_output_schema detects missing fields
- validate_output_schema detects leaked forbidden phrases
- split_dataset maintains total count and creates all splits
- CHECKPOINT_DIR configured correctly

---

## Test-Data Validation

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 16  |  **Failed:** 0

### Passed
- sample_test.json: file exists
- sample_test.json: valid JSON (5 records)
- sample_test.json: all 5 records pass schema validation
- sample_test.json: non-empty (5 records)
- edge_cases.json: file exists
- edge_cases.json: valid JSON (8 records)
- edge_cases.json: all 8 records pass schema validation
- edge_cases.json: non-empty (8 records)
- stress_cases.json: file exists
- stress_cases.json: valid JSON (3 records)
- stress_cases.json: all 3 records pass schema validation
- stress_cases.json: non-empty (3 records)
- e2e_cases.json: file exists
- e2e_cases.json: valid JSON (3 records)
- e2e_cases.json: all 3 records pass schema validation
- e2e_cases.json: non-empty (3 records)

---

## Inference Schema Tests

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 19  |  **Failed:** 0

### Passed
- sample_test.json[0]: expected output passes schema
- sample_test.json[1]: expected output passes schema
- sample_test.json[2]: expected output passes schema
- sample_test.json[3]: expected output passes schema
- sample_test.json[4]: expected output passes schema
- edge_cases.json[0]: expected output passes schema
- edge_cases.json[1]: expected output passes schema
- edge_cases.json[2]: expected output passes schema
- edge_cases.json[3]: expected output passes schema
- edge_cases.json[4]: expected output passes schema
- edge_cases.json[5]: expected output passes schema
- edge_cases.json[6]: expected output passes schema
- edge_cases.json[7]: expected output passes schema
- stress_cases.json[0]: expected output passes schema
- stress_cases.json[1]: expected output passes schema
- stress_cases.json[2]: expected output passes schema
- e2e_cases.json[0]: expected output passes schema
- e2e_cases.json[1]: expected output passes schema
- e2e_cases.json[2]: expected output passes schema

---

## Edge Case Tests

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 12  |  **Failed:** 0

### Passed
- Loaded 8 edge case records
- Edge case 0: expected output schema valid
- Edge case 0 (minimal note): ≥6 'Not specified' fields — OK
- Edge case 1: expected output schema valid
- Edge case 1 (minimal note): ≥6 'Not specified' fields — OK
- Edge case 2: expected output schema valid
- Edge case 2 (minimal note): ≥6 'Not specified' fields — OK
- Edge case 3: expected output schema valid
- Edge case 4: expected output schema valid
- Edge case 5: expected output schema valid
- Edge case 6: expected output schema valid
- Edge case 7: expected output schema valid

---

## Stress Tests

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 8  |  **Failed:** 0

### Passed
- Loaded 3 stress records
- Stress set: convert_dataset stable on all records
- Stress case 0: expected output schema valid
- Stress case 1: expected output schema valid
- Stress case 2: expected output schema valid
- Stress case 0: message construction OK (note len=610)
- Stress case 1: message construction OK (note len=344)
- Stress case 2: message construction OK (note len=400)

---

## End-to-End Tests

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 6  |  **Failed:** 0

### Passed
- Loaded 3 e2e records
- E2E: all records pass schema validation
- E2E: 3 records converted to chat format
- E2E: system instruction correctly embedded in all samples
- E2E: checkpoint directory exists and contains 11 files
- E2E: dataset split integrity OK (train=3, val=0, test=0)

---

## Regression Tests

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 5  |  **Failed:** 0

### Passed
- Regression: REQUIRED_FIELDS count == 12
- Regression: REQUIRED_FIELDS names and order unchanged
- Regression: ROOT_DIR and SVC_DIR exist
- Regression: SYSTEM_INSTRUCTION length consistent (>100 chars)
- Regression: validate_output_schema accepts well-formed output

---

## Checkpoint Validation

**Status:** ✅ PASS  |  **Duration:** 0.0s  |  **Passed:** 7  |  **Failed:** 0

### Passed
- Checkpoint directory exists: /Users/ishanmane/prompt-optimisation/sft_checkpoints/phi3_rewriter_sft
- Checkpoint artifact found: adapter_config.json
- Tokenizer artifact found: tokenizer_config.json
- Tokenizer artifact found: tokenizer.json
- Tokenizer artifact found: special_tokens_map.json
- LoRA adapter weights found (bin or safetensors)
- config.json found in checkpoint

---

## Checkpoint Validation

Checkpoint directory: `/Users/ishanmane/prompt-optimisation/sft_checkpoints/phi3_rewriter_sft`

Existence check performed as part of 'Checkpoint Validation' category above.

---

## Known Issues

- Full model inference tests require a trained checkpoint to be present.
- Training on CPU/MPS is significantly slower than on CUDA; use GPU for production runs.
- BitsAndBytes 4-bit quantisation is only available on CUDA; MPS/CPU use float16/float32.

---

## Final Status

**PASS** — 83 assertions passed, 0 failed, 0 errors.

---

_Report auto-generated by `test_runner.py`_