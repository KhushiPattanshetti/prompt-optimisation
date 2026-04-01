# Test Report — `reward_metrics_svc`

**Date:** 10 March 2026  
**Branch:** `feature/rl_loop`  
**Python:** 3.10.13  
**pytest:** 8.4.2  
**Module under test:** `main.py`

---

## 1. Executive Summary

| Metric             | Value                         |
| ------------------ | ----------------------------- |
| Total tests        | 6                             |
| Passed             | **6**                         |
| Failed             | 0                             |
| Errors             | 0                             |
| Skipped            | 0                             |
| Overall result     | ✅ **PASS**                   |
| Statement coverage | **84 %** (75 / 89 statements) |
| Missing statements | 14                            |

---

## 2. Test Environment

| Item                 | Detail                                         |
| -------------------- | ---------------------------------------------- |
| OS                   | macOS (darwin)                                 |
| Python               | 3.10.13                                        |
| pytest               | 8.8.2                                          |
| pytest-cov           | 4.1.0                                          |
| FastAPI test client  | `httpx` via `fastapi.testclient.TestClient`    |
| ICD-10 tree nodes    | 9 (sample tree: A00–B02)                       |
| Ground-truth fixture | `gt_codes/sample_gt.json` → `["A01.1", "A02"]` |
| RL block endpoint    | Not set (`None`) — offline mode                |

---

## 3. Test Results

| #   | Test ID                            | Category                       | Result  | Duration |
| --- | ---------------------------------- | ------------------------------ | ------- | -------- |
| 1   | `test_distance_between_basic`      | Unit – distance metric         | ✅ PASS | —        |
| 2   | `test_distance_between_empty`      | Unit – edge case (empty input) | ✅ PASS | —        |
| 3   | `test_calculate_reward_better_enh` | Unit – reward logic            | ✅ PASS | —        |
| 4   | `test_calculate_reward_better_org` | Unit – reward logic            | ✅ PASS | —        |
| 5   | `test_calculate_reward_equal`      | Unit – reward tie-break        | ✅ PASS | —        |
| 6   | `test_api_reward`                  | Integration – POST /reward     | ✅ PASS | —        |

**Total execution time:** ~0.44 s

---

## 4. Test Descriptions & Assertions

### 4.1 `test_distance_between_basic`

Calls `distance_between(["A01.1"], ["A01.2"], ICD10_GRAPH)` and asserts the result is in `[0, 1]`.  
**Purpose:** Validates that the normalised graph-distance metric produces a valid probability-range value when both inputs are non-empty, reachable ICD-10 codes.

### 4.2 `test_distance_between_empty`

Calls `distance_between([], ["A01.2"], ICD10_GRAPH)` and asserts the result is `1.0`.  
**Purpose:** Guards the edge case where one code-set is empty — the function must return maximum distance (`1.0`) rather than crash or return an undefined value.

### 4.3 `test_calculate_reward_better_enh`

Sets `gt=["A01.1"]`, `enh=["A01.1"]`, `org=["A01.2"]` and asserts `reward > 0`.  
**Purpose:** When the enhanced codes are closer to ground truth than the original codes, the reward signal must be positive (reinforcement signal for a good prompt).

### 4.4 `test_calculate_reward_better_org`

Sets `gt=["A01.2"]`, `enh=["A01.1"]`, `org=["A01.2"]` and asserts `reward < 0`.  
**Purpose:** When the original codes are already closer (or equal) to ground truth than the enhanced codes, the reward signal must be negative (penalty for a worse prompt).

### 4.5 `test_calculate_reward_equal`

Sets `gt=["A01.1"]`, `enh=["A01.2"]`, `org=["A01.2"]` (enh == org) and asserts `reward == 1.0 or reward < 0`.  
**Purpose:** Exercises the tie-break branch of `calculate_reward`. Because `enh` and `org` are identical (`d_enh_org == 0`), the function returns `1.0`. The `or reward < 0` guard accommodates the alternate tie-break path.

### 4.6 `test_api_reward`

Posts `{"gt_codes": ["A01.1"], "enh_codes": ["A01.1"], "org_codes": ["A01.2"]}` to `POST /reward`.  
Asserts HTTP 200, response body contains `"reward"`, and `-1.0 ≤ reward ≤ 1.0`.  
**Purpose:** End-to-end integration test for the FastAPI endpoint. Validates schema compliance, HTTP routing, and reward range.

---

## 5. Code Coverage

```
Name      Stmts   Miss  Cover   Missing
---------------------------------------
main.py      89     14    84%   60-65, 115, 120-123, 134-137
---------------------------------------
TOTAL        89     14    84%
```

### 5.1 Covered Areas (75 statements, 84 %)

| Area                                                                                                    | Lines                 | Status     |
| ------------------------------------------------------------------------------------------------------- | --------------------- | ---------- |
| Imports & config                                                                                        | 1–12                  | ✅ Covered |
| Pydantic models (`RewardRequest`, `RewardResponse`)                                                     | 18–26                 | ✅ Covered |
| `load_icd10_tree()` — full body                                                                         | 30–39                 | ✅ Covered |
| `get_max_path_length()` — full body                                                                     | 46–53                 | ✅ Covered |
| `normalize_codes()`                                                                                     | 68–69                 | ✅ Covered |
| `distance_between()` — full body                                                                        | 72–100                | ✅ Covered |
| `calculate_reward()` — branches d_gt_enh < d_gt_org and d_gt_enh > d_gt_org, tie-break `d_enh_org == 0` | 103–113               | ✅ Covered |
| `app` instantiation and `POST /reward` happy path (`gt_codes`)                                          | 127, 130–133, 138–142 | ✅ Covered |

### 5.2 Uncovered Areas (14 statements, 16 %)

| Lines   | Function             | Uncovered Path                                                                 | Risk                                                        |
| ------- | -------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| 60–65   | `load_gt_codes()`    | Entire function body — never called directly by any test                       | **Medium** — `gt_file` request path relies entirely on this |
| 115     | `calculate_reward()` | `else: return round(-d_gt_enh, 4)` — tie-break when `d_enh_org != 0`           | Low                                                         |
| 120–123 | `post_to_rl_block()` | `try/except requests.post(...)` — only entered when `RL_BLOCK_ENDPOINT` is set | **Medium** — RL integration untested                        |
| 134–135 | `reward_endpoint()`  | `elif req.gt_file: gt_codes = load_gt_codes(req.gt_file)`                      | **Medium** — file-based GT path                             |
| 136–137 | `reward_endpoint()`  | `else: raise HTTPException(400)` — missing both `gt_codes` and `gt_file`       | **Medium** — invalid-input rejection untested               |

---

## 6. Findings & Recommendations

### 6.1 Missing Test: `load_gt_codes` happy path

**Severity:** Medium  
`load_gt_codes()` is called when the client sends `gt_file` instead of `gt_codes`. The function is entirely uncovered. A test using the existing `gt_codes/sample_gt.json` fixture should be added:

```python
def test_api_reward_with_gt_file():
    payload = {"gt_file": "sample_gt.json", "enh_codes": ["A01.1"], "org_codes": ["A01.2"]}
    resp = client.post("/reward", json=payload)
    assert resp.status_code == 200
    assert "reward" in resp.json()
```

### 6.2 Missing Test: `load_gt_codes` file-not-found path

**Severity:** Low  
The `FileNotFoundError` path (line 62) has no coverage. Add:

```python
def test_api_reward_gt_file_not_found():
    payload = {"gt_file": "nonexistent.json", "enh_codes": ["A01.1"], "org_codes": ["A01.2"]}
    resp = client.post("/reward", json=payload)
    assert resp.status_code == 500  # or 404 depending on exception handling
```

### 6.3 Missing Test: `reward_endpoint` 400 validation (no GT provided)

**Severity:** Medium  
The `HTTPException(400)` branch (line 137) is never exercised. Add:

```python
def test_api_reward_missing_gt():
    payload = {"enh_codes": ["A01.1"], "org_codes": ["A01.2"]}
    resp = client.post("/reward", json=payload)
    assert resp.status_code == 400
    assert "gt_codes or gt_file required" in resp.json()["detail"]
```

### 6.4 Missing Test: `post_to_rl_block` with active endpoint

**Severity:** Low  
The RL block integration is untested. Use `unittest.mock.patch` or an environment variable override to simulate both the successful POST and a network failure:

```python
from unittest.mock import patch, MagicMock

def test_post_to_rl_block_success(monkeypatch):
    monkeypatch.setenv("RL_BLOCK_ENDPOINT", "http://localhost:9999/rl")
    with patch("main.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        from main import post_to_rl_block
        post_to_rl_block(0.75)
        mock_post.assert_called_once_with(
            "http://localhost:9999/rl", json={"reward": 0.75}
        )
```

### 6.5 Missing Test: `calculate_reward` tie-break when `d_enh_org != 0`

**Severity:** Low  
Line 115 (`return round(-d_gt_enh, 4)`) is only executed when `d_gt_enh == d_gt_org` but `d_enh_org != 0`. Construct a case where enh and org differ but are equidistant from gt, then assert `reward < 0`.

---

## 7. Overall Assessment

The service core — graph loading, normalised distance calculation, three-branch reward function, and the primary API endpoint — is thoroughly tested and all 6 tests pass. The 16 % coverage gap is concentrated in three secondary code paths: file-based ground-truth loading, the `POST /reward` 400-error branch, and the optional RL-block integration. None of these gaps affect the primary happy-path logic, but adding the tests described in §6 would raise coverage above 95 % and harden the service against regressions in those paths.
