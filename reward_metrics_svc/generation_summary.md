# Generation Summary

## Microservice: Reward Metrics (v3.0.0)

- **Goal:** Compute a scalar reward in `[-1.0, 1.0]` using a hybrid Wu-Palmer + Jaccard + structure signal over the ICD-10 hierarchy, for RL input (spec §13).
- **Inputs:**
  - `icd10_tree.json` — ICD-10 hierarchy (loaded at startup via BFS into a `_TreeState` singleton)
  - Ground truth ICD-10 codes (`gt_codes`) supplied per request via the API
  - Enhanced ICD-10 codes from the Task LLM (`enh_codes`)
  - Non-enhanced (original) ICD-10 codes from the Task LLM (`org_codes`)
  - Validation metadata: `invalid_codes`, `duplicate_codes`, `parsing_success`
  - PPO rollout fields: `state`, `action`, `log_prob_old`, `value_estimate`
- **Output:**
  - Scalar reward `tanh(w_tree·R_tree + w_exact·R_exact + w_structure·R_structure)`
  - `reward_components`: `{R_tree, R_exact, R_structure}`
  - `diagnostics`: `{worst_gt_coverage_code, worst_pred_match_code, invalid_codes, duplicate_codes}`
  - Full distance breakdown: `D_enh`, `D_org`, `delta_D`, `components_enh`, `components_org`

- **Reward Components (spec §13):**
  - `R_tree = D_org − D_enh` — improvement in Wu-Palmer tree distance
  - `R_exact = 2·J − 1` — Jaccard-based exact match signal ∈ [−1, 1]
  - `R_structure = 1 − p_invalid − p_dupes`, clamped [−1, 1]; forced −1 if `parsing_success=False`
  - **Weights:** `W_TREE=0.6, W_EXACT=0.25, W_STRUCTURE=0.15` (env-var overridable)

- **Core Logic (modular):**
  - `config.py` — `W_TREE`, `W_EXACT`, `W_STRUCTURE`, `RL_SERVICE_URL`, `ROLLOUT_DIR`; loaded from env vars with defaults
  - `tree.py` — pure-Python JSON loader; builds `parent_map` and `depth_map` via BFS; exposes a singleton `_TreeState`; virtual root `__ROOT__` anchors all top-level codes
  - `similarity.py` — Wu-Palmer `sim(a,b) = 2·depth(lca) / (depth(a) + depth(b))`; string-heuristic depth fallback for out-of-tree codes; **string-prefix LCA fallback** recognises parent/sibling relationships (e.g. `J45` is parent of `J45.0`)
  - `metrics.py` — `set_distance`, `coverage_penalty`, `extra_penalty`, `cardinality_penalty`, `distance_between`, `distance_with_components`
  - `reward.py` — `compute_reward(gt, enh, org, invalid_codes, duplicate_codes, parsing_success)` → `(float, dict)`
  - `rollout.py` — file-based rollout queue (`rollouts/pending/` → `rollouts/acked/` or `rollouts/failed/`); HTTP POST to `RL_SERVICE_URL`; stdlib only
  - `log_utils.py` — per-call DEBUG traces + rolling aggregate stats every 10 samples
  - `schemas.py` — Pydantic v2: `RewardRequest`, `RewardResponse`, `RewardComponents`, `ComponentMetrics`, `Diagnostics`, `RewardMetrics`
  - `app.py` — FastAPI v3.0.0 with lifespan tree reload; routes `POST /compute_reward`, `GET|POST /health`
  - `main.py` — lean entry point (logging setup + uvicorn on port 8002)
  - `simulate.py` — standalone simulation; runs all 14+1 spec §19 scenarios without an HTTP server

- **API:**
  - `POST /compute_reward` — accepts full `RewardRequest` (note_id, codes, validation metadata, PPO fields); returns `RewardResponse` with reward + `reward_components` + `diagnostics` + component breakdowns
  - `GET /health` / `POST /health` — liveness check with `tree_loaded` and `node_count`
  - `parsing_success=false` forces `R_structure=-1`, strongly anchoring total reward toward negative

- **Testing (152 tests, all passing):**
  - `test/test_tree.py` — tree state, parent/depth maps, reload idempotency
  - `test/test_similarity.py` — `_infer_depth` heuristic, `lca`, `sim`, `distance`
  - `test/test_metrics.py` — all penalty functions, `distance_between`, components dict
  - `test/test_reward.py` — hybrid formula components, sign semantics, bounds, diagnostics structure
  - `test/test_api.py` — spec §19 scenarios, parsing failure, deduplication, case normalisation, rollout wiring, edge cases

- **Tech Stack:**
  - Python 3.10, FastAPI, uvicorn, Pydantic v2, pytest, httpx
  - No networkx — tree traversal uses pure-Python dicts (BFS)

- **Directory Structure:**
  ```
  reward_metrics_svc/
    config.py · tree.py · similarity.py · metrics.py
    reward.py · rollout.py · log_utils.py · schemas.py
    app.py · main.py · simulate.py
    icd10_tree.json
    gt_codes/
    rollouts/  (pending/ · acked/ · failed/ — .json files git-ignored)
    test/  (conftest.py + 5 test files)
    requirements.txt · llm-v2.txt
  ```

---

Service upgraded from v1 (tanh(k·ΔD)) to v3.0.0 (hybrid three-component reward). Design spec in `llm-v2.txt`.
