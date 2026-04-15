# Reward Metrics Microservice

## Overview

Computes a scalar reward in `[-1.0, 1.0]` using a hybrid Wu-Palmer + Jaccard + structure signal over the ICD-10 hierarchy, for use as input to a Reinforcement Learning (RL) loop. Implemented as a modular FastAPI service (v3.0.0).

## Reward Formula (spec §13)

$$R_{total} = \tanh(w_{tree} \cdot R_{tree} + w_{exact} \cdot R_{exact} + w_{structure} \cdot R_{structure})$$

| Component     | Formula                                                                              | Default weight |
| ------------- | ------------------------------------------------------------------------------------ | -------------- |
| `R_tree`      | `D_org − D_enh` (Wu-Palmer tree-distance improvement)                                | 0.60           |
| `R_exact`     | `2·J − 1`, where J = Jaccard(gt ∩ enh) / Jaccard(gt ∪ enh)                           | 0.25           |
| `R_structure` | `1 − p_invalid − p_dupes`, clamped [−1, 1]; forced **−1** if `parsing_success=False` | 0.15           |

Weights are overridable via env vars `W_TREE`, `W_EXACT`, `W_STRUCTURE`.

## Features

- Wu-Palmer similarity and composite distance metric over the ICD-10 tree (spec §6–11).
- Three-component hybrid reward: tree signal, exact match (Jaccard), and structural quality.
- String-prefix LCA fallback for out-of-tree ICD-10 codes (e.g. `J45` is correctly treated as parent of `J45.0`).
- Rollout queue: file-based `pending/acked/failed` queue with HTTP POST to RL service (spec §16).
- Explainable DEBUG-level logging: per-code LCA, similarity, coverage, extra, and cardinality breakdowns.
- Standalone simulation — no HTTP server required to generate logs.
- 152-test modular pytest suite.

## Module Structure

```
reward_metrics_svc/
  config.py        # W_TREE/W_EXACT/W_STRUCTURE weights; RL_SERVICE_URL; env-var overrides
  tree.py          # ICD-10 JSON loader, BFS depth-map, _TreeState singleton
  similarity.py    # Wu-Palmer sim(), lca(), depth(); string-prefix LCA fallback
  metrics.py       # set_distance(), coverage_penalty(), extra_penalty(), cardinality_penalty()
  reward.py        # compute_reward() → hybrid tanh reward + RewardComponents + Diagnostics
  rollout.py       # File-based rollout queue (pending/acked/failed) + HTTP POST to RL service
  log_utils.py     # Per-call DEBUG traces + rolling aggregate stats every 10 samples
  schemas.py       # Pydantic v2: RewardRequest, RewardResponse, RewardComponents, Diagnostics
  app.py           # FastAPI v3.0.0; lifespan tree reload; POST /compute_reward, GET|POST /health
  main.py          # Entry point — logging setup + uvicorn on port 8002
  simulate.py      # Standalone simulation — all 14+1 spec §19 scenarios
  icd10_tree.json  # Sample ICD-10 hierarchy (9 nodes: A00–B02)
  gt_codes/        # Sample ground-truth code files
  test/            # 152-test modular pytest suite
  rollouts/        # Runtime rollout queue dirs (JSON files git-ignored; dirs kept via .gitkeep)
```

## Setup

Python 3.10+ and a virtual environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Service

```bash
# HTTP server (port 8002)
uvicorn reward_metrics_svc.app:app --host 0.0.0.0 --port 8002 --log-level debug

# Or via the entry point
python -m reward_metrics_svc.main
```

## Standalone Simulation (logs without a server)

```bash
python -m reward_metrics_svc.simulate                     # DEBUG logs to stdout
python -m reward_metrics_svc.simulate 2>&1 | tee run.log  # save to file
```

Runs all 14+1 spec §19 scenarios directly against the Python functions, emitting full `DEBUG`-level traces for every LCA, similarity, penalty, and reward computation.

## API Endpoints

### `POST /compute_reward`

**Request:**

```json
{
  "note_id": "note_001",
  "gt_codes": ["A01.1"],
  "enh_codes": ["A01.2"],
  "org_codes": ["B01"],
  "invalid_codes": [],
  "duplicate_codes": [],
  "parsing_success": true,
  "state": "<prompt_state>",
  "action": "<rewrite_action>",
  "log_prob_old": 0.1,
  "value_estimate": 0.5
}
```

**Response:**

```json
{
  "note_id": "note_001",
  "reward": 0.4859,
  "metrics": {
    "D_enh": 0.2178,
    "D_org": 0.6533,
    "delta_D": 0.4356,
    "reward_components": {
      "R_tree": 0.4356,
      "R_exact": 1.0,
      "R_structure": 1.0
    },
    "components_enh": {
      "D_set": 0.0,
      "P_cov": 0.0,
      "P_extra": 0.0,
      "P_card": 0.0
    },
    "components_org": {
      "D_set": 1.0,
      "P_cov": 0.667,
      "P_extra": 0.267,
      "P_card": 0.0
    },
    "diagnostics": {
      "worst_gt_coverage_code": "A01.1",
      "worst_pred_match_code": "A01.2",
      "invalid_codes": [],
      "duplicate_codes": []
    }
  }
}
```

If `parsing_success` is `false`, `R_structure` is forced to `-1.0`, which strongly anchors the total reward toward the negative end.

### `GET /health` · `POST /health`

Returns `{ "status": "ok", "tree_loaded": true, "node_count": 9 }`.

## Testing

```bash
python -m pytest test/ -v
```

| File                      | Coverage                                                        |
| ------------------------- | --------------------------------------------------------------- |
| `test/test_tree.py`       | Tree loading, BFS depth/parent maps                             |
| `test/test_similarity.py` | `_infer_depth`, `lca`, `sim`, `distance`, prefix-LCA fallback   |
| `test/test_metrics.py`    | All penalty functions, `distance_between`                       |
| `test/test_reward.py`     | `compute_reward` — hybrid components, sign, bounds, diagnostics |
| `test/test_api.py`        | Full API — spec §19 scenarios, edge cases, rollout wiring       |

---

For design details and spec, see `llm-v2.txt`.
