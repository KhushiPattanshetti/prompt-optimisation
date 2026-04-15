# trajectory_store_svc

A standalone Python microservice for generating, storing, and preparing RL training rollouts. It handles the full pre-training data pipeline: rollout generation → JSONL storage → preprocessing → advantage computation → batch file output.

> **Scope:** This service does **not** implement model training, PPO loss, checkpointing, or reward computation. Rewards are consumed as-is from an external source (stubbed in simulation mode).

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Batch Output Format](#batch-output-format)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [Simulation Mode](#simulation-mode)
- [API](#api)
- [Testing](#testing)

---

## Architecture

```
Rollout Generator
      │
      ▼
 get_reward() stub  ◄── (external reward_metrics_svc in production)
      │
      ▼
 JSONL Storage  (rollouts_store/rollouts_<timestamp>.jsonl)
      │
      ▼
 RolloutLoader  (incremental reads, offset-tracked, restart-safe)
      │
      ▼
 Preprocessing  (group_id, sample_weight)
      │
      ▼
 RolloutBuffer  (accumulates until batch_size)
      │
      ▼
 Advantage Computation  (standard | grpo | hybrid)
      │
      ▼
 BatchWriter  (prepared_batches/prepared_batch_<id>.json)
```

---

## Project Structure

```
trajectory_store_svc/
├── main.py                  FastAPI application entry-point
├── config.py                All configuration (env-var overrideable)
├── pytest.ini
│
├── api/
│   └── routes.py            GET /status endpoint
│
├── schemas/
│   └── rollout_schema.py    Pydantic Rollout model
│
├── storage/
│   └── rollout_loader.py    JSONL writer + incremental reader
│
├── buffer/
│   └── rollout_buffer.py    RolloutBuffer + RolloutBatch
│
├── processing/
│   ├── preprocessing.py     group_id resolution, sample_weight
│   ├── reward_processing.py Normalisation guard
│   └── advantage.py         standard / grpo / hybrid advantage modes
│
├── serving/
│   └── batch_writer.py      Writes batch JSON files, manages reuse state
│
├── simulation/
│   └── simulator.py         End-to-end pipeline runner with synthetic data
│
├── tests/
│   ├── test_unit.py         31 unit tests
│   ├── test_integration.py  12 integration tests
│   └── test_simulation.py   11 failure + e2e tests
│
└── utils/
    └── logging.py           Structured colour-coded logger
```

Runtime directories (git-ignored, created automatically):

| Directory           | Contents                                                                     |
| ------------------- | ---------------------------------------------------------------------------- |
| `rollouts_store/`   | Timestamped JSONL rollout files + `seen_files.json` + `segment_offsets.json` |
| `prepared_batches/` | Output batch JSON files consumed by the training service                     |

---

## Pipeline

Each simulation run executes the following stages in sequence:

| Stage | Module                        | Description                                               |
| ----- | ----------------------------- | --------------------------------------------------------- |
| 1     | `simulation/simulator.py`     | Generate synthetic rollouts, call `get_reward()` stub     |
| 2     | `storage/rollout_loader.py`   | Append rollouts to timestamped JSONL file                 |
| 3     | `storage/rollout_loader.py`   | Load new rollouts incrementally (offset-tracked)          |
| 4     | `processing/preprocessing.py` | Attach `group_id` (SHA-256 of prompt) and `sample_weight` |
| 5     | `buffer/rollout_buffer.py`    | Accumulate until `batch_size` is reached                  |
| 6     | `processing/advantage.py`     | Compute advantages and returns                            |
| 7     | `serving/batch_writer.py`     | Write `prepared_batch_<id>.json`, manage batch reuse      |

---

## Batch Output Format

Each file in `prepared_batches/` contains:

```json
{
  "batch_id": "uuid",
  "rollout_ids": ["uuid", "..."],
  "rewards": [0.65, "..."],
  "advantages": [0.21, "..."],
  "returns": [0.65, "..."],
  "group_ids": ["1888ba25", "..."],
  "sample_weights": [1.0, "..."]
}
```

---

## Configuration

All values have defaults and can be overridden via `TRAJ_*` environment variables.

| Variable                         | Default            | Description                                                   |
| -------------------------------- | ------------------ | ------------------------------------------------------------- |
| `TRAJ_BASE_DIR`                  | service directory  | Root for data directories                                     |
| `TRAJ_ROLLOUTS_DIR`              | `rollouts_store`   | Subdirectory for JSONL files                                  |
| `TRAJ_BATCHES_DIR`               | `prepared_batches` | Subdirectory for batch output                                 |
| `TRAJ_BATCH_SIZE`                | `8`                | Rollouts per batch                                            |
| `TRAJ_NORMALIZE_REWARDS`         | `false`            | z-score normalise raw rewards                                 |
| `TRAJ_REWARD_ALREADY_NORMALIZED` | `true`             | Skip normalisation if rewards are pre-normalised              |
| `TRAJ_ADVANTAGE_MODE`            | `grpo`             | `standard` / `grpo` / `hybrid`                                |
| `TRAJ_GRPO_ENABLED`              | `true`             | Enable GRPO grouping                                          |
| `TRAJ_GRPO_GROUP_SIZE`           | `4`                | Target group size                                             |
| `TRAJ_GRPO_MIN_GROUP_SIZE`       | `2`                | Groups smaller than this get advantage = 0                    |
| `TRAJ_HYBRID_LAMBDA`             | `0.5`              | Weight of GRPO term in hybrid mode: `λ·grpo + (1-λ)·standard` |
| `TRAJ_BATCH_REUSE_MODE`          | `single_pass`      | `single_pass` / `repeat`                                      |
| `TRAJ_MAX_BATCH_REUSE_COUNT`     | `3`                | Max reuses per batch in `repeat` mode                         |
| `TRAJ_HOST`                      | `0.0.0.0`          | FastAPI bind host                                             |
| `TRAJ_PORT`                      | `8200`             | FastAPI bind port                                             |
| `TRAJ_SIM_NUM_ROLLOUTS`          | `32`               | Default rollouts per simulation run                           |
| `TRAJ_SIM_REWARD_MEAN`           | `0.5`              | Mean of Gaussian reward distribution                          |
| `TRAJ_SIM_REWARD_STD`            | `0.2`              | Std of Gaussian reward distribution                           |

---

## Running the Service

### Prerequisites

```bash
# Uses the shared venv from reward_metrics_svc (Python 3.10)
source ../reward_metrics_svc/.venv/bin/activate
cd trajectory_store_svc
```

### Start the FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8200 --reload
```

---

## Simulation Mode

Run the full pipeline end-to-end without an HTTP server.

```bash
# Default: 32 rollouts, batch_size=8, GRPO single-pass
python -m simulation.simulator

# Custom rollout count
python -m simulation.simulator 64

# Repeat mode with hybrid advantages
TRAJ_BATCH_REUSE_MODE=repeat \
TRAJ_MAX_BATCH_REUSE_COUNT=3 \
TRAJ_ADVANTAGE_MODE=hybrid \
TRAJ_HYBRID_LAMBDA=0.6 \
python -m simulation.simulator 16
```

Batch files are written to `prepared_batches/`. The console output is structured and stage-separated for easy debugging:

```
[2026-04-15 21:58:21] [INFO    ] [simulation.simulator]  ── SIMULATION START ──
[2026-04-15 21:58:21] [INFO    ] [simulation.simulator]  Simulation parameters  num_rollouts=32  batch_size=8 ...
[2026-04-15 21:58:21] [INFO    ] [simulation.simulator]  ── STAGE 1: Generate & Store ──
...
[2026-04-15 21:58:21] [INFO    ] [serving.batch_writer]  Batch written to disk  batch_id='...'  size=8  reward_mean='0.49'
[2026-04-15 21:58:21] [INFO    ] [simulation.simulator]  ── SIMULATION COMPLETE ──
```

---

## API

### `GET /status`

Returns service health and counters.

```json
{
  "service": "trajectory_store_svc",
  "status": "ok",
  "total_rollouts": 32,
  "total_batches": 4,
  "current_mode": "grpo",
  "batch_reuse_mode": "single_pass"
}
```

---

## Testing

```bash
# Run all 54 tests
pytest

# Run a specific suite
pytest tests/test_unit.py
pytest tests/test_integration.py
pytest tests/test_simulation.py
```

| Suite                 | Tests | Covers                                                                                                                |
| --------------------- | ----- | --------------------------------------------------------------------------------------------------------------------- |
| `test_unit.py`        | 31    | Schema validation, JSONL I/O, offset tracking, preprocessing, buffer, reward normalisation guard, all advantage modes |
| `test_integration.py` | 12    | Full pipeline, repeat/single-pass modes, no-duplicate guarantee, batch size correctness                               |
| `test_simulation.py`  | 11    | Corrupted JSONL, partial writes, empty buffer, insufficient rollouts, e2e simulator, restart safety                   |
