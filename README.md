# Prompt Optimisation Pipeline

End-to-end multi-service pipeline for clinical note prompt rewriting, ICD-10 coding, reward shaping, and RL training.

## Documentation Map

- Project state overview: [reports/project_state_2026-04-07/00_overview.md](reports/project_state_2026-04-07/00_overview.md)
- Dataset service: [reports/project_state_2026-04-07/01_dataset_svc.md](reports/project_state_2026-04-07/01_dataset_svc.md)
- Rewriter service: [reports/project_state_2026-04-07/02_rewriter_inference_svc.md](reports/project_state_2026-04-07/02_rewriter_inference_svc.md)
- ICD10 service: [reports/project_state_2026-04-07/03_icd10_coding_svc.md](reports/project_state_2026-04-07/03_icd10_coding_svc.md)
- Reward service: [reports/project_state_2026-04-07/04_reward_metrics_svc.md](reports/project_state_2026-04-07/04_reward_metrics_svc.md)
- RL loop service: [reports/project_state_2026-04-07/05_rl_loop_svc.md](reports/project_state_2026-04-07/05_rl_loop_svc.md)
- Theory meta-doc: [reports/project_state_2026-04-07/07_theoretical_concepts_flow_methods_schemas.md](reports/project_state_2026-04-07/07_theoretical_concepts_flow_methods_schemas.md)

## 1) What This Project Runs

The default pipeline runs these services:

- dataset_svc (port 8003): serves notes and GT ICD labels
- rewriter_svc (port 8000): rewrites note into coding-oriented prompt
- icd10_svc (port 8001): predicts ICD codes for enhanced and original prompts
- reward_svc (port 8002): computes reward and queues rollouts
- rl_loop_svc (port 8004): ingests rollouts and runs training cycles

Orchestrator script:

- [scripts/full_dataset_train_eval.py](scripts/full_dataset_train_eval.py)

Flow per note:

1. Read note from dataset_svc.
2. Rewrite prompt via rewriter_svc.
3. Generate ICD predictions via icd10_svc.
4. For train split, enqueue rollout and periodically trigger RL train.
5. For val/test split, compute reward only (no training leakage).

## 2) Prerequisites

- Linux
- Docker Engine with Docker Compose v2
- NVIDIA drivers + NVIDIA Container Toolkit
- GPUs visible to Docker

Quick checks:

```bash
nvidia-smi
docker --version
docker compose version
```

## 3) Clone And Enter Project

```bash
git clone <your-repo-url>
cd prompt-optimisation-merged
```

## 4) Prepare Data

Expected files:

- [data/notes.csv](data/notes.csv)
- [data/diagnoses.csv](data/diagnoses.csv)
- [data/section111_valid_icd10_october2025.xlsx](data/section111_valid_icd10_october2025.xlsx)

If missing, download them:

```bash
# Optional but recommended for ICD spreadsheet auto-download:
# export ICD10_SECTION111_XLSX_FILE_ID=<google_drive_file_id>
python data/download_data.py
```

Generate the ICD-10 hierarchy JSON locally (not tracked in git):

```bash
python scripts/build_icd10_tree.py
```

This creates:

- [reward_metrics_svc/icd10_tree.json](reward_metrics_svc/icd10_tree.json)

Run this once after downloading/updating the ICD spreadsheet.

## 5) Start Services

### Option A: Baseline (default compose)

```bash
docker compose up --build -d
```

### Option B: RL baseline override (single-process RL settings)

```bash
docker compose -f docker-compose.yml -f docker-compose.rl-baseline.yml up --build -d
```

### Option C: RL distributed max override

```bash
docker compose -f docker-compose.yml -f docker-compose.rl-distributed-max.yml up --build -d
```

## 6) Verify Service Health

```bash
curl -fsS http://localhost:8003/health
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8001/health
curl -fsS http://localhost:8002/health
curl -fsS http://localhost:8004/status
```

If any service is not ready yet, check logs:

```bash
docker compose logs -f --tail=200
```

## 7) Run End-to-End Pipeline (Beginner Safe)

Recommended: run orchestrator inside a running container so dependencies and network names are already configured.

### Small smoke run

```bash
docker compose exec dataset_svc python scripts/full_dataset_train_eval.py \
  --dataset-url http://dataset_svc:8003 \
  --rewriter-url http://rewriter_svc:8000 \
  --icd10-url http://icd10_svc:8001 \
  --reward-url http://reward_svc:8002 \
  --rl-url http://rl_loop_svc:8004 \
  --max-notes 20 \
  --batch-size 4 \
  --train-every 4
```

### Full run over complete dataset

```bash
docker compose exec dataset_svc python scripts/full_dataset_train_eval.py \
  --dataset-url http://dataset_svc:8003 \
  --rewriter-url http://rewriter_svc:8000 \
  --icd10-url http://icd10_svc:8001 \
  --reward-url http://reward_svc:8002 \
  --rl-url http://rl_loop_svc:8004 \
  --max-notes 0
```

Notes:

- `--max-notes 0` means full dataset.
- The script auto-waits for services and can reset reward observability at run start.
- Exit code `0` means run guards passed.
- Exit code `2` means run guard failure (for example train-cycle failure or transport degradation).

## 8) Optional: Run Orchestrator From Host

If you prefer local execution (outside Docker), create a Python env with required packages and run:

```bash
python scripts/full_dataset_train_eval.py
```

Defaults target localhost service ports from compose:

- dataset: `http://localhost:8003`
- rewriter: `http://localhost:8000`
- icd10: `http://localhost:8001`
- reward: `http://localhost:8002`
- rl: `http://localhost:8004`

## 9) Where Outputs Go

- Rewriter outputs: [inference_outputs](inference_outputs)
- GT cache: [gt_codes](gt_codes)
- Reward queue files: [reward_queue](reward_queue)
- RL checkpoints: [rl_checkpoints](rl_checkpoints)
- RL rollout segments: [rl_loop_svc/rollouts](rl_loop_svc/rollouts)
- Logs and reports: [logs](logs), [reports](reports)

## 10) Stop And Clean Up

Stop services:

```bash
docker compose down
```

Stop and remove volumes too:

```bash
docker compose down -v
```

## 11) Quick Troubleshooting

- Service does not become healthy:
  - Run `docker compose logs -f <service_name>` and check model/data path errors.
- Data file missing errors:
  - Ensure [data/notes.csv](data/notes.csv), [data/diagnoses.csv](data/diagnoses.csv), and [data/section111_valid_icd10_october2025.xlsx](data/section111_valid_icd10_october2025.xlsx) exist, or run `python data/download_data.py`.
- Missing ICD tree JSON:
  - Run `python scripts/build_icd10_tree.py` to regenerate [reward_metrics_svc/icd10_tree.json](reward_metrics_svc/icd10_tree.json).
- RL train-cycle failures:
  - Check [rl_loop_svc](rl_loop_svc) logs and `http://localhost:8004/status` fields such as `last_train_success` and `last_train_error`.
- Reward transport degradation:
  - Check `http://localhost:8002/observability` and queue status endpoints.
