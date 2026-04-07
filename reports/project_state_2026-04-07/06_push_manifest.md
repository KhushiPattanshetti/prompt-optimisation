# Push Manifest for Reproducible Local Runs

## Goal

Push only source/config/test assets required to run the pipeline locally, while excluding runtime artifacts, caches, checkpoints, and local outputs.

## Must be pushed

Root config and orchestration:
- .gitignore
- .dockerignore
- Dockerfile
- docker-compose.yml
- docker-compose.rl-baseline.yml
- docker-compose.rl-distributed-max.yml
- pytest.ini
- .env.rewriter-sft.example

Core services:
- dataset_svc/**
- rewriter_inference_svc/**
- icd10_coding_svc/**
- reward_metrics_svc/**
- rl_loop_svc/**

Important scripts:
- scripts/build_icd10_tree.py
- scripts/ci_test_contracts.sh
- scripts/full_dataset_train_eval.py
- scripts/smoke_test.sh
- scripts/validate_rl_single_note.sh
- scripts/launch_100note_tmux.sh
- scripts/run_100note_baseline_vs_distributed_tmux.sh
- scripts/download_dataset.sh

Data contract/support files:
- data/.gitignore
- data/download_data.py
- reward_metrics_svc/icd10_tree.json (if using prebuilt tree)

Reports/documentation:
- reports/project_state_2026-04-07/**

## Must NOT be pushed (now ignored)

- rl_checkpoints/
- sft_checkpoints/
- hf_cache/
- inference_outputs/
- reward_queue/
- logs/
- gt_codes/
- data/*.xlsx
- rl_loop_svc/rollouts/*.jsonl
- rl_loop_svc/rollouts/.seen_files
- rl_loop_svc/rollouts/.seen_rollout_ids
- scripts/__pycache__/
- .venv/ and local env files

## Local run prerequisites for new users

1. Clone repository and check out the pushed branch.
2. Copy environment template and set local values.
3. Provision dataset files using scripts/download_dataset.sh or equivalent local data path setup.
4. Build and start services using Docker Compose.
5. Run smoke/e2e checks:
   - scripts/smoke_test.sh
   - scripts/full_dataset_train_eval.py (small max-notes probe first)

## Current caveat

Distributed RL can still OOM on constrained VRAM scenarios. The latest pipeline now detects and fails these train cycles explicitly, with per-rank failure artifacts available for debugging.
