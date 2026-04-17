# Prompt-Opt-v2 Resource Requirement Report (RAM and VRAM)

Date: 2026-04-13
Scope: prompt-opt-v2/prompt-optimisation
Method: Static analysis from existing code, compose configuration, and existing artifacts/log-derived outputs only (no fresh benchmark run for this report).

## 1. Evidence Used

### 1.1 Service and GPU placement from compose
- Rewriter pinned to GPU 0 via NVIDIA_VISIBLE_DEVICES=0 and device_ids=["0"] (docker-compose.yml).
- ICD service pinned to GPU 1 via NVIDIA_VISIBLE_DEVICES=1 and device_ids=["1"] (docker-compose.yml).
- RL loop configured with policy on GPU 2 and reference on GPU 3:
  - RL_POLICY_CUDA_DEVICE=2
  - RL_REFERENCE_CUDA_DEVICE=3
  - device_ids=["0","1","2","3"]

### 1.2 Model choices and precision
- ICD model: m42-health/Llama3-Med42-8B (icd10_coding_svc/config.py).
- Rewriter model: ishanmane/phi3-rewriter-sft (Phi-3-mini family) (rewriter_inference_svc/config.py).
- ICD, Rewriter, RL Policy, and RL Reference all load with 4-bit NF4 quantization (BitsAndBytesConfig load_in_4bit=True in respective model loaders).

### 1.3 Data and cache footprint from existing files
- data/notes.csv: 1.4G
- data/diagnoses.csv: 39M
- hf_cache/models--m42-health--Llama3-Med42-8B: 15G
- hf_cache/models--microsoft--Phi-3-mini-4k-instruct: 7.2G
- reward_metrics_svc/icd10_tree.json: 26M

### 1.4 Observed pipeline stage timing from existing run artifact
From run_results/run_20260413T133451Z.notes.jsonl (20-note run):
- avg_rewrite_sec: 40.92
- avg_icd10_sec: 30.00
- avg_reward_sec: 0.074
- avg_note_total_sec: 70.99

This indicates model inference dominates runtime; reward compute is lightweight.

## 2. What Drives Memory Usage

### 2.1 VRAM drivers
1. ICD service (Med42 8B in 4-bit) is the heaviest single inference service.
2. RL loop keeps two model instances active (policy + reference) on separate GPUs.
3. Rewriter service keeps a Phi-3-mini-based model live for continuous rewriting.

### 2.2 RAM drivers
1. dataset_svc loads full notes and diagnoses into in-memory Python indexes (DatasetStore).
2. Multiple Python services run concurrently (dataset, rewriter, icd10, reward, rl_loop).
3. Tokenization, request payloads, and model serving buffers add transient memory overhead.

## 3. Minimum Necessary RAM and VRAM

## 3.1 Full stack (as currently configured, including RL training)

Required minimum to run reliably:
- GPUs: 4
- VRAM per GPU (practical minimum):
  - GPU0 (rewriter): 12 GB
  - GPU1 (icd10/Med42): 20 GB
  - GPU2 (RL policy training): 24 GB
  - GPU3 (RL reference): 16 GB
- System RAM (host): 96 GB

Recommended for stable operation under sustained load:
- GPUs: 4 x 32 GB VRAM
- System RAM: 128 GB

Why this is the minimum:
- Compose hard-codes a 4-GPU topology for full pipeline operation.
- RL policy and reference are split across dedicated GPUs.
- ICD Med42 is larger than rewriter and requires the largest inference VRAM budget.
- Dataset is fully materialized in memory, so RAM pressure is non-trivial even before model-serving overhead.

## 3.2 Inference-only mode (no RL training cycle)

If RL training is disabled/skipped and only dataset + rewriter + icd10 + reward are used:
- GPUs: 2 (rewriter + icd10)
- VRAM practical minimum:
  - Rewriter GPU: 12 GB
  - ICD GPU: 20 GB
- System RAM minimum: 48 GB
- System RAM recommended: 64 GB

## 3.3 Hard lower bound from code-level guardrails
In RL settings, distributed launch guardrails include:
- distributed_min_free_ram_gb = 64
- distributed_min_free_vram_gb_per_gpu = 8

These are guardrails, not full-stack sufficiency numbers. Realistic full operation needs higher resources than these minima.

## 3.4 Single-device minimum requirement

Interpretation used here: single device means one host with one GPU.

### 3.4.1 As-is codebase (no topology changes)
- Not feasible on one GPU for full stack with RL training.
- Reason: current compose and service design keep multiple model services alive concurrently, and RL expects separate policy/reference device placement.

### 3.4.2 Single GPU, inference-only (no RL training)
Minimum practical target:
- GPU count: 1
- VRAM: 24 GB minimum (32 GB recommended)
- RAM: 64 GB minimum (96 GB recommended)

Assumptions:
- Run dataset + rewriter + icd10 + reward only.
- Keep Med42 as ICD model and Phi-3-mini family rewriter.
- Avoid concurrent RL model residency.

### 3.4.3 Single GPU, full flow including RL training (with runtime/config changes)
Minimum practical target:
- GPU count: 1
- VRAM: 48 GB minimum (80 GB preferred)
- RAM: 128 GB minimum

Assumptions:
- Policy and reference models must share one GPU or be time-sliced.
- Throughput will drop significantly versus multi-GPU setup.
- Additional engineering is needed for robust scheduling to avoid OOM and contention.

## 4. Storage Requirement (supporting requirement)

Required local disk (practical minimum):
- Model cache and checkpoints: at least 35-50 GB free
- Dataset + outputs + growth headroom: at least 20 GB free

Recommended total free disk before runs:
- 80 GB+

Rationale:
- Existing HF cache already includes a 15G Med42 snapshot and 7.2G Phi-3-mini snapshot.
- Additional snapshots/checkpoints and run artifacts accumulate over time.

## 5. Final Recommendation

For this exact prompt-opt-v2 setup (Med42 ICD + Phi-3-mini rewriter + RL loop active), provision at least:
- RAM: 96 GB (128 GB preferred)
- VRAM: 4 GPUs with at least 12/20/24/16 GB respectively (4 x 32 GB preferred)

If you want a single-number procurement target that avoids edge-case instability:
- 4 GPUs x 32 GB VRAM and 128 GB RAM.
