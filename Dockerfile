FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY dataset_svc/requirements.txt             requirements_dataset.txt
COPY rewriter_inference_svc/requirements.txt  requirements_rewriter.txt
COPY icd10_coding_svc/requirements.txt        requirements_icd10.txt
COPY reward_metrics_svc/requirements.txt      requirements_reward.txt
COPY rl_loop_svc/requirements.txt             requirements_rl.txt

RUN pip install --no-cache-dir \
    torch==2.2.1 \
    "transformers>=4.40.0,<5.0.0" \
    "peft>=0.10.0" \
    "bitsandbytes>=0.43.0" \
    "accelerate>=0.29.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.2.0" \
    pandas \
    networkx \
    requests \
    httpx \
    gdown \
    pytest \
    -r requirements_dataset.txt \
    -r requirements_rewriter.txt \
    -r requirements_icd10.txt \
    -r requirements_reward.txt \
    -r requirements_rl.txt

COPY . .

RUN mkdir -p data rl_checkpoints sft_checkpoints/med42 \
    inference_outputs/icd10 gt_codes \
    rl_loop_svc/rollouts hf_cache logs \
    best_prompts_cache

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV PYTHONPATH=/app
ENV DATASET_SVC_URL=http://dataset_svc:8003
ENV REWARD_SERVICE_URL=http://reward_svc:8002
ENV RL_BLOCK_ENDPOINT=http://rl_loop_svc:8004/rollout
ENV REWRITER_SERVICE_URL=http://rewriter_svc:8000
ENV RL_CHECKPOINT_DIR=/app/rl_checkpoints
ENV RL_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
ENV RL_HIDDEN_SIZE=3072
ENV RL_BATCH_SIZE=4
ENV RL_GRADIENT_ACCUMULATION_STEPS=4
