# dataset_svc

## Overview

`dataset_svc` is the single source of truth for all dataset access in the
`prompt_optimisation` pipeline. It loads `notes.csv` (clinical notes) and
`diagnoses.csv` (ICD-10 diagnosis codes) once at startup, builds in-memory
indexes, and serves data to all other microservices via a REST API. No other
service in the pipeline is permitted to load the CSV files directly — all
dataset access must go through this service.

## Setup

### Step 1: Download the data

```bash
pip install gdown
python data/download_data.py
```

### Step 2: Install dependencies

```bash
pip install -r dataset_svc/requirements.txt
```

### Step 3: Start the service

```bash
uvicorn dataset_svc.app:app --host 0.0.0.0 --port 8003
```

### Step 4: Verify the service is ready

```bash
curl http://localhost:8003/health
```

Wait until the response contains `"status": "ok"`.

## Important

- **Do not start any other service** until `dataset_svc` returns
  `"status": "ok"` from `/health`.
- The data files are **not committed** to the repository. Always run
  `download_data.py` on a new machine before running any service.
