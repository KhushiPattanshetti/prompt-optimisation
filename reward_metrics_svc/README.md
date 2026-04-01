# Reward Metrics Microservice

## Overview

This microservice computes a scalar reward in [-1.0, 1.0] using ICD-10 code evaluations and the ICD-10 tree structure, for use as input to a Reinforcement Learning (RL) block.

## Features

- Computes semantic distance between ICD-10 code sets using the ICD-10 hierarchy.
- Calculates reward based on enhanced, original, and ground truth codes.
- REST API for reward calculation.
- Forwards reward to RL block endpoint.

## Usage Instructions

### 1. Setup

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Directory Structure

- `main.py` : API and core logic
- `icd10_tree.json` : ICD-10 tree structure (JSON)
- `gt_codes/` : Directory with ground truth codes (JSON files)
- `test/` : Unit and integration tests
- `llm.json`, `llm.txt` : Design and requirements

### 3. Running the Service

```bash
uvicorn main:app --reload
```

### 4. API Endpoints

- `POST /reward` :
  - **Request JSON:**
    ```json
    {
      "enh_codes": ["A01.1", "B02"],
      "org_codes": ["A01.2"],
      "gt_codes": ["A01.1", "A02"] // optional, else loaded from ./gt_codes/
    }
    ```
  - **Response JSON:**
    ```json
    { "reward": 0.85 }
    ```

### 5. Testing

```bash
pytest
```

### 6. Example

See `test/` for sample payloads and test cases.

---

For more details, see `llm.json` and `llm.txt`.
