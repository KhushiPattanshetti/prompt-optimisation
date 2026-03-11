# icd10_coding_svc

Frozen LLM inference microservice for ICD-10 clinical coding. Part of the `prompt_optimisation` PPO pipeline.

## Pipeline Position

```
Prompt Rewriter Service (port 8000)
        ↓
icd10_coding_svc (port 8001)        ← THIS SERVICE
        ↓
reward_metric_svc (port 8002)
```

## What It Does

1. Receives a `note_id`, `original_prompt`, and `rewritten_prompt`.
2. Runs **two** frozen inference passes using `m42-health/Llama3-Med42-8B` (4-bit NF4 quantised):
   - **Enhanced pass** → `enh_codes` (from `rewritten_prompt`)
   - **Original pass** → `org_codes` (from `original_prompt`)
3. Fetches **ground truth** ICD-10 codes (`gt_codes`) from the diagnoses dataset via `note_id`.
4. Returns all three code sets and forwards them asynchronously to `reward_metric_svc`.

## Setup

```bash
pip install -r requirements.txt
```

Ensure the following data files exist relative to the project root:

- `data/notes.csv` — columns: `note_id`, `subject_id_x`, `hadm_id`, `text`
- `data/diagnoses.csv` — columns: `note_id`, `seq_num`, `icd_code`

## Running

```bash
cd icd10_coding_svc
uvicorn app:app --host 0.0.0.0 --port 8001
```

## Endpoints

| Method | Path               | Description                                       |
|--------|--------------------|----------------------------------------------------|
| POST   | `/generate_codes`  | Run dual inference + fetch GT codes                |
| GET    | `/health`          | Returns model name and `weights_frozen: true`      |

### POST /generate_codes

**Request:**
```json
{
  "note_id": "12345",
  "original_prompt": "Extract ICD-10 codes from: Patient has chest pain...",
  "rewritten_prompt": "Step 1: Identify symptoms... Step 2: Map to ICD-10..."
}
```

**Response:**
```json
{
  "note_id": "12345",
  "enh_codes": ["R07.9", "I20.9"],
  "org_codes": ["R07.9"],
  "gt_codes": ["R07.9", "I20.9", "R06.0"],
  "enh_raw_output": "...",
  "org_raw_output": "...",
  "parsing_success": true
}
```

## Testing

```bash
pytest tests/unit/              # Unit tests (no GPU needed)
pytest tests/integration/       # Integration tests (mocked model)
pytest tests/                   # All tests
```
