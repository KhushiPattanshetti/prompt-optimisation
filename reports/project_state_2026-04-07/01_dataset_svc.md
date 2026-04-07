# dataset_svc Detailed Report

## Navigation

- [Overview](00_overview.md)
- [Dataset Service](01_dataset_svc.md)
- [Rewriter Service](02_rewriter_inference_svc.md)
- [ICD10 Service](03_icd10_coding_svc.md)
- [Reward Service](04_reward_metrics_svc.md)
- [RL Loop Service](05_rl_loop_svc.md)
- [Theory Meta-Doc](07_theoretical_concepts_flow_methods_schemas.md)

## 1. What this service does

dataset_svc is the source-of-truth data service for the entire pipeline.

It provides:
- clinical note text
- ground-truth ICD code labels
- batch slices for iterative processing
- service readiness metadata

Beginner mental model:
- Think of this service as a read-only data API on top of CSV files.
- Every downstream service depends on it for consistent note_id-centered data.

## 2. Core concepts (beginner friendly)

## 2.1 Source of truth

Two files define the data universe:
- notes.csv: note_id -> free-text clinical note
- diagnoses.csv: note_id -> ordered ICD labels

The pipeline assumes these are authoritative for evaluation and training.

## 2.2 In-memory indexing

At startup, dataset_svc loads data once and builds indexes in memory:
- notes_index: dict[note_id] -> text
- gt_codes_index: dict[note_id] -> [icd_code sequence]
- note_ids_list: stable ordered ID list for slicing

Why this matters:
- downstream requests become fast lookups
- expensive CSV parsing does not happen per request

## 2.3 Multiline-safe note parsing

Clinical notes can contain line breaks inside text fields.
The store therefore uses a custom row boundary detector (regex) instead of naive line-based CSV parsing for notes.csv.

## 2.4 Deterministic pagination

The /batch and /note_ids endpoints use offset-size pagination.
This allows orchestration scripts to process data in deterministic chunks.

## 3. Startup and runtime flow

1. FastAPI lifespan starts.
2. validate_data_files checks required CSV paths.
3. DatasetStore loads notes.csv and diagnoses.csv.
4. Indexes are built and attached to app.state.
5. Endpoints serve read-only lookups and slices.

If files are missing, startup fails early with explicit instructions.

## 4. API endpoints and behavior

## 4.1 GET /note/{note_id}

Returns one note text.
If note_id is missing, returns 404.

## 4.2 GET /gt_codes/{note_id}

Returns ordered GT ICD list for note_id.
If note_id has no codes, returns 404.

## 4.3 GET /batch

Returns a slice of records with:
- note_id
- text
- gt_codes

Supports offset and size query params with bounds.

## 4.4 GET /note_ids

Returns only note_id slices for clients that do separate fetch patterns.

## 4.5 GET /health

Returns loading status and dataset metrics.

## 5. Schemas explained

All schemas are in dataset_svc/schemas.py.

## 5.1 NoteResponse

Fields:
- note_id: string key
- text: full clinical note

## 5.2 GTCodesResponse

Fields:
- note_id
- gt_codes: list of ICD labels (already ordered)

## 5.3 BatchRecord

Fields:
- note_id
- text
- gt_codes

This is the canonical per-note payload used by orchestration.

## 5.4 BatchResponse

Fields:
- batch: list[BatchRecord]
- offset
- size (actual returned size)
- total (total notes)

## 5.5 NoteIdsResponse

Fields:
- note_ids: list[str]
- offset
- size
- total

## 5.6 HealthResponse

Fields:
- status: loading or ok
- total_notes
- total_coded_notes
- loading_time_sec

## 6. Method-level walkthrough

Key methods in dataset_svc/store.py:

## 6.1 _load_notes

Purpose:
- read notes.csv safely with multiline text
- deduplicate note_id keeping first occurrence

Important detail:
- row boundary regex identifies start of each new note record

## 6.2 _flush_note_row

Purpose:
- parse one fully assembled row
- extract note_id and text
- commit into notes_index

## 6.3 _load_diagnoses

Purpose:
- load diagnoses.csv with pandas
- sort by note_id and seq_num
- group into gt_codes_index

Why sorting by seq_num matters:
- preserves clinical coding order semantics

## 6.4 get_batch

Purpose:
- return contiguous note windows for pipeline loops
- include GT labels inline for immediate downstream use

## 7. How this service connects to others

- Orchestration script starts by checking /health.
- It repeatedly calls /batch.
- note_id and gt_codes become reference identity for rewrite, ICD inference, reward, and RL rollout grouping.

## 8. Current limitations

- No built-in dataset version endpoint for strict reproducibility checks.
- Expects local files to exist and be correctly formatted.
- In-memory approach is simple and fast, but not designed for very large-scale sharded datasets.

## 9. Future improvements

1. Add dataset manifest with version/hash metadata.
2. Add strict schema validation and data quality diagnostics at startup.
3. Add optional persistent index or lightweight DB adapter for very large datasets.
