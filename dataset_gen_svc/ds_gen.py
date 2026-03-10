import os

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache/hub"
os.environ["HF_ASSETS_CACHE"] = "/workspace/hf_cache/assets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs("/workspace/hf_cache", exist_ok=True)
os.makedirs("/workspace/data", exist_ok=True)

import gc
import re
import time
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "m42-health/Llama3-Med42-8B"

INPUT_CSV = "/workspace/mimic_clean_10k_03.csv"
OUTPUT_CSV = "/workspace/data/structured_notes.csv"
CHECKPOINT_FILE = "/workspace/data/checkpoint.txt"
TEXT_COLUMN = "clean_text"

# A40 / ~50 GB GPU tuned settings
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 96
PROMPT_BATCH_SIZE = 16      # if OOM, reduce to 12 or 8
NOTE_WINDOW_SIZE = 64       # number of notes processed together
CHUNK_OVERLAP = 48
SAVE_EVERY_NOTES = 256

FIELDS = [
    "Age",
    "Gender",
    "Primary Diagnosis",
    "Secondary Diagnoses",
    "Symptoms",
    "Duration",
    "Investigations",
    "Procedures",
    "Comorbidities",
    "Risk Factors",
    "Complications",
]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

device = "cuda"
print("torch:", torch.__version__)
print("cuda device:", torch.cuda.get_device_name(0))

# =========================================================
# LOAD TOKENIZER + MODEL
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    use_fast=True,
    cache_dir="/workspace/hf_cache"
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_INPUT_TOKENS

# sdpa is fast and avoids extra dependencies
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    cache_dir="/workspace/hf_cache"
).to(device)

model.eval()

# =========================================================
# PROMPT WITH YOUR RULES
# =========================================================
PROMPT_TEMPLATE = """Return exactly these fields in this order:
Patient Demographics:
Age
Gender
Primary Diagnosis:
Secondary Diagnoses:
Symptoms:
Duration:
Investigations:
Procedures:
Comorbidities:
Risk Factors:
Complications:

RULES:
1. Extract only information explicitly mentioned in the clinical note.
2. Do NOT infer, assume or hallucinate medical information.
3. Do NOT add clinical explanations, reasoning, or commentary.
4. Do NOT paraphrase, rewrite, or summarize the clinical note. Only extract and organize relevant medical information.
5. Include only medically relevant information useful for diagnosis and clinical coding.
6. Do NOT add any extra text outside the required output format.
7. Do NOT include headings such as "Clinical Note", "Structured Format", or similar labels.
8. Follow the exact output structure provided. Do not add or remove fields.
9. If information for a field is not present in the clinical note, write: Not specified.
10. Multiple items within the same field must appear on separate lines.
11. Use clear medical terminology when listing diagnoses, symptoms, investigations, or procedures.
12. Do not repeat the same information across multiple fields.
13. Do not include explanations, justifications, or interpretation of the note.
14. Ensure the output contains ONLY the structured fields defined in the output format.
15. All fields must appear in the output exactly in the specified order, even if the value is "Not specified".
16. Extract concise medical terms or phrases instead of copying long narrative sentences from the note.
17. Symptoms must not be listed as diagnoses unless explicitly documented as a diagnosis in the note.

Clinical Note:
{note}

Structured Output:
"""

# =========================================================
# TOKEN HELPERS
# =========================================================
def fast_token_ids(text: str):
    return tokenizer.backend_tokenizer.encode(
        text,
        add_special_tokens=False
    ).ids

EMPTY_PROMPT_TOKENS = len(fast_token_ids(PROMPT_TEMPLATE.format(note="")))
NOTE_CHUNK_TOKENS = MAX_INPUT_TOKENS - EMPTY_PROMPT_TOKENS - 24

if NOTE_CHUNK_TOKENS <= 128:
    raise ValueError(f"Prompt too long. NOTE_CHUNK_TOKENS={NOTE_CHUNK_TOKENS}")

print("Prompt tokens:", EMPTY_PROMPT_TOKENS)
print("Max note chunk tokens:", NOTE_CHUNK_TOKENS)

# =========================================================
# SAFE TOKEN-LEVEL CHUNKING
# =========================================================
def split_note_into_chunks(note: str, chunk_tokens: int = NOTE_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP):
    ids = fast_token_ids(note)
    if not ids:
        return [""]

    chunks = []
    step = max(1, chunk_tokens - overlap)

    for start in range(0, len(ids), step):
        piece = ids[start:start + chunk_tokens]
        if not piece:
            continue
        text = tokenizer.decode(piece, skip_special_tokens=True)
        chunks.append(text)
        if start + chunk_tokens >= len(ids):
            break

    return chunks

# =========================================================
# OUTPUT PARSING
# =========================================================
FIELD_PATTERN = re.compile(
    r"^(Age|Gender|Primary Diagnosis|Secondary Diagnoses|Symptoms|Duration|Investigations|Procedures|Comorbidities|Risk Factors|Complications):\s*$",
    re.IGNORECASE
)

def empty_record():
    return {field: [] for field in FIELDS}

def normalize_item(text: str):
    text = text.strip().strip("-• ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def clean_generated_text(text: str):
    if "Structured Output:" in text:
        text = text.split("Structured Output:", 1)[-1]
    return text.strip()

def parse_structured_output(text: str):
    record = empty_record()
    current = None
    text = clean_generated_text(text)

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = FIELD_PATTERN.match(line)
        if m:
            current = next(f for f in FIELDS if f.lower() == m.group(1).lower())
            continue

        if current is None:
            continue

        val = normalize_item(line)
        if not val:
            continue

        low = val.lower()
        if low == "not specified":
            continue
        if low.startswith("rules"):
            continue
        if low.startswith("clinical note"):
            continue
        if low.startswith("structured output"):
            continue
        if low.startswith("patient demographics"):
            continue

        if val not in record[current]:
            record[current].append(val)

    return record

def merge_records(records):
    merged = empty_record()
    for rec in records:
        for field in FIELDS:
            for item in rec[field]:
                if item not in merged[field]:
                    merged[field].append(item)
    return merged

def format_record(record):
    lines = []
    lines.append("Patient Demographics:")
    lines.append("Age")
    if record["Age"]:
        lines.extend(record["Age"])
    else:
        lines.append("Not specified")

    lines.append("Gender")
    if record["Gender"]:
        lines.extend(record["Gender"])
    else:
        lines.append("Not specified")

    for field in [
        "Primary Diagnosis",
        "Secondary Diagnoses",
        "Symptoms",
        "Duration",
        "Investigations",
        "Procedures",
        "Comorbidities",
        "Risk Factors",
        "Complications",
    ]:
        lines.append(f"{field}:")
        if record[field]:
            lines.extend(record[field])
        else:
            lines.append("Not specified")

    return "\n".join(lines)

# =========================================================
# BATCH GENERATION
# only decode newly generated tokens
# =========================================================
def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@torch.inference_mode()
def generate_batch(prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    attention_mask = inputs["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    cleaned = []
    for i in range(outputs.shape[0]):
        gen_ids = outputs[i, input_lengths[i]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        cleaned.append(clean_generated_text(text))

    return cleaned

# =========================================================
# PROCESS MULTIPLE NOTES TOGETHER
# =========================================================
def extract_structured_window(notes_window):
    note_to_chunk_outputs = [[] for _ in range(len(notes_window))]
    all_prompts = []
    mapping = []

    for note_idx, note in enumerate(notes_window):
        chunks = split_note_into_chunks(note)
        for chunk in chunks:
            all_prompts.append(PROMPT_TEMPLATE.format(note=chunk))
            mapping.append(note_idx)

    generated_texts = []
    for prompt_batch in batched(all_prompts, PROMPT_BATCH_SIZE):
        texts = generate_batch(prompt_batch)
        generated_texts.extend(texts)

    for text, note_idx in zip(generated_texts, mapping):
        note_to_chunk_outputs[note_idx].append(text)

    final_outputs = []
    for chunk_outputs in note_to_chunk_outputs:
        parsed = [parse_structured_output(t) for t in chunk_outputs]
        merged = merge_records(parsed)
        final_outputs.append(format_record(merged))

    return final_outputs

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv(INPUT_CSV)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Column '{TEXT_COLUMN}' not found. Available columns: {list(df.columns)}")

notes = df[TEXT_COLUMN].fillna("").astype(str).tolist()
print(f"Loaded {len(notes)} notes")

# =========================================================
# RESUME SUPPORT
# =========================================================
start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        content = f.read().strip()
        if content.isdigit():
            start_idx = int(content)

print(f"Resuming from note index: {start_idx}")

if not os.path.exists(OUTPUT_CSV):
    pd.DataFrame(columns=["original_note", "optimized_note"]).to_csv(OUTPUT_CSV, index=False)

# =========================================================
# MAIN LOOP
# =========================================================
buffer_rows = []
start_time = time.time()

progress = tqdm(range(start_idx, len(notes), NOTE_WINDOW_SIZE), desc="Processing note windows")

for window_start in progress:
    window_end = min(window_start + NOTE_WINDOW_SIZE, len(notes))
    notes_window = notes[window_start:window_end]

    try:
        structured_window = extract_structured_window(notes_window)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()

        # fallback on smaller batch
        fallback_batch = max(1, PROMPT_BATCH_SIZE // 2)
        note_to_outputs = []

        for note in notes_window:
            chunks = split_note_into_chunks(note)
            prompts = [PROMPT_TEMPLATE.format(note=c) for c in chunks]
            chunk_outputs = []
            for pb in batched(prompts, fallback_batch):
                texts = generate_batch(pb)
                chunk_outputs.extend(texts)

            parsed = [parse_structured_output(t) for t in chunk_outputs]
            note_to_outputs.append(format_record(merge_records(parsed)))

        structured_window = note_to_outputs

    for original_note, optimized_note in zip(notes_window, structured_window):
        buffer_rows.append({
            "original_note": original_note,
            "optimized_note": optimized_note
        })

    if len(buffer_rows) >= SAVE_EVERY_NOTES:
        pd.DataFrame(buffer_rows).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=False,
            index=False
        )
        buffer_rows = []

        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(window_end))

        elapsed = time.time() - start_time
        done = window_end - start_idx
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(notes) - window_end) / rate if rate > 0 else 0
        progress.set_postfix({
            "notes_done": window_end,
            "notes_per_sec": f"{rate:.2f}",
            "eta_hr": f"{remaining / 3600:.2f}"
        })

        torch.cuda.empty_cache()
        gc.collect()

if buffer_rows:
    pd.DataFrame(buffer_rows).to_csv(
        OUTPUT_CSV,
        mode="a",
        header=False,
        index=False
    )

with open(CHECKPOINT_FILE, "w") as f:
    f.write(str(len(notes)))

total_elapsed = time.time() - start_time
print("Done.")
print("Saved output to:", OUTPUT_CSV)
print("Checkpoint saved to:", CHECKPOINT_FILE)
print(f"Total runtime: {total_elapsed / 3600:.2f} hours")