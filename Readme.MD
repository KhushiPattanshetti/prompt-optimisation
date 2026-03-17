# Prompt Rewriter SFT (Phi-3 Mini)

## Overview
This branch implements the Prompt Rewriter module using Supervised Fine-Tuning (SFT) as part of the Prompt Optimisation project.

The objective of this module is to fine-tune a Phi-3 Mini model to transform unstructured clinical notes into a standardized structured clinical format. This serves as the foundational learning stage before applying reinforcement learning or downstream ICD coding tasks.

---

## Objective
The model is designed to:

- Accept raw clinical notes as input  
- Extract only explicitly mentioned medical information  
- Convert the extracted information into a fixed structured format  
- Avoid hallucination, inference, summarization, or additional explanation  

---

## Model Details

| Component            | Description                              |
|---------------------|------------------------------------------|
| Base Model          | microsoft/Phi-3-mini-4k-instruct         |
| Training Method     | Supervised Fine-Tuning (SFT)             |
| PEFT Technique      | LoRA / QLoRA                             |
| Framework           | Hugging Face Transformers and PEFT       |

---

## Input Format

The model accepts a clinical note in JSON format:

```json
{
  "clinical_note": "Patient is a 58-year-old male with diabetes and chest pain for 2 days."
}
```

## Output Format

```json
{
Patient Demographics:
Age: 58
Gender: Male

Primary Diagnosis:
Not specified

Secondary Diagnoses:
Diabetes

Symptoms:
Chest pain

Duration:
2 days

Investigations:
Not specified

Procedures:
Not specified

Comorbidities:
Diabetes

Risk Factors:
Not specified

Complications:
Not specified
}
```

## Repo structure

```json
{
prompt-optimisation/
│
├── rewriter_sft_svc/
│   ├── app.py
│   ├── train_sft.py
│   ├── model_trainer.py
│   ├── test_runner.py
│   ├── config.py
│   └── requirements.txt
│
├── sft_checkpoints/
│   └── phi3_rewriter_sft/
│
├── llm.txt
├── .gitignore
└── README.md
}
```
