# Generation Summary

## Microservice: Reward Metrics

- **Goal:** Compute a scalar reward in [-1.0, 1.0] using ICD-10 code evaluations and tree structure, for RL input.
- **Inputs:**
  - ICD-10 tree structure dataset (JSON)
  - Ground truth ICD-10 codes for a doctor's note (from ./gt_codes/)
  - Enhanced ICD-10 codes from Task LLM (via API POST)
  - Non-enhanced ICD-10 codes from Task LLM (via API POST)
- **Output:**
  - Scalar reward value in [-1.0, 1.0] (POSTed to RL block endpoint)
- **Core Logic:**
  - Implements a semantic distance metric for ICD-10 codes using the tree structure (networkx graph).
  - Computes reward based on the rules described in llm.json.
- **API:**
  - POST /reward: Accepts codes, computes and returns reward.
  - Forwards reward to RL block endpoint.
- **Testing:**
  - Unit tests for distance metric and reward logic.
  - Integration tests for API and end-to-end flow.
- **Tech Stack:**
  - Python (FastAPI)
  - networkx
  - pytest
- **Directory Structure:**
  - ./gt_codes/
  - icd10_tree.json
  - main.py
  - test/
  - README.md
  - llm.json, llm.txt

---

This summary was generated based on the requirements and design in llm.json and llm.txt.
