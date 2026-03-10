# Testing Summary — `rl_loop_svc`

**Date:** 11 March 2026  
**Branch:** `feature/rl_loop`  
**Python:** 3.10.13 (Anaconda)  
**pytest:** 9.0.2  
**Service root:** `rl_loop_svc/`

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Total tests | **42** |
| Passed | **42** |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Overall result | ✅ **ALL PASS** |
| Total execution time | ~2.05 s |

---

## 2. Test Environment

| Item | Detail |
|---|---|
| OS | macOS (darwin) |
| Python | 3.10.13 |
| Virtual environment | `rl_loop_svc/venv/` |
| PyTorch | 2.x (CPU-only build) |
| Transformers | 4.x (HuggingFace) |
| FastAPI | 0.111+ |
| pytest | 9.0.2 |
| pytest-asyncio | 1.3.0 |
| Locust | 2.43.3 |
| Stub models | `_StubPolicyModel`, `_StubReferenceModel`, `_StubValueHead` (no real weights loaded) |

---

## 3. Test Suite Structure

```
tests/
├── conftest.py                 # Shared fixtures (rollout data, tensor helpers)
├── unit/
│   ├── test_rollout_buffer.py  # 8 tests
│   ├── test_advantage.py       # 8 tests
│   ├── test_kl_divergence.py   # 8 tests
│   └── test_ppo_loss.py        # 4 tests
├── integration/
│   └── test_training_pipeline.py  # 7 tests
├── e2e/
│   └── test_full_cycle.py      # 7 tests
└── stress/
    └── locustfile.py           # Locust load-test (separate runner)
```

---

## 4. Detailed Results by Category

### 4.1 Unit Tests — `RolloutBuffer` (8 tests ✅)

| Test | Assertion | Result |
|---|---|---|
| `test_store_and_len` | Buffer length increments after store | ✅ |
| `test_multiple_stores` | 10 sequential stores → len==10 | ✅ |
| `test_build_shapes` | Output tensors all have shape (4,) | ✅ |
| `test_returns_equals_advantage_plus_value` | `returns == advantages + values` | ✅ |
| `test_clear_resets_buffer` | `clear()` resets length to 0 | ✅ |
| `test_build_prompts_preserved` | Original/rewritten prompt strings round-trip | ✅ |
| `test_negative_rewards_stored_correctly` | Negative reward (-1.0) stored without error | ✅ |
| `test_build_tensor_dtype` | Output tensors are `float32` | ✅ |

### 4.2 Unit Tests — `compute_gae` (8 tests ✅)

| Test | Assertion | Result |
|---|---|---|
| `test_output_shape` | `advantages.shape == rewards.shape` | ✅ |
| `test_output_dtype` | Output is `float32` | ✅ |
| `test_normalised_mean_near_zero` | `|mean| < 1e-5` after normalisation | ✅ |
| `test_normalised_std_near_one` | `std ≈ 1.0` after normalisation | ✅ |
| `test_single_step` | T=1: `delta = r + 0 - V`; no normalisation applied | ✅ |
| `test_high_gamma_propagates_reward` | Raw adv[0] with γ=0.99 > raw adv[0] with γ=0.01 | ✅ |
| `test_zero_rewards_zero_unadjusted` | All-zero rewards → advantages near zero | ✅ |
| `test_custom_gamma_lambda` | Different hyper-params produce different advantages | ✅ |

### 4.3 Unit Tests — `KLController` (8 tests ✅)

| Test | Assertion | Result |
|---|---|---|
| `test_compute_kl_positive_when_policy_higher` | KL > 0 when lp_policy > lp_ref | ✅ |
| `test_compute_kl_negative_when_policy_lower` | KL < 0 when lp_policy < lp_ref | ✅ |
| `test_compute_kl_zero_when_equal` | KL == 0 when both log-probs equal | ✅ |
| `test_last_kl_updated` | `last_kl` property updated after compute | ✅ |
| `test_adjust_rewards_subtracts_beta_kl` | `r_adj = r - β * KL` computed correctly | ✅ |
| `test_adjust_rewards_shape_preserved` | Output shape == input shape | ✅ |
| `test_kl_output_shape` | KL shape == (B,) | ✅ |
| `test_different_beta_scales_penalty` | Larger β → lower adjusted reward | ✅ |

### 4.4 Unit Tests — `PPOTrainer` / PPO Loss (4 tests ✅)

| Test | Assertion | Result |
|---|---|---|
| `test_update_returns_components` | All 5 loss components returned as floats | ✅ |
| `test_clipping_limits_large_ratio` | Loss is finite for very large ratio (clip active) | ✅ |
| `test_loss_finite_for_random_inputs` | `isfinite(total_loss)` for random tensors | ✅ |
| `test_kl_penalty_increases_loss` | Large KL term → higher total loss | ✅ |

### 4.5 Integration Tests — Rollout Pipeline (7 tests ✅)

| Test | Components | Result |
|---|---|---|
| `test_loader_fills_buffer` | RolloutLoader → RolloutBuffer | ✅ |
| `test_loader_new_entries_only` | Second `load_new()` returns empty | ✅ |
| `test_incremental_loading_accumulates` | New file added mid-session detected | ✅ |
| `test_advantage_computed_from_loader_data` | GAE on real loaded rewards is finite | ✅ |
| `test_buffer_build_after_loader` | Full loader → buffer → batch pipeline | ✅ |
| `test_malformed_file_skipped` | Invalid JSON file skipped with error log | ✅ |
| `test_loader_reset_allows_reload` | `reset()` allows reloading same files | ✅ |

### 4.6 End-to-End Tests — Full RL Cycle (7 tests ✅)

| Test | Components | Result |
|---|---|---|
| `test_run_once_returns_true_when_rollouts_present` | Full COLLECT→TRAIN→CKPT→IDLE | ✅ |
| `test_run_once_returns_false_when_no_new_rollouts` | IDLE skip when nothing new | ✅ |
| `test_state_returns_to_idle_after_cycle` | State machine ends in IDLE | ✅ |
| `test_training_step_incremented` | Step count equals `ppo_epochs × cycles` | ✅ |
| `test_checkpoint_created_after_cycle` | All 4 checkpoint files created | ✅ |
| `test_rollouts_loaded_count_accumulates` | `rollouts_loaded == 4` after 1 cycle | ✅ |
| `test_multiple_checkpoint_files_increment_index` | 3 cycles → `checkpoint_0001`…`checkpoint_0003` | ✅ |

### 4.7 Stress Tests — Locust (manual runner)

The Locust file at `tests/stress/locustfile.py` is designed to run against the live service.

```bash
# Start the service first (from rl_loop_svc/):
venv/bin/uvicorn app.main:app --reload

# Then in a separate terminal:
venv/bin/locust -f tests/stress/locustfile.py \
  --host http://localhost:8000 \
  --users 10 --spawn-rate 2 --run-time 60s --headless
```

Simulated scenarios:
- `GET /status` — 5× weight (heavy polling load)
- `POST /train` — 2× weight (concurrent training triggers)
- `GET /checkpoint` — 1× weight (metadata queries)

Stress tests require a running service instance and are excluded from automated `pytest` runs.

---

## 5. Bugs Found & Fixed During Testing

| # | Bug | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | `test_high_gamma_propagates_reward` failed | The assertion compared normalised advantages but normalisation inverts the expected ordering | Added `normalize: bool = True` flag to `compute_gae`; test uses `normalize=False` for raw comparison |
| 2 | All 7 e2e tests failed with `RuntimeError: element 0 of tensors does not require grad` | Stub model `forward()` returned detached constant tensors with no grad_fn; `total_loss.backward()` had nothing to differentiate | Stub model routes output through `self.linear` to attach to autograd graph |
| 3 | `test_multiple_checkpoint_files_increment_index` failed (1 checkpoint instead of 3) | Test wrote all 3 rollout files before the loop; `load_new()` loaded all 3 in the first call → 1 checkpoint | Changed test to write files one-at-a-time, interleaved with `run_once()` calls |

---

## 6. Component Coverage Summary

| Module | Category | Tests |
|---|---|---|
| `rl/rollout_buffer.py` | Unit | 8 |
| `rl/advantage.py` | Unit | 8 |
| `rl/kl_controller.py` | Unit | 8 |
| `rl/ppo_trainer.py` | Unit | 4 |
| `storage/rollout_loader.py` | Integration | 7 |
| `rl/training_loop.py` (full lifecycle) | E2E | 7 |
| `storage/checkpoint_manager.py` | E2E (implicit) | 7 |
| `rl/lifecycle_manager.py` | E2E (implicit) | 7 |
| `app/api_routes.py` | Stress (Locust) | — |

---

## 7. Service Readiness Checklist

| Item | Status |
|---|---|
| Virtual environment (`rl_loop_svc/venv/`) created | ✅ |
| All dependencies installed from `requirements.txt` | ✅ |
| `rollouts/` directory present (with `.gitkeep`) | ✅ |
| `rl_checkpoints/` directory present at project root | ✅ |
| Root `.gitignore` excludes `rl_checkpoints/` | ✅ |
| `rl_loop_svc/.gitignore` excludes `venv/`, `llm.*`, `rollouts/*.json` | ✅ |
| All 42 automated tests pass | ✅ |
| Lifecycle state machine (COLLECT→TRAIN→CKPT→IDLE) verified | ✅ |
| Checkpoint auto-pruning (`max_checkpoints=5`) implemented | ✅ |
| `scripts/setup_env.sh` ready for fresh environment setup | ✅ |
| Stress test `locustfile.py` ready for manual load testing | ✅ |
| Service startable with `uvicorn app.main:app --reload` | ✅ |
