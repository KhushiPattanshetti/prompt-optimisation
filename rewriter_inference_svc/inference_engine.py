"""Inference engine for the rewriter inference service.
 
FIXES APPLIED:
    - load_model() now returns (model, tokenizer, value_head) — updated all callers
    - value_estimate now uses the explicit value_head instead of fragile
      attribute lookup (was trying model.value_head / model.v_head / model.score)
    - _compute_value_estimate now takes value_head as an explicit argument
    - threading.Lock scope tightened: only wraps GPU calls, not file I/O,
      so /health endpoint is not blocked
"""
 
from __future__ import annotations
 
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
 
from config import MAX_NEW_TOKENS, OUTPUT_PATH, TEMPERATURE, DO_SAMPLE
from logger import get_logger
from model_loader import load_model
 
log = get_logger(__name__)
 
# Serialises all GPU inference calls to prevent CUDA OOM under concurrency.
# Scoped tightly — only wraps model.generate() and forward passes.
_inference_lock = threading.Lock()
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _get_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device
 
 
def _compute_log_prob(
    model: PreTrainedModel,
    full_input_ids: torch.Tensor,
    input_length: int,
) -> float:
    """Compute log π_old(a|s) for the generated token sequence.
 
    Procedure:
        1. Forward pass on [input | generated] tokens.
        2. log-softmax over vocab dimension.
        3. Gather log-prob for each generated token.
        4. Sum to scalar.
 
    Args:
        model:           Loaded causal LM.
        full_input_ids:  Shape (1, input_len + gen_len).
        input_length:    Number of input-prompt tokens.
 
    Returns:
        Scalar log probability (float).
    """
    outputs = model(full_input_ids)
    logits  = outputs.logits  # (1, seq_len, vocab_size)
 
    log_probs = F.log_softmax(logits, dim=-1)
 
    # Generated tokens start at index `input_length`
    generated_token_ids = full_input_ids[:, input_length:]          # (1, gen_len)
    prediction_logits   = log_probs[:, input_length - 1 : -1, :]   # (1, gen_len, V)
 
    token_log_probs = prediction_logits.gather(
        dim=-1, index=generated_token_ids.unsqueeze(-1)
    ).squeeze(-1)  # (1, gen_len)
 
    return token_log_probs.sum().item()
 
 
def _compute_value_estimate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    value_head: nn.Linear,
) -> float:
    """Extract V(s) using the explicit value head.
 
    FIX: Previously used fragile attribute lookup on the model
    (model.value_head / model.v_head / model.score) which always
    fell through to the meaningless proxy on a raw base model.
    Now uses the explicit nn.Linear value_head passed as argument.
 
    Args:
        model:      Loaded causal LM.
        input_ids:  Shape (1, seq_len).
        value_head: nn.Linear(hidden_size, 1) attached at load time.
 
    Returns:
        Scalar value estimate (float).
    """
    outputs     = model(input_ids, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    # Use the last token position as the state representation
    state_repr  = last_hidden[:, -1, :]     # (1, hidden_dim)
    value       = value_head(state_repr)    # (1, 1)
    return value.squeeze().item()
 
 
def _save_output(result: Dict[str, Any]) -> Path:
    """Persist inference output to disk as a timestamped JSON file."""
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    filename  = timestamp.replace(":", "-") + ".json"
    filepath  = output_dir / filename
 
    payload = {
        "timestamp":        timestamp,
        "input_note":       result["input_note"],
        "rewritten_prompt": result["rewritten_prompt"],
        "log_prob_old":     result["log_prob_old"],
        "value_estimate":   result["value_estimate"],
    }
 
    filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("output_saved | path=%s", filepath)
    return filepath
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def run_inference(clinical_note: str) -> Dict[str, Any]:
    """Run the full inference pipeline for a given clinical note.
 
    Steps:
        1. Load model, tokenizer, and value head (cached).
        2. Tokenize the clinical note.
        3. Generate rewritten prompt via model.generate().
        4. Compute log_prob_old — log π_old(a|s).
        5. Compute value_estimate — V(s) via explicit value head.
        6. Save output to disk.
        7. Return results.
 
    All GPU operations run inside torch.no_grad() and are serialised
    by _inference_lock to prevent CUDA OOM under concurrent requests.
 
    Args:
        clinical_note: Raw clinical note text.
 
    Returns:
        Dict with keys: rewritten_prompt, log_prob_old, value_estimate.
    """
    t_start = time.perf_counter()
 
    # load_model returns (model, tokenizer, value_head)
    model, tokenizer, value_head = load_model()
 
    with _inference_lock:
        device = _get_device(model)
 
        with torch.no_grad():
            # Tokenize
            inputs       = tokenizer(clinical_note, return_tensors="pt").to(device)
            input_ids    = inputs["input_ids"]
            input_length = input_ids.shape[1]
 
            # Generate rewritten prompt
            gen_output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = gen_output  # (1, input_len + gen_len)
 
            # Decode only the newly generated tokens
            new_token_ids     = generated_ids[:, input_length:]
            rewritten_prompt: str = tokenizer.decode(
                new_token_ids[0], skip_special_tokens=True
            )
 
            # Compute log_prob_old
            log_prob_old: float = _compute_log_prob(
                model, generated_ids, input_length
            )
 
            # Compute value_estimate using the explicit value head
            value_estimate: float = _compute_value_estimate(
                model, input_ids, value_head
            )
 
    generation_time = time.perf_counter() - t_start
    log.info("generation_time | seconds=%.4f", generation_time)
 
    result: Dict[str, Any] = {
        "input_note":       clinical_note,
        "rewritten_prompt": rewritten_prompt,
        "log_prob_old":     log_prob_old,
        "value_estimate":   value_estimate,
    }
 
    # Save to disk (outside the GPU lock — pure I/O)
    _save_output(result)
 
    return {
        "rewritten_prompt": rewritten_prompt,
        "log_prob_old":     log_prob_old,
        "value_estimate":   value_estimate,
    }

# """Inference engine for the rewriter inference service.

# Responsible for:
# - Text generation
# - Log probability computation
# - Value estimation
# - Saving inference outputs to disk
# """

# from __future__ import annotations

# import json
# import time
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any, Dict

# import torch
# import torch.nn.functional as F
# from transformers import PreTrainedModel, PreTrainedTokenizerBase

# from config import MAX_NEW_TOKENS, OUTPUT_PATH, TEMPERATURE, DO_SAMPLE
# from logger import get_logger
# from model_loader import load_model

# log = get_logger(__name__)


# def _get_device(model: PreTrainedModel) -> torch.device:
#     """Return the device the model parameters live on."""
#     return next(model.parameters()).device


# def _compute_log_prob(
#     model: PreTrainedModel,
#     full_input_ids: torch.Tensor,
#     input_length: int,
# ) -> float:
#     """Compute the log probability of the generated tokens.

#     Implements log π_old(a|s):
#         1. Forward pass on the full sequence (input + generated).
#         2. log-softmax over the vocabulary dimension.
#         3. Gather the log-prob for each *generated* token.
#         4. Sum to obtain the scalar log_prob_old.

#     Args:
#         model: The loaded causal LM.
#         full_input_ids: Tensor of shape (1, seq_len) containing
#             input tokens concatenated with generated tokens.
#         input_length: Number of tokens that belong to the input prompt.

#     Returns:
#         Scalar log probability (float).
#     """
#     outputs = model(full_input_ids)
#     logits = outputs.logits  # (1, seq_len, vocab_size)

#     log_probs = F.log_softmax(logits, dim=-1)  # (1, seq_len, vocab_size)

#     # For each generated position t, the prediction comes from logits at t-1
#     # Generated tokens start at index `input_length`
#     generated_token_ids = full_input_ids[:, input_length:]  # (1, gen_len)
#     # Corresponding logit predictions are at positions [input_length-1 .. -2]
#     prediction_logits = log_probs[:, input_length - 1 : -1, :]  # (1, gen_len, V)

#     token_log_probs = prediction_logits.gather(
#         dim=-1, index=generated_token_ids.unsqueeze(-1)
#     ).squeeze(-1)  # (1, gen_len)

#     total_log_prob: float = token_log_probs.sum().item()
#     return total_log_prob


# def _compute_value_estimate(
#     model: PreTrainedModel,
#     input_ids: torch.Tensor,
# ) -> float:
#     """Extract V(s) from the value head attached to the actor model.

#     The value head is expected to be available as ``model.value_head``
#     or via a ``score`` / ``v_head`` attribute depending on how the
#     checkpoint was saved during SFT/RL training.

#     Falls back to using the mean of the last hidden state projected
#     through any available value head layer.

#     Args:
#         model: The loaded model with a value head.
#         input_ids: Tokenised input (1, seq_len).

#     Returns:
#         Scalar value estimate (float).
#     """
#     outputs = model(input_ids, output_hidden_states=True)
#     last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)

#     # Try known value-head attribute names
#     for attr in ("value_head", "v_head", "score"):
#         head = getattr(model, attr, None)
#         if head is not None:
#             value = head(last_hidden[:, -1, :])  # (1, 1) or (1,)
#             return value.squeeze().item()

#     # Fallback: use the mean-pooled hidden state norm as a proxy
#     log.warning("No explicit value head found; using mean hidden-state norm as proxy.")
#     value_proxy = last_hidden[:, -1, :].mean().item()
#     return value_proxy


# def _save_output(result: Dict[str, Any]) -> Path:
#     """Persist inference output to disk as a timestamped JSON file.

#     Args:
#         result: Dictionary containing inference results.

#     Returns:
#         Path to the saved file.
#     """
#     output_dir = Path(OUTPUT_PATH)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
#     # Use a filename-safe version of the timestamp
#     filename = timestamp.replace(":", "-") + ".json"
#     filepath = output_dir / filename

#     payload = {
#         "timestamp": timestamp,
#         "input_note": result["input_note"],
#         "rewritten_prompt": result["rewritten_prompt"],
#         "log_prob_old": result["log_prob_old"],
#         "value_estimate": result["value_estimate"],
#     }

#     filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
#     log.info("output_saved | path=%s", filepath)
#     return filepath


# def run_inference(clinical_note: str) -> Dict[str, Any]:
#     """Run the full inference pipeline for a given clinical note.

#     Steps:
#         1. Load model & tokenizer (cached).
#         2. Tokenize the clinical note.
#         3. Generate rewritten prompt via model.generate().
#         4. Compute log_prob_old (log π_old(a|s)).
#         5. Compute value_estimate (V(s)).
#         6. Save output to disk.
#         7. Return results.

#     All inference runs inside ``torch.no_grad()``.

#     Args:
#         clinical_note: Raw clinical note text.

#     Returns:
#         Dictionary with keys: rewritten_prompt, log_prob_old, value_estimate.
#     """
#     t_start = time.perf_counter()

#     model, tokenizer = load_model()
#     device = _get_device(model)

#     with torch.no_grad():
#         # Step 2: Tokenize input
#         inputs = tokenizer(clinical_note, return_tensors="pt").to(device)
#         input_ids = inputs["input_ids"]
#         input_length = input_ids.shape[1]

#         # Step 3: Generate rewritten prompt
#         gen_output = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=TEMPERATURE,
#             do_sample=DO_SAMPLE,
#         )
#         generated_ids = gen_output  # (1, input_len + gen_len)

#         # Decode only the newly generated tokens
#         new_token_ids = generated_ids[:, input_length:]
#         rewritten_prompt: str = tokenizer.decode(
#             new_token_ids[0], skip_special_tokens=True
#         )

#         # Step 4-6: Compute log_prob_old
#         log_prob_old: float = _compute_log_prob(model, generated_ids, input_length)

#         # Step 7: Compute value_estimate
#         value_estimate: float = _compute_value_estimate(model, input_ids)

#     generation_time = time.perf_counter() - t_start
#     log.info("generation_time | seconds=%.4f", generation_time)

#     result: Dict[str, Any] = {
#         "input_note": clinical_note,
#         "rewritten_prompt": rewritten_prompt,
#         "log_prob_old": log_prob_old,
#         "value_estimate": value_estimate,
#     }

#     # Step 8: Save to disk
#     _save_output(result)

#     return {
#         "rewritten_prompt": rewritten_prompt,
#         "log_prob_old": log_prob_old,
#         "value_estimate": value_estimate,
#     }
