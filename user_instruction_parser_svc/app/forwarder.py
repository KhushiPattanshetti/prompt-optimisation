"""
forwarder.py
Forwards the validated clinical note to the Prompt Rewriter Block (Phi-3 Mini SLM)

- Uses real HTTP call to rewriter_sft_svc
- Falls back to simulation if URL is not set
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

# ✅ IMPORTANT: local development URL (not Docker)
PROMPT_REWRITER_URL = os.environ.get(
    "PROMPT_REWRITER_URL",
    "http://127.0.0.1:8000/rewrite"
)

TIMEOUT_SECONDS = int(os.environ.get("PROMPT_REWRITER_TIMEOUT", "300"))


def forward_to_prompt_rewriter(
    filename: str,
    clinical_note: str,
    instruction_id: str,
) -> dict:
    """
    Forward clinical note to rewriter_sft_svc

    Returns:
        {
            forwarded: bool,
            destination: str,
            instruction_id: str,
            simulated: bool,
            response: dict (if success),
            error: str (if failure)
        }
    """

    payload = {
        "instruction_id": instruction_id,
        "filename": filename or "uploaded_note.txt",
        "clinical_note": clinical_note or "",
    }

    # ── Simulation mode ───────────────────────────────────────────────
    if not PROMPT_REWRITER_URL:
        logger.info("[FORWARDER] Simulation mode (no URL set)")
        return {
            "forwarded": True,
            "destination": "simulated",
            "instruction_id": instruction_id,
            "simulated": True,
            "payload_preview": {
                "filename": filename,
                "note_length": len(clinical_note),
                "instruction_id": instruction_id,
            },
        }

    # ── Real HTTP call ───────────────────────────────────────────────
    try:
        logger.info("[FORWARDER] Calling rewriter at %s", PROMPT_REWRITER_URL)

        response = requests.post(
            PROMPT_REWRITER_URL,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            logger.error("[FORWARDER] Response not JSON")
            return {
                "forwarded": False,
                "destination": PROMPT_REWRITER_URL,
                "instruction_id": instruction_id,
                "simulated": False,
                "error": "Invalid JSON response from rewriter",
                "raw_response": response.text,
            }

        logger.info("[FORWARDER] Successfully forwarded")

        return {
            "forwarded": True,
            "destination": PROMPT_REWRITER_URL,
            "instruction_id": instruction_id,
            "simulated": False,
            "response": data,
        }

    except requests.Timeout:
        logger.exception("[FORWARDER] Timeout error")
        return {
            "forwarded": False,
            "destination": PROMPT_REWRITER_URL,
            "instruction_id": instruction_id,
            "simulated": False,
            "error": f"Timeout after {TIMEOUT_SECONDS}s",
        }

    except requests.RequestException as e:
        logger.exception("[FORWARDER] Request failed")
        return {
            "forwarded": False,
            "destination": PROMPT_REWRITER_URL,
            "instruction_id": instruction_id,
            "simulated": False,
            "error": str(e),
        }
