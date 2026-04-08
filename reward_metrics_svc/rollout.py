"""
rollout.py – Rollout queue system for the PPO training loop.  spec §16

Directory layout
----------------
rollouts/
  pending/    ← saved immediately after reward computation
  acked/      ← moved here after RL service confirms receipt
  failed/     ← moved here when RL service is unreachable / errors

Public API
----------
enqueue_rollout(note_id, state, action, reward,
                log_prob_old, value_estimate) → None

Flow
----
1. Compute reward (caller's responsibility)
2. Save rollout JSON to pending/
3. Attempt POST to RL service URL
4. On HTTP 200 → move file to acked/
5. On any other result → move file to failed/
"""

import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from typing import Optional

from config import RL_SERVICE_URL, ROLLOUT_DIR

logger = logging.getLogger("reward_metrics_svc.rollout")

# ─────────────────────────────────────────────────────────────────────────────
# Directory bootstrap
# ─────────────────────────────────────────────────────────────────────────────

_PENDING = os.path.join(ROLLOUT_DIR, "pending")
_ACKED = os.path.join(ROLLOUT_DIR, "acked")
_FAILED = os.path.join(ROLLOUT_DIR, "failed")


def _ensure_dirs() -> None:
    for d in (_PENDING, _ACKED, _FAILED):
        os.makedirs(d, exist_ok=True)


_ensure_dirs()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def enqueue_rollout(
    note_id: str,
    state: str,
    action: str,
    reward: float,
    log_prob_old: float,
    value_estimate: float,
) -> None:
    """
    Persist a rollout to disk and attempt to deliver it to the RL service.

    Parameters mirror the spec §16 rollout payload.
    """
    payload = {
        "note_id": note_id,
        "state": state,
        "action": action,
        "reward": reward,
        "log_prob_old": log_prob_old,
        "value_estimate": value_estimate,
    }

    # ── Step 1: save to pending/ ──────────────────────────────────────────────
    filename = f"{note_id}.json"
    pending_path = os.path.join(_PENDING, filename)
    with open(pending_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.debug(
        "ROLLOUT | saved to pending | note_id=%s  path=%s", note_id, pending_path
    )

    # ── Step 2: attempt POST to RL service ───────────────────────────────────
    _post_and_settle(pending_path, filename, payload)


def _post_and_settle(pending_path: str, filename: str, payload: dict) -> None:
    """POST payload to RL service; move file to acked/ or failed/."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RL_SERVICE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                _move(pending_path, _ACKED, filename)
                logger.info(
                    "ROLLOUT | acked  | note_id=%s  rl_url=%s",
                    payload["note_id"],
                    RL_SERVICE_URL,
                )
            else:
                _move(pending_path, _FAILED, filename)
                logger.warning(
                    "ROLLOUT | failed | note_id=%s  http_status=%d",
                    payload["note_id"],
                    resp.status,
                )
    except (urllib.error.URLError, OSError) as exc:
        _move(pending_path, _FAILED, filename)
        logger.warning(
            "ROLLOUT | failed | note_id=%s  error=%s", payload["note_id"], exc
        )


def _move(src: str, dest_dir: str, filename: str) -> None:
    dest = os.path.join(dest_dir, filename)
    shutil.move(src, dest)
    logger.debug("ROLLOUT | moved %s → %s", src, dest)


# ─────────────────────────────────────────────────────────────────────────────
# Utils (useful for tests / monitoring)
# ─────────────────────────────────────────────────────────────────────────────


def counts() -> dict:
    """Return {'pending': n, 'acked': n, 'failed': n}."""
    return {
        "pending": len(os.listdir(_PENDING)),
        "acked": len(os.listdir(_ACKED)),
        "failed": len(os.listdir(_FAILED)),
    }
