"""
storage/rollout_loader.py — Incremental JSONL reader with offset tracking.

Design goals
────────────
• Never re-read a line that was already consumed (offset-based).
• Survive process restarts via seen_files.json + segment_offsets.json.
• Skip and log corrupted JSON lines rather than crashing.
• Thread-safe append (file is opened in append mode per write call).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Generator, List

from config import cfg
from schemas.rollout_schema import Rollout
from utils.logging import get_logger

log = get_logger("storage.loader")


# ── Writer ────────────────────────────────────────────────────────────────────


def append_rollout(rollout: Rollout, filepath: Path) -> None:
    """Append a single rollout as one JSON line to *filepath*."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rollout.to_dict()) + "\n")
    log.debug(
        "Appended rollout to JSONL",
        extra={"rollout_id": rollout.rollout_id, "file": str(filepath)},
    )


def new_rollout_filepath() -> Path:
    """Return a timestamped JSONL path inside rollouts_dir."""
    ts = int(time.time())
    return cfg.rollouts_dir / f"rollouts_{ts}.jsonl"


# ── Offset persistence ────────────────────────────────────────────────────────


def _load_json_state(path: Path) -> dict:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            log.warning(
                "State file corrupted — starting fresh", extra={"path": str(path)}
            )
    return {}


def _save_json_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)  # atomic on POSIX


# ── RolloutLoader ─────────────────────────────────────────────────────────────


class RolloutLoader:
    """
    Read rollout JSONL files incrementally, tracking byte offsets so that
    restart does not re-read already-processed entries.
    """

    def __init__(self) -> None:
        self._seen: dict[str, bool] = _load_json_state(cfg.seen_files_path)
        self._offsets: dict[str, int] = _load_json_state(cfg.segment_offsets_path)
        log.info(
            "RolloutLoader initialised",
            extra={
                "seen_files": len(self._seen),
                "tracked_offsets": len(self._offsets),
            },
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_new(self) -> List[Rollout]:
        """Return all rollouts not yet seen, across all JSONL files in rollouts_dir."""
        rollouts: List[Rollout] = []
        for path in sorted(cfg.rollouts_dir.glob("rollouts_*.jsonl")):
            rollouts.extend(self._read_incremental(path))
        self._persist_state()
        log.info("Load complete", extra={"new_rollouts": len(rollouts)})
        return rollouts

    def mark_file_complete(self, filepath: Path) -> None:
        self._seen[str(filepath)] = True
        self._persist_state()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_incremental(self, path: Path) -> List[Rollout]:
        key = str(path)
        offset = self._offsets.get(key, 0)
        results: List[Rollout] = []

        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            log.error("File disappeared mid-scan", extra={"file": key})
            return results

        if offset >= file_size:
            return results  # nothing new

        with path.open("r", encoding="utf-8") as fh:
            fh.seek(offset)
            for raw_line in fh:
                offset += len(raw_line.encode("utf-8"))
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rollout = Rollout.from_dict(data)
                    results.append(rollout)
                except (json.JSONDecodeError, Exception) as exc:
                    log.error(
                        "Skipping corrupted JSONL line",
                        extra={
                            "file": key,
                            "error": str(exc),
                            "line_preview": line[:80],
                        },
                    )
            self._offsets[key] = offset

        if results:
            log.debug(
                "Read from JSONL",
                extra={
                    "file": path.name,
                    "new_entries": len(results),
                    "offset": offset,
                },
            )
        return results

    def _persist_state(self) -> None:
        _save_json_state(cfg.seen_files_path, self._seen)
        _save_json_state(cfg.segment_offsets_path, self._offsets)
