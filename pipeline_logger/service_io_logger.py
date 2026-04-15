"""
pipeline_logger/service_io_logger.py

Writes pretty, human-readable I/O log blocks to per_service/<svc>_io.log.

Format of each block:

  ─────────────── rewriter_svc  2026-04-15 14:30:22 UTC ───────────────
    direction   : INPUT
    run_id      : 2026-04-15_143022_Hybrid_SBMi
    note_id     : note_0042
    prompt_hash : a1b2c3d4e5f6
  ──────────────────────────────────────────────────────────────────────

Thread-safe: each write() call is a single os.write to the open file
(append mode, one open per process lifetime).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_context import get_run_dir

_WIDTH = 78
_LOCK = threading.Lock()
_HANDLES: dict[str, Any] = {}  # service_name → open file handle


def _get_handle(service_name: str) -> Any | None:
    """Return (creating if needed) the open append-mode file for this service."""
    if service_name in _HANDLES:
        return _HANDLES[service_name]

    run_dir = get_run_dir()
    if run_dir is None:
        return None

    log_path: Path = run_dir / "per_service" / f"{service_name}_io.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    _HANDLES[service_name] = fh
    return fh


def _header_line(service_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    centre = f"  {service_name}  {ts}  "
    dashes = "─" * max(0, (_WIDTH - len(centre)) // 2)
    return f"{dashes}{centre}{dashes}"


def _separator() -> str:
    return "─" * _WIDTH


class ServiceIOLogger:
    """
    Lightweight pretty-logger for one microservice.

    Usage:
        _io = ServiceIOLogger("rewriter_svc")

        # at endpoint entry:
        _io.log_input(note_id="note_042", prompt_hash="a1b2c3d4")

        # at endpoint exit:
        _io.log_output(note_id="note_042", rewritten_hash="f6e5d4c3", elapsed_ms=312)
    """

    def __init__(self, service_name: str) -> None:
        self._name = service_name

    def log_input(self, **fields: Any) -> None:
        self._write("INPUT", **fields)

    def log_output(self, **fields: Any) -> None:
        self._write("OUTPUT", **fields)

    def _write(self, direction: str, **fields: Any) -> None:
        with _LOCK:
            fh = _get_handle(self._name)
            if fh is None:
                return  # no run initialised — silent no-op

            _, run_id = _get_run_context_cached()
            key_width = max((len(k) for k in fields), default=8) + 2
            lines: list[str] = [
                "",
                _header_line(self._name),
                f"  {'direction':{key_width}}: {direction}",
            ]
            if run_id:
                lines.append(f"  {'run_id':{key_width}}: {run_id}")
            for k, v in fields.items():
                lines.append(f"  {k:{key_width}}: {v}")
            lines.append(_separator())
            lines.append("")
            fh.write("\n".join(lines) + "\n")


# ── Avoid circular import by inlining the context fetch ───────────────────────


def _get_run_context_cached() -> tuple[Any, str | None]:
    from .run_context import get_run_context

    return get_run_context()
