"""
pipeline_logger/system_logger.py

Writes two artefacts — used ONLY by rl_loop_svc:

1. system_exec.csv  — one row per (prompt × epoch), all required fields.
2. batch_summaries/batch_NNN_iter_MMM.txt — ASCII box-table per epoch.

Both files live under pipeline_logs/<run_id>/.
"""

from __future__ import annotations

import csv
import io
import textwrap
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_context import get_run_dir

_CSV_FIELDS = [
    "ts",
    "run_id",
    "mode1",
    "mode2",
    "batch_num",
    "iter_num",
    "batch_size",
    "max_iters",
    "rollout_id",
    "prompt_hash",
    "rewritten_hash",
    "og_codes",
    "enh_codes",
    "gt_codes",
    "reward",
]

_LOCK = threading.Lock()
_csv_handle: Any = None
_csv_writer: Any = None


# ── CSV ───────────────────────────────────────────────────────────────────────


def _get_csv_writer():
    global _csv_handle, _csv_writer
    if _csv_writer is not None:
        return _csv_writer

    run_dir = get_run_dir()
    if run_dir is None:
        return None

    csv_path: Path = run_dir / "system_exec.csv"
    needs_header = not csv_path.exists()
    _csv_handle = open(csv_path, "a", encoding="utf-8", newline="")
    _csv_writer = csv.DictWriter(
        _csv_handle, fieldnames=_CSV_FIELDS, extrasaction="ignore"
    )
    if needs_header:
        _csv_writer.writeheader()
        _csv_handle.flush()
    return _csv_writer


def write_csv_rows(rows: list[dict]) -> None:
    """Append one CSV row per prompt entry.  Called once per epoch."""
    with _LOCK:
        writer = _get_csv_writer()
        if writer is None:
            return
        for row in rows:
            # Serialise list fields as semicolon-separated strings so cells are
            # clean in any spreadsheet (no Python repr brackets).
            serialised = dict(row)
            for field in ("og_codes", "enh_codes", "gt_codes"):
                val = serialised.get(field, [])
                serialised[field] = (
                    "; ".join(val) if isinstance(val, list) else str(val)
                )
            writer.writerow(serialised)
        _csv_handle.flush()


# ── ASCII table ───────────────────────────────────────────────────────────────

_BOX_W = 90  # total width of the ASCII box


def _box_top(text: str) -> str:
    inner = f" {text} "
    pad = max(0, _BOX_W - 2 - len(inner))
    return "╔" + inner + "═" * pad + "╗"


def _box_bottom() -> str:
    return "╚" + "═" * (_BOX_W - 2) + "╝"


def _box_row(text: str) -> str:
    inner = f" {text}"
    pad = max(0, _BOX_W - 2 - len(inner))
    return "║" + inner + " " * pad + "║"


def _box_divider() -> str:
    return "╠" + "═" * (_BOX_W - 2) + "╣"


def _codes_str(codes: list[str], width: int = 24) -> str:
    joined = ", ".join(codes) if codes else "—"
    return (joined[: width - 1] + "…") if len(joined) > width else joined.ljust(width)


def _reward_str(r: float) -> str:
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.4f}"


def write_batch_summary(
    *,
    run_id: str,
    mode1: str,
    mode2: str,
    batch_num: int,
    iter_num: int,
    max_iters: int,
    batch_size: int,
    entries: list[dict],
) -> None:
    """
    Write a human-readable ASCII table for one (batch, epoch) pair.

    entries: list of dicts with keys:
        prompt_hash, rewritten_hash, og_codes, enh_codes, gt_codes, reward
    """
    run_dir = get_run_dir()
    if run_dir is None:
        return

    fname = f"batch_{batch_num:03d}_iter_{iter_num:03d}.txt"
    out_path: Path = run_dir / "batch_summaries" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rewards = [e["reward"] for e in entries]
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    min_r = min(rewards) if rewards else 0.0
    max_r = max(rewards) if rewards else 0.0

    # Column widths
    ph_w = 14  # prompt_hash display
    rh_w = 14  # rewritten_hash display
    og_w = 20
    enh_w = 22
    gt_w = 22
    rw_w = 9

    col_header = (
        f" {'prompt':^{ph_w}} │ {'rewritten':^{rh_w}} │"
        f" {'og_codes':^{og_w}} │ {'enh_codes':^{enh_w}} │"
        f" {'gt_codes':^{gt_w}} │ {'reward':^{rw_w}} "
    )
    col_sep = (
        "─" * (ph_w + 2)
        + "┼"
        + "─" * (rh_w + 2)
        + "┼"
        + "─" * (og_w + 2)
        + "┼"
        + "─" * (enh_w + 2)
        + "┼"
        + "─" * (gt_w + 2)
        + "┼"
        + "─" * (rw_w + 2)
    )

    lines: list[str] = [
        _box_top(
            f"BATCH {batch_num:03d}  ·  ITER {iter_num:03d}/{max_iters:03d}  ·  {mode1} / {mode2}"
        ),
        _box_row(f"{ts}  ·  run_id: {run_id}"),
        _box_row(f"batch_size: {batch_size}  ·  prompts_this_epoch: {len(entries)}"),
        _box_divider(),
        _box_row(col_header),
        _box_row(col_sep),
    ]

    for e in entries:
        ph = str(e.get("prompt_hash", ""))[:ph_w].ljust(ph_w)
        rh = str(e.get("rewritten_hash", ""))[:rh_w].ljust(rh_w)
        og = _codes_str(e.get("og_codes", []), og_w)
        enh = _codes_str(e.get("enh_codes", []), enh_w)
        gt = _codes_str(e.get("gt_codes", []), gt_w)
        rw = _reward_str(e["reward"]).rjust(rw_w)

        row = f" {ph} │ {rh} │ {og} │ {enh} │ {gt} │ {rw} "
        lines.append(_box_row(row))

    lines.append(_box_divider())
    summary_text = (
        f"SUMMARY  "
        f"mean_reward={_reward_str(mean_r)}   "
        f"min={_reward_str(min_r)}   "
        f"max={_reward_str(max_r)}   "
        f"n={len(entries)}"
    )
    lines.append(_box_row(summary_text))
    lines.append(_box_bottom())
    lines.append("")

    with _LOCK:
        out_path.write_text("\n".join(lines), encoding="utf-8")
