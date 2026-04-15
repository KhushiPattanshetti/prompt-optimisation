import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..app.config import settings
from ..schemas.rollout_schema import RolloutEntry, RolloutFile

logger = logging.getLogger(__name__)

_SEEN_INDEX_FILENAME = ".seen_files"
_SEGMENT_OFFSETS_FILENAME = ".segment_offsets.json"
_ICD_REGEX = re.compile(r"\b[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?\b")
_ROLLOUT_FILE_GLOB = "rollout_batch_*.json"
_ROLLOUT_SEGMENT_GLOB = "rollout_segment_*.jsonl"

# trajectory_store_svc writes JSONL files with this naming pattern and bare JSON lines
_TRAJ_STORE_GLOB = "rollouts_*.jsonl"


def _map_traj_store_line(data: dict) -> Optional[RolloutEntry]:
    """
    Map a single trajectory_store_svc rollout line to a RolloutEntry.

    trajectory_store_svc uses 'prompt' for the original prompt field; RolloutEntry
    uses 'original_prompt'.  group_id and sample_weight are already set by the
    upstream preprocessing step in trajectory_store_svc.
    """
    try:
        return RolloutEntry(
            rollout_id=data.get("rollout_id"),
            group_id=data.get("group_id"),
            original_prompt=data["prompt"],
            rewritten_prompt=data["rewritten_prompt"],
            reward=data["reward"],
            log_prob_old=data["log_prob_old"],
            value_estimate=data.get("value_estimate"),
            sample_weight=data.get("sample_weight", 1.0),
        )
    except Exception as exc:
        logger.warning(
            "Failed to map trajectory_store_svc rollout line: %s | data=%s", exc, data
        )
        return None


class RolloutLoader:
    def __init__(
        self, rollouts_dir: Path, traj_store_dir: Optional[Path] = None
    ) -> None:
        self.rollouts_dir = Path(rollouts_dir)
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self._seen_index = self.rollouts_dir / _SEEN_INDEX_FILENAME
        self._segment_offsets_path = self.rollouts_dir / _SEGMENT_OFFSETS_FILENAME
        self._loaded_files: Set[str] = self._load_seen_index()
        self._segment_offsets: Dict[str, int] = self._load_segment_offsets()

        # Optional directory for trajectory_store_svc JSONL rollout files.
        # When set, load_new() also reads bare-JSONL files produced by that service.
        self._traj_store_dir: Optional[Path] = (
            Path(traj_store_dir) if traj_store_dir else None
        )
        if self._traj_store_dir:
            logger.info(
                "RolloutLoader: trajectory_store_svc integration enabled | dir=%s",
                self._traj_store_dir,
            )

    def load_all(self) -> List[RolloutEntry]:
        entries: List[RolloutEntry] = []
        for path in sorted(self.rollouts_dir.glob(_ROLLOUT_FILE_GLOB)):
            entries.extend(self._parse_file(path))

        for path in sorted(self.rollouts_dir.glob(_ROLLOUT_SEGMENT_GLOB)):
            batch, _ = self._parse_segment(path, start_offset=0)
            entries.extend(batch)

        logger.info(
            "Loaded %d rollout entries from %s", len(entries), self.rollouts_dir
        )
        return entries

    def load_new(self) -> List[RolloutEntry]:
        entries: List[RolloutEntry] = []
        newly_seen: List[str] = []

        for path in sorted(self.rollouts_dir.glob(_ROLLOUT_FILE_GLOB)):
            if path.name in self._loaded_files:
                continue
            batch = self._parse_file(path)
            entries.extend(batch)
            newly_seen.append(path.name)

        if newly_seen:
            self._loaded_files.update(newly_seen)
            self._persist_seen_index()
            logger.info(
                "Loaded %d new rollout entries from %d new files",
                len(entries),
                len(newly_seen),
            )

        segment_updates = False
        for path in sorted(self.rollouts_dir.glob(_ROLLOUT_SEGMENT_GLOB)):
            start_offset = self._segment_offsets.get(path.name, 0)
            batch, end_offset = self._parse_segment(path, start_offset=start_offset)
            entries.extend(batch)
            if end_offset != start_offset:
                self._segment_offsets[path.name] = end_offset
                segment_updates = True

        if segment_updates:
            self._persist_segment_offsets()

        if segment_updates and entries:
            logger.info(
                "Loaded %d new rollout entries from append-only segments", len(entries)
            )

        # Also read from trajectory_store_svc rollouts directory if configured
        if self._traj_store_dir and self._traj_store_dir.exists():
            traj_entries = self._load_traj_store_new()
            if traj_entries:
                logger.info(
                    "Loaded %d new rollout entries from trajectory_store_svc at %s",
                    len(traj_entries),
                    self._traj_store_dir,
                )
            entries.extend(traj_entries)

        return entries

    def _load_traj_store_new(self) -> List[RolloutEntry]:
        """
        Read new rollouts from trajectory_store_svc's JSONL store incrementally.

        trajectory_store_svc writes bare JSON lines (one Rollout object per line)
        to rollouts_*.jsonl files.  Offset state is shared with the main segment
        offset store, keyed by the full file name so there's no collision.
        """
        entries: List[RolloutEntry] = []
        path = self._traj_store_dir
        if path is None:
            return entries

        updated = False
        for jsonl_file in sorted(path.glob(_TRAJ_STORE_GLOB)):
            key = f"traj_store:{jsonl_file.name}"
            start_offset = self._segment_offsets.get(key, 0)
            batch, end_offset = self._parse_traj_store_segment(jsonl_file, start_offset)
            entries.extend(batch)
            if end_offset != start_offset:
                self._segment_offsets[key] = end_offset
                updated = True

        if updated:
            self._persist_segment_offsets()

        return entries

    def _parse_traj_store_segment(
        self, path: Path, start_offset: int
    ) -> Tuple[List[RolloutEntry], int]:
        """
        Read trajectory_store_svc JSONL lines incrementally from start_offset.

        Each line is a bare JSON object with fields from trajectory_store_svc's
        Rollout schema (prompt, rewritten_prompt, log_prob_old, value_estimate,
        reward, group_id, sample_weight, rollout_id).
        """
        entries: List[RolloutEntry] = []
        end_offset = start_offset
        strict_filters_enabled = not (
            settings.ppo_debug_mode or settings.ppo_debug_disable_rollout_filters
        )
        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(start_offset)
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    end_offset = fh.tell()
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                        entry = _map_traj_store_line(data)
                        if entry is None:
                            continue
                        if strict_filters_enabled and entry.log_prob_old == 0.0:
                            logger.warning(
                                "Discarded traj_store rollout: log_prob_old == 0.0 | file=%s",
                                path.name,
                            )
                            continue
                        if strict_filters_enabled and _ICD_REGEX.search(
                            entry.rewritten_prompt
                        ):
                            logger.warning(
                                "Discarded traj_store rollout: rewritten_prompt contains ICD codes | file=%s",
                                path.name,
                            )
                            continue
                        entries.append(entry)
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse traj_store JSONL line from %s: %s",
                            path.name,
                            exc,
                        )
        except Exception as exc:
            logger.warning("Failed to read traj_store JSONL segment %s: %s", path, exc)

        return entries, end_offset

    def reset(self) -> None:
        self._loaded_files.clear()
        self._segment_offsets.clear()
        if self._seen_index.exists():
            self._seen_index.unlink()
        if self._segment_offsets_path.exists():
            self._segment_offsets_path.unlink()
        logger.info("RolloutLoader reset complete")

    def _parse_rollouts_from_data(
        self, data: dict, source_name: str
    ) -> List[RolloutEntry]:
        parsed = RolloutFile.model_validate(data)
        strict_filters_enabled = not (
            settings.ppo_debug_mode or settings.ppo_debug_disable_rollout_filters
        )

        valid_rollouts: List[RolloutEntry] = []
        for rollout in parsed.rollouts:
            if strict_filters_enabled and rollout.log_prob_old == 0.0:
                logger.warning(
                    "Discarded rollout from %s: log_prob_old == 0.0", source_name
                )
                continue
            if strict_filters_enabled and _ICD_REGEX.search(rollout.rewritten_prompt):
                logger.warning(
                    "Discarded rollout from %s: rewritten_prompt contains ICD codes",
                    source_name,
                )
                continue
            valid_rollouts.append(rollout)

        return valid_rollouts

    def _parse_file(self, path: Path) -> List[RolloutEntry]:
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return self._parse_rollouts_from_data(data, path.name)
        except Exception as exc:
            logger.warning("Failed to parse rollout file %s: %s", path, exc)
            return []

    def _parse_segment(
        self, path: Path, start_offset: int
    ) -> Tuple[List[RolloutEntry], int]:
        entries: List[RolloutEntry] = []
        end_offset = start_offset
        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(start_offset)
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    end_offset = fh.tell()
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                        entries.extend(self._parse_rollouts_from_data(data, path.name))
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse rollout segment line from %s: %s",
                            path.name,
                            exc,
                        )
        except Exception as exc:
            logger.warning("Failed to read rollout segment %s: %s", path, exc)

        return entries, end_offset

    def _persist_seen_index(self) -> None:
        try:
            content = "\n".join(sorted(self._loaded_files))
            self._seen_index.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist seen-files index: %s", exc)

    def _persist_segment_offsets(self) -> None:
        try:
            self._segment_offsets_path.write_text(
                json.dumps(self._segment_offsets, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist segment offsets: %s", exc)

    def _load_seen_index(self) -> Set[str]:
        if not self._seen_index.exists():
            return set()

        try:
            lines = self._seen_index.read_text(encoding="utf-8").splitlines()
            return {line.strip() for line in lines if line.strip()}
        except Exception as exc:
            logger.warning("Failed to read seen-files index: %s", exc)
            return set()

    def _load_segment_offsets(self) -> Dict[str, int]:
        if not self._segment_offsets_path.exists():
            return {}

        try:
            payload = json.loads(self._segment_offsets_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            return {
                str(name): int(offset)
                for name, offset in payload.items()
                if isinstance(name, str)
            }
        except Exception as exc:
            logger.warning("Failed to read segment offsets index: %s", exc)
            return {}
