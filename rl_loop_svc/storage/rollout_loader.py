import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ..app.config import settings
from ..schemas.rollout_schema import RolloutEntry, RolloutFile

logger = logging.getLogger(__name__)

_SEEN_INDEX_FILENAME = ".seen_files"
_SEGMENT_OFFSETS_FILENAME = ".segment_offsets.json"
_ICD_REGEX = re.compile(r"\b[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?\b")
_ROLLOUT_FILE_GLOB = "rollout_batch_*.json"
_ROLLOUT_SEGMENT_GLOB = "rollout_segment_*.jsonl"


class RolloutLoader:
    def __init__(self, rollouts_dir: Path) -> None:
        self.rollouts_dir = Path(rollouts_dir)
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self._seen_index = self.rollouts_dir / _SEEN_INDEX_FILENAME
        self._segment_offsets_path = self.rollouts_dir / _SEGMENT_OFFSETS_FILENAME
        self._loaded_files: Set[str] = self._load_seen_index()
        self._segment_offsets: Dict[str, int] = self._load_segment_offsets()

    def load_all(self) -> List[RolloutEntry]:
        entries: List[RolloutEntry] = []
        for path in sorted(self.rollouts_dir.glob(_ROLLOUT_FILE_GLOB)):
            entries.extend(self._parse_file(path))

        for path in sorted(self.rollouts_dir.glob(_ROLLOUT_SEGMENT_GLOB)):
            batch, _ = self._parse_segment(path, start_offset=0)
            entries.extend(batch)

        logger.info("Loaded %d rollout entries from %s", len(entries), self.rollouts_dir)
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
            logger.info("Loaded %d new rollout entries from append-only segments", len(entries))

        return entries

    def reset(self) -> None:
        self._loaded_files.clear()
        self._segment_offsets.clear()
        if self._seen_index.exists():
            self._seen_index.unlink()
        if self._segment_offsets_path.exists():
            self._segment_offsets_path.unlink()
        logger.info("RolloutLoader reset complete")

    def _parse_rollouts_from_data(self, data: dict, source_name: str) -> List[RolloutEntry]:
        parsed = RolloutFile.model_validate(data)
        strict_filters_enabled = not (
            settings.ppo_debug_mode
            or settings.ppo_debug_disable_rollout_filters
        )

        valid_rollouts: List[RolloutEntry] = []
        for rollout in parsed.rollouts:
            if strict_filters_enabled and rollout.log_prob_old == 0.0:
                logger.warning("Discarded rollout from %s: log_prob_old == 0.0", source_name)
                continue
            if strict_filters_enabled and _ICD_REGEX.search(rollout.rewritten_prompt):
                logger.warning("Discarded rollout from %s: rewritten_prompt contains ICD codes", source_name)
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

    def _parse_segment(self, path: Path, start_offset: int) -> Tuple[List[RolloutEntry], int]:
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
                        logger.warning("Failed to parse rollout segment line from %s: %s", path.name, exc)
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
