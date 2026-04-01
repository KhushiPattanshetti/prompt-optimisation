"""
Rollout loader: scans the rollouts/ directory and deserialises JSON files
into Python objects for the training loop.

FIX SUMMARY (from review — Issue 20):
  - Previously _loaded_files was an in-memory set[str].
    On service restart, this set was lost and ALL existing rollout
    files in rollouts/ would be reprocessed, corrupting training.
  - Fixed by persisting the seen-files index to disk at
    rollouts/.seen_files (one filename per line).
  - On restart, the index is reloaded from disk so already-processed
    files are never replayed.
  - reset() now clears both the in-memory set and the disk index.
"""

import json
import logging
from pathlib import Path
from typing import List, Set

from schemas.rollout_schema import RolloutEntry, RolloutFile

logger = logging.getLogger(__name__)

_SEEN_INDEX_FILENAME = ".seen_files"


class RolloutLoader:
    """
    Loads rollout trajectory files from a directory.

    Tracks which files have already been processed using a persistent
    on-disk index so that service restarts do not replay old rollouts.

    Args:
        rollouts_dir: Path to the directory containing rollout JSON files.
    """

    def __init__(self, rollouts_dir: Path) -> None:
        self.rollouts_dir  = Path(rollouts_dir)
        self._seen_index   = self.rollouts_dir / _SEEN_INDEX_FILENAME
        self._loaded_files: Set[str] = self._load_seen_index()

    # ── Public API ─────────────────────────────────────────────────────────

    def load_all(self) -> List[RolloutEntry]:
        """Load ALL rollout files (including already-seen files).

        Used for full reprocessing (e.g. debugging).

        Returns:
            Flat list of RolloutEntry objects.
        """
        entries: List[RolloutEntry] = []
        for path in sorted(self.rollouts_dir.glob("*.json")):
            entries.extend(self._parse_file(path))
        logger.info(
            "Loaded %d rollout entries from %s", len(entries), self.rollouts_dir
        )
        return entries

    def load_new(self) -> List[RolloutEntry]:
        """Load only NEW rollout files not seen in any previous call.

        Persists the updated seen-files index to disk after loading
        so restarts never replay the same files.

        Returns:
            Flat list of RolloutEntry objects from new files only.
        """
        entries: List[RolloutEntry] = []
        newly_seen: List[str] = []

        for path in sorted(self.rollouts_dir.glob("*.json")):
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
                len(entries), len(newly_seen),
            )

        return entries

    def reset(self) -> None:
        """Clear the seen-files tracking both in memory and on disk.

        After reset(), load_new() will return ALL files in the directory
        as if they had never been loaded. Use with caution.
        """
        self._loaded_files.clear()
        if self._seen_index.exists():
            self._seen_index.unlink()
        logger.info("RolloutLoader reset — seen-files index cleared")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _load_seen_index(self) -> Set[str]:
        """Load the persisted seen-files index from disk.

        Returns an empty set if the index does not exist yet
        (first run, or after reset()).
        """
        if not self._seen_index.exists():
            return set()
        try:
            lines = self._seen_index.read_text(encoding="utf-8").splitlines()
            seen  = {line.strip() for line in lines if line.strip()}
            logger.info(
                "Seen-files index loaded | %d files already processed", len(seen)
            )
            return seen
        except OSError as exc:
            logger.warning(
                "Could not read seen-files index %s: %s — starting fresh",
                self._seen_index, exc,
            )
            return set()

    def _persist_seen_index(self) -> None:
        """Write the current seen-files set to disk."""
        try:
            self.rollouts_dir.mkdir(parents=True, exist_ok=True)
            self._seen_index.write_text(
                "\n".join(sorted(self._loaded_files)), encoding="utf-8"
            )
        except OSError as exc:
            logger.error(
                "Failed to persist seen-files index: %s", exc
            )

    def _parse_file(self, path: Path) -> List[RolloutEntry]:
        """Parse a single rollout JSON file into RolloutEntry objects."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rollout_file = RolloutFile(**data)
            return rollout_file.rollouts
        except Exception as exc:
            logger.error("Failed to parse rollout file %s: %s", path.name, exc)
            return []

# """
# Rollout loader: scans the rollouts/ directory and deserialises JSON files
# produced by the Trajectory Store into Python objects.
# """

# import json
# import logging
# from pathlib import Path
# from typing import List

# from schemas.rollout_schema import RolloutEntry, RolloutFile

# logger = logging.getLogger(__name__)


# class RolloutLoader:
#     """
#     Loads rollout trajectory files from a directory.

#     Args:
#         rollouts_dir: Path to the directory containing rollout JSON files.
#     """

#     def __init__(self, rollouts_dir: Path) -> None:
#         self.rollouts_dir = Path(rollouts_dir)
#         self._loaded_files: set[str] = set()

#     def load_all(self) -> List[RolloutEntry]:
#         """
#         Load *all* rollout files (previously loaded files included).

#         Returns:
#             Flat list of RolloutEntry objects.
#         """
#         entries: List[RolloutEntry] = []
#         for path in sorted(self.rollouts_dir.glob("*.json")):
#             entries.extend(self._parse_file(path))
#         logger.info(
#             "Loaded %d rollout entries from %s", len(entries), self.rollouts_dir
#         )
#         return entries

#     def load_new(self) -> List[RolloutEntry]:
#         """
#         Load only *new* rollout files (not seen in previous calls).

#         Returns:
#             Flat list of RolloutEntry objects from new files only.
#         """
#         entries: List[RolloutEntry] = []
#         for path in sorted(self.rollouts_dir.glob("*.json")):
#             if path.name not in self._loaded_files:
#                 batch = self._parse_file(path)
#                 entries.extend(batch)
#                 self._loaded_files.add(path.name)
#         if entries:
#             logger.info("Loaded %d new rollout entries", len(entries))
#         return entries

#     def _parse_file(self, path: Path) -> List[RolloutEntry]:
#         """Parse a single rollout JSON file."""
#         try:
#             with open(path, "r") as f:
#                 data = json.load(f)
#             rollout_file = RolloutFile(**data)
#             return rollout_file.rollouts
#         except Exception as exc:
#             logger.error("Failed to parse rollout file %s: %s", path, exc)
#             return []

#     def reset(self) -> None:
#         """Clear the set of already-loaded file names."""
#         self._loaded_files.clear()
