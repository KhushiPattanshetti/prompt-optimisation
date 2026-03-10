"""
Rollout loader: scans the rollouts/ directory and deserialises JSON files
produced by the Trajectory Store into Python objects.
"""

import json
import logging
from pathlib import Path
from typing import List

from schemas.rollout_schema import RolloutEntry, RolloutFile

logger = logging.getLogger(__name__)


class RolloutLoader:
    """
    Loads rollout trajectory files from a directory.

    Args:
        rollouts_dir: Path to the directory containing rollout JSON files.
    """

    def __init__(self, rollouts_dir: Path) -> None:
        self.rollouts_dir = Path(rollouts_dir)
        self._loaded_files: set[str] = set()

    def load_all(self) -> List[RolloutEntry]:
        """
        Load *all* rollout files (previously loaded files included).

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
        """
        Load only *new* rollout files (not seen in previous calls).

        Returns:
            Flat list of RolloutEntry objects from new files only.
        """
        entries: List[RolloutEntry] = []
        for path in sorted(self.rollouts_dir.glob("*.json")):
            if path.name not in self._loaded_files:
                batch = self._parse_file(path)
                entries.extend(batch)
                self._loaded_files.add(path.name)
        if entries:
            logger.info("Loaded %d new rollout entries", len(entries))
        return entries

    def _parse_file(self, path: Path) -> List[RolloutEntry]:
        """Parse a single rollout JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            rollout_file = RolloutFile(**data)
            return rollout_file.rollouts
        except Exception as exc:
            logger.error("Failed to parse rollout file %s: %s", path, exc)
            return []

    def reset(self) -> None:
        """Clear the set of already-loaded file names."""
        self._loaded_files.clear()
