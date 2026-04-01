"""Configuration constants for dataset_svc.

All file paths and service settings are defined here.
No other module may define or hardcode file paths.
"""

from __future__ import annotations

from pathlib import Path

# --- Path configuration ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
NOTES_CSV_PATH: Path = PROJECT_ROOT / "data" / "notes.csv"
DIAGNOSES_CSV_PATH: Path = PROJECT_ROOT / "data" / "diagnoses.csv"

# --- Service configuration ---
SERVICE_HOST: str = "0.0.0.0"
SERVICE_PORT: int = 8003
DEFAULT_BATCH_SIZE: int = 32
MAX_BATCH_SIZE: int = 512


def validate_data_files() -> None:
    """Check that both CSV data files exist at their configured paths.

    Must be called at service startup before any loading begins.

    Raises:
        FileNotFoundError: If either CSV file is missing, with a message
            listing the missing paths and instructions to download.
    """
    missing: list[Path] = []

    if not NOTES_CSV_PATH.exists():
        missing.append(NOTES_CSV_PATH)
    if not DIAGNOSES_CSV_PATH.exists():
        missing.append(DIAGNOSES_CSV_PATH)

    if missing:
        paths_str = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing data files:\n  {paths_str}\n\n"
            f"Run: python data/download_data.py"
        )
