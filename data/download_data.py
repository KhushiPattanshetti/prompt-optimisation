"""Download dataset files from Google Drive using gdown.

Usage:
    pip install gdown
    python data/download_data.py
"""

from pathlib import Path
import os

import gdown

# --- Google Drive file IDs (update these when files change) ---
NOTES_FILE_ID = "1UbaMm5bG8Axacwc6MWhhMzZO2PqC6Ibs"
DIAGNOSES_FILE_ID = "14N_NjkppC_-xUlvseP_oVxUNmeyrqHT3"
# Optional: ICD-10 source spreadsheet used by scripts/build_icd10_tree.py
# Set via env var or edit this constant directly.
ICD10_SECTION111_XLSX_FILE_ID = os.environ.get(
    "ICD10_SECTION111_XLSX_FILE_ID",
    "18aYookEAjw9nn7KPM90lQBzKJlYpva4R",
).strip()

# --- Destination paths (resolved relative to this script) ---
DATA_DIR = Path(__file__).resolve().parent
NOTES_CSV_PATH = DATA_DIR / "notes.csv"
DIAGNOSES_CSV_PATH = DATA_DIR / "diagnoses.csv"
ICD10_SECTION111_XLSX_PATH = DATA_DIR / "section111_valid_icd10_october2025.xlsx"

FILES = [
    {"name": "notes.csv", "file_id": NOTES_FILE_ID, "dest": NOTES_CSV_PATH},
    {"name": "diagnoses.csv", "file_id": DIAGNOSES_FILE_ID, "dest": DIAGNOSES_CSV_PATH},
]

if ICD10_SECTION111_XLSX_FILE_ID:
    FILES.append(
        {
            "name": "section111_valid_icd10_october2025.xlsx",
            "file_id": ICD10_SECTION111_XLSX_FILE_ID,
            "dest": ICD10_SECTION111_XLSX_PATH,
        }
    )


def download_file(name: str, file_id: str, dest: Path) -> None:
    """Download a single file from Google Drive if it does not already exist."""
    if dest.exists():
        print(f"[SKIP] {name} already exists at {dest}")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[DOWNLOAD] Downloading {name} to {dest} ...")
    try:
        gdown.download(url, str(dest), quiet=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {name}.\n"
            f"  URL: {url}\n"
            f"  Destination: {dest}\n\n"
            f"Ensure gdown is installed: pip install gdown\n"
            f"Then retry: python data/download_data.py"
        ) from exc

    if not dest.exists():
        raise RuntimeError(
            f"Download appeared to succeed but {dest} was not created.\n"
            f"Please retry: python data/download_data.py"
        )
    print(f"[OK] {name} saved to {dest}")


def main() -> None:
    """Download all required dataset files."""
    for file_info in FILES:
        download_file(file_info["name"], file_info["file_id"], file_info["dest"])

    if not ICD10_SECTION111_XLSX_FILE_ID and not ICD10_SECTION111_XLSX_PATH.exists():
        print(
            "[WARN] ICD spreadsheet file ID is not configured and section111_valid_icd10_october2025.xlsx "
            "is missing. Set ICD10_SECTION111_XLSX_FILE_ID to enable auto-download."
        )

    print("\nAll dataset files are ready.")


if __name__ == "__main__":
    main()
