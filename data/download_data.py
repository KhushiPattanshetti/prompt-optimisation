"""Download dataset CSV files from Google Drive using gdown.

Usage:
    pip install gdown
    python data/download_data.py
"""

from pathlib import Path

import gdown

# --- Google Drive file IDs (update these when files change) ---
NOTES_FILE_ID = "https://drive.google.com/file/d/1UbaMm5bG8Axacwc6MWhhMzZO2PqC6Ibs/view?usp=sharing"
DIAGNOSES_FILE_ID = "https://drive.google.com/file/d/14N_NjkppC_-xUlvseP_oVxUNmeyrqHT3/view?usp=sharing"

# --- Destination paths (resolved relative to this script) ---
DATA_DIR = Path(__file__).resolve().parent
NOTES_CSV_PATH = DATA_DIR / "notes.csv"
DIAGNOSES_CSV_PATH = DATA_DIR / "diagnoses.csv"

FILES = [
    {"name": "notes.csv", "file_id": NOTES_FILE_ID, "dest": NOTES_CSV_PATH},
    {"name": "diagnoses.csv", "file_id": DIAGNOSES_FILE_ID, "dest": DIAGNOSES_CSV_PATH},
]


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
    print("\nAll dataset files are ready.")


if __name__ == "__main__":
    main()
