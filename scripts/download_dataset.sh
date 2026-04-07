#!/bin/bash
set -e
echo "Downloading dataset files..."
pip install gdown --quiet
python3 data/download_data.py

if [ -f data/section111_valid_icd10_october2025.xlsx ]; then
	echo "Building local ICD-10 tree JSON..."
	python3 scripts/build_icd10_tree.py
fi

echo "Dataset download complete."
ls -lh data/
