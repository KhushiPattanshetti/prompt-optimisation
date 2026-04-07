#!/bin/bash
set -e
echo "Downloading dataset files..."
pip install gdown --quiet
python3 data/download_data.py
echo "Dataset download complete."
ls -lh data/
