#!/usr/bin/env bash
# Activate the virtual environment and launch the Streamlit app.
# Usage:  bash run.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found. Run:"
    echo "  python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "Starting Panorama Stitcher …"
venv/bin/streamlit run app.py

