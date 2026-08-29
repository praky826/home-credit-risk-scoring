#!/bin/bash
# Shell wrapper script to download Home Credit Default Risk data using the Python ingestion script.

# Determine python command
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
    if [ -f ".venv/Scripts/python.exe" ]; then
        PYTHON_CMD=".venv/Scripts/python.exe"
    fi
else
    PYTHON_CMD="python"
fi

echo "Running data download via $PYTHON_CMD..."
$PYTHON_CMD scripts/download_data.py
