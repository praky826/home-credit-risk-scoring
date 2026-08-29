#!/bin/bash
# Shell wrapper script to run the feature engineering pipeline.

# Determine python command
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
    if [ -f ".venv/Scripts/python.exe" ]; then
        PYTHON_CMD=".venv/Scripts/python.exe"
    fi
else
    PYTHON_CMD="python"
fi

echo "Running feature pipeline via $PYTHON_CMD..."
$PYTHON_CMD -m src.features.pipeline
