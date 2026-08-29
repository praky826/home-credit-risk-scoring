#!/bin/bash
# Shell wrapper script to train all 5 models sequentially.

# Determine python command
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
    if [ -f ".venv/Scripts/python.exe" ]; then
        PYTHON_CMD=".venv/Scripts/python.exe"
    fi
else
    PYTHON_CMD="python"
fi

echo "=== Starting Home Credit Model Training Pipeline ==="

# 1. XGBoost
echo "Training XGBoost..."
$PYTHON_CMD -m src.models.train --model xgboost

# 2. LightGBM
echo "Training LightGBM..."
$PYTHON_CMD -m src.models.train --model lgbm

# 3. MLP
echo "Training MLP..."
$PYTHON_CMD -m src.models.train --model mlp

# 4. Entity Embedding
echo "Training Entity Embedding..."
$PYTHON_CMD -m src.models.train --model entity_embedding

# 5. TabTransformer
echo "Training TabTransformer..."
$PYTHON_CMD -m src.models.train --model tab_transformer

echo "=== Training Pipeline Complete ==="
