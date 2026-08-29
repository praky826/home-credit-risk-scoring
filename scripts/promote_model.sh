#!/bin/bash
# Usage: bash scripts/promote_model.sh --run-id abc123 --stage Production

set -e

RUN_ID=""
STAGE="Production"

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --stage)  STAGE="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ -z "$RUN_ID" ]; then
  echo "Error: --run-id required"
  exit 1
fi

# Determine python command
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
    if [ -f ".venv/Scripts/python.exe" ]; then
        PYTHON_CMD=".venv/Scripts/python.exe"
    fi
else
    PYTHON_CMD="python"
fi

"$PYTHON_CMD" - "$RUN_ID" "$STAGE" <<'EOF'
import os
import sys
import mlflow
client = mlflow.tracking.MlflowClient()
model_name = "home_credit_default_risk"
run_id = sys.argv[1]
stage = sys.argv[2]

# Find the model version with this run_id
versions = client.search_model_versions(f"name='{model_name}'")
target = next((v for v in versions if v.run_id == run_id), None)
if not target:
    # If not registered yet, register it first
    # (Since this is local tracking URI, we can register the model from the run)
    print(f"Registering model home_credit_default_risk from run_id {run_id}...")
    try:
        run = client.get_run(run_id)
        # Let's search artifacts to find the model name logged
        artifacts = client.list_artifacts(run_id)
        model_artifact = next((a.path for a in artifacts if "model_" in a.path), None)
        if model_artifact:
            mlflow.register_model(f"runs:/{run_id}/{model_artifact}", model_name)
            # Re-fetch
            versions = client.search_model_versions(f"name='{model_name}'")
            target = next((v for v in versions if v.run_id == run_id), None)
    except Exception as e:
        print(f"Registration failed: {e}")

if not target:
    # Fallback: create mock version or raise error
    raise ValueError(f"No model version found for run_id={run_id}. Make sure the model is registered first.")

client.transition_model_version_stage(
    name=model_name,
    version=target.version,
    stage=stage,
    archive_existing_versions=True
)
print(f"Promoted {model_name} v{target.version} (run {target.run_id}) to {stage}")
EOF
