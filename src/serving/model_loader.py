import os
import yaml
import torch
import boto3
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname
import mlflow
from mlflow.tracking import MlflowClient

_model = None
_model_version = ""
_model_name = "LendingClub XGBoost Default Risk Classifier"
_model_type = "xgboost"
_model_stage = "Production"

def _format_display_name(m_type: str, stage: str = "Production") -> str:
    type_labels = {
        "xgboost": "LendingClub XGBoost Default Risk Classifier",
        "lgbm": "LendingClub LightGBM Default Risk Classifier",
        "entity_embedding": "LendingClub Entity Embedding Neural Network",
        "mlp": "LendingClub Multi-Layer Perceptron",
        "tab_transformer": "LendingClub TabTransformer Model",
    }
    base = type_labels.get(m_type.lower(), f"{m_type.capitalize()} Risk Model")
    if stage and stage != "Production":
        return f"{base} [{stage}]"
    return base

def load_model(config_path: str = "configs/serving.yaml") -> None:
    """
    Called once at FastAPI startup.
    Reads serving.yaml for registry_model_name/model_stage or a fallback artifact_path.
    """
    global _model, _model_version, _model_name, _model_type, _model_stage

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Serving config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_type = config.get("model_type", "xgboost")
    registry_model_name = config.get("registry_model_name")
    registry_stage = config.get("registry_stage", "Production")
    artifact_path = config.get("artifact_path")


    if not model_type:
        raise ValueError("model_type must be specified in serving.yaml")

    local_path = None

    if registry_model_name:
        client = MlflowClient()
        versions = client.get_latest_versions(
            name=registry_model_name,
            stages=[registry_stage],
        )
        if not versions:
            raise RuntimeError(
                f"No MLflow model found for name={registry_model_name!r} in stage={registry_stage!r}"
            )

        target = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        download_uri = client.get_model_version_download_uri(registry_model_name, target.version)
        parsed_url = urlparse(download_uri)
        local_dir = "data/processed"
        os.makedirs(local_dir, exist_ok=True)

        if parsed_url.scheme in ("", "file"):
            local_path = url2pathname(parsed_url.path)
        else:
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=download_uri,
                    dst_path=local_dir,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MLflow model artifact from registry: {e}"
                )

        _model_version = f"{registry_model_name}:v{target.version}@{registry_stage}"
        _model_type = model_type
        _model_stage = registry_stage
        _model_name = _format_display_name(model_type, registry_stage)

    elif artifact_path:
        # If artifact_path starts with S3, download it
        if artifact_path.startswith("s3://"):
            print(f"Downloading model artifact from S3: {artifact_path}...")
            parsed_url = urlparse(artifact_path)
            bucket_name = parsed_url.netloc
            key = parsed_url.path.lstrip("/")

            # Local filename
            local_filename = os.path.basename(key)
            local_dir = "data/processed"
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, local_filename)

            try:
                s3_client = boto3.client("s3")
                s3_client.download_file(bucket_name, key, local_path)
                print(f"Downloaded model artifact to {local_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model artifact from S3: {e}")
        else:
            local_path = artifact_path
        _model_version = os.path.basename(local_path)
        _model_type = model_type
        _model_stage = "Production"
        _model_name = _format_display_name(model_type, "Production")
    else:
        raise ValueError(
            "Either registry_model_name or artifact_path must be specified in serving.yaml"
        )

    if not os.path.exists(local_path):
        raise RuntimeError(f"Model artifact not found at local path: {local_path}")

    print(f"Loading {model_type} model from {local_path}...")
    try:
        if model_type == "xgboost":
            from src.models.xgboost_model import XGBoostModel
            _model = XGBoostModel.load(local_path)
        elif model_type == "lgbm":
            from src.models.lgbm_model import LGBMModel
            _model = LGBMModel.load(local_path)
        elif model_type == "mlp":
            from src.models.mlp import MLPModel
            _model = MLPModel.load(local_path)
        elif model_type == "entity_embedding":
            from src.models.entity_embedding import EntityEmbeddingModel
            _model = EntityEmbeddingModel.load(local_path)
        elif model_type == "tab_transformer":
            from src.models.tab_transformer import TabTransformerModel
            _model = TabTransformerModel.load(local_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_type} from {local_path}: {e}")

def get_model() -> Any:
    global _model
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
    return _model

def get_model_version() -> str:
    global _model_version
    return _model_version

def get_model_name() -> str:
    global _model_name
    return _model_name

def get_model_type() -> str:
    global _model_type
    return _model_type

def get_model_stage() -> str:
    global _model_stage
    return _model_stage

def get_model_info() -> dict[str, Any]:
    return {
        "model_name": get_model_name(),
        "model_version": get_model_version(),
        "model_type": get_model_type(),
        "model_stage": get_model_stage(),
        "model_loaded": _model is not None,
    }
