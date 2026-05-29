"""Utilities for uploading and downloading models from Hugging Face Hub."""

import os
import pickle
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, login, whoami

from gait_classification.utils import ModelType


HF_REPO_ID = "slugroom/gait-classification-models"


def _get_hf_token() -> str:
    """Get HF token from environment or use cached credentials."""
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            whoami()
            token = None
        except Exception:
            raise ValueError(
                "HF_TOKEN not set. Please set HF_TOKEN environment variable or run 'huggingface-cli login'"
            )
    return token


def authenticate_hf(token: str | None = None):
    """Authenticate with Hugging Face Hub."""
    if token:
        login(token=token)
    else:
        token = _get_hf_token()
        if token:
            login(token=token)


def download_model_checkpoint(model_type: ModelType, cache_dir: Path | None = None) -> Path:
    """Download model checkpoint from HF Hub.

    Args:
        model_type: Type of model (transformer, lstm, etc.)
        cache_dir: Directory to cache the downloaded file. Defaults to HF cache.

    Returns:
        Path to the downloaded checkpoint file.
    """
    filename = f"final_model_{model_type.value}.pt" if model_type.value == "transformer" else f"best_model_{model_type.value}.pt"

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def download_scaler(cache_dir: Path | None = None) -> Path:
    """Download scaler from HF Hub."""
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="scaler.pkl",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def download_centroids(model_type: ModelType, cache_dir: Path | None = None) -> Path:
    """Download centroids from HF Hub."""
    filename = f"centroids_{model_type.value}.pkl"
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def upload_models(checkpoints_dir: Path, token: str | None = None):
    """Upload all model artifacts to HF Hub.

    Args:
        checkpoints_dir: Directory containing model files to upload.
        token: HF token. If not provided, will try to use cached credentials.
    """
    authenticate_hf(token)
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id=HF_REPO_ID)
    except Exception:
        print(f"Creating repo {HF_REPO_ID}...")
        api.create_repo(repo_id=HF_REPO_ID, private=False)

    # Upload model files
    files_to_upload = [
        "final_model_transformer.pt",
        "best_model_lstm.pt",
        "scaler.pkl",
        "centroids_transformer.pkl",
        "centroids_lstm.pkl",
    ]

    for filename in files_to_upload:
        file_path = checkpoints_dir / filename
        if file_path.exists():
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=HF_REPO_ID,
            )
        else:
            print(f"⚠ {filename} not found, skipping...")


def upload_model_from_training(model_state_dict: dict, model_type: ModelType, checkpoints_dir: Path, token: str | None = None):
    """Upload a single model after training."""
    authenticate_hf(token)
    api = HfApi()

    if model_type.value == "transformer":
        filename = "final_model_transformer.pt"
    else:
        filename = f"best_model_{model_type.value}.pt"

    file_path = checkpoints_dir / filename
    if file_path.exists():
        print(f"Uploading {filename} to {HF_REPO_ID}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
        )
