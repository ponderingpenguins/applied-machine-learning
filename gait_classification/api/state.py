import pickle
from pathlib import Path

import torch

from gait_classification.hf_utils import (
    download_centroids,
    download_model_checkpoint,
    download_scaler,
)
from gait_classification.models.models import construct_model
from gait_classification.utils import ModelType, TrainConfig

_model_cache = {}
trusted_users = {
    ModelType.TRANSFORMER: [],
    ModelType.LSTM: [],
}


def get_model_scaler_centroids(model_type: ModelType):
    cache_key = f"{model_type.value}_model"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Try to load from HuggingFace, fallback to local files
    try:
        checkpoint_path = download_model_checkpoint(
            model_type, cache_dir=checkpoints_dir
        )
    except Exception as e:
        print(f"Failed to download model from HF: {e}, trying local fallback...")
        if model_type.value == "transformer":
            checkpoint_path = checkpoints_dir / "final_model_transformer.pt"
        else:
            checkpoint_path = checkpoints_dir / f"best_model_{model_type.value}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model not found at {checkpoint_path} and HF download failed"
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(
        model_type=checkpoint["model_type"],
        embedding_size=checkpoint["embedding_size"],
    )
    model = construct_model(config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Try to load scaler from HF, fallback to local
    try:
        scaler_path = download_scaler(cache_dir=checkpoints_dir)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        with open(checkpoints_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    # Try to load centroids from HF, fallback to local
    centroids = {}
    try:
        centroids_path = download_centroids(model_type, cache_dir=checkpoints_dir)
        with open(centroids_path, "rb") as f:
            centroids = pickle.load(f)
    except Exception:
        centroids_path = checkpoints_dir / f"centroids_{model_type.value}.pkl"
        if centroids_path.exists():
            with open(centroids_path, "rb") as f:
                centroids = pickle.load(f)

    result = (model, scaler, centroids)
    _model_cache[cache_key] = result
    return result
