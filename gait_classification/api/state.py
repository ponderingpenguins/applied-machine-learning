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

    # Special handling for FFT+centroids (no neural model)
    if model_type == ModelType.FFT_CENTROIDS:
        try:
            fft_scaler_path = checkpoints_dir / "scaler_fft.pkl"
            with open(fft_scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            raise FileNotFoundError(f"FFT scaler not found at {fft_scaler_path}")

        fft_centroids = {}
        try:
            fft_centroids_path = checkpoints_dir / "centroids_fft_centroids.pkl"
            with open(fft_centroids_path, "rb") as f:
                fft_centroids = pickle.load(f)
        except Exception:
            raise FileNotFoundError(f"FFT centroids not found")

        result = (None, scaler, fft_centroids)
        _model_cache[cache_key] = result
        return result

    # Standard neural model loading
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
    dim_feedforward = checkpoint.get("transformer_dim_feedforward") or checkpoint.get(
        "dim_feedforward"
    )
    config = TrainConfig(
        model_type=checkpoint["model_type"],
        embedding_size=checkpoint["embedding_size"],
        **({"transformer_dim_feedforward": dim_feedforward} if dim_feedforward else {}),
    )
    model = construct_model(config, torch.device("cpu"))
    state_dict = checkpoint["model_state_dict"]
    # Strip wrapper prefix if checkpoint was saved from a wrapped model
    if any(k.startswith("base_model.") for k in state_dict):
        state_dict = {
            k[len("base_model."):]: v
            for k, v in state_dict.items()
            if k.startswith("base_model.")
        }
    model.load_state_dict(state_dict)
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
