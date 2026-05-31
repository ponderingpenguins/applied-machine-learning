import pickle
from pathlib import Path

import torch

from gait_classification.hf_utils import (
    download_model_checkpoint,
    download_scaler,
)
from gait_classification.models.models import construct_model
from gait_classification.utils import ModelType, TrainConfig

_model_cache = {}


def get_model_and_scaler(model_type: ModelType):
    cache_key = f"{model_type.value}_model"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Special handling for FFT (no neural model)
    if model_type == ModelType.FFT_CENTROIDS:
        try:
            fft_scaler_path = checkpoints_dir / "scaler_fft.pkl"
            with open(fft_scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FFT scaler not found at {fft_scaler_path}. "
                "Run: python -m gait_classification.compute_fft_centroids "
                "to generate FFT centroids and scaler."
            )

        result = (None, scaler)
        _model_cache[cache_key] = result
        return result

    # Standard neural model loading
    try:
        checkpoint_path = download_model_checkpoint(model_type, cache_dir=checkpoints_dir)
    except Exception as e:
        print(f"Failed to download model from HF: {e}, trying local fallback...")
        if model_type.value == "transformer":
            checkpoint_path = checkpoints_dir / "final_model_transformer.pt"
        else:
            checkpoint_path = checkpoints_dir / f"best_model_{model_type.value}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found at {checkpoint_path} and HF download failed")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract transformer config from checkpoint (handle old and new key names)
    config_kwargs = {
        "model_type": checkpoint["model_type"],
        "embedding_size": checkpoint["embedding_size"],
    }
    
    # Handle transformer-specific config
    if checkpoint.get("model_type") == "transformer":
        # Prefer new key names, fall back to old ones
        d_model = checkpoint.get("transformer_d_model") or checkpoint.get("d_model")
        nhead = checkpoint.get("transformer_nhead") or checkpoint.get("nhead")
        num_layers = checkpoint.get("transformer_num_layers") or checkpoint.get("num_layers")
        dim_feedforward = checkpoint.get("transformer_dim_feedforward") or checkpoint.get("dim_feedforward")
        
        if d_model:
            config_kwargs["transformer_d_model"] = d_model
        if nhead:
            config_kwargs["transformer_nhead"] = nhead
        if num_layers:
            config_kwargs["transformer_num_layers"] = num_layers
        if dim_feedforward:
            config_kwargs["transformer_dim_feedforward"] = dim_feedforward
    
    config = TrainConfig(**config_kwargs)
    model = construct_model(config, torch.device("cpu"))
    state_dict = checkpoint["model_state_dict"]
    # Strip wrapper prefix if checkpoint was saved from a wrapped model
    if any(k.startswith("base_model.") for k in state_dict):
        state_dict = {
            k[len("base_model.") :]: v for k, v in state_dict.items() if k.startswith("base_model.")
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

    result = (model, scaler)
    _model_cache[cache_key] = result
    return result
