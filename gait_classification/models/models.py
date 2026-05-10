import torch
from torch import nn

from gait_classification.models.lstm import LSTM
from gait_classification.models.transformer import GaitTransformer
from gait_classification.utils import ModelType, TrainConfig


def construct_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    """Construct the model based on the configuration."""
    if cfg.model_type == ModelType.LSTM or cfg.model_type == "lstm":
        return LSTM(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            embedding_size=cfg.embedding_size,
        ).to(device)
    elif cfg.model_type == ModelType.TRANSFORMER or cfg.model_type == "transformer":
        return GaitTransformer(
            input_size=6,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            embedding_size=cfg.embedding_size,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
