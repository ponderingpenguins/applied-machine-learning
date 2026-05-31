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
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            embedding_size=cfg.embedding_size,
            dropout=cfg.dropout,
        ).to(device)
    elif cfg.model_type == ModelType.TRANSFORMER or cfg.model_type == "transformer":
        return GaitTransformer(
            input_size=6,
            d_model=cfg.transformer_d_model,
            nhead=cfg.transformer_nhead,
            num_layers=cfg.transformer_num_layers,
            dim_feedforward=cfg.transformer_dim_feedforward,
            embedding_size=cfg.embedding_size,
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
