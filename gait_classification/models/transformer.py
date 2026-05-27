import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from gait_classification.utils import TrainConfig


class GaitTransformer(nn.Module):
    """
    Transformer-based model for gait classification using triplet loss.
    Uses self-attention to learn which timesteps are most important for identification.
    """

    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        embedding_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.embedding_size = embedding_size
        self.dropout_rate = dropout

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.embedding_head = nn.Linear(d_model, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Embedding tensor of shape (batch_size, embedding_size), L2-normalized
        """
        x = self.input_projection(x)

        x = self.encoder(x)

        x = x.mean(dim=1)
        x = self.dropout(x)

        embedding = self.embedding_head(x)

        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def save_a_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        cfg: TrainConfig,
    ) -> None:
        """Save a model checkpoint."""
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_type": cfg.model_type,
                "embedding_size": cfg.embedding_size,
                "input_size": self.input_size,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")
