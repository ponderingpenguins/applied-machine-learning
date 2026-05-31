import os

import torch
import torch.nn.functional as F
from torch import nn

from gait_classification.utils import TrainConfig


class LSTM(nn.Module):
    """
    LSTM model for gait classification, trained using triplet loss.
    input is 3D gyro and accelerometer data, output is an embedding vector.
    """

    def __init__(
        self,
        input_size=6,
        hidden_size=128,
        num_layers=2,
        embedding_size=64,
        dropout=0.1,
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.dropout_rate = dropout
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        """Forward pass through the LSTM model."""
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size)
        # Take the last time step's output
        last_output = lstm_out[:, -1, :]  # last_output shape: (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        embedding = self.fc(last_output)  # embedding shape: (batch_size, embedding_size)
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
        path = os.path.join(cfg.checkpoint_dir, f"best_model_{cfg.model_type}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_type": cfg.model_type,
                "embedding_size": cfg.embedding_size,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_rate,
                "d_model": self.embedding_size,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")
