import torch
import torch.nn as nn
import torch.nn.functional as F


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

        embedding = self.embedding_head(x)

        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
