from torch import nn


class LSTM(nn.Module):
    """
    LSTM model for gait classification, trained using triplet loss.
    input is 3D gyro and accelerometer data, output is an embedding vector.
    """

    def __init__(self, input_size=6, hidden_size=128, num_layers=2, embedding_size=64):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        """Forward pass through the LSTM model."""
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(
            x
        )  # lstm_out shape: (batch_size, seq_length, hidden_size)
        # Take the last time step's output
        last_output = lstm_out[:, -1, :]  # last_output shape: (batch_size, hidden_size)
        embedding = self.fc(
            last_output
        )  # embedding shape: (batch_size, embedding_size)
        return embedding


class GaitTransformer(nn.Module):
    """
    Transformer model for gait classification, trained using triplet loss.
    input is 3D gyro and accelerometer data, output is an embedding vector.
    """

    def __init__(
        self,
        input_size=6,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        embedding_size=64,
    ):
        super(GaitTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, embedding_size)

    def forward(self, x):
        """Forward pass through the Transformer model."""
        # x shape: (batch_size, seq_length, input_size)
        x = self.input_proj(x)  # shape: (batch_size, seq_length, d_model)
        x = x.permute(1, 0, 2)  # shape: (seq_length, batch_size, d_model)
        transformer_out = self.transformer_encoder(
            x
        )  # shape: (seq_length, batch_size, d_model)
        last_output = transformer_out[-1]  # shape: (batch_size, d_model)
        embedding = self.fc(last_output)  # shape: (batch_size, embedding_size)
        return embedding
