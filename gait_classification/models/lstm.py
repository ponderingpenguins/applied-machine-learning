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
