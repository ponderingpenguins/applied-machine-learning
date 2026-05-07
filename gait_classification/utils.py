"""Utility functions and classes for gait classification."""

from dataclasses import dataclass
from enum import StrEnum


class ModelType(StrEnum):
    """Model types"""

    LSTM = "lstm"


@dataclass
class TrainConfig:
    """Configuration for training"""

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
