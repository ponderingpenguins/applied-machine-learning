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

    seed: int = 67  # Six seven...
    fft_threshold: float = 0.95  # Chosen in the preliminary data look notebook by plotting the t-sne of the FFT features to see which threshold gives the best separation.
    # Would be nice to use the elbow method.
    n_folds: int = 5

    data_dir: str = "../Gait-Datasets-TIFS20/Dataset #1"
    train_dir: str = f"{data_dir}/train"
    test_dir: str = f"{data_dir}/test"
    signals_dir: str = f"{train_dir}/Inertial Signals"
    y_path: str = f"{train_dir}/y_train.txt"
    CHANNEL_FILES: dict[str, str] = {
        "ACCx": "train_acc_x",
        "ACCy": "train_acc_y",
        "ACCz": "train_acc_z",
        "GYRx": "train_gyr_x",
        "GYRy": "train_gyr_y",
        "GYRz": "train_gyr_z",
    }
