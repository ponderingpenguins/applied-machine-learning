"""Utility functions and classes for gait classification."""

from dataclasses import dataclass, field
from enum import StrEnum


class ModelType(StrEnum):
    """Model types"""

    LSTM = "lstm"
    TRANSFORMER = "transformer"


@dataclass
class TrainConfig:
    """
    Configuration for training.
    Args:
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        seed: Random seed for reproducibility.
        fft_threshold: Threshold for selecting FFT features based on cumulative energy.
        n_folds: Number of folds for cross-validation.
        seq_len: Length of input sequences for the model.
        window_stride: Stride for the sliding window when creating input sequences.
        embedding_size: Size of the embedding layer in the model.
        triplet_margin: Margin for the triplet loss function.
        train_split: Proportion of participants to use for training.
        val_split: Proportion of participants to use for validation.
        max_samples: Maximum number of samples to use (0 for all).
        checkpoint_dir: Directory to save model checkpoints.
        data_dir: Base directory for the dataset.
        train_dir: Directory for training data.
        test_dir: Directory for test data.
        signals_dir: Directory for inertial signal files.
        y_path: Path to the file containing participant labels.
        CHANNEL_FILES: Dictionary mapping channel keys to their corresponding file names.
        sampling_rate: Sampling rate of the signals in Hz.
        cutoff_freq: Cutoff frequency for low-pass filtering in Hz.
        filter_order: Order of the Butterworth filter.
    """

    figures_dir: str = "./figures/"
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001

    model_type: str = "lstm"

    seed: int = 67  # Six seven...
    fft_threshold: float = 0.95  # Chosen in the preliminary data look notebook by plotting the t-sne of the FFT features to see which threshold gives the best separation.
    # Would be nice to use the elbow method.
    n_folds: int = 5

    seq_len: int = 128
    window_stride: int = 128
    embedding_size: int = 64
    triplet_margin: float = 0.3

    train_split: float = 0.70
    val_split: float = 0.15

    max_samples: int = 0

    checkpoint_dir: str = "checkpoints"

    # Data configuration
    data_dir: str = "../Gait-Datasets-TIFS20/Dataset #1"
    train_dir: str = f"{data_dir}/train"
    test_dir: str = f"{data_dir}/test"
    signals_dir: str = f"{train_dir}/Inertial Signals"
    y_path: str = f"{train_dir}/y_train.txt"
    CHANNEL_FILES: dict[str, str] = field(
        default_factory=lambda: {
            "ACCx": "train_acc_x",
            "ACCy": "train_acc_y",
            "ACCz": "train_acc_z",
            "GYRx": "train_gyr_x",
            "GYRy": "train_gyr_y",
            "GYRz": "train_gyr_z",
        }
    )
    sampling_rate: float = 50.0  # Hz
    cutoff_freq: float = 5.0  # Hz
    filter_order: int = 4  # Order of the Butterworth filter
