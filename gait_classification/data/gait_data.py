from curses import raw

import numpy as np
import pandas as pd
from scipy.fft import rfft
from tqdm import tqdm

from gait_classification.utils import TrainConfig


def load_signal(cfg: TrainConfig, file_name: str):
    """Load a signal file into a NumPy array of shape (n_samples, n_timesteps)."""
    return pd.read_csv(
        f"{cfg.signals_dir}/{file_name}.txt", sep=r"\s+", header=None
    ).to_numpy()


def extract_fft_features(signals, n_samples):
    """Extract FFT features for each sample and channel, returning a 2D array of shape (n_samples, n_features)."""
    features = []
    for i in range(n_samples):
        sample_features = []
        for arr in signals.values():
            yf = rfft(arr[i, :])
            sample_features.extend(np.abs(yf[:500]))
        features.append(sample_features)
    return np.array(features)


def select_features_by_contribution(fft_features, threshold):
    """Select top features that contribute to the given cumulative energy threshold."""
    contributions = np.abs(fft_features).sum(axis=0)
    contributions /= contributions.sum()
    cumulative = np.cumsum(contributions)
    n_keep = np.searchsorted(cumulative, threshold) + 1
    return n_keep


def load_and_preprocess_data(
    cfg: TrainConfig, preprocess_functions: list
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the data, returning the processed features and labels."""
    channels = []
    for channel_key in cfg.CHANNEL_FILES.keys():
        signal = load_signal(cfg, cfg.CHANNEL_FILES[channel_key])
        channels.append(signal)

    raw = np.stack(channels, axis=2)

    y = np.loadtxt(cfg.y_path, dtype=int)

    # Limit the number of samples if max_samples is set
    n_samples = raw.shape[0]
    if cfg.max_samples > 0:
        n_samples = min(n_samples, cfg.max_samples)
        raw = raw[:n_samples]
        y = y[:n_samples]

    for func in tqdm(preprocess_functions, desc="Preprocessing data"):
        raw = func(raw)

    return raw, y


def build_windowed_data(
    cfg: TrainConfig, raw: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all 6 channel files, stack them, and apply sliding window.
    Returns windowed raw signals and per-window participant labels.
    Args:
        cfg: TrainConfig object.
        raw: 3D array of shape (n_samples, n_timesteps, n_channels) containing the raw signals.
        y: 1D array of shape (n_samples,) containing the participant labels for each sample.
    Returns:
        Tuple of (windows, labels) where windows is (N_windows, seq_len, 6) and labels is (N_windows,).
    """
    # Limit the number of samples if max_samples is set
    n_samples = raw.shape[0]
    if cfg.max_samples > 0:
        n_samples = min(n_samples, cfg.max_samples)
        raw = raw[:n_samples]
        y = y[:n_samples]

    n_samples, n_timesteps, n_channels = raw.shape
    windows_list = []
    labels_list = []

    for i in tqdm(range(n_samples), desc="Windowing data"):
        for start in range(0, n_timesteps - cfg.seq_len + 1, cfg.window_stride):
            window = raw[i, start : start + cfg.seq_len, :]
            windows_list.append(window)
            labels_list.append(y[i])

    windows = np.array(windows_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=int)

    return windows, labels
