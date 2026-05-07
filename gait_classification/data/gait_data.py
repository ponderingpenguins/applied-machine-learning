import numpy as np
import pandas as pd
from scipy.fft import rfft

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
