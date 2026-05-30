"""Compute and save FFT-based centroids."""

import os
import pickle
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.fft import rfft
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from gait_classification.data.gait_data import (
    build_windowed_data,
    load_and_preprocess_data,
)
from gait_classification.data.filters import construct_filters
from gait_classification.utils import TrainConfig


def extract_fft_features_from_window(window):
    """Extract FFT features from a single window (seq_len, 6)."""
    features = []
    for ch in range(window.shape[1]):
        yf = rfft(window[:, ch])
        features.extend(np.abs(yf[:250]))
    return np.array(features, dtype=np.float32)


def select_features_by_contribution(fft_features_2d, threshold=0.95):
    """
    Select top features based on cumulative energy contribution.
    Args:
        fft_features_2d: shape (n_samples, n_features)
        threshold: cumulative energy threshold (e.g., 0.95)
    Returns:
        indices of selected features
    """
    contributions = np.abs(fft_features_2d).sum(axis=0)
    contributions /= contributions.sum()
    cumulative = np.cumsum(contributions)
    n_keep = np.searchsorted(cumulative, threshold) + 1
    return np.arange(n_keep)


def main():
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:
        print(f"Error: {e}")
        print("Usage: python -m gait_classification.compute_fft_centroids max_samples=500")
        sys.exit(1)

    print("Loading and preprocessing data...")
    preprocess_functions = construct_filters(cfg)
    raw, y = load_and_preprocess_data(cfg, preprocess_functions=preprocess_functions)
    windows, labels = build_windowed_data(cfg, raw, y)

    print(f"Extracting FFT features for {len(windows)} windows...")
    fft_features_list = []
    for window in tqdm(windows, desc="FFT extraction"):
        fft_feat = extract_fft_features_from_window(window)
        fft_features_list.append(fft_feat)

    fft_features = np.array(fft_features_list, dtype=np.float32)
    print(f"FFT features shape: {fft_features.shape}")

    print("Selecting features by cumulative energy threshold (0.95)...")
    selected_indices = select_features_by_contribution(fft_features, threshold=0.95)
    fft_features_selected = fft_features[:, selected_indices]
    n_selected = len(selected_indices)
    n_total = fft_features.shape[1]
    print(f"Selected {n_selected} features out of {n_total}")

    print("Scaling features...")
    scaler = StandardScaler()
    fft_features_scaled = scaler.fit_transform(fft_features_selected)

    print("Computing FFT-based centroids...")
    centroids = {}
    for pid in np.unique(labels):
        mask = labels == pid
        centroid = fft_features_scaled[mask].mean(axis=0)
        centroids[pid] = centroid
        norm = np.linalg.norm(centroid)
        n_windows = (mask).sum()
        print(f"  Person {pid}: {n_windows} windows, centroid norm={norm:.3f}")

    # Save FFT scaler
    scaler_fft_path = os.path.join(os.path.dirname(__file__), cfg.checkpoint_dir, "scaler_fft.pkl")
    with open(scaler_fft_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved FFT scaler to {scaler_fft_path}")

    # Save FFT centroids
    centroids_fft_path = os.path.join(
        os.path.dirname(__file__), cfg.checkpoint_dir, "centroids_fft_centroids.pkl"
    )
    with open(centroids_fft_path, "wb") as f:
        pickle.dump(centroids, f)
    print(f"Saved {len(centroids)} FFT-based centroids to {centroids_fft_path}")


if __name__ == "__main__":
    main()
