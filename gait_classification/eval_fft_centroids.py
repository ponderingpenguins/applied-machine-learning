"""Evaluate EER of FFT + centroid method."""

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


def _compute_single_resample_metrics(
    test_feat_by_pid: dict[int, np.ndarray],
    known_pids: np.ndarray,
    unknown_pids: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """Compute EER/FAR/FRR for one enrollment-probe resample."""
    rng = np.random.default_rng(seed)
    feature_size = next(iter(test_feat_by_pid.values())).shape[1]

    centroids = np.zeros((len(known_pids), feature_size), dtype=np.float32)
    probe_known_features = []

    for i, pid in enumerate(known_pids):
        features = test_feat_by_pid[pid]
        indices = rng.permutation(len(features))
        enroll_count = max(1, len(indices) // 2)
        enroll_idx = indices[:enroll_count]
        probe_idx = indices[enroll_count:]

        centroids[i] = features[enroll_idx].mean(axis=0)
        if len(probe_idx) > 0:
            probe_known_features.append((pid, features[probe_idx]))

    distances_known = []
    distances_unknown = []

    for pid, features in probe_known_features:
        pid_idx = np.where(known_pids == pid)[0][0]
        centroid = centroids[pid_idx]
        dists = np.linalg.norm(features - centroid, axis=1)
        distances_known.extend(dists.tolist())

    for pid in unknown_pids:
        features = test_feat_by_pid[pid]
        min_dists = np.min(
            np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        distances_unknown.extend(min_dists.tolist())

    distances_known = np.asarray(distances_known, dtype=float)
    distances_unknown = np.asarray(distances_unknown, dtype=float)

    if len(distances_known) == 0 or len(distances_unknown) == 0:
        raise ValueError("Open-set evaluation requires both known probe trials and unknown trials.")

    thresholds = np.linspace(0, np.max(np.concatenate([distances_known, distances_unknown])), 100)
    fars = []
    frrs = []

    for threshold in thresholds:
        far = np.sum(distances_unknown < threshold) / len(distances_unknown)
        frr = np.sum(distances_known > threshold) / len(distances_known)
        fars.append(far)
        frrs.append(frr)

    fars = np.asarray(fars, dtype=float)
    frrs = np.asarray(frrs, dtype=float)
    eer_idx = int(np.argmin(np.abs(fars - frrs)))

    eer = float((fars[eer_idx] + frrs[eer_idx]) / 2)
    far = float(fars[eer_idx])
    frr = float(frrs[eer_idx])
    return eer, far, frr


def compute_far_frr_eer_fft(
    test_feat_by_pid: dict[int, np.ndarray],
    seed: int = 67,
    n_resamples: int = 10,
) -> tuple[float, float, float]:
    """
    Compute FAR, FRR, and EER using FFT-based centroid distance.
    """
    rng = np.random.default_rng(seed)
    eligible_known_pids = [pid for pid, feat in test_feat_by_pid.items() if len(feat) >= 2]
    if not eligible_known_pids:
        raise ValueError(
            "Open-set evaluation requires at least one held-out participant with two or more windows."
        )

    shuffled_known = rng.permutation(eligible_known_pids)
    split_idx = max(1, len(shuffled_known) // 2)
    if split_idx == len(shuffled_known):
        split_idx = len(shuffled_known) - 1

    known_pids = np.array(sorted(shuffled_known[:split_idx]))
    unknown_pids = np.array(sorted(shuffled_known[split_idx:]))
    remaining_pids = np.array(
        sorted([pid for pid in test_feat_by_pid.keys() if pid not in eligible_known_pids])
    )
    unknown_pids = np.concatenate([unknown_pids, remaining_pids])
    if len(unknown_pids) == 0:
        raise ValueError(
            "Open-set evaluation requires both known probe participants and unknown participants."
        )

    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    resample_eers = []
    resample_fars = []
    resample_frrs = []

    for _ in range(n_resamples):
        resample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        eer, far, frr = _compute_single_resample_metrics(
            test_feat_by_pid,
            known_pids,
            unknown_pids,
            seed=resample_seed,
        )
        resample_eers.append(eer)
        resample_fars.append(far)
        resample_frrs.append(frr)

    eer_mean = float(np.mean(resample_eers))
    far_mean = float(np.mean(resample_fars))
    frr_mean = float(np.mean(resample_frrs))

    return eer_mean, far_mean, frr_mean


def main():
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:
        print(f"Error: {e}")
        print("Usage: python eval_fft_centroids.py")
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
    print(f"Selected {len(selected_indices)} features out of {fft_features.shape[1]}")

    print("Scaling features...")
    scaler = StandardScaler()
    fft_features_scaled = scaler.fit_transform(fft_features_selected)

    print("Organizing features by participant...")
    test_feat_by_pid = {}
    for pid in np.unique(labels):
        mask = labels == pid
        test_feat_by_pid[pid] = fft_features_scaled[mask]

    print(f"Evaluating EER for {len(test_feat_by_pid)} participants...")
    eer, far, frr = compute_far_frr_eer_fft(test_feat_by_pid, seed=cfg.seed, n_resamples=10)

    print(f"\n{'='*50}")
    print(f"FFT + Centroid Method Results (n_resamples=10)")
    print(f"{'='*50}")
    print(f"EER (Equal Error Rate): {eer:.4f}")
    print(f"FAR (False Acceptance Rate): {far:.4f}")
    print(f"FRR (False Rejection Rate): {frr:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
