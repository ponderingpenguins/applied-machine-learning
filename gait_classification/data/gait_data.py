import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.fft import rfft
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from gait_classification.utils import TrainConfig


def load_signal(signal_dir: str, file_name: str):
    """Load a signal file into a NumPy array of shape (n_samples, n_timesteps)."""
    return pd.read_csv(f"{signal_dir}/{file_name}.txt", sep=r"\s+", header=None).to_numpy()


def _split_channel_file_name(file_name: str, split: str) -> str:
    """Convert a train_ channel filename into the matching split filename."""
    if file_name.startswith("train_"):
        return f"{split}_{file_name[len('train_') :]}"
    return file_name


def _load_split_data(cfg: TrainConfig, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one dataset split (train or test) into a raw tensor and labels."""
    base_dir = cfg.train_dir if split == "train" else cfg.test_dir
    signals_dir = f"{base_dir}/Inertial Signals"
    y_path = f"{base_dir}/y_{split}.txt"

    channels = []
    for channel_key in cfg.CHANNEL_FILES.keys():
        file_name = _split_channel_file_name(cfg.CHANNEL_FILES[channel_key], split)
        signal = load_signal(signals_dir, file_name)
        channels.append(signal)

    raw = np.stack(channels, axis=2)
    y = np.loadtxt(y_path, dtype=int)
    return raw, y


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
    train_raw, train_y = _load_split_data(cfg, "train")
    test_raw, test_y = _load_split_data(cfg, "test")

    raw = np.concatenate([train_raw, test_raw], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)

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


def participant_split(
    participants: np.ndarray, cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic participant-wise 70/15/15 split.
    Args:
        participants: Array of unique participant IDs.
        cfg: TrainConfig containing seed and split ratios.
    Returns:
        Tuple of (train_pids, val_pids, test_pids).
    """
    rng = np.random.default_rng(cfg.seed)
    shuffled = rng.permutation(participants)

    n = len(shuffled)
    train_cut = int(n * cfg.train_split)
    val_cut = int(n * (cfg.train_split + cfg.val_split))

    train_pids = shuffled[:train_cut]
    val_pids = shuffled[train_cut:val_cut]
    test_pids = shuffled[val_cut:]

    # Sanity check to ensure splits are disjoint and cover all participants
    assert len(set(train_pids) & set(val_pids)) == 0, "Train and Val splits overlap!"
    assert len(set(train_pids) & set(test_pids)) == 0, "Train and Test splits overlap!"
    assert len(set(val_pids) & set(test_pids)) == 0, "Val and Test splits overlap!"
    assert len(train_pids) + len(val_pids) + len(test_pids) == n, (
        "Splits do not cover all participants!"
    )

    return train_pids, val_pids, test_pids


def make_kfold_splits(
    participants: np.ndarray, cfg: TrainConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Make participant-wise train/validation folds inside a development pool, dont know if we want nested or something else so this for now"""
    rng = np.random.default_rng(cfg.seed)
    shuffled = rng.permutation(participants)
    folds = np.array_split(shuffled, cfg.n_folds)

    splits = []
    for k in range(cfg.n_folds):
        val_pids = folds[k]
        train_pids = np.concatenate([folds[j] for j in range(cfg.n_folds) if j != k])
        splits.append((train_pids, val_pids))

    return splits


def fit_scaler(
    cfg: TrainConfig, windows: np.ndarray, labels: np.ndarray, train_pids: np.ndarray
) -> StandardScaler:
    """
    Fit a StandardScaler on training windows only.
    Args:
        cfg: TrainConfig object.
        windows: Array of shape (N_windows, seq_len, 6).
        labels: Array of shape (N_windows,) with participant IDs.
        train_pids: Array of participant IDs in the training set.
    Returns:
        Fitted StandardScaler.
    """
    train_mask = np.isin(labels, train_pids)
    train_windows = windows[train_mask]

    train_windows_flat = train_windows.reshape(-1, 6)

    scaler = StandardScaler()  # z-score normalization
    scaler.fit(train_windows_flat)

    # Pickle and save the scaler for later use in inference
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(f"{cfg.checkpoint_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return scaler


def apply_scaler(windows: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply a fitted StandardScaler to windows.
    Args:
        windows: Array of shape (N_windows, seq_len, 6).
        scaler: Fitted StandardScaler.
    Returns:
        Scaled windows of the same shape.
    """
    n_windows, seq_len = windows.shape[:2]
    windows_flat = windows.reshape(-1, 6)
    windows_scaled_flat = scaler.transform(windows_flat)
    windows_scaled = windows_scaled_flat.reshape(n_windows, seq_len, 6)
    return windows_scaled


def generate_triplets(
    labels: np.ndarray,
    pids: np.ndarray,
    n_neg_per_pair: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[int, int, int]]:
    """
    Generate offline triplets for metric learning.
    For each participant, sample random anchor-positive pairs and negatives from other participants.
    Args:
        labels: Array of window participant IDs.
        pids: Array of participant IDs to consider.
        n_neg_per_pair: Number of negatives to sample per anchor-positive pair.
        rng: Random number generator.
    Returns:
        List of (anchor_idx, pos_idx, neg_idx) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    idx_by_pid = {}
    for pid in pids:
        idx_by_pid[pid] = np.where(labels == pid)[0].tolist()

    triplets = []

    for pid in tqdm(pids, desc="Generating triplets"):
        pos_indices = idx_by_pid[pid]
        if len(pos_indices) < 2:
            continue

        neg_pids = [p for p in pids if p != pid]
        if not neg_pids:
            continue

        neg_indices = []
        for neg_pid in neg_pids:
            neg_indices.extend(idx_by_pid[neg_pid])

        n_samples_per_anchor = min(5, len(pos_indices) // 2 + 1)

        for anchor_idx in tqdm(pos_indices, desc=f"Processing PID {pid}", leave=False):
            pos_candidates = [idx for idx in pos_indices if idx != anchor_idx]
            if not pos_candidates:
                continue

            sampled_pos = rng.choice(
                pos_candidates,
                size=min(n_samples_per_anchor, len(pos_candidates)),
                replace=False,
            )

            for pos_idx in sampled_pos:
                for _ in range(n_neg_per_pair):
                    neg_idx = rng.choice(neg_indices)
                    triplets.append((anchor_idx, pos_idx, neg_idx))

    return triplets


class GaitWindowDataset(Dataset):
    """Dataset for online triplet mining on gait signals."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        Args:
            windows: Array of shape (N, seq_len, 6) with windowed signals.
            labels: Array of shape (N,) with participant IDs.
        """
        self.windows = torch.tensor(windows, dtype=torch.float32)
        
        if not isinstance(labels, torch.Tensor):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.windows[idx], self.labels[idx]


