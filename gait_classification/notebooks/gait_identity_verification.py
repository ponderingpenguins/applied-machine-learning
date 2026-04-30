"""
Gait-based person identification using accelerometer & gyroscope signals.

Approach:
  - Extract FFT features from 6-channel inertial data (3-axis accel, 3-axis gyro)
  - Select features by cumulative energy contribution (threshold-based)
  - Build per-participant centroids on training data
  - Evaluate via k-fold CV with participant-wise splits
  - Tune distance threshold on dev set to maximize separation of known vs unknown users
  - Report Equal Error Rate (EER), False Acceptance Rate (FAR), False Rejection Rate (FRR)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RANDOM_STATE = 67  # Six seven...
FFT_THRESHOLD = 0.95  # Chosen in the preliminary data look notebook by plotting the t-sne of the FFT features to see which threshold gives the best separation.
# Would be nice to use the elbow method.
N_FOLDS = 5

TRAIN_DIR = "../../Gait-Datasets-TIFS20/Dataset #1/train"
SIGNALS_DIR = f"{TRAIN_DIR}/Inertial Signals"
Y_PATH = f"{TRAIN_DIR}/y_train.txt"

CHANNEL_FILES = {
    "ACCx": "train_acc_x",
    "ACCy": "train_acc_y",
    "ACCz": "train_acc_z",
    "GYRx": "train_gyr_x",
    "GYRy": "train_gyr_y",
    "GYRz": "train_gyr_z",
}


def load_signal(file_name):
    """Load a signal file into a NumPy array of shape (n_samples, n_timesteps)."""
    return pd.read_csv(
        f"{SIGNALS_DIR}/{file_name}.txt", sep=r"\s+", header=None
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


def nearest_cluster_distance(X, centroids):
    """Compute distance from each sample in X to nearest centroid."""
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    min_idx = dists.argmin(axis=1)
    min_dist = dists[np.arange(len(X)), min_idx]
    return min_dist, min_idx


def tune_threshold(dist_pos, dist_neg):
    """Find the distance threshold that best separates positive (known) from negative (unknown) samples on the dev set."""
    all_dists = np.concatenate([dist_pos, dist_neg])
    labels = np.concatenate([np.ones(len(dist_pos)), np.zeros(len(dist_neg))])
    best_t, best_f1 = 0.0, 0.0
    for t in np.percentile(all_dists, np.linspace(1, 99, 200)):
        preds = (all_dists <= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def compute_far_frr_curve(dist, nearest_idx, sample_labels, binary_labels, known):
    """Return (thresholds, macro_far%, macro_frr%) curves for the test set."""
    thresholds = np.percentile(dist, np.linspace(0, 100, 300))
    neg_mask = binary_labels == 0
    fars, frrs = [], []
    for t in thresholds:
        preds = (dist <= t).astype(int)
        far_sum, frr_sum, count = 0.0, 0.0, 0
        for i, pid in enumerate(known):
            pos_mask = (binary_labels == 1) & (sample_labels == pid)
            n_pos = pos_mask.sum()
            if n_pos == 0:
                continue
            n_neg = neg_mask.sum()
            frr_p = (preds[pos_mask] == 0).sum() / n_pos
            far_p = (
                ((preds[neg_mask] == 1) & (nearest_idx[neg_mask] == i)).sum() / n_neg
                if n_neg > 0
                else 0.0
            )
            far_sum += far_p
            frr_sum += frr_p
            count += 1
        fars.append(far_sum / count * 100)
        frrs.append(frr_sum / count * 100)
    return thresholds, np.array(fars), np.array(frrs)


def eer_from_curves(fars, frrs):
    """Find the Equal Error Rate (EER) where FAR and FRR are closest."""
    idx = np.argmin(np.abs(fars - frrs))
    return (fars[idx] + frrs[idx]) / 2


def run_fold(fft_raw, y, known, unknown, fold_rng):
    """Run one fold: fit scaler/features on known-train, tune on dev, evaluate on test."""
    # Split known participants' windows 60 / 20 / 20 (I found that we needed more participants in dev/test to get stable tuning and evaluation metrics, so I went with 60/20/20 instead of 70/15/15 or 80/10/10)
    known_idx = np.where(np.isin(y, known))[0]
    perm = fold_rng.permutation(len(known_idx))
    known_idx = known_idx[perm]
    n_k = len(known_idx)
    train_idx = known_idx[: int(n_k * 0.60)]
    dev_pos_idx = known_idx[int(n_k * 0.60) : int(n_k * 0.80)]
    test_pos_idx = known_idx[int(n_k * 0.80) :]

    # Unknown participants: split half to dev-neg, half to test-neg
    unk_idx = np.where(np.isin(y, unknown))[0]
    mid = len(unk_idx) // 2
    dev_neg_idx = unk_idx[:mid]
    test_neg_idx = unk_idx[mid:]

    # Feature selection and scaling
    n_keep = select_features_by_contribution(fft_raw[train_idx], FFT_THRESHOLD)
    fft = fft_raw[:, :n_keep].copy()

    scaler = StandardScaler()
    fft[train_idx] = scaler.fit_transform(fft[train_idx])
    fft[dev_pos_idx] = scaler.transform(fft[dev_pos_idx])
    fft[dev_neg_idx] = scaler.transform(fft[dev_neg_idx])
    fft[test_pos_idx] = scaler.transform(fft[test_pos_idx])
    fft[test_neg_idx] = scaler.transform(fft[test_neg_idx])

    # Centroids from train windows
    centroids = np.array(
        [fft[train_idx][y[train_idx] == pid].mean(axis=0) for pid in known]
    )

    # Tune threshold on dev (the threshold is a distance to nearest centroid, so we want to find the point that best separates dev-pos from dev-neg, the EER point on the dev set)
    dev_idx = np.concatenate([dev_pos_idx, dev_neg_idx])
    dev_dist, _ = nearest_cluster_distance(fft[dev_idx], centroids)
    threshold, _ = tune_threshold(
        dev_dist[: len(dev_pos_idx)], dev_dist[len(dev_pos_idx) :]
    )

    # Evaluate on test
    test_idx = np.concatenate([test_pos_idx, test_neg_idx])
    test_binary = np.concatenate(
        [np.ones(len(test_pos_idx)), np.zeros(len(test_neg_idx))]
    )
    test_pids = np.concatenate(
        [y[test_pos_idx], np.zeros(len(test_neg_idx), dtype=int)]
    )
    test_dist, test_nearest = nearest_cluster_distance(fft[test_idx], centroids)

    thresholds, fars, frrs = compute_far_frr_curve(
        test_dist, test_nearest, test_pids, test_binary, known
    )
    eer = eer_from_curves(fars, frrs)

    # Per-participant FAR/FRR at tuned threshold
    preds = (test_dist <= threshold).astype(int)
    neg_mask = test_binary == 0
    rows = []
    for i, pid in enumerate(known):
        pos_mask = (test_binary == 1) & (test_pids == pid)
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        frr_p = (preds[pos_mask] == 0).sum() / n_pos if n_pos > 0 else float("nan")
        far_p = (
            ((preds[neg_mask] == 1) & (test_nearest[neg_mask] == i)).sum() / n_neg
            if n_neg > 0
            else float("nan")
        )
        rows.append({"participant": pid, "FRR": frr_p, "FAR": far_p})

    per_part = pd.DataFrame(rows).set_index("participant")
    macro_frr = per_part["FRR"].mean()
    macro_far = per_part["FAR"].mean()

    return eer, macro_far, macro_frr, thresholds, fars, frrs


def plot_kfold_curves(all_thresholds, all_fars, all_frrs, eers):
    """Plot FAR/FRR curves for all folds, with EER summary in title."""
    _, ax = plt.subplots(figsize=(9, 6))
    for ts, fars, frrs in zip(all_thresholds, all_fars, all_frrs):
        # Normalise thresholds to [0, 100] so folds are comparable on x-axis
        ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-9) * 100
        ax.plot(ts_norm, frrs, color="red", alpha=0.3, linewidth=1)
        ax.plot(ts_norm, fars, color="blue", alpha=0.3, linewidth=1)

    mean_eer = np.mean(eers)
    std_eer = np.std(eers)

    # Dummy lines for legend
    ax.plot([], [], color="red", label="FRR (per fold)")
    ax.plot([], [], color="blue", label="FAR (per fold)")
    ax.set_xlabel("Normalised Distance Threshold")
    ax.set_ylabel("Error (%)")
    ax.set_title(
        f"FAR / FRR - {N_FOLDS}-fold participant-wise CV\nEER = {mean_eer:.2f}% ± {std_eer:.2f}%"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../figures/far_frr_kfold.png", dpi=150)
    print("Plot saved to ../figures/far_frr_kfold.png")


def main():
    """Main function to run the gait identity verification classification."""

    print("Loading signals...")
    signals = {label: load_signal(fname) for label, fname in CHANNEL_FILES.items()}
    y = pd.read_csv(Y_PATH, header=None).squeeze("columns").to_numpy(dtype=int)
    print(f"Total samples: {len(y)}")

    print("Extracting FFT features (once, shared across folds)...")
    fft_raw = extract_fft_features(signals, len(y))

    # Shuffle participants and assign to k folds
    participants = np.unique(y)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(participants)
    folds = np.array_split(participants, N_FOLDS)

    eers, macro_fars, macro_frrs = [], [], []
    all_thresholds, all_fars_curves, all_frrs_curves = [], [], []

    for k, test_fold in tqdm(enumerate(folds), total=N_FOLDS, desc="CV folds"):
        known = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        unknown = test_fold
        print(
            f"\n=== Fold {k + 1}/{N_FOLDS}  known={len(known)}  unknown={len(unknown)} ==="
        )

        fold_rng = np.random.default_rng(RANDOM_STATE + k)
        eer, mfar, mfrr, ts, fars, frrs = run_fold(fft_raw, y, known, unknown, fold_rng)

        eers.append(eer)
        macro_fars.append(mfar)
        macro_frrs.append(mfrr)
        all_thresholds.append(ts)
        all_fars_curves.append(fars)
        all_frrs_curves.append(frrs)

        print(f"  EER: {eer:.2f}%  Macro FAR: {mfar:.4f}  Macro FRR: {mfrr:.4f}")

    print("\n=== Cross-validation summary ===")
    print(f"  EER       : {np.mean(eers):.2f}% ± {np.std(eers):.2f}%")
    print(f"  Macro FAR : {np.mean(macro_fars):.4f} ± {np.std(macro_fars):.4f}")
    print(f"  Macro FRR : {np.mean(macro_frrs):.4f} ± {np.std(macro_frrs):.4f}")

    plot_kfold_curves(all_thresholds, all_fars_curves, all_frrs_curves, eers)


if __name__ == "__main__":
    main()
