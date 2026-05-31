"""
This was donw with a help of LLM, I wanted to quickly explore some simple interpretability methods to understand the FFT feature space used for gait verification.

xAI analysis of the FFT feature space used for gait verification.

Method 1: Discriminative frequency spectrum
  For each FFT bin, compute the between-participant variance vs within-participant
  variance (F-ratio). High F-ratio = that frequency separates people well.
  Plotted as a spectrum so peaks can be mapped back to Hz.

Method 2: Centroid visualisation (PCA + t-SNE)
  Project the per-participant centroids to 2D and colour by participant.
  Tight, separated clusters explain the low EER.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FFT_THRESHOLD = 0.95
SAMPLING_RATE = 50  # Hz
N_FFT_COEFFS = 500

# Construct paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, "../../Gait-Datasets-TIFS20/Dataset #1/train")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "../figures")
SIGNALS_DIR = os.path.join(TRAIN_DIR, "Inertial Signals")
Y_PATH = os.path.join(TRAIN_DIR, "y_train.txt")

CHANNEL_FILES = {
    "ACCx": "train_acc_x",
    "ACCy": "train_acc_y",
    "ACCz": "train_acc_z",
    "GYRx": "train_gyr_x",
    "GYRy": "train_gyr_y",
    "GYRz": "train_gyr_z",
}
CHANNELS = list(CHANNEL_FILES.keys())


def load_signal(file_name):
    return pd.read_csv(f"{SIGNALS_DIR}/{file_name}.txt", sep=r"\s+", header=None).to_numpy()


def extract_fft_features(signals, n_samples):
    features = []
    for i in range(n_samples):
        row = []
        for arr in signals.values():
            yf = rfft(arr[i, :])
            row.extend(np.abs(yf[:N_FFT_COEFFS]))
        features.append(row)
    return np.array(features)


def select_features_by_contribution(fft_features, threshold):
    contributions = np.abs(fft_features).sum(axis=0)
    contributions /= contributions.sum()
    cumulative = np.cumsum(contributions)
    return np.searchsorted(cumulative, threshold) + 1


# ---------------------------------------------------------------------------
# Method 1: Discriminative frequency spectrum (F-ratio)
# ---------------------------------------------------------------------------


def compute_f_ratio(fft_features, y):
    """
    For each feature (FFT bin), compute the Fisher F-ratio:
      F = between-class variance / within-class variance
    Higher F = more discriminative.
    """
    participants = np.unique(y)
    grand_mean = fft_features.mean(axis=0)

    between = np.zeros(fft_features.shape[1])
    within = np.zeros(fft_features.shape[1])

    for pid in participants:
        mask = y == pid
        n_p = mask.sum()
        class_mean = fft_features[mask].mean(axis=0)
        between += n_p * (class_mean - grand_mean) ** 2
        within += ((fft_features[mask] - class_mean) ** 2).sum(axis=0)

    between /= len(participants) - 1
    within /= len(y) - len(participants)

    f_ratio = np.where(within > 0, between / within, 0.0)
    return f_ratio


def plot_f_ratio_spectrum(
    f_ratio,
    n_keep,
    n_total_features,
    title="Discriminative Frequency Spectrum (F-ratio)",
):
    """
    Plot F-ratio per FFT bin, x-axis in Hz, one panel per channel.
    Only the kept bins (after energy filtering) are shown.
    """
    bins_per_channel = n_total_features // len(CHANNELS)
    freqs = rfftfreq(bins_per_channel * 2 - 2, d=1 / SAMPLING_RATE)[:bins_per_channel]

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()

    for ch_idx, (ch_name, ax) in enumerate(zip(CHANNELS, axes)):
        start = ch_idx * bins_per_channel
        # Only bins that survived the energy filter
        kept_end = min(n_keep - start, bins_per_channel)
        if kept_end <= 0:
            ax.set_title(f"{ch_name} (all filtered out)")
            continue

        ch_f = f_ratio[start : start + kept_end]
        ch_freqs = freqs[:kept_end]

        ax.fill_between(ch_freqs, ch_f, alpha=0.4, color="steelblue")
        ax.plot(ch_freqs, ch_f, color="steelblue", linewidth=0.8)

        peak_idx = ch_f.argmax()
        ax.axvline(
            ch_freqs[peak_idx],
            color="red",
            linestyle="--",
            linewidth=0.8,
            label=f"peak {ch_freqs[peak_idx]:.1f} Hz",
        )
        ax.set_title(ch_name)
        ax.set_ylabel("F-ratio")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[-2:]:
        ax.set_xlabel("Frequency (Hz)")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "xai_f_ratio_spectrum.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Method 2: Centroid visualisation (PCA + t-SNE)
# ---------------------------------------------------------------------------


def plot_centroids_pca(centroids, known, title="PCA of participant centroids"):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = pca.fit_transform(centroids)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=np.arange(len(known)),
        cmap="tab20",
        s=60,
        alpha=0.85,
        linewidths=0.4,
        edgecolors="k",
    )
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="Participant index")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "xai_centroids_pca.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved {output_path}")


def plot_centroids_tsne(centroids, known, title="t-SNE of participant centroids"):
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(known) // 3),
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )
    emb = tsne.fit_transform(centroids)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=np.arange(len(known)),
        cmap="tab20",
        s=60,
        alpha=0.85,
        linewidths=0.4,
        edgecolors="k",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="Participant index")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "xai_centroids_tsne.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading signals...")
    signals = {label: load_signal(fname) for label, fname in CHANNEL_FILES.items()}
    y = pd.read_csv(Y_PATH, header=None).squeeze("columns").to_numpy(dtype=int)
    print(f"Total samples: {len(y)}")

    print("Extracting FFT features...")
    fft_raw = extract_fft_features(signals, len(y))

    # Use all data to build centroids for xAI (no train/test split needed here —
    # we are analysing the feature space, not evaluating generalisation).
    n_keep = select_features_by_contribution(fft_raw, FFT_THRESHOLD)
    fft_trimmed = fft_raw[:, :n_keep]
    print(f"Kept {n_keep} / {fft_raw.shape[1]} FFT features")

    scaler = StandardScaler()
    fft_scaled = scaler.fit_transform(fft_trimmed)

    participants = np.unique(y)
    centroids = np.array([fft_scaled[y == pid].mean(axis=0) for pid in participants])
    print(f"Centroids: {centroids.shape}")

    # --- Method 1: F-ratio spectrum ---
    print("\nComputing F-ratio per FFT bin...")
    f_ratio = compute_f_ratio(fft_scaled, y)
    plot_f_ratio_spectrum(f_ratio, n_keep, fft_raw.shape[1])

    # --- Method 2: Centroid projections ---
    print("\nPCA projection of centroids...")
    plot_centroids_pca(centroids, participants)

    print("\nt-SNE projection of centroids...")
    plot_centroids_tsne(centroids, participants)


if __name__ == "__main__":
    main()
