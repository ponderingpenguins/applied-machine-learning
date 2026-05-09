from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from gait_classification.data.filters import (
    ButterworthLowPassFilter,
    KalmanFilter,
    LowPassFFTFilter,
)
from gait_classification.data.gait_data import load_and_preprocess_data
from gait_classification.utils import TrainConfig

# testing purposes
if __name__ == "__main__":
    cfg = OmegaConf.structured(TrainConfig)

    lowpass_filter = LowPassFFTFilter(cutoff_freq=cfg.cutoff_freq, fs=cfg.sampling_rate)
    basic_fft_filter = LowPassFFTFilter(
        cutoff_freq=cfg.cutoff_freq, fs=cfg.sampling_rate
    )
    butter_filter = ButterworthLowPassFilter(
        cutoff_freq=cfg.cutoff_freq, fs=cfg.sampling_rate, order=cfg.filter_order
    )
    kalman_filter = KalmanFilter(process_variance=1e-3, measurement_variance=1e-3)

    # load sample signals from the dataset for demonstration
    raw, y = load_and_preprocess_data(cfg, preprocess_functions=[])

    # Create figure with grid of subplots
    n_samples_to_show = 6
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for sample_idx in range(min(n_samples_to_show, raw.shape[0])):
        x = raw[sample_idx, :, 0]  # Take the sample and first channel
        t = np.arange(len(x)) / cfg.sampling_rate  # Time vector based on sampling rate

        filtered_signal_low_pass = lowpass_filter.apply(x)
        filtered_signal_butter = butter_filter.apply(x)
        filtered_signal_basic_fft = basic_fft_filter.apply(x)
        filtered_signal_kalman = kalman_filter.apply(x)

        ax = axes[sample_idx]
        ax.plot(t, x, alpha=0.5, label="Original Signal", linewidth=1.5)
        ax.plot(t, filtered_signal_low_pass, label="Low-pass FFT", linewidth=1)
        ax.plot(t, filtered_signal_butter, label="Butterworth", linewidth=1)
        ax.plot(t, filtered_signal_basic_fft, label="Basic FFT", linewidth=1)
        ax.plot(t, filtered_signal_kalman, label="Kalman", linewidth=1)
        ax.legend(framealpha=0.9, shadow=True, fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_xlabel("t (s)", fontsize=9)
        ax.set_ylabel("x(t)", fontsize=9)
        ax.set_title(f"Sample {sample_idx} (Class: {y[sample_idx]})", fontsize=10)

    plt.suptitle(
        "Comparison of Filters on Multiple Samples", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    output_path = cfg.figures_dir + "preprocessing_demo.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
