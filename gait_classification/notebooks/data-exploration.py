import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy.signal import butter, sosfiltfilt

BASE = "Gait-Datasets-TIFS20/Dataset #1/train/Inertial Signals"


def load(name):
    return pd.read_csv(f"{BASE}/{name}.txt", sep=r"\s+", header=None).to_numpy()


ax_ = load("train_acc_x")
ay_ = load("train_acc_y")
az_ = load("train_acc_z")
gx_ = load("train_gyr_x")
gy_ = load("train_gyr_y")
gz_ = load("train_gyr_z")

sample_index = 1
fs = 50.0  # sampling rate
dt = 1.0 / fs
gyro_in_degrees = True  # set False if gyro is already in rad/s

# visualize raw data
t = np.arange(ax_.shape[1]) * dt
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ax_[sample_index], label="acc_x")
plt.plot(t, ay_[sample_index], label="acc_y")
plt.plot(t, az_[sample_index], label="acc_z")
plt.title("Accelerometer")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t, gx_[sample_index], label="gyr_x")
plt.plot(t, gy_[sample_index], label="gyr_y")
plt.plot(t, gz_[sample_index], label="gyr_z")
plt.title("Gyroscope")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (deg/s)")
plt.legend()
plt.tight_layout()
plt.savefig("gait_classification/figures/raw_signals.png")


# Smoothing via FFT low-pass filter
def low_pass_filter(signal, cutoff_freq, fs):
    N = signal.shape[0]
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_coeffs = np.fft.rfft(signal, axis=0)
    fft_coeffs[freqs > cutoff_freq] = 0
    return np.fft.irfft(fft_coeffs, n=N, axis=0)


def butterworth_filter(
    signal: np.ndarray, cutoff_freq: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to the input signal.

    Parameters:
    signal (np.ndarray): The input signal to be filtered.
    cutoff_freq (float): The cutoff frequency of the low-pass filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the Butterworth filter.

    Returns:
    np.ndarray: The filtered signal.
    """
    sos = butter(order, cutoff_freq / (0.5 * fs), output="sos")
    return sosfiltfilt(sos, signal, axis=0)


acc = np.stack([ax_[sample_index], ay_[sample_index], az_[sample_index]], axis=1)  # (N, 3)
gyr = np.stack([gx_[sample_index], gy_[sample_index], gz_[sample_index]], axis=1)
if gyro_in_degrees:
    gyr = np.deg2rad(gyr)
N = acc.shape[0]

# Apply low-pass filter to smooth the signals
cutoff_freq = 5.0  # Hz
acc_smooth = low_pass_filter(acc, cutoff_freq, fs)
gyr_smooth = low_pass_filter(gyr, cutoff_freq, fs)

acc_butterworth = butterworth_filter(acc, cutoff_freq, fs)
gyr_butterworth = butterworth_filter(gyr, cutoff_freq, fs)

# visualize smoothed data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, acc_smooth[:, 0], label="acc_x")
plt.plot(t, acc_smooth[:, 1], label="acc_y")
plt.plot(t, acc_smooth[:, 2], label="acc_z")
plt.title("Smoothed Accelerometer")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t, gyr_smooth[:, 0], label="gyr_x")
plt.plot(t, gyr_smooth[:, 1], label="gyr_y")
plt.plot(t, gyr_smooth[:, 2], label="gyr_z")
plt.title("Smoothed Gyroscope")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.tight_layout()
plt.savefig("gait_classification/figures/smoothed_signals.png")

# visualize smoothed data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, acc_butterworth[:, 0], label="acc_x")
plt.plot(t, acc_butterworth[:, 1], label="acc_y")
plt.plot(t, acc_butterworth[:, 2], label="acc_z")
plt.title("Butterworth Filtered Accelerometer")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t, gyr_smooth[:, 0], label="gyr_x")
plt.plot(t, gyr_smooth[:, 1], label="gyr_y")
plt.plot(t, gyr_smooth[:, 2], label="gyr_z")
plt.title("Smoothed Gyroscope")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.tight_layout()
plt.savefig("gait_classification/figures/butterworth_signals.png")
plt.close()
