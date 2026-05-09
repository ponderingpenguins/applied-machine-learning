from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt

def low_pass_filter(signal: np.ndarray, cutoff_freq: float, fs: float) -> np.ndarray:
    """
    Apply a low-pass filter to the input signal using FFT.

    Parameters:
    signal (np.ndarray): The input signal to be filtered.
    cutoff_freq (float): The cutoff frequency of the low-pass filter.
    fs (float): The sampling frequency of the signal.

    Returns:
    np.ndarray: The filtered signal.
    """
    N = signal.shape[0]
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_coeffs = np.fft.rfft(signal, axis=0)
    fft_coeffs[freqs > cutoff_freq] = 0
    return np.fft.irfft(fft_coeffs, n=N, axis=0)


def butterworth_filter(signal: np.ndarray, cutoff_freq: float, fs: float, order: int = 4) -> np.ndarray:
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
    sos = butter(order, cutoff_freq / (0.5 * fs), output='sos')
    return sosfiltfilt(sos, signal, axis=0)

# testing purposes
if __name__ == "__main__":
    n = 201
    t = np.linspace(0, 1, n)
    x = 1 + (t < 0.5) - 0.25*t**2 + 0.05*np.random.standard_normal(n)
    
    filtered_signal_low_pass = low_pass_filter(x, cutoff_freq=5.0, fs=50.0)
    
    filter_coeffs = butter(4, 5.0 / (0.5 * 50.0), output='sos')
    
    filtered_signal_butter = sosfiltfilt(filter_coeffs, x)
    
    plt.plot(t, x, alpha=0.5, label='x(t)')
    plt.plot(t, filtered_signal_low_pass, label='Filtered (Low-pass)')
    plt.plot(t, filtered_signal_butter, label='Filtered (Butterworth)')
    plt.legend(framealpha=1, shadow=True)
    plt.grid(alpha=0.25)
    plt.xlabel('t')
    output_path = Path(__file__).resolve().parents[1] / "figures" / "preprocessing_demo.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
