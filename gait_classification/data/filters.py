"""
FFT filtering:
- Turn time series into frequency domain using FFT.
- Select top features that contribute to a certain cumulative energy threshold (e.g., 95%).
- This can reduce noise.
- Turn the selected features back to time domain using inverse FFT


Kalman filtering:
- A recursive algorithm that estimates the state of a dynamic system from noisy measurements.
- It consists of two steps: prediction and update.
- In the prediction step, the algorithm uses a model of the system to predict the next state and the uncertainty of that prediction.
- In the update step, the algorithm incorporates new measurements to refine the state estimate and reduce uncertainty.
- Kalman filtering can be used to smooth the time series data and reduce noise, which can improve the performance of the gait classification model.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from gait_classification.utils import TrainConfig


## Filter class
class Filter:
    """Base class for filters"""

    def apply(self, data):
        """Apply the filter to the data"""
        raise NotImplementedError("Filter subclasses must implement the apply method")

    def inverse(self, data):
        """Apply the inverse of the filter to the data"""
        raise NotImplementedError("Filter subclasses must implement the inverse method")

    def __call__(self, data):
        """Apply the filter to the data when the object is called"""
        return self.apply(data)


# FFT filter
class FFTFilter(Filter):
    """FFT filter"""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.selected_features = None

    def apply(self, data):
        """Apply the FFT filter to the data"""
        # Compute the FFT of the data
        fft_data = np.fft.fft(data, axis=0)
        # Compute the energy of each feature
        energy = np.sum(np.abs(fft_data) ** 2, axis=0)
        # Compute the cumulative energy
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        # Select features that contribute to the given cumulative energy threshold
        self.selected_features = np.where(cumulative_energy <= self.threshold)[0]
        # Keep only the selected features
        filtered_fft_data = fft_data[:, self.selected_features]
        # Inverse FFT to get back to time domain
        filtered_data = np.fft.ifft(filtered_fft_data, axis=0).real
        return filtered_data

    def inverse(self, data):
        """Apply the inverse of the FFT filter to the data"""
        if self.selected_features is None:
            raise ValueError("FFTFilter must be applied before calling inverse")
        # Compute the FFT of the data
        fft_data = np.fft.fft(data, axis=0)
        # Create an array of zeros for all features
        full_fft_data = np.zeros((fft_data.shape[0], len(self.selected_features)), dtype=complex)
        # Fill in the selected features with the filtered FFT data
        full_fft_data[:, self.selected_features] = fft_data
        # Inverse FFT to get back to time domain
        reconstructed_data = np.fft.ifft(full_fft_data, axis=0).real
        return reconstructed_data


class KalmanFilter(Filter):
    """Kalman filter"""

    def __init__(self, process_variance: float, measurement_variance: float):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_state = None
        self.estimated_covariance = None

    def _apply_1d(self, data: np.ndarray) -> np.ndarray:
        """Apply the Kalman filter to a single 1D time series."""
        data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        self.estimated_state = np.zeros((n_samples, n_features))
        self.estimated_covariance = np.zeros((n_samples, n_features, n_features))

        for i in range(n_samples):
            if i == 0:
                self.estimated_state[i] = data[i]
                self.estimated_covariance[i] = np.eye(n_features)
            else:
                predicted_state = self.estimated_state[i - 1]
                predicted_covariance = self.estimated_covariance[
                    i - 1
                ] + self.process_variance * np.eye(n_features)

                innovation = data[i] - predicted_state
                innovation_covariance = predicted_covariance + self.measurement_variance * np.eye(
                    n_features
                )
                kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
                self.estimated_state[i] = predicted_state + kalman_gain @ innovation
                self.estimated_covariance[i] = (
                    np.eye(n_features) - kalman_gain
                ) @ predicted_covariance

        return self.estimated_state.squeeze()

    def apply(self, data):
        """Apply the Kalman filter to the data"""
        if data.ndim == 1:
            return self._apply_1d(data)

        if data.ndim == 2:
            return self._apply_1d(data)

        if data.ndim != 3:
            raise ValueError(
                "KalmanFilter expects a 1D, 2D, or 3D array with time on the second axis"
            )

        filtered = np.empty_like(data, dtype=np.float32)
        for sample_idx in range(data.shape[0]):
            for channel_idx in range(data.shape[2]):
                filtered[sample_idx, :, channel_idx] = self._apply_1d(
                    data[sample_idx, :, channel_idx]
                )

        return filtered

    def inverse(self, data):
        """The kalman filter is not invertible, so don't implement this method"""
        raise NotImplementedError("KalmanFilter does not implement the inverse method")


class LowPassFFTFilter(FFTFilter):
    """Low-pass filter using FFT"""

    def __init__(self, cutoff_freq: float, fs: float):
        super().__init__(threshold=0.95)  # Use the default threshold for feature selection
        self.cutoff_freq = cutoff_freq
        self.fs = fs

    def apply(self, data):
        """Apply the low-pass filter to the data using FFT"""
        N = data.shape[1]
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
        fft_coeffs = np.fft.rfft(data, axis=1)
        fft_coeffs[:, freqs > self.cutoff_freq, ...] = 0
        return np.fft.irfft(fft_coeffs, n=N, axis=1)


class ButterworthLowPassFilter(Filter):
    """Butterworth low-pass filter"""

    def __init__(self, cutoff_freq: float, fs: float, order: int = 4):
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order

    def apply(self, data):
        """Apply the Butterworth low-pass filter to the data"""
        sos = butter(self.order, self.cutoff_freq / (0.5 * self.fs), output="sos")
        return sosfiltfilt(sos, data, axis=1)

    def inverse(self, data):
        """The Butterworth filter is not invertible, so don't implement this method"""
        raise NotImplementedError("ButterworthLowPassFilter does not implement the inverse method")


def construct_filters(cfg: TrainConfig) -> list[Filter]:
    """Construct filter objects based on the configuration."""
    filters = []
    for filter_name in cfg.preprocess_filters:
        if filter_name == "butterworth_lowpass":
            butterworth_filter = ButterworthLowPassFilter(
                cutoff_freq=cfg.cutoff_freq,
                fs=cfg.sampling_rate,
                order=cfg.filter_order,
            )
            filters.append(butterworth_filter)
        elif filter_name == "kalman":
            kalman_filter = KalmanFilter(process_variance=1e-5, measurement_variance=1e-2)
            filters.append(kalman_filter)
        elif filter_name == "fft_lowpass":
            fft_filter = LowPassFFTFilter(cutoff_freq=cfg.cutoff_freq, fs=cfg.sampling_rate)
            filters.append(fft_filter)

    return filters
