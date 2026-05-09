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


## Filter class
class Filter:
    """Base class for filters"""

    def apply(self, data):
        """Apply the filter to the data"""
        raise NotImplementedError("Filter subclasses must implement the apply method")

    def inverse(self, data):
        """Apply the inverse of the filter to the data"""
        raise NotImplementedError("Filter subclasses must implement the inverse method")


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
        full_fft_data = np.zeros(
            (fft_data.shape[0], len(self.selected_features)), dtype=complex
        )
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

    def apply(self, data):
        """Apply the Kalman filter to the data"""
        n_samples, n_features = data.shape
        self.estimated_state = np.zeros((n_samples, n_features))
        self.estimated_covariance = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            if i == 0:
                # Initialize the state and covariance
                self.estimated_state[i] = data[i]
                self.estimated_covariance[i] = np.eye(n_features)
            else:
                # Prediction step
                predicted_state = self.estimated_state[i - 1]
                predicted_covariance = self.estimated_covariance[
                    i - 1
                ] + self.process_variance * np.eye(n_features)
                # Update step
                kalman_gain = predicted_covariance / (
                    predicted_covariance
                    + self.measurement_variance * np.eye(n_features)
                )
                self.estimated_state[i] = predicted_state + kalman_gain @ (
                    data[i] - predicted_state
                )
                self.estimated_covariance[i] = (
                    np.eye(n_features) - kalman_gain
                ) @ predicted_covariance
        return self.estimated_state

    def inverse(self, data):
        """The kalman filter is not invertible, so don't implement this method"""
        raise NotImplementedError("KalmanFilter does not implement the inverse method")
