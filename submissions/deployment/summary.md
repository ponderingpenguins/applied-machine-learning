# Project Status Summary

Repository: [ponderingpenguins/applied-machine-learning](https://github.com/ponderingpenguins/applied-machine-learning)

The project is a deep learning-based gait authentication system using IMU sensor data. The current implementation includes a training pipeline, a deployed FastAPI API, and a web interface for inspecting and using the models.

## Implemented Components

Core ML Pipeline:

- Dual model architectures (LSTM and Transformer) for gait classification, with embedding-based inference.
- K-fold cross-validation training pipeline with configurable preprocessing (Butterworth, Kalman, FFT filters)
- Centroid-based classification with embedding extraction
- Automatic model storage and retrieval via Hugging Face Hub

Web API & Deployment:

- FastAPI backend serving inference endpoints (/model/{model_type}/encode-recording, /model/{model_type}/authenticate) with full documentation
- Interactive web UI with model selection, real-time sensor input, and result visualization
- Recent improvements: better UI/homepage, micro-interactions, green/red result screens, timer with audio feedback
- Docker Compose deployment configuration

Data Pipeline:

- Signal preprocessing with sliding window segmentation
- StandardScaler normalization (participant-aware)
- FFT-based feature selection
- Support for raw IMU data (accel x/y/z, gyro x/y/z)

Development & Quality:

- Pre-commit hooks with flake8 linting
- Code formatted with black
- Dependency management via uv package manager
- Multiple utility scripts (training curves plotting, centroid computation, HF model management)



## Development Notes

- Pre-commit hooks are configured, including flake8 and formatting checks.
- The project uses `uv` for dependency management.
- Supporting scripts exist for training curves, centroid computation, and Hugging Face model management.