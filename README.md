# Gait Classification

Deep learning–based gait recognition using IMU sensor data (LSTM and Transformer models).

Based on [Deep Learning-Based Gait Recognition Using Smartphones in the Wild](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones) by Zou Q, Wang Y, Zhao Y, Wang Q and Li Q.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Install Dependencies

```bash
uv sync
```

## Dataset Setup

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1KOm-zROeOZH3e2tqYUpHAvIaBZSJGFm_) and place the zip in the repo root, then unzip:

```bash
unzip Gait-Datasets-TIFS20.zip
cd Gait-Datasets-TIFS20 && unzip "Dataset #1.zip" && cd ..
```

## Launch the API

**Development mode** (auto-reload on file changes):

```bash
uv run uvicorn gait_classification.api:app --reload
```

The API will be available at <http://localhost:8000>, with interactive docs at <http://localhost:8000/docs>.

**Custom host/port:**

```bash
uv run uvicorn gait_classification.api:app --host 0.0.0.0 --port 8000
```

**Docker Compose** (recommended for deployment):

```bash
docker-compose up --build
```

> The API loads model checkpoints from `gait_classification/checkpoints/` on startup. Train a model first (see below) or pull existing checkpoints from Hugging Face.

## Training

```bash
# LSTM
uv run python -m gait_classification.train model_type=lstm

# Transformer
uv run python -m gait_classification.train model_type=transformer

# Fast smoke-test (small dataset, no filters)
uv run python -m gait_classification.train max_samples=500 batch_size=128 model_type=lstm 'preprocess_filters=[]' n_folds=1
```

Models are saved to `gait_classification/checkpoints/`. To also push to Hugging Face:

```bash
uv run python -m gait_classification.train model_type=transformer push_to_hf=true
```

## Hugging Face

```bash
# Authenticate
huggingface-cli login

# Push existing checkpoints
uv run python push_models_to_hf.py
```

Models are stored at [slugroom/gait-classification-models](https://huggingface.co/slugroom/gait-classification-models).

## Linting

```bash
flake8 gait_classification
black gait_classification --line-length 100
pre-commit run --all-files
```
