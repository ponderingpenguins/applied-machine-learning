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

The API will be available at [http://localhost:8000](http://localhost:8000), with interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

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

### Training Configuration

Training uses OmegaConf-style CLI overrides, so any field in `TrainConfig` can be changed from the command line just as we do above

Possible parameters:

- Optimization: `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`, `dropout`, `early_stopping_patience`, `early_stopping_min_delta`, `evaluation_resamples`
- Model / loss: `model_type`, `loss_type`, `embedding_size`, `triplet_margin`, `cosface_margin`, `cosface_scale`
- Architecture: `lstm_hidden_size`, `lstm_num_layers`, `transformer_d_model`, `transformer_nhead`, `transformer_num_layers`, `transformer_dim_feedforward`
- Data split and windowing: `n_folds`, `train_split`, `val_split`, `seq_len`, `window_stride`, `max_samples`, `seed`
- Preprocessing: `preprocess_filters`, `fft_threshold`, `sampling_rate`, `cutoff_freq`, `filter_order`
- Paths and outputs: `data_dir`, `train_dir`, `test_dir`, `signals_dir`, `y_path`, `checkpoint_dir`, `figures_dir`, `push_to_hf`

### Finetuning

The finetuning script runs a grid search over a smaller set of promising hyperparameters and writes the results to `gait_classification/checkpoints/finetuning/`.

```bash
# CosFace finetuning for the Transformer
uv run python -m gait_classification.finetune model_type=transformer loss_type=cosface

# Triplet finetuning for the LSTM
uv run python -m gait_classification.finetune model_type=lstm loss_type=triplet
```

Finetuning uses the same configuration keys as training, plus the search is centered around:

- Shared search space: `learning_rate`, `weight_decay`, `dropout`, `embedding_size`, `n_folds`, `batch_size`, `num_epochs`
- Triplet-specific search: `triplet_margin`
- CosFace-specific search: `cosface_margin`, `cosface_scale`
- LSTM search: `lstm_hidden_size`, `lstm_num_layers`
- Transformer search: `transformer_d_model`, `transformer_nhead`, `transformer_num_layers`, `transformer_dim_feedforward`

## Hugging Face

```bash
# Authenticate
huggingface-cli login 

or 

hf auth login

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
