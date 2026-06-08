# Gait Classification

## [Try it Live, gait-authentication.sigurdurhaukur.com](https://gait-authentication.sigurdurhaukur.com)

Deep learning–based gait recognition using IMU sensor data (Transformer, LSTM, and FFT + Centroid models) with a FastAPI backend for authentication. The API provides endpoints for encoding gait recordings into embeddings and authenticating users based on their gait patterns. The project includes training scripts, model checkpoints, and Hugging Face integration for model sharing.

Inspired by [Deep Learning-Based Gait Recognition Using Smartphones in the Wild](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones) by Zou Q, Wang Y, Zhao Y, Wang Q and Li Q.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Docker](https://docs.docker.com/get-docker/) (optional, for deployment)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start#installing-the-cli) For downloading and pushing model checkpoints (optional, unless you want to download our trained models or push your own)

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

The API will be available at [http://localhost:8050](http://localhost:8050), with interactive docs at [http://localhost:8050/docs](http://localhost:8050/docs).

**Custom host/port:**

```bash
uv run uvicorn gait_classification.api:app --host 0.0.0.0 --port 8050
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

### Hyperparameter Finetuning

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

## Evaluation

Evaluate the saved Transformer, LSTM, FFT + centroid baseline, and IID-random control on the same participant-disjoint holdout:

```bash
uv run python -m gait_classification.benchmark_models
```

By default it loads the following which needs to be setup manually. Results are written by default to `benchmark_results/benchmark_scores.json`.:

- `checkpoints/final_model_transformer.pt`
- `checkpoints/best_model_lstm.pt`

Checkpoint and output paths can both be overridden if you need/want:

```bash
uv run python -m gait_classification.benchmark_models \
  --transformer-checkpoint checkpoints/final_model_transformer.pt \
  --lstm-checkpoint checkpoints/best_model_lstm.pt \
  --output benchmark_results/benchmark_scores.json \
  --n-bootstrap 2000
```


To add participant/window bootstrap 95% confidence intervals:

```bash
uv run python -m gait_classification.benchmark_models --n-bootstrap 2000
```

Plot cross-validation training loss and validation EER/FAR/FRR from saved fold histories:

```bash
uv run python -m gait_classification.plot_training_curves \
  --checkpoint-dir checkpoints/transformer_history_unseeded \
  --output benchmark_results/overfitting_diagnostics.png
```

## Hugging Face

Integration with Hugging Face Hub allows us to store and load trained models easily. The API automatically downloads models from HF on startup, and you can push your own models after training.

```bash
# Authenticate
huggingface-cli login

or

hf auth login

# Push existing checkpoints
uv run python push_models_to_hf.py
```

Models are stored at [slugroom/gait-classification-models](https://huggingface.co/slugroom/gait-classification-models).

## API Usage

Web app for enrolling and classifying users by their gait. Uses a trained LSTM or Transformer model to produce gait embeddings from phone sensor data.

### Running the app

```bash
uvicorn gait_classification.api:app --reload
```

Open `http://localhost:8050` in a browser.

### Usage

1. **Select a model** — choose Transformer, LSTM, or FFT + Centroid.
2. **Enroll a trusted user** — go to the model page and click **Start Recording**. Walk naturally for 60 seconds with your phone in your pocket. The gait is encoded and stored.
3. **Classify a user** — go to the classify page, click **Start Recording**, and walk for a few seconds. The app compares the embedding against enrolled users and returns the closest match with a confidence score.

### iOS testing with ngrok (required for DeviceMotion)

iOS requires HTTPS for the `DeviceMotionEvent` API. Use [ngrok](https://ngrok.com) to expose the local server over a public HTTPS URL.

**1. Install ngrok**

```bash
brew install ngrok
```

Or download from [ngrok.com/download](https://ngrok.com/download).

**2. Start the app**

```bash
uvicorn gait_classification.api:app --host 0.0.0.0 --port 8050
```

**3. In a separate terminal, expose it**

```bash
ngrok http 8050
```

ngrok will print a URL like `https://abc123.ngrok-free.app`. Open that URL on your iPhone — motion recording will work over the HTTPS tunnel.

> Note: The free ngrok tier is enough for testing. You do not need an account for short sessions.

### CSV fallback (desktop / no motion sensors)

If the browser does not support `DeviceMotionEvent` (desktop, some Android), the enrollment page automatically shows a CSV upload form instead. The file should have six columns per row with no required header:

```
acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
0.51,0.22,9.81,1.02,0.48,0.11
...
```

## Docker Deployment

### Quick Start

```bash
docker-compose up --build
```

The app will be available at `http://localhost:8050`

### Accessing the App

- **Web UI**: http://localhost:8050
- **API Docs**: http://localhost:8050/docs
- **ReDoc**: http://localhost:8050/redoc

### Environment Variables

The following can be set in the `docker-compose.yml` file:

- `PYTHONUNBUFFERED=1` — Ensures Python output is streamed directly (recommended)

### Volume Mounts

The docker-compose.yml includes a volume for the checkpoints directory:
```yaml
volumes:
  - ./gait_classification/checkpoints:/app/gait_classification/checkpoints
```

This allows pre-downloaded models to be reused across container restarts without re-downloading.

## Hugging Face Integration

This project uses Hugging Face Hub to store and load trained models. Models are automatically downloaded during API startup and can be pushed after training.

### Repository

All models are stored in the Hugging Face Hub repository:
- **Repository**: [slugroom/gait-classification-models](https://huggingface.co/slugroom/gait-classification-models)

### Setup

#### Install Dependencies

The `huggingface-hub` dependency is already included in `pyproject.toml`:

```bash
pip install huggingface-hub
```

#### Authenticate with Hugging Face

You have two options:

**Option 1: Using CLI (Recommended)**
```bash
huggingface-cli login
# This will prompt you for your HF token and save it locally
```

**Option 2: Using Environment Variable**
```bash
export HF_TOKEN=<your-hugging-face-token>
```

To get your token, visit: https://huggingface.co/settings/tokens

### Usage

#### Loading Models from HF (Automatic)

The API automatically loads models from Hugging Face when it starts:

```bash
uvicorn gait_classification.api:app --reload
```

The first time, models will be downloaded and cached. Subsequent runs will use the cached versions.

#### Pushing Models to HF After Training

##### Option 1: During Training (Automatic)
Add `push_to_hf=true` when running training:

```bash
python gait_classification/train.py \
  max_samples=500 \
  batch_size=128 \
  model_type=transformer \
  push_to_hf=true
```

##### Option 2: After Training (Manual)
Use the provided script to push models from the checkpoints directory:

```bash
python push_models_to_hf.py
```

With a specific token:
```bash
python push_models_to_hf.py --token YOUR_HF_TOKEN
```

With a custom checkpoints directory:
```bash
python push_models_to_hf.py --checkpoints-dir /path/to/checkpoints
```

### Model Files

The following files are stored in the HF repository:

- `final_model_transformer.pt` - Transformer model checkpoint
- `best_model_lstm.pt` - LSTM model checkpoint
- `scaler.pkl` - Feature scaler for preprocessing
- `centroids_transformer.pkl` - Transformer model centroids
- `centroids_lstm.pkl` - LSTM model centroids

### API Load Behavior

The API loading logic tries the following in order:

1. **Try to load from HF Hub** - Downloads from Hugging Face (cached locally)
2. **Fallback to local files** - Uses local checkpoints if HF download fails
3. **Cache in memory** - Subsequent requests use the in-memory cache

This ensures the API works offline after the first download, while allowing updates when models are pushed to HF.

### Troubleshooting

#### "HF_TOKEN not set" Error
```bash
# Option 1: Set environment variable
export HF_TOKEN=<your-token>

# Option 2: Use CLI login
huggingface-cli login

# Option 3: Pass token explicitly to push script
python push_models_to_hf.py --token <your-token>
```

#### Models Not Found
- Ensure the HF repository exists and is accessible
- Check that model files have been uploaded to HF
- Verify your HF token has sufficient permissions

#### Offline Usage
- Models are cached locally after first download
- Set `HF_HUB_OFFLINE=1` environment variable to use cache-only mode
- Ensure local checkpoints exist as fallback

#### Example: Complete Workflow

```bash
# 1. Install dependencies
pip install -e .

# 2. Authenticate with HF
huggingface-cli login

# 3. Train model and push to HF
python gait_classification/train.py \
  max_samples=1000 \
  batch_size=128 \
  model_type=transformer \
  num_epochs=20 \
  push_to_hf=true

# 4. Verify upload on HF
# Visit: https://huggingface.co/slugroom/gait-classification-models

# 5. Start API (models auto-download from HF)
uvicorn gait_classification.api:app --reload
```

## Linting

```bash
flake8 gait_classification
black gait_classification --line-length 100
pre-commit run --all-files
```
