# Hugging Face Integration

This project uses Hugging Face Hub to store and load trained models. Models are automatically downloaded during API startup and can be pushed after training.

## Repository

All models are stored in the Hugging Face Hub repository:
- **Repository**: [slugroom/gait-classification-models](https://huggingface.co/slugroom/gait-classification-models)

## Setup

### Install Dependencies

The `huggingface-hub` dependency is already included in `pyproject.toml`:

```bash
pip install huggingface-hub
```

### Authenticate with Hugging Face

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

## Usage

### Loading Models from HF (Automatic)

The API automatically loads models from Hugging Face when it starts:

```bash
uvicorn gait_classification.api:app --reload
```

The first time, models will be downloaded and cached. Subsequent runs will use the cached versions.

### Pushing Models to HF After Training

#### Option 1: During Training (Automatic)
Add `push_to_hf=true` when running training:

```bash
python gait_classification/train.py \
  max_samples=500 \
  batch_size=128 \
  model_type=transformer \
  push_to_hf=true
```

#### Option 2: After Training (Manual)
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

## Model Files

The following files are stored in the HF repository:

- `final_model_transformer.pt` - Transformer model checkpoint
- `best_model_lstm.pt` - LSTM model checkpoint
- `scaler.pkl` - Feature scaler for preprocessing
- `centroids_transformer.pkl` - Transformer model centroids
- `centroids_lstm.pkl` - LSTM model centroids

## API Load Behavior

The API loading logic tries the following in order:

1. **Try to load from HF Hub** - Downloads from Hugging Face (cached locally)
2. **Fallback to local files** - Uses local checkpoints if HF download fails
3. **Cache in memory** - Subsequent requests use the in-memory cache

This ensures the API works offline after the first download, while allowing updates when models are pushed to HF.

## Troubleshooting

### "HF_TOKEN not set" Error
```bash
# Option 1: Set environment variable
export HF_TOKEN=<your-token>

# Option 2: Use CLI login
huggingface-cli login

# Option 3: Pass token explicitly to push script
python push_models_to_hf.py --token <your-token>
```

### Models Not Found
- Ensure the HF repository exists and is accessible
- Check that model files have been uploaded to HF
- Verify your HF token has sufficient permissions

### Offline Usage
- Models are cached locally after first download
- Set `HF_HUB_OFFLINE=1` environment variable to use cache-only mode
- Ensure local checkpoints exist as fallback

## Example: Complete Workflow

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
