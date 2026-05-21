# Gait Classification Frontend

A minimal web app that collects gyroscope and accelerometer data from your phone, sends it to a server for gait classification, and returns the identified person with a confidence score.

## Setup

Install server dependencies:
```bash
uv sync --group server
```

## Running the Server

**Locally:**
```bash
uv run --group server uvicorn gait_classification.frontend.app:app --port 8000
```

**For testing on iPhone (requires HTTPS):**

Terminal 1 - Start the server:
```bash
uv run --group server uvicorn gait_classification.frontend.app:app --host 0.0.0.0 --port 8000
```

Terminal 2 - Create HTTPS tunnel with ngrok:
```bash
ngrok http 8000
```

Then open the HTTPS URL in **Safari** on your iPhone.

## How It Works

1. **Home page** - User clicks "Start" button
2. **Permission** - Browser requests motion sensor access (iOS requires explicit permission)
3. **Collection** - App collects 128 samples of accelerometer + gyroscope data (~1.3 seconds at 100Hz)
4. **Upload** - Data is sent to server
5. **Classification** - Server:
   - Scales data using training set statistics
   - Computes 64-dim embedding via LSTM model
   - Finds nearest person centroid
   - Returns person ID + confidence score
6. **Result** - Frontend displays classification result

## Files

- `app.py` - FastAPI server with classification endpoint
- `templates/index.html` - Single-page app (Tailwind + HTMX + vanilla JS)
- `../checkpoints/best_model_lstm.pt` - Trained LSTM model
- `../checkpoints/scaler.pkl` - Feature scaler (fit on training data)
- `../checkpoints/centroids_lstm.pkl` - Per-person embedding centroids

## Environment Variables

`MODEL_TYPE` - Which model checkpoint to load (default: `lstm`)
```bash
MODEL_TYPE=transformer uvicorn gait_classification.frontend.app:app
```

## Computing Centroids

If you need to recompute centroids from a checkpoint:
```bash
uv run python -m gait_classification.compute_centroids max_samples=500
```

## Debug Output

The server prints to console:
- Raw sensor samples
- Scaled samples after preprocessing
- Full 64-dim embedding vector
- Classification result

Check server logs to debug sensor data quality.
