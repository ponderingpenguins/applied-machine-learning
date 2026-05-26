# Gait Classification API

Web app for enrolling and classifying users by their gait. Uses a trained LSTM or Transformer model to produce gait embeddings from phone sensor data.

## Running the app

```bash
uvicorn gait_classification.api:app --reload
```

Open `http://localhost:8000` in a browser.

## Usage

1. **Select a model** — choose LSTM or Transformer on the home page.
2. **Enroll a trusted user** — go to the model page and click **Start Recording**. Walk naturally for 60 seconds with your phone in your pocket. The gait is encoded and stored.
3. **Classify a user** — go to the classify page, click **Start Recording**, and walk for a few seconds. The app compares the embedding against enrolled users and returns the closest match with a confidence score.

## iOS testing with ngrok (required for DeviceMotion)

iOS requires HTTPS for the `DeviceMotionEvent` API. Use [ngrok](https://ngrok.com) to expose the local server over a public HTTPS URL.

**1. Install ngrok**

```bash
brew install ngrok
```

Or download from [ngrok.com/download](https://ngrok.com/download).

**2. Start the app**

```bash
uvicorn gait_classification.api:app --host 0.0.0.0 --port 8000
```

**3. In a separate terminal, expose it**

```bash
ngrok http 8000
```

ngrok will print a URL like `https://abc123.ngrok-free.app`. Open that URL on your iPhone — motion recording will work over the HTTPS tunnel.

> Note: The free ngrok tier is enough for testing. You do not need an account for short sessions.

## CSV fallback (desktop / no motion sensors)

If the browser does not support `DeviceMotionEvent` (desktop, some Android), the enrollment page automatically shows a CSV upload form instead. The file should have six columns per row with no required header:

```
acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
0.51,0.22,9.81,1.02,0.48,0.11
...
```
