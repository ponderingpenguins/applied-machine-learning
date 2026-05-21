import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from gait_classification.models.models import construct_model
from gait_classification.utils import TrainConfig

app = FastAPI()
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

_model_cache = {}


def get_model_scaler_centroids():
    if all(k in _model_cache for k in ("model", "scaler", "centroids")):
        return (
            _model_cache["model"],
            _model_cache["scaler"],
            _model_cache["centroids"],
        )

    model_type = os.getenv("MODEL_TYPE", "lstm")
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), f"../checkpoints/best_model_{model_type}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(
        model_type=checkpoint["model_type"],
        embedding_size=checkpoint["embedding_size"],
    )
    model = construct_model(config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(
        os.path.join(os.path.dirname(__file__), "../checkpoints/scaler.pkl"), "rb"
    ) as f:
        scaler = pickle.load(f)

    centroids_path = os.path.join(
        os.path.dirname(__file__), f"../checkpoints/centroids_{model_type}.pkl"
    )
    try:
        with open(centroids_path, "rb") as f:
            centroids = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: centroids file not found at {centroids_path}")
        centroids = {}

    _model_cache["model"] = model
    _model_cache["scaler"] = scaler
    _model_cache["centroids"] = centroids
    return model, scaler, centroids


class SensorSample(BaseModel):
    acc_x: float
    acc_y: float
    acc_z: float
    gyr_x: float
    gyr_y: float
    gyr_z: float


class GaitData(BaseModel):
    samples: list[SensorSample]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/classify", response_class=HTMLResponse)
async def classify(data: GaitData):
    model, scaler, centroids = get_model_scaler_centroids()

    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in data.samples
    ]
    arr = np.array(samples, dtype=np.float32)

    print(f"Raw samples shape: {arr.shape}")
    print(f"First 3 samples:\n{arr[:3]}")

    arr = scaler.transform(arr)
    print(f"Scaled samples (first 3):\n{arr[:3]}")

    tensor = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    embedding_np = embedding.cpu().numpy()[0]
    print(f"Embedding shape: {embedding_np.shape}")
    print(f"Embedding: {embedding_np}")

    if centroids:
        distances = {
            pid: float(np.linalg.norm(embedding_np - centroid))
            for pid, centroid in centroids.items()
        }
        best_pid = min(distances.items(), key=lambda x: x[1])[0]
        best_dist = distances[best_pid]
        confidence = max(0, 1.0 - (best_dist / 2.0))
        result_text = f"Person {best_pid}"
    else:
        embedding_norm = float(np.linalg.norm(embedding_np))
        confidence = min(1.0, embedding_norm)
        result_text = "Gait Pattern"

    n = len(data.samples)
    return f"""
    <div class="text-center">
      <div class="text-6xl mb-4">✓</div>
      <h2 class="text-2xl font-bold text-green-400 mb-2">Classification: {result_text}</h2>
      <p class="text-gray-300 mb-2">Confidence: {confidence*100:.1f}%</p>
      <p class="text-gray-400 mb-6">Processed {n} sensor samples</p>
      <button onclick="location.reload()"
              class="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition">
        Try Again
      </button>
    </div>
    """
