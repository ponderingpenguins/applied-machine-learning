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

from gait_classification.models.lstm import LSTM
from gait_classification.utils import TrainConfig

app = FastAPI()
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

config = TrainConfig()

model = LSTM(
    input_size=6, hidden_size=128, num_layers=2, embedding_size=config.embedding_size
)
checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), "../checkpoints/best_model.pt"),
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(
    os.path.join(os.path.dirname(__file__), "../checkpoints/scaler.pkl"), "rb"
) as f:
    scaler = pickle.load(f)


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
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/classify", response_class=HTMLResponse)
async def classify(data: GaitData):
    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in data.samples
    ]
    arr = np.array(samples, dtype=np.float32)
    arr = scaler.transform(arr)
    tensor = torch.tensor(arr).unsqueeze(0)  # (1, seq_len, 6)

    print(samples)

    with torch.no_grad():
        model(tensor)

    n = len(data.samples)
    return f"""
    <div class="text-center">
      <div class="text-6xl mb-4">✓</div>
      <h2 class="text-2xl font-bold text-green-400 mb-2">Gait Recorded</h2>
      <p class="text-gray-400 mb-6">Successfully processed {n} sensor samples.</p>
      <button onclick="location.reload()"
              class="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition">
        Try Again
      </button>
    </div>
    """
