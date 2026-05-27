from fastapi import FastAPI, HTTPException, status, Request
from fastapi import File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
import pickle
from pathlib import Path

from gait_classification.models.models import construct_model
from gait_classification.utils import ModelType, TrainConfig

app = FastAPI(
    title="Gait Classification API",
    description="API for encoding and classifying gait data",
    version="1.0.0",
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["label"] = lambda m: m.value.replace("_", " ").title()

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

_model_cache = {}


def _get_model_scaler_centroids(model_type: ModelType):
    cache_key = f"{model_type.value}_model"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    checkpoint_path = checkpoints_dir / f"best_model_{model_type.value}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(
        model_type=checkpoint["model_type"],
        embedding_size=checkpoint["embedding_size"],
    )
    model = construct_model(config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(checkpoints_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    centroids_path = checkpoints_dir / f"centroids_{model_type.value}.pkl"
    try:
        with open(centroids_path, "rb") as f:
            centroids = pickle.load(f)
    except FileNotFoundError:
        centroids = {}

    result = (model, scaler, centroids)
    _model_cache[cache_key] = result
    return result


models = {
    ModelType.TRANSFORMER: None,
    ModelType.LSTM: None,
}

trusted_users = {
    ModelType.TRANSFORMER: [],
    ModelType.LSTM: [],
}

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
async def root(request: Request, model_type: ModelType | None = None):
    model_options = list(models.keys())
    selected = model_type or model_options[0]
    return templates.TemplateResponse(
        request,
        name="model_selection.html",
        context={"request": request, "model_types": model_options, "selected_model": selected}
    )


@app.get("/models", response_class=HTMLResponse)
async def list_models(request: Request, model_type: ModelType | None = None):
    model_options = list(models.keys())
    selected = model_type or model_options[0]
    return templates.TemplateResponse(
        request,
        name="model_selection.html",
        context={"request": request, "model_types": model_options, "selected_model": selected}
    )


@app.get("/models/{model_type}", response_class=HTMLResponse)
async def model_page(request: Request, model_type: ModelType):
    return templates.TemplateResponse(
        request,
        name="model_page.html",
        context={
            "request": request,
            "selected_model_value": model_type.value,
            "selected_label": model_type.value.replace("_", " ").title(),
        }
    )


def _build_trusted_user_embedding_from_gait_data(
    model_type: ModelType,
    gait_data: GaitData,
) -> list[float]:
    model, scaler, centroids = _get_model_scaler_centroids(model_type)

    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in gait_data.samples
    ]
    arr = np.array(samples, dtype=np.float32)
    arr = scaler.transform(arr)
    tensor = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.cpu().numpy()[0].tolist()


@app.post("/models/{model_type}/encode-recording", response_class=HTMLResponse)
async def encode_from_recording(request: Request, model_type: ModelType, data: GaitData):
    if not data.samples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No samples provided")

    embedding = _build_trusted_user_embedding_from_gait_data(model_type, data)
    trusted_users[model_type].append(embedding)

    return templates.TemplateResponse(
        request,
        name="fragments/enroll_status.html",
        context={
            "request": request,
            "status_message": f"Trusted user enrolled from recording ({len(data.samples)} samples).",
        }
    )


@app.post("/models/{model_type}/encode", response_class=HTMLResponse)
async def encode_from_file(
    request: Request,
    model_type: ModelType,
    trusted_user_file: UploadFile | None = File(default=None),
):
    if trusted_user_file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    content = (await trusted_user_file.read()).decode("utf-8")
    samples = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("acc"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            samples.append(SensorSample(
                acc_x=float(parts[0]),
                acc_y=float(parts[1]),
                acc_z=float(parts[2]),
                gyr_x=float(parts[3]),
                gyr_y=float(parts[4]),
                gyr_z=float(parts[5]),
            ))
        except ValueError:
            continue

    if not samples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not parse any sensor samples from file")

    gait_data = GaitData(samples=samples)
    embedding = _build_trusted_user_embedding_from_gait_data(model_type, gait_data)
    trusted_users[model_type].append(embedding)

    filename = trusted_user_file.filename or "uploaded file"
    return templates.TemplateResponse(
        request,
        name="fragments/enroll_status.html",
        context={
            "request": request,
            "status_message": f"Trusted user enrolled from {filename} ({len(samples)} samples).",
        }
    )

@app.get("/models/{model_type}/classify", response_class=HTMLResponse)
async def model_classify_page(request: Request, model_type: ModelType):
    return templates.TemplateResponse(
        request,
        name="classify_user.html",
        context={
            "request": request,
            "selected_model_value": model_type.value,
            "selected_label": model_type.value.replace("_", " ").title(),
        }
    )


@app.post("/models/{model_type}/classify", response_class=HTMLResponse)
async def classify_user(model_type: ModelType, data: GaitData):
    model, scaler, centroids = _get_model_scaler_centroids(model_type)

    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in data.samples
    ]
    arr = np.array(samples, dtype=np.float32)
    arr = scaler.transform(arr)
    tensor = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    embedding_np = embedding.cpu().numpy()[0]

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
    <div class="classification-result">
        <div class="result-icon">✓</div>
        <h2>Classification: {result_text}</h2>
        <p>Confidence: {confidence*100:.1f}%</p>
        <p>Processed {n} sensor samples</p>
        <button onclick="location.reload()" class="button-row">
            Try Again
        </button>
    </div>
    """


@app.get("/models/data")
async def list_models_data():
    return {"available_models": [model_type.value for model_type in models.keys()]}
