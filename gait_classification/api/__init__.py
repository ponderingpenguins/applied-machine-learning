import pickle
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn.manifold import TSNE

from gait_classification.hf_utils import (
    download_centroids,
    download_model_checkpoint,
    download_scaler,
)
from gait_classification.models.models import construct_model
from gait_classification.utils import ModelType, TrainConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload available models
    for model_type in ModelType:
        try:
            _get_model_scaler_centroids(model_type)
            print(f" Preloaded {model_type.value} model")
        except FileNotFoundError as e:
            print(f"Skipping {model_type.value} model: {e}")
    yield
    # Cleanup (optional)


app = FastAPI(
    title="Gait Classification API",
    description="API for encoding and classifying gait data",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["label"] = lambda m: m.value.replace("_", " ").title()

app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)

_model_cache = {}


def _get_model_scaler_centroids(model_type: ModelType):
    cache_key = f"{model_type.value}_model"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Try to load from HuggingFace, fallback to local files
    try:
        checkpoint_path = download_model_checkpoint(
            model_type, cache_dir=checkpoints_dir
        )
    except Exception as e:
        print(f"Failed to download model from HF: {e}, trying local fallback...")
        if model_type.value == "transformer":
            checkpoint_path = checkpoints_dir / "final_model_transformer.pt"
        else:
            checkpoint_path = checkpoints_dir / f"best_model_{model_type.value}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model not found at {checkpoint_path} and HF download failed"
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(
        model_type=checkpoint["model_type"],
        embedding_size=checkpoint["embedding_size"],
    )
    model = construct_model(config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Try to load scaler from HF, fallback to local
    try:
        scaler_path = download_scaler(cache_dir=checkpoints_dir)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        with open(checkpoints_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    # Try to load centroids from HF, fallback to local
    centroids = {}
    try:
        centroids_path = download_centroids(model_type, cache_dir=checkpoints_dir)
        with open(centroids_path, "rb") as f:
            centroids = pickle.load(f)
    except Exception:
        centroids_path = checkpoints_dir / f"centroids_{model_type.value}.pkl"
        if centroids_path.exists():
            with open(centroids_path, "rb") as f:
                centroids = pickle.load(f)

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


class ClassifyWithReference(BaseModel):
    samples: list[SensorSample]
    reference_embedding: list[float]


@app.get("/", response_class=HTMLResponse)
async def root(request: Request, model_type: ModelType | None = None):
    model_options = list(models.keys())
    selected = model_type or model_options[0]
    return templates.TemplateResponse(
        request,
        name="model_selection.html",
        context={
            "request": request,
            "model_types": model_options,
            "selected_model": selected,
        },
    )


@app.get("/models", response_class=HTMLResponse)
async def list_models(request: Request, model_type: ModelType | None = None):
    model_options = list(models.keys())
    selected = model_type or model_options[0]
    return templates.TemplateResponse(
        request,
        name="model_selection.html",
        context={
            "request": request,
            "model_types": model_options,
            "selected_model": selected,
        },
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
        },
    )


def _build_trusted_user_embedding_from_gait_data(
    model_type: ModelType,
    gait_data: GaitData,
) -> list[float]:
    model, scaler, centroids = _get_model_scaler_centroids(model_type)

    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z]
        for s in gait_data.samples
    ]
    arr = np.array(samples, dtype=np.float32)
    arr = scaler.transform(arr)
    tensor = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.cpu().numpy()[0].tolist()


@app.post("/models/{model_type}/encode-recording")
async def encode_from_recording(model_type: ModelType, data: GaitData):
    if not data.samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No samples provided"
        )

    embedding = _build_trusted_user_embedding_from_gait_data(model_type, data)
    return {"embedding": embedding}


@app.post("/models/{model_type}/encode", response_class=HTMLResponse)
async def encode_from_file(
    request: Request,
    model_type: ModelType,
    trusted_user_file: UploadFile | None = File(default=None),
):
    if trusted_user_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded"
        )

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
            samples.append(
                SensorSample(
                    acc_x=float(parts[0]),
                    acc_y=float(parts[1]),
                    acc_z=float(parts[2]),
                    gyr_x=float(parts[3]),
                    gyr_y=float(parts[4]),
                    gyr_z=float(parts[5]),
                )
            )
        except ValueError:
            continue

    if not samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse any sensor samples from file",
        )

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
        },
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
        },
    )


@app.post("/models/{model_type}/classify")
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

    return {
        "result": result_text,
        "confidence": confidence,
        "samples_processed": len(data.samples),
    }


@app.post("/models/{model_type}/authenticate")
async def authenticate_user(model_type: ModelType, data: ClassifyWithReference):
    model, scaler, _ = _get_model_scaler_centroids(model_type)

    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in data.samples
    ]
    arr = np.array(samples, dtype=np.float32)
    arr = scaler.transform(arr)
    tensor = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    embedding_np = embedding.cpu().numpy()[0]
    reference_np = np.array(data.reference_embedding, dtype=np.float32)

    distance = float(np.linalg.norm(embedding_np - reference_np))
    threshold = 0.5
    is_match = distance < threshold
    confidence = max(0, 1.0 - (distance / 2.0))

    return {
        "is_match": is_match,
        "distance": distance,
        "confidence": confidence,
        "samples_processed": len(data.samples),
        "embedding": embedding_np.tolist(),
    }


@app.get("/models/data")
async def list_models_data():
    return {"available_models": [model_type.value for model_type in models.keys()]}


class EmbeddingVisualization(BaseModel):
    reference_embedding: list[float]
    auth_history: list[list[float]]


@app.post("/embeddings/plot")
async def plot_embeddings(data: EmbeddingVisualization):
    ref = np.array(data.reference_embedding, dtype=np.float32)
    embeddings = [ref]

    for auth_emb in data.auth_history:
        embeddings.append(np.array(auth_emb, dtype=np.float32))

    combined = np.vstack(embeddings)
    perplexity = min(30, len(embeddings) - 1) if len(embeddings) > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(10, 7))
    threshold = 0.5

    # Enrollment point
    ax.scatter(
        embeddings_2d[0, 0],
        embeddings_2d[0, 1],
        s=250,
        c="blue",
        label="Enrollment",
        marker="o",
        edgecolors="black",
        linewidths=2,
        zorder=5,
    )

    # Auth attempts with color gradient (oldest to newest)
    from matplotlib import cm

    colors_gradient = cm.get_cmap("Reds")(np.linspace(0.4, 0.9, len(data.auth_history)))

    for i, (auth_emb, color) in enumerate(zip(data.auth_history, colors_gradient)):
        distance = float(np.linalg.norm(auth_emb - ref))
        is_match = distance < threshold
        attempt_num = i + 1

        ax.scatter(
            embeddings_2d[i + 1, 0],
            embeddings_2d[i + 1, 1],
            s=200,
            c=[color],
            label=f"Attempt {attempt_num} (d={distance:.3f})",
            marker="s",
            edgecolors="black" if is_match else "gray",
            linewidths=2 if is_match else 1,
            zorder=4,
        )

        # Connect enrollment to this attempt
        ax.plot(
            [embeddings_2d[0, 0], embeddings_2d[i + 1, 0]],
            [embeddings_2d[0, 1], embeddings_2d[i + 1, 1]],
            "k--",
            alpha=0.3,
            linewidth=1,
            zorder=1,
        )

    # Add threshold info as text
    threshold_text = (
        f"Match threshold: {threshold}\n(Solid border = match, Gray = no match)"
    )
    ax.text(
        0.02,
        0.98,
        threshold_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Authentication History (Last 5 Attempts)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")
