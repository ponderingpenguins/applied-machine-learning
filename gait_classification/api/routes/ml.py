from io import BytesIO

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn.manifold import TSNE

from gait_classification.api.state import get_model_scaler_centroids
from gait_classification.utils import ModelType

router = APIRouter()


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


class EmbeddingVisualization(BaseModel):
    reference_embedding: list[float]
    auth_history: list[list[float]]


models = {
    ModelType.TRANSFORMER: None,
    ModelType.LSTM: None,
}


@router.get("/models/data")
async def list_models_data():
    return {"available_models": [model_type.value for model_type in models.keys()]}


def _build_embedding_from_gait_data(
    model_type: ModelType,
    gait_data: GaitData,
) -> list[float]:
    model, scaler, _ = get_model_scaler_centroids(model_type)

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


@router.post("/models/{model_type}/encode-recording")
async def encode_from_recording(model_type: ModelType, data: GaitData):
    if not data.samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No samples provided"
        )

    embedding = _build_embedding_from_gait_data(model_type, data)
    return {"embedding": embedding}


@router.post("/models/{model_type}/classify")
async def classify_user(model_type: ModelType, data: GaitData):
    model, scaler, centroids = get_model_scaler_centroids(model_type)

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


@router.post("/models/{model_type}/authenticate")
async def authenticate_user(model_type: ModelType, data: ClassifyWithReference):
    model, scaler, _ = get_model_scaler_centroids(model_type)

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


@router.post("/embeddings/plot")
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
