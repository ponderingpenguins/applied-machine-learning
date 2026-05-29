from io import BytesIO

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn.manifold import TSNE

from scipy.fft import rfft as scipy_rfft

from gait_classification.api.state import get_model_and_scaler
from gait_classification.utils import ModelType

router = APIRouter()

FFT_SEQ_LEN = 128
FFT_BINS_PER_CHANNEL = 250


def _fft_embedding_from_raw(arr: np.ndarray, scaler) -> np.ndarray:
    """Extract and scale an FFT embedding from a raw (n_samples, 6) recording.

    Windows the recording into seq_len=128 chunks, extracts FFT magnitude
    features (matching compute_fft_centroids.py), then averages across windows.
    Uses scaler.n_features_in_ to select the same number of features as training.
    """
    n_keep = scaler.n_features_in_

    def _window_features(window):
        feats = []
        for ch in range(window.shape[1]):
            yf = scipy_rfft(window[:, ch])
            feats.extend(np.abs(yf[:FFT_BINS_PER_CHANNEL]))
        return np.array(feats[:n_keep], dtype=np.float32)

    windows = [arr[s:s + FFT_SEQ_LEN] for s in range(0, len(arr) - FFT_SEQ_LEN + 1, FFT_SEQ_LEN)]
    if not windows:
        padded = np.pad(arr, ((0, FFT_SEQ_LEN - len(arr)), (0, 0)))
        windows = [padded]

    mean_features = np.mean([_window_features(w) for w in windows], axis=0)
    return scaler.transform(mean_features.reshape(1, -1))[0]


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
    ModelType.FFT_CENTROIDS: None,
}


@router.get("/models/data")
async def list_models_data():
    return {"available_models": [model_type.value for model_type in models.keys()]}


def _build_embedding_from_gait_data(
    model_type: ModelType,
    gait_data: GaitData,
) -> list[float]:
    model, scaler = get_model_and_scaler(model_type)

    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z]
        for s in gait_data.samples
    ]
    arr = np.array(samples, dtype=np.float32)

    # Handle FFT centroids separately
    if model_type == ModelType.FFT_CENTROIDS:
        return _fft_embedding_from_raw(arr, scaler).tolist()

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


@router.post("/models/{model_type}/authenticate")
async def authenticate_user(model_type: ModelType, data: ClassifyWithReference):
    model, scaler = get_model_and_scaler(model_type)

    samples = [
        [s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in data.samples
    ]
    arr = np.array(samples, dtype=np.float32)

    # Handle FFT centroids separately
    if model_type == ModelType.FFT_CENTROIDS:
        embedding_np = _fft_embedding_from_raw(arr, scaler)
    else:
        arr = scaler.transform(arr)
        tensor = torch.tensor(arr).unsqueeze(0)
        with torch.no_grad():
            embedding = model(tensor)
        embedding_np = embedding.cpu().numpy()[0]

    reference_np = np.array(data.reference_embedding, dtype=np.float32)

    distance = float(np.linalg.norm(embedding_np - reference_np))
    # Neural embeddings are L2-normalised (unit vectors, d ∈ [0, 2]).
    # FFT embeddings are StandardScaler-normalised; distance scale differs.
    threshold = 0.5 if model_type != ModelType.FFT_CENTROIDS else 5.0
    is_match = distance < threshold
    # Confidence = 1.0 at distance=0, 0.5 at the decision boundary, 0.0 at 2×threshold.
    confidence = float(max(0.0, min(1.0, 1.0 - distance / (2.0 * threshold))))

    return {
        "is_match": is_match,
        "distance": distance,
        "threshold": threshold,
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
