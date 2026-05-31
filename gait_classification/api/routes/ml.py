from io import BytesIO

import numpy as np
import torch
from fastapi import APIRouter, Body, HTTPException, Path, status
from fastapi.responses import StreamingResponse
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from sklearn.manifold import TSNE

from scipy.fft import rfft as scipy_rfft

from gait_classification.api.state import get_model_and_scaler
from gait_classification.utils import ModelType

router = APIRouter(tags=["Gait Encoding & Authentication"])

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

    windows = [arr[s : s + FFT_SEQ_LEN] for s in range(0, len(arr) - FFT_SEQ_LEN + 1, FFT_SEQ_LEN)]
    if not windows:
        padded = np.pad(arr, ((0, FFT_SEQ_LEN - len(arr)), (0, 0)))
        windows = [padded]

    mean_features = np.mean([_window_features(w) for w in windows], axis=0)
    return scaler.transform(mean_features.reshape(1, -1))[0]


class SensorSample(BaseModel):
    """A single IMU sensor reading with 6 degrees of freedom.

    Represents one sample of acceleration (3 axes) and gyroscope (3 axes) data.
    """

    acc_x: float = Field(..., description="Acceleration in X direction (m/s²)")
    acc_y: float = Field(..., description="Acceleration in Y direction (m/s²)")
    acc_z: float = Field(..., description="Acceleration in Z direction (m/s²)")
    gyr_x: float = Field(..., description="Angular velocity around X axis (rad/s)")
    gyr_y: float = Field(..., description="Angular velocity around Y axis (rad/s)")
    gyr_z: float = Field(..., description="Angular velocity around Z axis (rad/s)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "acc_x": 0.5,
                "acc_y": -0.2,
                "acc_z": 9.8,
                "gyr_x": 0.1,
                "gyr_y": 0.05,
                "gyr_z": -0.15,
            }
        }
    }


class GaitData(BaseModel):
    """A sequence of IMU sensor readings from a gait recording.

    Contains multiple timesteps of acceleration and gyroscope data.
    At least one sample is required.
    """

    samples: list[SensorSample] = Field(
        ...,
        description="List of sensor readings. Minimum 1 sample required.",
        min_items=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "samples": [
                    {
                        "acc_x": 0.5,
                        "acc_y": -0.2,
                        "acc_z": 9.8,
                        "gyr_x": 0.1,
                        "gyr_y": 0.05,
                        "gyr_z": -0.15,
                    },
                    {
                        "acc_x": 0.6,
                        "acc_y": -0.1,
                        "acc_z": 9.7,
                        "gyr_x": 0.12,
                        "gyr_y": 0.07,
                        "gyr_z": -0.12,
                    },
                ]
            }
        }
    }


class ClassifyWithReference(BaseModel):
    """Request to authenticate a user against a reference embedding.

    Computes an embedding from the input samples and compares it to the
    reference embedding using L2 distance and a learned threshold.
    """

    samples: list[SensorSample] = Field(
        ...,
        description="Gait sensor readings to authenticate",
        min_items=1,
    )
    reference_embedding: list[float] = Field(
        ...,
        description="Reference embedding vector from enrollment (typically 128-256 dimensions)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "samples": [
                    {
                        "acc_x": 0.5,
                        "acc_y": -0.2,
                        "acc_z": 9.8,
                        "gyr_x": 0.1,
                        "gyr_y": 0.05,
                        "gyr_z": -0.15,
                    }
                ],
                "reference_embedding": [0.12, -0.08, 0.25, 0.15, -0.33, 0.08],
            }
        }
    }


class EmbeddingVisualization(BaseModel):
    """Request to generate a t-SNE visualization of authentication attempts.

    Creates a 2D scatter plot showing the enrollment point and all
    authentication attempts in embedding space.
    """

    reference_embedding: list[float] = Field(
        ...,
        description="The enrollment/reference embedding",
    )
    auth_history: list[list[float]] = Field(
        ...,
        description="List of authentication attempt embeddings in chronological order",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "reference_embedding": [0.12, -0.08, 0.25, 0.15, -0.33, 0.08],
                "auth_history": [
                    [0.13, -0.07, 0.26, 0.14, -0.32, 0.09],
                    [0.11, -0.09, 0.24, 0.16, -0.34, 0.07],
                ],
            }
        }
    }


models = {
    ModelType.TRANSFORMER: None,
    ModelType.LSTM: None,
    ModelType.FFT_CENTROIDS: None,
}


@router.get(
    "/models/data",
    summary="List available models",
    response_description="Dictionary containing list of available model types",
    tags=["Model Management"],
)
async def list_models_data():
    """Retrieve all available gait classification models.

    Returns the list of model types that can be used for encoding and authentication.
    Currently supports: transformer, lstm, and fft_centroids.
    """
    return {"available_models": [model_type.value for model_type in models.keys()]}


def _build_embedding_from_gait_data(
    model_type: ModelType,
    gait_data: GaitData,
) -> list[float]:
    model, scaler = get_model_and_scaler(model_type)

    samples = [[s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in gait_data.samples]
    arr = np.array(samples, dtype=np.float32)

    # Handle FFT centroids separately
    if model_type == ModelType.FFT_CENTROIDS:
        return _fft_embedding_from_raw(arr, scaler).tolist()

    arr = scaler.transform(arr)
    tensor = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.cpu().numpy()[0].tolist()


@router.post(
    "/models/{model_type}/encode-recording",
    summary="Generate gait embedding from sensor data",
    response_description="Embedding vector extracted from the gait recording",
    tags=["Gait Encoding & Authentication"],
    responses={
        200: {
            "description": "Successfully encoded the gait data",
            "content": {
                "application/json": {
                    "example": {"embedding": [0.12, -0.08, 0.25, 0.15, -0.33, 0.08]}
                }
            },
        },
        400: {
            "description": "Invalid request - no samples provided",
        },
        422: {
            "description": "Validation error - invalid model type or malformed data",
        },
    },
)
async def encode_from_recording(
    model_type: ModelType = Path(
        ..., description="Model type to use: 'transformer', 'lstm', or 'fft_centroids'"
    ),
    data: GaitData = Body(..., description="Gait sensor recordings to encode"),
):
    """Encode a gait recording into a fixed-size embedding vector.

    Takes raw IMU sensor data and processes it through the selected model
    to produce a compact embedding representation suitable for authentication
    or distance-based similarity comparisons.

    The resulting embedding is normalized and can be used as a reference for
    enrollment or compared against reference embeddings for authentication.
    """
    if not data.samples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No samples provided")

    embedding = _build_embedding_from_gait_data(model_type, data)
    return {"embedding": embedding}


@router.post(
    "/models/{model_type}/authenticate",
    summary="Authenticate user against reference embedding",
    response_description="Authentication decision with confidence score and distance metrics",
    tags=["Gait Encoding & Authentication"],
    responses={
        200: {
            "description": "Authentication decision computed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "is_match": True,
                        "distance": 0.25,
                        "threshold": 0.5,
                        "confidence": 0.75,
                        "samples_processed": 128,
                        "embedding": [0.13, -0.07, 0.26, 0.14, -0.32, 0.09],
                    }
                }
            },
        },
        400: {
            "description": "Invalid request - missing required fields",
        },
        422: {
            "description": "Validation error - invalid model type or embedding dimension mismatch",
        },
    },
)
async def authenticate_user(
    model_type: ModelType = Path(
        ..., description="Model type to use: 'transformer', 'lstm', or 'fft_centroids'"
    ),
    data: ClassifyWithReference = Body(..., description="Gait data and reference embedding"),
):
    """Authenticate a user by comparing their gait to a reference enrollment.

    Computes an embedding from the input sensor data and compares it to the
    reference embedding using L2 distance. Returns an authentication decision
    (match or no match), the distance metric, a confidence score, and the
    computed embedding for further analysis.

    **Decision Thresholds:**
    - Neural models (transformer/lstm): threshold = 0.5 (L2 distance on unit vectors)
    - FFT centroids model: threshold = 5.0 (StandardScaler-normalized distance)

    **Confidence Scoring:**
    - Confidence = 1.0 at distance=0 (perfect match)
    - Confidence = 0.5 at the decision boundary (distance = threshold)
    - Confidence = 0.0 at distance = 2×threshold (or beyond)
    """
    model, scaler = get_model_and_scaler(model_type)

    samples = [[s.acc_x, s.acc_y, s.acc_z, s.gyr_x, s.gyr_y, s.gyr_z] for s in data.samples]
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


@router.post(
    "/embeddings/plot",
    summary="Generate t-SNE visualization of embeddings",
    response_description="PNG image of authentication attempt visualization",
    tags=["Gait Encoding & Authentication"],
    responses={
        200: {
            "description": "Successfully generated t-SNE plot",
            "content": {"image/png": {"description": "Scatter plot visualization"}},
        },
        422: {
            "description": "Validation error - missing embeddings or dimension mismatch",
        },
    },
)
async def plot_embeddings(
    data: EmbeddingVisualization = Body(
        ..., description="Reference enrollment and authentication attempt embeddings"
    ),
):
    """Generate a t-SNE visualization of authentication history in embedding space.

    Creates a 2D scatter plot showing:
    - The enrollment point (blue circle)
    - Authentication attempts (red squares, gradient from light to dark)
    - Distance lines connecting enrollment to each attempt
    - Thick black borders indicate successful matches; thin gray borders indicate rejections

    The plot helps visualize how consistent a user's gait biometric is across
    multiple authentication attempts and shows the spread of embeddings in
    learned feature space.
    """
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
    threshold_text = f"Match threshold: {threshold}\n(Solid border = match, Gray = no match)"
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
