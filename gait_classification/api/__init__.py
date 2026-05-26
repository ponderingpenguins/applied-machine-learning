from fastapi import FastAPI, HTTPException, status
from fastapi import File, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import hashlib

from gait_classification.api.model_selection_page import (
    render_classify_user_page,
    render_model_page,
    render_model_selection_page,
)
from gait_classification.models.models import construct_model
from gait_classification.utils import ModelType, TrainConfig

app = FastAPI(
    title="Gait Classification API",
    description="API for encoding and classifying gait data",
    version="1.0.0",
)

_transformer = construct_model(
    TrainConfig(model_type="transformer", embedding_size=128),
    device=torch.device("cpu"),
)
_lstm = construct_model(
    TrainConfig(model_type="lstm", embedding_size=128),
    device=torch.device("cpu"),
)

models = {
    ModelType.TRANSFORMER: _transformer,
    ModelType.LSTM: _lstm,
}

trusted_users = {
    ModelType.TRANSFORMER: [],
    ModelType.LSTM: [],
}

class GaitDataStep(BaseModel):
    gyr_x: float
    gyr_y: float
    gyr_z: float
    acc_x: float
    acc_y: float
    acc_z: float


class GaitData(BaseModel):
    steps: list[GaitDataStep]


@app.get("/", response_class=HTMLResponse)
async def root(model_type: ModelType | None = None):
    return render_model_selection_page(models.keys(), model_type)


@app.get("/models", response_class=HTMLResponse)
async def list_models(model_type: ModelType | None = None):
    return render_model_selection_page(models.keys(), model_type)


@app.get("/models/{model_type}", response_class=HTMLResponse)
async def model_page(model_type: ModelType):
    return render_model_page(model_type)


def _build_trusted_user_embedding(
    model_type: ModelType,
    source_bytes: bytes,
    source_mode: str,
) -> list[float]:
    digest = hashlib.sha256(model_type.value.encode("utf-8") + b":" + source_mode.encode("utf-8") + b":" + source_bytes).digest()
    return [round(byte / 255.0, 6) for byte in digest[:16]]


@app.post("/models/{model_type}/encode", response_class=HTMLResponse)
async def model_encode_page(
    model_type: ModelType,
    trusted_user_file: UploadFile | None = File(default=None),
    source_mode: str = Form(default="upload"),
):
    model = models.get(model_type)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    if source_mode == "record":
        source_bytes = source_mode.encode("utf-8")
        source_label = "recorded data"
    elif trusted_user_file is not None:
        source_bytes = await trusted_user_file.read()
        source_label = trusted_user_file.filename or "uploaded file"
    else:
        source_bytes = source_mode.encode("utf-8")
        source_label = "recorded data"

    embedding = _build_trusted_user_embedding(model_type, source_bytes, source_mode)
    trusted_users[model_type].append(embedding)

    return render_model_page(
        model_type,
        status_message=f"Trusted user embedding added from {source_label}.",
    )

@app.get("/models/{model_type}/classify", response_class=HTMLResponse)
async def model_classify_page(model_type: ModelType):
    model = models.get(model_type)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return render_classify_user_page(model_type)


@app.get("/models/data")
async def list_models_data():
    return {"available_models": [model_type.value for model_type in models.keys()]}
