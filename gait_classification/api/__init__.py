from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch

from gait_classification.api.model_selection_page import render_model_selection_page
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

gaits = {
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
    return render_model_selection_page(models.keys(), model_type)


@app.get("/models/data")
async def list_models_data():
    return {"available_models": [model_type.value for model_type in models.keys()]}


@app.post("/models/{model_type}/encode_gait/")
async def encode_gait(model_type: ModelType, data: GaitData):
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input data"
        )
    model = models.get(model_type)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    embedding = model.forward(data)
    return {"embedding": embedding.tolist()}


@app.post("/models/{model_type}/classify_gait/")
async def classify_gait(model_type: ModelType, data: GaitData):
    if gaits.get(model_type) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No gaits available for classification. First encode some gaits using the /models/{model_type}/encode_gait/ endpoint."
        )
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input data"
        )
    model = models.get(model_type)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    prediction = model.forward(data)
    return {"prediction": prediction}