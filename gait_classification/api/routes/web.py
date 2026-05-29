import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from gait_classification.api.routes.ml import SensorSample, GaitData
from gait_classification.api.state import get_model_and_scaler
from gait_classification.utils import ModelType

router = APIRouter()

templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

_LABELS = {
    "transformer": "Transformer",
    "lstm": "LSTM",
    "fft_centroids": "FFT Centroids",
}
templates.env.filters["label"] = lambda m: _LABELS.get(m.value, m.value.replace("_", " ").title())

_DESCRIPTIONS = {
    "transformer": "Uses a Transformer encoder to learn walking patterns from raw sensor data. Best overall accuracy.",
    "lstm": "Uses an LSTM network to model the sequence of motion over time. Good for varied walking speeds.",
    "fft_centroids": "Extracts frequency features from the motion signal and matches against stored centroids. Fastest inference.",
}
templates.env.filters["description"] = lambda m: _DESCRIPTIONS.get(m.value, "")

DEFAULT_MODEL = ModelType(os.getenv("DEFAULT_MODEL", "transformer"))

models = {
    ModelType.TRANSFORMER: None,
    ModelType.LSTM: None,
    ModelType.FFT_CENTROIDS: None,
}


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, name="home.html", context={"request": request})


@router.get("/methods", response_class=HTMLResponse)
async def methods(request: Request, model_type: ModelType | None = None):
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


@router.get("/models", response_class=HTMLResponse)
async def list_models(request: Request, model_type: ModelType | None = None):
    return RedirectResponse(url="/methods", status_code=302)


@router.get("/enroll", response_class=HTMLResponse)
async def enroll(request: Request):
    return templates.TemplateResponse(
        request,
        name="model_page.html",
        context={
            "request": request,
            "selected_model_value": DEFAULT_MODEL.value,
            "selected_label": _LABELS.get(DEFAULT_MODEL.value, DEFAULT_MODEL.value),
        },
    )


@router.get("/models/{model_type}", response_class=HTMLResponse)
async def model_page(request: Request, model_type: ModelType):
    return templates.TemplateResponse(
        request,
        name="model_page.html",
        context={
            "request": request,
            "selected_model_value": model_type.value,
            "selected_label": _LABELS.get(model_type.value, model_type.value),
        },
    )


