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
templates.env.filters["label"] = lambda m: m.value.replace("_", " ").title()

DEFAULT_MODEL = ModelType(os.getenv("DEFAULT_MODEL", "transformer"))

models = {
    ModelType.TRANSFORMER: None,
    ModelType.LSTM: None,
    ModelType.FFT_CENTROIDS: None,
}


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, name="home.html", context={"request": request})


@router.get("/dev", response_class=HTMLResponse)
async def dev(request: Request, model_type: ModelType | None = None):
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
    return RedirectResponse(url="/dev", status_code=302)


@router.get("/enroll", response_class=HTMLResponse)
async def enroll(request: Request):
    return templates.TemplateResponse(
        request,
        name="model_page.html",
        context={
            "request": request,
            "selected_model_value": DEFAULT_MODEL.value,
            "selected_label": DEFAULT_MODEL.value.replace("_", " ").title(),
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
            "selected_label": model_type.value.replace("_", " ").title(),
        },
    )


