from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from gait_classification.api.routes.ml import SensorSample, GaitData
from gait_classification.api.state import get_model_scaler_centroids, trusted_users
from gait_classification.utils import ModelType

router = APIRouter()

templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")
templates.env.filters["label"] = lambda m: m.value.replace("_", " ").title()

models = {
    ModelType.TRANSFORMER: None,
    ModelType.LSTM: None,
}


def _build_enrollment_embedding_from_gait_data(
    model_type: ModelType,
    gait_data: GaitData,
) -> list[float]:
    model, scaler, _ = get_model_scaler_centroids(model_type)

    import numpy as np
    import torch

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


@router.get("/", response_class=HTMLResponse)
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


@router.get("/models", response_class=HTMLResponse)
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


@router.get("/models/{model_type}/classify", response_class=HTMLResponse)
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


@router.post("/models/{model_type}/encode", response_class=HTMLResponse)
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
    embedding = _build_enrollment_embedding_from_gait_data(model_type, gait_data)
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
