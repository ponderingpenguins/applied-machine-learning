from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from gait_classification.api.routes.ml import router as ml_router
from gait_classification.api.routes.web import router as web_router
from gait_classification.api.state import get_model_and_scaler
from gait_classification.utils import ModelType


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload available models
    for model_type in ModelType:
        try:
            get_model_and_scaler(model_type)
            print(f"Preloaded {model_type.value} model")
        except FileNotFoundError as e:
            print(f"Skipping {model_type.value} model: {e}")
    yield


app = FastAPI(
    title="Gait Classification API",
    description="API for encoding and classifying gait data",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)

app.include_router(ml_router)
app.include_router(web_router)
