"""
api/main.py
-----------
FastAPI service that serves predictions from the trained model.

Run locally (no keys, no paid services):
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

# Make `src` importable when running `uvicorn api.main:app` from the repo root.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import DiagnosisModel  # noqa: E402

app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description=(
        "A free, self-hosted ML inference API. Predicts whether a tumor is "
        "benign or malignant from 30 numeric cell-nuclei measurements "
        "(Breast Cancer Wisconsin Diagnostic dataset)."
    ),
    version="1.0.0",
)

# Allow a local frontend (e.g. a Streamlit or React dev server) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = DiagnosisModel()
except FileNotFoundError as exc:
    # Let the app still start so /health can report the problem clearly,
    # instead of crashing with a confusing traceback on import.
    model = None
    _model_load_error = str(exc)
else:
    _model_load_error = None


def _build_input_schema() -> type[BaseModel]:
    """Dynamically build a Pydantic model with one float field per feature,
    so request validation and the /docs page always match the trained model."""
    if model is None:
        return BaseModel
    fields = {name: (float, Field(..., description=name)) for name in model.feature_names}
    return create_model("PatientFeatures", **fields)


PatientFeatures = _build_input_schema()


class PredictionResponse(BaseModel):
    prediction: str
    prediction_index: int
    probabilities: dict[str, float]


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Breast Cancer Diagnosis API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Meta"])
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error}")
    return {"status": "ok", "n_features": len(model.feature_names)}


@app.get("/features", tags=["Meta"])
def features():
    """List the exact feature names/order the model expects."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"feature_names": model.feature_names}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: PatientFeatures):  # type: ignore[valid-type]
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error}")
    try:
        result = model.predict(payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
