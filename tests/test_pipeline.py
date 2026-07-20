"""
tests/test_pipeline.py
-----------------------
Basic tests for the trained model wrapper and the FastAPI service.

Run with:
    pytest -v
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import DiagnosisModel  # noqa: E402

MODEL_PATH = ROOT_DIR / "models" / "model.joblib"


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found - run `python src/train.py` first.")
    return DiagnosisModel()


def test_model_loads(model):
    assert model.pipeline is not None
    assert len(model.feature_names) == 30
    assert set(model.target_names) == {"malignant", "benign"}


def test_predict_returns_expected_shape(model):
    sample = {name: 0.0 for name in model.feature_names}
    result = model.predict(sample)

    assert "prediction" in result
    assert result["prediction"] in model.target_names
    assert set(result["probabilities"].keys()) == set(model.target_names)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_predict_missing_feature_raises(model):
    incomplete = {name: 0.0 for name in model.feature_names[:-1]}
    with pytest.raises(ValueError):
        model.predict(incomplete)


def test_api_endpoints():
    """Integration test for the FastAPI app. Skipped automatically if
    fastapi/httpx aren't installed in the current environment."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found - run `python src/train.py` first.")

    from api.main import app

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    features_resp = client.get("/features")
    feature_names = features_resp.json()["feature_names"]

    payload = {name: 0.0 for name in feature_names}
    pred = client.post("/predict", json=payload)
    assert pred.status_code == 200
    body = pred.json()
    assert body["prediction"] in {"malignant", "benign"}
