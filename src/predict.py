"""
predict.py
----------
Small, reusable wrapper around the saved model artifact so both the
FastAPI service and the Streamlit demo share exactly the same inference
logic (no duplicated preprocessing code = no drift between the two).
"""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"


class DiagnosisModel:
    """Loads the trained pipeline once and exposes a simple .predict() API."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python src/train.py` first."
            )
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.feature_names = artifact["feature_names"]
        self.target_names = artifact["target_names"]

    def predict(self, features: Dict[str, float]) -> dict:
        """
        features: dict mapping each of self.feature_names to a numeric value.
        Returns the predicted class label and class probabilities.
        """
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        row = pd.DataFrame([[features[f] for f in self.feature_names]],
                            columns=self.feature_names)

        pred_idx = int(self.pipeline.predict(row)[0])
        proba = self.pipeline.predict_proba(row)[0]

        return {
            "prediction": str(self.target_names[pred_idx]),
            "prediction_index": pred_idx,
            "probabilities": {
                str(self.target_names[i]): round(float(p), 4)
                for i, p in enumerate(proba)
            },
        }


if __name__ == "__main__":
    # Quick manual smoke test using the mean value of each feature.
    model = DiagnosisModel()
    dummy_input = {name: 0.0 for name in model.feature_names}
    print(model.predict(dummy_input))
