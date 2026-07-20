"""
train.py
--------
Trains a breast-cancer diagnosis classifier and saves a single, ready-to-serve
artifact (preprocessing + model) to disk.

Dataset: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn.
No internet connection, API key, or paid service is required to run this.

Usage:
    python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the built-in Breast Cancer Wisconsin dataset as a DataFrame."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")  # 0 = malignant, 1 = benign
    return X, y, list(dataset.target_names)


def build_pipeline() -> Pipeline:
    """Scaler + RandomForest wrapped in a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )


def train_and_evaluate() -> dict:
    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()

    # Small, fast grid search so the script still runs in seconds on a laptop.
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 6, 10],
    }
    search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1
    )

    start = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - start

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_params": search.best_params_,
        "train_time_seconds": round(train_time, 2),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the fitted pipeline together with everything the API needs
    # to validate incoming requests and decode predictions.
    artifact = {
        "pipeline": best_model,
        "feature_names": list(X.columns),
        "target_names": target_names,
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


if __name__ == "__main__":
    results = train_and_evaluate()
    print("Training complete.\n")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"F1 score : {results['f1_score']:.4f}")
    print(f"ROC AUC  : {results['roc_auc']:.4f}")
    print(f"Best params: {results['best_params']}")
    print(f"\nModel saved to  : {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
