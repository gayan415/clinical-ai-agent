"""Model evaluation — compute metrics to assess model quality.

Metrics explained:
- AUC (Area Under ROC Curve): How well the model separates classes. 1.0 = perfect, 0.5 = random.
- Accuracy: % of predictions that are correct. Can be misleading with imbalanced data.
- Precision: "When the model says HIGH RISK, how often is it right?"
- Recall: "Of all patients who died, how many did the model catch?"
  In healthcare, recall matters MORE — missing a sick patient is worse than a false alarm.
- F1: Harmonic mean of precision and recall. Balances both.
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "xgboost",
) -> dict[str, float]:
    """Evaluate a trained model and return metrics.

    Args:
        model: Trained model (XGBClassifier or dict with PyTorch model + scaler)
        X_test: Test features (n_patients, 12)
        y_test: Test labels (n_patients,)
        model_type: "xgboost" or "pytorch"

    Returns:
        Dict of metric_name → value
    """
    if model_type == "pytorch":
        y_pred, y_prob = _predict_pytorch(model, X_test)
    else:
        y_pred, y_prob = _predict_xgboost(model, X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall_class_0": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "recall_class_1": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
    }


def _predict_xgboost(model: Any, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions from XGBoost model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def _predict_pytorch(model_dict: Any, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions from PyTorch model.

    PyTorch requires:
    1. Scale the input (same scaler used during training)
    2. Convert to tensor
    3. Run through model with no_grad() (skip gradient computation — we're not training)
    4. Convert back to numpy
    """
    scaler = model_dict["scaler"]
    model = model_dict["model"]

    X_scaled = scaler.transform(X_test)
    X_tensor = torch.FloatTensor(X_scaled)

    with torch.no_grad():  # Don't compute gradients during inference
        y_prob = model(X_tensor).squeeze().numpy()

    y_pred = (y_prob >= 0.5).astype(np.float32)  # Threshold at 0.5
    return y_pred, y_prob
