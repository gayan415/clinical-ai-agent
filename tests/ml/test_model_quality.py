"""Model quality gate tests — models must pass these to be deployable.

These are not unit tests. They train actual models and check if the
metrics meet minimum thresholds. If a model can't beat these gates,
it's not good enough for clinical use.

In CI, these run as part of the ML validation stage.
"""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from model.evaluate import evaluate_model
from model.features import extract_features
from model.train import train_pytorch, train_xgboost


@pytest.fixture(scope="module")
def data_split():
    """Load data and split once for all quality tests.

    scope="module" = runs once per test file, not once per test.
    Fixed random_state=42 ensures reproducible splits.
    """
    df = pd.read_csv("model/data/heart_failure.csv")
    X, y = extract_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_xgboost(data_split):
    """Train XGBoost model once for all quality tests."""
    X_train, X_test, y_train, y_test = data_split
    model = train_xgboost(X_train, y_train, seed=42)
    return model, X_test, y_test


@pytest.fixture(scope="module")
def trained_pytorch(data_split):
    """Train PyTorch model once for all quality tests."""
    X_train, X_test, y_train, y_test = data_split
    model = train_pytorch(X_train, y_train, seed=42)
    return model, X_test, y_test


@pytest.mark.ml
class TestXGBoostQuality:
    """Quality gates for XGBoost model."""

    def test_auc_above_threshold(self, trained_xgboost):
        """AUC must be > 0.75 — can the model separate alive from dead patients?"""
        model, X_test, y_test = trained_xgboost
        metrics = evaluate_model(model, X_test, y_test, model_type="xgboost")
        assert metrics["auc"] > 0.75, f"XGBoost AUC {metrics['auc']:.3f} below 0.75"

    def test_recall_per_class(self, trained_xgboost):
        """Both classes must have recall > 0.60 — don't miss sick patients OR over-alarm."""
        model, X_test, y_test = trained_xgboost
        metrics = evaluate_model(model, X_test, y_test, model_type="xgboost")
        assert metrics["recall_class_0"] > 0.60, f"Class 0 recall {metrics['recall_class_0']:.3f}"
        assert metrics["recall_class_1"] > 0.60, f"Class 1 recall {metrics['recall_class_1']:.3f}"


@pytest.mark.ml
class TestPyTorchQuality:
    """Quality gates for PyTorch model."""

    def test_auc_above_threshold(self, trained_pytorch):
        """AUC must be > 0.70 — slightly lower bar than XGBoost (neural nets need more data)."""
        model, X_test, y_test = trained_pytorch
        metrics = evaluate_model(model, X_test, y_test, model_type="pytorch")
        assert metrics["auc"] > 0.70, f"PyTorch AUC {metrics['auc']:.3f} below 0.70"
