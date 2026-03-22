"""Integration tests for the FastAPI model prediction endpoint.

Tests the HTTP API contract — what goes in, what comes out.
Uses FastAPI's TestClient (no real server needed).
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from model.features import extract_features
from model.predict import create_app
from model.train import train_xgboost


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create a test client with a real trained model."""
    import joblib

    df = pd.read_csv("model/data/heart_failure.csv")
    X, y = extract_features(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_xgboost(X_train, y_train, seed=42)

    model_path = str(tmp_path_factory.mktemp("models") / "xgboost_hf_risk.pkl")
    joblib.dump(model, model_path)

    app = create_app(model_path=model_path)
    return TestClient(app)


VALID_PATIENT = {
    "age": 65,
    "anaemia": 0,
    "creatinine_phosphokinase": 150,
    "diabetes": 1,
    "ejection_fraction": 30,
    "high_blood_pressure": 1,
    "platelets": 250000,
    "serum_creatinine": 1.2,
    "serum_sodium": 137,
    "sex": 1,
    "smoking": 0,
    "time": 120,
}


@pytest.mark.integration
class TestHealthEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_ready(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


@pytest.mark.integration
class TestPredictEndpoint:
    def test_valid_prediction(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        assert response.status_code == 200

        body = response.json()
        assert 0 <= body["risk_score"] <= 1
        assert 0 <= body["confidence"] <= 1
        assert "model_version" in body
        assert body["latency_ms"] >= 0

    def test_invalid_age_rejected(self, client):
        bad_patient = {**VALID_PATIENT, "age": -5}
        response = client.post("/predict", json=bad_patient)
        assert response.status_code == 422

    def test_missing_fields_rejected(self, client):
        response = client.post("/predict", json={"age": 65})
        assert response.status_code == 422

    def test_response_includes_latency(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        assert response.json()["latency_ms"] >= 0
