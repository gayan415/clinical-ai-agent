"""FastAPI prediction endpoint for heart failure risk model.

Serves the trained XGBoost model over HTTP. The LangChain agent
calls this endpoint to get risk scores for patients.

Endpoints:
- POST /predict — send patient features, get risk score back
- GET /health  — liveness check (is the service running?)
- GET /ready   — readiness check (is the model loaded?)
"""

import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.features import FEATURE_COLUMNS, validate_input


class PatientInput(BaseModel):
    """Request schema — all 12 patient features required.

    Pydantic validates types automatically. Our validate_input()
    checks clinical ranges on top of that.
    """

    age: float = Field(..., ge=0, le=120)
    anaemia: float = Field(..., ge=0, le=1)
    creatinine_phosphokinase: float = Field(..., ge=0)
    diabetes: float = Field(..., ge=0, le=1)
    ejection_fraction: float = Field(..., ge=0, le=100)
    high_blood_pressure: float = Field(..., ge=0, le=1)
    platelets: float = Field(..., ge=0)
    serum_creatinine: float = Field(..., ge=0)
    serum_sodium: float = Field(..., ge=100, le=160)
    sex: float = Field(..., ge=0, le=1)
    smoking: float = Field(..., ge=0, le=1)
    time: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Response schema — risk score + metadata."""

    risk_score: float
    confidence: float
    model_version: str
    latency_ms: float


def create_app(model_path: str) -> FastAPI:
    """Create FastAPI app with the given model.

    Factory pattern — makes the app testable by injecting the model path.
    """
    app = FastAPI(title="Heart Failure Risk Prediction API")

    # Load model at startup
    model = joblib.load(model_path)
    model_loaded = True

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy"}

    @app.get("/ready")
    def ready() -> dict:
        return {"model_loaded": model_loaded, "model_version": "xgboost_v1"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientInput) -> PredictionResponse:
        # Validate clinical ranges
        patient_dict = patient.model_dump()
        errors = validate_input(patient_dict)
        if errors:
            raise HTTPException(status_code=422, detail=errors)

        # Build feature array in correct column order
        features = np.array(
            [[patient_dict[col] for col in FEATURE_COLUMNS]],
            dtype=np.float32,
        )

        # Predict with timing
        start = time.perf_counter()
        probabilities = model.predict_proba(features)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        risk_score = float(probabilities[1])  # probability of death
        confidence = float(max(probabilities))  # how sure the model is

        return PredictionResponse(
            risk_score=risk_score,
            confidence=confidence,
            model_version="xgboost_v1",
            latency_ms=round(elapsed_ms, 2),
        )

    return app
