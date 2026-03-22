"""Behavioral tests — does the model make clinically sensible predictions?

Quality gate tests check METRICS (AUC, recall).
Behavioral tests check LOGIC:
- Does sicker patient get higher risk? (directional)
- Does the model crash on weird inputs? (edge cases)
- Are obvious cases classified correctly? (minimum functionality)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from model.features import FEATURE_COLUMNS, extract_features
from model.train import train_xgboost


@pytest.fixture(scope="module")
def xgb_model():
    """Train XGBoost once for all behavioral tests."""
    df = pd.read_csv("model/data/heart_failure.csv")
    X, y = extract_features(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_xgboost(X_train, y_train, seed=42)


def _make_patient(**overrides: float) -> np.ndarray:
    """Create a patient feature array with sensible defaults."""
    defaults = {
        "age": 65,
        "anaemia": 0,
        "creatinine_phosphokinase": 150,
        "diabetes": 0,
        "ejection_fraction": 35,
        "high_blood_pressure": 0,
        "platelets": 250000,
        "serum_creatinine": 1.2,
        "serum_sodium": 137,
        "sex": 1,
        "smoking": 0,
        "time": 120,
    }
    defaults.update(overrides)
    return np.array([[defaults[col] for col in FEATURE_COLUMNS]], dtype=np.float32)


@pytest.mark.ml
class TestDirectional:
    """Sicker patients should get higher risk scores."""

    def test_lower_ejection_fraction_higher_risk(self, xgb_model):
        """Heart pumping 15% should be riskier than 50%."""
        healthy_heart = _make_patient(ejection_fraction=50)
        weak_heart = _make_patient(ejection_fraction=15)

        risk_healthy = xgb_model.predict_proba(healthy_heart)[0][1]
        risk_weak = xgb_model.predict_proba(weak_heart)[0][1]
        assert risk_weak > risk_healthy

    def test_worse_kidneys_higher_risk(self, xgb_model):
        """Creatinine 5.0 (kidney failure) should be riskier than 1.0 (normal)."""
        good_kidneys = _make_patient(serum_creatinine=1.0)
        bad_kidneys = _make_patient(serum_creatinine=5.0)

        risk_good = xgb_model.predict_proba(good_kidneys)[0][1]
        risk_bad = xgb_model.predict_proba(bad_kidneys)[0][1]
        assert risk_bad > risk_good

    def test_older_patient_higher_risk(self, xgb_model):
        """85-year-old should be riskier than 40-year-old."""
        young = _make_patient(age=40)
        old = _make_patient(age=85)

        risk_young = xgb_model.predict_proba(young)[0][1]
        risk_old = xgb_model.predict_proba(old)[0][1]
        assert risk_old > risk_young


@pytest.mark.ml
class TestMinimumFunctionality:
    """Obvious cases must be classified correctly."""

    def test_very_sick_patient_flagged(self, xgb_model):
        """85yo, heart pumping 14%, kidneys failing, low sodium, seen for 4 days then died."""
        very_sick = _make_patient(
            age=85,
            ejection_fraction=14,
            serum_creatinine=9.0,
            serum_sodium=113,
            time=4,
            anaemia=1,
            diabetes=1,
        )
        assert xgb_model.predict(very_sick)[0] == 1

    def test_healthy_patient_not_flagged(self, xgb_model):
        """45yo, strong heart, normal labs, long follow-up."""
        healthy = _make_patient(
            age=45,
            ejection_fraction=55,
            serum_creatinine=0.8,
            serum_sodium=140,
            time=250,
        )
        assert xgb_model.predict(healthy)[0] == 0


@pytest.mark.ml
class TestEdgeCases:
    """Model should not crash on weird inputs."""

    def test_all_zeros(self, xgb_model):
        zeros = np.zeros((1, 12), dtype=np.float32)
        assert xgb_model.predict(zeros)[0] in [0, 1]

    def test_extreme_values(self, xgb_model):
        extreme = _make_patient(
            age=100,
            ejection_fraction=80,
            serum_creatinine=0.5,
            platelets=800000,
            creatinine_phosphokinase=7000,
        )
        assert xgb_model.predict(extreme)[0] in [0, 1]
