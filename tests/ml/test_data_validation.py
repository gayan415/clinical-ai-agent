"""Data validation tests — ensure the UCI dataset is clean and usable.

These tests run BEFORE training. If the data is corrupt, missing columns,
or has impossible values, we catch it here — not during model training
where the error would be cryptic.

Think of this as input validation for your ML pipeline.
"""

import pandas as pd
import pytest

from model.features import FEATURE_COLUMNS, TARGET_COLUMN, extract_features, validate_input


@pytest.mark.ml
class TestDatasetIntegrity:
    """Validate the UCI Heart Failure dataset itself."""

    @pytest.fixture()
    def df(self):
        return pd.read_csv("model/data/heart_failure.csv")

    def test_dataset_has_expected_columns(self, df):
        """All 12 features + 1 target must be present."""
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            assert col in df.columns, f"Missing column: {col}"

    def test_dataset_has_299_rows(self, df):
        """UCI dataset should have exactly 299 patients."""
        assert len(df) == 299

    def test_no_missing_values(self, df):
        """No NaN/null values in any feature or target column."""
        assert df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum() == 0

    def test_age_in_valid_range(self, df):
        """Age should be between 0 and 120."""
        assert df["age"].between(0, 120).all()

    def test_ejection_fraction_in_valid_range(self, df):
        """EF should be between 0 and 100 (it's a percentage)."""
        assert df["ejection_fraction"].between(0, 100).all()

    def test_binary_columns_are_binary(self, df):
        """Columns that represent yes/no should only have 0 or 1."""
        binary_cols = [
            "anaemia",
            "diabetes",
            "high_blood_pressure",
            "sex",
            "smoking",
            "DEATH_EVENT",
        ]
        for col in binary_cols:
            assert set(df[col].unique()).issubset({0, 1}), f"{col} has non-binary values"

    def test_target_is_not_extremely_imbalanced(self, df):
        """Death rate should be between 10-90%. Extreme imbalance = model can't learn."""
        death_rate = df[TARGET_COLUMN].mean()
        assert 0.1 < death_rate < 0.9, f"Target imbalance: {death_rate:.2%} death rate"


@pytest.mark.ml
class TestFeatureExtraction:
    """Test the feature extraction and validation functions."""

    def test_extract_features_returns_correct_shape(self):
        """Should return X with 12 columns and y with same number of rows."""
        df = pd.read_csv("model/data/heart_failure.csv")
        X, y = extract_features(df)
        assert X.shape[1] == len(FEATURE_COLUMNS)
        assert len(y) == len(df)

    def test_validate_rejects_negative_age(self):
        """Age can't be negative — validation should catch this."""
        errors = validate_input({"age": -5, "ejection_fraction": 30, "serum_creatinine": 1.0})
        assert len(errors) > 0

    def test_validate_rejects_impossible_ef(self):
        """EF can't be 200% — validation should catch this."""
        errors = validate_input({"age": 65, "ejection_fraction": 200, "serum_creatinine": 1.0})
        assert len(errors) > 0

    def test_validate_accepts_good_data(self):
        """A valid patient record should pass validation."""
        good_data = {
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
        errors = validate_input(good_data)
        assert len(errors) == 0
