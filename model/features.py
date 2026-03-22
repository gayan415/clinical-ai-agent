"""Feature extraction and validation for heart failure risk prediction.

This module defines:
- Which columns from the UCI dataset are features vs target
- How to extract features into numpy arrays for model training
- How to validate incoming patient data at the API boundary

The validation rules are based on clinically plausible ranges —
a patient can't have negative age or EF > 100%.
"""

import numpy as np
import pandas as pd

# 12 input features from the UCI Heart Failure dataset
FEATURE_COLUMNS = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]

# Binary target: 1 = patient died, 0 = survived
TARGET_COLUMN = "DEATH_EVENT"

# Clinically plausible ranges for input validation
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "anaemia": (0, 1),
    "creatinine_phosphokinase": (0, 50000),
    "diabetes": (0, 1),
    "ejection_fraction": (0, 100),
    "high_blood_pressure": (0, 1),
    "platelets": (0, 1_000_000),
    "serum_creatinine": (0, 20),
    "serum_sodium": (100, 160),
    "sex": (0, 1),
    "smoking": (0, 1),
    "time": (0, 1000),
}


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a DataFrame.

    Args:
        df: DataFrame with columns matching FEATURE_COLUMNS and TARGET_COLUMN

    Returns:
        Tuple of (X, y) where X is shape (n_patients, 12) and y is shape (n_patients,)
    """
    X = np.array(df[FEATURE_COLUMNS].values, dtype=np.float32)
    y = np.array(df[TARGET_COLUMN].values, dtype=np.float32)
    return X, y


def validate_input(data: dict) -> list[str]:
    """Validate a single patient's input data against clinically plausible ranges.

    Used at the API boundary to reject garbage input before it reaches the model.
    Returns a list of error messages (empty list = valid).

    Args:
        data: Dict of feature_name → value

    Returns:
        List of validation error strings. Empty = all valid.
    """
    errors = []

    for feature, (min_val, max_val) in FEATURE_RANGES.items():
        if feature in data:
            val = data[feature]
            if not isinstance(val, int | float):
                errors.append(f"{feature}: expected number, got {type(val).__name__}")
            elif val < min_val or val > max_val:
                errors.append(f"{feature}: {val} outside valid range [{min_val}, {max_val}]")

    return errors
