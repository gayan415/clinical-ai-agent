"""Tests for drift detection — catch when real-world data shifts from training data."""

import numpy as np
import pytest

from mlops.drift import check_drift, compute_baseline, compute_psi


@pytest.mark.unit
class TestDriftDetection:
    def test_no_drift_same_distribution(self):
        """Same data as training = PSI well below significance threshold (0.2)."""
        rng = np.random.RandomState(42)
        training = rng.normal(loc=65, scale=10, size=200)
        current = rng.normal(loc=65, scale=10, size=100)

        baseline = compute_baseline(training)
        psi = compute_psi(baseline, current)
        # With small samples, PSI has natural variance. Below 0.2 = no significant drift.
        assert psi < 0.2

    def test_detects_significant_drift(self):
        """Very different distribution = PSI > 0.2."""
        rng = np.random.RandomState(42)
        training = rng.normal(loc=65, scale=10, size=200)
        # Current data shifted — patients are much older
        current = rng.normal(loc=85, scale=5, size=100)

        baseline = compute_baseline(training)
        psi = compute_psi(baseline, current)
        assert psi > 0.2

    def test_check_drift_returns_drifted_features(self):
        """check_drift should return names of features that drifted."""
        rng = np.random.RandomState(42)
        baselines = {
            "age": compute_baseline(rng.normal(65, 10, 200)),
            "ejection_fraction": compute_baseline(rng.normal(35, 10, 200)),
        }
        current_data = {
            "age": rng.normal(85, 5, 100),  # drifted
            "ejection_fraction": rng.normal(35, 10, 100),  # stable
        }
        drifted = check_drift(baselines, current_data, threshold=0.2)
        assert "age" in drifted
        assert "ejection_fraction" not in drifted

    def test_check_drift_empty_when_stable(self):
        rng = np.random.RandomState(42)
        baselines = {
            "age": compute_baseline(rng.normal(65, 10, 200)),
        }
        current_data = {
            "age": rng.normal(65, 10, 100),
        }
        drifted = check_drift(baselines, current_data)
        assert len(drifted) == 0
