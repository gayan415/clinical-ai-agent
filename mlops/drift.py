"""Feature drift detection using Population Stability Index (PSI).

After deploying a model, the real-world data distribution may shift:
- Patient population gets older
- New treatments change lab values
- Seasonal patterns in admissions

PSI measures how much the current data distribution differs from training:
- PSI < 0.1  → no significant drift
- PSI 0.1-0.2 → moderate drift, investigate
- PSI > 0.2  → significant drift, consider retraining

This is the same concept as monitoring infrastructure metrics for anomalies —
but applied to model input features.
"""

import numpy as np


def compute_baseline(training_data: np.ndarray, n_bins: int = 10) -> dict:
    """Compute baseline distribution from training data.

    Splits the training data into n_bins equal-width bins and records
    the proportion of values in each bin. This becomes the reference
    distribution for drift comparison.
    """
    min_val = float(np.min(training_data))
    max_val = float(np.max(training_data))
    # Add small buffer to include max value in last bin
    bin_edges = np.linspace(min_val, max_val + 1e-6, n_bins + 1)
    counts, _ = np.histogram(training_data, bins=bin_edges)
    # Convert to proportions, add small epsilon to avoid division by zero
    proportions = (counts / len(training_data)) + 1e-6

    return {
        "bin_edges": bin_edges.tolist(),
        "proportions": proportions.tolist(),
        "n_samples": len(training_data),
    }


def compute_psi(baseline: dict, current_data: np.ndarray) -> float:
    """Compute Population Stability Index between baseline and current data.

    PSI = sum( (current_i - baseline_i) * ln(current_i / baseline_i) )

    Where current_i and baseline_i are the proportions of values in bin i.
    Higher PSI = more drift.
    """
    bin_edges = np.array(baseline["bin_edges"])
    baseline_proportions = np.array(baseline["proportions"])

    counts, _ = np.histogram(current_data, bins=bin_edges)
    current_proportions = (counts / len(current_data)) + 1e-6

    psi = float(
        np.sum(
            (current_proportions - baseline_proportions)
            * np.log(current_proportions / baseline_proportions)
        )
    )
    return psi


def check_drift(
    baselines: dict[str, dict],
    current_data: dict[str, np.ndarray],
    threshold: float = 0.2,
) -> list[str]:
    """Check all features for drift, return names of drifted features.

    Args:
        baselines: Dict of feature_name → baseline (from compute_baseline)
        current_data: Dict of feature_name → numpy array of current values
        threshold: PSI threshold for significant drift (default 0.2)

    Returns:
        List of feature names where PSI > threshold
    """
    drifted = []
    for feature_name, baseline in baselines.items():
        if feature_name in current_data:
            psi = compute_psi(baseline, current_data[feature_name])
            if psi > threshold:
                drifted.append(feature_name)
    return drifted
