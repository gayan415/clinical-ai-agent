"""SLO (Service Level Objective) definitions and checking.

SLOs define the reliability targets for the model service.
Same concept as SRE SLOs for any production service, applied to ML:
- Availability: is the model service up?
- Latency: how fast are predictions?
- Error rate: how often do predictions fail?
- Quality: is the model still accurate?
"""

from typing import Any

# SLO definitions — thresholds for alerting
SLOS: dict[str, dict[str, Any]] = {
    "model_availability": {
        "description": "Model API availability",
        "target": 0.999,  # 99.9%
        "unit": "ratio",
    },
    "model_latency_p99": {
        "description": "Model prediction latency (p99)",
        "target": 200.0,  # milliseconds
        "unit": "ms",
    },
    "model_error_rate": {
        "description": "Model prediction error rate",
        "target": 0.001,  # 0.1%
        "unit": "ratio",
    },
    "model_auc": {
        "description": "Model prediction quality (AUC)",
        "target": 0.75,
        "unit": "score",
    },
    "agent_response_p95": {
        "description": "Agent end-to-end response time (p95)",
        "target": 10000.0,  # 10 seconds
        "unit": "ms",
    },
}


def check_slo(slo_name: str, current_value: float) -> dict[str, Any]:
    """Check if a current metric value meets its SLO target.

    Returns dict with: met (bool), slo_name, target, current, margin.
    """
    if slo_name not in SLOS:
        return {"met": False, "error": f"Unknown SLO: {slo_name}"}

    slo = SLOS[slo_name]
    target = slo["target"]

    # For latency/error rate: lower is better (current must be <= target)
    # For availability/AUC: higher is better (current must be >= target)
    if slo_name in ("model_latency_p99", "model_error_rate", "agent_response_p95"):
        met = current_value <= target
    else:
        met = current_value >= target

    return {
        "met": met,
        "slo_name": slo_name,
        "description": slo["description"],
        "target": target,
        "current": current_value,
        "unit": slo["unit"],
    }
