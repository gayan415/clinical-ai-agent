"""Clinical safety module — disclaimers, confidence checks, audit trail.

In healthcare AI, safety is non-negotiable:
1. Every response must include a disclaimer (agent recommends, never decides)
2. Low-confidence predictions must be flagged for clinician review
3. Every prediction must be logged for audit (FDA 21 CFR Part 11 awareness)
"""

from datetime import UTC, datetime
from typing import Any

CLINICAL_DISCLAIMER = (
    "AI-assisted recommendation — clinical judgment required. "
    "Do not use for direct patient care without clinician review."
)


def format_disclaimer(recommendation: str) -> str:
    """Wrap any recommendation with the clinical disclaimer."""
    return f"{recommendation}\n\n---\n{CLINICAL_DISCLAIMER}"


def check_confidence(
    confidence: float,
    threshold: float = 0.70,
) -> dict[str, Any]:
    """Check if a prediction's confidence meets the review threshold.

    Below threshold = clinician must review before acting.
    In healthcare, low-confidence predictions are dangerous —
    better to flag for human review than silently serve a bad prediction.
    """
    if confidence >= threshold:
        return {"requires_review": False, "message": "Confidence within acceptable range."}

    return {
        "requires_review": True,
        "message": f"Low confidence ({confidence:.0%}) — clinician review required.",
    }


def format_audit_entry(
    input_features: dict,
    prediction: float,
    confidence: float,
    model_version: str,
    tools_called: list[str],
) -> dict[str, Any]:
    """Create an immutable audit log entry for a prediction.

    Every prediction gets logged with:
    - What was the input (patient features)
    - What was the output (prediction + confidence)
    - Which model version made the prediction
    - Which tools the agent called
    - When it happened (ISO timestamp)

    In production, these go to an append-only log (S3, CloudWatch).
    FDA requires traceability for clinical decision support systems.
    """
    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "input_features": input_features,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version,
        "tools_called": tools_called,
    }
