"""Tests for clinical safety module.

In healthcare AI, safety is not optional:
- Every response must include a disclaimer
- Low-confidence predictions must be flagged
- Every decision must be logged for audit (FDA requirement)
"""

import pytest

from agent.safety import (
    CLINICAL_DISCLAIMER,
    check_confidence,
    format_audit_entry,
    format_disclaimer,
)


@pytest.mark.unit
class TestDisclaimer:
    def test_disclaimer_wraps_recommendation(self):
        """Any recommendation must be wrapped with the clinical disclaimer."""
        result = format_disclaimer("Consider adding SGLT2 inhibitor.")
        assert CLINICAL_DISCLAIMER in result
        assert "SGLT2 inhibitor" in result

    def test_disclaimer_on_empty_recommendation(self):
        """Even empty recommendations get a disclaimer."""
        result = format_disclaimer("")
        assert CLINICAL_DISCLAIMER in result


@pytest.mark.unit
class TestConfidenceCheck:
    def test_low_confidence_flagged(self):
        """Below 70% confidence = clinician must review."""
        result = check_confidence(0.55)
        assert result["requires_review"] is True
        assert "low confidence" in result["message"].lower()

    def test_high_confidence_not_flagged(self):
        """Above 70% confidence = no special flag."""
        result = check_confidence(0.85)
        assert result["requires_review"] is False

    def test_threshold_is_configurable(self):
        """Different use cases may need different thresholds."""
        result = check_confidence(0.65, threshold=0.60)
        assert result["requires_review"] is False

    def test_edge_case_exactly_at_threshold(self):
        """At exactly 70%, should NOT require review (>= threshold passes)."""
        result = check_confidence(0.70, threshold=0.70)
        assert result["requires_review"] is False


@pytest.mark.unit
class TestAuditTrail:
    def test_audit_entry_has_required_fields(self):
        """Every audit entry must have these fields for FDA compliance."""
        entry = format_audit_entry(
            input_features={"age": 65, "ejection_fraction": 30},
            prediction=0.82,
            confidence=0.89,
            model_version="xgboost_v1",
            tools_called=["retrieve_clinical_context", "predict_risk"],
        )
        assert "timestamp" in entry
        assert entry["prediction"] == 0.82
        assert entry["confidence"] == 0.89
        assert entry["model_version"] == "xgboost_v1"
        assert "retrieve_clinical_context" in entry["tools_called"]

    def test_audit_entry_has_timestamp(self):
        """Timestamp must be ISO format for log parsing."""
        entry = format_audit_entry(
            input_features={},
            prediction=0.5,
            confidence=0.5,
            model_version="v1",
            tools_called=[],
        )
        # ISO format: 2026-03-22T...
        assert "T" in entry["timestamp"]
