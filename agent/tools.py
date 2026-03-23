"""LangChain tool definitions for the clinical AI agent.

Each tool wraps one of our components (RAG, model, recommender)
so the LLM can call them by name. The @tool decorator tells
LangChain: "This function is callable by the agent."

The agent reads the docstring to decide when to use each tool.
"""

import json
import os

import httpx
from langchain_core.tools import tool

from agent.safety import check_confidence, format_disclaimer
from rag.retriever import ClinicalRetriever
from sre.circuit_breaker import CircuitBreaker

# Lazy-initialized components (created on first use)
_retriever: ClinicalRetriever | None = None
_model_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


def _get_retriever() -> ClinicalRetriever:
    """Get or create the RAG retriever (lazy init)."""
    global _retriever
    if _retriever is None:
        persist_dir = os.environ.get("CHROMA_DB_PATH", "chroma_db")
        _retriever = ClinicalRetriever(persist_dir=persist_dir)
    return _retriever


@tool
def retrieve_clinical_context(query: str) -> str:
    """Search clinical guidelines for relevant information.

    Use this tool when you need clinical context about heart failure,
    NYHA classifications, GDMT drug recommendations, CardioMEMS protocols,
    or risk factors. Returns relevant excerpts from ACC/AHA guidelines.

    Args:
        query: A clinical question like "What is NYHA Class III?" or
               "GDMT recommendations for HFrEF"
    """
    retriever = _get_retriever()
    results = retriever.query(query)

    if not results:
        return "No relevant clinical guidelines found for this query."

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        formatted.append(f"[{i}] (Source: {source})\n{doc.page_content}")

    return "\n\n".join(formatted)


@tool
def predict_risk(patient_data_json: str) -> str:
    """Predict heart failure mortality risk for a patient.

    Use this tool when you have specific patient measurements and need
    a risk score. Input must be a JSON string with patient features.

    Args:
        patient_data_json: JSON string with keys: age, anaemia,
            creatinine_phosphokinase, diabetes, ejection_fraction,
            high_blood_pressure, platelets, serum_creatinine,
            serum_sodium, sex, smoking, time
    """
    # Circuit breaker check
    if not _model_circuit_breaker.allow_request():
        return (
            "Model service unavailable (circuit breaker open). "
            "Unable to generate risk prediction — clinician review required."
        )

    model_url = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8000")

    try:
        patient_data = json.loads(patient_data_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Please provide valid patient data."

    try:
        response = httpx.post(
            f"{model_url}/predict",
            json=patient_data,
            timeout=10.0,
        )
        response.raise_for_status()
        _model_circuit_breaker.record_success()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        _model_circuit_breaker.record_failure()
        return (
            f"Model service error: {e}. "
            "Unable to generate risk prediction — clinician review required."
        )

    result = response.json()
    risk_score = result["risk_score"]
    confidence = result["confidence"]
    latency = result["latency_ms"]

    # Check confidence threshold
    confidence_check = check_confidence(confidence)
    confidence_warning = ""
    if confidence_check["requires_review"]:
        confidence_warning = f"\n⚠️ {confidence_check['message']}"

    risk_level = "HIGH" if risk_score > 0.5 else "LOW"

    return (
        f"Risk Score: {risk_score:.2f} ({risk_level})\n"
        f"Confidence: {confidence:.0%}\n"
        f"Model: {result['model_version']}\n"
        f"Latency: {latency:.1f}ms"
        f"{confidence_warning}"
    )


@tool
def recommend_treatment(clinical_context: str) -> str:
    """Recommend evidence-based treatment based on clinical context.

    Use this tool after retrieving clinical context and/or risk assessment.
    Searches GDMT guidelines and returns treatment recommendations
    with citations. Always used as the last step before final response.

    Args:
        clinical_context: Summary of patient's condition, risk level,
            and any relevant clinical findings
    """
    retriever = _get_retriever()
    results = retriever.query(f"GDMT treatment recommendations for: {clinical_context}")

    if not results:
        return format_disclaimer("No specific treatment guidelines found for this context.")

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        formatted.append(f"[{i}] (Source: {source})\n{doc.page_content}")

    recommendations = "\n\n".join(formatted)
    return format_disclaimer(recommendations)
