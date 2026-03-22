"""RAG retrieval quality tests — does the system return the RIGHT documents?

These are ML-specific tests, not unit tests. They verify that the embedding
model + vector search actually understands clinical semantics. If these fail,
either the embeddings are bad or the clinical content is poorly written.
"""

import pytest

from rag.retriever import ClinicalRetriever


@pytest.mark.ml
class TestRetrievalQuality:
    """Known clinical queries must return expected documents."""

    def test_nyha_query_returns_nyha_content(self, ingested_db):
        """Asking about NYHA should return NYHA classification content."""
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What are the NYHA functional classifications?")

        all_content = " ".join(r.page_content for r in results).lower()
        assert "nyha" in all_content
        assert any(term in all_content for term in ["class i", "class ii", "class iii", "class iv"])

    def test_gdmt_query_returns_drug_recommendations(self, ingested_db):
        """Asking about HFrEF treatment should return GDMT drug info."""
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What medications are recommended for HFrEF?")

        all_content = " ".join(r.page_content for r in results).lower()
        # Should mention at least 2 of the 4 GDMT drug classes
        drug_classes = ["ace inhibitor", "arni", "beta-blocker", "sglt2", "spironolactone"]
        matches = sum(1 for drug in drug_classes if drug in all_content)
        assert matches >= 2, f"Expected ≥2 GDMT drug classes, found {matches}"

    def test_risk_query_returns_prognostic_factors(self, ingested_db):
        """Asking about risk factors should return ejection fraction, creatinine, etc."""
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What are the risk factors for heart failure mortality?")

        all_content = " ".join(r.page_content for r in results).lower()
        risk_factors = ["ejection fraction", "creatinine", "age"]
        matches = sum(1 for factor in risk_factors if factor in all_content)
        assert matches >= 2, f"Expected ≥2 risk factors, found {matches}"
