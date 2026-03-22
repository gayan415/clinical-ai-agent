"""Tests for RAG retriever — querying the vector store.

The retriever takes a natural language question, embeds it,
and finds the most similar document chunks in ChromaDB.
"""

import pytest

from rag.retriever import ClinicalRetriever


@pytest.mark.unit
class TestClinicalRetriever:
    """Test the retriever can find relevant clinical content."""

    def test_query_returns_results(self, ingested_db):
        """A valid clinical query should return at least one result."""
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What is NYHA Class III?")
        assert len(results) > 0

    def test_results_have_content_and_source(self, ingested_db):
        """Each result should have text content and source metadata."""
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("GDMT recommendations")

        for result in results:
            assert result.page_content  # not empty
            assert "source" in result.metadata  # knows where it came from

    def test_top_k_limits_results(self, ingested_db):
        """Setting top_k=2 should return at most 2 results."""
        retriever = ClinicalRetriever(persist_dir=ingested_db, top_k=2)
        results = retriever.query("heart failure")
        assert len(results) <= 2

    def test_relevant_content_returned(self, ingested_db):
        """Asking about NYHA should return content mentioning NYHA, not GDMT drugs."""
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What are the NYHA functional classifications?")

        # At least one result should mention NYHA
        all_content = " ".join(r.page_content for r in results)
        assert "NYHA" in all_content or "nyha" in all_content.lower()
