"""Clinical document retriever — queries ChromaDB for relevant content.

Takes a natural language question, converts it to a vector using the
same embedding model used during ingestion, and finds the most similar
document chunks via cosine similarity.
"""

import chromadb
from langchain_core.documents import Document

from rag.ingest import get_chroma_embedding_function


class ClinicalRetriever:
    """Retrieves relevant clinical documents from ChromaDB.

    Usage:
        retriever = ClinicalRetriever(persist_dir="chroma_db")
        results = retriever.query("What meds for HFrEF?")
        # Returns list of Document objects with page_content and metadata
    """

    def __init__(self, persist_dir: str, top_k: int = 5) -> None:
        self._top_k = top_k

        chroma_ef = get_chroma_embedding_function()

        client = chromadb.PersistentClient(path=persist_dir)
        self._collection = client.get_or_create_collection(
            name="clinical_docs",
            embedding_function=chroma_ef,  # type: ignore[arg-type]
        )

    def query(self, question: str) -> list[Document]:
        """Search for documents relevant to the question.

        Args:
            question: Natural language clinical question

        Returns:
            List of Document objects, ranked by similarity (most relevant first).
            Each Document has .page_content (text) and .metadata (source file, etc.)
        """
        if not question.strip():
            return []

        results = self._collection.query(
            query_texts=[question],
            n_results=min(self._top_k, self._collection.count()),
        )

        # Convert ChromaDB results to LangChain Documents
        documents = []
        if results["documents"] and results["metadatas"]:
            for content, metadata in zip(
                results["documents"][0],
                results["metadatas"][0],
                strict=False,
            ):
                documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata or {},
                    )
                )

        return documents
