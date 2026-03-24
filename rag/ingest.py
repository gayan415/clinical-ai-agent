"""RAG document ingestion pipeline.

Loads clinical documents (markdown, PDF), splits them into chunks,
embeds them using HuggingFace or Bedrock, and stores in ChromaDB.

Pipeline: Load → Chunk → Embed → Store
"""

# Fix SSL for HuggingFace model downloads on macOS.
# Must run before any imports that trigger httpx/huggingface_hub.
import truststore

truststore.inject_into_ssl()

import os  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

if TYPE_CHECKING:
    import chromadb


def load_pdf_files(directory: str) -> list[Document]:
    """Load all .pdf files from a directory into LangChain Documents.

    Uses pypdf to extract text from each page, then combines all pages
    into one Document per file. Metadata includes the source file path.

    PDFs may have messy text (headers, footers, page numbers) — the
    chunking step downstream handles splitting into clean pieces.
    """
    from pypdf import PdfReader

    docs = []
    dir_path = Path(directory)

    for pdf_file in sorted(dir_path.glob("*.pdf")):
        reader = PdfReader(str(pdf_file))
        # Extract text from all pages, join with newlines
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()

        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(pdf_file)},
            )
        )

    return docs


def load_markdown_files(directory: str) -> list[Document]:
    """Load all .md files from a directory into LangChain Documents.

    Each Document gets metadata with the source file path,
    which is used later for citations in agent responses.
    """
    docs = []
    dir_path = Path(directory)

    for md_file in sorted(dir_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(md_file)},
            )
        )

    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into smaller overlapping chunks for embedding.

    Why: Embedding models have max input (~512 tokens). Smaller chunks
    also mean more precise retrieval — you get the relevant paragraph,
    not the whole document.

    The overlap ensures sentences at chunk boundaries aren't lost.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


def get_chroma_embedding_function() -> "chromadb.EmbeddingFunction[list[str]]":  # type: ignore[type-arg]
    """Return a ChromaDB-compatible embedding function.

    Default: HuggingFace all-MiniLM-L6-v2 (local, free, no API calls)
    Optional: Bedrock Titan (set EMBEDDING_PROVIDER=bedrock)

    ChromaDB has its own embedding function interface — we use its native
    SentenceTransformerEmbeddingFunction for HuggingFace, and a custom
    wrapper for Bedrock.
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()

    if provider == "bedrock":
        from chromadb import EmbeddingFunction, Embeddings

        class BedrockEmbeddingFunction(EmbeddingFunction):  # type: ignore[type-arg]
            """Wraps AWS Bedrock Titan Embed for ChromaDB."""

            def __init__(self) -> None:
                from langchain_aws import BedrockEmbeddings

                self._model = BedrockEmbeddings(
                    model_id="amazon.titan-embed-text-v2:0",
                    region_name=os.environ.get("AWS_REGION", "us-east-1"),
                )

            def __call__(self, input: list[str]) -> Embeddings:
                return self._model.embed_documents(input)  # type: ignore[return-value]

        return BedrockEmbeddingFunction()

    # Default: ChromaDB's native SentenceTransformer support
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def ingest_documents(data_dir: str, persist_dir: str) -> None:
    """Full ingestion pipeline: load → chunk → embed → store in ChromaDB.

    Args:
        data_dir: Path to directory containing rag/data/ subdirectories
        persist_dir: Path where ChromaDB will persist its data
    """
    import chromadb

    # Load documents from all subdirectories
    all_docs: list[Document] = []

    snippets_dir = Path(data_dir) / "snippets"
    if snippets_dir.exists():
        all_docs.extend(load_markdown_files(str(snippets_dir)))

    guidelines_dir = Path(data_dir) / "guidelines"
    if guidelines_dir.exists():
        all_docs.extend(load_markdown_files(str(guidelines_dir)))
        all_docs.extend(load_pdf_files(str(guidelines_dir)))

    if not all_docs:
        print("No documents found to ingest.")
        return

    # Chunk
    chunks = chunk_documents(all_docs, chunk_size=500, chunk_overlap=50)
    print(f"Split {len(all_docs)} documents into {len(chunks)} chunks.")

    # Embed and store
    chroma_ef = get_chroma_embedding_function()

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="clinical_docs",
        embedding_function=chroma_ef,  # type: ignore[arg-type]
    )

    # Upsert chunks using content hash as ID.
    # Same content = same hash = no duplicates on re-ingestion.
    # Updated content = new hash = inserted as new chunk.
    import hashlib

    ids = [hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16] for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

    print(f"Upserted {len(chunks)} chunks into ChromaDB at {persist_dir}")


if __name__ == "__main__":
    ingest_documents(
        data_dir="rag/data",
        persist_dir="chroma_db",
    )
