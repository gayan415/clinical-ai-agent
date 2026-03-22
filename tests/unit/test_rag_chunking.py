"""Tests for RAG ingestion — chunking and document loading.

What we're testing:
- load_markdown_files: reads .md files from a directory, returns LangChain Documents
- chunk_documents: splits Documents into smaller overlapping pieces for embedding
"""

import pytest

from rag.ingest import chunk_documents, load_markdown_files, load_pdf_files


@pytest.mark.unit
class TestLoadMarkdown:
    """Test that we can load markdown files into LangChain Document objects."""

    def test_loads_all_files_from_directory(self, tmp_path):
        """Given a directory with 2 markdown files, should return 2 Documents."""
        (tmp_path / "doc1.md").write_text("# Doc 1\nFirst document content.")
        (tmp_path / "doc2.md").write_text("# Doc 2\nSecond document content.")

        docs = load_markdown_files(str(tmp_path))
        assert len(docs) == 2

    def test_document_has_page_content(self, tmp_path):
        """Each Document should contain the text from the file."""
        (tmp_path / "test.md").write_text("# Heart Failure\nGDMT is important.")

        docs = load_markdown_files(str(tmp_path))
        assert "GDMT is important" in docs[0].page_content

    def test_document_has_source_metadata(self, tmp_path):
        """Each Document should know which file it came from (for citations)."""
        (tmp_path / "nyha.md").write_text("# NYHA\nClass III is severe.")

        docs = load_markdown_files(str(tmp_path))
        assert "source" in docs[0].metadata
        assert "nyha.md" in docs[0].metadata["source"]

    def test_ignores_non_markdown_files(self, tmp_path):
        """Should only load .md files, ignore everything else."""
        (tmp_path / "good.md").write_text("# Valid markdown")
        (tmp_path / "bad.txt").write_text("Not a markdown file")
        (tmp_path / "bad.py").write_text("print('hello')")

        docs = load_markdown_files(str(tmp_path))
        assert len(docs) == 1

    def test_empty_directory_returns_empty(self, tmp_path):
        """Empty directory should return empty list, not crash."""
        docs = load_markdown_files(str(tmp_path))
        assert docs == []


@pytest.mark.unit
class TestLoadPDF:
    """Test that we can load PDF files into LangChain Document objects.

    pypdf reads the text layer from each page of a PDF.
    We combine all pages into one Document per file.
    """

    def test_loads_pdf_file(self, tmp_path):
        """Should load a PDF and extract its text content."""
        # Create a minimal PDF using pypdf
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        # pypdf blank pages have no text, so we test with a real-ish approach
        pdf_path = tmp_path / "test.pdf"
        writer.write(str(pdf_path))

        docs = load_pdf_files(str(tmp_path))
        # Blank PDF returns a Document (even if content is empty)
        assert len(docs) == 1
        assert "source" in docs[0].metadata
        assert "test.pdf" in docs[0].metadata["source"]

    def test_ignores_non_pdf_files(self, tmp_path):
        """Should only load .pdf files, ignore everything else."""
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.write(str(tmp_path / "good.pdf"))
        (tmp_path / "bad.md").write_text("# Not a PDF")

        docs = load_pdf_files(str(tmp_path))
        assert len(docs) == 1

    def test_empty_directory_returns_empty(self, tmp_path):
        """No PDFs = empty list."""
        docs = load_pdf_files(str(tmp_path))
        assert docs == []


@pytest.mark.unit
class TestChunkDocuments:
    """Test that documents are split into smaller overlapping chunks.

    Why chunking matters:
    - Embedding models have max input length (~512 tokens)
    - Smaller chunks = more precise retrieval (you get the relevant paragraph,
      not the whole document)
    - Overlap ensures we don't cut a sentence in half
    """

    def test_long_document_gets_split(self, tmp_path):
        """A document longer than chunk_size should be split into multiple chunks."""
        (tmp_path / "long.md").write_text("word " * 500)  # ~2500 chars
        docs = load_markdown_files(str(tmp_path))

        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_short_document_stays_intact(self, tmp_path):
        """A document shorter than chunk_size should not be split."""
        (tmp_path / "short.md").write_text("Brief content.")
        docs = load_markdown_files(str(tmp_path))

        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert "Brief content" in chunks[0].page_content

    def test_chunks_preserve_source_metadata(self, tmp_path):
        """After splitting, each chunk should still know its source file."""
        (tmp_path / "guidelines.md").write_text("word " * 500)
        docs = load_markdown_files(str(tmp_path))

        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "guidelines.md" in chunk.metadata["source"]

    def test_empty_input_returns_empty(self):
        """No documents in = no chunks out."""
        chunks = chunk_documents([], chunk_size=500, chunk_overlap=50)
        assert chunks == []
