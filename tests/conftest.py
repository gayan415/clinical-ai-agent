"""Shared pytest fixtures for the test suite.

Fixtures are reusable test setup functions. Any test can request a fixture
by name as a function parameter, and pytest injects it automatically.
"""

# Fix SSL certificate verification for HuggingFace model downloads.
# macOS Python may not trust system certificates by default.
# truststore makes httpx (used by huggingface_hub) use the OS cert store.
import truststore

truststore.inject_into_ssl()

import pytest  # noqa: E402

from rag.ingest import (  # noqa: E402
    chunk_documents,
    get_chroma_embedding_function,
    load_markdown_files,
)


@pytest.fixture(scope="session")
def ingested_db(tmp_path_factory):
    """Create a temporary ChromaDB with test clinical documents.

    scope="session" means this runs ONCE for the entire test session,
    not once per test. This saves time because embedding is slow (~5s).

    The fixture:
    1. Creates temp markdown files with known clinical content
    2. Loads and chunks them
    3. Embeds and stores in a temp ChromaDB
    4. Returns the path to the ChromaDB directory
    """
    import chromadb

    # Create temp directory with test clinical documents
    data_dir = tmp_path_factory.mktemp("rag_data")

    (data_dir / "nyha.md").write_text(
        "# NYHA Classification\n\n"
        "NYHA Class I: No limitation of physical activity.\n"
        "NYHA Class II: Slight limitation. Ordinary activity causes fatigue.\n"
        "NYHA Class III: Marked limitation. Less than ordinary activity causes symptoms. "
        "Patients are candidates for CardioMEMS implantation.\n"
        "NYHA Class IV: Unable to carry on any physical activity without discomfort."
    )

    (data_dir / "gdmt.md").write_text(
        "# GDMT Recommendations for HFrEF\n\n"
        "The four pillars of guideline-directed medical therapy:\n"
        "1. ACE inhibitor, ARB, or ARNI (sacubitril/valsartan)\n"
        "2. Beta-blocker (carvedilol, metoprolol succinate, bisoprolol)\n"
        "3. Mineralocorticoid receptor antagonist (spironolactone, eplerenone)\n"
        "4. SGLT2 inhibitor (dapagliflozin, empagliflozin)\n\n"
        "All four classes reduce mortality in heart failure with reduced ejection fraction."
    )

    (data_dir / "risk_factors.md").write_text(
        "# Heart Failure Risk Factors\n\n"
        "Key prognostic indicators include:\n"
        "- Ejection fraction: normal 55-70%, reduced (HFrEF) ≤ 40%\n"
        "- Serum creatinine: elevated > 1.5 mg/dL indicates renal impairment\n"
        "- Serum sodium: hyponatremia < 135 mEq/L is a poor prognostic sign\n"
        "- Age: risk increases significantly after age 65\n"
        "- Anemia: hemoglobin < 13 g/dL in men, < 12 g/dL in women"
    )

    # Load, chunk, embed, store
    docs = load_markdown_files(str(data_dir))
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    chroma_ef = get_chroma_embedding_function()

    persist_dir = str(tmp_path_factory.mktemp("chroma_db"))
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="clinical_docs",
        embedding_function=chroma_ef,
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

    return persist_dir
