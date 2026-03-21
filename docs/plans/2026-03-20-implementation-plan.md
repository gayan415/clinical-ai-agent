# Clinical AI Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade clinical AI agent for heart failure risk assessment with full test pyramid, SRE practices, and MLOps — as a portfolio piece.

**Architecture:** LangChain agent on AWS Bedrock (Claude) with 3 tools: RAG retriever (HuggingFace embeddings + ChromaDB), risk prediction (XGBoost + PyTorch), and treatment recommender. Wrapped with circuit breakers, graceful degradation, inference logging, drift detection, and clinical safety guardrails.

**Tech Stack:** Python 3.11+, LangChain, AWS Bedrock, HuggingFace sentence-transformers, ChromaDB, XGBoost, PyTorch, FastAPI, Docker, pytest, ruff, mypy, GitHub Actions

**Project Root:** `/Users/gjayasun/git/AI/clinical-ai-agent`

**Development Methodology:** TDD + Agentic Development — engineer architects, directs, and reviews; AI agents execute under strict TDD discipline. Every task: write failing test → implement → verify → review → commit.

---

## Day 1 — Friday March 20 (2 hours)

### Task 1: Initialize Git Repo + Python Project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `Makefile`

**Step 1: Initialize git repo**

```bash
cd /Users/gjayasun/git/AI/clinical-ai-agent
git init
```

**Step 2: Create `.python-version`**

```
3.11
```

**Step 3: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual env
.venv/
venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.env.local

# ML artifacts
models/*.pkl
models/*.pt
models/*.joblib
*.h5

# ChromaDB
chroma_db/

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# Docker
*.tar

# Jupyter
.ipynb_checkpoints/
```

**Step 4: Create `pyproject.toml`**

```toml
[project]
name = "clinical-ai-agent"
version = "0.1.0"
description = "Clinical AI Agent for Heart Failure Risk Assessment"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "Gayan Jayasundara"}]

dependencies = [
    # LLM & Agent
    "langchain>=0.3.0",
    "langchain-aws>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-huggingface>=0.1.0",

    # RAG
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "pypdf>=4.0.0",

    # ML Models
    "xgboost>=2.0.0",
    "torch>=2.0.0",
    "scikit-learn>=1.4.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",

    # Model Serving
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "httpx>=0.27.0",

    # Observability
    "structlog>=24.0.0",

    # CLI
    "rich>=13.0.0",
    "typer>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "bandit>=1.7.0",
    "locust>=2.28.0",
]

[project.scripts]
clinical-agent = "cli:main"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "ml: ML-specific tests (data validation, model quality, behavioral)",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "perf: Performance tests",
]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

**Step 5: Create `Makefile`**

```makefile
.PHONY: install dev lint type-check security test test-unit test-ml test-integration test-e2e test-perf test-all clean docker-build docker-run

# Setup
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Quality
lint:
	ruff check .
	ruff format --check .

lint-fix:
	ruff check --fix .
	ruff format .

type-check:
	mypy cli.py agent/ model/ rag/ mlops/ sre/

security:
	bandit -r agent/ model/ rag/ mlops/ sre/ -ll

# Testing
test-unit:
	pytest tests/unit/ -v -m unit

test-ml:
	pytest tests/ml/ -v -m ml

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test-perf:
	python -m perf.benchmark

test: test-unit test-ml
	@echo "Core tests passed."

test-all: test-unit test-ml test-integration test-e2e
	@echo "All tests passed."

# Quality gate (CI)
ci: lint type-check security test
	@echo "CI pipeline passed."

# Docker (model service only)
docker-build:
	docker build -t clinical-ai-model:latest -f model/Dockerfile .

docker-run:
	docker run -p 8000:8000 clinical-ai-model:latest

# RAG
ingest:
	python -m rag.ingest

# Training
train:
	python -m model.train

# Agent
run:
	python cli.py

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
```

**Step 6: Commit**

```bash
git add .gitignore .python-version pyproject.toml Makefile docs/
git commit -m "feat: initialize project with pyproject.toml, Makefile, and design doc"
```

---

### Task 2: Create Directory Structure + Package Init Files

**Files:**
- Create: all `__init__.py` files and directory structure
- Create: all `tests/` subdirectories with `__init__.py`

**Step 1: Create all directories and init files**

```bash
# Source packages
mkdir -p agent model/data rag/data/guidelines rag/data/snippets mlops sre perf models

# Test packages
mkdir -p tests/unit tests/ml tests/integration tests/e2e

# Init files
touch agent/__init__.py model/__init__.py rag/__init__.py mlops/__init__.py sre/__init__.py perf/__init__.py
touch tests/__init__.py tests/unit/__init__.py tests/ml/__init__.py tests/integration/__init__.py tests/e2e/__init__.py

# Placeholder for model artifacts
touch models/.gitkeep

# GitHub Actions
mkdir -p .github/workflows
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: create project directory structure"
```

---

### Task 3: Create CI/CD Pipeline Scaffold

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create GitHub Actions workflow**

```yaml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Stage 1: Quality Gates
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Lint
        run: make lint
      - name: Type check
        run: make type-check
      - name: Security scan
        run: make security

  # Stage 2: Unit + ML Tests
  test-core:
    needs: quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Unit tests
        run: make test-unit
      - name: ML tests
        run: make test-ml

  # Stage 3: Integration Tests
  test-integration:
    needs: test-core
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Build model container
        run: make docker-build
      - name: Integration tests
        run: make test-integration

  # Stage 4: Performance (runs on main only)
  test-perf:
    needs: test-integration
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Performance benchmark
        run: make test-perf
```

**Step 2: Commit**

```bash
git add .github/
git commit -m "feat: add GitHub Actions CI pipeline scaffold"
```

---

### Task 4: Create README Skeleton

**Files:**
- Create: `README.md`

**Step 1: Write README skeleton** (will be polished on Day 6)

```markdown
# Clinical AI Agent for Heart Failure Risk Assessment

A production-grade clinical AI agent that demonstrates end-to-end ML engineering for healthcare: agentic workflows, RAG retrieval, predictive modeling, MLOps, and SRE practices.

## Architecture

[Architecture diagram from design doc]

### Request Flow

1. User inputs patient scenario via CLI
2. LangChain agent (AWS Bedrock / Claude) reasons about the scenario
3. Agent calls **retrieve_clinical_context** — RAG over ACC/AHA heart failure guidelines
4. Agent calls **predict_risk** — containerized ML model (XGBoost + PyTorch) returns risk score
5. Agent calls **recommend_treatment** — RAG + LLM reasoning returns GDMT-based recommendations
6. Agent synthesizes a clinical assessment with citations and safety disclaimers

## Tech Stack

| Component | Technology |
|---|---|
| LLM Orchestration | LangChain + AWS Bedrock (Claude) |
| RAG | HuggingFace sentence-transformers + ChromaDB |
| ML Models | XGBoost + PyTorch (champion/challenger) |
| Model Serving | FastAPI (containerized with Docker) |
| Testing | pytest (unit, ML behavioral, integration, e2e, performance) |
| MLOps | Custom model registry, inference logging, drift detection |
| SRE | Circuit breakers, graceful degradation, SLOs, runbook |
| CI/CD | GitHub Actions (5-stage pipeline) |

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (or Colima)
- AWS credentials with Bedrock access

### Setup
```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/clinical-ai-agent.git
cd clinical-ai-agent
pip install -e ".[dev]"

# Ingest clinical documents into vector store
make ingest

# Train ML models
make train

# Build model container
make docker-build

# Run the agent
make run
```

### Run Tests
```bash
make test          # Unit + ML tests
make test-all      # All tests including integration + e2e
make ci            # Full CI pipeline (lint + typecheck + security + tests)
```

## Project Structure

```
clinical-ai-agent/
├── agent/          # LangChain agent, tools, prompts, safety guardrails
├── model/          # XGBoost + PyTorch training, FastAPI serving, Dockerfile
├── rag/            # Document ingestion, embedding, ChromaDB retrieval
├── mlops/          # Model registry, inference logging, drift detection
├── sre/            # Circuit breakers, health checks, graceful degradation
├── perf/           # Performance benchmarks, load testing
├── tests/          # Full test pyramid (unit, ml, integration, e2e)
├── docs/           # Design docs, AWS deployment guide, runbook
└── cli.py          # CLI entry point
```

## Production Engineering Highlights

### Testing Pyramid
- **Unit tests:** Data validation, feature extraction, RAG chunking, circuit breaker state
- **ML behavioral tests:** Invariance, directional, minimum functionality, edge cases
- **Integration tests:** Agent ↔ RAG, Agent ↔ Model API, container health checks
- **Performance tests:** Latency benchmarks, load testing, regression gates

### MLOps
- File-based model registry with champion/challenger pattern
- Inference logging (structured JSON, CloudWatch-ready)
- Feature drift detection (Population Stability Index)
- Model quality gates (AUC > 0.75, recall > 0.60, latency < 100ms p99)

### SRE Practices
- Circuit breakers for model service and LLM calls
- Graceful degradation hierarchy (full → degraded → minimal)
- SLO definitions (availability, latency, error rate)
- Health check endpoints (/health, /ready)
- Operational runbook

### Clinical Safety
- Confidence thresholds with clinician review flags
- Immutable audit trail for every prediction and agent decision
- Human-in-the-loop: agent recommends, never decides
- All responses include clinical disclaimer

## AWS Deployment Architecture (Documented)

See [docs/aws-deployment.md](docs/aws-deployment.md) for the production AWS architecture:
SageMaker endpoints, ECR, ECS/Fargate, CloudWatch, S3, Step Functions, HIPAA-compliant VPC design.

## Author

**Gayan Jayasundara** — Senior Engineering Manager / SRE Leader
- 15+ years building production systems at scale
- Built AgenticOps: multi-agent AI platform on AWS Bedrock (90% incident resolution time reduction)
- 22-person SRE org, 300TB+ daily data processing, 99.96%+ uptime
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "feat: add README skeleton with architecture and project overview"
```

---

### Task 5: Install Dependencies + Verify Setup

**Step 1: Create virtual environment and install**

```bash
cd /Users/gjayasun/git/AI/clinical-ai-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 2: Verify tools work**

```bash
ruff check .
pytest --co  # collect tests (should find 0 for now)
```

**Step 3: Write a smoke test to prove the setup works**

Create `tests/unit/test_smoke.py`:

```python
import pytest


@pytest.mark.unit
def test_project_imports():
    """Verify all packages are importable."""
    import agent
    import model
    import rag
    import mlops
    import sre
    import perf


@pytest.mark.unit
def test_python_version():
    """Verify we're running Python 3.11+."""
    import sys
    assert sys.version_info >= (3, 11)
```

**Step 4: Run it**

```bash
pytest tests/unit/test_smoke.py -v -m unit
```

Expected: 2 PASSED

**Step 5: Commit**

```bash
git add tests/unit/test_smoke.py
git commit -m "test: add smoke tests verifying project setup"
```

---

### Task 6: Create GitHub Repo + Push

**Step 1: Create repo on GitHub**

```bash
gh repo create clinical-ai-agent --public --source=. --remote=origin --description="Clinical AI Agent for Heart Failure Risk Assessment — LangChain + RAG + ML + MLOps + SRE"
```

**Step 2: Push**

```bash
git push -u origin main
```

---

## Day 2 — Saturday March 21 (3 hours)

### Task 7: Create Curated Clinical Snippets

**Files:**
- Create: `rag/data/snippets/nyha_classification.md`
- Create: `rag/data/snippets/gdmt_recommendations.md`
- Create: `rag/data/snippets/cardiomems_protocol.md`
- Create: `rag/data/snippets/hf_risk_factors.md`

**Step 1: Write curated clinical content**

Write 4 markdown files with authoritative clinical content covering:
- NYHA Class I-IV definitions and symptoms
- GDMT recommendations for each HF stage (ACEi/ARB, beta-blockers, MRA, SGLT2i, ARNI)
- CardioMEMS PA pressure monitoring protocol (normal ranges, intervention thresholds)
- Heart failure risk factors (age, EF, creatinine, sodium, anemia, diabetes, hypertension)

Each file should be 200-400 words with specific clinical values and thresholds.

**Step 2: Commit**

```bash
git add rag/data/snippets/
git commit -m "feat: add curated clinical guideline snippets for RAG"
```

---

### Task 8: RAG Ingestion — TDD

**Files:**
- Create: `tests/unit/test_rag_chunking.py`
- Create: `rag/ingest.py`

**Step 1: Write failing tests for chunking**

```python
import pytest
from rag.ingest import chunk_text, load_markdown_files


@pytest.mark.unit
class TestChunking:
    def test_chunk_splits_text(self):
        text = "word " * 1000  # ~1000 words
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_overlap_exists(self):
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        # Last tokens of chunk[0] should appear in start of chunk[1]
        assert chunks[0][-50:] in chunks[1][:100] or len(chunks) == 1

    def test_chunk_preserves_all_content(self):
        text = "The quick brown fox jumps over the lazy dog. " * 100
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        reassembled = " ".join(chunks)
        # All original sentences should appear somewhere
        assert "quick brown fox" in reassembled

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", chunk_size=500, chunk_overlap=50)
        assert chunks == []


@pytest.mark.unit
class TestLoadMarkdown:
    def test_loads_snippets_directory(self, tmp_path):
        # Create test markdown files
        (tmp_path / "test1.md").write_text("# Test 1\nContent one.")
        (tmp_path / "test2.md").write_text("# Test 2\nContent two.")
        docs = load_markdown_files(str(tmp_path))
        assert len(docs) == 2

    def test_returns_metadata_with_source(self, tmp_path):
        (tmp_path / "test.md").write_text("# Test\nContent.")
        docs = load_markdown_files(str(tmp_path))
        assert "source" in docs[0].metadata
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_rag_chunking.py -v -m unit
```

Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Implement `rag/ingest.py`**

Implement:
- `load_markdown_files(directory: str) -> list[Document]` — loads all `.md` files from a directory
- `load_pdf(path: str) -> list[Document]` — loads a PDF and extracts text
- `chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]` — splits text into overlapping chunks
- `ingest_documents(data_dir: str, persist_dir: str) -> None` — full pipeline: load → chunk → embed → store in ChromaDB

Use:
- `langchain_community.document_loaders` for PDF loading
- `langchain.text_splitter.RecursiveCharacterTextSplitter` for chunking
- `langchain_huggingface.HuggingFaceEmbeddings` with model `all-MiniLM-L6-v2`
- `langchain_community.vectorstores.Chroma` for storage

**Step 4: Run tests — verify they pass**

```bash
pytest tests/unit/test_rag_chunking.py -v -m unit
```

Expected: ALL PASSED

**Step 5: Commit**

```bash
git add rag/ingest.py tests/unit/test_rag_chunking.py
git commit -m "feat: add RAG ingestion pipeline with chunking and markdown loading"
```

---

### Task 9: RAG Retriever — TDD

**Files:**
- Create: `tests/unit/test_retriever.py`
- Create: `rag/retriever.py`

**Step 1: Write failing tests for retriever**

```python
import pytest
from rag.retriever import ClinicalRetriever


@pytest.mark.unit
class TestClinicalRetriever:
    def test_query_returns_results(self, ingested_db):
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What is NYHA Class III?")
        assert len(results) > 0

    def test_results_have_content_and_source(self, ingested_db):
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("GDMT recommendations")
        for result in results:
            assert result.page_content
            assert "source" in result.metadata

    def test_top_k_limits_results(self, ingested_db):
        retriever = ClinicalRetriever(persist_dir=ingested_db, top_k=2)
        results = retriever.query("heart failure")
        assert len(results) <= 2

    def test_empty_query_returns_empty(self, ingested_db):
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("")
        assert isinstance(results, list)
```

Note: `ingested_db` is a pytest fixture that creates a temporary ChromaDB with test documents. Define it in `tests/conftest.py`.

**Step 2: Run tests — verify they fail**

**Step 3: Implement `rag/retriever.py`**

Implement:
- `ClinicalRetriever` class with `__init__(persist_dir, top_k=5)` and `query(question: str) -> list[Document]`
- Uses ChromaDB similarity search under the hood
- Returns documents with content and metadata

**Step 4: Run tests — verify they pass**

**Step 5: Commit**

```bash
git add rag/retriever.py tests/unit/test_retriever.py tests/conftest.py
git commit -m "feat: add clinical retriever with ChromaDB similarity search"
```

---

### Task 10: RAG Retrieval Quality Tests

**Files:**
- Create: `tests/ml/test_rag_quality.py`

**Step 1: Write retrieval quality tests**

```python
import pytest
from rag.retriever import ClinicalRetriever


@pytest.mark.ml
class TestRetrievalQuality:
    """Known queries must return expected documents."""

    def test_nyha_query_returns_nyha_doc(self, ingested_db):
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What are the NYHA functional classifications?")
        sources = [r.metadata.get("source", "") for r in results]
        assert any("nyha" in s.lower() for s in sources)

    def test_gdmt_query_returns_treatment_doc(self, ingested_db):
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What medications are recommended for HFrEF?")
        content = " ".join([r.page_content for r in results])
        assert any(drug in content.lower() for drug in ["ace inhibitor", "beta-blocker", "arni", "sglt2"])

    def test_risk_query_returns_risk_factors(self, ingested_db):
        retriever = ClinicalRetriever(persist_dir=ingested_db)
        results = retriever.query("What are the risk factors for heart failure mortality?")
        content = " ".join([r.page_content for r in results])
        assert any(factor in content.lower() for factor in ["ejection fraction", "creatinine", "age"])
```

**Step 2: Run and verify**

```bash
pytest tests/ml/test_rag_quality.py -v -m ml
```

**Step 3: Commit**

```bash
git add tests/ml/test_rag_quality.py
git commit -m "test: add RAG retrieval quality tests for known clinical queries"
```

---

### Task 11: Run Full Ingestion + Verify

**Step 1: Add a `__main__` block to `rag/ingest.py`**

```python
if __name__ == "__main__":
    ingest_documents(
        data_dir="rag/data",
        persist_dir="chroma_db",
    )
    print("Ingestion complete.")
```

**Step 2: Run ingestion**

```bash
make ingest
```

**Step 3: Verify with a manual query test**

```bash
python -c "from rag.retriever import ClinicalRetriever; r = ClinicalRetriever('chroma_db'); print(r.query('NYHA Class III')[0].page_content[:200])"
```

**Step 4: Commit**

```bash
git add rag/
git commit -m "feat: complete RAG pipeline — ingest, embed, retrieve"
```

---

## Day 3 — Sunday March 22 (3 hours)

### Task 12: Download UCI Heart Failure Dataset

**Files:**
- Create: `model/data/heart_failure.csv`

**Step 1: Download the dataset**

```bash
# UCI Heart Failure Clinical Records Dataset (299 rows, 13 features)
# Source: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv" -o model/data/heart_failure.csv
```

**Step 2: Verify**

```bash
head -5 model/data/heart_failure.csv
wc -l model/data/heart_failure.csv  # Should be 300 (299 + header)
```

**Step 3: Commit**

```bash
git add model/data/heart_failure.csv
git commit -m "data: add UCI heart failure clinical records dataset"
```

---

### Task 13: Data Validation Tests — TDD

**Files:**
- Create: `tests/ml/test_data_validation.py`
- Create: `model/features.py`

**Step 1: Write failing data validation tests**

```python
import pytest
import pandas as pd
from model.features import validate_dataframe, extract_features, FEATURE_COLUMNS, TARGET_COLUMN


@pytest.mark.ml
class TestDataValidation:
    def test_dataset_has_expected_columns(self):
        df = pd.read_csv("model/data/heart_failure.csv")
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_values(self):
        df = pd.read_csv("model/data/heart_failure.csv")
        assert df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum() == 0

    def test_age_in_valid_range(self):
        df = pd.read_csv("model/data/heart_failure.csv")
        assert df["age"].between(0, 120).all()

    def test_ejection_fraction_in_valid_range(self):
        df = pd.read_csv("model/data/heart_failure.csv")
        assert df["ejection_fraction"].between(0, 100).all()

    def test_binary_columns_are_binary(self):
        df = pd.read_csv("model/data/heart_failure.csv")
        binary_cols = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]
        for col in binary_cols:
            assert set(df[col].unique()).issubset({0, 1}), f"{col} is not binary"

    def test_target_distribution_not_extreme(self):
        df = pd.read_csv("model/data/heart_failure.csv")
        ratio = df[TARGET_COLUMN].mean()
        assert 0.1 < ratio < 0.9, f"Target is too imbalanced: {ratio}"


@pytest.mark.ml
class TestFeatureExtraction:
    def test_extract_features_returns_correct_shape(self):
        df = pd.read_csv("model/data/heart_failure.csv")
        X, y = extract_features(df)
        assert X.shape[1] == len(FEATURE_COLUMNS)
        assert len(y) == len(df)

    def test_validate_input_rejects_bad_data(self):
        bad_data = {"age": -5, "ejection_fraction": 200}
        errors = validate_dataframe(bad_data)
        assert len(errors) > 0

    def test_validate_input_accepts_good_data(self):
        good_data = {
            "age": 65, "anaemia": 0, "creatinine_phosphokinase": 150,
            "diabetes": 1, "ejection_fraction": 30, "high_blood_pressure": 1,
            "platelets": 250000, "serum_creatinine": 1.2, "serum_sodium": 137,
            "sex": 1, "smoking": 0, "time": 120,
        }
        errors = validate_dataframe(good_data)
        assert len(errors) == 0
```

**Step 2: Run — verify fail**

**Step 3: Implement `model/features.py`**

Implement:
- `FEATURE_COLUMNS`: list of 12 feature column names
- `TARGET_COLUMN`: `"DEATH_EVENT"`
- `extract_features(df) -> tuple[np.ndarray, np.ndarray]`: returns X, y
- `validate_dataframe(data: dict) -> list[str]`: validates input ranges, returns list of errors

**Step 4: Run — verify pass**

**Step 5: Commit**

```bash
git add model/features.py tests/ml/test_data_validation.py
git commit -m "feat: add feature extraction and data validation with tests"
```

---

### Task 14: Train XGBoost Model — TDD

**Files:**
- Create: `tests/ml/test_model_quality.py`
- Create: `model/train.py`
- Create: `model/evaluate.py`

**Step 1: Write failing model quality tests**

```python
import pytest
from model.train import train_xgboost, train_pytorch
from model.evaluate import evaluate_model


@pytest.mark.ml
class TestModelQuality:
    """Quality gates — model must meet these to deploy."""

    def test_xgboost_auc_above_threshold(self, trained_xgboost):
        model, X_test, y_test = trained_xgboost
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["auc"] > 0.75, f"AUC {metrics['auc']:.3f} below threshold 0.75"

    def test_xgboost_recall_per_class(self, trained_xgboost):
        model, X_test, y_test = trained_xgboost
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["recall_0"] > 0.60, f"Class 0 recall {metrics['recall_0']:.3f} below 0.60"
        assert metrics["recall_1"] > 0.60, f"Class 1 recall {metrics['recall_1']:.3f} below 0.60"

    def test_pytorch_auc_above_threshold(self, trained_pytorch):
        model, X_test, y_test = trained_pytorch
        metrics = evaluate_model(model, X_test, y_test, model_type="pytorch")
        assert metrics["auc"] > 0.70, f"AUC {metrics['auc']:.3f} below threshold 0.70"

    def test_model_size_under_limit(self, trained_xgboost):
        import os
        model, _, _ = trained_xgboost
        # Save temporarily to check size
        import joblib, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            joblib.dump(model, f.name)
            size_mb = os.path.getsize(f.name) / (1024 * 1024)
        assert size_mb < 50, f"Model size {size_mb:.1f}MB exceeds 50MB limit"
```

Note: `trained_xgboost` and `trained_pytorch` are pytest fixtures in `tests/conftest.py` that train models with a fixed seed on the UCI dataset.

**Step 2: Run — verify fail**

**Step 3: Implement `model/train.py`**

Implement:
- `train_xgboost(X_train, y_train, seed=42) -> XGBClassifier`
- `train_pytorch(X_train, y_train, seed=42, epochs=100) -> nn.Module`
  - Architecture: `Linear(12, 64) → ReLU → Dropout(0.3) → Linear(64, 32) → ReLU → Dropout(0.3) → Linear(32, 1) → Sigmoid`
- `train_and_save(data_path, output_dir, seed=42) -> dict` — full pipeline: load data → split → train both → evaluate → save best as active

**Step 4: Implement `model/evaluate.py`**

Implement:
- `evaluate_model(model, X_test, y_test, model_type="xgboost") -> dict` — returns accuracy, auc, precision, recall (per class), f1
- `compare_models(results: dict) -> str` — generates side-by-side comparison report
- `print_classification_report(metrics: dict) -> None`

**Step 5: Run — verify pass**

**Step 6: Commit**

```bash
git add model/train.py model/evaluate.py tests/ml/test_model_quality.py tests/conftest.py
git commit -m "feat: add XGBoost + PyTorch training with quality gate tests"
```

---

### Task 15: Behavioral Tests — TDD

**Files:**
- Create: `tests/ml/test_behavioral.py`

**Step 1: Write behavioral tests**

```python
import pytest
import numpy as np


@pytest.mark.ml
class TestBehavioral:
    """CheckList-style behavioral tests for model correctness."""

    def test_invariance_name_change(self, trained_xgboost):
        """Changing non-feature data shouldn't affect prediction."""
        model, _, _ = trained_xgboost
        features = np.array([[65, 0, 150, 1, 30, 1, 250000, 1.2, 137, 1, 0, 120]])
        pred1 = model.predict_proba(features)[0]
        pred2 = model.predict_proba(features)[0]  # Same features = same prediction
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_directional_lower_ef_higher_risk(self, trained_xgboost):
        """Lower ejection fraction should increase risk."""
        model, _, _ = trained_xgboost
        # EF=50 (normal)
        high_ef = np.array([[65, 0, 150, 1, 50, 1, 250000, 1.2, 137, 1, 0, 120]])
        # EF=15 (very low)
        low_ef = np.array([[65, 0, 150, 1, 15, 1, 250000, 1.2, 137, 1, 0, 120]])
        risk_high_ef = model.predict_proba(high_ef)[0][1]
        risk_low_ef = model.predict_proba(low_ef)[0][1]
        assert risk_low_ef > risk_high_ef, "Lower EF should mean higher risk"

    def test_directional_higher_creatinine_higher_risk(self, trained_xgboost):
        """Higher serum creatinine should increase risk."""
        model, _, _ = trained_xgboost
        normal_cr = np.array([[65, 0, 150, 1, 30, 1, 250000, 1.0, 137, 1, 0, 120]])
        high_cr = np.array([[65, 0, 150, 1, 30, 1, 250000, 5.0, 137, 1, 0, 120]])
        risk_normal = model.predict_proba(normal_cr)[0][1]
        risk_high = model.predict_proba(high_cr)[0][1]
        assert risk_high > risk_normal, "Higher creatinine should mean higher risk"

    def test_known_high_risk_flagged(self, trained_xgboost):
        """Patient with very poor indicators must be flagged high risk."""
        model, _, _ = trained_xgboost
        # Old, low EF, high creatinine, low sodium, short follow-up
        high_risk = np.array([[85, 1, 7000, 1, 14, 1, 100000, 9.0, 113, 1, 1, 4]])
        pred = model.predict(high_risk)[0]
        assert pred == 1, "Known high-risk patient should be flagged"

    def test_edge_case_all_zeros(self, trained_xgboost):
        """Model should not crash on edge case inputs."""
        model, _, _ = trained_xgboost
        zeros = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        pred = model.predict(zeros)
        assert pred[0] in [0, 1], "Should return valid prediction"
```

**Step 2: Run — verify pass (model already trained from Task 14 fixtures)**

**Step 3: Commit**

```bash
git add tests/ml/test_behavioral.py
git commit -m "test: add behavioral tests — invariance, directional, edge cases"
```

---

### Task 16: FastAPI Prediction Endpoint — TDD

**Files:**
- Create: `tests/integration/test_model_api.py`
- Create: `model/predict.py`

**Step 1: Write failing API tests**

```python
import pytest
from fastapi.testclient import TestClient
from model.predict import create_app


@pytest.mark.integration
class TestModelAPI:
    def test_health_endpoint(self):
        app = create_app(model_path="models/xgboost_hf_risk.pkl")
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_ready_endpoint_with_model(self):
        app = create_app(model_path="models/xgboost_hf_risk.pkl")
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_predict_valid_input(self):
        app = create_app(model_path="models/xgboost_hf_risk.pkl")
        client = TestClient(app)
        response = client.post("/predict", json={
            "age": 65, "anaemia": 0, "creatinine_phosphokinase": 150,
            "diabetes": 1, "ejection_fraction": 30, "high_blood_pressure": 1,
            "platelets": 250000, "serum_creatinine": 1.2, "serum_sodium": 137,
            "sex": 1, "smoking": 0, "time": 120,
        })
        assert response.status_code == 200
        body = response.json()
        assert "risk_score" in body
        assert "confidence" in body
        assert "model_version" in body
        assert 0 <= body["risk_score"] <= 1

    def test_predict_invalid_input_returns_422(self):
        app = create_app(model_path="models/xgboost_hf_risk.pkl")
        client = TestClient(app)
        response = client.post("/predict", json={"age": -5})
        assert response.status_code == 422

    def test_predict_logs_inference(self):
        app = create_app(model_path="models/xgboost_hf_risk.pkl")
        client = TestClient(app)
        response = client.post("/predict", json={
            "age": 65, "anaemia": 0, "creatinine_phosphokinase": 150,
            "diabetes": 1, "ejection_fraction": 30, "high_blood_pressure": 1,
            "platelets": 250000, "serum_creatinine": 1.2, "serum_sodium": 137,
            "sex": 1, "smoking": 0, "time": 120,
        })
        body = response.json()
        assert "latency_ms" in body
```

**Step 2: Run — verify fail**

**Step 3: Implement `model/predict.py`**

Implement FastAPI app with:
- `POST /predict` — accepts patient features, validates, runs prediction, returns risk_score + confidence + model_version + latency_ms
- `GET /health` — liveness check
- `GET /ready` — readiness check (model loaded)
- `create_app(model_path: str) -> FastAPI` — factory function for testability
- Inference timing with `time.perf_counter()`
- Structured logging with `structlog`

**Step 4: Run — verify pass**

**Step 5: Commit**

```bash
git add model/predict.py tests/integration/test_model_api.py
git commit -m "feat: add FastAPI prediction endpoint with health checks and logging"
```

---

### Task 17: Dockerfile for Model Service

**Files:**
- Create: `model/Dockerfile`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only model serving dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn xgboost scikit-learn numpy pandas joblib structlog

# Copy model code and artifacts
COPY model/ model/
COPY models/ models/
COPY mlops/ mlops/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "model.predict:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Build and test**

```bash
make docker-build
docker run -d -p 8000:8000 --name clinical-model clinical-ai-model:latest
curl http://localhost:8000/health
curl http://localhost:8000/ready
docker stop clinical-model && docker rm clinical-model
```

**Step 3: Commit**

```bash
git add model/Dockerfile
git commit -m "feat: add Dockerfile for model serving container"
```

---

## Day 4 — Monday March 24 (90 minutes)

### Task 18: Clinical Safety Module — TDD

**Files:**
- Create: `tests/unit/test_safety.py`
- Create: `agent/safety.py`
- Create: `agent/prompts.py`

**Step 1: Write failing safety tests**

```python
import pytest
from agent.safety import (
    format_disclaimer,
    check_confidence,
    format_audit_entry,
    CLINICAL_DISCLAIMER,
)


@pytest.mark.unit
class TestClinicalSafety:
    def test_disclaimer_always_present(self):
        disclaimer = format_disclaimer("Some recommendation")
        assert CLINICAL_DISCLAIMER in disclaimer

    def test_low_confidence_flagged(self):
        result = check_confidence(0.55)
        assert result["requires_review"] is True
        assert "low confidence" in result["message"].lower()

    def test_high_confidence_not_flagged(self):
        result = check_confidence(0.85)
        assert result["requires_review"] is False

    def test_audit_entry_has_required_fields(self):
        entry = format_audit_entry(
            patient_id="P001",
            input_features={"age": 65},
            prediction=0.75,
            model_version="v1",
            tools_called=["predict_risk"],
        )
        assert "timestamp" in entry
        assert entry["patient_id"] == "P001"
        assert entry["model_version"] == "v1"
```

**Step 2: Run — verify fail**

**Step 3: Implement `agent/safety.py` and `agent/prompts.py`**

`agent/safety.py`:
- `CLINICAL_DISCLAIMER` constant
- `check_confidence(score: float, threshold: float = 0.7) -> dict`
- `format_disclaimer(recommendation: str) -> str`
- `format_audit_entry(...) -> dict` — immutable audit log entry with timestamp

`agent/prompts.py`:
- System prompt that prevents diagnosis, prevents prescribing, requires citations, includes disclaimer
- Tool descriptions for the 3 agent tools

**Step 4: Run — verify pass**

**Step 5: Commit**

```bash
git add agent/safety.py agent/prompts.py tests/unit/test_safety.py
git commit -m "feat: add clinical safety module — disclaimers, confidence checks, audit trail"
```

---

### Task 19: Circuit Breaker — TDD

**Files:**
- Create: `tests/unit/test_circuit_breaker.py`
- Create: `sre/circuit_breaker.py`

**Step 1: Write failing circuit breaker tests**

```python
import pytest
from sre.circuit_breaker import CircuitBreaker, CircuitState


@pytest.mark.unit
class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_raises(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        with pytest.raises(Exception, match="circuit.*open"):
            cb.check()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)  # 0s timeout for test
        cb.record_failure()
        assert cb.state == CircuitState.HALF_OPEN  # Immediately half-open due to 0s timeout
```

**Step 2: Run — verify fail**

**Step 3: Implement `sre/circuit_breaker.py`**

Implement:
- `CircuitState` enum: CLOSED, OPEN, HALF_OPEN
- `CircuitBreaker` class with: `check()`, `record_success()`, `record_failure()`, `state` property
- Time-based recovery from OPEN → HALF_OPEN

**Step 4: Run — verify pass**

**Step 5: Commit**

```bash
git add sre/circuit_breaker.py tests/unit/test_circuit_breaker.py
git commit -m "feat: add circuit breaker with CLOSED/OPEN/HALF_OPEN states"
```

---

### Task 20: LangChain Agent + Tools

**Files:**
- Create: `agent/tools.py`
- Create: `agent/agent.py`
- Create: `tests/integration/test_agent_rag.py`
- Create: `tests/integration/test_agent_model.py`

**Step 1: Implement `agent/tools.py`**

Three LangChain tools:
- `retrieve_clinical_context(query: str) -> str` — calls RAG retriever, formats results
- `predict_risk(patient_data: str) -> str` — parses patient features, calls model API via httpx, wraps with circuit breaker
- `recommend_treatment(context: str) -> str` — RAG over GDMT guidelines + formats recommendation with disclaimer

**Step 2: Implement `agent/agent.py`**

- `ClinicalAgent` class wrapping LangChain agent executor
- Uses Bedrock Claude via `langchain_aws.ChatBedrock`
- System prompt from `agent/prompts.py`
- Graceful degradation: if model service down, return RAG-only results
- Audit logging for every interaction

**Step 3: Write integration tests**

Test agent ↔ RAG (does it retrieve relevant context?) and agent ↔ model (does it call the API and handle failures?). Use mocks for Bedrock to avoid real API calls in tests.

**Step 4: Run — verify pass**

**Step 5: Commit**

```bash
git add agent/tools.py agent/agent.py tests/integration/
git commit -m "feat: add LangChain agent with RAG, prediction, and recommendation tools"
```

---

### Task 21: CLI Entry Point

**Files:**
- Create: `cli.py`

**Step 1: Implement CLI**

```python
"""Clinical AI Agent — CLI entry point."""
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def assess(scenario: str = typer.Argument(..., help="Patient scenario description")):
    """Run clinical AI assessment for a patient scenario."""
    # Initialize agent, run scenario, print results with rich formatting
    ...

@app.command()
def demo():
    """Run demo with predefined patient scenarios."""
    ...

def main():
    app()

if __name__ == "__main__":
    main()
```

Uses `rich` for formatted terminal output (colored risk scores, boxed recommendations, citations).

**Step 2: Test manually**

```bash
python cli.py assess "65-year-old male, ejection fraction 30%, serum creatinine 1.9, NYHA Class III, on lisinopril and metoprolol"
```

**Step 3: Commit**

```bash
git add cli.py
git commit -m "feat: add CLI entry point with rich formatted output"
```

---

## Day 5 — Tuesday March 25 (90 minutes)

### Task 22: Model Registry — TDD

**Files:**
- Create: `tests/unit/test_registry.py`
- Create: `mlops/registry.py`

Implement file-based model registry:
- `register_model(name, version, path, metrics) -> None`
- `get_active_model() -> dict`
- `promote_model(name, version) -> None`
- `rollback(name) -> None`
- `list_models() -> list[dict]`

TDD: write tests first for register, promote, rollback, list. Use `tmp_path` fixture.

**Commit:** `"feat: add file-based model registry with champion/challenger"`

---

### Task 23: Inference Monitor — TDD

**Files:**
- Create: `tests/unit/test_monitor.py`
- Create: `mlops/monitor.py`

Implement:
- `InferenceLogger` — logs every prediction to structured JSON (append-only)
- `log_inference(input_features, prediction, confidence, model_version, latency_ms) -> None`
- `get_metrics_summary() -> dict` — p50/p95/p99 latency, prediction distribution, error rate
- `log_llm_call(tokens_in, tokens_out, latency_ms, cost_estimate) -> None`

TDD: test logging, test metrics aggregation, test append-only behavior.

**Commit:** `"feat: add inference monitoring with structured logging and metrics"`

---

### Task 24: Drift Detection — TDD

**Files:**
- Create: `tests/unit/test_drift.py`
- Create: `mlops/drift.py`

Implement:
- `compute_baseline(training_data: pd.DataFrame) -> dict` — per-feature distribution stats
- `compute_psi(baseline: dict, current: pd.DataFrame) -> dict` — Population Stability Index per feature
- `check_drift(psi_scores: dict, threshold: float = 0.2) -> list[str]` — returns list of drifted features

TDD: test PSI computation with known distributions, test drift detection thresholds.

**Commit:** `"feat: add feature drift detection using Population Stability Index"`

---

### Task 25: SLO Config + Health Endpoints

**Files:**
- Create: `mlops/slo.py`
- Create: `sre/health.py`
- Create: `sre/resilience.py`

Implement:
- `slo.py` — SLO definitions as config, SLO checker that reads from inference logs
- `health.py` — `/health` and `/ready` endpoint logic
- `resilience.py` — Graceful degradation logic (full → degraded → minimal)

**Commit:** `"feat: add SLO definitions, health checks, and graceful degradation"`

---

### Task 26: Performance Benchmark

**Files:**
- Create: `perf/benchmark.py`
- Create: `perf/baseline.json`

Implement:
- Single request latency (cold + warm, 100 iterations)
- Batch inference (10, 100, 1000 patients)
- Report: p50, p95, p99 latency + throughput
- Save results to `perf/baseline.json`
- Compare against baseline — fail if regression > 10%

**Commit:** `"feat: add performance benchmarks with regression detection"`

---

### Task 27: Runbook

**Files:**
- Create: `docs/runbook.md`

Write operational runbook covering:
- Model latency spike → diagnosis → remediation
- Prediction drift detected → investigation → retrain
- LLM response quality degradation → fallback → recovery
- RAG returning irrelevant results → re-index
- Complete service outage → graceful degradation activation

**Commit:** `"docs: add operational runbook for production incidents"`

---

## Day 6 — Wednesday March 26 (60 minutes)

### Task 28: AWS Deployment Architecture Doc

**Files:**
- Create: `docs/aws-deployment.md`

Document:
- Architecture diagram (SageMaker, ECR, ECS, S3, CloudWatch, Step Functions)
- HIPAA considerations (encryption, audit, BAA, VPC)
- Component-by-component migration from local → AWS
- Cost estimation
- Scaling strategy

**Commit:** `"docs: add AWS deployment architecture guide"`

---

### Task 29: Polish README

Update README.md with:
- Final architecture diagram
- Actual model metrics from training
- Demo recording link or instructions
- Any adjustments based on what was actually built

**Commit:** `"docs: polish README with final architecture and metrics"`

---

### Task 30: Record Demo

**Step 1: Record CLI demo**

```bash
# Use asciinema or screen recording
asciinema rec demo.cast
python cli.py demo
# Ctrl+D to stop
```

Or use a screen recording tool and save as GIF.

**Step 2: Add to README**

**Step 3: Final commit + push**

```bash
git add .
git commit -m "feat: add demo recording and final polish"
git push
```

---

## Summary

| Day | Tasks | Commits |
|---|---|---|
| 1 (Fri) | Tasks 1-6: Repo, structure, CI, README, install, GitHub | 5-6 |
| 2 (Sat) | Tasks 7-11: Clinical snippets, RAG ingest, retriever, quality tests | 4-5 |
| 3 (Sun) | Tasks 12-17: Dataset, data validation, training, behavioral tests, API, Docker | 5-6 |
| 4 (Mon) | Tasks 18-21: Safety, circuit breaker, agent + tools, CLI | 4-5 |
| 5 (Tue) | Tasks 22-27: Registry, monitor, drift, SLOs, perf benchmark, runbook | 6 |
| 6 (Wed) | Tasks 28-30: AWS doc, README polish, demo recording | 3 |
| **Total** | **30 tasks** | **~30 commits** |
