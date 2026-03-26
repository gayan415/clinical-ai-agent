# Clinical AI Agent for Heart Failure Risk Assessment

A production-grade clinical AI agent that demonstrates end-to-end ML engineering for healthcare: agentic workflows, RAG retrieval, predictive modeling, MLOps, and SRE practices.

**Built with Agentic Development + TDD** — engineer architects and reviews, AI agents execute under strict test-driven discipline.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              CLI Interface                                │
│         "Assess patient: 65yo male, EF 30%..."           │
└──────────────────┬───────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────┐
│           LangChain Agent (AWS Bedrock / Claude)          │
│      Decides which tools to call, in what order           │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Tool 1: retrieve_clinical_context                        │
│  └── RAG: HuggingFace embeddings → ChromaDB → retrieval  │
│      Sources: ACC/AHA heart failure guidelines            │
│                                                           │
│  Tool 2: predict_risk                                     │
│  └── Containerized FastAPI endpoint                       │
│      Models: XGBoost + PyTorch (champion/challenger)      │
│      Data: UCI Heart Failure dataset                      │
│                                                           │
│  Tool 3: recommend_treatment                              │
│  └── RAG over GDMT guidelines + LLM reasoning            │
│      Evidence-based recommendations with citations        │
│                                                           │
├──────────────────────────────────────────────────────────┤
│  MLOps: Model registry, inference logging, drift detect  │
│  SRE: Circuit breakers, graceful degradation, SLOs       │
│  Safety: Confidence thresholds, audit trail, disclaimers │
└──────────────────────────────────────────────────────────┘
```

### Request Flow

1. User inputs patient scenario via CLI
2. LangChain agent (AWS Bedrock / Claude) reasons about the scenario
3. Agent calls **retrieve_clinical_context** — RAG over ACC/AHA heart failure guidelines
4. Agent calls **predict_risk** — containerized ML model returns risk score with confidence
5. Agent calls **recommend_treatment** — RAG + LLM reasoning returns GDMT-based recommendations
6. Agent synthesizes a clinical assessment with citations and safety disclaimers

## Tech Stack

| Component | Technology |
|---|---|
| LLM Orchestration | LangChain + AWS Bedrock (Claude) |
| RAG | HuggingFace sentence-transformers + ChromaDB |
| ML Models | XGBoost + PyTorch (champion/challenger) |
| Model Serving | FastAPI (containerized with Docker) |
| Testing | pytest — unit, ML behavioral, integration, e2e, performance |
| MLOps | Custom model registry, inference logging, drift detection (PSI) |
| SRE | Circuit breakers, graceful degradation, SLOs, runbook |
| CI/CD | GitHub Actions (4-stage pipeline) |
| Clinical Safety | Confidence thresholds, audit trail, human-in-the-loop |

## Quick Start

### Prerequisites
- Python 3.11+
- macOS: `brew install libomp` (required by XGBoost)
- Docker or Colima (optional — for containerized model service)
- AWS credentials with Bedrock access (for the agent LLM)

### Setup
```bash
git clone https://github.com/gayan415/clinical-ai-agent.git
cd clinical-ai-agent

# Full setup: install deps + download embedding model (~80MB)
make setup

# Ingest clinical documents into vector store (one-time, idempotent)
make ingest

# Train models — saves XGBoost + PyTorch with registry
make train
```

### Running Locally (2 terminals)

**Terminal 1 — Model prediction service:**
```bash
source .venv/bin/activate
uvicorn model.predict:app --port 8000
# Wait for: "Uvicorn running on http://0.0.0.0:8000"
```

**Terminal 2 — Agent CLI:**
```bash
source .venv/bin/activate
export AWS_REGION=us-east-1
export AWS_PROFILE=<your_aws_profile>

# Run a patient assessment
python cli.py assess "65-year-old male, ejection fraction 30%, creatinine 1.9, NYHA Class III"

# Run demo with predefined scenarios
python cli.py demo
```

### Environment Variables
```bash
# Embedding provider (default: huggingface — local, free)
export EMBEDDING_PROVIDER=huggingface  # or "bedrock" for AWS Titan Embed

# AWS (only needed for agent LLM and Bedrock embeddings)
export AWS_REGION=us-east-1
export AWS_PROFILE=<your_aws_profile>

# Model service URL (default: http://localhost:8000)
export MODEL_SERVICE_URL=http://localhost:8000
```

### Run Tests
```bash
make test          # Unit + ML tests (51 tests)
make test-all      # All tests including integration + e2e
make ci            # Full CI pipeline (lint + typecheck + security + tests)
```

### Verify Setup
```bash
# Check vector store has data
python -c "import chromadb; c=chromadb.PersistentClient('chroma_db'); print(f'Chunks: {c.get_or_create_collection(\"clinical_docs\").count()}')"

# Check model is trained
python -c "import joblib; m=joblib.load('models/xgboost_hf_risk.pkl'); print('Model loaded')"

# Check model API is running
curl http://localhost:8000/health
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

## Production Engineering

### Testing Pyramid
- **Unit tests:** Data validation, feature extraction, RAG chunking, circuit breaker state
- **ML behavioral tests:** Invariance, directional, minimum functionality, edge cases
- **Integration tests:** Agent-RAG, Agent-Model API, container health checks
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
- Confidence thresholds — low confidence flags clinician review
- Immutable audit trail for every prediction and agent decision
- Human-in-the-loop: agent recommends, never decides
- All responses include clinical disclaimer

## Development Methodology: TDD + Human-in-the-Loop Agentic

This project uses **TDD + Agentic Coding** — not vibe coding. Every feature follows this flow:

```
┌─────────────────────────────┐
│  1. DESIGN                  │  Human + AI brainstorm architecture,
│     (Spec & Plan)           │  validate approach, write design doc
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  2. TEACH                   │  AI explains new concepts (RAG, embeddings,
│     (Learn before code)     │  XGBoost, PyTorch) before writing any code
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  3. TEST                    │  Write failing tests first — tests define
│     (TDD red phase)        │  the behavior, not the implementation
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  4. IMPLEMENT               │  AI agent codes until tests pass
│     (TDD green phase)       │  (agentic execution with constraints)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  5. REVIEW & OWN            │  Human reviews PR, asks questions,
│     (Human-in-the-loop)     │  doesn't merge until every line is understood
└─────────────────────────────┘
```

**Why this works:**
- **Design** catches "building the wrong thing"
- **Tests** catch "building the thing wrong"
- **AI agent** handles execution speed
- **Human review** ensures judgment and ownership

**What this is NOT:**
- Not **vibe coding** — no accepting AI output blindly
- Not **prompt-driven** — human architects, reviews, and owns every decision
- Not **auto-pilot** — AI stops and teaches when introducing new concepts

30 tasks, 30 commits, full test pyramid, pre-commit security gates, branch + PR workflow.

## AWS Deployment Architecture

See [docs/aws-deployment.md](docs/aws-deployment.md) for the full production architecture:
- ECS Fargate (model + agent services), OpenSearch Serverless (vector search)
- Multi-AZ high availability, multi-region disaster recovery (RPO < 1h, RTO < 4h)
- 7-layer security (WAF, VPC isolation, KMS encryption, IAM roles, audit trail)
- HIPAA compliance (BAA, encryption at rest/transit, FDA 21 CFR Part 11 awareness)
- Blue/green model deployment, canary agent rollout, automated retraining pipeline
- Cost estimation: ~$660/month pilot, ~$2,950/month production (100K+ patients)

## Author

**Gayan Jayasundara** — Senior Engineering Manager / SRE Leader
- 15+ years building production systems at scale
- Built AgenticOps: multi-agent AI platform on AWS Bedrock (90% incident resolution time reduction)
- 22-person SRE org, 300TB+ daily data processing, 99.96%+ uptime
