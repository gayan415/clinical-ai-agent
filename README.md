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
- Docker (or Colima)
- AWS credentials with Bedrock access

### Setup
```bash
git clone https://github.com/gayan415/clinical-ai-agent.git
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

## Development Methodology

**TDD + Agentic Development** — This project was built using AI agents (Claude Code) as development partners, directed by the engineer with strict engineering discipline:

1. **Architect** — all design decisions made and validated by the engineer
2. **Direct** — work broken into 30 TDD tasks, constraints set, approaches reviewed
3. **Review** — every line of code reviewed before commit, tests must pass
4. **Own** — engineer understands and can defend every component and decision

This is how production AI teams will operate — engineers directing AI agents with engineering rigor, not blindly accepting generated code.

## AWS Deployment Architecture

See [docs/aws-deployment.md](docs/aws-deployment.md) for the production AWS architecture:
SageMaker endpoints, ECR, ECS/Fargate, CloudWatch, S3, Step Functions, HIPAA-compliant VPC design.

## Author

**Gayan Jayasundara** — Senior Engineering Manager / SRE Leader
- 15+ years building production systems at scale
- Built AgenticOps: multi-agent AI platform on AWS Bedrock (90% incident resolution time reduction)
- 22-person SRE org, 300TB+ daily data processing, 99.96%+ uptime
