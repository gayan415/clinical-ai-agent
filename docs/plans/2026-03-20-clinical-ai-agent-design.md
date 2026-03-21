# Clinical AI Agent — Design Document

**Date:** 2026-03-20
**Author:** Gayan Jayasundara
**Status:** Approved

---

## 1. Overview

A clinical AI agent for heart failure risk assessment that demonstrates production-grade engineering practices for healthcare AI systems. The agent receives patient scenarios, retrieves relevant clinical guidelines (RAG), predicts heart failure risk (ML models), and recommends evidence-based treatments.

**Primary goals:**
- Demonstrate full-stack AI/ML architecture (breadth)
- Show production engineering rigor — TDD, SRE practices, observability (depth)
- Serve as a talking piece for technical interviews

**Non-goals:**
- Production AWS deployment (documented, not built)
- Mobile application
- Real patient data or HIPAA-compliant infrastructure
- Image analysis / computer vision

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│              CLI Interface (Phase 1)                      │
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
│      Sources: ACC/AHA guidelines PDF + curated snippets   │
│                                                           │
│  Tool 2: predict_risk                                     │
│  └── HTTP call to containerized FastAPI endpoint          │
│      Models: XGBoost + PyTorch (champion/challenger)      │
│      Data: UCI Heart Failure dataset (299 patients)       │
│                                                           │
│  Tool 3: recommend_treatment                              │
│  └── RAG over GDMT guidelines + LLM reasoning            │
│      Returns evidence-based recommendations with sources  │
│                                                           │
├──────────────────────────────────────────────────────────┤
│  MLOps Layer                                              │
│  ├── Model registry (file-based, versioned)               │
│  ├── Inference logging (request/response, latency)        │
│  ├── Drift detection (input feature distributions)        │
│  └── Performance benchmarks                               │
├──────────────────────────────────────────────────────────┤
│  SRE Layer                                                │
│  ├── Circuit breakers (model service + LLM)               │
│  ├── Graceful degradation hierarchy                       │
│  ├── Health check endpoints                               │
│  ├── SLO definitions                                      │
│  └── Runbook                                              │
├──────────────────────────────────────────────────────────┤
│  Clinical Safety Layer                                    │
│  ├── Confidence thresholds                                │
│  ├── Audit trail (immutable logging)                      │
│  ├── Human-in-the-loop (recommend, never decide)          │
│  └── Disclaimer on every response                         │
└──────────────────────────────────────────────────────────┘

  Containerized (Docker/Colima):        Local Python:
  ┌─────────────────────┐    ┌──────────────────────────┐
  │ FastAPI Prediction   │    │ Agent + RAG + MLOps      │
  │ Service (port 8000)  │    │ ChromaDB (in-process)    │
  └─────────────────────┘    └──────────────────────────┘
```

### Request Flow

1. User inputs patient scenario via CLI
2. Agent reasons about the scenario, calls `retrieve_clinical_context` for relevant guidelines
3. Agent calls `predict_risk` with patient features → gets risk score from containerized model
4. Agent calls `recommend_treatment` with risk score + patient context → gets GDMT-based recommendations
5. Agent synthesizes everything into a clinical assessment with citations and disclaimer

---

## 3. Tech Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Industry standard for ML |
| LLM Orchestration | LangChain | JD requirement, agent framework |
| LLM Provider | AWS Bedrock (Claude) | Consistent with AgenticOps experience |
| Embeddings | HuggingFace sentence-transformers | JD requirement, high quality |
| Vector Store | ChromaDB (in-process) | Simple, no infra overhead |
| ML Models | XGBoost + PyTorch | XGBoost for tabular performance, PyTorch for JD requirement |
| Model Serving | FastAPI | Lightweight, async, OpenAPI docs |
| Containerization | Docker (Colima) | JD requirement, model deployment unit |
| Testing | pytest + pytest-cov | Standard, with ML-specific test patterns |
| CI/CD | GitHub Actions | Free, integrated with repo |
| Linting | ruff | Fast, replaces flake8+isort+black |
| Type Checking | mypy | Production rigor |
| Security Scan | bandit | Python security linter |
| Performance Testing | Custom scripts + locust | Load testing and benchmarks |

---

## 4. RAG Pipeline

### Document Sources
- **ACC/AHA Heart Failure Guidelines** — official clinical PDF
- **Curated snippets** — markdown summaries of key GDMT recommendations, CardioMEMS protocols, NYHA classifications

### Pipeline
```
PDF / Markdown → Text Extraction → Chunking (500 tokens, 50 overlap)
    → HuggingFace embedding (all-MiniLM-L6-v2) → ChromaDB storage
    → Retrieval (top-k=5, similarity search) → Context for agent
```

### Quality Controls
- Known queries must return expected documents (retrieval tests)
- Relevance scoring on retrieved chunks
- Empty retrieval handled gracefully
- No hallucinated citations (agent must cite source docs)

---

## 5. Prediction Models

### Dataset
- UCI Heart Failure Clinical Records (299 patients, 13 features)
- Features: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time, DEATH_EVENT
- Binary classification: death event (0/1)

### Models

**XGBoost (primary candidate):**
- Strong on tabular data, interpretable feature importance
- Expected AUC: 0.85+

**PyTorch Neural Network (challenger):**
- Simple feedforward network (input → 64 → 32 → 1)
- Demonstrates deep learning framework knowledge
- Compare against XGBoost on same metrics

### Champion/Challenger Pattern
- Both models trained and evaluated
- Side-by-side comparison report (accuracy, AUC, precision, recall, F1, latency)
- Best performer promoted to "active" in model registry
- Other kept as "candidate" for rollback

### Quality Gates (must pass to deploy)
- AUC > 0.75
- No class recall < 0.60
- Prediction latency < 100ms (p99)
- Model size < 50MB

---

## 6. Testing Strategy

### Testing Pyramid

**Unit Tests:**
- Data validation (schema, ranges, types, missing values)
- Feature extraction logic
- RAG chunking and embedding
- Prompt template rendering
- Model registry operations
- Circuit breaker state transitions

**ML-Specific Tests (Behavioral):**
- Invariance: changing patient name shouldn't change prediction
- Directional: higher age + lower EF should increase risk score
- Minimum functionality: known high-risk patients must be flagged
- Edge cases: all zeros, all max values, out-of-range inputs

**Integration Tests:**
- Agent ↔ RAG retrieval
- Agent ↔ Model API (HTTP contract)
- Model container health check
- End-to-end: scenario in → assessment out

**Performance Tests:**
- Single request latency (cold + warm start)
- Batch inference (10, 100, 1000 patients)
- Concurrent request load test
- Memory/CPU profiling
- Regression gates: fail if p99 latency regresses > 10%

---

## 7. MLOps Layer

### Model Registry (file-based)
```json
{
  "models": [
    {
      "name": "xgboost_hf_risk",
      "version": "v1",
      "path": "models/v1/xgboost_hf_risk.pkl",
      "status": "active",
      "metrics": {"auc": 0.87, "accuracy": 0.82, "f1": 0.79},
      "trained_at": "2026-03-22T10:00:00Z",
      "training_data_hash": "sha256:abc123..."
    }
  ]
}
```

### Inference Logging
- Every prediction logged: input_features, model_version, prediction, confidence, latency, timestamp
- Structured JSON format (CloudWatch/ELK-ready)
- Append-only audit trail

### Drift Detection
- Baseline: training data feature distributions
- Monitor: incoming request feature distributions
- Metric: Population Stability Index (PSI) per feature
- Alert threshold: PSI > 0.2 (significant drift)

### LLM Observability
- Token usage per request
- Tool call sequence logging
- Latency per tool + total agent response time
- Cost estimation per request

---

## 8. SRE Practices

### SLO Definitions
| Service | SLI | Target |
|---|---|---|
| Model API | Availability | 99.9% |
| Model API | Latency (p99) | < 200ms |
| Model API | Error rate | < 0.1% |
| Model API | Prediction quality (AUC) | > 0.75 |
| Agent | End-to-end response (p95) | < 10s |
| Agent | Tool call success rate | > 99% |

### Circuit Breakers
**Model Service:**
- CLOSED → OPEN: 5 failures in 30 seconds
- OPEN → HALF-OPEN: after 60 seconds
- Fallback: "Unable to generate risk prediction — clinician review required"

**LLM (Bedrock):**
- Timeout: 30s max
- Retry: exponential backoff, 3 attempts
- Fallback: return RAG results without agent reasoning

### Graceful Degradation Hierarchy
1. **Full:** Agent + Model + RAG
2. **Degraded:** RAG results only (no prediction)
3. **Minimal:** "Service unavailable — please consult clinician directly"
4. **CRITICAL:** Never silently fail on patient-facing predictions

### Health Check Endpoints
- `GET /health` — basic liveness (is the service running)
- `GET /ready` — readiness (model loaded, can serve predictions)

---

## 9. Clinical Safety

### Confidence Thresholds
- Prediction confidence < 70% → "Low confidence — clinician review required"
- Always show risk score with confidence interval
- Flag out-of-distribution inputs

### Audit Trail
- Every prediction: patient_id, input_features, model_version, prediction, confidence, timestamp
- Every agent decision: tools_called, reasoning_chain, sources_cited
- Immutable append-only log

### Safety Guardrails
- Agent RECOMMENDS, never DECIDES
- All outputs labeled: "AI-assisted recommendation — clinical judgment required"
- System prompt prevents diagnosis or medication prescription
- All recommendations must cite clinical guidelines
- Disclaimer on every response

---

## 10. CI/CD Pipeline (GitHub Actions)

### Stage 1: Quality Gates
- Lint (ruff)
- Type check (mypy)
- Unit tests (pytest)
- Security scan (bandit)

### Stage 2: ML Validation
- Data validation tests
- Train model (deterministic seed)
- Model quality gate (AUC > 0.75)
- Behavioral tests

### Stage 3: Integration
- Build Docker image
- Container health check
- Integration tests (API contracts)
- RAG retrieval quality tests
- Agent behavior tests

### Stage 4: Performance
- Inference latency benchmark
- Throughput test
- Compare against baseline metrics
- Fail if regression > 10%

### Stage 5: Deploy
- Push image to registry
- Update model registry (candidate)
- Smoke test
- Promote to active (or rollback)

---

## 11. Project Structure

```
clinical-ai-agent/
├── README.md
├── pyproject.toml
├── Makefile
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docs/
│   ├── plans/
│   │   └── 2026-03-20-clinical-ai-agent-design.md
│   ├── aws-deployment.md
│   └── runbook.md
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py              # PDF/markdown loading, chunking, embedding
│   ├── retriever.py           # Query interface
│   └── data/
│       ├── guidelines/        # ACC/AHA PDF
│       └── snippets/          # Curated markdown files
│
├── model/
│   ├── __init__.py
│   ├── train.py               # XGBoost + PyTorch training
│   ├── predict.py             # FastAPI inference endpoint
│   ├── evaluate.py            # Metrics, comparison report
│   ├── features.py            # Feature extraction & validation
│   ├── Dockerfile
│   └── data/
│       └── heart_failure.csv
│
├── agent/
│   ├── __init__.py
│   ├── agent.py               # LangChain agent definition
│   ├── tools.py               # Tool definitions (RAG, predict, recommend)
│   ├── prompts.py             # System prompts with safety guardrails
│   └── safety.py              # Confidence thresholds, disclaimers
│
├── mlops/
│   ├── __init__.py
│   ├── registry.py            # Model versioning & champion/challenger
│   ├── monitor.py             # Inference logging & metrics
│   ├── drift.py               # Feature drift detection (PSI)
│   └── slo.py                 # SLO definitions & tracking
│
├── sre/
│   ├── __init__.py
│   ├── circuit_breaker.py     # Circuit breaker implementation
│   ├── health.py              # Health check endpoints
│   └── resilience.py          # Graceful degradation logic
│
├── perf/
│   ├── __init__.py
│   ├── benchmark.py           # Inference latency benchmarks
│   ├── load_test.py           # Concurrent request load testing
│   └── baseline.json          # Performance baseline for regression
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_rag_chunking.py
│   │   ├── test_registry.py
│   │   ├── test_circuit_breaker.py
│   │   └── test_safety.py
│   ├── ml/
│   │   ├── test_data_validation.py
│   │   ├── test_model_quality.py
│   │   └── test_behavioral.py
│   ├── integration/
│   │   ├── test_agent_rag.py
│   │   ├── test_agent_model.py
│   │   └── test_model_api.py
│   └── e2e/
│       └── test_full_pipeline.py
│
└── cli.py                      # CLI entry point
```

---

## 12. Build Schedule

| Day | Date | Task | Hours |
|---|---|---|---|
| 1 | Fri Mar 20 | Repo setup, project structure, README skeleton, CI scaffold, design doc | 2h |
| 2 | Sat Mar 21 | RAG pipeline + RAG tests (unit + retrieval quality) | 3h |
| 3 | Sun Mar 22 | Train XGBoost + PyTorch, evaluation, behavioral tests, containerize | 3h |
| 4 | Mon Mar 24 | LangChain agent, circuit breaker, graceful degradation, agent tests | 90m |
| 5 | Tue Mar 25 | MLOps layer, performance benchmarks, SLO config, runbook | 90m |
| 6 | Wed Mar 26 | Polish README, record demo, AWS deployment doc, final review | 60m |
| 7 | Thu Mar 27 | Breathing exercise → call at 9:30am PST | — |

---

## 13. Deployment Strategy (Document Only)

Full AWS deployment architecture documented in `docs/aws-deployment.md`:
- SageMaker endpoints for model serving
- ECR for container registry
- ECS/Fargate for model service
- S3 for model artifacts and training data
- CloudWatch for observability
- Step Functions for ML pipeline orchestration
- VPC with private subnets (HIPAA consideration)
- KMS encryption at rest, TLS in transit
- Cost estimation for running at scale

---

## 14. Decisions Log

| Decision | Choice | Rationale |
|---|---|---|
| LLM Provider | AWS Bedrock (Claude) | Consistent with AgenticOps, deep AWS expertise |
| ML Models | XGBoost + PyTorch | XGBoost for tabular performance, PyTorch for JD coverage |
| Embeddings | HuggingFace sentence-transformers | JD requirement, quality embeddings |
| Vector Store | ChromaDB | Simple, no infra overhead for local demo |
| MLOps | Custom lightweight | Learn fundamentals, more impressive than pip install mlflow |
| Docker scope | Model service only | Realistic deployment unit, rest runs locally |
| Deployment | Local only | AWS architecture documented, not built |
| Testing | TDD, full pyramid | Differentiator — most ML projects have zero tests |
| Dev Methodology | TDD + Agentic Development | Engineer architects/reviews, AI agents execute under TDD discipline. Demonstrates how production AI teams will operate |
