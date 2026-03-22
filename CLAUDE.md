# CLAUDE.md — Clinical AI Agent for Heart Failure Risk Assessment

## Project Overview
Portfolio project demonstrating production-grade clinical AI engineering. LangChain agent on AWS Bedrock (Claude) with RAG retrieval, ML risk prediction, and treatment recommendations — wrapped with full SRE practices, MLOps, and clinical safety.

## Development Methodology: TDD + Agentic Coding (NOT Vibe Coding)
This project is built using **Agentic Coding** — AI agents (Claude Code) as development partners, directed by the engineer. The engineer:
1. **Architects** — makes all design decisions, validates approaches, approves before any code
2. **Directs** — breaks work into tasks, sets constraints, reviews output
3. **Reviews** — every line of code is reviewed before commit, tests must pass
4. **Owns** — understands every component, can explain and defend every decision

### Task Flow (MANDATORY for every task)
1. **Teach** — explain the new concept before writing any code
2. **Test** — write the failing test, explain what it's testing and why
3. **Build** — implement the code, explain key decisions
4. **Verify** — run tests, engineer confirms understanding
5. **Commit** — feature branch + PR, engineer reviews and merges

### What this is NOT
- NOT vibe coding — no generating walls of code the engineer can't explain
- NOT auto-pilot — agent stops and teaches when introducing new concepts
- NOT "trust the AI" — every line must be understood before merge

Combined with **Test-Driven Development**:
- Write failing test first → implement → verify → commit
- Full test pyramid: unit, ML behavioral, integration, e2e, performance
- Tests are the specification, not an afterthought

This is how production AI teams will work — engineers directing AI agents with engineering rigor.

## Project Root
`/Users/gjayasun/git/AI/clinical-ai-agent`

## Related Directory
Interview prep lives at `/Users/gjayasun/git/AI/Abbott` — do not modify files there from this project.

## Design & Plan
- Design doc: `docs/plans/2026-03-20-clinical-ai-agent-design.md`
- Implementation plan: `docs/plans/2026-03-20-implementation-plan.md`
- 30 tasks across 6 days (Mar 20-26), interview Mar 27

## Tech Stack
- Python 3.11+, LangChain, AWS Bedrock (Claude), HuggingFace sentence-transformers
- ChromaDB, XGBoost, PyTorch, FastAPI, Docker (Colima — no Docker Desktop)
- pytest, ruff, mypy, bandit, GitHub Actions

## Key Decisions
- LLM: AWS Bedrock (Claude) — matches AgenticOps narrative
- Models: Train BOTH XGBoost + PyTorch, champion/challenger pattern
- Embeddings: HuggingFace all-MiniLM-L6-v2
- Docker: Model service only (FastAPI prediction endpoint)
- Deployment: Local only — AWS architecture documented in `docs/aws-deployment.md`
- MLOps: Custom lightweight (no MLflow) — file-based registry, inference logging, PSI drift detection
- Testing: TDD, full pyramid (unit, ML behavioral, integration, e2e, performance)

## Build Schedule
| Day | Date | Focus |
|---|---|---|
| 1 | Fri Mar 20 | Repo setup, structure, CI, README (Tasks 1-6) |
| 2 | Sat Mar 21 | RAG pipeline + tests (Tasks 7-11) |
| 3 | Sun Mar 22 | ML models + behavioral tests + Docker (Tasks 12-17) |
| 4 | Mon Mar 24 | Agent + safety + circuit breaker (Tasks 18-21) |
| 5 | Tue Mar 25 | MLOps + perf benchmarks + runbook (Tasks 22-27) |
| 6 | Wed Mar 26 | Polish + demo + AWS doc (Tasks 28-30) |

## Git Workflow Rules
- **NEVER push directly to main** — always create a feature branch
- **Every task = branch + PR** — engineer reviews and merges manually
- **Pre-commit hooks run automatically** — ruff (lint), bandit (security), file hygiene
- Branch naming: `feat/`, `fix/`, `chore/`, `test/`, `docs/` prefixes

## Current Progress
- [x] Design doc written and approved
- [x] Implementation plan written (30 tasks)
- [x] Day 1: Tasks 1-6 COMPLETE (repo, structure, CI, README, deps, GitHub push)
- [x] Pre-commit hooks configured (ruff, bandit, file hygiene)
- [x] CI fixes merged (missing cli.py, empty tests, missing Dockerfile, missing benchmark)
- [x] Day 2: Tasks 7-11 COMPLETE (RAG pipeline — 18 tests passing)
- [ ] Day 3: Tasks 12-17 (ML models) — IN PROGRESS
  - [x] Task 12: UCI Heart Failure dataset downloaded (299 patients, 13 columns)
  - [x] Task 13: Data validation + features.py — 11/11 tests passing
  - [x] Task 14: XGBoost + PyTorch training + quality gates — 3/3 tests passing
    - XGBoost: AUC > 0.75, class imbalance handled with scale_pos_weight
    - PyTorch: AUC > 0.70, feedforward net (12→64→32→1)
  - [ ] Task 15: Behavioral tests (directional, invariance, edge cases)
  - [ ] Task 16: FastAPI prediction endpoint + tests
  - [ ] Task 17: Dockerfile for model service

## Environment Variables
```bash
# Embedding provider (default: huggingface, runs locally)
EMBEDDING_PROVIDER=huggingface   # or "bedrock" for AWS Titan Embed
AWS_REGION=us-east-1             # only needed if EMBEDDING_PROVIDER=bedrock
AWS_PROFILE=sbg-bedrock          # only needed if EMBEDDING_PROVIDER=bedrock
```

## Commands
```bash
make dev          # Install with dev deps
make lint         # Ruff check + format
make type-check   # Mypy
make test         # Unit + ML tests
make test-all     # All tests
make ci           # Full CI pipeline
make ingest       # RAG document ingestion
make train        # Train ML models
make docker-build # Build model container
make run          # Run agent CLI
```

## Code Style
- Use ruff for linting/formatting (line-length=100)
- Use mypy for type checking (disallow_untyped_defs)
- Use structlog for all logging (structured JSON)
- TDD: write failing test first, then implement
- Commit after each task completes

## Clinical Safety Rules
- Agent RECOMMENDS, never DECIDES
- Every response includes clinical disclaimer
- Confidence < 70% = "clinician review required"
- All predictions logged to immutable audit trail
- System prompt prevents diagnosis or prescribing

## Who This Is For
Built by Gayan Jayasundara as a portfolio piece for Abbott Acelis Principal SDE (AI/ML) interview with Shekhar Venkat (ex-Amazon Bar Raiser). Must demonstrate production engineering rigor, not tutorial-grade code.
