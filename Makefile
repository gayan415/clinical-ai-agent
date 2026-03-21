.PHONY: install dev lint lint-fix type-check security test test-unit test-ml test-integration test-e2e test-perf test-all ci clean docker-build docker-run ingest train run

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
