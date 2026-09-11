.DEFAULT_GOAL := help
SHELL := /bin/bash

# Use the project venv if present, otherwise whatever python is on PATH.
VENV := .venv
ifeq ($(OS),Windows_NT)
	PY := $(VENV)/Scripts/python.exe
else
	PY := $(VENV)/bin/python
endif
ifeq ($(wildcard $(PY)),)
	PY := python
endif

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: venv
venv: ## Create the virtual environment
	python -m venv $(VENV)

.PHONY: install
install: ## Install the project with dev dependencies
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

.PHONY: install-postgres
install-postgres: ## Add the Postgres driver (needed for migrations)
	$(PY) -m pip install -e ".[dev,postgres]"

.PHONY: env
env: ## Create .env from the example if absent
	@test -f .env || (cp .env.example .env && echo "created .env - add CCE_GROQ_API_KEY and the CCE_MODEL_* identifiers")

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
.PHONY: lint
lint: ## Lint
	$(PY) -m ruff check .

.PHONY: format
format: ## Format
	$(PY) -m ruff format .

.PHONY: typecheck
typecheck: ## Type-check (strict)
	$(PY) -m mypy app tests

.PHONY: test
test: ## Run the full suite (no credentials, no network)
	$(PY) -m pytest

.PHONY: test-unit
test-unit: ## Unit tests only
	$(PY) -m pytest tests/unit -q

.PHONY: test-contract
test-contract: ## Architectural and specification contract tests
	$(PY) -m pytest tests/contract -q

.PHONY: spec-check
spec-check: ## Assert the specification agrees with docs/DOMAIN_CHECKPOINT.md
	$(PY) -m scripts.check_spec_consistency

.PHONY: ingest-corpus
ingest-corpus: ## Ingest data/sample_corpus and print what the deterministic layer produced
	$(PY) -m scripts.ingest_corpus --twice

.PHONY: coupling-report
coupling-report: ## Build the coupling index and print the knowledge-decay report
	$(PY) -m scripts.coupling_report

.PHONY: coupling-delta
coupling-delta: ## Propagate the worked example change through the coupling index
	$(PY) -m scripts.coupling_report --delta 'retention.window=90d->180d'

.PHONY: retrieval-benchmark
retrieval-benchmark: ## Score the lexical retriever against the labelled query suite
	$(PY) -m scripts.retrieval_benchmark

.PHONY: retrieval-benchmark-detail
retrieval-benchmark-detail: ## Same, with each query's trace and expectations
	$(PY) -m scripts.retrieval_benchmark --per-query

.PHONY: retrieval-benchmark-vector
retrieval-benchmark-vector: ## Score the vector retriever against the same labelled queries
	$(PY) -m scripts.retrieval_benchmark --strategy vector

.PHONY: retrieval-benchmark-hybrid
retrieval-benchmark-hybrid: ## Score reciprocal rank fusion over both strategies
	$(PY) -m scripts.retrieval_benchmark --strategy hybrid

.PHONY: embeddings-check
embeddings-check: ## Report whether configuration is ready for the one live embedding call
	$(PY) -m scripts.record_embeddings --check

.PHONY: record-embeddings
record-embeddings: ## Make the one live embedding call and commit the vectors (needs a key)
	$(PY) -m scripts.record_embeddings

.PHONY: check
check: lint typecheck spec-check test ## Everything CI runs

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
.PHONY: run
run: ## Start the API with reload
	$(PY) -m uvicorn app.main:app --reload --port 8000

.PHONY: agents
agents: ## Print agent -> tier -> provider/model resolution without calling a model
	$(PY) -m scripts.show_agents

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
.PHONY: up
up: ## Start Postgres + Redis
	docker compose up -d

.PHONY: down
down: ## Stop infrastructure
	docker compose down

.PHONY: reset-db
reset-db: ## Destroy and recreate the database volume
	docker compose down -v
	docker compose up -d postgres

.PHONY: migrate
migrate: ## Apply migrations
	$(PY) -m alembic upgrade head

.PHONY: migration
migration: ## Create a migration: make migration m="add sources"
	$(PY) -m alembic revision --autogenerate -m "$(m)"

.PHONY: clean
clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
