# Repository Folder Structure

```text
change-consequence-engine/
│
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Makefile
│
├── .claude/
│   └── agents/
│       ├── change-decomposer.md
│       ├── policy-analyst.md
│       ├── process-analyst.md
│       ├── technology-analyst.md
│       ├── people-analyst.md
│       ├── data-analyst.md
│       ├── risk-analyst.md
│       ├── historical-analyst.md
│       ├── consequence-hunter.md
│       ├── contradiction-hunter.md
│       ├── adversarial-challenger.md
│       ├── blind-spot-hunter.md
│       ├── evidence-arbiter.md
│       ├── consequence-judge.md
│       └── report-composer.md
│
├── docs/
│   ├── IDEA.md
│   ├── ARCHITECTURE.md
│   ├── FOLDER_STRUCTURE.md
│   ├── DEVELOPMENT_PLAN.md
│   ├── EVALUATION.md
│   ├── DATA_MODEL.md
│   ├── SECURITY.md
│   └── STATUS.md
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── deps.py
│   │   ├── routes/
│   │   │   ├── analyses.py
│   │   │   ├── changes.py
│   │   │   ├── sources.py
│   │   │   └── health.py
│   │   └── schemas/
│   │       ├── analysis.py
│   │       ├── change.py
│   │       └── source.py
│   │
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── change.py
│   │   │   ├── evidence.py
│   │   │   ├── consequence.py
│   │   │   ├── graph.py
│   │   │   └── analysis.py
│   │   ├── value_objects/
│   │   │   ├── confidence.py
│   │   │   ├── severity.py
│   │   │   └── identifiers.py
│   │   ├── interfaces/
│   │   │   ├── llm.py
│   │   │   ├── embeddings.py
│   │   │   ├── retrieval.py
│   │   │   └── repositories.py
│   │   └── services/
│   │       ├── scope_lattice.py          # equality, containment, intersection, distance
│   │       ├── value_comparison.py       # unit-aware; a mismatch BLOCKS coupling
│   │       ├── temporal_intervals.py     # half-open [from, to) arithmetic
│   │       ├── coupling_strength.py      # versioned; cannot see an agent
│   │       ├── span_verification.py
│   │       ├── typed_parse.py
│   │       ├── consequence_scoring.py
│   │       ├── evidence_validation.py
│   │       └── graph_builder.py
│   │
│   ├── application/
│   │   ├── use_cases/
│   │   │   ├── analyze_change.py
│   │   │   ├── ingest_source.py
│   │   │   ├── resolve_coupling.py       # the only store-touching step in Class A
│   │   │   └── evaluate_analysis.py
│   │   ├── orchestration/
│   │   │   ├── analysis_orchestrator.py
│   │   │   ├── agent_router.py
│   │   │   ├── phase_runner.py
│   │   │   └── policies.py
│   │   └── contracts/
│   │       ├── agent_contracts.py
│   │       └── pipeline_contracts.py
│   │
│   ├── agents/
│   │   └── runtime/
│   │       ├── agent_loader.py
│   │       ├── agent_executor.py
│   │       ├── prompt_registry.py
│   │       └── output_parser.py
│   │
│   ├── assumptions/
│   │   ├── extraction.py             # LLM, fast tier, per source version
│   │   ├── prompting.py              # untrusted-content envelope
│   │   ├── proposals.py              # untrusted extractor output schema
│   │   ├── admission.py              # the deterministic admission gate
│   │   ├── registry.py               # closed key registry: resolve, quarantine, aliases
│   │   ├── temporal.py               # resolve_authority(key, scope, as_of) | Ambiguous
│   │   ├── collisions.py             # the role matrix; three-way divergence rule
│   │   ├── coupling.py               # coupling index keyed on (key, scope, as_of)
│   │   ├── invalidation.py           # AssumptionDelta -> invalidated / scope-affected
│   │   ├── class_a.py                # Class A generation; Class B merge and rediscovery
│   │   └── decay.py                  # the standing knowledge-decay report
│   │
│   ├── retrieval/
│   │   ├── lexical.py
│   │   ├── vector.py
│   │   ├── metadata.py
│   │   ├── temporal.py
│   │   ├── graph.py
│   │   ├── reranker.py
│   │   └── fusion.py
│   │
│   ├── ingestion/
│   │   ├── pipeline.py
│   │   ├── parsers/
│   │   │   ├── markdown.py
│   │   │   ├── pdf.py
│   │   │   ├── html.py
│   │   │   └── json.py
│   │   ├── chunking.py
│   │   ├── metadata.py
│   │   └── connectors/
│   │       ├── filesystem.py
│   │       └── github.py
│   │
│   ├── infrastructure/
│   │   ├── llm/
│   │   │   ├── gateway.py            # tier resolution, budgets, retries, accounting
│   │   │   ├── registry.py           # provider registration
│   │   │   └── providers/
│   │   │       ├── groq.py           # default remote provider
│   │   │       ├── gemini.py         # optional
│   │   │       ├── mistral.py        # optional
│   │   │       ├── ollama.py         # optional, local
│   │   │       └── fake.py           # deterministic, for tests
│   │   ├── embeddings/
│   │   ├── database/
│   │   │   ├── session.py
│   │   │   ├── models/
│   │   │   └── repositories/
│   │   ├── storage/
│   │   ├── queue/
│   │   ├── observability/
│   │   └── security/
│   │
│   ├── evaluation/
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   ├── baselines/
│   │   └── datasets.py
│   │
│   └── config/
│       ├── settings.py
│       └── agents.yaml               # agent -> tier / provider / model binding
│
├── data/
│   ├── sample_corpus/
│   ├── benchmarks/
│   └── fixtures/
│
├── migrations/
│
├── scripts/
│   ├── seed_demo_data.py
│   ├── build_embeddings.py
│   └── run_benchmark.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   ├── evaluation/
│   └── fixtures/
│
└── frontend/
    └── ...
```

## Important convention

`.claude/agents/*.md` are the actual Claude Code project subagent definitions.

They are not runtime Python implementations.

The Python runtime loads the agent definitions and invokes model/tool execution according to the orchestration layer.

Claude Code officially supports project subagents in `.claude/agents/` as Markdown files with YAML frontmatter.

### Separation of prompt from runtime binding

The Markdown file owns behavior:
role, responsibilities, reasoning constraints, evidence requirements, expected behavior,
structured output requirements.

`app/config/agents.yaml` owns runtime binding:
tier, provider, model, token budget, temperature, retry policy.

These are separated because prompts are reviewed content and provider bindings are
deployment configuration that changes per experiment. A `model:` value in the Markdown
frontmatter is honored by the Claude Code interactive harness only; the production runtime
resolves models through the gateway and ignores it.

Prompt versioning is carried by a `version:` field in the frontmatter plus a content hash
computed by the loader, so a finding can be attributed to the exact prompt text that
produced it without moving prompts out of `.claude/agents/`.
