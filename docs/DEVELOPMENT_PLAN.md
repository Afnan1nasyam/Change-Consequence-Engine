# Development Plan for Claude Code

## Principle

Develop in vertical slices. Do not ask Claude Code to build the whole platform in one pass.

Guidance followed here: incremental progress, explicit state tracking, testing at each
step, and specialized subagents for focused work.

Two ordering rules override convenience:

1. **Schemas before prompts.** A prompt told to "match the runtime schema" against a
   schema that does not exist produces invented fields and fabricated citations.
2. **Deterministic before agentic.** The core mechanism is the assumption store, coupling
   index, and invalidation resolver. Agents explain and challenge its output. Building the
   agent layer first produces a system whose recall depends on context length.

## Phase 0 — Repository bootstrap

Claude Code tasks:
- initialize Python project
- create folder structure
- configure linting/type checking/testing
- LLM provider abstraction (`ProviderPort`) in the domain layer
- LLM gateway with tier resolution, token budgets, retries, usage accounting
- Groq adapter (default remote provider, env-based credentials)
- deterministic fake adapter for tests (no credentials, no network)
- optional adapter extension points for Gemini / Mistral / Ollama
- create Docker Compose for Postgres + pgvector + Redis
- add `.env.example`
- add initial CI
- create STATUS.md

Acceptance:
- project installs
- tests execute with no provider credentials present
- FastAPI health endpoint works
- database migration configuration is valid
- gateway resolves agent -> tier -> provider/model from configuration only
- no provider SDK is imported by the domain or application layer (enforced by a test)

## Phase 1 — Domain model

Implement:
- ChangeRequest
- ChangeSpec
- AssumptionRecord
- AssumptionDelta
- CouplingSet
- EvidenceItem
- SpecialistFinding
- ConsequenceHypothesis
- ChallengeResult
- ArbitrationDecision
- FindingDisposition
- GraphNode
- GraphEdge

Acceptance:
- Pydantic models
- validation tests
- serialization tests
- an AssumptionRecord without an evidence_id fails validation
- a GraphEdge with a relationship outside the MVP taxonomy fails validation
- these schemas are frozen before any agent prompt is written against them

## Phase 2 — Corpus ingestion and assumption extraction

Start with Markdown, plain text, and JSON.

Implement:
- source registration
- versioning
- chunking
- metadata
- embeddings
- persistence
- assumption extraction (fast tier, once per source version, cached)
- registry resolution and quarantine — a key normalizer is deliberately excluded, because
  canonical form is declared per registry entry, never learned (`DOMAIN_CHECKPOINT.md` 4.1.5)
- coupling index build
- collision detection
- temporal authority resolution

Acceptance — all met:
- synthetic corpus ingests
- evidence IDs remain stable
- source versions are queryable
- re-ingesting an unchanged source performs zero extraction calls
- coupling sets are reproducible from the store alone
- a seeded threshold contradiction is detected by key comparison, with no LLM call
- overlapping effective ranges resolve to `Ambiguous` rather than newest-wins

Delivered in two slices: the **ingestion slice** (parse → version → chunk → extract →
admit) and the **coupling slice** (index → authority → collisions → Class A →
knowledge-decay report). The split is why the last three criteria above were outstanding
for a while; they belong to the second.

The coupling slice also delivered, beyond the stated acceptance criteria:
- the three-way divergence rule with the authority comparison recorded on every derivation
- role-aware collision detection per the checkpoint 3.3 matrix, including the cases that
  must emit **nothing**
- invalidation propagation with `scope_inheritance_affected` kept distinct from
  `invalidates_assumption`
- Class A generation by producer, with evidence materialized as a marked derived union
- the Class B merge gate and `class_b_rediscovery_rate`
- the versioned coupling-strength function

Checkpoint: this phase alone produces the knowledge-decay report (`IDEA.md` 15.5). If that
output is not useful on a realistic corpus, stop and fix the corpus before proceeding.
**Passed** — see the metric table in `docs/EVALUATION.md`.

## Phase 3 — Retrieval

Implement:
- lexical
- vector
- metadata
- hybrid fusion
- reranking abstraction

Acceptance:
- retrieval benchmark
- relevant evidence appears for known scenarios
- permissions can filter evidence

## Phase 3.5 — Deterministic consequence path

Most of this arrived early, with the coupling slice of Phase 2: the invalidation resolver
and coupling propagation are prerequisites of the knowledge-decay report, and building them
twice would have been waste.

Implemented:
- invalidation resolver (`AssumptionDelta` → invalidated keys), including the
  scope-inheritance and pre-existing-defect distinctions
- coupling propagation → candidate affected artifacts
- the versioned coupling-**strength** function

Still outstanding:
- the deterministic **confidence** scorer. `IDEA.md` 13 names the factors and directions
  but no weights, and `ConsequenceScore` persists `factor_inputs` plus
  `confidence_fn_version` so a score stays recomputable. Weights stay untuned until the
  feedback loop supplies data, so there is nothing honest to compute yet.
- conversion of `ClassAFinding` → `Consequence`, which needs the judge to supply the
  `severity` the resolver must not invent (see `docs/STATUS.md`).

Acceptance:
- Class A consequences are produced with zero reasoning-tier calls — **met**, zero calls of
  any tier
- every Class A consequence carries evidence IDs by construction — **met**, as a marked
  derived union
- confidence scores are recomputable from persisted inputs — **outstanding**, no scorer yet
- removing a supporting document changes the result (ablation works) — **met**

## Phase 4 — First agent vertical slice

Implement:
- change-decomposer (emits AssumptionDelta)
- policy-analyst
- process-analyst
- technology-analyst
- consequence-hunter (Class B only)
- evidence-arbiter
- report-composer

Each specialist issues its own QueryPlan. Do not fan one shared EvidenceBundle out to all
specialists; see `ARCHITECTURE.md` 4a.

Acceptance:
- one change can be analyzed end-to-end
- every final consequence references evidence IDs
- unsupported claim rate is measured
- specialist evidence slices are measurably distinct
- agents run against frozen Phase 1 schemas with validation failures recorded, not coerced

## Phase 5 — Multi-agent differentiation

Add:
- data-analyst
- people-analyst
- risk-analyst
- historical-analyst
- contradiction-hunter

Acceptance:
- specialist findings can disagree
- contradictions are visible
- routing can skip irrelevant agents

## Phase 6 — Adversarial pipeline

Add:
- adversarial-challenger (citation-obligated)
- blind-spot-hunter (over a disjoint evidence partition)
- consequence-judge

Acceptance:
- full pipeline produces challenge outcomes
- a challenge without a contradicting evidence_id is rejected by the schema, not by
  prompt wording
- `no_refutation_found` is a first-class, recorded outcome
- judge can reject hypotheses
- the blind-spot partition is verifiably disjoint from specialist evidence slices
- blind-spot findings are measured separately

## Phase 7 — Consequence graph

Implement graph nodes/edges and provenance.

Acceptance:
- every inference edge traceable to findings/evidence
- graph JSON returned by API
- graph validation tests

## Phase 8 — Evaluation

Implement benchmark suite and baselines.

Required comparisons:
- retrieval + prompt
- single-agent RAG
- self-critique
- multi-agent
- full CCE

Also required:
- mandatory negative controls (~25% of scenarios)
- corpus-ablation variants
- Class A / Class B metric separation
- coupling recall
- provider-diversity conditions F1-F5

Acceptance:
- reproducible evaluation command
- metrics stored with confidence intervals
- comparison report generated
- unseeded-but-correct findings are not counted as false positives

## Phase 9 — API and UI

Add:
- change submission
- analysis status
- result retrieval
- graph visualization
- evidence drill-down
- unresolved questions
- per-finding disposition capture (`IDEA.md` 21.2)
- persistent suppression of findings dispositioned `wrong` / `not_material`
- knowledge-decay report view

Alert-load constraints are product requirements, not polish:
- hard cap on findings surfaced by default
- default view limited to `supported` / `likely` at HIGH / CRITICAL
- everything else behind an explicit disclosure
- findings routed to an owner; an unowned finding is not actionable

## Phase 10 — Production hardening

Add:
- auth
- tenant isolation
- audit logs
- tracing
- rate limits
- retries
- cost controls
- security tests
- prompt/version registry
- deployment manifests

## Claude Code working rules

At the start of each phase:
1. read CLAUDE.md
2. read relevant architecture/spec documents
3. inspect repository state
4. implement the smallest complete slice
5. write tests
6. run tests and checks
7. update STATUS.md
8. do not start unrelated refactors

At the end of each phase, create a git checkpoint.
