# Change Consequence Engine — Claude Code Project Instructions

## Project identity

Project name: Change Consequence Engine (CCE)

CCE is a production-oriented multi-agent system that analyzes a proposed organizational/business/technical change and discovers both explicit and latent consequences across heterogeneous organizational knowledge.

Core question:

> "If we change X, what else may silently become wrong, inconsistent, risky, or operationally impossible?"

This is NOT a generic chatbot, NOT a simple document Q&A system, and NOT a conventional keyword-based impact analyzer.

## Core mechanism

The mechanism is **assumption-coupling analysis** (`docs/IDEA.md` 15).

A latent consequence exists because two or more artifacts independently bind the same
unstated proposition without referencing each other. The unit of analysis is therefore the
**assumption**, not the document and not the entity.

- assumptions are extracted once per source version at ingestion and cached
- the coupling set of an assumption key is every artifact binding it, derived from
  co-binding rather than cross-reference
- a change is expressed as a delta over assumptions
- invalidation and propagation are **deterministic** and contain no model call
- divergence between declarers resolves deterministically three ways by scope and
  `authority_class`: contradiction, legitimate override, or unauthorized override
- falsification carries a citation obligation: a contradicting evidence ID, or the explicit
  outcome `no_refutation_found`

Class A consequences (assumption-mediated) are discovered deterministically and are
evidence-backed by construction. Class B consequences (behavioral/quantitative) are LLM
hypotheses and require a historical analogue with an observed outcome, or the status
`speculative`.

The agent layer explains, rates, and challenges the deterministic output. It is not the
thing that finds Class A consequences. Any design in which recall depends on how much
corpus fits in a context window is the wrong design.

## Non-negotiable product principles

1. Evidence first. Every material claim must be traceable to source evidence.
2. Separate facts, inferred relationships, hypotheses, and judgments.
3. Agents must have distinct responsibilities and should be allowed to disagree.
4. Do not make every component an LLM call. Use deterministic code wherever deterministic logic is sufficient.
5. Preserve provenance at chunk, document, repository object, ticket, and graph-edge level.
6. Never silently invent dependencies. A **model-inferred** relationship must state its mechanism and is a hypothesis until validated. A **deterministically derived** relationship carries its `rule_id` and input record ids instead of evidence of its own — its evidence set is the computed union of its inputs, and demanding a direct citation there would manufacture false provenance.
7. The system must be useful even when the corpus is incomplete; uncertainty is a first-class output.
8. The system must identify blind spots and unresolved questions, not only produce a polished answer.
9. Do not optimize for a flashy demo at the expense of auditability, reproducibility, cost controls, or correctness.
10. Prefer small, explicit interfaces over hidden state and implicit agent coordination.
11. Avoid framework lock-in where practical. Business logic should not depend deeply on a single agent framework.
12. All agent prompts live in `.claude/agents/*.md`. Do not create Python agent classes merely to hold prompts.
13. Agent `.md` files describe behavior and contracts. Runtime orchestration is implemented in Python. Runtime binding (tier, provider, model, budget) lives in `app/config/agents.yaml`, not in the prompt.
14. Never expose hidden chain-of-thought. Persist concise rationales, evidence references, decisions, and structured findings only.
15. Confidence is computed by deterministic, versioned code over evidence features. A model must never be asked to emit a confidence number.
16. Corroboration counts independent sources, never independent agents. Several agents reading the same evidence are one source.
17. Analysis diversity comes from differing inputs, not differing personas.
18. Schemas are authored and frozen before the prompts that must satisfy them.
19. No provider SDK may be imported above the infrastructure layer.
20. **An LLM may propose a semantic relationship. It can never promote that relationship into the trusted deterministic layer by itself.** Unknown keys go to `KeyRegistryGap`; possible key equivalences go to a `KeyAlias` proposal. Both require human review and a versioned update before any Class A derivation can use them. There is no confidence threshold that bypasses either gate.
21. The assumption key registry is **closed** for the MVP. Only registered keys participate in Class A coupling; unregistered keys are quarantined, never indexed.
22. Class A means **produced by the deterministic resolver** — classification is by producer, not by subject matter. `consequence_class` is pipeline-assigned and appears on no agent-authored schema.
23. The consequence graph is a **pure projection**, never a second source of truth. No agent writes an edge.

## Technology direction

Default implementation target:
- Python 3.12+
- FastAPI
- Pydantic v2
- PostgreSQL
- pgvector
- SQLAlchemy
- Alembic
- Redis for queues/cache where justified
- object storage for raw artifacts
- OpenTelemetry-compatible tracing
- pytest
- Docker / docker compose for local development

Use LangGraph only if it provides a concrete orchestration benefit. The core domain model and orchestration contracts must remain framework-independent.

Use an embedding provider behind an interface. Do not hard-wire business logic to one embedding vendor.

### LLM providers

CCE is provider-agnostic. **No single vendor is a required dependency.**

```text
Client / API -> Analysis Orchestrator -> Agent Runtime -> LLM Gateway / ProviderPort
                                                              |
                                    groq | gemini | mistral | ollama | fake
                                                              |
                                                    model from configuration
```

- **Groq** is the default remote provider and the only one required to run the project.
- **Gemini**, **Mistral**, and **Ollama** are optional. They must not be needed for
  development, tests, or CI, and their absence must never be an error.
- A deterministic **fake** provider serves tests. The suite runs green with no
  credentials and no network.
- Agents request a capability **tier** (`reasoning` | `standard` | `fast`). They never
  name a model.
- Model identifiers come from environment/configuration only. Do not commit model names
  to source, and do not invent model names that have not been configured.
- Per-agent provider override must be a configuration change, because provider diversity
  is an open evaluation question (`docs/EVALUATION.md`), not a settled decision.

Forbidden imports in `app/domain/` and `app/application/`: any provider SDK
(groq, google-genai, mistralai, ollama), LangChain, LangGraph. This is enforced by a test,
not by convention.

### Cost discipline

- retrieval before reasoning; never reason over what could have been filtered
- deterministic code wherever deterministic logic suffices
- assumption extraction runs on the `fast` tier, never on `reasoning`
- token budgets per call and per analysis, enforced before dispatch
- structured outputs with bounded repair retries, then discard
- cache on (prompt version, schema version, provider, model, params, input hash)
- retries with exponential backoff and jitter
- per-call usage and cost accounting attributed to agent and analysis
- a cheap gate must be able to reject an analysis as not worth running

## Repository rules

Before modifying an unfamiliar area:
- read the relevant files
- inspect existing tests
- inspect interfaces and schemas
- preserve established patterns

Do not generate large speculative abstractions.

Do not add a dependency merely because it makes a small task convenient.

Every new production component must have:
- clear ownership/responsibility
- typed input/output
- tests
- logging/observability hooks
- failure behavior

## Agent rules

Agents should:
- operate on explicit task contracts
- consume structured context
- cite evidence IDs
- return structured JSON matching Pydantic schemas
- state uncertainty
- distinguish direct evidence from inference
- avoid repeating retrieval work performed by another agent unless independently validating it

Agents should NOT:
- directly mutate the production database
- perform irreversible external actions
- fabricate citations
- use unsupported source content
- make business-policy decisions on behalf of the customer

## Development approach

Build in vertical slices:
1. Domain model, evidence model, and assumption model
2. Ingestion, assumption extraction, coupling index, temporal authority
3. Deterministic invalidation and propagation (Class A consequences)
4. Retrieval
5. Single-agent consequence hypothesis generation (Class B)
6. Multi-agent parallel analysis over per-specialist evidence slices
7. Adversarial challenge and blind-spot discovery
8. Consequence graph
9. Evaluation harness
10. API and UI
11. Production hardening

Deterministic before agentic. The knowledge-decay report (`docs/IDEA.md` 15.5) is
deliverable after slice 2 and is the earliest honest proof that the corpus supports the
product at all.

At the end of each implementation slice:
- run focused tests
- run relevant static checks
- update documentation
- update `docs/STATUS.md`

## Current product boundary

MVP target corpus:
- policies
- SOPs/runbooks
- internal knowledge documents
- tickets/incidents
- change requests
- sample source-code/configuration metadata
- organizational ownership metadata

The MVP should simulate a realistic enterprise corpus using synthetic/open sample data. Do not depend on confidential company data.

## Quality bar

A good result is not "the model produced a convincing report."

A good result is:
- evidence-grounded
- traceable
- contradiction-aware
- uncertainty-aware
- repeatable enough to evaluate
- able to identify missed/hidden consequences
- demonstrably better than a single-agent baseline on defined benchmark cases

Two additional bars, because they are the ones most easily failed while everything above
looks satisfied:

- **Ablation-honest.** Remove a supporting document and the dependent finding must
  degrade to `unresolved`. A finding that survives removal of its evidence was coming from
  model priors.
- **Quiet on quiet changes.** On a negative-control change the system must report
  approximately nothing. A system that always finds consequences has not been shown to
  find any.
