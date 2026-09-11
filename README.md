# Change Consequence Engine (CCE)

Multi-agent evidence analysis for discovering hidden consequences of organizational and
technical changes.

> If we change X, what else silently becomes wrong, inconsistent, risky, or operationally
> impossible?

## The mechanism

CCE is not a RAG chatbot over policy documents. The core is **assumption-coupling
analysis**:

A latent consequence exists because two or more artifacts **independently bind the same
unstated proposition** without referencing each other. The refund policy, the support
training deck, and the reconciliation runbook all commit to
`refund.approval_threshold == 10000`. None cites another. All three are coupled.

So the unit of analysis is the **assumption**, not the document and not the entity.

| step | kind |
|---|---|
| extract typed `AssumptionRecord`s per source version | LLM, fast tier, cached at ingestion |
| build the coupling index (key -> holders) | **deterministic** |
| express the change as an assumption delta | structured |
| propagate invalidated keys to affected artifacts | **deterministic** |
| articulate why invalidation causes harm | LLM, focused |
| falsify, with a citation obligation | LLM |

Discovery of the latent edge is *computed*, not model-asserted. A divergent binding of the
same key is a contradiction found by comparison rather than inference, so it does not
degrade as the corpus grows. The agents explain, rate and challenge that output; they are
not what finds it.

See `docs/IDEA.md` section 15, including the honest scope limit: assumption-coupling owns
documentation and configuration consistency (Class A) and does *not* cover behavioural
or load effects (Class B), which stay LLM hypotheses requiring a historical analogue.

## Status

**Phase 0 complete — repository bootstrap.** No analysis pipeline yet. See
`docs/STATUS.md` for exactly what is and is not implemented, and
`docs/DEVELOPMENT_PLAN.md` for what comes next.

## Provider independence

No vendor LLM SDK is a dependency. Every provider is reached over HTTP behind a domain
port, and agents request a capability *tier* rather than naming a model.

```text
Client / API -> Analysis Orchestrator -> Agent Runtime -> LLM Gateway / ProviderPort
                                                              |
                                    groq | gemini | mistral | ollama | fake
                                                              |
                                                    model from configuration
```

- **Groq** is the default and the only provider required.
- **Gemini / Mistral / Ollama** are optional and never needed to run, develop, or test.
- **fake** is a deterministic in-process provider; the suite is green with no credentials
  and no network.
- Model identifiers come from the environment. None are built in — vendor inventory
  changes, and a stale hard-coded identifier is an outage.

Per-agent provider overrides are configuration, because whether provider diversity reduces
correlated reasoning error is an open evaluation question (`docs/EVALUATION.md`), not a
settled decision.

## Quick start

```bash
make venv && make install     # or: python -m venv .venv && pip install -e ".[dev]"
make env                      # copy .env.example -> .env
make check                    # lint + strict types + tests, no credentials needed
make run                      # http://127.0.0.1:8000/health/ready
```

To make real model calls, set in `.env`:

```bash
CCE_GROQ_API_KEY=...
CCE_MODEL_REASONING=...   # judge, challenger, consequence hunter, blind-spot hunter
CCE_MODEL_STANDARD=...    # specialist analysts, contradiction hunter, report composer
CCE_MODEL_FAST=...        # assumption extraction (whole corpus - keep cheap), decomposition
```

Then check resolution without spending a token:

```bash
make agents            # or: GET /health/agents
```

Local infrastructure (Postgres + pgvector, Redis) — needed from Phase 2 onward:

```bash
make up
make install-postgres && make migrate
```

## Layout

```text
.claude/agents/*.md     agent behaviour: role, constraints, evidence and output requirements
app/config/agents.yaml  agent runtime binding: tier, provider, model env, budget
app/domain/             entities, value objects, ports. No vendor SDK, enforced by a test
app/application/        orchestration and use cases
app/infrastructure/     provider adapters, gateway, persistence, observability
app/assumptions/        extraction, normalization, coupling, collisions, invalidation
docs/                   specification (authoritative)
tests/contract/         architectural and specification invariants
```

Agent behaviour lives in Markdown; runtime binding lives in YAML. They are separate
because prompts are reviewed content and provider bindings are deployment configuration
that changes per experiment. No Python class exists merely to hold a prompt.

## Reading order

1. `CLAUDE.md` — working rules and non-negotiable principles
2. `docs/IDEA.md` — product specification and the core mechanism (section 15)
3. `docs/ARCHITECTURE.md` — layering, pipelines, LLM gateway (section 11)
4. `docs/DATA_MODEL.md` — entities including `AssumptionRecord`
5. `docs/DEVELOPMENT_PLAN.md` — phases and acceptance criteria
6. `docs/EVALUATION.md` — baselines, metrics, anti-flattery rules
7. `docs/SECURITY.md` — threat model and prompt-injection stance
8. `docs/STATUS.md` — current state

## Non-negotiables

- Evidence first: every material claim traces to source evidence.
- Confidence is deterministic versioned code over evidence features. A model is never
  asked for a confidence number.
- Corroboration counts independent **sources**, never independent agents. Several agents
  reading the same evidence are one source.
- Analysis diversity comes from differing inputs, not differing personas.
- Uncertainty is an output, not a gap to be smoothed over. Absence of evidence is not
  evidence of absence.
- Schemas are frozen before the prompts that must satisfy them.
- Deterministic code wherever deterministic logic suffices.
