# CCE Development Status

Status: **PHASE 2 COMPLETE — the central Class A claim is demonstrable. PHASE 4 IN PROGRESS:
four agentic stages are built, gated and tested — `consequence-hunter` (Class B discovery,
whose unanswered questions become `EvidenceGap` claims only where a real search returned
nothing), `contradiction-hunter` (materiality over the deterministic divergence set, which
cannot search, reclassify or suppress), `adversarial-challenger` (falsification under a
citation obligation, which cannot refute a derivation on mechanism grounds) and
`consequence-judge` (status and severity over the whole pipeline, with confidence computed by
a deterministic domain service whose inputs are now **measured** from pipeline data). The
pipeline produces `Consequence`, `ArbitrationDecision` and `ConsequenceScore` records, and
the `ConsequenceGraph` projects them deterministically. **The core intelligence layer is now
complete: the blind-spot hunter (an evidence partition disjoint from the main hunt, not a
critic persona), the evidence arbiter (a qualitative position under a deterministic ceiling
computed from measured corroboration) and the report composer (deterministic sections and a
deterministic summary; the agent may only narrow and order the surface) run as stages 8, 9 and
12 of a twelve-stage pipeline. Reviewer verdicts are wired: a finished analysis persists, and
the next run honours `wrong`/`not_material` suppression and measures the already-known rate.
Both evaluation bars still hold, and ablation honesty is now visible on a second, independently
built axis. **All seven specialist agents are integrated as a deterministic fan-out ahead of
the consequence-hunter, so every agent definition in the repository now has a runtime and a
gate. All seven persistence ports have PostgreSQL-backed implementations behind one
Alembic revision, and a complete analysis written by one process is read back by another - so
the reviewer loop survives a restart.** What is missing is a verification run against a real
Postgres server, and the UI.**
**Phase 3 in progress: lexical retrieval, permission filtering, the retrieval benchmark, both
benchmark-driven tuning changes and vector retrieval are delivered. Lexical recall@10 is
91.1% over the labelled suite; the vector strategy measures 80.4% and finds a strict subset
of what lexical finds, so there is no case for fusion yet. The real-embedding experiment was
attempted and is blocked on a missing credential: the recorder and the replay path are built
and proven offline, no synthetic fixture was committed, and no semantic claim is made. A
Gemini `gemini-embedding-001` adapter at 768 dimensions is now prepared and unenabled,
awaiting one run on a machine that holds the key. Hybrid fusion is built and measured: RRF
recovers the lexical result set exactly and ranks it worse, because the weak vector input
supplies eight times the volume and no unique finds. All three are now reachable through one
application entry point that selects by configuration or request and refuses rather than
substitutes.**
Last updated: 2026-09-11

Phase 0 delivered the provider-agnostic LLM substrate. Phase 1 froze the domain contracts.
Phase 2 delivered ingestion, then the coupling resolver. The pipeline now runs end to end:

```text
Ingested source -> verified assumption bindings -> coupling index
  -> authority + scope + temporal resolution -> deterministic Class A findings
  -> knowledge-decay report
```

**Zero model calls occur anywhere in that path**, and the enforcement is structural: a
contract test asserts per file that no module in the Class A path imports a provider, an
HTTP client, the LLM gateway or the agent runtime. The claim is held by the absence of a
seam rather than by a promise.

On the real 13-document corpus, with extraction replayed and nothing else substituted:

| | |
|---|---|
| bindings coupled | 21 across 9 registry keys |
| index cells | 10, keyed `(key, scope, as_of)` |
| derived relations | 8 |
| Class A findings | 14 |
| legitimate overrides | 3, recorded and **silent** |
| ambiguous authority | 1 key, no winner named |
| knowledge-decay entries | 17 |
| Class A recall / false Class A rate | **1.0 / 0.0** |
| live extraction calls | **0** |

All four seeded `must_find` cases are found, all three `must_not_find` cases stay quiet,
and the ablation and newest-wins guards both hold. See `docs/EVALUATION.md` for the metric
table and the mutation-testing result.

## Sequence so far

1. **Specification revision** — `IDEA.md` 15 rewritten around assumption-coupling
   analysis; the edge taxonomy cut to a fixed set; confidence made a deterministic
   versioned function; negative controls made mandatory.
2. **Phase 0** — LLM gateway, provider adapters, agent runtime, API skeleton,
   observability, infrastructure. 107 tests.
3. **Domain checkpoint** — `docs/DOMAIN_CHECKPOINT.md` authored and marked `DECIDED`,
   freezing the five-part epistemic ontology, the closed registry, the two promotion
   gates, and the Pydantic/service invariant split. It is authoritative: where another
   document disagreed, the other document was amended.
4. **Documentation amendment pass** — all seven amendments in checkpoint 15 applied, plus
   the authority-gated override decision, verified mechanically.
5. **Phase 1** — the frozen contracts implemented as Pydantic v2 models.
6. **Phase 2 (ingestion slice)** — parsers, deterministic identity, chunking, span
   verification, the closed registry, extraction behind the provider port, and the
   deterministic admission gate; plus a synthetic corpus and a replay harness.
7. **Phase 2 (coupling slice)** — the coupling index, temporal authority resolution, the
   role-aware collision matrix, invalidation propagation, Class A generation, the Class B
   merge gate, and the knowledge-decay report. This is the point where the product's
   central claim stopped being a design and became a measurement.
8. **Phase 3 (lexical retrieval slice)** — the retrieval contracts and port, then lexical
   retrieval over the stored evidence spans with source-type, scope and as-of filtering.
   Nothing agentic; retrieval exists so that Class B has a corpus to reason over.
9. **Phase 3 (permission filtering)** — authorization on the retrieval path, enforced by
   narrowing the index rather than by a late check, plus the `permission_scope_hash` that
   makes two authorization scopes distinguishable and reproducible.
10. **Phase 3 (retrieval benchmark)** — 35 labelled queries with 62 expectations, a
    deterministic runner, and recall at *k* separated from coverage and from authorization.
    The first number in the project that says how good retrieval actually is, and the
    prerequisite for adding a second strategy.
11. **Phase 3 (stopword list, `tokens-2`)** — the first change made because of a benchmark
    figure, landed alone and re-measured: negative-control results 24 → 5, recall up at @1,
    @3 and @5, and the same 45 expectations retrieved. Issue 32 closed.
12. **Phase 3 (plural rule, `tokens-3`)** — the second, and the last cheap lexical change
    available: one guarded suffix rule, recall@10 80.4% → 91.1%, six previously missed
    expectations recovered, negative-control noise unchanged. Issue 26 closed.
13. **Phase 3 (vector retrieval)** — the second strategy behind the existing
    `EmbeddingPort`: cosine similarity, the same filters and the same authorization
    guarantees as lexical, a deterministic offline embedder, and a replay seam for a real
    provider. Scored separately against the same labels and **not** fused. Measured
    recall@10 80.4%, and zero finds lexical misses.
14. **Phase 3 (embedding adapter foundation)** — what an adapter owes rather than what a
    caller needs: per-vector dimension validation, normalized-vector validation where
    declared, deterministic batching, provenance metadata, and a configuration path with no
    defaults for the model or the credential. A remote adapter exists behind
    `httpx.MockTransport` and is **not enabled**; no live call has been made.
15. **Phase 3 (real embedding experiment — attempted, blocked)** — the recorder that makes
    the one live call, the fixture provenance reproduction needs, and a preflight that
    diagnoses the gap. No credential and no local provider exist here, so **no measurement
    was produced and no fixture was committed**. Two real defects surfaced anyway: the
    provenance label cap was too small for a real embedder's fingerprint, and a model
    identifier is not a version.
16. **Phase 3 (Gemini adapter, prepared)** - `gemini-embedding-001` at 768 dimensions
    behind the same adapter base: its own wire format, the shared error translation, and 58
    `MockTransport` tests. **Not enabled, no live call made, no embeddings generated.**
17. **Phase 3 (hybrid fusion)** - reciprocal rank fusion over the two strategies, as a pure
    function of their two outcomes. Neither retriever modified; every filter and
    authorization guarantee inherited; per-strategy provenance preserved. Measured: it ties
    lexical at @10 and is **worse at @1, @3 and @5**.
18. **Phase 3 (retrieval orchestration)** - one application entry point selecting lexical,
    vector or hybrid from configuration or the request, returning one common
    `RetrievalOutcome` whichever ran. Dispatch only: no filter, no scoring, no merging.
    Unknown strategies and a missing embedder both fail explicitly.
19. **Phase 4 (consequence-hunter)** - the first agentic stage. One call through the
    existing `AgentCompletionPort`, a deterministic gate over its output, and the existing
    `merge_class_b` suppression against the Class A set. The agent cannot supply confidence,
    severity, status or a consequence class, and cannot promote anything to Class A.
20. **Phase 4 (EvidenceGap producer)** - the hunter's unanswered questions become type-E
    claims only where an executed search over a readable corpus returned nothing. The
    provenance comes from the real `RetrievalTrace`; the model contributes the question and
    nothing else. "I don't know" produces no gap.
21. **Phase 4 (contradiction-hunter)** - the second agentic stage, and the first whose input
    is a Class A conclusion. It rates the deterministic divergence set for materiality and
    temporal position; it cannot search, cannot reclassify, cannot suppress and cannot emit
    a number. An empty divergence set costs zero provider calls.
22. **Phase 4 (adversarial-challenger)** - the third agentic stage. Two rulebooks chosen by
    producer: a Class B hypothesis is challengeable on anything, a Class A finding only on
    the four frozen `ClassAGround` members. Three outcomes, each priced - a citation, a
    citation, or a falsification test. An inadmissible or uncited objection is refused and
    counted, never softened into a verdict.
23. **Phase 4 (consequence-judge)** - the fourth agentic stage and the first consumer of the
    whole pipeline. It assigns status and severity, inherits the upstream `consequence_id`,
    and emits `Consequence` / `ArbitrationDecision` / `ConsequenceScore`. Confidence is
    **computed** by a new versioned domain service whose input type cannot express agent
    agreement. Status is proposed then clamped downward only; `no_refutation_found` raises
    nothing, enforced by the clamp having no parameter for it.
24. **Phase 4 (confidence-input assembler)** - the eight checkpoint 9.1 factors are now
    *measured* from pipeline data rather than defaulted to zero. Weights and formula
    unchanged. A narrower permission scope can never score higher, because the assembler
    reads only what the retrievers returned. Measured on the sample corpus, and the
    measurement falsified a prediction recorded in this file.
25. **Phase 8 (graph projection)** - one pure function over admitted records, producing all
    seven frozen edge types with their admissible derivations and directionality. Content-
    derived ids make two rebuilds byte-identical; a record that cannot legally project is
    reported rather than approximated; `EvidenceGap` can annotate a node and never an edge.
    The `retention.window` join is visible with all five holders.
26. **The end-to-end orchestrator** - one use case composing all nine stages, with per-stage
    statuses that keep `empty`, `failed`, `skipped` and `not_configured` apart. **Both
    evaluation bars now measured**: a quiet change produces 0 material findings and 1 recorded
    gap; removing a supporting document removes all 4 findings that rested on it and leaves 0
    citations of its spans.
27. **`FindingDisposition`** - the primary feedback signal has a producer. One verdict per
    `(analysis_id, consequence_id)`, five frozen values, reversible with a full audit trail.
    `wrong` and `not_material` suppress the foreground without touching the consequence;
    `already_known` never suppresses, because it is the signal the product thesis rests on.
28. **Analysis persistence (in-memory)** - `AnalysisStore` + `StoredAnalysis`, the review
    snapshot of a finished run. Tenant-safe reads, the fingerprint stored verbatim, and the
    two features that were inert without it now wired: dispositions validate against the
    stored analysis, and standing suppressions are readable by a later run.
29. **The core intelligence layer** - blind-spot hunter, evidence arbiter, report composer,
    integrated as a twelve-stage pipeline with reviewer feedback flowing back into it. Every
    frozen contract in the checkpoint now has a producer *and* a consumer.
30. **PostgreSQL persistence across all seven ports** - twelve tables, one Alembic revision,
    JSONB payloads with projected columns, and the in-memory implementations kept. A complete
    analysis survives a process restart, verified by writing it in a subprocess and reading it
    back.
31. **Blind-spot findings are reviewed** - `blind_spots` moved ahead of `challenge`, so both
    Class B hunts face the same falsification obligation and a blind-spot finding can arrive
    at the judge `rejected`. Pipeline version `analysis-pipeline-2`.
32. **The composition root** - `app/composition.py`. One `Application.configured()` opens the
    database, builds the seven SQL stores through the existing `SqlStores` seam, and wires
    ingestion and the analysis over them. `python -m app.composition` is the entry point, and
    the PostgreSQL stores are no longer implemented-but-unconstructed.
33. **The specialist fan-out** - seven domain analysts, each over its own source-type slice,
    feeding the existing consequence-hunter. Observations only: no class, no status, no
    severity, no number, and no new persistence model. Pipeline version
    `analysis-pipeline-3`.

## Phase 0 deliverables (unchanged, still verified)

**LLM provider abstraction** — `ProviderPort` / `AgentCompletionPort` / `EmbeddingPort` /
`UsageLedgerPort` in `app/domain/interfaces/llm.py`, no vendor SDK. `LlmGateway` resolves
agent → tier → provider/model from configuration only, enforces the token budget *before*
dispatch, retries with full-jitter backoff honouring `Retry-After`, and validates
structured output with exactly one bounded repair attempt. Adapters: **groq** (default),
**mistral**, **ollama** over a shared OpenAI-compatible base; **gemini** with its own
`generateContent` mapping; **fake** deterministic in-process provider.

**Agent runtime** — `agent_loader.py` parses `.claude/agents/*.md`, validates
name/filename agreement, and content-hashes each prompt so a finding is attributable to
exact prompt text. No Python class holds a prompt.

**API / observability / infrastructure** — `create_app` factory with correlation-ID
middleware and three health endpoints; JSON logging with recursive secret redaction;
in-memory usage ledger; `docker-compose.yml` (pgvector/pg16, Redis 7, MinIO), Alembic
scaffolding, GitHub Actions CI.

## Authority-gated override (decided, then specified)

Divergence between two `declares` bindings of the same key resolves **deterministically**,
three ways, by scope and `authority_class`:

| scope | authority | outcome | edge |
|---|---|---|---|
| equal | any | contradiction | `conflicts_with` (symmetric) |
| narrower ⊂ broader | narrower ≥ broader | legitimate override | none |
| narrower ⊂ broader | narrower < broader | governance defect | `unauthorized_override` (directed) |
| disjoint, or non-overlapping time | any | not coupled for contradiction | none |

Both bindings and both scopes are preserved; the authority comparison and
`authority_model_version` are recorded in the derivation; `resolve_authority` still returns
the broader higher-authority value and flags the narrowing. **No agent discovers,
classifies, or rates this.** The taxonomy grew from six to seven edges by the documented
extension path — written semantics and an admissible derivation, decided before any agent
could emit a finding that projects to it.

Specified in `IDEA.md` 12, `ARCHITECTURE.md`, `DATA_MODEL.md`, `DOMAIN_CHECKPOINT.md`
5.4.3, and the three affected agent prompts (`contradiction-hunter`,
`consequence-judge`, `adversarial-challenger`).

## Assumption key normalization — decided against

There is **no key normalizer** in the MVP, and none is planned for Phase 2. The closed
`AssumptionKeyRegistry` replaces it:

- unknown keys are quarantined as `KeyRegistryGap`, never indexed
- an unknown key never enters deterministic coupling; there is no fallback path
- semantic equivalence between two keys requires a `KeyAlias`, and only a human-approved
  alias participates in Class A coupling
- normalization is registry-declared `canonical_form` per entry (polarity included), not
  learned; no clustering, no embedding-based key merging

The earlier acceptance criterion naming a versioned key normalizer is **withdrawn** as
inconsistent with checkpoint 4.1.5. The obsolete references in `DEVELOPMENT_PLAN.md`,
`EVALUATION.md`, `FOLDER_STRUCTURE.md`, and `.claude/agents/contradiction-hunter.md` were
corrected, and `scripts/check_spec_consistency.py` now treats it as retired vocabulary.

## Phase 1 deliverables (verified)

### Value objects — `app/domain/value_objects/`

| module | contents |
|---|---|
| `enums.py` | `Derivation`, `Support`, `ClaimType`, `BindingRole`, `Explicitness`, `ValueType`, `AuthorityClass`, `CouplingOutcome`, `DeltaKind`, `DeltaAssertedBy`, `RegistryGapStatus`, `AliasStatus`, `ConsequenceClass`, `Origin`, `Status`, `Severity`, `EvidencePosition`, `ChallengeOutcome`, `ClassAGround`, `Disposition`, `SourceType`; plus `AUTHORITY_RANK`, `EXPLICITNESS_RANK`, `UNIT_REQUIRED_TYPES`, `ASSERTIVE_STATUSES`, `STATUSES_ABOVE_SPECULATIVE`, `SUPPRESSING_DISPOSITIONS` |
| `identifiers.py` | prefixed 32-hex ids for 23 kinds, `new_id`, `id_kind`, `ASSUMPTION_KEY_PATTERN`, `VersionLabel` |
| `scope.py` | `ScopeDimension`, `ScopeSelector` |
| `taxonomy.py` | `Relationship` (7), `Directionality`, `NodeType` (5), `ADMISSIBLE_DERIVATIONS`, `DIRECTIONALITY`, `DERIVED_RELATIONSHIPS`, `CONSEQUENCE_BEARING`, `PROJECTION_ONLY` |
| `typed_value.py` | `TypedValue`, `Unit` |
| `provenance.py` | `ModelProvenance`, `DerivationProvenance`, `RunProvenance`, `Sha256` |
| `tiers.py` | `CapabilityTier` (Phase 0) |

### Entities — `app/domain/entities/`, 30 models

- **base** — `DomainModel` (frozen, `extra="forbid"`), `TimestampedModel`,
  `DOMAIN_SCHEMA_VERSION`
- **source** — `SourceDocument`, `SourceVersion`, `Chunk`, `EvidenceItem`,
  `HistoricalEvent`
- **registry** — `AssumptionKeyRegistryEntry`, `KeyRegistryGap`, `KeyAlias`
- **assumption** — `AssumptionRecord` (type B), `CouplingHolder`, `CouplingSet`,
  `AssumptionDelta`
- **change** — `ChangeRequest`, `ChangeSpec`
- **claims** — `ObservedFact` (A), `DerivedRelation` (C), `ConsequenceHypothesis` (D),
  `EvidenceGap` (E), `Claim` discriminated union, `ClaimEnvelope`
- **findings** — `SpecialistFinding`, `FalsificationTest`, `ChallengeResult`,
  `ArbitrationDecision`, `ConsequenceScore`, `Consequence`, `FindingDisposition`,
  `AnalysisReport`
- **graph** — `GraphNode`, `GraphEdge`, `ConsequenceGraph`

Public surface is `app/domain/entities/__init__.py`; a contract test asserts every exported
entity has a sample and therefore cannot escape the cross-cutting invariants.

### Domain services — `app/domain/services/`

`scope_lattice.py` — `equals`, `contains`, `strictly_contains`, `comparable`, `intersects`,
`is_disjoint`, `distance`, `narrower_of`, `broader_of`. Pure functions, no store.

## Invariants enforced at the Pydantic level

**Claim shape**
- types A–E are a discriminated union on `claim_type`; each variant's epistemic axes are
  `Literal`s, so a binding cannot relabel itself a hypothesis
- `evidence_ids` non-empty for `ObservedFact`, and for `refuted` / `weakened`
- `evidence_id` required on `AssumptionRecord`
- `evidence_ids` **empty** for `DerivedRelation`, `EvidenceGap`, `no_refutation_found`
- `DerivedRelation` requires `rule_id` (via `DerivationProvenance`) and non-empty
  `input_record_ids`, and may only express the three derivable relationships
- `EvidenceGap` requires non-empty strategies, source types, `as_of`, and a resolution
  statement
- `no_refutation_found` requires a `falsification_test_id` and forbids evidence and any
  contradiction assertion
- `ChallengeResult.outcome` is exactly three values, with per-outcome requirements
- `ObservedFact` support restricted to `verified_span` / `multi_span`, and `multi_span`
  requires ≥2 spans

**Confidence, class, status, severity**
- `confidence` exists on `ConsequenceScore` alone — asserted by enumeration over all
  entities, not by review
- `ConsequenceScore` requires non-empty `factor_inputs` and a `confidence_fn_version`
- the scorer's reachable field set contains no agent, prompt, or model identity, so it
  **cannot** count agents
- `consequence_class` appears only on `Consequence` and must agree with `origin`
  (`resolver` ⇒ Class A, `agent` ⇒ Class B) in both directions
- Class A requires ≥1 `derived_relation_id` and ≥1 assumption key; Class B requires ≥1
  `hypothesis_id` and forbids a derived relation
- assertive status (`supported` / `likely`) requires evidence
- Class B above `speculative` requires a `historical_analogue_id`
- `severity` is categorical, confined to `ArbitrationDecision` and `Consequence`
- `merged_hypothesis_ids` valid only on a Class A finding

**Bindings**
- `unit` required for `scalar` / `duration`, forbidden elsewhere; `raw` must match the
  declared type; a `bool` is not a scalar; an interval has exactly two endpoints
- `effective_from < effective_to`; validity is half-open
- key matches the registry *shape*; scope is a well-formed lattice path with no repeated
  dimension
- `role` and `explicitness` both required; the retired `binding_kind` is unrepresentable
- the retired `extraction_confidence`, `likelihood`, `has_collision` and `unknowns` are gone
- `registry_version` required on records, deltas, gaps, and aliases

**Registry and aliases**
- approved alias requires `approved_by`, `approved_at`, `registry_version`; approval fields
  cannot be pre-filled while unapproved; a rejected alias requires a reason
- `source_key != target_key`; no confidence or similarity field exists to bypass the gate
- `KeyRegistryGap` requires the proposed key, its evidence, and `as_of`

**Graph**
- `relationship` ∈ the seven MVP edges; `derivation` must be admissible per relationship
- `directionality` fixed by the relationship, not chosen per edge
- `consequence_id` required for consequence-bearing edges; `derived_relation_id` required
  for `derived` edges and forbidden otherwise
- `derived` edges carry **no** direct evidence; `observed` / `extracted` edges must cite
- `is_hypothesized` true iff `derivation = inferred`
- no self-loops; symmetric edges stored once in canonical order
- no `confidence`, no `severity`, and no model attribution path on an edge
- **no relationship admits `absent`**, so an `EvidenceGap` can never become an edge; it may
  only annotate a node
- a projection's edge endpoints must all be projected nodes; ids unique;
  `is_projection` is `Literal[True]`

**Identifiers**
- every id is prefix + 32 lowercase hex, so `"EV-12"`, `"evidence_3"`, a wrong-kind id, or
  a truncated body all fail at the schema boundary. A fabricated citation is a parse error,
  not a detection problem.

## Invariants deliberately deferred to domain services

Cross-object, requires the store. None of these are implemented; the checkpoint lists all
of them and Phases 2–3.5 own them.

- span verification (the anti-hallucination primitive), typed parse, registry resolution,
  quarantine — Phase 2
- unit-aware typed comparison; unit mismatch **blocks** coupling
- scope semantics in anger: equality for contradiction, containment for override,
  intersection for invalidation
- **override legitimacy test** — `narrower.authority_class >= broader.authority_class`
- temporal authority — `resolve_authority(key, scope, as_of) -> SourceVersion | Ambiguous`,
  newest-wins forbidden
- `CouplingSet` construction on `(key, scope, as_of)`; role-and-scope-aware collision
  detection; alias gate; coupling strength
- invalidation propagation; Class A/B classification by producer; Class B merge against the
  Class A set; Class A refutation admissibility; transitive evidence materialization
- confidence computation and corroboration counting over distinct source versions and
  strategies
- graph projection; provenance completeness; reproducibility diff

`ClassAGround` and `GraphEdge` shape make the admissibility and projection rules
*expressible*; enforcing them needs the store.

## Verification

Executed in this environment (Python 3.14.0, Windows):

| command | result |
|---|---|
| `pytest -q` | **677 passed**, 2 warnings, ~3.6s |
| `mypy app tests` | **success, 95 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 125 files already formatted |
| `python scripts/check_spec_consistency.py` | **CONSISTENCY: CLEAN** — 12 docs + 15 agent files |

Test suite runs with **no credentials and no network**.

### Test inventory

| suite | tests |
|---|---|
| `tests/contract/test_domain_invariants.py` | 276 |
| `tests/unit/domain/test_findings.py` | 58 |
| `tests/unit/domain/test_assumption.py` | 49 |
| `tests/unit/domain/test_claims.py` | 45 |
| `tests/unit/domain/test_graph.py` | 44 |
| `tests/unit/domain/test_identifiers.py` | 32 |
| `tests/unit/domain/test_registry.py` | 26 |
| `tests/unit/domain/test_typed_value.py` | 22 |
| `tests/unit/domain/test_scope.py` | 18 |
| Phase 0 suites (unchanged) | 107 |

570 of the 677 are new. The suite is deliberately negative-heavy: the majority of new
assertions prove an invalid state is **rejected**, not that a valid object can be built.
`tests/fixtures/domain.py` builds one valid instance of every entity so a negative test can
change exactly one field and attribute the failure to it.

### Defects found by writing the tests

1. **`AssumptionRecord`'s epistemic axes were overridable.** `claim_type`, `derivation`, and
   `support` were enum fields with defaults and `frozen=True`, which blocks *assignment* but
   not *construction* — `AssumptionRecord(claim_type=HYPOTHESIS)` validated. A type B record
   able to declare itself a hypothesis, or to claim `rule_over_records` support it never had,
   defeats the ontology. Changed to `Literal` types, matching the other four variants.
2. **`typed_value.py` interval check was untyped.** `len(tuple(self.raw))` after an
   `isinstance` against a runtime type tuple narrowed to `object` and failed strict mypy.
   Restructured against a literal type union.
3. **A symmetric edge could be stored twice.** Nothing prevented emitting both A→B and B→A
   for one `conflicts_with`, which would double-count every contradiction and make two
   projections of the same store differ. Now canonically ordered.

## Not implemented (deliberately)

- Ingestion, chunking, embeddings, assumption extraction, coupling index — Phase 2
- Retrieval — Phase 3
- Deterministic invalidation and propagation — Phase 3.5
- Any agent execution; only the runtime and gateway beneath it exist
- ORM models and migrations — none exist yet, by the schemas-before-tables rule. No
  persistence was added for Phase 1; the domain tests need none.
- The confidence function itself — weights stay untuned until the feedback loop supplies
  data
- Auth, tenancy, ACL enforcement, audit logs, tracing, rate limits — Phase 10

## Unresolved issues

Carried from Phase 0:

1. **Groq wire contract is unverified against the live API.** Implemented against the
   documented OpenAI-compatible shape and tested through `httpx.MockTransport`; no real
   call has been made.
2. **Gemini, Mistral, and Ollama adapters are untested code.** Written from documented
   contracts, never executed. Not enabled by default.
3. **Docker Compose was never run.** No Docker daemon in this environment.
4. **Migrations never applied to a real database.** There are no revisions yet.
5. **`psycopg` not installed here**; wheel availability on Python 3.14 unverified. CI pins
   the migrations job to 3.12.
6. **No `LICENSE`.** Choosing one is the owner's decision.
7. **Starlette deprecation warning** — cosmetic, 2 warnings.

New in Phase 1:

8. **`ScopeSelector` containment assumes a strict tree.** `intersects` currently delegates
   to `comparable`, which is correct only while scope is a single ordered path. If a key
   ever needs two independent dimensions to intersect without one containing the other
   (`region:EU` ∩ `tier:enterprise`), the lattice becomes a partial product and `intersects`
   must be rewritten. Registered here rather than pre-solved: no benchmark scenario needs it,
   and guessing the shape now would be a speculative abstraction.
9. **`ConsequenceGraph.analysis_id` is typed as short text, not `AnalysisId`.** A projection
   is currently constructible against a non-id label. Harmless while nothing persists, worth
   tightening when the projector lands in Phase 8.
10. **`AUTHORITY_RANK` gaps of 10 are arbitrary.** The ordering is what matters and it is
    versioned as `authority-1`; the numeric spacing carries no meaning and should not be
    read as a distance.
11. **`extraction_quality` is an unweighted mean of three booleans.** It gates admission
    (all three must pass), so the intermediate values are currently informational only. If
    it ever feeds a score, the weighting needs deciding.
12. **The confidence factor list is specified but unweighted.** `IDEA.md` 13 names the
    factors and directions; `ConsequenceScore` persists `factor_inputs` and
    `confidence_fn_version` so a score is recomputable, but no function exists to produce
    one.

## Contract-freeze attestation

No production prompt was written or revised against an unfrozen contract. The ordering was:
`DOMAIN_CHECKPOINT.md` authored and marked `DECIDED` → documentation amended to match →
entities implemented → tests. The only agent-prompt change in this phase was replacing an
obsolete key-normalization question in `contradiction-hunter.md` with the mis-filed-key and
alias questions the closed registry actually raises, which brings the prompt *into* line
with the frozen contract rather than ahead of it.
`scripts/check_spec_consistency.py` enforces the agreement and is re-run at the start of
every later phase.

## Phase 2 deliverables (verified)

### The path

```text
bytes + SourceRegistration
   -> parse            canonical text + section map, offsets document-absolute
   -> version          version_id = f(source_id, content_hash)  <- the cache key
   -> chunk            section-aware, non-overlapping, deterministic ids
   -> extract          ONE LLM call per chunk, fast tier, behind AgentCompletionPort
   -> ADMISSION GATE   deterministic, no model call
        pass -> AssumptionRecord + EvidenceItem -> assumption store
        fail -> QuarantinedProposal (+ KeyRegistryGap when the key is unregistered)
```

### Modules

| module | responsibility |
|---|---|
| `app/ingestion/identity.py` | canonical text, content hashing, every derived id |
| `app/ingestion/parsers/` | markdown, JSON, plaintext behind one `SourceParser` protocol |
| `app/ingestion/chunking.py` | deterministic section-aware chunking with provenance |
| `app/ingestion/redaction.py` | credential masking at the provider/log boundary only |
| `app/domain/services/span_verification.py` | exact span location and verification |
| `app/domain/services/typed_parse.py` | verbatim value to `TypedValue`; unit mismatch blocks |
| `app/domain/interfaces/repositories.py` | store ports; tenant is a parameter, not a filter |
| `app/assumptions/registry.py` | closed registry, resolution, approved-alias gate |
| `app/assumptions/proposals.py` | `AssumptionProposal` / `ExtractionBatch` (untrusted) |
| `app/assumptions/prompting.py` | untrusted-content envelope, per-chunk nonce delimiter |
| `app/assumptions/extraction.py` | the one LLM call, bounded and attributed |
| `app/assumptions/admission.py` | the admission gate and its 15 refusal reasons |
| `app/application/use_cases/ingest_source.py` | sequencing plus the cache decision |
| `app/infrastructure/database/repositories/memory.py` | the Phase 2 store implementations |
| `app/evaluation/replay.py` | recorded-proposal extractor for reproducible runs |
| `.claude/agents/assumption-extractor.md` | the extractor prompt (principle 12) |
| `app/config/assumption_keys.yaml` | the seed registry: 9 keys, 8 domains, `registry-1` |

### Determinism

Identity is a pure function of its inputs, which is what makes re-ingestion and stable
citations properties of the design rather than optimizations:

```
source_id     = f(tenant_id, uri)
version_id    = f(source_id, content_hash(canonical_text))
chunk_id      = f(version_id, ordinal, char_start, char_end)
evidence_id   = f(version_id, char_start, char_end)
assumption_id = f(version_id, key, scope, role, evidence_id, extractor_version)
gap_id        = f(version_id, proposed_key, evidence_id)
```

Persisted version labels: `parser-1`, `chunker-1` (plus its budget fingerprint),
`registry-1`, and `extractor_version` = code version + prompt template version + prompt
hash. A prompt change therefore produces *new* records rather than overwriting ones a past
analysis cited.

Tenant is part of the `source_id` recipe, not a filter applied afterwards, so two tenants
ingesting the same URI cannot collide onto one document history.

Extraction is skipped on an explicit per-version marker (`extracted_with`), not on whether
records exist, so a document that binds nothing is cached like any other. The marker holds
the `extractor_version`, so a prompt or parser change re-extracts rather than leaving stale
records in place.

### The admission gate

Eight checks, in the order `ARCHITECTURE.md` 3.1 specifies, with no model call anywhere:

1. holder metadata — the version belongs to the supplied document
2. the chunk exists, and the quoted span exists **verbatim** inside it
3. the claimed value appears inside the span cited for it
4. typed parse under the declared type, with a unit where required
5. registry resolution against the **closed** registry, or an **approved** alias
6. registry agreement — the parsed value is the type and unit the entry declares
7. scope well-formed, and every dimension declared by that key's entry
8. temporal validity, inheriting the holder version's interval when the text states none

Parse precedes registry resolution deliberately: a `KeyRegistryGap` records the value that
was proposed, so it can only be written once that value is known to be well typed. An
unparseable value is an extraction failure, not a registry gap, and conflating them would
pollute the signal that says where to grow the registry.

The three verification flags on an `AssumptionRecord` are set **by the gate**. Reaching the
end of the gate is the proof; the extractor has no way to assert it.
`InMemoryAssumptionStore` refuses any record whose flags are not all set, so the trusted
store cannot hold an unverified binding.

### Registry, gaps and aliases

- **Closed.** An unregistered key has no path into the index — not a low-confidence path,
  not a quarantine-then-retry path. It becomes a `KeyRegistryGap`.
- **No key normalizer**, by the Phase 1 decision. Resolution is a dictionary lookup plus an
  approved-alias lookup; near misses (`refund.approval_limit`, `refund_approval_threshold`,
  differing case) resolve to nothing. Canonical form is declared per entry.
- **Gaps are reviewable and counted.** One gap per `(version, key, span)`, with an
  occurrence count, so a repeated proposal does not inflate the volume metric.
- **Aliases are gated, not scored.** A `proposed` alias resolves to nothing and is reported
  as `alias_not_approved` rather than as a generic gap, so a reviewer sees a pending gate.
  There is no confidence field on `KeyAlias` to threshold.

### Security

| control | where | negative test |
|---|---|---|
| source text is data, never instruction | envelope + extractor prompt | `test_an_injected_document_produces_a_normal_prompt` |
| a document cannot impersonate the operator | per-chunk nonce delimiter, literal occurrences neutralized | `test_a_document_impersonating_the_delimiter_is_neutralized` |
| a document cannot fabricate authority | `authority_class` comes from the manifest | `test_the_injected_document_cannot_fabricate_authority` |
| a document cannot add a registry key | closed registry | `test_the_injected_document_cannot_create_a_registry_key` |
| a document cannot approve an alias | human-only approval | `test_the_injected_document_cannot_approve_an_alias` |
| a document cannot select a provider or model | resolved from configuration | `test_document_content_cannot_change_the_resolved_provider_or_model` |
| an extractor cannot emit confidence | `extra="forbid"` on the proposal schema | `test_output_carrying_a_confidence_number_is_discarded` |
| credentials do not cross the provider boundary | redaction, applied to prompt payloads only | `test_a_chunk_with_credentials_reports_them_redacted` |
| tenant isolation | tenant is a port parameter | `test_a_document_is_not_readable_from_another_tenant` |

**One boundary stated plainly, because it is easy to mistake for a defect.** A hostile
document *can* create a binding for a key that is already registered, when the span really
does say it — the vendor notes in the corpus produce
`refund.approval_threshold = 1 USD @global`. What it cannot do is make that binding
authoritative: it holds `wiki` authority, so at coupling time it can only ever be an
unauthorized override of the real policy. This is the documented design in
`docs/SECURITY.md`, and
`test_the_injected_documents_own_bindings_are_recorded_but_powerless` pins it so nobody
later "fixes" it by letting document content influence authority.

Redaction is applied to what crosses the boundary, **not** to the stored canonical text.
Redacting the stored text would break every offset and make verified spans unverifiable
later; the artifact stays byte-faithful and is protected by access control instead.

### Corpus and harness

`data/sample_corpus/` — 11 synthetic documents with registration metadata in
`manifest.yaml`. Gold labels live only in `data/benchmarks/ingestion_expectations.yaml`;
nothing under `app/` reads that file.

`python -m scripts.ingest_corpus --twice` exercises the path locally and prints the store,
the gaps and the alias state. `make ingest-corpus` does the same.

## Acceptance criteria

From `docs/DEVELOPMENT_PLAN.md` Phase 2, stated honestly:

| criterion | status |
|---|---|
| synthetic corpus ingests | **met** — 13 sources, 21 admitted bindings, 2 gaps |
| evidence IDs remain stable | **met** — derived from `(version_id, offsets)`; asserted across a fresh store |
| source versions are queryable | **met** — `versions_for` returns history in order |
| re-ingesting an unchanged source performs zero extraction calls | **met** — skipped on identity; asserted by call count |
| coupling sets are reproducible from the store alone | **met** — two builds over the same store are byte-identical, and order-independent |
| a seeded threshold contradiction is detected by key comparison, with no LLM call | **met** — `10,000` vs `7,500 USD @global`, both `declares`, resolved to `conflicts_with` |
| overlapping effective ranges resolve to `Ambiguous` rather than newest-wins | **met** — `sla.p1_response_time`, two `sop_runbook` peers, and the *newer* one does not win |

The knowledge-decay report (`IDEA.md` 15.5) is delivered: 17 entries over the sample corpus,
separated into known inconsistency, known ambiguity and missing knowledge.

## Verification

Executed in this environment (Python 3.14.0, Windows):

| command | result |
|---|---|
| `pytest -q` | **958 passed**, 2 warnings, ~6s |
| `mypy app tests` | **success, 126 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | all formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** — 12 docs + 16 agent files |
| `python -m scripts.ingest_corpus --twice` | 11 sources, 19 admitted, 2 gaps; second pass 0 extraction calls |

Test suite runs with **no credentials and no network**.

### Test inventory (Phase 2 additions)

| suite | tests |
|---|---|
| `tests/unit/assumptions/test_admission.py` | 32 |
| `tests/unit/domain/test_typed_parse.py` | 32 |
| `tests/integration/test_ingestion_pipeline.py` | 33 |
| `tests/unit/ingestion/test_parsers.py` | 27 |
| `tests/unit/ingestion/test_redaction.py` | 26 |
| `tests/unit/assumptions/test_key_registry.py` | 25 |
| `tests/unit/assumptions/test_proposals.py` | 22 |
| `tests/unit/domain/test_span_verification.py` | 17 |
| `tests/unit/ingestion/test_identity.py` | 17 |
| `tests/unit/assumptions/test_extraction.py` | 16 |
| `tests/unit/ingestion/test_chunking.py` | 15 |
| `tests/unit/assumptions/test_prompting.py` | 12 |
| `tests/contract/test_spec_consistency.py` | 3 |

281 new tests; 958 total. `test_every_failure_reason_is_reachable` fails if an
`AdmissionFailure` is added without a test that reaches it.

### The tests were checked against mutations

Three deliberate mutations of `admission.py` — dropping registry agreement, admitting
unregistered keys, and neutering span verification — produced 10 failures and 14 errors
across the unit and integration suites. A green suite here means the checks are wired in,
not merely present.

### Defects found while building Phase 2

1. **Chunk offsets did not recover the stored text.** `Chunk.text` is whitespace-stripped by
   the frozen `DomainModel` config, so `text[char_start:char_end] != chunk.text` whenever a
   span had surrounding blank lines. The chunker now trims spans to their content.
2. **`AgentCompletionPort` was missing `complete_json`.** The gateway implemented it; the
   port did not declare it, so every structured-output caller was untyped against the port.
3. **A test that never ran.** `text_stored_per_version_is_independent` was named `text_`
   rather than `test_`, so pytest never collected it.
4. **`VALUE_NOT_IN_SPAN` is unreachable in practice.** The proposal schema refuses the shape
   and `QuarantinedProposal` re-validates the embedded proposal, so it cannot even be
   recorded. Kept as depth for a future non-schema caller, and the test says so instead of
   faking a bypass.
5. **The extraction cache was keyed on the wrong thing.** The skip decision asked whether
   a version had records, so a document that legitimately binds nothing re-extracted on
   every ingestion. On a real corpus most documents bind nothing, which would have
   re-extracted almost the whole corpus every run and destroyed the amortization this stage
   depends on. Found by running the corpus twice and noticing the negative control was the
   one file still reporting `[extracted]`. The store now carries an explicit
   `mark_extracted` / `extracted_with` marker, which also correctly re-extracts when the
   extractor version changes.
6. **The consistency checker was not enforced.** It existed only as a script, so CLEAN
   depended on somebody running it. Now `run_checks()` is importable, asserted by
   `tests/contract/test_spec_consistency.py`, and run in CI and `make check`.

## Not implemented (deliberately)

- Embeddings and vector storage — Phase 3 owns retrieval, and nothing in the
  deterministic path needs them
- Retrieval — Phase 3
- Any agent execution beyond the extractor
- The **confidence** scorer. `IDEA.md` 13 names the factors and directions but no
  weights, and there is no calibration data, so there is nothing honest to compute.
  `ConsequenceScore` persists `factor_inputs` and `confidence_fn_version` so a score
  stays recomputable once weights exist. Coupling **strength** is implemented and
  versioned separately; it is reported, not used as a filter.
- The graph projector — Phase 8. `ClassAFinding.relationship` is already constrained to
  the three derivable edges, and a contract test asserts each is admissible as a
  `derived` edge, so projection is a mechanical step rather than a design question.
- **Postgres persistence.** The ports are defined and the Phase 2 implementations are
  in-memory. Nothing in the acceptance criteria needs durability, and tables are written
  against a frozen contract by the schemas-before-tables rule. A SQL implementation is a new
  class behind the same Protocol, with the same tests.
- PDF, HTML, Git and ticket connectors — the parser registry is the extension point
- Auth, ACL enforcement, audit logs, tracing, rate limits — Phase 10

## Unresolved issues

Carried forward:

1. **Groq wire contract is unverified against the live API.** No real call has been made.
2. **Gemini, Mistral and Ollama adapters are untested code.** Not enabled by default.
3. **Docker Compose was never run.** No Docker daemon in this environment.
4. **Migrations never applied to a real database.** There are no revisions yet.
5. **`psycopg` not installed here**; CI pins the migrations job to 3.12.
6. **No `LICENSE`.**
7. **Starlette deprecation warning** — cosmetic, 2 warnings.
8. **`ScopeSelector.intersects` assumes a strict tree.** Correct while scope is a single
   ordered path; a partial product would need a rewrite.
9. **`AUTHORITY_RANK` gaps of 10 are arbitrary.** Only the ordering carries meaning.
10. **`extraction_quality` is an unweighted mean of three booleans.** It gates admission, so
    intermediate values are currently informational.
11. **The confidence factor list is specified but unweighted.** No function exists yet.

New in Phase 2:

12. **Extraction has never run against a live provider.** Every extraction test uses the
    fake provider or the replay harness. The gateway path is exercised; the *quality* of
    real proposals is entirely unmeasured, and it is the largest unknown in the project.
    Until a real run happens, the recall of the mechanism is unknown even though its
    precision is enforced.
13. **`ConsequenceGraph.analysis_id` is short text, not `AnalysisId`.** Carried from Phase 1;
    tighten when the projector lands.
14. **No unit conversion anywhere.** A declared unit must match the registry exactly.
    Correct for now (a mismatch must block), but a corpus writing `1440 minutes` where the
    registry says `h` will produce a quarantine rather than a binding, and the fix belongs
    in one audited place with the registry.
15. **The chunker budget is untuned.** 1200/2000 characters was chosen to keep a sentence
    intact, not measured against extraction quality. It is a `ChunkingConfig`, and its
    fingerprint is provenance, so retuning is a config change with a visible id change.
16. ~~**Intra-document divergence is not yet reported.**~~ **Closed by the coupling
    slice.** Two divergent declarations inside one source version now emit an
    `intra_document_divergence` anomaly rather than a `conflicts_with` relation, so a
    single planted document cannot surface in the Class A tier as two independent
    authorities in conflict.
17. **Gap review has no surface.** `KeyRegistryGap` is persisted and queryable by key, but
    there is no admin endpoint to register a key or approve an alias. The checkpoint allows
    a persisted approval state plus an admin API for the MVP; only the state exists.

New in the coupling slice:

18. **`ClassAFinding` deviates from the checkpoint entity map.** The reasoning and the
    decision needed are recorded in the *Open review item* section below. This is the only
    place in the slice where the frozen contract was not followed literally, and it was not
    changed silently.
19. **The coupling-strength weights are untuned.** They encode the orderings checkpoint 5.7
    states and nothing more; there is no calibration data. That is why the function is
    versioned as `strength-1`, and why strength is currently *reported* rather than used as
    a filter or a threshold. Nothing downstream branches on it.
20. **`AnomalyKind` values are reported but not prioritized.** A `consumer_divergence` and a
    `blocked_by_unit_mismatch` land in the report with equal weight, which is honest but
    will not scale: on a large corpus unit mismatches are probably registry defects worth
    surfacing separately from genuine scope questions.
21. **The alias-mediated path in coupling strength is untested against a real alias.** No
    `KeyAlias` has ever been approved, so `alias_mediated=True` is exercised only by unit
    tests passing the flag directly. The gate itself is tested; the end-to-end promotion of
    an approved alias into a Class A derivation is not.
22. **`resolve_authority` assumes the strict-tree lattice.** Carried from issue 8: the level
    walk relies on "two governing scopes at equal depth are the same scope", which holds for
    a single ordered path and fails for a partial product (`region:EU` ∩ `tier:enterprise`
    with neither containing the other). No benchmark scenario needs it, and the
    `intersects` rewrite and this walk would have to change together.
23. **The decay report has no volume control.** Seventeen entries over thirteen documents is
    readable; the same density over ten thousand documents is not. `alert_load` is an
    `EVALUATION.md` metric and the report currently has no ranking, no suppression and no
    paging. `FindingDisposition` exists to drive suppression but nothing consumes it yet.
24. **Delta propagation reaches broader bindings from a narrower change.** Checkpoint 5.5
    specifies *intersection* for invalidation, and `global` intersects `region:EU`, so a
    regional change reports the global binding as invalidated. That is arguably correct —
    the reader does have to decide whether the global rule still holds — but it is a
    judgment about usefulness that the checkpoint does not settle, and on a corpus with many
    regions it may prove noisy. Pinned by a test that names the behaviour rather than
    asserting it away.

## Phase 2 coupling slice (verified)

### The path

```text
Assumption Store (admitted bindings only)
   -> join to holder authority        authority_class from REGISTRATION, never content
   -> ResolvedBinding[]               pure values from here down; no store, no model
   -> resolve_authority               Resolved | Ambiguous | NoDeclarer
   -> collision matrix                role- and scope-aware (checkpoint 3.3 / 5.4)
   -> DerivedRelation[]               type C: rule_id + inputs, NO evidence of its own
   -> ClassAFinding[]                 one per affected artifact, evidence = derived union
   -> KnowledgeDecayReport            inconsistency | ambiguity | missing knowledge
```

### Modules

| module | responsibility |
|---|---|
| `app/domain/value_objects/rules.py` | the six named rules and the version labels pinned onto their output |
| `app/domain/services/value_comparison.py` | unit-aware three-way compare; a mismatch **blocks** |
| `app/domain/services/temporal_intervals.py` | half-open `[from, to)` arithmetic |
| `app/domain/services/coupling_strength.py` | the versioned score; cannot see an agent |
| `app/domain/entities/coupling.py` | `ResolvedBinding`, `AuthorityResolution`, `OverrideRecord`, `CouplingAnomaly`, `CouplingStrength` |
| `app/domain/entities/class_a.py` | `ClassAFinding`, `ClassBMergeOutcome` |
| `app/domain/entities/decay.py` | `KnowledgeDecayReport`, `DecayEntry`, `DecayParticipant` |
| `app/assumptions/temporal.py` | `resolve_authority(key, scope, as_of)` |
| `app/assumptions/collisions.py` | the role matrix and the three-way divergence rule |
| `app/assumptions/coupling.py` | the index, keyed `(key, scope, as_of)` |
| `app/assumptions/invalidation.py` | delta propagation, with scope inheritance kept distinct |
| `app/assumptions/class_a.py` | Class A generation, Class B merge, rediscovery rate |
| `app/assumptions/decay.py` | the standing report |
| `app/application/use_cases/resolve_coupling.py` | the only store-touching step |
| `app/evaluation/corpus.py` | one corpus loader, shared by scripts and tests |
| `scripts/coupling_report.py` | the end-to-end demonstration, with `--delta` |

### The decisions that were not obvious

**Staleness is evaluated against resolved authority, not pairwise.** The literal reading of
"declares vs consumes, divergent" compares a consumer against *every* declarer. On this
corpus that reports the training deck stale because a lower-authority operations standard
disagrees with the policy the deck correctly follows — a false positive in the tier whose
whole value is that it has none. So a consumer is stale iff its value differs from what
`resolve_authority` returns for its scope. Where authority is `Ambiguous`, **no** staleness
finding is emitted: if the corpus cannot say what the value is, it cannot say a consumer has
the wrong one.

**The override authority test is applied per candidate, not per level.** One scope can hold
both a legitimate regional regulation and an unauthorized team wiki page — the sample corpus
does, at `region:EU`. Rejecting the whole level would lose the regulation; accepting it would
bless the wiki.

**Legitimacy for scope inheritance is decided by `resolve_authority`, not by the existence
of an override record.** These are not the same test, and conflating them was a real bug
caught during the slice: the EMEA wiki page legitimately outranks *another wiki page* at
`global` while remaining an unauthorized narrowing of the retention *policy*. Matching on the
override record alone told the reader to "confirm" a governance defect the resolver had
already flagged — exactly the merge checkpoint 5.5 forbids.

**A symmetric conflict yields two findings but one decay row.** Both authors have something
to reconcile and a reviewer filtering by document must see it from either side, but it is one
inconsistency in the corpus and counting it twice would inflate the defect count.

**`ClassAFinding` exists because a `Consequence` requires a `severity`.** See the open
review item below.

### Determinism

Relation and finding ids are derived, not minted:

```
relation_id = f(rule_id, key, scope, sorted(input_record_ids))
finding_id  = f(relation_id, affected_source_version_id)
```

Sorting the inputs is what makes a symmetric `conflicts_with` collapse to one relation
instead of being emitted once per direction. Every ordering inside the resolver is by
`assumption_id`, so store iteration order cannot change a finding. Two builds over the same
store are byte-identical, and the reproducibility tests assert both equality and
order-independence.

Three version labels travel with every derivation, because each can change output with no
document changing: `coupling_index_version` (`coupling-1`), `registry_version`, and
`authority_model_version` (`authority-1`). `COUPLING_STRENGTH_VERSION` is separate so
retuning weights cannot look like a change to the collision rules.

### Corpus additions

Two documents were added, because nothing in the original eleven produced the corresponding
case and the phase requires both:

- `dashboard/retention-analytics-definition.md` — `consumes retention.window = 365d` where
  the policy declares 90d. Every consumer in the original corpus **agreed** with its
  governing declarer, so the staleness rule only fired on a delta; section A of the decay
  report needs a divergence that exists with no change request at all.
- `sop/incident-response-runbook.md` — a second `sop_runbook` declaring
  `sla.p1_response_time = 30 min` against the existing 15. Deliberately the **newer**
  document (2025-09-01 vs 2025-02-01), so an implementation with a newest-wins fallback
  would resolve to it and pass everything else in the suite.

### Verification

Executed in this environment (Python 3.14.0, Windows):

| command | result |
|---|---|
| `pytest -q` | **1403 passed**, 2 warnings, ~6.5s |
| `mypy app tests` | **success, 155 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 200 files formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** — 12 docs + 16 agent files |
| `python -m scripts.ingest_corpus --twice` | 13 sources, 21 admitted, 2 gaps; second pass 0 extraction calls |
| `python -m scripts.coupling_report` | 9 keys, 21 bindings, 10 cells, 8 relations, 14 Class A findings, 17 decay entries, **0 model calls** |
| `python -m scripts.coupling_report --delta 'retention.window=90d->180d'` | 3 invalidated, 1 confirm, 1 not-affected, 1 pre-existing defect reported separately |

Test suite runs with **no credentials and no network**.

### Test inventory (coupling slice additions)

| suite | tests |
|---|---|
| `tests/contract/test_trust_invariants.py` | 50 |
| `tests/unit/assumptions/test_collisions.py` | 43 |
| `tests/evaluation/test_class_a_benchmark.py` | 37 |
| `tests/unit/assumptions/test_decay.py` | 34 |
| `tests/unit/assumptions/test_temporal_authority.py` | 30 |
| `tests/unit/assumptions/test_coupling_index.py` | 30 |
| `tests/unit/assumptions/test_class_a_generation.py` | 30 |
| `tests/unit/domain/test_temporal_intervals.py` | 28 |
| `tests/unit/domain/test_value_comparison.py` | 25 |
| `tests/unit/assumptions/test_invalidation.py` | 25 |
| `tests/unit/domain/test_coupling_strength.py` | 23 |

445 new tests; 1403 total. The suite is deliberately negative-heavy: the tests that carry
the most weight are the ones asserting that **nothing** happens — a legitimate override is
silent, a `describes` binding never produces a finding, two consumers disagreeing is not a
contradiction, a unit mismatch blocks rather than weakens, and a consumer that agrees with
its governing policy is not stale merely because some lower-authority document disagrees.

### The tests were checked against mutations

Eight plausible wrong implementations were introduced one at a time; the suite had to fail
on each. **All eight were caught**: a newest-wins fallback in `resolve_authority`, scope
containment conferring override rights without the authority test, a unit mismatch weakening
instead of blocking, key-only (scope-blind) coupling, `describes` bindings participating in
collisions, pairwise instead of authority-based staleness, adjacent intervals treated as
overlapping, and legitimate overrides reported as findings.

### Defects found while building the coupling slice

1. **The affected artifact was resolved through a version-keyed map.** One document version
   routinely holds several bindings — the EMEA notes hold a global refund threshold *and* an
   EU retention window — so the lookup silently attached the wrong scope and the wrong value
   to a finding. Visible in the first report run as "narrows retention.window to 10000USD at
   global". Now resolved from the relation's own `input_record_ids`.
2. **Scope inheritance blessed an unauthorized override.** `_legitimate_override_of` matched
   on the existence of *any* override record naming the binding, and the EMEA wiki page
   legitimately outranks another wiki page at `global`. The delta pass therefore asked the
   reader to "confirm" a governance defect the standing resolver had already flagged. Now
   decided by `resolve_authority` at the binding's own scope.
3. **`merge_class_b` had an order-dependent tie break.** Four `(key, artifact)` pairs in the
   corpus are shared by two findings — an artifact can be both a peer in a same-scope
   contradiction and the deviating side of an unauthorized override on the same key — so the
   absorbing finding depended on iteration order. Now the lowest `finding_id` wins.
4. **The span-verification gate in the scorer is unreachable.** `ResolvedBinding` refuses a
   record that has not passed all three admission checks, which is the stronger guarantee.
   The gate is kept as defence in depth for a future caller that hydrates without
   re-validating, and the test says so rather than faking a bypass.
5. **A circular import between the fixture modules.** `tests/fixtures/domain.py` merges the
   coupling sample inventory at the bottom while `coupling.py` imports its primitives from
   it, so whichever was imported first decided whether the suite could collect. The shared
   primitives moved to a leaf module, `tests/fixtures/ids.py`.
6. **The false-positive metric was scored per artifact.** That counted a true positive as a
   false one: the EU data-protection policy must never be flagged as an *unauthorized
   override*, but it is legitimately reported as a same-scope contradiction against the EMEA
   wiki page. Scored per `(artifact, rule)` now, and the gold labels state which rules are
   forbidden for each case.
7. **The first ablation test asserted the wrong thing.** Removing the retention policy leaves
   the vendor wiki page as the only global declarer, so a staleness finding against the CLV
   definition correctly still exists — from different records, citing different spans. The
   honest claim is that nothing survives *unchanged*, which is now asserted three ways: no
   `finding_id` survives, no evidence span from the removed document is cited, and the
   surviving finding names a different authority.

## Open review item — `ClassAFinding` versus `Consequence`

**This is a deliberate deviation from the checkpoint's entity map (§11) and needs an
architectural decision.**

Checkpoint §11 shows `DerivedRelation -> Consequence(class_a)`. The resolver does not
produce a `Consequence`, because `Consequence.severity` is required and §9.4 is explicit
that severity is **judgment** — a function of business impact the corpus does not contain —
assigned by the judge, attributed to a prompt/provider/model, and reviewer-overridable. The
resolver has no model and no business context. The three ways to satisfy the map as written
were all worse than deviating:

- invent a severity per rule — laundering a hard-coded guess into the trust-bearing tier;
- default everything to `LOW` — which misrepresents an unauthorized override of a
  regulatory retention policy;
- make `severity` optional on `Consequence` — a change to a frozen contract, for the
  benefit of one producer.

So the resolver emits a `ClassAFinding`: fully derived, fully evidenced, carrying no score
and no judgment, with `consequence_class` and `origin` as `Literal`s. The judge converts it
into a `Consequence` in a later phase by adding exactly the field it is qualified to add.

Nothing was silently changed: `DOMAIN_CHECKPOINT.md` is untouched, and the deviation is
recorded here and in `docs/DATA_MODEL.md`. **Decision needed:** amend §11 to insert
`ClassAFinding` between `DerivedRelation` and `Consequence`, or collapse it back when the
judge lands.

## Phase 3 — lexical retrieval, permission filtering and the benchmark (verified)

Six slices, in order: lexical retrieval, authorization over it, a labelled benchmark that
measures both, the two tokenizer changes that benchmark justified, then vector retrieval
scored against the same labels. **No reranking, no hybrid fusion, no stemmer, no
lemmatizer, no synonyms, no query expansion, no fuzzy matching, no graph retrieval and no
agent.** Vector search and embeddings were absent for the first five slices and are the
subject of the sixth. The plan order stands: retrieval comes
before any agentic layer, because a Class B hypothesis over an unretrieved corpus is a
hypothesis about nothing — and the benchmark comes before the second retrieval strategy,
because a fusion weight tuned against no number is not a decision. The two tuning changes
sit between them for the same reason: they were cheap, they were measurable, and doing them
first means the vector comparison starts from a lexical baseline that has had its obvious
defects fixed rather than from a straw man.

### The path

```text
Evidence store (admitted spans only)
   -> join to version + document        tenant, source_type and authority from REGISTRATION
   -> join to admitted bindings         the scopes and keys the span actually commits to
   -> EvidenceCandidate[]               pure values from here down; no store, no model
   -> narrow to the tenant              a partition, not a filter
   -> narrow to the authorized slice    AUTHORIZATION, before any content filter
   -> filters, fixed order              source_type -> as_of -> scope
   -> BM25 over the authorized slice    statistics derived after authorization, versioned
   -> rank, tie-break, truncate         descending score, then ascending evidence_id
   -> RetrievalOutcome                  results + trace + permission_scope_hash + versions
```

### Modules

| module | responsibility |
|---|---|
| `app/domain/interfaces/retrieval.py` | the contracts: `EvidenceCandidate`, `RetrievalQuery`, `RetrievalResult`, `RetrievalTrace`, `RetrievalOutcome`, `RetrievalPort`, `RETRIEVAL_CONFIG_VERSION` |
| `app/domain/value_objects/permissions.py` | `PermissionScope`, the label match rule, and the deterministic `scope_hash` |
| `app/retrieval/tokens.py` | the tokenizer, one function applied to both sides of a match |
| `app/retrieval/lexical.py` | `LexicalRetriever` — Okapi BM25, the filters, the ranking |
| `app/application/use_cases/retrieve_evidence.py` | the only store-touching step: the candidate join, and the authorized accessor over it |
| `app/domain/value_objects/enums.py` | `RetrievalStrategy`, with the single member that exists |
| `app/domain/interfaces/repositories.py` | `SourceStore.evidence_for_tenant` — the corpus enumeration retrieval needs |

### The retrieval interface

One strategy, one filter set, one `top_k`. `ARCHITECTURE.md` 6 calls the multi-strategy
form a QueryPlan; a plan composes queries once there is more than one strategy to compose,
and writing it now would be guessing at the shape of fusion.

Three properties are structural rather than left to each retriever:

**Provenance travels by reference, never by copy.** An `EvidenceCandidate` holds the stored
`EvidenceItem`, its `SourceVersion` and its `SourceDocument`. Nothing is flattened into a
retrieval-local shape, so `evidence_id`, the offsets, the version and the owning document
cannot drift from the store, and a citation taken from a result is the same citation the
admission gate verified. A model validator rejects a candidate whose evidence belongs to a
different version, or whose version belongs to a different document — a citation assembled
out of genuine parts is still fabricated.

**A result cannot carry a zero score.** `score > 0` is on the type. A span sharing no term
with the query is not a weak result; it is not a result, and it must not reach an agent
looking as though retrieval had endorsed it.

**Every excluded candidate is counted.** `RetrievalTrace` charges each candidate to the
first filter it failed, so the buckets sum to `considered` — asserted by a test. This is
what makes "retrieval returned nothing" a reportable observation about the corpus:
*nothing matched the words* and *everything was filtered out before scoring* are different
findings, and one empty list cannot tell them apart. It is the raw material for negative
provenance on an `EvidenceGap`, which is why it exists before the gap producer does.

### Lexical retrieval behaviour

**The retrievable unit is the `EvidenceItem`** — the verified span — not the chunk and not
the document. That is what makes a result citable with no further work: the thing retrieval
returns is exactly the thing a finding may cite, offsets included.

**Scoring is Okapi BM25** over the span text alone. Section labels and titles are excluded:
a match must be attributable to words inside the citation, or a result is justified by
metadata the reader cannot see in the quoted span. The IDF form is the non-negative variant;
the textbook form goes negative for a term present in more than half the corpus, which on a
corpus this small would let a common term *subtract* from a score and rank a matching span
below a non-matching one.

**One tokenizer runs over both the query and the span.** Case is folded and digit-group
separators are dropped, so the corpus writing `10,000` matches a change request writing
`10000` — the most important term in a threshold question. Terms split on any
non-alphanumeric character, so `retention.window` typed as a query reaches the prose that
binds it. There was no stopword list, no suffix rule and no synonym table: each an untuned
knob with no benchmark yet to justify it. Two have since been added and measured — the
stopword list in `tokens-2` and one guarded plural-`s` rule in `tokens-3`, each with a
before-figure recorded below. A synonym table, a stemmer and a lemmatizer are still absent.

**Filters select; they do not re-weight.** Corpus statistics are computed before any filter
runs, so a source-type filter changes *which* spans come back and never the score of a span
that survives. A filtered run is therefore comparable with an unfiltered one, which a
benchmark will need.

| filter | semantics |
|---|---|
| authorization | any label on the holder document matched by a grant; unlabelled means unrestricted. Applied **before** everything below |
| source type | `document.source_type ∈ named types`; empty means every type, so a caller cannot silence retrieval with an empty allow-list |
| scope | any bound scope of the span **intersects** the queried scope |
| as of | the holder `SourceVersion` is in force, `[from, to)` |
| tenant | structural, not a filter: the index is partitioned per tenant |

Authorization appears in that table but is deliberately **not** one of those filters — see
the next section for why the distinction is load-bearing rather than cosmetic.

**Scope uses intersection, not equality**, the same reading invalidation uses (checkpoint
5.5): a `global` declaration is relevant to a `region:EU` question, and a `region:EU` span is
excluded from a `region:US` question because siblings are disjoint. A span that binds
*nothing* has no scope, so a scope-filtered query excludes it and counts it in
`excluded_unscoped` rather than in `excluded_by_scope` — that bucket is a coverage signal
saying the registry needs expanding, not that the query needs rewriting.

**As-of is evaluated at version level.** Binding-level temporal validity decides which value
is *authoritative*, and that is `resolve_authority`'s question, already answered
deterministically in Phase 2. Retrieval asks only whether the artifact governs at all.

**Tenant isolation is partitioning, not filtering.** `evidence_for_tenant` is the only
corpus enumeration on the port and it takes a tenant; the retriever then indexes per tenant,
so another tenant's spans are never iterated. Document frequency is per tenant too, so a
ranking cannot depend on who else is on the system.

### Permission filtering

`docs/SECURITY.md` 4 requires authorization *before* retrieval output enters an agent. The
requirement has two halves that pull against each other: an unauthorized span must be
withheld **and** the withholding must be reported, because a silent disappearance is
indistinguishable from an empty corpus. The whole design is in what the report is allowed to
say.

**The scope is a required input.** `RetrievalQuery.permission_scope` has no default and no
`None` case. A scope defaulting to "no filtering" would make *forgetting* the parameter the
unrestricted case, and the rule this project keeps returning to is that the unsafe thing
must be the hard thing to write.

**The model is an ACL, decided rather than left implicit.** `SourceDocument.permissions` is
a set of labels, put there by registration like `authority_class` and never readable from
document content, so a planted file can neither widen its own audience nor claim to be
readable. A `PermissionScope` holds grants. **Any** single matching label grants read: the
labels answer "who may read this", so they are read disjunctively and not as a clearance
requirement. Matching is case-folded on **both** sides, because an asymmetric comparison
makes access depend on where the shift key landed. There is no wildcard grant.

**Enforcement is by narrowing the index, not by adding a check.** `_authorized_slice`
reduces the tenant's index to the readable spans before anything is filtered, scored,
ranked or counted. Everything downstream iterates a collection an unauthorized span is not
in, so no ordering mistake, later edit or early `return` can leak one — leaking one would
require putting it back. A per-candidate check alongside the content filters was the first
draft and is strictly weaker: it makes the guarantee rest on the order of a chain of `if`
statements.

**Authorization runs first because the count must not become an oracle.** The trace reports
`excluded_by_permission`, and that number is deliberately independent of the query: it says
how much of the tenant's corpus is closed to the requester, never whether any of it matched
their words. Applied *after* scoring, the same field would answer "how many restricted
documents mention my search terms?" — a content oracle over documents the requester cannot
read. Pinned by a test that runs a matching and a non-matching query and asserts the count
is identical, and by another that probes with the exact wording of a restricted span.

**Corpus statistics are derived after authorization.** Document frequency and mean length
are computed over the authorized slice, so a score cannot depend on a document the requester
may not read. The consequence is that two scopes legitimately rank the same span
differently, and that is not a defect: a scope is *the corpus you have*, not a filter over a
corpus everyone shares. It is exactly what the scope hash exists to record.

**What escapes about a denied document: a count.** No id, no title, no uri, no source type,
no authority, no label, no span, no offsets. Asserted by serializing the whole outcome and
searching it for every field the denied document has, and structurally by the fact that
every field on `RetrievalTrace` is typed `int` — there is nowhere to put an identifier. The
one place restricted *wording* can appear in an outcome is the echoed query, which is the
requester's own text; a test asserts it appears nowhere else.

**`permission_scope_hash` on the outcome.** A SHA-256 over `permission-1` plus the canonical
grant list, newline-joined. Deterministic for the *effective* scope: `("b","a")`,
`("a","b","a")` and `("A"," b ")` all hash alike, so a cache key cannot fragment on
spelling, while different access never collides — a grant may not contain a newline, so one
grant cannot forge the canonical form of two. The model version is inside the preimage,
because a change to the match rule must change every hash rather than let a stored scope
keep claiming an access decision the code no longer makes. A validator on `RetrievalOutcome`
refuses a hash that disagrees with the query it answers, so results computed under a wide
scope cannot be relabelled and served to a narrow one — the cross-ACL replay
`ARCHITECTURE.md` 14.1 forbids.

**Tenant isolation is untouched.** It remains a partition applied before authorization, and
a grant does not cross it: holding the label a foreign tenant's document carries does not
make it readable. Two tests pin that, one of them asserting that a foreign restricted span
is charged to the tenant bucket rather than the permission one.

**The public surface has one candidate accessor and it filters.** `authorized_candidates`
takes a scope; the raw join that includes unreadable spans is private, and its only consumer
is the retriever. A caller wanting "all of them" has to say which grants it holds.

### Determinism

Ordering is by descending score, then **ascending `evidence_id`**. Scores are rounded before
comparison, so two spans quoting the same words tie and are broken by id rather than by
floating-point noise or dictionary order. The candidate list is itself sorted by
`evidence_id` before indexing, so store iteration order cannot decide what an analysis saw.
Two runs of one query are byte-identical, asserted over the real corpus.

Three labels travel with every outcome, and they are separate on purpose:
`permission_scope_hash` says *who* the results were computed for;
`retrieval_config_version` (`retrieval-1`) covers which filters exist and how they compose;
`scorer_version` is the scorer plus the tokenizer plus the tuning parameters
(`lexical-bm25-1:tokens-3:k1=1.2/b=0.75`), so retuning `k1` cannot look like a change to the
filters. That fingerprint has since earned its keep twice: the stopword list moved the
tokenizer component to `tokens-2` and the plural rule to `tokens-3`, and on both occasions
the benchmark's pin refused the run until the recall figures were re-measured rather than
reprinted.

### The retrieval benchmark

The third retrieval slice, and the one that turns every claim above into a number. Full
definition and results in `docs/EVALUATION.md`, "Retrieval recall, as measured"; what
follows is the shape of it and the findings.

| module | responsibility |
|---|---|
| `data/benchmarks/retrieval_queries.yaml` | the gold labels: 35 queries, 62 expectations, the scopes, and the ACL overlay |
| `app/evaluation/retrieval_metrics.py` | the recall arithmetic — pure functions over labelled outcomes, no corpus |
| `app/evaluation/retrieval_benchmark.py` | the suite schema, the loader, the label resolver, the runner |
| `scripts/retrieval_benchmark.py` | the report: `make retrieval-benchmark`, `--per-query`, `--json` |

**A label names a source path and a quote, never an `evidence_id`.** The runner resolves
the pair to the span containing it. An id is opaque, so a reviewer cannot judge whether the
label is right — and, decisively, a stale id is indistinguishable from a span that was
never admitted, which is the distinction the whole benchmark is built on. A quote must occur
exactly once in its document, matched against the canonical text; the same discipline the
extraction fixtures already use.

**`expect` is a relevance judgment about the question**, not a prediction about the
retriever and not a statement about who may read the evidence. That is what lets one
expectation be a hit under a scope holding a grant and a correct withholding under one that
does not, with the label unchanged. It is also why the region:US query does *not* label the
EU spans it excludes: they are irrelevant to a US question, and labelling irrelevant
evidence to demonstrate an exclusion would corrupt the judgments. That exclusion is read
from the query's own trace instead.

**Three ways an expectation can fail to arrive, kept apart, because only one is retrieval's:**

| | count | owner |
|---|---|---|
| retrieval miss | 11 | the retriever |
| never admitted — no verified binding covers the text | 4 | ingestion (issue 25) |
| withheld by authorization | 2 | nobody; correct behaviour |
| removed by a filter the query asked for | 0 | the suite, if ever non-zero |
| label defect | 0 | the suite |

**Every query runs at `MAX_TOP_K` and recall@k is derived by truncating the ranking.** Exactly
equivalent, because the ordering is total and deterministic, so a prefix of the long run *is*
the short run — and it buys one thing separate runs could not: a miss is reported as *shares
no term with the query* rather than merely *ranked below k*, because a span that scored at
all has a rank.

**The suite ships its own ACL overlay.** The sample corpus carries no permission labels
(issue 30), so authorization recall was unmeasurable against it. The suite declares one
label on the retention policy and the runner applies it through the store port, which is
what an operator applying an ACL does. `manifest.yaml` stays untouched.

#### Results

Measured under `tokens-3`, the current scorer. The earlier rows are the same suite under
the two previous tokenizers, so the effect of each tuning change is on the record.

| | | @1 | @3 | @5 | @10 |
|---|---|---|---|---|---|
| **recall** (of the 56 reachable expectations) | `tokens-3` | **35.7%** | **75.0%** | **89.3%** | **91.1%** |
| | `tokens-2` | 35.7% | 67.9% | 80.4% | 80.4% |
| | `tokens-1` | 33.9% | 62.5% | 78.6% | 80.4% |
| end-to-end recall (of all 62 labelled) | `tokens-3` | 32.3% | 67.7% | 80.6% | 82.3% |
| mean per query | `tokens-3` | 53.5% | 85.5% | 94.1% | 94.8% |
| authorization-filtered recall (3 restricted queries) | all three | 50.0% | 100% | 100% | 100% |

Coverage — the reachable share of the labelled evidence — is **90.3%**, unchanged by either
tuning change. Withheld evidence returned anyway: **0**. Negative controls: 4 queries,
**3 silent, 5 results returned** (`tokens-1`: 2 silent, 24 results; unchanged by the plural
rule). Expectations retrieved / missed: **51 / 5**, from 45 / 11 under `tokens-2`.

Both recall denominators are printed side by side, always. Quoting only the first flatters
the retriever; quoting only the second blames it for the ingestion boundary.

#### What the benchmark found

1. **Typing an assumption key does not reconstruct its coupling set.** `retention.window`
   is bound by five artifacts; only the injected vendor wiki page spells the key out, and
   the policy that governs writes "retained for 90 days". So the query returns the *least*
   trustworthy member of the coupling set at rank 1 and misses the governing policy: 1 of 5,
   and neither tuning change moved it. This is the empirical case for the closed registry,
   and for Class A recall not depending on retrieval at all.
2. **A paraphrase can fail for two different reasons, and this list confused them.** "who
   has to sign off on a large refund" was recorded here as sharing no content word with the
   2 spans that answer it, and called the vector-retrieval case. It was a *morphological*
   gap: the spans write "Refunds", the query writes "refund", and the plural rule alone
   recovered both. The genuinely semantic gap is one query, not three — see issue 26 below.
   Correcting this downward is the more useful result.
3. ~~**The missing stemming costs the `known_lexical_miss` category 80% of its recall.**~~
   **Closed** in `tokens-3`: 20% flat across every *k* became 60 / 80 / 80% at @3 / @5 / @10,
   with negative-control noise unchanged. See "The plural rule (issue 26, closed)".
4. **Coverage is the ceiling and it is not retrieval's to raise.** One of the four
   unreachable spans is a material policy exception ("a refund arising from a confirmed
   billing defect may be released without manager approval"); another is blocked by
   *registry* coverage rather than extraction, because the EMEA escalation ceiling proposes
   an unregistered key. Registry coverage is a retrieval ceiling, which was not obvious
   before it was measured.
5. **Authorization costs nothing in recall for what the requester may read.** Not a
   foregone conclusion: statistics are recomputed over the authorized slice, so a restricted
   requester scores every span differently. Run on the identical query with and without the
   grant, the entitled evidence comes back at the same ranks.
6. **The absence of a stopword list was expensive in precision.** "the coffee machine is
   coming with us" returned 13 spans and "pack your personal items into the crate" returned
   11, on `the`, `is` and `personal`. This is the one finding that has since been acted on —
   see the next section — and it was recorded as a number first, which is what made the fix
   attributable.

`k1=1.2` / `b=0.75` remain the untuned defaults (issue 27). Nothing is fitted to the suite:
the stopword list is the only change it has caused, and its contents were chosen by
grammatical class rather than from the suite's own failures, for exactly that reason.

### The stopword list (issue 32, closed)

The first change made *because* of a benchmark figure, and deliberately the smallest one
available. `app/retrieval/tokens.py` now drops a closed-class list of English function words
from **both** the query and the span; `TOKENIZER_VERSION` moved to `tokens-2`. Nothing else
changed — no stemming, no synonyms, no query expansion, no second strategy, no scorer
parameter.

| | `tokens-1` | `tokens-2` | |
|---|---|---|---|
| recall@1 | 33.9% | **35.7%** | +1.8 |
| recall@3 | 62.5% | **67.9%** | +5.4 |
| recall@5 | 78.6% | **80.4%** | +1.8 |
| recall@10 | 80.4% | **80.4%** | — |
| negative controls silent | 2 of 4 | **3 of 4** | +1 |
| negative-control results returned | 24 | **5** | −19 |
| expectations retrieved / missed | 45 / 11 | 45 / 11 | — |
| coverage, withheld, leaks | 90.3% / 2 / 0 | 90.3% / 2 / 0 | — |

**The last two rows are the result.** Recall rose at 1, 3 and 5, was flat at 10, and the
outcome counts are identical — the same 45 expectations arrive and the same 11 are missed.
So nothing was discovered and nothing was lost: every gain is an expectation that was
already coming back, arriving higher up, because a span can no longer earn score for
matching `the`. A change that traded recall for precision would look nothing like this, and
the outcome counts are asserted beside the recall figures so the distinction cannot blur.

Most of the value is in results that stopped coming back, which a recall suite cannot see —
so the per-query counts are pinned by name: `paraphrase_password_reset_link_validity` 15→3,
`numeric_retention_ninety_days` 15→8, `negative_coffee_machine` 13→0,
`negative_pack_personal_items` 11→5, `paraphrase_who_signs_off_large_refunds` 7→2,
`exact_revenue_close_lag` 5→1.

Three things worth naming:

- **The governing policy now outranks the unofficial wiki.** Asked "how long do we keep
  customer data once an account is closed", `tokens-1` put the EMEA wiki page — an
  unauthorized override at `wiki` authority — 2nd and the retention policy 3rd. It is now
  policy 1st, wiki 4th. Retrieval does not know about authority and must not; this is a side
  effect of function words no longer contributing score. But an agent reads results in
  order, so it is pinned by a test.
- **Two equal-authority peers swapped places**, and this is the one ranking that arguably
  got worse. In `as_of_p1_target_2026_01` the 15-minute runbook was 1st and the 30-minute
  one 2nd; they are now reversed. Both come back either way so no figure moves, and neither
  order is correct — they are peers and `resolve_authority` answers `Ambiguous` for this
  exact pair. Reported rather than presented as a win.
- **The surviving negative control cannot be fixed by any stopword list.** "pack your
  personal items into the crate" still returns 5 spans, on the *content* words `personal`
  and `items`. That is the floor for this class of change, not an unfinished list.

**Negation, comparatives, quantifiers and modals are deliberately kept as terms**, though a
standard English stopword list removes all four. "a verbal approval is *not* sufficient",
"10,000 USD *or more*", "*every* refund requires manager approval" and "a manager *must*
approve" each change what the sentence commits to. Dropping negation is the best-known
defect of the off-the-shelf lists and it is not repeated here;
`tests/unit/retrieval/test_tokens.py` asserts the boundary word by word so a later "just use
the standard list" edit fails loudly.

**The version pin fired, which was its first real test.** The list is part of
`TOKENIZER_VERSION`, which is part of the scorer fingerprint the benchmark suite pins, so
the run failed until the figures were re-measured. Reprinting stale recall numbers under a
changed scorer was the specific accident the pin exists to prevent.

What the stopword list did **not** change: coverage, the unreachable count, the withheld
count, authorization-filtered recall, the leak count, `known_lexical_miss` (still 20% flat
across every *k*) and `assumption_key_lookup` (still 54.5% @10). None of those failures are
caused by function words — which is what pointed at morphology as the next thing to try.

### The plural rule (issue 26, closed)

The second benchmark-driven change, and the last cheap lexical one available. `tokens-2`
had left `known_lexical_miss` at 20% flat across every *k*, which is the signature of a
vocabulary mismatch rather than a ranking problem.

**The whole rule.** `depluralize()` in `app/retrieval/tokens.py` removes **one** trailing
`s` when four conditions hold: the token is **all letters**; it ends in `s`; it does **not**
end in `ss`, `us` or `is`; and the remaining stem is at least **3** characters. It runs
inside `tokenize()`, after folding and the stopword filter, so query and span go through
the same function and the rule cannot be applied to one side only.

**Not a stemmer, and it must not become one.** No `-ies` rule, no irregular plurals, no verb
inflection, no lemmatizer, no synonyms, no fuzzy matching. A Porter stemmer would fix more
plurals and also conflate `operate` with `operation`, which in this corpus are different
claims. The value of one rule with three exceptions is that it can be read in full.

**The guard principle: guard the endings where a trailing `s` is never a plural marker.**
No plural ends in `ss` (`access`, `address`, `business`, `class`, `loss`, and the retrieval
term `less`); `-us` words are singulars with Latin `-i` plurals (`status`, `corpus`,
`previous`); the plural of `-is` is `-es`, so `analysis`, `basis` and `axis` are never
plurals. `-as` is deliberately not guarded, because there both classes are real — `alias`
and `bias` are singulars, `areas` and `ideas` are plurals — so a guard would buy one and
sell the other.

**Validated against the full corpus, not the labels.** Over all 736 distinct tokens in the
canonical text of the 13 documents plus every benchmark query and quote, the rule changes 99
tokens and creates 40 groups where two raw tokens collapse onto one stem. **All 40 are
genuine singular/plural pairs** and not one unrelated merge. That check matters more than
the recall numbers: it is the only evidence that the guards are right rather than lucky, and
it does not depend on the 62 labelled expectations.

| | `tokens-2` | `tokens-3` | |
|---|---|---|---|
| recall@1 | 35.7% | 35.7% | — |
| recall@3 | 67.9% | **75.0%** | +7.1 |
| recall@5 | 80.4% | **89.3%** | +8.9 |
| recall@10 | 80.4% | **91.1%** | +10.7 |
| `known_lexical_miss` @3 / @5 / @10 | 20 / 20 / 20% | **60 / 80 / 80%** | +40 / +60 / +60 |
| `paraphrase` @3 / @10 | 57.1 / 71.4% | **85.7 / 100%** | +28.6 / +28.6 |
| `assumption_key_lookup` @10 | 54.5% | **63.6%** | +9.1 |
| expectations retrieved / missed | 45 / 11 | **51 / 5** | +6 / −6 |
| negative controls silent / results | 3 of 4 / 5 | 3 of 4 / 5 | — |
| coverage · withheld · leaks | 90.3% / 2 / 0 | 90.3% / 2 / 0 | — |

**A different shape of change from the stopword list**, and the difference is the honest
part. That one moved evidence up without altering what came back; this one alters what
comes back — six labelled spans that were reachable and missed are now retrieved. The
negative-control figures are **byte-identical**, which is the only evidence that recall was
not bought with noise, and the reason the suite cannot be read on recall alone (issue 33).

The six recoveries, all plural/singular mismatches: `lexical_miss_refund_singular` 3 misses
→ ranks 2, 3, 5; `paraphrase_who_signs_off_large_refunds` 2 misses → ranks 3, 5;
`key_refund_approval_threshold` training deck → rank 6.

**One clear ranking regression, and it undoes a stopword-era gain.** Asked "how long do we
keep customer data once an account is closed": `tokens-1` put the EMEA wiki page — an
unauthorized override at `wiki` authority — 2nd and the governing policy 3rd; the stopword
list made it policy 1st, wiki 4th; the plural rule reverses that to wiki 1st, policy 2nd,
because the wiki span writes "customers" and now matches the query's "customer". Every
expectation is still returned, so no recall figure moves — this is purely the order an agent
reads them in. The aggregate is not one-sided: on the refund-threshold queries the same
effect promoted the governing policy 3rd → 2nd and the training deck 5th → 3rd while the
wiki fell 2nd → 4th. Retrieval does not know about authority and must not; ordering evidence
by authority belongs to the consumer. Both directions are pinned by tests.

**Limitations, recorded rather than fixed:** `policies` does not reach `policy`; `analyses`
does not reach `analysis`; `retained` does not reach `retain`; `access` does not reach
`accesses` (the guard is one-directional); and `apis`/`kpis` are caught by the `-is` guard,
which cannot tell them from `basis` — rescuing them needs a length exception that would then
strip `axis`, trading a real protection for a marginal gain.

**The version pin fired again**, moving `tokens-2` → `tokens-3`, and the figures were
re-measured rather than reprinted.

## Phase 3 — vector retrieval (verified)

The second retrieval strategy. **No hybrid fusion, no reranking, no query expansion, no
graph retrieval, no agentic retrieval.** Full measured results in `docs/EVALUATION.md`,
"Vector retrieval, as measured"; the architecture is `docs/ARCHITECTURE.md` 6.1 and 11.4.

| module | responsibility |
|---|---|
| `app/retrieval/vectors.py` | `VectorRetriever` — cosine similarity, the filters, the ranking; a pure function of values |
| `app/infrastructure/embeddings/hashing.py` | `HashingEmbedder` — deterministic, offline, no API key. The default and the test embedder |
| `app/infrastructure/embeddings/replay.py` | `ReplayEmbedder` — recorded provider vectors; the seam a real provider fills |
| `app/application/use_cases/retrieve_evidence.py` | `execute_vector` — the join, the embedding call, the async boundary |
| `app/domain/value_objects/enums.py` | `RetrievalStrategy.VECTOR`, added with the retriever and not one slice earlier |

### The same shape as lexical, deliberately

Everything structural in the lexical retriever is repeated rather than reinvented, because
none of those properties was about BM25: the `EvidenceCandidate` join, the filter set in the
same fixed order (source type → as-of → scope), authorization applied **first** by narrowing
the index, descending score then ascending `evidence_id`, scores rounded before ranking, and
provenance carried by reference so a citation is the one the admission gate verified. Only
the scoring differs. `tests/unit/retrieval/test_vectors.py` mirrors `test_lexical.py`
assertion for assertion, because a second strategy that quietly relaxed one of those would
be a security or reproducibility regression wearing the clothes of a feature.

Two differences are structural rather than cosmetic:

**Cosine is pairwise, so there are no corpus statistics.** A span's score depends on the
query and on itself and on nothing else. That makes `excluded_by_permission`
query-independent for a structural reason rather than an arranged one, and it means a
requester's scope cannot change the score of a span they may read — which BM25 achieves only
by recomputing statistics after authorization.

**The tie break is load-bearing here, not defensive.** Cosine over floats produces near-ties
constantly; without rounding plus a total tie break the ordering would drift between
platforms.

**The vector index does not reuse the lexical tokenizer.** It has its own minimal splitter,
deliberately: corroboration counts evidence reached by *distinct* strategies (checkpoint
9.3), and two strategies that agree because they share a preprocessing stage are closer to
one strategy than to two. It is also what makes the two independently tunable.

### `RetrievalPort` does not fit, and was not changed

The port is synchronous on purpose — "a lexical scan is computation, not I/O" — but
interpreting a query for this strategy needs an embedding call. So `VectorRetriever.retrieve`
takes a `VectorQuery` (the query plus its embedding) and stays pure, and the async
port-shaped surface is `execute_vector`. A real mismatch in the contract as authored,
recorded as issue 39 rather than patched: widening the port now would be guessing at the
shape of fusion.

### The embedder, and what it does and does not prove

`HashingEmbedder` is feature hashing over a bag of words, L2-normalized, hashed with
`hashlib` and never the built-in `hash()` — which is salted per process, so an index built
with it would rank differently on every run.

**It carries no semantic knowledge at all.** Two texts are close only when they share
literal tokens. Everything measured below is therefore a measurement of the vector
*machinery* plus a term-overlap signal, and **not** of semantic retrieval. That is stated
here, in the module docstring, in the suite's pin, in the benchmark test's docstring and in
`docs/EVALUATION.md`, because reading it as a semantic result is the whole risk of shipping
a fake.

`ReplayEmbedder` is what would make it a semantic measurement: vectors recorded once from a
real provider, keyed by `sha256(text)`, committed, replayed offline. **A cache miss raises**
rather than substituting another embedder or a zero vector — either fallback would put two
embedders in one index, where the cosine between a recorded vector and a substituted one is
meaningless and the recall figure would read as a result rather than a defect. No fixture is
committed: recording one needs credentials this environment does not have, and that absence
is asserted by a test rather than left for a reader to notice.

### Results

Scored against the **same 62 expectations** as the lexical baseline. The labels are
strategy-independent — an expectation is a relevance judgment about the question — so
nothing in the suite changed, and coverage, the unreachable set and the withheld set come out
identical under both strategies (asserted, not assumed).

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **vector recall** (of the 56 reachable) | **30.4%** | **60.7%** | **76.8%** | **80.4%** |
| lexical baseline (`tokens-3`, frozen) | 35.7% | 75.0% | 89.3% | 91.1% |
| vector end-to-end recall (of all 62) | 27.4% | 54.8% | 69.4% | 72.6% |
| vector authorization-filtered recall | 50.0% | 100% | 100% | 100% |

| | vector | lexical |
|---|---|---|
| unreachable (never admitted) | **4** | 4 |
| withheld by authorization | **2** | 2 |
| withheld evidence returned anyway | **0** | 0 |
| coverage | 90.3% | 90.3% |
| expectations retrieved / missed | 46 / 10 | 51 / 5 |
| negative controls silent / results | 1 of 4 / **28** | 3 of 4 / 5 |

**The result of the slice: vector finds a strict subset of what lexical finds.** 46 found by
both, **0 by vector alone**, 5 by lexical alone, 5 by neither. A second strategy earns its
place by reaching evidence the first cannot; by that test this embedder earns nothing, so
there is no case for fusion yet — and building it now would mean building against a
measurement that says not to.

Four findings, kept apart because they have different causes:

1. **Lexical wins on preprocessing, not on ranking.** All five lexical-only finds are
   singular/plural pairs the `tokens-3` suffix rule merges; the embedder has no suffix rule
   by design. `known_lexical_miss` is 20% @10 against lexical's 80% for exactly that reason.
2. **Paraphrase recall got *worse*** — 71.4% @10 against 100% — which contradicts the reason
   this slice was scheduled. An embedder with no semantics cannot help a paraphrase, and
   measuring that is worth more than having assumed the opposite.
3. **The strategies agree wherever a filter or literal wording decides the answer.** Seven
   categories reach 100% @10 under both, which is the evidence that the machinery is wired
   correctly rather than merely returning something.
4. **Neither strategy reaches the same five expectations** — four of them the coupling set of
   `retention.window`. A second, independent strategy failing on that set is stronger
   evidence for the closed registry than one strategy failing was.

**Cosine has no natural cutoff, and it costs precision:** 28 negative-control results against
lexical's 5. A similarity floor would fix it and would be an untuned knob, so it is measured
rather than pre-empted. Note the converse too: the quietness this embedder *does* show is an
artefact of sparsity (an unrelated text lands in no shared coordinate and scores exactly
0.0), and a real dense embedder would have no zeros. The control figures here say nothing
about how a real embedder behaves.

### Verification

| command | result |
|---|---|
| `pytest -q` | **1978 passed**, 2 warnings, ~16s |
| `mypy app tests` | **success, 180 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 226 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |
| `python -m scripts.retrieval_benchmark --strategy vector` | exit 0 — no label defect, no leak, no scorer drift |

146 new tests: 66 for the retriever, 43 for the vector benchmark, 21 integration, 16 for the
replay embedder. Still **no credentials and no network**: the lexical path makes zero model
calls and the vector path makes embedding calls only, against the offline hashing embedder.

### Defects and surprises found while building the slice

1. **The version pin caught a missing pin.** The first vector run exited non-zero with "the
   suite pins no scorer for the vector strategy" — the guard working before there was
   anything to guard. Each strategy is now compared against its own pin, so a vector run
   cannot satisfy the lexical pin or be reprinted under its heading.
2. **`isinstance` against a `runtime_checkable` protocol proves nothing useful.**
   `VectorRetriever` passes `isinstance(..., RetrievalPort)` on attribute *names* alone,
   despite `retrieve` taking a different argument type. The test asserting the mismatch had
   to become behavioural. Worth recording because the same false comfort is available
   anywhere a protocol is checked structurally.
3. **A leak test can pass for the wrong reason.** The first version searched the serialized
   outcome for the denied document's title — which every fixture document shares by default,
   so it found an *allowed* document. The restricted fixture now carries a distinct title.
4. **`model_copy` skips validators**, so the first "a mis-joined candidate is refused" test
   asserted nothing. It constructs the candidate directly now.
5. **The measured result contradicted the plan.** This slice was scheduled because paraphrase
   recall was the lexical strategy's weakest category; the vector strategy made it worse and
   added no unique finds. Reported as measured rather than reframed as a success, and it is
   the reason the next step is a recorded provider rather than fusion.

#### Defects found while building the benchmark

1. **The authorization metric had no data on its first run.** Both permission-restricted
   queries asked a restricted requester a question whose only answer was closed to them, so
   the suite measured withholding and had nothing left to compute recall over — the metric
   printed a dash and looked fine. Fixed by adding
   `restricted_refund_threshold_least_privilege`: a restricted requester asking a question
   they are fully entitled to have answered, which is the case the metric was for.
2. **`excluded_by_tenant` is never charged on the production path.** The foreign-tenant
   query was authored expecting 21 spans in the tenant bucket. Measured, the trace reports
   `considered: 0`: the store enumeration takes a tenant, so a foreign tenant's index is
   empty before the retriever runs. The bucket exists for a candidate list that spans
   tenants, which the join never builds. Isolation is upstream of the retriever's own
   accounting, which is the stronger place for it — recorded rather than "fixed".
3. **The retriever's filter chain is private, so the benchmark needs its own copy** to
   explain *why* an expectation did not arrive. Rather than widening the retriever's surface
   — its authorization guarantee rests on never iterating an unreadable span — the
   diagnostic reuses the same public predicates (`is_readable_by`, `is_in_force`,
   `scope_lattice.intersects`) and a test asserts that every bucket it charges is matched by
   a non-zero count in the real trace. Divergence is caught rather than assumed.

### Verification

Executed in this environment (Python 3.14.0, Windows):

| command | result |
|---|---|
| `pytest -q` | **1978 passed**, 2 warnings, ~16s |
| `mypy app tests` | **success, 180 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 226 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** — 12 docs + 16 agent files |
| `python -m scripts.retrieval_benchmark` | exit 0 — no label defect, no leak, no scorer drift |

Test suite still runs with **no credentials and no network**. The benchmark makes **zero
model calls**: extraction is replayed, and nothing on the retrieval path can reach a
provider.

### Test inventory (retrieval slice additions)

| suite | tests |
|---|---|
| `tests/unit/retrieval/test_tokens.py` | 146 |
| `tests/unit/retrieval/test_lexical.py` | 68 |
| `tests/evaluation/test_retrieval_benchmark.py` | 47 |
| `tests/unit/evaluation/test_retrieval_metrics.py` | 43 |
| `tests/unit/retrieval/test_permission_filtering.py` | 39 |
| `tests/unit/domain/test_permissions.py` | 29 |
| `tests/unit/evaluation/test_retrieval_suite.py` | 28 |
| `tests/integration/test_lexical_retrieval.py` | 27 |
| `tests/contract/test_layering.py` (new guarded layer) | 2 |

429 new tests over the five slices — 103 for lexical retrieval, 75 for permission filtering,
111 for the benchmark, 50 for the stopword list, 90 for the plural rule; 1832 total.

Both tokenizer changes are tested the same way, and it is worth saying why: **the boundary
of a rule gets more tests than the rule.** 25 tests assert which words are *not* stopwords
(negation, comparatives, quantifiers, modals — each removed by a standard English list, each
changing what an organizational sentence commits to), and 46 assert which words the plural
rule must *not* touch (`-ss`/`-us`/`-is` singulars, short words, quantities, tokens with
digits, and a vocabulary-collision check over real corpus terms). A wrong merge is worse
than a missed one: a missed plural costs recall on one query, while conflating `access` with
something else changes what a citation is about. Negative-heavy again, and
deliberately so: a retriever that
returns everything passes every positive test and destroys precision downstream. The tests
carrying the most weight assert that a span is **not** returned — a foreign tenant, a
disjoint scope, a version out of force, a source type nobody has, a query sharing no term —
and that the trace says which of those happened.

`tests/integration/test_lexical_retrieval.py` re-asserts span verification on the retrieval
path: every returned result's offsets must still recover its quoted text from the canonical
version. Retrieval is the path that hands evidence to an agent, so a result whose offsets no
longer verify would be a citation the reader cannot check.

### The tests were checked against mutations

Ten plausible wrong implementations were introduced one at a time and the suite had to fail
on each: ignoring the tenant partition, pooling corpus statistics across tenants, scope
equality instead of intersection, letting an unscoped span pass a scope filter, dropping the
as-of filter, dropping the source-type filter, dropping the tie break, never truncating to
`top_k`, indexing a duplicated span twice, and tokenizing the query differently from the
span. **All ten were caught.**

Ten more were introduced against permission filtering: authorization doing nothing,
statistics computed before authorization instead of after, authorization moved out of the
narrowing step and into the content filters, authorization applied after scoring, the outcome
stamping a constant hash, an unlabelled scope reading everything, every label required
instead of any, the hash ignoring the grants, case-sensitive label matching, and the public
candidate accessor not filtering. **All ten were caught.**

The first run of that exercise caught only nine, and the survivor was a defect in a *test*
rather than in the code: the structural assertion that authorization is not decided inside
``_first_failing_filter`` used ``"permission" not in co_names``, and exact membership does
not match ``"permission_scope"``. It is a substring scan now, and the mutation that survived
is one of the two late-check variants above. Worth recording because it is the failure mode
mutation testing exists to find - a test that reads correctly and asserts nothing.

### Defects found while building the slice

1. **`SourceStore` could not enumerate a corpus.** `get_evidence(evidence_id)` was the only
   read, which forces a caller to already know the ids it wants — usable for verification,
   useless for retrieval. `evidence_for_tenant` was added to the port with tenant as a
   parameter, so an unscoped enumeration of every span in the system is not a query anyone
   can write by accident.
2. **A plural does not match its singular.** `refund` missed `Refunds`, because there was no
   suffix rule. Found by a test that assumed it would match, and pinned by a test that named
   the behaviour rather than asserting it away — which is what made it available to be
   measured (issue 26) and then fixed in `tokens-3`, two slices later. The sequence is the
   point: name the defect, price it, then change one thing.
3. **A per-candidate tenant check would have been the weaker design.** The first draft
   filtered tenant alongside source type. Partitioning the index per tenant is stronger — a
   span that is not in the loop cannot be returned by a filter-order bug — and it is also
   what makes document frequency per tenant rather than global.

### Not implemented (deliberately)

- **Reranking, query expansion, weighted or learned fusion.** Hybrid fusion has since been
  built (see "Phase 3 - hybrid retrieval") and measured worse than lexical alone at every
  cut-off but the deepest. Weighting the inputs would improve that figure and is deliberately
  not done: it would be fitting a fusion parameter to compensate for an embedder nobody has
  measured. Reranking is a further stage on top of a fusion that is not yet earning its
  place.
- **Analysis-level scope keying.** Retrieval now records the scope hash;
  `ARCHITECTURE.md` 14.1 also requires a stored *analysis* to be served only to a matching
  scope, and there is no analysis store yet to enforce it against.
- **An authorization audit log.** `docs/SECURITY.md` 7 requires source access to be audited
  in production. Retrieval logs nothing per access today, and inventing a log format before
  there is a request identity to attribute it to would be guesswork.
- **Anything ACL-shaped beyond a label match**: no principal/group resolution, no
  hierarchical or negative labels, no row- or field-level redaction. A label match is what
  `SourceDocument.permissions` can express, and the rest needs an identity model.
- **Any parameter fitted to the benchmark.** `k1`/`b` are untouched, and a stemmer, a
  lemmatizer and a synonym table are all still absent. Two changes have been made because of
  the suite — the stopword list and one guarded plural rule — each landing alone, after the
  labels, and each specified by a linguistic principle rather than read off the suite's
  failures: a closed-class function vocabulary, and "guard the endings where a trailing `s`
  is never a plural marker". A word list assembled from the 24 noisy results, or a guard
  list assembled from this corpus's `-as` words, would each have scored better and meant
  less. The plural rule was additionally validated against the full 736-token corpus
  vocabulary rather than only the 62 labelled expectations, so its safety claim does not
  rest on the labels at all.
- **Precision and ranking-quality metrics.** The labels are recall labels; a span that came
  back unlabelled is not thereby wrong, and scoring it as a false positive would punish the
  behaviour the product wants. MRR and nDCG were considered and left out: with 62
  expectations they would move on single rank swaps and read as signal.
- **Metadata-only retrieval.** Every query carries text; a filters-only query is a different
  contract and belongs with the QueryPlan.
- **A QueryPlan, and multi-strategy merge or deduplication.** One strategy cannot be fused
  with itself.
- **Caching.** Candidates are rebuilt per call. Honest for 21 spans and wrong for a real
  corpus; a projection keyed on corpus version is a measurable change rather than a guess.
- **`EvidenceGap` production.** `RetrievalTrace` is the negative provenance a gap needs, but
  the producer belongs with the agent layer that asks the unanswered question.

### New unresolved issues

Continuing the numbering of the Phase 2 list above.

25. **Only spans that produced an admitted binding are retrievable** — now measured, and it
    is the largest single limitation in the slice. An `EvidenceItem` exists exactly where
    the admission gate admitted something, so the retrievable corpus is 21 spans over 13
    documents rather than the whole text. Every other sentence is parsed, chunked and
    stored, and unreachable. **Benchmark: coverage 90.3%, four labelled expectations
    unreachable**, and one of them is a material policy exception a refund question needs.
    The benchmark also split the cause in two, which was not visible before: three of the
    four bind no registered key at all, so *registry coverage* is itself a retrieval
    ceiling. The fix remains a decision, not a bug fix: either chunks become retrievable
    alongside spans, or an `EvidenceItem` is minted per chunk at ingestion. Both change what
    a citation means, so neither was done quietly.
26. ~~**No stemming, so a plural misses its singular.**~~ **Closed** in `tokens-3` by a
    guarded plural-`s` rule, and the feared trade did not materialize: `access`, `analysis`,
    `status` and their kind are protected by three ending guards, and over the full 736-token
    corpus vocabulary the rule produced 40 merge groups of which **all 40 are genuine
    singular/plural pairs**. `known_lexical_miss` went 20% flat → 60 / 80 / 80% at
    @3 / @5 / @10 and overall recall@10 went 80.4% → 91.1%, with negative-control noise
    unchanged. Full write-up in "The plural rule (issue 26, closed)". **What remains is
    narrower than the issue claimed:** `-ies` (`policies`/`policy`), irregular plurals
    (`analyses`/`analysis`), verb inflection (`retained`/`retain`) and the `-is` guard
    catching `apis`. Each is a *missed* merge rather than a wrong one, and each would need a
    second rule nobody has measured — so they stay open as limitations, not as work.
27. **BM25 parameters are untuned** — still, on purpose. `k1=1.2` / `b=0.75` are the
    standard defaults, chosen because they are conventional and not because they were
    measured here. Nothing was fitted to the new suite, because tuning in the change that
    authored the labels produces a meaningless number. They live in a `LexicalConfig` whose
    fingerprint is provenance, so retuning is a config change with a visible version change
    — and the benchmark pins the fingerprint, so retuning without re-measuring now fails.
28. **The candidate join is rebuilt per query.** It is O(corpus) per call, which is fine for
    21 spans and wrong for a real one. Deliberate: the alternative is a cache, and a cache
    key is a design decision that wants a measurement behind it.
29. **An unlabelled document is readable by the whole tenant.** This is the one fail-open
    edge in the authorization model, and it is deliberate: `permissions` is a statement about
    who may read a document, and an empty statement restricts nothing. The consequence is
    that a connector which forgets to populate `permissions` publishes a restricted document
    tenant-wide. Fail-closed on an empty label set was considered and rejected for the MVP
    because it makes an unlabelled corpus — which is what the sample corpus is — entirely
    unretrievable, so the safe default would be indistinguishable from a broken one. The
    right fix is per-source-type registration policy, and it belongs with the connector work
    rather than here.
30. **The whole sample corpus is unlabelled**, so both the integration test and the
    benchmark apply an ACL through the store port rather than measuring against a corpus
    that ships with ACLs. The benchmark improves on the test — it declares the overlay in
    the label file, restricts the most-referenced document in the corpus so withholding is
    visible in the numbers, and reports authorization-filtered recall separately — but it is
    still **one label on one document**. Recall under a realistic label distribution, where
    a requester's readable slice is a fraction of the corpus rather than 19 spans of 21, is
    still unmeasured, and the honest statement of the result is "authorization cost nothing
    here" rather than "authorization costs nothing".
31. **`excluded_by_permission` is a real, if small, side channel.** It tells a requester how
    many spans in their tenant are closed to them. That is the reported behaviour the slice
    was asked for, and it is deliberately query-independent so it cannot become a content
    oracle — but on a corpus where restricted document *counts* are themselves sensitive, the
    field would need suppressing or bucketing. Recorded rather than pre-solved: no benchmark
    scenario needs it, and a threshold chosen now would be a guess.

Added by the benchmark slice:

32. ~~**There is no stopword list, and it costs precision on ordinary questions.**~~
    **Closed.** Fixed in `tokens-2` and measured: negative-control results fell from 24 to
    5, three of four controls are now silent, and recall rose 1.8 / 5.4 / 1.8 points at
    @1 / @3 / @5 with the same 45 expectations retrieved. Full before/after in "The stopword
    list (issue 32, closed)" above. **What remains is not a stopword problem:** the one
    control still returning results does so on the content words `personal` and `items`, so
    the residual noise is content-word ambiguity and belongs to a later strategy. Precision
    is still not scored by the suite (issue 33), so the 24→5 improvement is visible only in
    per-query result counts.
33. **The benchmark measures recall and says nothing about precision.** A returned span that
    no label mentions is not scored, in either direction. That is the right choice for a
    recall suite and it means `numeric_retention_ninety_days` returning 15 of 21 spans is
    invisible in every headline number. The negative controls are the only precision-adjacent
    signal, and three of them is not a precision measurement.
34. **62 expectations cannot separate two retrieval strategies.** They can show that one
    fails on a shape of query, which is what they do. When vector retrieval lands, the
    comparison needs either a larger suite or a separately authored held-out set — the
    caution `docs/EVALUATION.md` already states for the consequence benchmark applies here
    unchanged, and the suite was authored by the same people who wrote the retriever.
35. **The benchmark keeps its own copy of the retriever's filter order.** Needed to explain
    *why* an expectation did not arrive, and written rather than imported because the
    retriever's chain is private on purpose — its authorization guarantee rests on never
    iterating an unreadable span. The copy reuses the same public predicates and a test
    asserts every bucket it charges is matched in the real trace, so divergence fails loudly;
    but it is still a second statement of one rule, and a filter added to the retriever
    without a matching benchmark update would be classified as a plain retrieval miss.

Added by the vector slice:

36. **No embedding fixture from a real provider ships, so the semantic question is
    unanswered — and the attempt to answer it was blocked, not skipped.** Every vector
    figure was measured with `HashingEmbedder`, which is feature hashing over a bag of words
    and knows no semantics. The machinery is now complete and proven on this side:
    `scripts/record_embeddings.py` performs the one live call, `HttpEmbeddingProvider`
    speaks the wire format, `ReplayEmbedder` reads a committed fixture keyed by
    `sha256(text)` and raises on a miss, and 30 tests drive the whole recording → replay →
    benchmark path through `httpx.MockTransport`. **What is missing is only the credential.**
    `python -m scripts.record_embeddings --check` reports five unset values, and no local
    provider is listening either (ports 11434, 1234, 8080, 8000 all refused). A synthetic
    fixture was deliberately **not** committed: one labelled with a provider that never ran
    would make every downstream recall figure a fabrication, which is worse than an open
    issue. **No semantic claim about vector retrieval in this repository is supported by
    anything.**
37. **The vector index is rebuilt and re-embedded on every query.** The largest practical
    issue in the slice. It is O(corpus) embedding calls per query, which is fine for 21 spans
    and ruinous for a real corpus, where it would dominate both latency and cost — and it is
    a *cost-discipline* violation rather than merely slow, since `CLAUDE.md` requires
    retrieval before reasoning to be cheap. The fix is a stored vector index keyed on
    (corpus version, embedder) and it belongs with pgvector; it was not done here because a
    correctness-first implementation is the one whose recall figure can be trusted.
38. **There is no similarity floor, and cosine has no natural cutoff.** 28 negative-control
    results against lexical's 5. Deliberately not fixed: a threshold is an untuned knob, and
    the measurement that would justify one needs a real embedder (issue 36). Recorded with a
    before-figure so a floor can be judged rather than guessed.
39. **`RetrievalPort` does not model a strategy that needs I/O to interpret a query.** The
    port is synchronous by design; embedding a query is I/O. So `VectorRetriever` takes an
    already-embedded `VectorQuery` and the port-shaped surface is `execute_vector` on the use
    case. The port was left alone deliberately — widening it now would be guessing at the
    shape of fusion — but the consequence is that "every strategy implements one interface"
    is currently not true, and a `QueryPlan` cannot treat the two uniformly until it is.
40. **The two strategies cannot yet corroborate each other.** Checkpoint 9.3 counts evidence
    reached by distinct strategies, and the groundwork is in place (`RetrievalStrategy` has
    two members, and the vector index deliberately does not share the lexical tokenizer so
    agreement between them is evidence rather than an artefact). But nothing consumes it:
    there is no `EvidenceBundle` producer and no confidence scorer wired to strategy counts.

Added by the adapter-foundation slice:

41. **`HttpEmbeddingProvider` has never made a live call, so it is untested integration
    code.** The wire *mapping* is verified through `httpx.MockTransport` — request shape,
    bearer header, response parsing, provider-index ordering, four status-code translations
    and five malformed-response shapes. The wire *contract* is not: a real provider may name
    its fields differently, page its responses, cap batch sizes lower than configured, or
    return a width its documentation does not mention. The first real run should be treated
    as a debugging exercise, not a measurement.
42. **Embeddings have no retry, backoff or caching.** Completions get all three from the
    gateway; embeddings reach their provider directly, so a rate limit or a 5xx propagates
    to the caller as a failed run. Acceptable while the only enabled adapters are offline,
    and the first thing a real provider will expose. Sharing the gateway's retry machinery
    is the likely fix, and it was not attempted here because it is a refactor of the
    gateway rather than of embeddings.
45. **Hybrid fusion is implemented but is not the default, and should not become one on
    this evidence.** RRF at `k=60` over the current two inputs measures 33.9 / 66.1 / 83.9 /
    91.1 against lexical's 35.9-point-better @3, and it inherits the vector input's
    negative-control noise wholesale (28 results against 5). The fix is not a fusion
    parameter - it is a vector input worth fusing (issue 36). Until then `execute` remains
    what a caller should use, and `execute_hybrid` exists to be re-measured the moment the
    embedder changes.
46. **RRF gives a positive contribution to every returned result, however weak.** There is no
    rank cut-off and no per-strategy weight, which is why 104 vector-only results can push
    relevant lexical ones down. Both are single-parameter changes and both are deliberately
    unmade: choosing either now would tune fusion against an embedder that carries no
    semantics, and the measurement would describe the tuning.
43. **Nothing consumes `EmbeddingMetadata` yet beyond `name`.** The adapter records
    provider, model, dimensions and adapter version; `VectorRetriever` folds only the
    derived `name` into `scorer_version`. Enough to make two indexes distinguishable, and
    short of what run provenance will want when an analysis has to record which embedder
    produced the evidence it reasoned over.
44. **The asymmetric retrieval task type cannot be expressed.** Gemini - and most embedding
    providers - distinguish embedding a *document* from embedding a *query*, and for
    retrieval Google recommends `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` rather than one
    symmetric type. `EmbeddingPort.embed` takes an undifferentiated list of texts, and the
    recorder embeds spans and queries in one pass, so the distinction has nowhere to live.
    The Gemini adapter sends one symmetric `SEMANTIC_SIMILARITY`. This is the most likely
    single reason a real-embedder measurement would understate what the provider can do, and
    fixing it means widening a domain contract - deliberately not done as part of an adapter
    slice.
47. **Nothing consumes an `EvidenceGap` yet.** The producer is built and tested, but no
    report, no API response and no analysis record reads a `GapRecord`. Until something does,
    "uncertainty is a first-class output" (`CLAUDE.md` 7) is true of the data model and
    unproven of the product. The consumer is the report composer, which is not built.
61. **The orchestrator holds no persistence, so an analysis exists only in memory.** Every
    record it produces - consequences, decisions, scores, the graph - is returned and dropped.
    That is fine for evaluation and impossible for a product: `FindingDisposition` needs a
    stored `consequence_id` to attach to, and the reproducibility diff (checkpoint 13.21)
    needs two runs to compare. The stores are all in-memory implementations of ports that
    already exist, so this is wiring rather than design.
62. **Confidence inputs are measured from retrieval only, so `explicitness` is never reached
    end to end.** The orchestrator passes `bindings={}` to the judge, because the bindings it
    holds are keyed on assumption id while the confidence assembler wants the candidate's own
    records - a small join nobody has written. Every score in the measured runs therefore
    reports `explicitness` as unmeasured.
59. **RESOLVED by the end-to-end orchestrator.** Originally: Every stage is verified against its own
    inputs and no run passes a change through ingestion, coupling, retrieval, the four agentic
    stages, arbitration and projection in one go. The interfaces between stages have only been
    exercised by test fixtures, which is the largest remaining risk in the repository - and it
    blocks the two quality bars most easily failed while every slice looks green: ablation
    honesty and quiet-on-quiet-changes.
60. **The projector's gap annotation matches on an exact label, which is fragile.**
    `EvidenceGap.partition` is free text, so a gap only annotates a node when its partition
    string equals a node label exactly. Deliberate - interpreting the prose would put a
    judgement in the projector - but it means a gap phrased slightly differently annotates
    nothing and is reported as unmatched. The real fix is a typed node reference on the gap,
    which is a domain change and therefore a slice of its own.
56. **High strategy agreement is scored as reliability and currently means redundancy.**
    Measured at 0.875-1.000 on the sample corpus, because `HashingEmbedder` is feature hashing
    over the same tokens the lexical scorer uses - so the two strategies confirm each other by
    construction. The formula treats that as a positive (checkpoint 9.1, retrieval reliability,
    +). It is not wrong as a rule; it is uninformative against *this* embedder. The fix is a
    real embedder (issue 36) rather than a reweighting, because re-tuning against one
    uncalibrated embedder would be tuning to an artefact.
57. **Corroboration saturates at three source versions, so it stops discriminating early.**
    A broad query over six or seven documents scores 1.0, which is most queries on this corpus.
    Whether the saturation point is right is a calibration question with no data behind it yet,
    and it is the single most consequential unlabelled constant in the confidence function.
58. **`alias_mediated` is never measured, because nothing records it.** The factor exists on
    `ConfidenceInputs` and the assembler passes a caller flag through, but the collision matrix
    does not record whether a coupling was reached through an approved `KeyAlias`. Until it
    does, an alias-mediated Class A finding scores as though the key matched directly - which
    overstates it by the 0.85 factor the function would otherwise apply.
53. **The confidence weights are uncalibrated, and the number is not a probability.** The
    function encodes the orderings checkpoint 9.1 states and nothing more. Until there is
    labelled outcome data, `confidence` is an ordering aid - which is why
    `CONFIDENCE_FN_VERSION` exists, and why checkpoint 14's ban on a probability-by-another-name
    field still holds. The risk to watch is a reader treating 0.62 as a 62% chance, which is a
    report-surface problem as much as a scoring one.
54. **RESOLVED by the confidence-input assembler.** The eight factors are measured from
    pipeline data and the judge assembles them per candidate; what remains unmeasured is
    reported by `FactorCoverage` rather than defaulted silently. Two factors are still
    unreachable in practice - see issues 56 and 58. Originally: `JudgeConsequencesUseCase` accepts
    `confidence_inputs` per consequence and defaults to an empty input, so every score is
    currently an honest zero unless a caller supplies them. Computing authority, freshness,
    corroborating versions and retrieval agreement needs the store and the retrieval trace,
    which is a slice of its own - and until it lands, `confidence` is wired rather than
    measured.
73. **The specialist slices add no recall on this corpus.** They partition the spans the
    main retrieval already returned, because `top_k=10` over 13 documents recovers everything
    lexically matching. The union mechanism that would let a hypothesis cite a span only a
    specialist reached is built, typed and unexercised. It needs either a corpus large enough
    to truncate or a semantic embedder - so it is blocked behind issue 36, and until then the
    fan-out's value is per-domain reading rather than wider evidence.
74. ~~**The historical analyst cannot supply the Class B anchor it exists for.**~~ **Closed**
    by `app/specialists/analogues.py`. It needed no store: `HistoricalEvent` already existed
    in the domain, and the event id is *derived* from the cited spans, so an analogue is
    re-derivable rather than looked up. The prediction above was wrong in an instructive way -
    it assumed the missing piece was persistence, when the missing piece was an identity
    recipe. It also missed that the gate was open in the **other** direction: the hunter's
    `historical_analogue_id` was never validated, so an invented well-formed `he_...` would
    have lifted a Class B finding above `speculative` with nothing behind it. Both halves are
    closed and a Class B finding now reaches `likely` on INC-4471's recorded outcome.
75. **Seven lexical retrievals per analysis, whether or not a specialist is invoked.** Slices
    are retrieved for all seven so the selection reason is a fact rather than a guess; only
    the non-empty ones cost a provider call. Cheap in-process today and worth pre-filtering on
    the corpus's source-type distribution if it stops being.
70. **The persistence layer has not been run against a real PostgreSQL server.** No
    Postgres, no Docker and nothing listening on 5432 in the environment it was built in, so
    every persistence suite ran on SQLite - the same repository code, ORM models and Alembic
    revision, with `JSONB` variant-ed to `JSON`. What that covers is stated in
    `tests/fixtures/databases.py` and what it does not is JSONB indexing, Postgres type
    coercion and concurrent-write behaviour. The offline `alembic upgrade head --sql` run
    proves the DDL compiles for the production dialect with 12 JSONB payload columns, which is
    the most a serverless environment can prove. `CCE_TEST_DATABASE_URL` runs the identical
    tests against the real thing.
71. **No cross-store transaction by default, and the ports cannot express one.** Each store
    call commits. `Database.unit_of_work()` covers the one case that needed it (the three
    coupling-projection writes) but a caller has to know to open it, and nothing in the port
    signatures says so. Acceptable because every port method is a complete operation; worth
    revisiting if a use case ever needs two stores to move together.
72. **RESOLVED by the composition root.** `app/composition.py` constructs the graph from
    configuration and `python -m app.composition` is the entry point. What is still absent is
    a *caller* - no route, no worker and no schedule decides when an analysis runs. Originally:
    Nothing in the application is wired to the SQL stores yet.
66. **Two of the four blind-spot partition slices are empty on this corpus.**
    `low_ranked` needs retrieval to return more spans than the hunter's 24-span budget, which
    6 results cannot do; `unqueried_source_type` needs the change's own wording to hook a
    different source type, which it does not. Both are honest zeroes and both would fill on a
    larger corpus - but until then the stage's real yield comes from one slice
    (`single_holder`), and a yield figure that averaged over four would overstate it.
67. **RESOLVED by the blind-spot ordering fix.** `blind_spots` now runs *before* `challenge`,
    the challenger receives the union of both hunts' hypotheses, and its evidence map includes
    the partition spans so a blind-spot candidate has something citable. Originally:
    Blind-spot findings reach the judge unchallenged.
68. **The report composer does not write the report's prose.** The executive summary is
    composed deterministically by `summarize(...)`; the agent's narration is advisory, gated on
    identifiers, and carried beside the report. That is the only way "introduces no new factual
    claim" can be a property rather than an instruction. The agent definition has been rewritten
    to describe the role it actually has (surface narrowing and ordering, version 2 -> 3), so
    the prompt and the runtime now agree - but this is a real narrowing of the capability the
    original definition described, and it is the open product question in this slice: a reader
    may well want generated prose, and if so the gate has to get stronger than an identifier
    check before it can be offered.
69. **`already_known_rate` is measured over re-encountered findings, not over a review
    queue.** The population is "findings a reviewer has already ruled on for the same
    `(assumption_key, source version)`", which answers *how often does the system re-deliver
    something a reviewer said they knew*. That is the useful question and it is not the same as
    the share of a reviewer's verdicts that were `already_known`. Both are worth having; only
    the first is implemented.
65. **RESOLVED by the core-intelligence slice** - `AnalyzeChangeUseCase` now persists its own
    snapshot and reads standing verdicts in the report stage. Originally: The orchestrator does
    not save its own result, and does not read suppressions.**
    `AnalysisStore` exists and `snapshot(...)` projects a result into it, but
    `AnalyzeChangeUseCase` neither writes nor consults it - a caller must call `snapshot` and
    `put` itself, and nothing filters a new analysis against standing suppressions. Both are
    now one wiring change rather than a design question.
63. **RESOLVED by the analysis persistence slice.** Originally: The disposition path has no
    analysis to validate against, so the caller supplies one.** `RecordDispositionUseCase.record` takes the consequences the reviewer was shown and
    checks membership against them. That is honest but weak: a caller passing the wrong list
    would have its verdict accepted. The fix is issue 61 - an analysis store - and until it
    lands the cross-analysis check relies on the disposition store's own index, which only
    catches a consequence some *other* analysis already has a verdict on.
64. **RESOLVED by the analysis persistence slice** - readable via `is_suppressed(...)`,
    scoped to the stored finding's affected source version. Originally: Persistent suppression
    is recorded and nothing consumes it.** A reviewer can ask for a
    finding to stay suppressed until the source is revised, and the scope and the expiry rule
    are both implemented and tested - but the orchestrator does not read them, so the next
    analysis re-reports the finding anyway. Wiring it needs the analysis to know the affected
    source version id, which the `Consequence` does not currently carry (it holds
    `affected_entity`, a title).
55. **RESOLVED by the FindingDisposition slice.** Originally: The entity
    exists and `SUPPRESSING_DISPOSITIONS` is defined, but nothing records a reviewer's verdict
    and nothing suppresses on one. `already_known` is the key product signal (`IDEA.md` 21.2)
    and it is currently unmeasurable.
51. **RESOLVED by the consequence-judge slice.** `consequence_id` is minted by the
    challenge layer and the judge now inherits it on all three records; an orphaned
    `ChallengeResult` raises at assembly rather than attaching to nothing. Originally:
    `ChallengeResult` and `FalsificationTest` are keyed on `consequence_id`, because a
    challenge attaches to the arbitrated finding - but the judge is not built, so
    `app/challenge/candidates.py` mints the id once per candidate. The judge slice must
    construct its `Consequence` with the id already on the `ChallengeTarget` rather than
    minting a second one: a challenge pointing at an id no consequence carries would be an
    orphaned proof obligation, and the failure would be silent.
52. **PARTLY RESOLVED.** The judge consumes the challenge verdicts, the contradiction
    ratings, the Class B hypotheses, the Class A findings and the gap ids. What still consumes
    nothing is the *judge's own* output - no report, API response or graph projection reads a
    `Consequence`. Originally: Same shape as issues 47 and 50. The
    verdicts, the falsification tests and the refusal accounting are produced and tested,
    and no report, API response or scorer reads them. The judge is the consumer.
49. **The contradiction-hunter's inferred half is unbuilt, and requirement 10 is why.** The
    agent definition's second job - contradictions no key comparison can express - needs a
    schema of its own *and* a rule that keeps "a corpus with no deterministic divergence
    produces no contradiction findings" true once the agent can propose one. The obvious
    rule is that an inferred contradiction is a Class B hypothesis with a mechanism and a
    citation obligation, never a peer conflict; that is a slice, not an addition.
50. **Nothing consumes a `ContradictionReport` yet.** Same shape as issue 47: the findings,
    the ratings and the suppressed divergences are produced and tested, and no report or API
    response reads them. The judge and the report composer are the consumers, and neither is
    built.
48. **A gap's `would_be_resolved_by` is caller-supplied prose.** The default is honest but
    generic. What would actually resolve a gap is usually a *source type* and a *scope* -
    "a regional retention policy for region:eu" - which is machine-checkable against the
    corpus and could therefore be re-evaluated automatically after the next ingest. It is
    prose today, so a resolved gap has to be noticed by a person.

## Phase 3 — the embedding adapter foundation (verified)

Foundation only, and deliberately small. It makes a **real** embedding provider possible
without changing `VectorRetriever`, the lexical retriever, permission semantics, the domain
contracts or the benchmark — and it stops short of calling one. Architecture in
`docs/ARCHITECTURE.md` 11.4.1.

| module | responsibility |
|---|---|
| `app/infrastructure/embeddings/base.py` | `EmbeddingMetadata`, `validate_vectors`, `batches`, `EmbeddingAdapter` — what an adapter owes, implemented once |
| `app/infrastructure/embeddings/http.py` | `HttpEmbeddingProvider` — OpenAI-compatible `POST /embeddings` over `httpx`. **Not enabled, never called live** |
| `app/infrastructure/embeddings/registry.py` | `EmbeddingRegistry` + factories — selection by configuration, enablement by decision |
| `app/config/settings.py` | the `CCE_EMBEDDING_*` configuration path; model identifier and credential with no defaults |
| `.env.example` | the same, documented for an operator |

### What the base class owes, and why it is not per-adapter

`EmbeddingPort` is three members wide and describes what a *caller* needs. Three obligations
sit outside it, and each is a way a remote provider produces a plausible wrong answer rather
than an error:

- **dimension validation, per vector.** `VectorRetriever` already refuses a mis-sized vector
  at index construction — the right last line of defence, but a late one: by then the batch
  is paid for and the provider's answer has been trusted. A mistyped model identifier is
  caught here instead, with the index of the vector that was wrong.
- **normalized-vector validation, where an adapter declares it.** Some providers return unit
  vectors, some do not, and the ones that do occasionally stop. True cosine survives either,
  but a *ranking* measured under one convention is not comparable with one measured under the
  other.
- **deterministic, order-preserving batching.** A remote provider has a request-size limit.
  If the caller splits the call, every caller reimplements it and one will reassemble out of
  order — and the vector index pairs spans to vectors positionally, so that failure is
  undetectable downstream.

Two cheaper checks are worth naming: a **non-finite component** is refused (a NaN compares
false against everything and would poison every cosine it enters), and an **empty request
makes no call at all**.

`EmbeddingMetadata` records provider, model, dimensions and adapter version, and derives the
`name` the retriever folds into `scorer_version`. It lives on the **adapter, not on the
port** — retrieval needs a name and a width, and widening a domain contract to carry
vendor-shaped provenance would serve no caller. The domain checkpoint is untouched.

### Enablement is a decision, not a consequence

`EmbeddingRegistry` mirrors `ProviderRegistry`. Membership in `EMBEDDING_FACTORIES` means CCE
knows how to build an adapter; using one additionally requires it in
`CCE_EMBEDDING_ENABLED_PROVIDERS`, which ships as `("hashing", "replay")` — the two that
reach no network. **`http` is absent by default**, because a remote embedder sends corpus
text, including spans from documents under an ACL, to a third party. Activating one because
a key appeared in the environment would widen the data-egress surface without anyone
deciding to.

Everything the remote adapter needs comes from configuration or the environment with **no
defaults**: `CCE_MODEL_EMBEDDING` (a vector index has to name the model that produced it),
`CCE_EMBEDDING_API_KEY` (environment only, never a literal), `CCE_EMBEDDING_BASE_URL`, and
`CCE_EMBEDDING_DIMENSIONS` (validated per vector — nothing in a response proves which model
answered). Each missing value raises `ProviderUnavailable` at selection time, never mid-run.

### What is deliberately not built

- **No live provider call.** The wire *mapping* is verified through `httpx.MockTransport`;
  the wire *contract* is not. A real provider may name its fields differently, page its
  responses, or return a width its documentation does not mention. This is untested
  integration code until one run happens against a real endpoint — issue 41, not a claim.
- **No benchmark run.** The vector benchmark still measures `hashing-1` at 256 dimensions,
  and every figure in `docs/EVALUATION.md` is unchanged and still says so.
- **No retry or backoff** (a gateway concern embeddings do not yet share), **no caching**,
  **no hybrid fusion**, **no reranking**.
- **`HashingEmbedder` and `ReplayEmbedder` were not retrofitted onto the new base.**
  Validation belongs where the data is untrusted, and their output is computed locally and
  already tested; putting them behind the base would have risked `HashingEmbedder.name` —
  and with it `scorer_version` and every pinned benchmark figure — for no guarantee gained.
  A test asserts the identity is still `hashing-1` at 256 dimensions.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2049 passed**, 2 warnings, ~18s |
| `mypy app tests` | **success, 184 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 230 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

71 new tests, all offline: dimension and normalization validation, batching and reassembly
order, provider-index ordering, the four error mappings, five malformed-response shapes,
client lifetime, registry enablement, and that the vector benchmark's embedder identity is
untouched. **No credentials and no network** — `httpx.MockTransport` throughout, and a test
asserts no embedding credential is configured under test settings.

## Phase 3 — the real embedding experiment (attempted, blocked on a credential)

**Outcome: no measurement.** The live embedding call could not be made in this environment,
so there are no real-embedding recall figures, and none are quoted anywhere. What was
delivered is the mechanism plus a precise diagnosis of the blocker. Both frozen baselines
are byte-identical to before this slice.

### The diagnosis

`python -m scripts.record_embeddings --check` performs the preflight and reports every gap
at once rather than one failure per run:

| check | state |
|---|---|
| `CCE_EMBEDDING_PROVIDER` | `hashing` — reaches no provider |
| `CCE_MODEL_EMBEDDING` | unset |
| `CCE_EMBEDDING_BASE_URL` | unset |
| `CCE_EMBEDDING_API_KEY` | **absent** |
| `CCE_EMBEDDING_DIMENSIONS` | unset |
| `.env` | does not exist; no `CCE_*` variable is set in the environment |
| local provider | nothing listening on 11434 (Ollama), 1234, 8080 or 8000 |

There is no credential to configure and no local endpoint to fall back to, so this is not a
configuration error that could be corrected — it is a missing input. The permitted "minimum
correction" does not extend to inventing a key.

**A synthetic fixture was deliberately not committed.** It would have made the pipeline look
finished and every downstream figure a fabrication, and it is exactly what `ReplayEmbedder`
raises to prevent. The absence is asserted by a test
(`TestNoFixtureIsCommitted`) rather than left for a reader to notice.

### What was delivered instead

| module | responsibility |
|---|---|
| `scripts/record_embeddings.py` | the one live call: preflight, collect, embed, validate, write. `--check` and `--dry-run` reach no provider |
| `app/infrastructure/embeddings/replay.py` | fixture provenance extended: `adapter_version`, `expects_unit_vectors`, `recorded_at`, `recorded_texts` |
| `app/domain/value_objects/identifiers.py` | `VersionLabel` widened 64 → 128 (see below) |
| `tests/integration/test_embedding_recording.py` | 30 tests driving recording → replay → benchmark through `httpx.MockTransport` |

The recorder embeds the **21 admitted span texts and the 35 benchmark query texts** — 56
texts, **51 distinct**, because the version-currency pairs ask one question at two dates and
the permission pair asks one question under two scopes. Texts are deduplicated and sorted by
their own digest, so two recordings send the same list in the same order and a diff of two
fixtures is meaningful. It sends span texts and not whole documents: this call ships corpus
text to a third party, so it ships the minimum that makes the benchmark runnable.

Validation is layered and every layer is tested: the adapter checks width, finiteness and
(where declared) unit length per batch; the recorder re-checks over the assembled result and
additionally proves **coverage** — every requested text has a distinct key, so a provider
that dropped one or returned a duplicate cannot produce a fixture. A wrong width or a
provider failure writes **nothing**: a partial recording is worse than none because it looks
complete.

**The API key never leaves the environment.** It builds one `Authorization` header. A test
serializes the written fixture and asserts the secret, `api_key` and `authorization` appear
nowhere in it.

### Two real findings from the attempt

1. **`VersionLabel`'s 64-character cap was too small for a real embedder's fingerprint.**
   A replayed remote embedder reports
   `vector-cosine-1:embed-replay-1:openai-compatible/<model>@<date>:d=<width>`, which
   exceeds 64 for most real model names — the benchmark run failed on a Pydantic validation
   error before producing a single number. 64 was sized when a fingerprint had two
   components. Widened to 128, which is the minimum correction: the alternative was
   truncating provenance, and a label that drops the model identifier to fit a cap is worse
   than a longer label. This would have been the first thing a live run hit.
2. **A model identifier is not a version.** Providers re-train behind a stable name, so
   `recorded_at` is now part of the fixture *and* of `ReplayEmbedder.name` — without it, two
   recordings of the "same" model are indistinguishable and the scorer pin cannot tell them
   apart.

## Phase 3 - the Gemini adapter (prepared, not run)

Preparation only, for a run planned on a machine that holds the key. **No live call was
made, no embeddings were generated, and nothing contacted Gemini.** Both frozen baselines
are byte-identical to before this slice, and the shipped configuration still selects the
offline `hashing` embedder.

| module | responsibility |
|---|---|
| `app/infrastructure/embeddings/gemini.py` | `GeminiEmbeddingProvider` - `gemini-embedding-001`, the Gemini wire format, 768 dimensions |
| `app/infrastructure/embeddings/http.py` | `post_json` and `raise_for_status` extracted so both adapters share one error translation |
| `app/infrastructure/embeddings/registry.py` | the `gemini` factory. Known, **not enabled** |
| `app/config/settings.py` | `CCE_EMBEDDING_TASK_TYPE` |
| `.env.example` | placeholders only, with the reasoning |
| `tests/unit/test_gemini_embeddings.py` | 58 `MockTransport` tests |

### Why a second adapter rather than a flag

Gemini differs from the OpenAI shape in four places at once - endpoint, auth header, request
body and response body (`docs/ARCHITECTURE.md` 11.4.2). A branch per difference is how one
adapter becomes unreadable. What the two share is shared as *code*: `post_json` and
`raise_for_status` were extracted from `HttpEmbeddingProvider` and now serve both, and both
inherit every obligation in `EmbeddingAdapter` - per-vector width validation, finiteness,
deterministic batching, provenance. The extraction changed no behaviour: all 71 pre-existing
adapter tests pass untouched.

### Three decisions worth recording

**The credential goes in a header, never the query string.** Gemini accepts `?key=`, which
the API treats as equivalent and which is much worse operationally: query strings land in
access logs, proxy logs and error reports. A test asserts the request URL carries no query
at all, and another asserts the key never appears in an error message.

**The response carries no index, so ordering is positional and unenforceable.** The OpenAI
shape lets the adapter sort by the provider's own `index`; `batchEmbedContents` returns
`embeddings[]` with nothing identifying which request each entry answers. The count is
checked - that catches a dropped or duplicated vector - but a *permuted* response would
produce a plausible wrong ranking rather than an error. It is documented rather than
validated away, because it cannot be validated away.

**768 is a documented width, not a tuned one.** `gemini-embedding-001` is Matryoshka: 3072
native, 1536 and 768 supported truncations. 768 keeps a 51-vector fixture reviewable in a
diff. Nothing has compared the three, and the adapter refuses an undocumented width before
making a request rather than discovering it as a 400 - or worse, as a silently different
vector.

### Two things to verify on the first real run, not to assume

Both are written into the module docstring so the first run checks them rather than trusting
this note:

1. **Normalization.** Google documents sub-3072 Matryoshka output as **not** unit length, so
   `expects_unit_vectors` is false and the adapter does not normalize. Cosine is
   scale-invariant, so ranking is identical either way, and transforming a provider's numbers
   inside a mapping adapter would make a recorded fixture something other than what the
   provider returned. If the first run shows unit vectors at 768, set the flag and nothing
   else changes.
2. **Task type.** Gemini accepts `taskType`, and for retrieval Google recommends the
   *asymmetric* pair `RETRIEVAL_DOCUMENT` for corpus text and `RETRIEVAL_QUERY` for queries.
   `EmbeddingPort.embed` takes one undifferentiated list of texts and cannot express that;
   widening it is a domain-contract change. So the adapter sends one symmetric
   `SEMANTIC_SIMILARITY`, which is the honest thing the current interface can do and which
   probably costs recall - issue 44, not a silent choice.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2137 passed**, 2 warnings, ~21s |
| `mypy app tests` | **success, 187 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 234 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |
| both retrieval benchmarks | byte-identical to before the slice |

58 new tests, all through `httpx.MockTransport` with a placeholder credential: request shape
(endpoint, header, per-text body entries, explicit `outputDimensionality`, task type,
batching at the documented limit of 100, empty request making no call), response shape
(order, count mismatch, unnormalized vectors accepted, five malformed bodies), dimensions
(all three supported widths, undocumented widths refused pre-request, wrong returned width,
non-finite components, every vector validated and not only the first), Google-shaped error
bodies across 400/401/403/429/500/503 plus timeout and connection failure, and selection
(known, not enabled, and unusable without configuration).

## Phase 3 - hybrid retrieval (built, measured, and not an improvement)

The third strategy, and the first one whose measurement argues against keeping it as the
default. Full figures in `docs/EVALUATION.md`, "Hybrid retrieval, as measured"; the design
is `docs/ARCHITECTURE.md` 6.2.

| module | responsibility |
|---|---|
| `app/retrieval/hybrid.py` | `fuse` - reciprocal rank fusion, `HybridResult`, `HybridOutcome`, `strategy_split` |
| `app/application/use_cases/retrieve_evidence.py` | `execute_hybrid` - runs both strategies, fuses their ranks |
| `app/evaluation/retrieval_benchmark.py` | `run_hybrid_suite`, and a `hybrid_scorer_version` pin |
| `app/domain/value_objects/enums.py` | `RetrievalStrategy.HYBRID` |
| `scripts/retrieval_benchmark.py` | `--strategy hybrid`, plus the contribution split |

### Fusion is a pure function of two outcomes

`fuse(lexical_outcome, vector_outcome)` takes what each strategy already produced. It
constructs no retriever, touches no store, embeds nothing, and **neither underlying retriever
knows it exists** - the use case calls the same two public methods any other caller would.

That is what makes every guarantee inherited rather than re-implemented. Tenant isolation,
authorization, source-type, scope and temporal filtering and provenance validation all ran
**twice** - once per strategy - before `fuse` saw a result. Fusion may merge, reorder and
truncate; it cannot introduce a span neither input returned, and a test asserts the fused set
is a subset of the union. A fusion that could add a candidate would be a retrieval strategy
running outside every filter its inputs passed.

### Why reciprocal rank fusion

`score(d) = Σ 1/(k + rank_s(d))` over the strategies that returned `d`, at `k = 60`.

The two inputs are on **incomparable scales**: BM25 is an unbounded sum over term weights,
cosine is bounded in `[-1, 1]`. Any fusion that combines *scores* must normalize first, and
every normalization is an unmeasured decision - min-max is unstable when a strategy returns
one result (that result becomes 1.0 however weak it was), and z-scores need a distribution a
21-span corpus does not have. RRF discards the scores and fuses ranks, so there is no
normalization to get wrong. `k = 60` is the original paper's value and is **untuned here**,
like `k1`/`b` before it.

### Strategy provenance is preserved, not flattened

`HybridResult.contributions` maps each strategy to the rank *it* gave a span, so
`lexical-only`, `vector-only` and `both` are readable off the outcome. That is a requirement:
checkpoint 9.3 counts corroboration across distinct strategies, and a fused list that forgot
which strategy found what could not support the count. Two contract consequences, both
deliberate:

- `RetrievalResult.strategy` is `hybrid`, because the fused *score* is what the result
  carries;
- `RetrievalOutcome.strategies_executed` is `(lexical, vector)` and **not** `(hybrid,)`,
  because both genuinely ran and a single entry would collapse two strategies into one.

`fuse` refuses anything that is not two rankings of the same question: different queries,
different permission scopes (fusing across scopes could surface a span the requester may not
read), different filter configurations, an already-fused input, or two rankings from the same
strategy. The shared trace buckets are **asserted equal** rather than summed - both
retrievers filter the same candidate set in the same fixed order, so a disagreement means the
fusion spans two corpora.

### Results

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **hybrid** (of the 56 reachable) | **33.9%** | **66.1%** | **83.9%** | **91.1%** |
| lexical baseline, frozen | 35.7% | 75.0% | 89.3% | 91.1% |
| vector baseline, frozen | 30.4% | 60.7% | 76.8% | 80.4% |
| hybrid authorization-filtered | 50.0% | 100% | 100% | 100% |

Coverage 90.3%, unreachable 4, withheld 2, leaks **0** - identical across all three runs.
Expectations retrieved / missed: **51 / 5**, the same as lexical.

Contribution over the 260 results the 35 fused queries returned:

| | count | share |
|---|---|---|
| both | 143 | 55.0% |
| lexical-only | 13 | 5.0% |
| vector-only | 104 | 40.0% |

### The result, and it is not the hoped-for one

**Fusion recovered the lexical set exactly and ranked it worse.** Same 51 hits, same 5
misses, same set - so it added nothing - while recall fell 1.8 points at @1, **8.9 at @3**,
5.4 at @5, tying only at @10.

The contribution table is the mechanism: the vector input supplies **104 vector-only results
against lexical's 13**, eight times the volume, while contributing no labelled evidence
lexical missed. Every one of them takes a rank and receives a positive RRF contribution, so
relevant lexical results are pushed down. RRF is working correctly - it beats the weak input
at every *k*, which is its job. Beating the *strong* input needs the second input to
contribute something, and this one does not.

Three further observations:

1. **Negative-control noise is inherited wholesale**: 28 results across four controls,
   identical to the vector input against lexical's 5. A union cannot be quieter than its
   noisiest member, and fusion adds no floor of its own.
2. **Every guarantee survived, and the measurement confirms it** rather than asserting it:
   coverage, unreachable, withheld, authorization-filtered recall and the leak count are
   identical across all three strategies.
3. **All three miss the same five expectations** - four the coupling set of
   `retention.window`, one the token-expiry wording. A third strategy built from the same two
   failing on the same five is stronger evidence for the closed registry than either failing
   alone.

**This is a verdict on fusing *these two* inputs, not on hybrid retrieval.** The vector side
has no semantics. Nothing was tuned in response: `k` stays at 60, no weighting was tried, and
neither input was touched - down-weighting the weak strategy would improve the numbers and
would be exactly the wrong lesson from one measurement over 62 labels.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2223 passed**, 2 warnings, ~12s |
| `mypy app tests` | **success, 190 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 237 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |
| both baseline benchmarks | recall byte-identical to before the slice |

86 new tests: 49 on the fusion arithmetic and its refusals, 37 on the benchmark and the
three-way comparison. All offline.

### A defect the tests found

**Fusing two rankings from the same strategy silently succeeded.** `fuse` checked that each
input named exactly one known strategy but not that the two were *different*, so
lexical+lexical produced a fused list whose `both` count reported corroboration that never
happened - and corroboration is precisely what checkpoint 9.3 counts. Found by a test written
to assert the refusal, which then failed. Now rejected explicitly, for both orderings.

## Phase 3 - the retrieval entry point (verified)

The finalization layer, and deliberately the smallest slice in the phase: it adds **dispatch
and nothing else**. Design in `docs/ARCHITECTURE.md` 6.3.

| module | responsibility |
|---|---|
| `app/application/orchestration/retrieval_service.py` | `RetrievalService`, `RetrievalRequest`, `FinalizedRetrieval`, and the two refusals |
| `app/config/settings.py` | `CCE_RETRIEVAL_STRATEGY`, defaulting to `lexical` |
| `.env.example` | the same, with why the default is what it is |
| `tests/unit/test_retrieval_service.py` | 66 tests: selection, refusal, permission propagation, output consistency |

### What it absorbs

Three strategies, three call shapes: `execute` is synchronous, `execute_vector` is async and
needs an embedder, `execute_hybrid` needs both and returns a richer object. That asymmetry is
correct one layer down and wrong to push onto every caller.
`RetrievalService.retrieve(request)` is the single entry point, and
`FinalizedRetrieval.outcome` is a plain `RetrievalOutcome` for all three - same type, same
validators - so a consumer needs no strategy-specific branch.

The envelope adds only what the contract cannot express: the **selected** strategy, and the
hybrid contribution detail. Those are different things - a fused run reports
`(lexical, vector)` in `strategies_executed` because both genuinely ran and corroboration
counts distinct strategies (checkpoint 9.3), so the selected strategy has nowhere else to
live. `__post_init__` requires the contribution detail to be present *exactly* for a hybrid
run, so a consumer can neither read provenance that does not exist nor miss some that does.

### Every guarantee is inherited, and that is observable

No filter, no scoring, no ranking and no merging happens in this layer. Tenant isolation and
authorization are applied inside each retriever by narrowing the index before anything is
scored; source-type, scope and temporal filters run in the same fixed order in all three;
provenance travels by reference.

Rather than assert that, the tests measure it: one filtered query is run under all three
strategies and their filter buckets - `considered`, tenant, permission, source type, as-of,
scope, unscoped - must be **identical**, with only the scoring buckets free to differ. A
second test requires the dispatched lexical outcome to serialize byte-identically to the use
case called directly, so the orchestration layer cannot have quietly become a fourth
strategy.

### Three explicit failures, and why the tempting one is refused

| condition | outcome |
|---|---|
| unknown name on a request | `ValidationError` - the request carries the enum, so it fails before the object exists |
| unknown `CCE_RETRIEVAL_STRATEGY` | `UnknownRetrievalStrategy`, naming the alternatives, raised when the service is built |
| `vector`/`hybrid` selected with no embedder | `EmbedderRequired` |
| enum member with no handler | `UnknownRetrievalStrategy` - dispatch is a table, not an `if` chain |

**`EmbedderRequired` rather than a fall back to lexical** is the decision worth recording. A
fallback is the obvious convenience and it is refused: a caller who asked for semantic
retrieval and received lexical results has no way to detect it, and a benchmark figure
produced that way would be attributed to a strategy that never ran. `available_strategies()`
reports what the service can actually serve, so a caller can present the choice honestly
instead of offering one that raises.

Both refusals share `RetrievalConfigurationError`, so a caller handling misconfiguration does
not have to enumerate its kinds.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2289 passed**, 2 warnings, ~24s |
| `mypy app tests` | **success, 192 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 239 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |
| all three benchmarks | recall byte-identical to before the slice |

### A defect the tests found

**A test that read correctly and asserted nothing.** The check that this layer imports no
infrastructure grepped the module source for `app.infrastructure` - and matched the docstring
that explains the rule, so it failed for the right reason by accident and would have passed
for the wrong one had the docstring been worded differently. It parses the imports with `ast`
now. Same failure mode as the `co_names` substring check found during the permission slice,
and worth recording twice because it is the one mutation testing exists to catch.

## Phase 4 - the consequence-hunter (verified)

The first agentic slice, and the first code in the system whose output is a *hypothesis*
rather than a derivation. Design in `docs/ARCHITECTURE.md` 6.4.

| module | responsibility |
|---|---|
| `app/consequences/proposals.py` | `HypothesisProposal` / `HypothesisBatch` - what the model is allowed to say |
| `app/consequences/prompting.py` | the task contract; evidence inside the untrusted envelope |
| `app/consequences/hunter.py` | `ConsequenceHunter` - one call, then the gate |
| `app/application/use_cases/hunt_consequences.py` | assemble, invoke, suppress Class A duplicates |
| `tests/contract/test_layering.py` | `app/consequences` declared as a new guarded layer |
| `.claude/agents/consequence-hunter.md` | a stale instruction corrected (below) |

### The same split as ingestion: propose, then admit

The model proposes; deterministic code admits. `app/consequences` is guarded for the reason
`app/assumptions` is - the code deciding what a model was allowed to have said must not
itself be able to make a network call - and it reaches its provider only through
`AgentCompletionPort`, so tier, provider and model stay configuration.

**Every check in the gate can only weaken a claim.** There is no branch that adds evidence,
raises support or assigns a class. That asymmetry is the design, and it is what makes the
stage safe to run against a model nobody has evaluated.

### The four refusals

| an agent supplying | outcome |
|---|---|
| `confidence` | batch rejected (`extra="forbid"`); computed by versioned code over evidence features |
| `severity` | batch rejected; the judge assigns it |
| `status` | batch rejected; absent a recorded analogue, Class B cannot exceed `speculative` and the runtime records the absence |
| `consequence_class` | **no field exists**, and `ConsequenceHypothesis` hard-codes `derivation=INFERRED` / `support=MECHANISM_ONLY` as `Literal`s |

The last is the strongest: the guarantee is a property of the type rather than of this
module's care, so a caller constructing a hypothesis by hand cannot promote it either. A
test asserts exactly that by trying.

### Evidence discipline

Only ids the model was **actually offered** survive; the rest are dropped and reported,
because a model inventing citations is a signal about the model and not only about one
hypothesis. Matching is exact: an id differing by a character is not a near miss to repair,
and repairing one would be the system inventing a citation on the model's behalf.

A hypothesis whose every citation was fabricated is weakened rather than discarded - it
becomes a hypothesis with no evidence, which is a legitimate result. The defect would be the
reverse: an unsupported claim presented as established, and there is no code path to it.

Retrieved corpus text is wrapped in the extractor's existing per-call nonce envelope. A span
that tries to impersonate the delimiter is neutralized and reported, which one test exercises
with a span demanding `severity CRITICAL`.

### Class A suppression

The prompt shows the hunter the Class A findings and asks it not to re-derive them; the
pipeline does not trust that. `merge_class_b` runs afterwards and suppresses any hypothesis
whose `(assumption_key, affected_source_version_id)` pair is already in the Class A set. The
rule is **reused, not reimplemented** - a second copy would drift, and this one is already
mutation-tested. Suppression merges: the absorbing finding records the hypothesis id, so the
mechanism text survives while the duplicate finding does not. A hypothesis naming no artifact
is retained, because ambiguity demotes to Class B (checkpoint 7.4).

`ClassBOutcome.rediscovery_rate` wires the `class_b_rediscovery_rate` metric
`docs/EVALUATION.md` reserved. A **high** value warns that the evidence slice is
misconfigured; it is not evidence of productivity.

### Failure handling

A provider failure, a rate limit, an unrepairable JSON body or a schema violation each yield
**no hypotheses and a recorded failure** - never a partial set, and never an empty result
that reads as "found nothing". Provenance is resolved without a model call, so a failed stage
still reports which prompt and which binding failed, and the Class A set is left untouched.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2336 passed**, 2 warnings, ~24s |
| `mypy app tests` | **success, 199 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 246 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

45 new tests, all through the real `LlmGateway` over `FakeProvider` - so the completion port,
its JSON repair loop and its error translation are genuinely exercised with **no credentials
and no network**.

### A defect found in the agent definition

**`.claude/agents/consequence-hunter.md` instructed the agent to do something the frozen
design forbids.** It said "Mark `consequence_class` explicitly", contradicting `CLAUDE.md` 22
- classification is pipeline-assigned, because Class A means "produced by the deterministic
resolver" and a hypothesis declaring its own class would be claiming a derivation it did not
perform. The schema has no field for it, so the instruction could only ever have produced a
rejected batch and a wasted repair round. Corrected, and a test now asserts the prompt and the
enforcement agree.

## Phase 4 - the EvidenceGap producer (verified)

The component that makes "we do not know" a *measured* statement instead of a hedge. Design in
`docs/ARCHITECTURE.md` 6.5.

| module | responsibility |
|---|---|
| `app/application/orchestration/evidence_gaps.py` | `SearchProvenance`, `GapRecord`, `produce_gaps`, `gaps_from_hunt` |
| `tests/unit/test_evidence_gaps.py` | 49 tests, every trace from a real retrieval over the sample corpus |

Nothing else changed. No retrieval algorithm, no domain entity, no checkpoint, and not
`hunt_consequences.py` — the producer is a separate seam the caller composes, which is
precisely what keeps the hunter without access to the trace it must not author.

### The creation rule

An `EvidenceGap` is checkpoint 2's type E: a claim **about the corpus**, not about the world.
Checkpoint 2.2 defines its support as *the executed QueryPlan that returned nothing*, so all
three conditions below are required and each is read off the trace:

1. **a strategy ran** — `strategies_executed` is non-empty;
2. **over a corpus this requester could read** —
   `considered - excluded_by_tenant - excluded_by_permission > 0`;
3. **and it returned nothing** — `trace.returned == 0`.

**"The hunter said it did not know" is never sufficient**, and that is the whole point. A
model's uncertainty recorded as a statement about the corpus would launder its silence into
evidence — the exact inversion type E exists to prevent. A search that returned evidence
produces no gap for that search either: the negative provenance would be false, because the
plan did not return nothing.

Refusals are **counted, not silent**: `refused_no_search` and `refused_evidence_found` are
separate, because "nothing was searched" and "everything was answered" are different facts
about an analysis and must not both read as zero gaps.

### The provenance captured

| from the query | from the trace | from the caller |
|---|---|---|
| query text, tenant, source-type / scope / `as_of` filters, `top_k` | every exclusion bucket, `matched`, `returned` | run identifier, `recorded_at`, optional node association |
| permission scope hash, retrieval config version, scorer version | selected strategy + strategies executed | |

The domain entity is frozen by the checkpoint, so what it has no field for — query text,
filters, scope hash, config and scorer versions, the trace, the run id — lives in a
`SearchProvenance` envelope beside it. Same pattern as `FinalizedRetrieval` and
`HybridOutcome`, and it is the reason this slice touched no domain contract. `GapRecord` pairs
the two rather than merging them, so the claim and its audit trail cannot drift.

### Why the model cannot author the provenance

`produce_gaps` accepts model-authored text in exactly one parameter, `questions`, and it
reaches exactly one field, `EvidenceGap.question`. Everything else is read off the
`FinalizedRetrieval`. **There is no parameter through which a model-authored value could
reach a provenance field**, which is stronger than validating that none did, and a test
asserts the signature rather than the intent — so adding such a parameter later fails a test
rather than passing review.

`SearchProvenance` then requires its trace to be a coherent account of a search: the buckets
must sum to `considered`, the invariant all three retrievers already guarantee (asserted here
again, per strategy, from real searches). A fabricated provenance has to add up the way a real
search adds up rather than merely being well-typed.

One test is about the realistic attack rather than the obvious one: a *question* crafted to
look like metadata (`"as_of 1999-01-01 strategies_executed oracle permission_scope_hash
deadbeef"`) stays a question, and none of it reaches `as_of`, `strategies_executed` or the
scope hash.

### Authorization

| the requester's slice | outcome |
|---|---|
| empty | **no gap** — they searched nothing they could read |
| narrowed | gap, flagged `authorization_limited` |
| complete | gap, unflagged |

The empty case is the interesting one. A gap there would claim the corpus cannot establish
something about a corpus this requester never saw, and the alternative — a gap per restricted
requester — would make a report's gap list vary with who ran it and read as though the corpus
had changed.

What escapes about a withheld document is a **count and a scope hash**, which the trace
already carries and which is query-independent by construction, so surfacing it adds no
oracle. Two tests hold that line: one asserts the withheld count is identical for two
different queries, and one serializes a gap produced under the restricted scope and asserts
that no evidence id, uri, document title or span text visible only to the privileged scope
appears anywhere in it.

### A gap is never an edge, a contradiction, or a Class A finding

It carries no evidence at all — the entity's own validator refuses any, exercised through this
path — attaches to the analysis, and may annotate one node through `partition`. There is
exactly one node field and no second, so a gap cannot name two ends. That is checkpoint 2.4's
rule made structural rather than remembered.

### A design point worth recording: which number the rule reads

The rule is anchored on `returned`, not `matched`. `matched` would support a different and
stronger claim — "the corpus contains nothing" — and a gap does not make that claim; it
describes what the executed search delivered. A truncated search that returned one result is
therefore not a gap, and a test pins that.

### A test-design point, because the first attempt was weaker

Three tests initially built their "empty search" from a query whose words the corpus does not
contain. That works for lexical and **not** for vector or hybrid: `HashingEmbedder` assigns a
non-zero cosine to unrelated text, so a vector search over absent vocabulary still returns
something. The tests now empty the result set with an `as_of` before any version was in force,
which is empty for every strategy — and is a more realistic negative-provenance shape anyway,
since "the filters excluded everything" is how a real gap usually arises.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2385 passed**, 2 warnings, ~25s |
| `mypy app tests` | **success, 201 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 248 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

49 new tests. Every trace under test comes from a real retrieval over the sample corpus
through the same orchestration layer production uses — a hand-built trace would let the
producer be tested against a search that never ran, which is the defect it exists to prevent.

## Phase 4 - the contradiction-hunter (verified)

The second agentic slice, and the first whose *input* is trust-bearing. Design in
`docs/ARCHITECTURE.md` 6.6.

| module | responsibility |
|---|---|
| `app/contradictions/divergence.py` | project the resolver's collisions into five classified buckets |
| `app/contradictions/proposals.py` | `DivergenceAssessment` / `AssessmentBatch` - what the model may say |
| `app/contradictions/prompting.py` | closed set, per-item evidence, untrusted envelope |
| `app/contradictions/hunter.py` | `ContradictionHunter` - one call, then the gate |
| `app/application/use_cases/hunt_contradictions.py` | assemble, rate, project into `SpecialistFinding` |
| `tests/contract/test_layering.py` | `app/contradictions` declared as a new guarded layer |
| `.claude/agents/contradiction-hunter.md` | an unwired half of the contract scoped out (below) |

### The risk is the opposite of the consequence-hunter's

There, the danger was a model claiming more than it earned. Here it is a model quietly
rewriting something the resolver already got right. Every safeguard below targets that.

### It cannot search, and that is a layer property

`ContradictionHunter.__init__` takes a completion port and an agent definition. No retriever,
no store. The layering contract now forbids `app/contradictions` from importing
`app.retrieval` at all, and a test parses the module's imports with `ast` rather than grepping
its source - the previous two slices each produced a test that read correctly and asserted
nothing because a substring matched the docstring explaining the rule.

### An empty divergence set costs zero calls

Not a filter over the model's output - a model that was never asked. `assess` returns before
reaching the provider when the set is empty, and a test asserts `fake_provider.call_count ==
0`. A filter can be wrong; a call that does not happen cannot be.

The negative control runs one layer deeper: two policies *agreeing* produce no divergence from
the real collision matrix, so the end-to-end run produces no findings and makes no call.

### The five classes, and where each comes from

| class | deterministic source | emitted as a finding? |
|---|---|---|
| `material_contradiction` | `conflicts_with` relation | yes |
| `stale_invalidation` | `invalidates_assumption` relation | yes |
| `unauthorized_override` | `unauthorized_override` relation | yes - governance |
| `legitimate_scoped_override` | `OverrideRecord` | **no** - correct behaviour |
| `non_material_divergence` | `CouplingAnomaly` | **no** - already a coupling anomaly |

Every one of these is produced in the tests by the **real** `collide_key`, one scenario per
class, parametrized so that a matrix change breaks the test rather than leaving a class nothing
reaches. Hand-building a `KeyCollisions` would have let the tests assert that the projection
behaves while saying nothing about whether the deterministic layer produces what it assumes.

### How Class A authority survives contact with a model

| the route a model might take | why it is closed |
|---|---|
| restate `divergence_class` | no such field; `extra="forbid"` rejects the batch |
| restate `relationship` / `relation_id` / `rule_id` | same, **and** the prompt never renders them, so the output cannot name a Class A relation at all |
| assess a divergence nobody found | handles verified against the offered set; the batch schema has no field for a new contradiction |
| suppress a finding by rating it immaterial | emission is decided by the deterministic class; the rating is recorded *on* the finding |
| promote a legitimate override to a contradiction | `reportable_as_contradiction` reads the item, never the assessment |
| have a rejected batch cost a finding | both failure paths return every divergence unassessed |

The last row is the shape of the whole guarantee: the agent's contribution can be lost
entirely - provider outage, malformed JSON, forbidden field - without costing the resolver's
conclusions anything.

`DivergenceItem.__post_init__` closes one more route that has nothing to do with models: a
*caller* cannot assemble an item whose class disagrees with its relationship, so a future
integration cannot mislabel a staleness finding as a peer conflict either.

### No numeric confidence

`Materiality` is three values, `TemporalPosition` five. `SpecialistFinding.evidence_position`
carries the position into the domain and is derived from two facts - does the finding carry
evidence, did the agent reach a determination - neither of which is a model-supplied score.

One asymmetry is deliberate and is the only place this stage can reduce what a reader sees:
`immaterial` argues against a divergence found by exact comparison, so it costs a reconciling
exception or an explicit rationale. Bare dismissal has no representation. `material` must name
who is affected, because "this matters" with no affected party is a label.

### Citations are checked per item, not against one pool

A real evidence id attached to an unrelated divergence is a fabricated citation assembled out
of genuine parts, and an existence check against a flat pool would admit it. The offered set is
per handle. A finding's own evidence is the participating bindings' spans, which the resolver
named - so a model that cited nothing still produces an evidenced finding, and a model that
cited more cannot produce a better-evidenced one.

Contextual evidence beyond the two bindings has exactly one route, because the agent definition
asks whether an exception clause reconciles two artifacts and such a clause usually sits in
neither of them. The caller supplies the span and names the items it bears on; a span naming an
unknown item raises. This was missing from the first draft of this slice - the field existed but
nothing could reach it, which would have made `reconciling_exception` answerable only from the
participants' own text.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2482 passed**, 2 warnings, ~29s |
| `mypy app tests` | **success, 209 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 256 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

97 new tests: 95 in `tests/unit/contradictions/`, plus two more cases in the layering
contract test, which is parametrized over `GUARDED_LAYERS` and now covers the new layer. All
of them run through the real `LlmGateway` over `FakeProvider`, with no credentials and no
network.

### A boundary in the agent definition, recorded rather than crossed

`.claude/agents/contradiction-hunter.md` describes a second job: contradictions no key
comparison can express - SOP versus observed behaviour, conflicting definitions, a procedure
that cannot be executed as written. **That half has no runtime path and no schema**, and it
cannot simply be added here: requirement 10 of this slice is that a corpus with no
deterministic divergence produces no contradiction findings, and the cheapest guarantee is a
schema with nowhere to put one.

Left unqualified, the definition would have asked a cooperative model for output guaranteed to
be rejected - the same defect found in the consequence-hunter's definition last slice, where it
instructed the agent to mark a `consequence_class` that has no field. The definition now names
the boundary and the version is bumped to 4 (the prompt hash is stamped onto every assessment,
so an edit that did not bump it would make two different prompts indistinguishable in
provenance). A test asserts the definition and the schema agree.

## Phase 4 - the adversarial challenger (verified)

The third agentic slice, and the one with the most inverted incentive. Design in
`docs/ARCHITECTURE.md` 6.7.

| module | responsibility |
|---|---|
| `app/challenge/candidates.py` | project finalized candidates; one rulebook per candidate, by origin |
| `app/challenge/proposals.py` | `ChallengeProposal` / `ChallengeBatch` - three outcomes, each priced |
| `app/challenge/prompting.py` | per-candidate rulebook, per-candidate evidence, untrusted envelope |
| `app/challenge/challenger.py` | `AdversarialChallenger` - one call, then the gate |
| `app/application/use_cases/challenge_findings.py` | assemble, challenge, report `ChallengeResult` + `FalsificationTest` |
| `tests/contract/test_layering.py` | `app/challenge` declared as a new guarded layer |
| `.claude/agents/adversarial-challenger.md` | closed-input framing corrected (below) |

### Producing objections is not evidence that this works

A model told to disagree will disagree, fluently, at whatever rate it is asked. So the
measurement that matters is `objection_refusal_rate` - the share of the model's verdicts the
gate would not admit - and it is a signal about the **model**, not the corpus. A high value
means it is producing objections it cannot support, which is the failure mode this entire
stage is designed around.

### Two rulebooks, chosen by producer

| candidate | challengeable on |
|---|---|
| Class B hypothesis (`origin=agent`) | everything, including the mechanism |
| Class A finding (`origin=resolver`) | **only** `disputed_span`, `disputed_key_or_scope`, `disputed_temporal_validity`, `superseding_version` |

Which rulebook applies is read off the candidate's `origin`, and the prompt states it
*against each candidate* rather than once at the top - the looser rulebook is the one that gets
applied when the two sit adjacent in one paragraph.

A Class A candidate's **mechanism is never rendered into the prompt**, and
`ChallengeTarget.__post_init__` refuses a Class A target that carries one. It is precisely
what may not be challenged, so showing it would invite the one objection the runtime must
reject. The inputs are shown instead: key, scope, effective date, evidence.

### Why there is no fifth ground

Requirement 3 of this slice lists "incorrect deterministic rule application", and the frozen
enum has no member for it. That is correct rather than an omission, and worth recording
because it looks like a gap:

Every version of a rule-application dispute that corpus evidence can support already reduces
to disputing an **input** - a precondition that did not hold is temporal, two keys that are
not one variable is key-or-scope, a value the span does not carry is a span dispute. What
remains is "the rule ran correctly on correct inputs and I disagree with the output", which
is a claim about the *code*. It cannot be evidence-backed, so requirement 5 (every
`refuted`/`weakened` carries valid evidence ids) makes it unrepresentable anyway - and routing
it here would let a model's opinion mark a derivation wrong without touching anything
checkable. A resolver defect is a bug report, not a `ChallengeResult`. A test asserts the enum
offers no mechanism member, so a future addition breaks it deliberately.

### The subtlest decision: refused, not downgraded

An inadmissible Class A objection could be converted to `no_refutation_found`. It must not be.
That outcome asserts "I looked and found nothing", which is **not** what happened - the model
found something it was not permitted to say. Converting would misreport the run *and*
fabricate the falsification test the outcome owes, since the model supplied no observation.

So it is refused, counted by reason, and the candidate is reported `unreviewed`. Four reasons
are kept apart because they mean different things and only one is about the corpus:
`invented_target`, `inadmissible_class_a_ground`, `class_a_ground_on_class_b`,
`no_admitted_citation`.

### The citation obligation, enforced twice

The per-outcome rules are duplicated from `ChallengeResult` onto the proposal schema
deliberately. Failing at the boundary lets the runtime record *that the model produced a
malformed objection*; failing at domain construction instead would surface from the wrong
layer, after the gate had already decided to admit it.

The case that matters most is a `refuted` whose every citation was fabricated - it reads to a
downstream consumer exactly like a proven refutation. The gate refuses it as uncited rather
than handing the domain a record it already knows is invalid. Citations are checked **per
candidate**: a real id belonging to another candidate is a fabricated citation assembled out
of genuine parts, and an existence check against one flat pool would admit it.

Contextual evidence is not optional here, unlike in the previous slice. The challenger's
single most valuable finding is *a newer source version supersedes the one the derivation
used*, which is by definition not in the candidate's own evidence set. The caller retrieves it
and names the candidates it is admissible against; a superseding span is flagged as such.

### Class A provenance is untouched, and that is a layer property

Nothing in `app/challenge` constructs, copies or re-provenances a `ClassAFinding` - asserted
by parsing the modules for `ClassAFinding(` calls and `model_copy` attributes rather than
grepping their text, which would match the docstrings explaining the rule. A `ChallengeResult`
carries its own `produced_by`, so the challenger's attribution and the resolver's cannot be
confused. One test dumps every candidate before and after a full run **on the refuted path**
and compares.

`app/challenge` also may not import `app.retrieval`: the challenger reviews what it is handed,
and a layer that could reach a store on its own could go looking for the objection it wants.

### Surviving review is not corroboration

Checkpoint 9.2, made visible in the naming. `no_refutation_found` means nobody found contrary
evidence, which is strictly weaker than evidence of correctness, and there is no field on the
proposal or the result through which it could raise a status. `ChallengeReport.survived` is
reported apart from `unreviewed` - conflating those is the one accounting error that would make
a failed stage look like a clean bill of health.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2573 passed**, 2 warnings, ~17s |
| `mypy app tests` | **success, 217 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 264 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

91 new tests: 89 in `tests/unit/challenge/`, plus two more cases in the layering contract
test, which is parametrized over `GUARDED_LAYERS`. All through the real `LlmGateway` over
`FakeProvider`, with no credentials and no network.

### A correction to the agent definition

`.claude/agents/adversarial-challenger.md` framed its input as "consequence hypotheses"
throughout and told the agent to seek "contrary evidence in the corpus". Neither matches the
runtime: it receives **both** candidate classes, and it has no corpus access or search tool.
Left as written it would have led a model to apply the looser Class B rulebook to a Class A
finding - the exact error this slice exists to prevent - and to look for evidence it was never
given. Corrected, version bumped to 4 (the prompt hash is stamped onto every verdict, so an
edit that did not bump it would make two different prompts indistinguishable in provenance),
and tests assert the definition states the input is closed and names the four grounds.

## Phase 4 - the consequence judge (verified)

The fourth agentic slice, the first consumer of the whole pipeline, and the first stage whose
output a reader acts on directly. Design in `docs/ARCHITECTURE.md` 6.8.

| module | responsibility |
|---|---|
| `app/domain/services/confidence.py` | **new.** The deterministic confidence function - the only place a confidence number is produced |
| `app/judgment/dossier.py` | assemble one dossier per candidate from all five upstream stages; inherit the identity |
| `app/judgment/status_rules.py` | the frozen checkpoint 9.2 gates, applied in code |
| `app/judgment/proposals.py` | `DispositionProposal` / `DispositionBatch` - status and severity, no number |
| `app/judgment/prompting.py` | per-candidate dossier, the gates stated, the closed evidence set |
| `app/judgment/judge.py` | `ConsequenceJudge` - one call, the gate, then the three domain records |
| `app/application/use_cases/judge_consequences.py` | assemble, arbitrate, report |
| `app/domain/value_objects/rules.py` | `CONFIDENCE_FN_VERSION` added |
| `tests/contract/test_layering.py` | `app/judgment` declared as a new guarded layer |
| `.claude/agents/consequence-judge.md` | a self-contradictory instruction corrected (below) |

### Confidence became real in this slice

`ConsequenceScore` has existed since Phase 1 with nothing producing it. Requirement 7 -
"confidence must remain the responsibility of deterministic scoring" - could not be satisfied
by absence, so `app/domain/services/confidence.py` now implements checkpoint 9.1's eight
factors and pins them behind `CONFIDENCE_FN_VERSION`.

**The input type is the enforcement of checkpoint 9.3.** `ConfidenceInputs` has no field for
an agent, an agent count, a model, a vote or a level of agreement between agents, and
`extra="forbid"` refuses one. The function *cannot* count agents, so it cannot be talked into
treating agreement as corroboration - which is the same structural approach
`coupling_strength.compute` already takes by accepting only two bindings.

Corroboration saturates over distinct source versions and is **gated by strategy reach**:
five versions reached by one strategy scores no higher than the strategies justify. That is
the arithmetic form of "a lexical retriever's blind spot must not become a confident number",
and it matters here because the vector strategy currently finds a strict subset of what
lexical finds (Phase 3).

The weights are **not tuned** and there is no calibration data. That is why the version label
exists, and why `docs/IDEA.md` 13 forbids calling this a probability: it is an ordering aid.

### Severity and confidence are never blended

| | decided by | tested |
|---|---|---|
| `severity` | the judge, four values | every value carried through untouched; CRITICAL survives a 0.0 confidence; LOW survives a >0.5 one |
| `confidence` | the versioned function | no field on the proposal; a batch carrying one is rejected whole |

Both directions are pinned, so the first test cannot pass by the runtime ignoring severity.
HIGH/LOW is a legitimate and important output (checkpoint 9.4), and a runtime that reconciled
them would destroy the one signal this stage adds.

### The status gates, and the one enforced by absence

Proposed by the model, then clamped in code: no admitted evidence caps below
`supported`/`likely`; Class B with no analogue caps at `speculative`; a refutation makes it
`rejected`; a narrowing caps at `disputed`.

**`no_refutation_found` raises nothing, and `clamp` has no parameter through which it could.**
A test asserts the signature; another asserts two otherwise identical candidates get the same
status whether or not a challenger cleared them. Testing by *difference* rather than by
assertion is what makes that one worth having.

`clamp` never returns something stronger than it was given, asserted over the **full
cross-product** of six statuses × two classes × four booleans rather than sampled cases. That
one-directional property is what makes proposing-then-clamping safe.

### A defect the tests found: a clamp can create an obligation nobody was asked about

Clamping a HIGH-severity finding to `unresolved` triggers `ArbitrationDecision`'s requirement
that it name what would settle it - but the judge proposed `supported`, which owes nothing, so
it supplied no action. The first implementation failed the whole batch on a Pydantic error.

Failing there would discard an entire arbitration because the model could not anticipate a
gate. So the runtime now derives the action from the clamp **reason**, which is mechanical
rather than invented: the reason *is* the thing to fix ("locate a span that supports this
claim and is retrievable under this analysis's filters, or record why none exists"). The
judge's own actions always come first, and an unclamped disposition gains nothing.

### Identity: inherited, and an orphan is refused

Issue 51 predicted this and it is now closed. The challenge layer minted the
`consequence_id`; the judge reuses it on all three records, and `dossiers_from` **raises**
when a `ChallengeResult` names a consequence no candidate carries.

The failure that guards against is silent: mint a second id and the upstream refutation
attaches to nothing, so the report shows a refuted finding as unchallenged.
`ArbitratedConsequence.__post_init__` re-checks it where the three records are paired.

### Disagreements survive a model that would rather not mention them

The runtime collects every refutation (with its admissible ground), every narrowing, every
rated divergence and every unanswered question, and **unions** it with whatever the judge
preserved. `_merge_disagreements` has no trimming parameter - a trimming option is how
"preserve material disagreement" quietly stops holding once a report looks crowded. The test
that carries the weight has the judge preserve *nothing* and asserts the upstream dissent
still reaches the reader.

### An EvidenceGap stays a corpus-knowledge statement

`dossiers_from` takes `evidence_gap_ids`, not gap records, so there is nothing to promote.
Gaps leave as `Consequence.evidence_gap_ids` and appear in no other field. Tests assert that
adding gaps adds no consequences and that a gap id reaches neither `evidence_ids` nor
`derived_relation_ids` nor `hypothesis_ids`.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2680 passed**, 2 warnings, ~22s |
| `mypy app tests` | **success, 227 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 274 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

107 new tests: 105 in `tests/unit/judgment/`, plus two more cases in the layering contract
test. All through the real `LlmGateway` over `FakeProvider`, with no credentials and no
network.

One mypy finding was a real dead branch rather than a type annoyance: `is_no_stronger` had a
comparison the type system proved could never be true, which meant the "a terminal status is
never raised out of" case was unreachable. Rewritten to return `False` there, which is the
correct answer and makes the branch meaningful if a future rule ever moves a status out of a
terminal one.

### A self-contradictory instruction in the agent definition

`.claude/agents/consequence-judge.md` said both **"You do not assign `consequence_class`"**
and, twelve lines later under Output discipline, **"Record `consequence_class`"**. The schema
has no field for it, so the second instruction could only ever have produced a rejected batch
and a wasted repair round - the third slice in a row where an agent definition asked for
output the contract forbids. Corrected, the status gates and the report-surface narrowing
added so the ask matches the enforcement, and the version bumped to 4 (the prompt hash is
stamped onto every disposition). A test asserts the contradictory line is gone.

## Phase 4 - the confidence-input assembler (verified)

`ConsequenceScore` existed from Phase 1 with nothing producing it; the judge slice wired it and
every score was an honest zero. This slice is the measurement. Numbers in
`docs/EVALUATION.md`, "Confidence inputs, as measured".

| module | responsibility |
|---|---|
| `app/judgment/confidence_inputs.py` | **new.** `assemble` - the eight factors, and `FactorCoverage` for what could not be reached |
| `app/application/use_cases/judge_consequences.py` | takes `outcomes` / `bindings` and measures per candidate; reports `confidence_coverage` |
| `tests/unit/judgment/test_confidence_inputs.py` | **new.** 52 tests, including the corpus-level measurement |

Neither the factor list nor the weights changed. Checkpoint 9.1 fixes the factors,
`app/domain/services/confidence.py` fixes the arithmetic, and this slice fixes only where each
number comes from - which is why `CONFIDENCE_INPUTS_VERSION` is separate from
`CONFIDENCE_FN_VERSION`: measuring authority differently changes the number without changing
the formula, and two runs differing only in that must still be diffable.

### Measured on the sample corpus

| query | distinct versions | authority | freshness | agreement | confidence |
|---|---|---|---|---|---|
| `refunds require manager approval` | 6 | 0.540 | 0.804 | 1.000 | **0.557** |
| `personal data ... 90 days ...` | 7 | 0.667 | 0.824 | 0.875 | **0.576** |
| `token expiry` | 1 | 0.556 | 0.883 | 1.000 | **0.401** |

Every manifest source states `effective_from`, so freshness is genuinely measured here. The one
factor a retrieval-only caller cannot reach is `explicitness` - retrieval returns spans, not
bindings - and `FactorCoverage` reports that rather than letting the zero read as "implied".

### The prediction this slice falsified

The judge slice's next-step note said corroboration on this corpus "should be low, because
lexical and vector reach the same source versions and the strategy gate is designed to say so."

**Wrong about which factor carries the signal.** Corroboration counts distinct *source
versions* and saturates at three, so a broad query over six or seven documents scores 1.0 and
the strategy gate never binds. The strategy dependence is real and lands on
`retrieval_agreement` at 0.875-1.000.

The relocation matters because the two factors point opposite ways: the formula scores high
agreement as **positive** (retrieval reliability), and on this corpus high agreement means
`HashingEmbedder` is bag-of-words over the same tokens as the lexical scorer, so the two
strategies confirm each other by construction. That is redundancy, not reliability. Filed as
issue 56 rather than fixed - re-weighting against one uncalibrated embedder would be tuning to
an artefact, and the weights are frozen for this slice by requirement.

The genuinely low case is visible in the same table: `token expiry` reaches one span in one
document, corroboration 1/3, confidence 0.401. The factor discriminates correctly when there is
something to discriminate.

### Authorization is a property, not a rule

| scope | distinct versions | authority | confidence |
|---|---|---|---|
| unrestricted | 6 | 0.593 | **0.489** |
| `privacy-office` | 7 | 0.667 | **0.502** |

The assembler measures only spans present in the `RetrievalOutcome` results, which the
retrievers already narrowed by tenant and by permission *before* scoring. So a withheld span
cannot enter the measurement at all, and **a narrower scope can never score higher**. The module
imports nothing from `app.infrastructure` and touches no store, asserted by parsing its imports
rather than grepping its text.

A cross-tenant candidate **raises** rather than being dropped: silently skipping one would turn
an isolation defect into a slightly lower number, which is the worst possible place for it to
surface.

### Three measurement decisions worth recording

**Agreement is reach, not score.** A BM25 score and a cosine similarity are not on a common
scale, so comparing them needs a normalization neither scorer supports - and any value produced
that way would describe the normalization. Whether two independent strategies both found a span
is a real, checkable fact.

**A candidate's own divergence does not penalize it.** For a Class A finding that *reports* a
contradiction, the divergence is the evidence. `contradiction_strength` is measured from
divergences whose participants are not the candidate's own input records; without the exclusion
every contradiction finding would be marked down for the thing it found.

**An undated version is excluded from freshness rather than scored stale.** Scoring it zero
would report an undated document as old, which the corpus does not say. It is excluded from the
mean and `FactorCoverage.unmeasured` says so - because a 0.0 meaning "measured as absent" and a
0.0 meaning "nothing to measure" are different facts about an analysis.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2732 passed**, 2 warnings, ~34s |
| `mypy app tests` | **success, 229 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 276 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

52 new tests. The eight required cases are covered, plus the corpus-level measurement, plus two
determinism tests (input ordering, and repeated runs over the real corpus).

## Phase 8 - the graph projection (verified)

The last frozen contract with no producer now has one. Design in `docs/ARCHITECTURE.md` 8.1,
edge sources in `docs/DATA_MODEL.md`.

| module | responsibility |
|---|---|
| `app/graph/projection.py` | **new.** `project(...)` - the whole projector, one pure function |
| `app/ingestion/identity.py` | `node_id_for` / `edge_id_for` added - content-derived graph ids |
| `tests/contract/test_layering.py` | `app/graph` declared the most tightly guarded layer |
| `tests/unit/graph/test_projection.py` | **new.** 65 tests, including the corpus demonstration |

The projector is the one component that must contain no judgement at all. Everything upstream
decides something; this decides nothing. It renders what those stages concluded, or it reports
that it could not.

### Four properties, each structural

**No agent can author a node or an edge, because the layer cannot reach one.** `app/graph` may
import `app.domain` and `app.ingestion` and nothing else - not `app.agents`, not any of the
four agentic packages, not `app.retrieval`, not `app.infrastructure`. Checkpoint 10.2 is a
property of the import graph, asserted twice: by the layering contract and by parsing the
module's imports with `ast`.

**Nothing is invented from absence.** Six skip reasons, each reported with the record id so a
reader can go and look at what the graph does not show. A projection that silently dropped a
contradiction and one that found none look identical otherwise.

**Type E never becomes an edge.** No relationship admits `Derivation.ABSENT`, so `GraphEdge`
is structurally incapable of carrying a gap - asserted directly. A gap annotates the node its
`partition` names, matched by **exact label**: interpreting its prose would put a judgement in
the projector.

**Two rebuilds are byte-identical.** Content-derived ids, sorted collections, and a
caller-supplied `projected_at` that also pins `created_at`. A test asserts the module calls no
`now()` at all.

### The `retention.window` demonstration

Measured over the real corpus - 21 bindings, 8 relations, no arbitrated consequences:

| | count |
|---|---|
| nodes | 22 (12 documents, 9 assumption keys, 1 change) |
| edges | 21, all `binds_assumption` |
| skipped | 8, all `no_arbitrated_consequence` |

`retention.window` has **five holders**: a Data Retention Policy, an EU Data Protection Policy,
a Customer Lifetime Value Metric Definition, Vendor Integration Notes and EMEA Support Team
Notes.

Lexical retrieval recovered **one** of them. So this is the gap between what text search shows
a reader and what co-binding establishes, and it is the entire argument for the mechanism:
nobody reading the retention policy would think to check the metric definition. Four other keys
show the same shape, `refund.approval_threshold` most strongly with six holders.

The taxonomy has no "document A mentions document B" edge, so every path between two holders
runs *through* the assumption-key node - which is the structural form of "coupling is derived
from co-binding rather than cross-reference". A test asserts that any direct edge between two
holders is `derived`, never a citation.

### What the measurement also shows

With no arbitrated consequences supplied, **no derived edge projects at all**. That is not a
defect: the three derived relationships are `CONSEQUENCE_BEARING`, so the edge validator
refuses one without a `consequence_id`, and the eight relations the resolver found are reported
as skipped rather than drawn. The graph shows what the pipeline *concluded*, not everything it
computed - and the difference is visible instead of silent.

### Two decisions worth recording

**A rejected consequence projects no edge, but its bindings survive.** My first test expected
an empty graph and was wrong: the `binds_assumption` edges are facts about what the documents
say, and a rejected *consequence* does not un-bind an assumption record. Only the derived
relationship the arbitration rejected disappears.

**Canonical ordering is the projector's job, not the resolver's.** `conflicts_with` between A
and B is one fact; the projector swaps endpoints into canonical order before minting the id, so
the entity's ordering validator never fires and the same relation named from either side yields
the same edge.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2799 passed**, 2 warnings, ~18s |
| `mypy app tests` | **success, 233 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 280 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

67 new tests: 65 in `tests/unit/graph/`, plus two more parametrized layering cases. All seven
edge types and all five node types are asserted reachable in one projection - a relationship
nothing can produce would be either dead taxonomy or a missing rule.

## The end-to-end orchestrator, and the two bars (verified)

Every stage was verified against its own inputs, which left the *pipeline* unproven: the
interfaces between stages had only ever been exercised by fixtures. That was the largest risk
in the repository. It is now closed. Design in `docs/ARCHITECTURE.md` 3a, numbers in
`docs/EVALUATION.md`, "End-to-end pipeline, as measured".

| module | responsibility |
|---|---|
| `app/application/orchestration/analysis.py` | **new.** `AnalyzeChangeUseCase` - the composition, and nothing else |
| `tests/fixtures/fake_agents.py` | **new.** A fake agent that answers from its own prompt |
| `tests/integration/test_analysis_pipeline.py` | **new.** 53 tests, including both evaluation bars |

No stage's logic was reimplemented and no stage's code changed. The orchestrator decides
order, dependency and what to report.

### The fake agent is the adversarial case, not a script

The default fake responder returns a digest that every agentic schema rejects, so a pipeline
run over it would be "quiet" because every stage *failed* - which proves nothing. A scripted
response is no better: pre-writing "no hypotheses" for the negative control would be asserting
the script.

So the fake agent **parses its own task contract** and answers for every handle and every
evidence id it was offered, at the strongest position the schema allows: it cites everything,
rates every divergence `material`, and asks the judge for `likely` on everything. Anything
weaker in the output was produced by a gate. The difference between the controls comes from
what the pipeline put in the prompt.

### Positive control

9 Class A findings, 6 retrieved spans, 1 Class B hypothesis, 10 consequences of which **9
material**, confidence 0.199-0.384, graph 23 nodes / 26 edges with all 5 `retention.window`
holders. The Class B consequence caps at `speculative` despite the fake asking for `likely`,
because it has no historical analogue.

### Negative control - and it is quiet across the whole pipeline

An espresso-machine maintenance change with no assumption delta: `class_a`, `retrieval`,
`class_b`, `contradictions`, `challenge` and `judgment` all **empty**. 0 consequences, 0
material, `is_quiet = True`, `succeeded = True`.

Getting there required one orchestration decision that the measurement forced. The first
version left the contradiction stage **corpus-scoped**, so an unrelated change still rated all
8 of the corpus's standing divergences - the analysis looked busy and paid for a provider call
to do it. Both `class_a` and `contradictions` are now scoped to the keys the change touches,
for the same reason: standing divergences are the knowledge-decay report's subject, not every
change's.

Quiet is **not silent**. One `EvidenceGap` is recorded, because a retrieval that returned
nothing over a readable corpus is exactly the negative provenance type E needs. The honest
quiet result is no findings *plus* a recorded gap.

### Ablation honesty

Corpus re-ingested without the retention policy, whole pipeline re-run. Class A 9 → 6,
material 9 → 6, retrieved 6 → 4, `retention.window` graph holders 5 → 4.

- **All 4 findings that rested on the removed spans disappeared**, matched by `finding_id`.
- **0 surviving findings, consequences or hypotheses cite a removed evidence id.** The sharpest
  anti-priors check available: the fake cites everything it is offered, so a removed id in a
  citation could only come from outside the corpus.
- **3 `retention.window` findings correctly survive** - five artifacts bind the key and two of
  them still genuinely disagree. My first test demanded the key vanish and was **wrong**: that
  would be demanding a false negative, which is the opposite failure and just as bad.
- **One finding reattributed** to a new governing declarer, because `resolve_authority` returns
  a different answer on the smaller corpus. Recomputation, not invention: a different finding
  with a different id, on spans that both still exist.

### Failure propagation

With every agent returning unparseable output: the four deterministic stages complete
untouched, the four agentic stages record `failed` with their details, `evidence_gaps` records
`skipped`, and `succeeded` is `False`. 9 Class A findings and 21 graph edges survive.

The same stage reads `empty` on the quiet change and `failed` here - requirement 5 as a
comparison rather than an assertion. A required deterministic stage failing raises
`AnalysisStageError` naming the stage, and the recorder's `__exit__` is annotated `-> None` so
the type checker forbids it ever suppressing one.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2852 passed**, 2 warnings, ~36s |
| `mypy app tests` | **success, 236 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 283 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

53 new tests. Every required scenario plus the two evaluation bars, all over the real sample
corpus through the real gateway with no credentials and no network.

## `FindingDisposition` - the human-signal loop (verified)

The last frozen contract with no producer. Design in `docs/ARCHITECTURE.md` 8.2, what it makes
measurable in `docs/EVALUATION.md`, "Reviewer disposition, as implemented".

| module | responsibility |
|---|---|
| `app/judgment/dispositions.py` | **new.** The rules, the envelope, the `DispositionStore` port, the foreground view, the signals |
| `app/application/use_cases/record_disposition.py` | **new.** `RecordDispositionUseCase` - record, change, read |
| `app/infrastructure/database/repositories/memory.py` | `InMemoryDispositionStore` added |
| `tests/unit/judgment/test_dispositions.py` | **new.** 52 tests |

No domain entity changed and no consequence-producing code was touched.

### Why this before the report composer

`already_known` is the failure **no internal metric can detect**. Precision, coupling recall
and confidence can all look excellent while the system reports only what a diligent reviewer
would have found anyway. The pipeline produces 9 material findings on the positive control and
until this slice there was no way to ask whether any was worth reading.

### One verdict per `(analysis_id, consequence_id)`

Identity is the pair. The same consequence reviewed in two analyses is two verdicts, because a
reviewer judges what they were *shown* - the evidence, the status and the confidence of that
run - and carrying a verdict forward silently would attribute an opinion to a report they never
saw.

### The two suppressions, and the one that never happens

| verdict | foreground | may request persistent suppression |
|---|---|---|
| `wrong`, `not_material` | **suppressed** | yes, scoped and expiring |
| `useful`, `already_known`, `unverifiable` | shown | **no** - the frozen entity refuses |

Foreground suppression is computed on read and reads **only the verdict**: no confidence
threshold, no severity, no age. Persistent suppression is scoped to
`(assumption_key, source_version_id)` and expires when the version changes - the reviewer said
the finding is wrong about *that text*, not that they never want to hear about the document.
Because the version id is content-derived, a revised document lifts the suppression by itself.

### Suppression never deletes, and that is what makes the signal safe to give

Nothing in either module constructs, copies or re-provenances a `Consequence`,
`ConsequenceScore`, `ArbitrationDecision` or graph record - asserted by parsing both files
rather than grepping them. A test serializes every consequence before and after a `wrong`
verdict and compares: identical. The finding is still evidenced, still scored, still in the
graph.

A verdict recorded with no way back is a verdict reviewers hesitate to give, and hesitant
feedback is worse than none. So `DispositionRecord` keeps every superseded verdict and one
`DispositionChange` per transition, with **both** values, the reviewer, the timestamp and the
reason - "changed to useful" is not auditable without knowing what it was. The port has no
`delete` and a test asserts the implementation's method set exactly.

### Three decisions worth recording

**`change` demands a reason whatever the value; `record` only for the two that subtract.** A
first verdict of `useful` needs no defence. A *change* is what a reviewer does when they got it
wrong, and an unexplained transition is the one thing an audit trail cannot use.

**`unverifiable` requires no reason.** "I could not tell" is a complete answer, and demanding
prose for it would push reviewers toward a value they can defend rather than the one that is
true.

**Idempotency is not "ignore the second one".** Re-submitting the same verdict returns the
existing record and appends nothing, so a double-click cannot fabricate a transition.
Equivalence compares the value, the note and the suppression request - and deliberately not the
id or the timestamp, since a retry carries a new one of each.

### A layering correction

My first attempt put `DispositionStore` in `app/domain/interfaces/repositories.py` alongside
the ingestion ports, with a `TYPE_CHECKING` import of `DispositionRecord`. That makes the
**domain import `app.judgment`**, which the layering contract forbids and correctly: the domain
must not know which envelope a later layer wraps its entities in. The port now lives beside its
envelope in `app/judgment/dispositions.py`, and the in-memory implementation stays in
infrastructure where it can import both.

### Verification

| command | result |
|---|---|
| `pytest -q` | **2904 passed**, 2 warnings, ~36s |
| `mypy app tests` | **success, 239 files**, strict mode |
| `ruff check .` | all checks passed |
| `ruff format --check .` | 286 files already formatted |
| `python -m scripts.check_spec_consistency` | **CONSISTENCY: CLEAN** |

52 new tests covering all eight required scenarios.

**No number is reported from this slice, and that is honest.** There are no reviewers and no
stored analyses, so the machinery is measured against synthetic verdicts and the *signal* is
not measured at all - it cannot be until a real reviewer looks at a real run.

## Analysis persistence (verified)

`AnalysisStore` + `StoredAnalysis` in `app/application/orchestration/analysis_store.py`,
`InMemoryAnalysisStore` in infrastructure. No Postgres yet.

**A snapshot, not a serialized `AnalysisResult`.** The result holds live working objects (the
coupling index, every retrieval candidate, each stage's internal outcome); storing those would
make the store a second copy of the pipeline's intermediate state, able to disagree with it.
`StoredAnalysis` keeps the review surface: analysis/run ids, the `ChangeSpec`, the
`RunProvenance`, the stage outcomes, one `ReviewedFinding` per consequence (claim + decision +
score + affected source version + whether the judge foregrounded it), the graph, the evidence
spans a citation resolves to, the gap ids, the open questions and the fingerprint.

**The fingerprint is stored verbatim, never recomputed.** It is what lets a verdict be checked
against the analysis that produced it: a re-run yielding a different fingerprint is a different
analysis, and a verdict from one does not apply to the other.

**Tenant.** Derived from the retrieval query - the boundary the analysis actually ran under -
and cross-checked against the Class A findings' own tenant; disagreement raises. Every read
takes `tenant_id` in its signature, and a cross-tenant read **raises `TenantMismatch` rather
than returning `None`**: an isolation defect must not look like an ordinary miss.

**Disposition validation.** `RecordDispositionUseCase.record_reviewed(...)` resolves the
analysis from the store, so membership is checked against what was stored for that tenant
rather than against a list the caller supplied. `foreground_reviewed` / `signals_reviewed`
reconstruct the review surface the same way. The list-based `record(...)` stays for a caller
with no store; both share one validation path.

**Suppression is now readable.** `suppressions(tenant_id)` and
`is_suppressed(tenant_id, assumption_key, artifact_version_id)` read
`DispositionRecord.suppression_scope` and delegate expiry to `suppression_has_expired` - no new
rule. The `artifact_ref` comes from the **stored finding's affected source version**, not the
caller, which is what closes issue 64: a `Consequence` records a title, and a title does not
change when a document is revised. A Class B finding stores `None` (a hypothesis spans several
artifacts, so no single revision refutes it) and is therefore never silently suppressed.

Measured on the positive control: 10 findings snapshotted, all with decision + score; every
Class A finding carries an affected version, the one Class B finding carries none; the stored
evidence set equals what the requester received.

| command | result |
|---|---|
| `pytest -q` | **2948 passed**, ~36s |
| `mypy app tests` | success, 241 files, strict |
| `ruff check .` / `format --check` | clean, 288 files |
| `check_spec_consistency` | CONSISTENCY: CLEAN |

44 new tests (33 unit, 11 end-to-end over a real pipeline run).

## The core intelligence layer (verified)

Three capabilities, integrated as one slice rather than three. The pipeline is now twelve
stages: `coupling -> class_a -> retrieval -> class_b -> evidence_gaps -> contradictions ->
challenge -> blind_spots -> evidence_arbitration -> judgment -> graph -> report`.

### Blind-spot hunter: `app/blindspots/`

A partition, not a personality. `EvidencePartition.__post_init__` **refuses to hold a span the
main consequence-hunter was offered**, so independence is a property of the type rather than a
claim about a prompt. Four slices in a frozen attribution order; the first two
(`single_holder`, `unwritten_rule`) are read off the existing decay report rather than
recomputed.

**It retrieves its own evidence through the same retriever** - three `model_copy` variants of
the main query changing only `text`, `top_k` and `source_types`, so tenant and `PermissionScope`
cannot be re-decided. Reading the coupling index directly would have been simpler and wrong:
the index is tenant-scoped but not permission-filtered.

**The third query is what earns the stage, and measurement is why it exists.** The first
implementation used only the change's own text at a widened depth plus the unqueried source
types; measured, the partition came out **empty** - all 6 spans it recovered had already gone
to the main hunter. Slices 3 and 4 are reachable only through the hook the main analysis
already pulled. Making the *undocumented keys* the query text moved the partition to 5 spans
and produced one finding on evidence the main analysis never saw.

Attribution is derived from the spans a finding cited, never taken from the model; a claimed
slice that disagrees lands in `misattributed_ids`. A self-reported restatement is dropped.
Corpus silence is a separate type with no claim, no mechanism and no citations - and is
deliberately **not** an `EvidenceGap`, because a gap carries negative provenance from an
executed query and this agent holds no trace.

### Evidence arbiter: `app/judgment/arbitration.py`

Answers on the **evidence axis** (`EvidencePosition`), never the status axis, and emits no
number. Independence is the factor that gets faked, so the model is not trusted with it: the
corroboration claim is intersected with the set `confidence_inputs.assemble` already measured,
and the **ceiling** on the position is computed from the measured count.

Three attacks foreclosed by absence rather than by a check: the schema has no field for an
agent, an agent count or an agreement level; the ceiling reads a *set* of `source_version_id`
so one document cannot be counted twice; and `ceiling_for` has no parameter for
`no_refutation_found`, exactly as `clamp` has none.

Disagreements leave on the ruling and reach the judge through
`dossiers_from(extra_disagreements=...)`, which has no trimming parameter. Nothing in the
status rules reads them, so an arbiter cannot raise a status by disagreeing eloquently.

### Report composer: `app/reporting/` + `compose_report.py`

**The executive summary is deterministic and the agent does not write the report's prose.**
That is a real narrowing of the role the agent definition describes (issue 68), and it is the
only way the no-new-factual-claims rule can be a property: sixteen sections that are all
projections of finalized records, and a summary assembled from counts by `summarize(...)`.

What the agent *is* admitted to do is structural and checkable: narrow the default surface (a
subset - and mandatory findings at `supported`/`likely` + HIGH/CRITICAL are restored whatever it
proposes) and order it (a preference; omissions are appended, never dropped). Its prose is
carried beside the report as advisory narration and refused **whole** if any identifier-shaped
token in it is absent from the finalized set.

### Reviewer feedback

One place: the report stage. Standing verdicts are read through the analysis **snapshot**
(`RecordDispositionUseCase.verdicts`, new), because the artifact version a suppression expires
on is a property of the stored finding. `wrong`/`not_material` leave the surface and stay in
`consequence_ids`; `already_known` suppresses nothing and is reported as a rate. A finished
analysis is then persisted, which closes issue 65 and is what gives the next run something to
read.

### Measured

| control | finding |
|---|---|
| positive | 11 consequences, 9 material, 9 foregrounded; partition 5 spans / 10 dropped as already seen / 1 finding; 11 rulings, 0 clamped, **all 11 corroboration claims trimmed** |
| quiet | 1 consequence (blind-spot, `speculative`), **0 material, 0 foregrounded**, `is_quiet` - and the partition is at its *largest* (9 spans) because the main query found nothing |
| ablated | 6 Class A (from 9), 6 foregrounded (from 9), and **3 of 8 rulings clamped to `weakly_supported`** for insufficient corroboration - ablation honesty on a second, independently built axis |
| challenger refutes | 11 rejected, 0 material, **11 of 11 rulings clamped to `disputed`** |
| reviewer, second run | foreground 9 -> **7**, 3 findings matched a standing verdict, 2 suppressed, already-known rate **0.33** |
| all agents broken | 6 agentic stages `failed` with details, gaps `skipped`, 9 Class A + 21 edges + a report survive |

**One verdict suppressed two findings**, and that is correct: the unit of a reviewer's rejection
is the `(assumption_key, source version)` they were looking at, not the consequence record.

### Three decisions worth recording

**`REPORT` is a required deterministic stage, and the composer's failure is not a stage
failure.** Composition cannot fail for a provider reason - every section is a projection - so a
failure there is a defect and aborts the run named. The agent's failure is recorded on
`ReportDocument.layout` and the report still exists with the deterministic surface. Collapsing
the two would have made "the model was unavailable" and "the composition code is broken" the
same status.

**The quiet control is no longer silent, and the definition of quiet did not move.** The
blind-spot stage legitimately produces a finding on a change the main query found nothing for -
that is the stage working, since every span is then unseen. `is_quiet` reads *material*
consequences and the default surface, both of which stay empty. Listing `BLIND_SPOTS` as
required-empty in the negative control would have demanded a false negative from the one stage
whose job is coverage.

**The evidence position is a second axis, never merged with status.** A reader is shown
`likely` (the judge) beside `supported` (the arbiter) and they mean different things: one is a
disposition under the frozen status gates, the other is what the evidence carries. Blending them
into one word is what a confidence number already does badly.

| command | result |
|---|---|
| `pytest -q` | **3048 passed**, ~37s |
| `mypy app tests` | success, 256 files, strict |
| `ruff check .` / `format --check` | clean, 303 files |
| `check_spec_consistency` | CONSISTENCY: CLEAN |

100 new tests: 74 unit (23 blind spots, 29 arbitration, 22 report composer) and 26 end-to-end,
including the ten named evaluation checks in `TestCoreIntelligenceLayer`.

## PostgreSQL persistence (verified)

`app/infrastructure/database/`: `models/tables.py` (twelve tables), `codecs.py` (the two
non-Pydantic envelopes), `repositories/sql.py` (the seven stores), `session.py` (the
`Database` handle), and `migrations/versions/0001_initial_persistence_schema.py`.

### Every port, and what backs it

| port | tables | notes |
|---|---|---|
| `SourceStore` | `source_documents`, `source_versions`, `chunks`, `evidence_items` | canonical text on the version row; tenant reached by join, never duplicated onto a span |
| `AssumptionStore` | `assumption_records` | refuses an unadmitted record, same message as in memory |
| `RegistryGapStore` | `registry_gaps` | `put_gap` accumulates the occurrence count |
| `AliasStore` | `key_aliases` | `approved()` filters on `status`, the field the property reads |
| `CouplingIndexStore` | `coupling_sets`, `derived_relations`, `class_a_findings` | tenant slice replaced wholesale, never merged |
| `DispositionStore` | `finding_dispositions` | history and superseded verdicts inside the payload |
| `AnalysisStore` | `stored_analyses` | cross-tenant read raises; fingerprint stored verbatim |

### The mapping, and the alternative it was chosen against

JSONB payload plus projected columns, not a normalized schema. Normalizing would put the rules
the frozen value objects already encode into CHECK constraints as well, and the two would
drift; round-tripping through `model_validate` means **every read re-runs every validator**.
And the ports ask no relational questions - every method is an id lookup, a tenant enumeration
or a filter on one key. The cost is SQL-level queryability inside an entity, which is the first
thing to revisit if a reporting surface needs it.

Three decisions worth recording:

**Projected columns can only make a filter wrong, never a value.** They are write-time
projections and the served value always comes from `payload`. A test corrupts a `status` column
and asserts exactly that asymmetry - the filter follows the column, the entity follows the
payload.

**Ordinals, because SQL has no insertion order.** `versions_for` is "oldest first" and
`coupling_sets` returns the resolver's tuple. Without a stored ordinal the two implementations
would satisfy the type and disagree about the answer, which is the failure mode a second
implementation actually has.

**One transaction per call, with an opt-in unit of work.** The faithful mapping of a port that
has no transaction. `Database.unit_of_work()` exists for the one case that needs it - the three
coupling-projection writes - and refuses to nest rather than flattening.

### Verification, and what it does and does not prove

The schema is created by **Alembic** on the tested path, not by `create_all`.

| check | result |
|---|---|
| migration vs `Base.metadata` | identical across columns, nullability, types, PKs, indexes, unique constraints and FKs, table for table |
| `upgrade -> downgrade base -> upgrade` | same schema, empty in between |
| offline Postgres DDL | 12 `CREATE TABLE`, **12 `payload JSONB NOT NULL`**, compiles with no server |
| single head, every revision has a real downgrade | asserted |
| `alembic.ini` carries no URL | asserted |

**Restart.** `tests/integration/test_persistence_restart.py` runs
`python -m tests.fixtures.persist_corpus <url>` as a **subprocess** - real ingestion of the
sample corpus, the coupling projection, the whole twelve-stage pipeline, two reviewer verdicts
and one reversal - then reads everything back in the test process. A second engine in the same
interpreter would share an import graph and a settings cache, so it could pass while nothing
reached disk; a separate interpreter shares nothing but the database.

Written and read back: 13 documents, 21 admitted assumption records, 2 open registry gaps, 10
coupling sets, 8 derived relations, 14 standing Class A findings, 9 change-relevant Class A
findings, 11 consequences with their decisions and scores, 27 graph edges, 6 retrieved evidence
spans, 2 disposition records with a one-transition audit trail, and the fingerprint verbatim.
Suppression expiry re-verified against the stored source version: it holds for the version the
reviewer saw, lifts for a different one, and never catches a Class B finding.

**The substrate is the honest caveat.** This environment has no Postgres server, no Docker and
nothing on 5432, so the suites run on SQLite - the *same* repository code, ORM models and
Alembic revision, with `JSONB` carrying a `JSON` variant. That exercises the SQL, the mapping,
the codecs, the ordering, the tenant predicates, the transaction boundaries and durability
across a restart. It proves nothing about JSONB indexing, Postgres type coercion or behaviour
under Postgres' isolation levels. Setting `CCE_TEST_DATABASE_URL` runs the identical tests
against the real server, `docs/RUNBOOK.md` has the command, and
`TestSubstrate::test_the_tests_report_which_substrate_they_ran_against` skips rather than
passing quietly so a green run says which one it was.

| command | result |
|---|---|
| `pytest -q` | **3162 passed, 1 skipped**, ~38s |
| `mypy app tests` | success, 265 files, strict |
| `ruff check .` / `format --check` | clean, 313 files |
| `check_spec_consistency` | CONSISTENCY: CLEAN |

115 new tests - 114 passing and 1 skipped, the skip being the substrate marker that stays
skipped until a Postgres URL is configured: 72 repository conformance (**31 shared assertions
run against both implementations**, plus 10 that are about being SQL), 20 restart, 11 migration
contract, 12 codec.

## The blind-spot ordering fix (verified)

`blind_spots` moved ahead of `challenge`, so the pipeline is now
`... contradictions -> blind_spots -> challenge -> evidence_arbitration -> judgment ...`.
Orchestration only: no domain model, no persistence, no agent definition and no stage's logic
changed.

**What was wrong.** The challenger ran first, so the blind-spot hunter's findings reached the
judge with no `ChallengeResult` on their dossiers. The newest and least corroborated findings
in the analysis were the only ones nobody had tried to refute - which is the wrong way round
for a stage whose whole premise is that it works a thinner evidence basis than the main hunt.

**What changed, in three places.**

1. `AnalysisStage` - the two members swapped, which is what `execute` and every stage-order
   test read.
2. `_stage_challenge` - takes the blind-spot report and passes the **union** of the two hunts'
   retained hypotheses, plus the assembled span map rather than building its own from
   retrieval alone. A failed hunt contributes an empty tuple, so degradation stays automatic.
3. `_Candidates.of` - both hypothesis sets are now treated identically. Previously the
   blind-spot set was passed to the judge as raw hypotheses *in addition to* the challenge
   targets; now, where the challenger ran, its targets carry the established `consequence_id`
   for every candidate including the blind-spot ones.

**The span map is the part that was easy to get wrong.** A blind-spot finding cites partition
spans, which are by construction absent from the main retrieval result. Handing the challenger
only the main spans would have left the candidate with nothing citable, so it could only ever
return `no_refutation_found` - a refusal that reads exactly like a survived challenge. The map
is therefore assembled before the challenge stage rather than after it, and a test asserts the
blind-spot target's citable set is a subset of the partition.

### Measured

| | before | after |
|---|---|---|
| challenge targets, positive control | 10 | **11** |
| blind-spot findings reviewed | 0 of 1 | **1 of 1** |
| `agent_sources` on a blind-spot consequence | hunter + judge | + **`adversarial-challenger`** |
| rulings clamped, challenger refutes | 10 of 11 | **11 of 11** |
| blind-spot consequence, challenger refutes | `speculative` | **`rejected`** |
| quiet control: material / foregrounded | 0 / 0 | **0 / 0** (unchanged) |

`no_refutation_found` still raises nothing: a surviving blind-spot finding stays
`speculative`, because the frozen status gates have no parameter for a challenger looking and
finding nothing. The negative control is unaffected - the blind-spot hypothesis on a quiet
change is now challenged and still comes out immaterial and off the default surface.

**One test changed rather than being added to.** `CHALLENGE` left the quiet control's
must-be-empty list: it now has that one hypothesis to review and correctly reports
`completed` with one result. A challenger reporting *nothing* while an unreviewed hypothesis
was on its way to the judge is precisely the failure this fixed, so asserting it stays empty
would have been asserting the defect.

**Failure isolation, both directions** - they are adjacent stages now, so both matter. A
broken blind-spot hunter leaves `blind_spots` `failed` and the challenger still reviews Class A
plus the main hunt (10 targets). A broken challenger leaves `challenge` `failed` and the
blind-spot finding still reaches judgment as an unreviewed candidate rather than being dropped
with the stage that could not review it.

`ANALYSIS_PIPELINE_VERSION` is now `analysis-pipeline-2`, because the order changed and the
label exists to say so.

| command | result |
|---|---|
| `pytest -q` | **3167 passed, 1 skipped**, ~70s |
| `mypy app tests` | success, 265 files, strict |
| `ruff check .` / `format --check` | clean, 314 files |
| `check_spec_consistency` | CONSISTENCY: CLEAN |

5 new tests, 1 rewritten.

## The composition root (verified)

`app/composition.py`. Until it existed, the PostgreSQL stores were implemented, migrated and
tested and **nothing constructed them** - issue 72.

One frozen dataclass and two constructors. `Application.configured()` is the production path:
settings, `Database.configured()`, `SqlStores.over(...)`, then the ingestion and analysis use
cases wired over the result. `Application.build()` takes the seven stores as parameters, which
is what keeps the in-memory composition first-class rather than a parallel wiring that can
drift - the store fields are typed as the **ports**, so the graph does not know which it got.

**No container, and that is the decision.** Wiring this graph is a hundred lines of constructor
calls; the value of writing them out is that a reader can see what depends on what without
learning a framework's resolution rules. No web framework, no worker, no DI library, no new
abstraction - asserted by a test in the module where they would be added by reflex.

Three things checked at construction rather than discovered later:

- **an unset `CCE_DATABASE_URL`** raises `DatabaseNotConfigured` naming the variable;
- **a missing agent definition** raises `CompositionError` naming the agent, instead of failing
  four stages into a run;
- **an unavailable embedder** yields `None` and `RetrievalService` then refuses a vector
  request explicitly. No silent fallback to lexical: a caller who asked for semantic retrieval
  and got lexical would have no way to know.

`python -m app.composition` constructs everything and prints a readiness report - dialect in
use, servable retrieval strategies, registry version, loaded agents, and whether the database
is reachable and migrated. It reports the **actual** dialect off the engine rather than the
string "postgres", because the production target and the test substrate differ and an operator
needs to be told which one they got.

One overlap left deliberately: `app/api/deps.py` keeps its own per-process gateway cache for
the health routes. Six lines of constructor calls are duplicated, and merging them would trade
that for a composition root importing the web layer. The lifetimes differ on purpose - those
are process-wide singletons serving requests, this is one owned graph a caller closes.

### Verification

One end-to-end test through `configured()`, not `build()`: a test that reached for `build()` to
avoid needing a credential would be testing a different function from the one that ships. It
ingests the real sample corpus into the application's own stores, resolves the coupling
projection inside a unit of work, runs all twelve stages, reads back the snapshot the
orchestrator persisted, and records a suppressing reviewer verdict against it. Substrate and
provider follow the existing rules: SQLite by default with Postgres on
`CCE_TEST_DATABASE_URL`, and the cooperative fake through the real `LlmGateway`.

| command | result |
|---|---|
| `pytest -q` | **3179 passed, 1 skipped**, ~64s |
| `mypy app tests` | success, 267 files, strict |
| `ruff check .` / `format --check` | clean, 316 files |
| `check_spec_consistency` | CONSISTENCY: CLEAN |

12 new tests.

## The specialist fan-out (verified)

`app/specialists/` plus `app/application/use_cases/consult_specialists.py`. The pipeline is now
thirteen stages: `... retrieval -> specialists -> class_b -> ...`.

Seven analysts read their own evidence slice and return evidence-backed **observations**. The
existing frozen `SpecialistFinding` carries them - so no new persistence model, and the entity
has no field for a class, a confidence, a severity or a status, which is the strongest form the
"must not bypass arbitration" requirement can take.

### The integration decision

The specialists feed the **existing** Class B stage. The consequence-hunter's own definition is
"generate consequence hypotheses by combining independent evidence from multiple domains"; the
fan-out is what makes that literal by producing the domains. Observations reach it as
attributed context alongside the union of the slices' spans, and from there a hypothesis is
challenged, arbitrated and judged like any other.

The alternative - converting observations into candidates directly - was rejected. It would
have meant asking each specialist for a mechanism and a weakest link, which is the
consequence-hunter's job, and the result would have been seven parallel hunters with one
report. There is one path from evidence to a consequence and the specialists feed its front.

### Four gates, and the one that is only defence in depth

**Citations are checked against *this* specialist's slice.** Another analyst's id is dropped
exactly like a fabricated one, so a specialist that cited only somebody else's slice ends up
with no evidence - which the ceiling then caps.

**The position is a ceiling computed from two deterministic facts**: whether a citation
survived, and whether the specialist called the observation an inference. No parameter for how
sure it sounded. An uncited claim is `unresolved`, never `unsupported` - nobody produced
evidence either way.

**The four pipeline-assigned fields have nowhere to arrive.** `extra="forbid"` on the proposal
schema, and the frozen entity has no field for any of them either.

**Observed bindings are recorded and never admitted** (`CLAUDE.md` 20, 21). The definitions ask
for them and they are reported, as proposals, with the value left as rendered text. This one is
defence in depth rather than the only gate: the closed registry and the ingestion-time
admission gate would refuse them anyway, and there is no store write on this path at all.

### Two suppression rules

Cross-specialist duplicates collapse on `(claim, evidence set)` in roster order. A Class A
restatement - an observation on a decided key resting *only* on spans that finding already
materialized - is suppressed. An observation on a decided key citing other evidence survives,
which is the half that makes the rule safe: suppressing everything on a Class A key would
silence the analysts exactly where the analysis is densest.

### Measured

| control | result |
|---|---|
| positive | **3 of 7 invoked** (policy 3 spans, process 2, data 1); 4 observations retained, 2 suppressed, 1 fabricated citation dropped, 3 bindings recorded and none admitted |
| quiet | **0 of 7 invoked**, all seven reported un-selected, stage `empty` - seven calls not made |
| one analyst broken | stage `failed` naming it, the other two analysts' findings retained, everything downstream complete |
| all analysts broken | stage `failed`, the hunter runs alone and still yields 11 consequences |
| not configured | `not_configured`, and the Class A finding ids match the agentic run exactly |

**A measurement that went against the design's assumption, recorded rather than smoothed
over.** On this corpus the three slices *partition* the six spans the main retrieval already
returned: zero spans were reached only by a specialist. The main query's `top_k` of 10 recovers
every lexically matching span, so the fan-out's value here is per-domain *reading*, not extra
recall - and the slice-union mechanism that would add recall is built and unexercised. It would
matter on a corpus where `top_k` truncates, or with a semantic embedder. Claiming otherwise
now would be claiming something the corpus does not show.

Two more decisions worth recording:

**Seven lexical retrievals per analysis, always.** Slices are retrieved for all seven and only
the non-empty ones are *invoked*, because the expensive thing is the provider call and the
selection reason has to be a fact rather than a guess. In-process lexical retrieval is cheap;
if it stops being cheap the fix is to pre-filter on the corpus's source-type distribution.

**One analyst's failure marks the stage `FAILED` while its siblings' findings survive.** That
is deliberate and consistent with every other agentic stage - the failure is named in the
stage detail, `retained` is not empty, and downstream runs. It does mean a single specialist
outage makes `result.succeeded` false, which is the same treatment a Class B failure gets.

| command | result |
|---|---|
| `pytest -q` | **3230 passed, 1 skipped**, ~68s |
| `mypy app tests` | success, 274 files, strict |
| `ruff check .` / `format --check` | clean, 323 files |
| `check_spec_consistency` | CONSISTENCY: CLEAN |

51 new tests: 34 unit (roster and selection, slice queries, the ceiling, the gate, the schema,
both suppression rules), 15 end-to-end over the real corpus covering the seven required
properties, and 2 layering-contract parameters for the new guarded layer.

## The historical analogue (verified)

`app/specialists/analogues.py`, plus an admission check in both hunters. This closes issue 74,
which was the last reachability gap in the Class B story: checkpoint 9.2 rule 2 says a
behavioural claim cannot exceed `speculative` without a historical analogue carrying an
observed outcome, and nothing in the system could produce one.

### How the analogue is represented

`HistoricalEvent` — which **already existed** in `app/domain/entities/source.py`, with
`qualifies_as_anchor` already spelling out the rule. No new domain model, no new table, no new
store. The prediction in issue 74 that this needed "one store and one ingestion path" was
wrong, and the reason is worth keeping:

**The event id is derived from the cited spans, not assigned by a store.**
`event_id_for(tenant_id, evidence_ids)` hashes the tenant and the sorted span ids, so two
analyses citing the same spans agree on the id without coordinating. The anchor is therefore
verified by *re-derivation* rather than by lookup, which is what removes the store from the
design. The title is deliberately **not** part of the identity: one incident described twice is
one precedent, so a model cannot split it in two by renaming it, nor corroborate itself by
describing it twice.

### Four admission rules, all deterministic

**There is no id field on the proposal schema.** `AnalogueProposal` has nowhere to put an
`he_...`, so "do not let the model invent an EventId" is enforced by absence rather than by a
check. A model wanting to influence the id would have to change which spans it cites, and those
are checked against its own slice.

**Citations are checked against the historical analyst's slice**, which came from the same
retriever under the same `PermissionScope`. An id from outside is dropped exactly like a
fabricated one, so an event can only rest on spans the requester was authorized to read.

**An event with no admitted citation is refused outright** — not recorded-but-unqualified.
Without a span there is nothing to be traceable to, and an untraceable event record is worse
than none.

**Only a recorded `observed_outcome` anchors.** An event with citations but no measured outcome
is *kept* and reported, and `qualifies_as_anchor` is `False`. That is the analyst's own
instruction expressed as a property of the record rather than a request in a prompt. A future
`occurred_at` is refused: an analogue that has not happened is a misreading, not a precedent.

One more rule sits outside `admit`: only the `historical-analyst` may propose an analogue at
all. One agent's definition carries the anchor obligation, so one agent may discharge it.

### The defect this found

The consequence-hunter's proposal has always carried `historical_analogue_id`, and **nothing
validated it**. A model emitting a well-formed invented `he_...` would have satisfied rule 2 and
lifted a Class B finding above `speculative` on an event that never existed. So the gate was
unreachable in the raising direction *and* open in the wrong one, and only building the
reachable half made the open half visible. The hunter now drops any anchor outside the
admissible set and reports it as `fabricated_analogue_ids`; the blind-spot hunter is offered no
anchors at all, so its anchors are dropped unconditionally and every blind-spot hypothesis
stays in `without_analogue_ids`.

The fakes were updated to be adversarial about exactly this: the consequence-hunter claims the
real anchor when offered one and an invented one otherwise, the blind-spot hunter always claims
an invented one, and every selected specialist proposes analogues whether or not it is entitled
to.

### Measured — the anchored control

A change to `refund.approval_threshold` (10,000 → 5,000 USD), which is the one change this
corpus can anchor: INC-4471 records a *measured* outcome for the previous reduction of the same
threshold.

| | |
|---|---|
| stages | 13 ran, 0 failed, 0 skipped (`evidence_gaps` empty) |
| Class A findings | **10** (6 `conflicts_with`, 4 `invalidates_assumption`) |
| retrieved spans | 9 |
| specialists | **5 of 7** invoked (policy 3 spans, process 4, people 1, risk 1, historical 1) |
| observations | 5 retained, 5 suppressed |
| historical events | 3 proposed → **1 admitted, 1 anchor** (`he_8cce2b46…`) |
| analogues refused | 13: **12 wrong agent**, 1 uncited |
| Class B hypotheses | 1 main hunt, 1 blind-spot partition |
| consequences | 12 (11 `likely`, **1** `speculative`) |
| **material** | **11** |
| graph | 25 nodes, 27 edges |

**The main-hunt Class B consequence is judged `likely`, carrying
`historical_analogue_id = he_8cce2b46…`.** It is the first non-speculative Class B finding the
project has produced, and it got there through the existing frozen gates: no new confidence
mechanism, no new scoring, no new status. Rule 2 simply stopped firing because the precedent
exists.

The blind-spot Class B consequence in the same run stays `speculative` with no analogue, which
is the contrast that makes the first number meaningful — one run, two Class B findings, and the
one with a precedent is the only one that rose.

**Three proposals collapsed to one event, and the reasons matter more than the count.** The
historical slice on this corpus is a **single** span, so the fake's outcome-carrying and
outcome-less proposals cite the same evidence — and identity is the spans, so they are one
event described twice. The recorded-without-outcome path is therefore unreachable end to end
here and is covered directly by unit tests instead of pretending otherwise.

**Permission filtering, demonstrated rather than asserted.** With INC-4471 labelled
`support-operations` in a manifest copy, the same change run twice gives: granted → the
incident's spans are retrieved and one anchor is admitted; denied → none of its spans are
retrieved, no event exists, and the Class B finding is back to `speculative`. The filtering
happens in retrieval, before the analyst is asked, which is the property that matters — an
analogue is the one output that can *raise* a status, so a leak here would hand an unauthorized
reader a **stronger** finding rather than a redacted one.

**Degradation is downward.** Break the historical analyst and the stage is `failed` naming it,
the other four analysts' findings survive, everything downstream completes, and the Class B
finding falls back to `speculative` rather than keeping a status it can no longer justify.
Refute the anchored hypothesis and the challenger still wins: the analogue raises the ceiling,
it does not overrule a refutation.

### One rough edge, recorded

`refused_analogues` carried **12** wrong-agent refusals in the measured run — four non-entitled
specialists × three proposals each. The gate is working and every refusal is accounted for, but
the field is noisy to read, and that noise is an artifact of the fake having every specialist
propose an analogue. A real provider following the prompt would leave `historical_events` empty
for the other six. Worth collapsing per agent if it ever reaches a report surface.

| command | result |
|---|---|
| `pytest -q` | **3264 passed, 1 skipped**, ~74s |
| `mypy app tests` | success, 276 files, strict |
| `ruff check .` / `format --check` | clean, 325 files |
| `check_spec_consistency` | CONSISTENCY: CLEAN |

34 new tests: 17 unit over the admission gate (the derived id, the tenant and span
sensitivity, each of the four refusals, the agent gate) and 17 end-to-end over the real corpus
covering the six required properties.

## Next step — the fifth retrieval slice

The benchmark is delivered, so the three open issues it was built to settle are now
measurements rather than guesses: coverage is 90.3% and the unreachable share splits into
extraction and *registry* coverage (25); the missing stemming costs the affected category
80% of its recall, flat across *k* (26); and authorization cost nothing in recall on a
one-document overlay, which is a weaker statement than the issue asked for (30).

Both cheap lexical changes have since landed, each alone and each re-measured. The stopword
list closed issue 32 (24 negative-control results → 5, recall up at @1/@3/@5, nothing lost);
the guarded plural rule closed issue 26 (recall@10 80.4% → 91.1%, six missed expectations
recovered, control noise unchanged). Recall@10 over the reachable labels is now **91.1%**.

**The cheap knobs are spent, and that is the useful conclusion.** What is left in the
tokenizer is a second suffix rule, then a third, then a stemmer — each smaller than the last
and each risking a wrong merge, which the guard tests exist to make expensive. The remaining
misses are no longer morphological:

- 4 of the 5 remaining misses are `key_retention_window`: the coupling set of an assumption
  key, which text search cannot reconstruct because the policy that governs writes "retained
  for 90 days" and never the key. No lexical rule reaches that.
- the 5th is `lexical_miss_token_expiry_wording`: "the link is valid for 24 hours" against
  "token expiry", with no shared word form under any inflection.
- `paraphrase` and `known_lexical_miss` are at 100% and 80% @10, so the categories that
  motivated both tuning changes have little left to give.

Vector retrieval has since landed against that reasoning, and **the measurement did not go
the way the plan assumed.** The strategy works — same filters, same authorization
guarantees, zero leaks, deterministic — and measures recall@10 80.4% against lexical's
91.1%, finding a strict subset of what lexical finds: 0 unique finds across 56 reachable
expectations. Paraphrase recall, the category that motivated the slice, got *worse*.

That result has one cause and it is known: the only embedder available offline
(`HashingEmbedder`) is feature hashing over a bag of words and carries no semantics, so it
cannot reach a paraphrase and has no suffix rule to match `refund` against `Refunds`. The
slice therefore delivered the machinery and a calibrated *negative* result about the
embedder, not about vector retrieval.

So the smallest useful next step is **one recorded run against a real embedding provider**
(issue 36), and the adapter foundation for it is now in place: `HttpEmbeddingProvider` speaks
the wire format, the configuration path takes a model identifier and a credential from the
environment with no defaults, per-vector width validation guards against a mistyped model,
and `ReplayEmbedder` reads a committed fixture and raises rather than substituting. The
benchmark already takes an embedder as a parameter and pins the vector scorer separately, so
the figures must be re-measured rather than reprinted.

**That run has now been attempted and is blocked on the credential alone** — see "Phase 3 —
the real embedding experiment". Everything around it is built and proven offline:
`scripts/record_embeddings.py` does the call, and 30 tests drive recording → replay →
benchmark through a mock transport. What remains is four environment values and one command:

```bash
# Gemini, gemini-embedding-001 at 768 dimensions. The adapter is prepared and tested; the
# key belongs in .env (gitignored) or the shell, never in .env.example.
export CCE_EMBEDDING_PROVIDER=gemini
export CCE_EMBEDDING_ENABLED_PROVIDERS='["hashing","replay","gemini"]'
export CCE_MODEL_EMBEDDING=gemini-embedding-001
export CCE_EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta
export CCE_EMBEDDING_DIMENSIONS=768
export CCE_EMBEDDING_API_KEY=...            # never committed, never logged

python -m scripts.record_embeddings --check  # preflight: expect no problems reported
python -m scripts.record_embeddings          # the one live call: 51 distinct texts

export CCE_EMBEDDING_PROVIDER=replay         # from here on, offline forever
make retrieval-benchmark-vector              # then update the suite's vector_scorer_version pin
```

Three things to expect rather than hope for. The first live call is a debugging exercise
against an unverified wire contract (issue 41). A dense embedder has no zero cosines, so the
negative-control figures will get *worse* and the similarity-floor question (issue 38)
becomes answerable for the first time. And the `vector_scorer_version` pin will refuse the
run until it is updated, which is the guard working.

Hybrid fusion has since been built and measured anyway, and the measurement agrees with
that reasoning rather than contradicting it: RRF over the current two inputs recovers the
lexical set exactly and ranks it worse, because the weak input supplies eight times the
volume and no unique finds. So fusion is **implemented and not the default**, and the
question it answers is still waiting on the same thing - a real embedder that contributes a
find lexical misses.

The agentic layer has since moved ahead of retrieval, deliberately: the
`consequence-hunter` and the `EvidenceGap` producer are both built, and both are limited by
the same thing every retrieval slice is — the corpus, not the machinery.

The `contradiction-hunter` has since been built on exactly that reasoning, and the negative
control held: on a corpus with no deterministic divergence it makes no provider call at all.

The `adversarial-challenger` has since been built on exactly that reasoning. Both predicted
behaviours hold: a mechanism objection to a Class A finding is refused, and a clean candidate
yields `no_refutation_found` with a falsification test rather than an invented objection.

The `consequence-judge` has since been built, and it closed the pipeline: four agentic stages
now feed one arbitration that emits `Consequence`, `ArbitrationDecision` and a **computed**
`ConsequenceScore`. Issue 51 is resolved and 52 is half-resolved - the judge consumes the
upstream records, and nothing yet consumes the judge's.

Confidence is now measured rather than wired (issue 54 resolved), and the measurement was
useful in the way a measurement should be: it falsified the prediction recorded here about
where the low-independence signal would land, and it turned that into two calibration issues
(56, 57) instead of a guess.

The graph projection has since been built, and it made the central claim visible: five
artifacts bind `retention.window`, lexical retrieval recovered one, and every path between two
holders runs through the assumption-key node because the taxonomy has no citation edge.

The pipeline is now composed and both evaluation bars are measured (issue 59 resolved). The
measurement was useful in the way a measurement should be: it forced one orchestration
correction (the contradiction stage was corpus-scoped and made an unrelated change look busy)
and it falsified one of my own test expectations (demanding the ablated key vanish entirely,
which would have been a false negative).

`FindingDisposition` is built, so **every frozen contract in the checkpoint now has a
producer**. What the repository lacks is no longer a component; it is the thing that would let
the components hold state between calls.

The analysis store is in place, so issues 61, 63 and 64 are closed and the reproducibility
diff (checkpoint 13.21) is now possible - two runs both persist, and
`AnalysisResult.fingerprint()` is the comparison.

**Working mode from here: terse.** One-line docstrings, comments only where a choice is not
obvious from the code, ~15-25 tests per slice covering the required cases, STATUS entry only
unless a contract changes, and larger slices per turn. The design decisions are settled and
recorded; the remaining work is execution.

**The core intelligence product is complete.** Every frozen contract in the checkpoint has a
producer and a consumer, the twelve stages compose, both evaluation bars hold, and reviewer
feedback closes the loop back into the next analysis.

Postgres and Alembic are delivered across all seven ports, so analyses now survive a restart
and the reviewer loop is no longer confined to one process.

**Next: one verification run against a real Postgres server** (issue 70). Everything is built
and exercised on SQLite; what is untested is JSONB on the production dialect, Postgres type
coercion, and concurrent writes under its isolation levels. It is one `docker compose up`, one
`createdb`, one environment variable and one `pytest` invocation - `docs/RUNBOOK.md` has all
four - and until it happens the persistence claim carries a named caveat rather than a
measurement.

**The core intelligence layer is complete.** Every agent definition in `.claude/agents/` now
has a runtime, a gate and a place in the pipeline; every frozen contract has a producer and a
consumer; the deterministic Class A path is unchanged by any of it.

What remains is not core intelligence. In order:

1. **One verification run against a real PostgreSQL server** (issue 70) - everything is
   exercised on SQLite and the caveat is named.
2. **A first real reviewer pass** - now possible across sessions, and the only thing that turns
   `already_known_rate` from measurable into measured.
3. **One real embedding run** (issue 36) - still the only way to settle whether vector
   retrieval helps, and the thing that would make the specialist slices add recall rather than
   re-read the same spans (issue 73).
4. **A second anchorable change** - the analogue path works (issue 74 is closed) but this
   corpus records a measured outcome for exactly one incident, so exactly one change can be
   anchored. One anchored finding is a demonstration, not a measurement.

UI, an API surface and production hardening are the natural stopping point.

Then, in order:

1. a stored vector index keyed on (corpus version, embedder), against issue 37. Re-embedding
   the corpus per query is a cost-discipline violation, not just slow, and it is the thing
   that has to be fixed before any of this runs on a real corpus.
2. metadata-only retrieval, which is a contract change rather than a strategy.
3. a `QueryPlan`, hybrid fusion and a reranking abstraction — which also needs issue 39
   resolved, since `RetrievalPort` currently cannot describe both strategies uniformly.
4. analysis-level scope keying, once there is an analysis store to key.

The deterministic layer stays a fixed point for all of it, and the benchmark strengthened
that argument rather than weakening it: lexical retrieval recovered 1 of the 5 artifacts
binding `retention.window`, so a coupling set reconstructed by text search would have missed
the governing policy. Class A does not depend on retrieval, and now there is a number
saying why it must not. It already produces the Class A set without retrieval, so no
regression in retrieval can damage Class A recall. What retrieval unlocks is Class B — and
`class_b_rediscovery_rate` is already wired to catch the failure mode where the agentic
layer merely re-verbalizes what the resolver found.
