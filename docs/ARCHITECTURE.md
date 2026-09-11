# CCE Architecture

## 1. Architectural principles

CCE uses a layered architecture:

1. API / presentation
2. application orchestration
3. agent runtime
4. LLM gateway / provider adapters
5. retrieval
6. evidence, assumption store, and knowledge graph
7. domain persistence
8. ingestion/connectors
9. observability/evaluation

The architecture keeps agent prompts in Markdown and runtime behavior in Python.

No layer above the gateway knows which vendor served a request.

## 2. High-level architecture

```text
                         ┌──────────────────────────┐
                         │       Client / UI         │
                         └────────────┬─────────────┘
                                      │ HTTPS
                                      ▼
                         ┌──────────────────────────┐
                         │       FastAPI API         │
                         │ auth / validation / IDs   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Analysis Application      │
                         │ Use-case orchestration     │
                         └────────────┬─────────────┘
                                      │
                      ┌───────────────┼────────────────┐
                      ▼               ▼                ▼
               Agent Router     Retrieval         Run Manager
                      │           Service               │
                      ▼               │                 │
        ┌────────────────────────┐    │                 │
        │  Claude Code subagent  │    │                 │
        │ definitions in MD      │    │                 │
        └────────────┬───────────┘    │                 │
                     │                │                 │
                     ▼                ▼                 ▼
             Specialized agents   Evidence Store   Job/Trace Store
                     │                │                 │
                     └───────────┬────┴─────────────────┘
                                 ▼
                      ┌──────────────────────┐
                      │ Consequence Graph    │
                      │ nodes / edges /      │
                      │ provenance / status  │
                      └──────────┬───────────┘
                                 │
                 ┌───────────────┼─────────────────┐
                 ▼               ▼                 ▼
           Judge/Arbiter    Evaluation       Report Composer
                 │               │                 │
                 └───────────────┼─────────────────┘
                                 ▼
                           Analysis Result
```

### 2.1 Guarded layers

Dependency direction is `domain <- application <- infrastructure`, and four packages are
**asserted vendor-free by `tests/contract/test_layering.py`** rather than by convention:

| package | may import | why it is guarded |
|---|---|---|
| `app/domain` | `app.domain` | the innermost layer depends on nothing of ours but itself |
| `app/ingestion` | `app.domain`, `app.ingestion` | parsers, chunker, identity recipes, redaction |
| `app/assumptions` | `app.domain`, `app.assumptions`, `app.ingestion`, `app.agents` | the closed registry and the Admission Gate |
| `app/application` | the above plus `app.application` | use-case orchestration |

`app/ingestion` and `app/assumptions` are guarded for the same reason the domain is: the
admission gate and the closed registry are the trust boundary, and a vendor SDK or HTTP
client reaching into them would mean the trusted layer could make a network call.

`app/assumptions` may import `app.agents` because the extractor consumes a prompt definition
from the agent runtime — a pure loader over `.claude/agents/*.md` with no vendor or transport
dependency of its own. That is how the extractor obtains the prompt hash it stamps onto the
provenance of every proposal, gap and quarantine record.

### 2.2 Ingestion module map

```text
app/ingestion/
  identity.py      canonical text, content hashing, derived ids (deterministic)
  parsers/         markdown | json | plaintext -> ParsedDocument; extensible
  chunking.py      section-aware, non-overlapping, document-absolute offsets
  redaction.py     credential masking at the provider/log boundary only

app/domain/services/
  span_verification.py   exact span location and verification (no fuzzy matching)
  typed_parse.py         verbatim value -> TypedValue; unit mismatch blocks

app/assumptions/
  registry.py      closed AssumptionKeyRegistry, resolution, approved-alias gate
  proposals.py     AssumptionProposal / ExtractionBatch (untrusted)
  prompting.py     untrusted-content envelope, per-chunk nonce delimiter
  extraction.py    the one LLM call, behind AgentCompletionPort, fast tier
  admission.py     the Admission Gate; deterministic, no model call

app/application/use_cases/
  ingest_source.py  parse -> version -> chunk -> extract -> admit, with the cache decision

app/domain/interfaces/repositories.py   store ports (tenant is a parameter, not a filter)
app/infrastructure/database/repositories/memory.py   the Phase 2 implementations
```

### 2.3 Coupling module map

The Class A path. Every module below is free of any provider, gateway or agent-runtime
import, and a contract test asserts it per file rather than per layer — the claim that
these consequences are found without an LLM is enforced by the absence of a seam.

```text
app/domain/services/
  scope_lattice.py        equality, containment, intersection, distance (pure)
  value_comparison.py     unit-aware three-way compare; a mismatch BLOCKS coupling
  temporal_intervals.py   half-open [from, to) arithmetic; adjacency is not overlap
  coupling_strength.py    versioned score over evidence properties; cannot see an agent

app/assumptions/
  temporal.py       resolve_authority(key, scope, as_of) -> Resolved | Ambiguous | NoDeclarer
  collisions.py     the role matrix (3.3) and the three-way divergence rule (5.4)
  coupling.py       CouplingSet on (key, scope, as_of); the index and its outcome summary
  invalidation.py   AssumptionDelta -> invalidated | scope_inheritance_affected
  class_a.py        Class A generation by producer; Class B merge; rediscovery rate
  decay.py          the standing knowledge-decay report

app/application/use_cases/
  resolve_coupling.py   the only store-touching step: join bindings to holder authority

app/evaluation/corpus.py   one corpus loader, shared by the scripts and the tests
scripts/coupling_report.py the end-to-end demonstration; --delta propagates a change
```

The dependency direction is `coupling -> collisions -> temporal -> domain services`, and
everything below `resolve_coupling.py` is a pure function over values. That is deliberate:
the collision matrix, the authority resolver and the decay report are all testable by
constructing bindings, so none of them can be made nondeterministic by store iteration
order.

## 3. Runtime pipeline

Two pipelines exist. The deterministic one carries the core mechanism; the agent one
explains and challenges its output and covers Class B consequences.

### 3.1 Ingestion-time (once per source version, cached)

```text
SourceVersion
   |
   v
[Chunker] --> Chunks --> [Embedder] --> vectors
   |
   v
[Assumption Extractor]                     (LLM, fast tier, amortized)
   |   proposes bindings; admits nothing
   v
[Admission Gate]                           (DETERMINISTIC - all three must pass)
   |   1. span verification   cited span exists and contains the value
   |   2. typed parse         value parses under value_type + unit
   |   3. registry resolution key resolves in the CLOSED AssumptionKeyRegistry
   |
   +--(fail 3)--> [KeyRegistryGap]  quarantined, NEVER indexed
   |                    |
   |                    v
   |              human review -> versioned registry update -> re-run resolver
   |
   +--(fail 1 or 2)--> quarantined, recorded as an extraction failure
   |
   v (all pass)
AssumptionRecords ---> Assumption Store
                            |
                            v
                 [Coupling Index]          (DETERMINISTIC)
                            |
                            +--> coupling sets, keyed (key, scope, as_of)
                            +--> outcome per set, by the three-way rule:
                            |      same scope           -> conflicts_with
                            |      narrower, authz >=   -> legitimate override
                            |      narrower, authz <    -> unauthorized_override
                            +--> approved KeyAlias resolution only
```

An unregistered key has **no fallback path** into the index. This is the closed-registry
decision (`DOMAIN_CHECKPOINT.md` 4): precision and trust of Class A take priority over
coverage for the MVP.

### 3.1a Standing resolution (no change request required)

Once the index exists, the resolver produces Class A findings and the knowledge-decay
report from the store alone. **No model call occurs anywhere in this path.**

```text
Assumption Store (admitted bindings only)
   |
   v
[Join to holder authority]                  resolve_coupling.py, the only store access
   |   authority_class from SourceDocument REGISTRATION, never from content
   v
ResolvedBinding[]                            pure values from here down
   |
   +--> [resolve_authority]  (key, scope, as_of) -> Resolved | Ambiguous | NoDeclarer
   |         walks the lattice narrowest-outward; per-candidate authority test;
   |         NO recency comparison, so newest-wins cannot happen
   |
   +--> [Collision matrix]   role- and scope-aware, per checkpoint 3.3 / 5.4
   |         declares/declares divergent -> conflicts_with | override | unauthorized
   |         declares/consumes divergent -> stale, AGAINST RESOLVED AUTHORITY
   |         consumes/consumes divergent -> anomaly, never a contradiction
   |         anything/describes          -> nothing, ever
   |
   v
DerivedRelation[]  (type C: rule_id + input_record_ids, NO evidence of its own)
   |
   +--> [Class A generation]  one ClassAFinding per affected artifact
   |         evidence = computed union of inputs, flagged evidence_is_derived_union
   |         consequence_class and origin are Literals: no agent can self-declare
   |
   +--> [Knowledge-decay report]  known inconsistency | ambiguity | missing knowledge
             absence is recorded, never derived
```

Staleness is evaluated against `resolve_authority`, **not pairwise against every
declarer**. The pairwise reading reports a consumer stale whenever any lower-authority
document disagrees with it — on the sample corpus that flags a training deck that correctly
follows the governing policy. Where authority is `Ambiguous`, no staleness finding is
emitted at all: if the corpus cannot say what the value is, it cannot say a consumer has
the wrong one.

### 3.2 Analysis-time

```text
ChangeRequest
   |
   v
[Change Decomposer]                        (LLM)
   |
   v
ChangeSpec  +  AssumptionDelta[]
   |
   v
[Invalidation Resolver]                    (DETERMINISTIC - no model call)
   |   resolves delta against Assumption Store, temporal authority,
   |   scope lattice, and the authority-gated override rule
   |
   |   Alias gate: only KeyAlias.status == approved participates here.
   |   A proposed alias may inform Class B only, marked as proposed.
   v
InvalidatedKeys --> [Coupling Propagation] (DETERMINISTIC)
   |
   v
DerivedRelation[]  --> CandidateAffectedArtifacts
   |                   (Class A, evidence-backed by construction)
   |
   |   consequence_class is assigned HERE, by the pipeline.
   |   No agent can declare a consequence Class A.
   |
   +-----------------------------+
   |                             |
   v                             v
[Retrieval Planner]        [Historical Search]
   |  per-specialist QueryPlans (NOT one shared bundle)
   v
EvidenceBundle(policy) ... EvidenceBundle(data)   -- disjoint-biased slices
   |                |                |                |
   v                v                v                v
Policy           Process         Technology         Data
Agent            Agent           Agent              Agent
   |                |                |                |
   +----------------+----------------+----------------+
                         |
                         v
                 SpecialistFindings
                         |
                         v
               [Consequence Hunter]        (Class B hypotheses only;
                         |                   Class A already derived)
                         v
                ConsequenceHypotheses
                         |
              +----------+----------+
              |                     |
              v                     v
     [Contradiction Hunter] [Adversarial Challenger]
      (materiality of              (citation-obligated:
       deterministic                evidence_id or
       collisions)                  no_refutation_found)
              |                     |
              +----------+----------+
                         |
                         v
                 [Blind-Spot Hunter]
                  over a DISJOINT evidence partition, not a different persona
                         |
                         v
                  [Evidence Arbiter]
                         |
                         v
                 [Consequence Judge]
                         |
                         v
      [Confidence Scorer]  (DETERMINISTIC, versioned function)
                         |
                         v
                 ConsequenceGraph
                         |
                         v
                 [Report Composer]
```

### 3.3 Why the deterministic stages come first

Class A consequences arrive at the agent layer already discovered and already
evidence-backed. The agents explain, rate materiality, and try to falsify them. They are
not asked to find them. This is what keeps recall independent of context length and keeps
cost sub-linear in corpus size.

### 3a. The analysis orchestrator, as implemented

`app/application/orchestration/analysis.py`. One use case, `AnalyzeChangeUseCase.execute`,
that composes the twelve stages and contains **no business logic at all** - it decides order,
dependency and what to report, and nothing else.

```text
 1 coupling             determ.  ResolveCouplingUseCase.execute
 2 class_a              determ.  apply_delta + derive_class_a, selected to the change's keys
 3 retrieval            determ.  RetrievalService.retrieve     <- the permission scope enters
 4 specialists          agentic  ConsultSpecialistsUseCase     <- one slice per domain analyst
 5 class_b              agentic  HuntConsequencesUseCase       <- reads the analysts' output
 6 evidence_gaps        determ.  gaps_from_hunt                <- the one hard dependency
 7 contradictions       agentic  HuntContradictionsUseCase     <- scoped to the changed keys
 8 blind_spots          agentic  HuntBlindSpotsUseCase         <- retrieves its own partition
 9 challenge            agentic  ChallengeFindingsUseCase      <- reviews both Class B hunts
10 evidence_arbitration agentic  ArbitrateEvidenceUseCase      <- positions, never scores
11 judgment             agentic  JudgeConsequencesUseCase      <- confidence measured here
12 graph                determ.  app.graph.projection.project
13 report               determ.  ComposeReportUseCase          <- reviewer verdicts enter here
```

Stages 10 and 11 are handed **one** candidate set (`_Candidates.of`), computed once and passed
to both the arbiter and the judge. Two stages that disagreed about the candidate set would
produce rulings addressed to consequences nobody arbitrated.

**Discovery finishes before review begins.** Both Class B producers - the consequence-hunter
and the blind-spot hunter - run before the challenger, so every hypothesis faces the same
falsification obligation. An earlier ordering ran the challenger first, which left blind-spot
findings reaching the judge unchallenged: the newest and least corroborated findings were the
only ones nobody had tried to refute.

Two consequences of that order, both wiring rather than logic:

- the challenger is handed the **union** of the two hunts' retained hypotheses;
- the evidence map it sees includes the blind-spot partition's spans. Without them a
  blind-spot candidate would be offered nothing citable and could only come back
  `no_refutation_found` for want of anything to cite - a refusal that reads like a survived
  challenge.

**Class A runs before retrieval.** That ordering is the product's central claim made
operational: Class A recall does not depend on retrieval, so no retrieval failure can damage
it. Retrieval feeds the *agentic* stages only, and a deterministic-only pipeline (no agent
runtimes supplied) still produces the Class A set and the graph - which a test asserts.

#### Required and optional

The five deterministic stages are **required**: they have no provider, so a failure is a
defect and `AnalysisStageError` aborts the run naming the stage. The six agentic stages are
optional and report their own failures. Each stage records a `StageOutcome`, and the statuses
are the whole of requirement 5:

| status | meaning |
|---|---|
| `completed` | ran, produced something |
| `empty` | ran, produced nothing. A legitimate result |
| `failed` | ran, reported a failure. **Not** the same as empty |
| `skipped` | did not run, because a stage it depends on failed |
| `not_configured` | did not run, because no runtime for it was supplied |

**One hard dependency**: evidence gaps require the hunt. A failed hunt asked nothing, and a
gap needs an executed search *and* an unanswered question - so the gap stage is `skipped`,
never reported as "no gaps", which would claim the corpus was interrogated and found complete.
Every other downstream stage degrades instead of skipping: the challenger reviews Class A
alone if the hunt failed, the blind-spot hunter works a *larger* partition if the hunt failed,
the arbiter positions what it was given, the judge arbitrates from the raw findings if the
challenger failed, and the graph still projects the bindings if the judge failed. A review
stage must not be skipped because a discovery stage failed.

`report` is required because composition is deterministic - every section is a projection of a
finalized record. The `report-composer` agent is optional and its failure is recorded on the
document rather than the stage: the report still exists, with the deterministic surface and the
deterministic executive summary. A test asserts exactly that.

#### Change-scoped, not corpus-scoped

The coupling index describes the **corpus**: standing divergences exist whether or not anybody
proposed a change. Reporting them as consequences *of* a change would make every analysis look
productive, so two stages are scoped to the keys the change touches:

- **Class A** is *selected*, not re-derived: the findings the change's deltas propagate to
  (via the existing `apply_delta`) plus the standing findings on a key the change touches.
- **contradictions** receives only those keys' collisions - which also means a change with no
  deltas costs no provider call for that stage.

A change with no assumption deltas therefore selects nothing, and the corpus's standing state
stays where it belongs: the knowledge-decay report.

#### Authorization enters once

The caller supplies a `RetrievalQuery` carrying the permission scope. Every agentic stage
downstream is handed the spans that query returned and nothing else - `_stored_spans` is built
from the retrieval results, and no stage below retrieval touches a store. A test asserts every
consequence's and every hypothesis's evidence ids are a subset of what the requester received.

#### Determinism

`analysis_id`, `run_id` and `now` are parameters. Given the same store, the same configuration
and the same provider responses, two runs agree on every status, count, score and graph shape;
`AnalysisResult.fingerprint()` is that claim in comparable form. Minted record ids differ by
design - they are fresh per run so a re-run cannot collide with a stored analysis - so the
fingerprint excludes them rather than pretending they are stable.

### 3a-bis. Specialist fan-out: seven domains, one path to a consequence

`app/specialists/` (roster, proposals, prompting, runtime) with the application path in
`app/application/use_cases/consult_specialists.py`.

Seven analysts - policy, process, technology, people, data, risk, historical - each read their
own evidence slice and return evidence-backed **observations**. Not consequences: a
`SpecialistFinding` has no field for a class, a status, a severity or a number, which is why no
new persistence model was needed and why nothing a specialist says can bypass arbitration.

**The slice is the only difference between them** (`CLAUDE.md` 17, and every specialist
definition says so in its own words). One query per specialist, each a `model_copy` of the
*caller's* `RetrievalQuery` changing only `source_types` and `top_k`. Tenant and
`PermissionScope` are untouched, so a specialist sees a subset of what the requester was
authorized to see and never more. Reading the corpus any other way - the coupling index, the
source store - would hand an agent spans the requester is not cleared for.

**Selection is deterministic and evidence-based.** A specialist is invoked iff its slice is
non-empty. A specialist with nothing to cite can only speculate, which its own definition
forbids, and paying for a call to be told "my slice is empty" is what the cheap gate exists to
avoid. `affected_domains` is deliberately *not* a gate: it is analyst-supplied prose, so
gating on it would let a mislabelled change silence a specialist that had evidence.

**The position is a ceiling, not a label.** `EvidencePosition` is computed from two
deterministic facts - whether any citation survived, and whether the specialist called the
observation an inference:

| admitted citations | `basis` | position |
|---|---|---|
| none | either | `unresolved` |
| at least one | `inference` | `weakly_supported` |
| at least one | `direct_evidence` | `supported` |

`unresolved` rather than `unsupported` for an uncited observation: "nobody produced evidence
for this" is not "the evidence is against it", and the second is a finding the specialist did
not make.

**Observed bindings are recorded and never admitted.** The definitions ask specialists to
report propositions their slice commits to, and they do - as proposals, carried on the outcome
for registry review. There is no code path from one to the coupling index, and that is
deliberate: the registry is closed and admission is an ingestion-time deterministic gate
(`CLAUDE.md` 20, 21). The value stays rendered text, because typed-parsing it here would be
the first half of a promotion.

**The historical analogue is derived, never proposed.** One specialist output can *raise* a
status rather than only inform one: a `HistoricalEvent` with a recorded outcome is what lets a
Class B consequence exceed `speculative` (checkpoint 9.2 rule 2). `app/specialists/analogues.py`
admits them under four rules, and the first is the one that makes the rest safe:

| rule | why |
|---|---|
| the event id is `event_id_for(tenant, spans)` | there is **no id field** on the proposal schema, so a model cannot mint an `EventId`, and two analyses citing the same spans agree on the id without a store |
| citations ∩ the analyst's own slice | the slice came from the same retriever under the same `PermissionScope`, so an event rests only on spans the requester could read |
| no admitted citation → refused outright | an untraceable event record is worse than none |
| no `observed_outcome` → kept, `qualifies_as_anchor` false | a precedent worth reading that is structurally unable to raise anything |

A future `occurred_at` is refused, and only the `historical-analyst` may propose one at all -
one agent's definition carries the anchor obligation, so one agent may discharge it. The title
is deliberately outside the identity, so renaming an incident cannot turn one precedent into
two.

The consequence-hunter is then offered the admitted **anchors only** and any
`historical_analogue_id` outside that closed set is dropped and reported. Before this the field
was passed through unvalidated, which made rule 2 satisfiable by an invented id. The blind-spot
hunter is offered no anchors, so its claims are dropped unconditionally.

**Two deterministic suppression rules**, both after the calls:

- *cross-specialist* - the same claim on the same citations from two analysts is one
  observation, kept in roster order so the survivor is not a race;
- *Class A restatement* - an observation on a decided key resting **only** on spans that
  finding already materialized. An observation on a decided key citing *other* evidence
  survives, which is the half of the rule that makes it safe to have.

#### Where the findings go

Into the existing Class B stage, not a second reasoning path. The consequence-hunter already
exists to combine "independent evidence from multiple domains"; the fan-out makes that literal
by producing the domains. It receives the observations as attributed context outside the
untrusted envelope - they are this system's gated output, not corpus text - alongside the union
of the slices' spans. The gate over what the hunter may *return* is byte-identical with and
without them, so a specialist cannot widen what a hypothesis is allowed to claim.

From there a hypothesis is challenged, arbitrated and judged like any other. There is exactly
one path from evidence to a consequence, and the specialists feed its front.

### 3b. Blind-spot hunter: a partition, not a personality

`app/blindspots/`, with the application path in
`app/application/use_cases/hunt_blind_spots.py`.

A second, more creative pass over the same evidence is the same analysis at a different
temperature. So the hunter's distinctness is a property of its **input**: `EvidencePartition`
excludes every span the main consequence-hunter was offered, and its `__post_init__` refuses to
hold one that does not. Four slices, in the frozen attribution order:

| slice | what it reaches |
|---|---|
| `single_holder` | keys exactly one artifact binds - nothing corroborates, nothing depends in writing |
| `unwritten_rule` | keys declared only by code, configuration or informal notes |
| `unqueried_source_type` | source types the main retrieval returned nothing from |
| `low_ranked` | the region the main query reached but ranked below what it handed over |

The first two are read off the **existing decay report** (`DecayCategory.SINGLE_HOLDER`,
`UNWRITTEN_RULE`) rather than recomputed: a second implementation would be a second definition
of "undocumented".

**It retrieves its own evidence, through the same retriever.** Three queries, each a
`model_copy` of the main `RetrievalQuery` changing only `text`, `top_k` and `source_types` -
so tenant and `PermissionScope` cannot be re-decided here. Reading the coupling index directly
would have been simpler and wrong: the index is tenant-scoped but not permission-filtered, so
assembling a partition from bindings would hand this agent spans the requester is not cleared
to read.

The third query is the one that earns the stage: **the undocumented keys become the query
text**. Slices 3 and 4 are reachable only through the change's own words, which is the hook the
main analysis already pulled; measured over the sample corpus they return nothing the main
hunter had not already seen. Querying the *subject* of the undocumented coupling reaches spans
the change's wording never would.

Three things the gate does, all one-directional:

- **citations are checked against the partition, not the analysis.** An id the main hunter was
  offered is dropped exactly like a fabricated one, and counted separately as
  `reused_main_evidence_ids`;
- **attribution is derived, never accepted.** The slice credited to a finding is the
  highest-priority slice it actually cited. A claimed slice that disagrees is recorded in
  `misattributed_ids` and discarded;
- **a self-reported restatement is dropped.** The agent is asked to flag when its partition led
  it back into the main analysis; when it does, the finding does not survive.

What it retains then goes to the **adversarial challenger** like any other Class B hypothesis,
so a blind-spot finding can be refuted on partition evidence and arrive at the judge
`rejected`. Its `agent_sources` names the hunter *and* the challenger, which is the observable
difference between a reviewed finding and one nobody looked at.

**Corpus silence is a separate kind.** `SilenceRecord` has no claim, no mechanism and no
citations, so there is no path by which one becomes a consequence. It is also not an
`EvidenceGap`: a gap carries negative provenance from an executed query and this agent holds no
trace. It leaves as an unresolved question plus the source that would answer it.

Class A duplicates are suppressed by the existing `merge_class_b`, the same rule and the same
`(assumption_key, affected_source_version_id)` comparison the main hunt uses.

### 3c. Evidence arbiter: a position, with a ceiling on it

`app/judgment/arbitration.py`, application path
`app/application/use_cases/arbitrate_evidence.py`.

Answers one question - *does the evidence carry this claim?* - on the **evidence axis**
(`EvidencePosition`), never on the status axis. Status is the judge's, severity is the judge's,
and the number is `app/domain/services/confidence.py`'s. There is no field here for any of the
three.

Independence is the factor that gets faked, so it is the one the model is not trusted with. The
arbiter may *report* a corroboration count; what survives is the intersection of its claim with
the set `app/judgment/confidence_inputs.py` already measured from the executed retrieval
outcomes. Overclaim is recorded in `overclaimed_source_version_ids` and discarded, and the
**ceiling** is computed from the measured count only:

| condition (deterministic) | ceiling | reason |
|---|---|---|
| no admitted supporting evidence | `unsupported` | `no_supporting_evidence` |
| admitted contradicting evidence | `disputed` | `contradicting_evidence` |
| the challenger refuted the candidate | `disputed` | `refuted_by_challenge` |
| fewer than 2 distinct measured source versions | `weakly_supported` | `insufficient_independent_corroboration` |

Every applicable reason is reported, not only the binding one: a claim that is both
uncorroborated and contradicted has two problems, and a reader shown one of them is misled
about which evidence would fix it.

Three attacks the shape forecloses, each by absence rather than by a check:

- **several agents citing one document.** The schema has no field for an agent, an agent count
  or an agreement level, and `ceiling_for`'s signature has no parameter one would fit.
- **one document reached twice.** The ceiling reads `len(measured distinct source versions)`,
  a set of `source_version_id`, so a version cannot be double-counted.
- **`no_refutation_found` raising a position.** `ceiling_for` has no parameter for it, exactly
  as `clamp` has none. Nobody finding contrary evidence raises nothing.

Disagreements are carried out, never resolved. Every upstream disagreement the dossier arrived
with plus the arbiter's `residual_disagreement` leaves on the ruling, and the orchestrator feeds
them to the judge through `dossiers_from(extra_disagreements=...)` - which has no trimming
parameter. Nothing in the status rules reads them, so an arbiter cannot raise a status by
disagreeing eloquently.

### 3d. Report composer: shorter and better ordered, never longer

`app/reporting/` (agent gate) plus `app/application/use_cases/compose_report.py`
(deterministic composition).

**Every section is a projection.** `ReportDocument` holds references to objects the pipeline
already produced - a `Consequence`, an `ArbitrationDecision`, a `ConsequenceScore`, a rated
divergence, a `ChallengeResult`, a `SilenceRecord`, a gap id. Nothing is re-derived, re-worded
or re-scored, and there is no code path here that constructs a claim. Sixteen named sections; a
section with nothing in it is listed in `empty_sections` rather than filled.

**The executive summary is deterministic.** `summarize(...)` composes it from counts. A
generated summary is the one part of a report whose truth cannot be checked against evidence,
so the report does not contain one - and that, not a prompt instruction, is why "introduces no
new factual claim" holds. This is a deliberate narrowing of the agent's role and is worth
stating plainly: the `report-composer` agent does not write the report's prose.

**What the agent is admitted to do is structural**, and both effects are checkable:

- **narrowing.** The final foreground is the deterministic eligible set intersected with what
  the composer proposed, so a handle it was not offered cannot be added. Findings the surface
  rule pins (`supported` or `likely` at HIGH or CRITICAL) are restored whatever it proposes -
  the floor that stops "reduce alert load" becoming "hide the alarming ones".
- **ordering.** A preference over the foreground. Anything omitted from the order is appended
  rather than dropped, so an omission cannot remove a finding from the surface.

Its prose is carried **beside** the report as advisory narration, and refused *whole* if any
identifier-shaped token in it is absent from the finalized set. Whole rather than repaired: a
summary with one invented citation removed is a summary whose other sentences came from the
same process.

#### Reviewer feedback

The report stage is the one place reviewer state enters. Standing verdicts are read through the
**analysis snapshot** (`RecordDispositionUseCase.verdicts`), because the artifact version a
suppression expires on is a property of the stored finding - a `Consequence` records a title,
and a title does not change when a document is revised.

- `wrong` / `not_material` take a finding off the default surface, leave it in
  `consequence_ids`, and are counted in `AnalysisReport.suppressed_count` with the verdict that
  did it. Expiry is delegated to the existing `suppression_has_expired`; there is no second
  rule.
- `already_known` suppresses nothing and is reported as a rate over the findings a reviewer has
  already ruled on for the same `(assumption_key, source version)` - which is the population
  the question is about: how often the system re-delivers something a reviewer said they knew.
- `ReviewerSignals.measured` distinguishes "nobody has reviewed" from "no store was wired". A
  rate of zero means something different in each case.

A finished analysis is then persisted (`AnalysisStore.put(snapshot(...))`), which is what gives
the *next* run something to read. That closes the wiring gap recorded as issue 65.

## 4. Agent orchestration model

The runtime, not the agents themselves, decides:
- which agents run
- in what parallel groups
- which findings move forward
- when additional retrieval is required
- when an expensive model is warranted
- when an analysis can terminate

Agent prompts define role behavior but not global workflow state.

### Recommended execution phases

Phase 1:
- change-decomposer

Phase 2, parallel:
- policy-analyst
- process-analyst
- technology-analyst
- people-analyst
- data-analyst
- risk-analyst
- historical-analyst

Phase 3:
- consequence-hunter
- contradiction-hunter

Phase 4, parallel:
- adversarial-challenger for high-impact hypotheses
- blind-spot-hunter

Phase 5:
- evidence-arbiter
- consequence-judge

Phase 6:
- report-composer

The router can skip irrelevant specialist agents based on ChangeSpec.

## 4a. Analysis diversity and correlated error

Persona diversity over identical context is not an ensemble. Several agents reading the
same EvidenceBundle with different system prompts produce heavily correlated output, and
the judge then reads what look like independent votes but are effectively one vote. This
silently corrupts the `independent corroboration` factor in the confidence function.

Three rules follow.

1. **Diversity comes from inputs, not personas.** Each specialist issues its own
   `QueryPlan` biased toward its own source types. Specialist evidence slices should
   overlap as little as the corpus allows.
2. **The blind-spot stage is a partition, not a personality.** It runs over a
   deliberately disjoint evidence basis: source types no specialist queried, corpus
   regions with the lowest retrieval scores, and single-holder assumption keys. Given the
   findings so far it would become a critic; denied them it would be a second consequence
   hunter. Neither is what the stage is for.
3. **Corroboration counts sources, not agents.** Enforced in the scoring function, not
   left to prompt wording.

Optional, and one of the project's open evaluation questions: **provider diversity.**
Because the gateway (section 11) allows per-agent provider binding, specialists and the
challenger can be served by different vendors and model families. Whether cross-provider
diversity reduces correlated reasoning error more than input diversity alone is an
experiment the harness must be able to run, not an assumption the architecture bakes in.

## 5. Agent communication contract

Agents communicate through structured artifacts, not free-form conversation.

Conceptual objects:
- ChangeSpec
- AssumptionRecord
- AssumptionDelta
- CouplingSet
- EvidenceBundle
- SpecialistFinding
- ConsequenceHypothesis
- ChallengeResult
- BlindSpotFinding
- ArbitrationDecision
- ConsequenceGraph
- AnalysisReport
- FindingDisposition

All objects are Pydantic models and versioned.

Schemas are authored **before** the prompts that must satisfy them. A prompt instructing
a model to "match the runtime schema" against a schema that does not yet exist produces
invented fields and fabricated citations.

## 6. Retrieval architecture

Retrieval is a separate service layer.

Strategies:
- lexical retrieval — **built** (`app/retrieval/lexical.py`, Okapi BM25)
- vector similarity — **built** (`app/retrieval/vectors.py`, cosine)
- hybrid fusion — **built** (`app/retrieval/hybrid.py`, reciprocal rank fusion)
- metadata filtering
- graph retrieval
- temporal retrieval — built as a filter on every strategy, not as a separate retriever
- source authority filtering
- historical similarity

`RetrievalStrategy` names only what exists, because the value is persisted and corroboration
counts distinct strategies (checkpoint 9.3). It currently has three members.

Do not rely on one retriever.

### 6.1 Two strategies, and one contract that does not fit both

The two built strategies are deliberately the **same shape**: same `EvidenceCandidate` join,
same filter set in the same fixed order, authorization applied first by narrowing the index,
descending score then ascending `evidence_id` for a total tie break, and provenance carried
by reference so a citation is the one the admission gate verified. Only the scoring differs.

They are also deliberately **independent**. The vector index does not reuse the lexical
tokenizer — it has its own minimal one — because corroboration counts evidence reached by
*distinct* strategies, and two strategies that agree because they share a preprocessing
stage are closer to one strategy than to two.

**`RetrievalPort` does not fit the vector strategy, and the port was not changed.** The port
is synchronous on purpose: "a lexical scan over stored spans is computation, not I/O". But
interpreting a *query* for the vector strategy requires embedding it, which is I/O. So:

- `VectorRetriever.retrieve` takes a `VectorQuery` — the query plus its embedding — and
  stays a pure function of values, testable with no provider;
- the async, `RetrievalPort`-shaped surface is `RetrieveEvidenceUseCase.execute_vector`,
  which owns the embedding call.

This is a real mismatch in the contract as authored, recorded rather than patched. Widening
the port now would mean guessing at the shape of fusion; it is the QueryPlan slice's problem
to resolve, and `docs/STATUS.md` carries it as an open issue.

### 6.3 The retrieval entry point

Three strategies exist and each is reached differently: `execute` is synchronous,
`execute_vector` is async and needs an embedder, `execute_hybrid` needs both and returns a
richer object. That asymmetry is correct one layer down — it reflects what each strategy
genuinely requires — and wrong to push onto every caller.
`app/application/orchestration/retrieval_service.py` absorbs it.

```text
RetrievalRequest(query, strategy?)
        |
        v
RetrievalService.retrieve            selection: request > configuration; no inference
        |
        +--> execute            (lexical)
        +--> execute_vector     (vector,  needs EmbeddingPort)
        +--> execute_hybrid     (hybrid,  needs EmbeddingPort)
        |
        v
FinalizedRetrieval
   .strategy          the strategy that was SELECTED
   .outcome           a plain RetrievalOutcome - same type for all three
   .hybrid            per-strategy rank contributions, hybrid only
```

**It orchestrates and finalizes; it does not retrieve.** Every path delegates to the use
case, which delegates to a retriever. No filter, no scoring, no ranking and no merging lives
here, so every guarantee is inherited rather than restated — and that is observable, not
asserted: a test runs one filtered query under all three strategies and requires their
filter buckets (`considered`, tenant, permission, source type, as-of, scope, unscoped) to be
identical. Only the *scoring* buckets may differ.

**Selection is configuration, overridable per request.** `CCE_RETRIEVAL_STRATEGY` names the
default (`lexical`, because it is the strongest measured strategy and the only one needing no
embedder). A request may name another. Neither is inferred from what happens to be
available.

**Three failures are explicit, and one of them matters most.** An unknown name on a request
is a Pydantic error before the request exists; an unknown `CCE_RETRIEVAL_STRATEGY` raises
`UnknownRetrievalStrategy` naming the alternatives; and selecting `vector` or `hybrid`
without an embedder raises `EmbedderRequired` rather than falling back to `lexical`. The
fallback is the tempting one and it is refused deliberately: a caller who asked for semantic
retrieval and received lexical results has no way to detect it, and a figure produced that
way would be attributed to a strategy that never ran. Dispatch is a table, so a strategy
added to the enum without a handler raises instead of resolving to a default.

**One shape, whichever ran.** `FinalizedRetrieval.outcome` is a `RetrievalOutcome` for all
three — same type, same validators — so a consumer needs no strategy-specific branch. The
envelope adds only what the contract cannot express: the *selected* strategy (a fused run
reports `(lexical, vector)` in `strategies_executed`, because both ran and corroboration
counts distinct strategies), and the hybrid contribution detail, present exactly when the
strategy is `hybrid`.

The service takes an `EmbeddingPort` and cannot construct one: this layer must not import
`app.infrastructure`, which `tests/contract/test_layering.py` enforces for the whole layer
and a focused test pins for this module by parsing its imports.

Deliberately absent: reranking, query expansion, caching, retry, and a QueryPlan that runs
several strategies and merges them. This layer adds dispatch; adding scoring to it would put
a fourth strategy where nobody would look for one.

### 6.2 Hybrid fusion

`app/retrieval/hybrid.py` combines the two rankings. It is a **pure function of two
outcomes** — `fuse(lexical_outcome, vector_outcome)` — and that is the design, not an
implementation detail:

- neither retriever was modified, and neither knows fusion exists;
- fusion touches no store, embeds nothing, and applies no filter of its own;
- every guarantee is therefore *inherited*. Tenant isolation, authorization, source-type,
  scope and temporal filtering and provenance validation all ran twice — once per strategy —
  before `fuse` saw a result. Fusion may merge, reorder and truncate; it **cannot** introduce
  a span neither input returned, and a test asserts the fused set is a subset of the union.

**Reciprocal rank fusion**, `score(d) = Σ 1/(k + rank_s(d))` over the strategies that
returned `d`. The two inputs are scored on incomparable scales — BM25 is an unbounded sum,
cosine is bounded in `[-1, 1]` — so any fusion that combines *scores* must first normalize
them, and every normalization is an unmeasured decision: min-max is unstable when a strategy
returns one result, and z-scores need a distribution a 21-span corpus does not have. RRF
discards the scores and fuses the ranks, which sidesteps the whole problem. `k = 60` is the
value from the original paper and is **untuned against this corpus**.

**Strategy provenance is preserved rather than flattened.** `HybridResult.contributions` maps
each strategy to the rank *it* assigned, so `lexical-only`, `vector-only` and `both` are
readable off the outcome. That is a requirement, not a convenience: checkpoint 9.3 counts
corroboration across distinct strategies, and a fused list that forgot which strategy found
what could not support the count.

Two consequences of that in the contract, both deliberate:

| field | value | why |
|---|---|---|
| `RetrievalResult.strategy` | `hybrid` | the fused *score* is what the result carries |
| `RetrievalOutcome.strategies_executed` | `(lexical, vector)` | both genuinely ran; a single `hybrid` entry would collapse two strategies into one and understate corroboration |

`fuse` refuses inputs that are not two rankings of the same question: different queries,
different permission scopes (fusing across scopes could surface a span the requester may not
read), different filter configurations, an already-fused input (double-counted ranks), or two
rankings from the *same* strategy (corroboration that never happened). The shared trace
buckets are **asserted equal** rather than summed — both retrievers filter the same candidate
set in the same fixed order, so a disagreement means the fusion spans two corpora.

**Still absent: reranking, query expansion, weighted or learned fusion, a third strategy.**
Each is separately measurable and none is justified by a number — see `docs/EVALUATION.md`,
"Hybrid retrieval, as measured", where fusion over the current two inputs measures *worse*
than the stronger input alone at every cut-off but the deepest.

A QueryPlan specifies:
- query strings
- filters
- source types
- time bounds
- top_k
- retrieval strategies

Results are merged, deduplicated, and reranked.

### 6.4 Class B: the consequence-hunter

The first agentic stage, and the only one built. Class A arrives already derived and
evidence-backed; Class B is what no assumption binding can express - load and capacity
shifts, timing and queue effects, escalation volume, reconciliation drift, second- and
third-order chain reactions.

```text
ChangeSpec + FinalizedRetrieval evidence + Class A findings
        |
        v
app/consequences/prompting.py     assemble; evidence inside the untrusted envelope
        |
        v
consequence-hunter                ONE call, AgentCompletionPort, reasoning tier
        |
        v
app/consequences/hunter.py        the gate: drop fabricated ids, refuse forbidden fields
        |
        v
app/application/use_cases/hunt_consequences.py
        |                         merge_class_b - deterministic Class A suppression
        v
ClassBOutcome    retained + suppressed + merge + rediscovery rate
```

Same split as ingestion: **the model proposes, deterministic code admits.** Nothing the
model returns reaches a store in the shape it was returned, and `app/consequences` is a
guarded layer for the same reason `app/assumptions` is - the code that decides what a model
was allowed to have said must not itself be able to make a network call.

#### The four things an agent cannot do

| | how it is prevented |
|---|---|
| supply `confidence` | no field on the proposal schema, and `extra="forbid"` makes one a rejection rather than a dropped key. Computed by versioned code over evidence features |
| supply `severity` | same, and it is the judge's to assign: a judgment about business impact the corpus does not contain |
| supply `status` | same. Absent an analogue with a recorded outcome, Class B cannot exceed `speculative`, and the runtime records the absence rather than the model asserting the status |
| promote to Class A | there is no `consequence_class` field to declare, and `ConsequenceHypothesis` hard-codes `derivation=INFERRED` / `support=MECHANISM_ONLY` as `Literal`s. Classification is by **producer** |

The last row is the important one: the guarantee is a property of the *type*, not of the
gate's care. Even a caller constructing a hypothesis by hand cannot raise its support.

#### Evidence discipline

The prompt lists the evidence ids the model may cite and says they are the only ones. The
gate then keeps **only ids that were actually offered** and reports the rest. Matching is
exact with no normalization: an id differing by one character is not a near miss to be
repaired, it is an id the model did not receive, and repairing one would be the system
inventing a citation on the model's behalf.

A hypothesis whose every citation was fabricated is **weakened, not discarded** - it becomes
a hypothesis with no evidence, which `ConsequenceHypothesis` already models correctly. Every
check in the gate can only weaken a claim; none can strengthen one.

Retrieved corpus text goes inside the same per-call nonce envelope the extractor uses
(`app/assumptions/prompting.wrap_untrusted`), reused rather than reimplemented - the corpus
contains a document that tries to issue instructions, and a second envelope would drift from
the first.

#### Class A suppression is deterministic and runs afterwards

The prompt shows the hunter the Class A set and asks it not to re-derive them. The pipeline
does not trust that: `merge_class_b` (checkpoint 7.5.2, already mutation-tested) suppresses
a hypothesis whose `(assumption_key, affected_source_version_id)` pair is already in the
Class A set. Suppression **merges rather than drops** - the absorbing finding records the
hypothesis id, so the model's articulated mechanism stays available while a second finding
claiming the same thing at an unearned trust level does not.

A hypothesis naming no artifact is **retained**: without knowing which artifact it concerns
we cannot know it is a duplicate, and ambiguity demotes to Class B (checkpoint 7.4).

Deliberately absent from this slice: contradiction-hunter, adversarial-challenger,
blind-spot-hunter, the judge, graph projection and any UI. The hunter also does not construct
an `EvidenceGap` - a gap is a claim about the *corpus* and needs the retrieval trace this
agent does not hold.

### 6.5 The EvidenceGap producer: negative provenance from the real trace

Retrieval's last consumer, and the only one interested in what retrieval *failed* to return.
An `EvidenceGap` is checkpoint 2's type E: **a claim about the corpus, not about the world**,
which is why it can never become a graph edge and why it needs the trace rather than a model's
account of one.

```text
HuntOutcome.unanswered_questions        FinalizedRetrieval
   (the only model-authored input)         (query + RetrievalTrace)
              |                                    |
              +----------------+-------------------+
                               v
        app/application/orchestration/evidence_gaps.py
                               |
                   two conditions, both required
                               |
                               v
              GapRecord = EvidenceGap + SearchProvenance
```

#### The rule

Checkpoint 2.2 defines `support: negative_provenance` as *the executed QueryPlan that
returned nothing*. Both halves are load bearing, so both are checked:

| condition | read from | why |
|---|---|---|
| a strategy ran | `strategies_executed` non-empty | a run that never reached a retriever searched nothing |
| over a readable corpus | `considered - excluded_by_tenant - excluded_by_permission > 0` | a requester whose whole slice was withheld looked in nothing |
| and returned nothing | `trace.returned == 0` | a plan that returned something cannot be cited as one that did not |

**"The hunter said it did not know" is never sufficient.** That is a model's uncertainty, and
recording it as a statement about the corpus would launder a model's silence into evidence -
the exact inversion type E exists to prevent. A search that *did* return evidence produces no
gap for that search either, even where the hunter still had questions.

#### The model cannot author the provenance

`produce_gaps` takes model-authored text in one parameter, `questions`, and it reaches one
field, `EvidenceGap.question`. Every other field is read off the `FinalizedRetrieval` by
`SearchProvenance.from_retrieval`. There is no parameter through which a model-authored value
could reach a provenance field, which is stronger than validating that none did — and a test
asserts the signature rather than the intent.

`SearchProvenance` additionally requires its trace to be a **coherent account of a search**:
the exclusion buckets must sum to `considered`, the invariant all three retrievers already
guarantee. A hand-assembled provenance therefore has to add up the way a real search adds up
rather than merely being well-typed.

#### What a gap carries, and where it lives

The domain entity is frozen by the checkpoint and holds `question`, `strategies_executed`,
`source_types_searched`, `as_of`, `would_be_resolved_by`, the `analysis_id` and the optional
`partition`. The rest of the required provenance — query text, tenant, filters, permission
scope hash, retrieval configuration and scorer versions, the trace itself, the run identifier
— lives in the `SearchProvenance` envelope beside it, the same pattern `FinalizedRetrieval`
and `HybridOutcome` use. `GapRecord` pairs the two rather than merging them, so the claim and
the audit trail cannot drift.

Two details are decisions rather than plumbing:

- **`source_types_searched` records `*` for an unfiltered search.** The field requires at
  least one entry, and an empty tuple would report a search of everything as a search of
  nothing.
- **A search with no `as_of` filter records when it ran.** A gap has to state the moment its
  claim about the corpus was true; "no filter" is not a moment, and a gap read a year later
  must not look like a claim about the corpus as it is then.

#### Authorization is represented, not leaked

| the requester's slice | outcome |
|---|---|
| empty — everything withheld or another tenant's | **no gap**: they searched nothing they could read, so nothing was established about the corpus |
| narrowed — some spans withheld | gap, flagged `authorization_limited`: it describes one requester's view, not the corpus |
| complete | gap, unflagged |

The first row matters more than it looks. The alternative — a gap per restricted requester —
would make a report's gap list vary with who ran it and read as though the corpus had changed.
What escapes about a withheld document is a **count and a scope hash**, exactly what the trace
already carries: `excluded_by_permission` is query-independent by construction, so surfacing
it adds no oracle. A test serializes a gap produced under a restricted scope and asserts that
no evidence id, uri, title or span text visible only to the privileged scope appears anywhere
in it.

#### A gap is never an edge

`EvidenceGap` carries no evidence at all (its own validator refuses any), attaches to the
analysis, and may *annotate* one node through `partition`. There is exactly one node field and
no second one, so a gap cannot name two ends — the structural form of checkpoint 2.4's rule
that a gap annotates a node and is never a relationship between nodes. Inventing a
relationship out of silence is treating absence of evidence as evidence of absence.

Deliberately absent: the contradiction-hunter, any change to a retrieval algorithm, and any
change to the domain checkpoint. The producer is a separate seam the caller composes rather
than a step inside `hunt_consequences.py`, which is what keeps the hunter without access to
the trace it must not author.

### 6.6 The contradiction-hunter: rating a set it did not find

The second agentic stage, and the first whose *input* is a trust-bearing Class A conclusion.
That inverts its risk profile relative to the consequence-hunter. There the danger was a model
claiming more than it earned. Here it is a model quietly rewriting something the resolver
already got right - a stale consumer presented as a peer conflict, a legitimate override
presented as a governance defect, or an empty corpus presented as a contradiction.

```text
KeyCollisions (relations + overrides + anomalies)   from app/assumptions/collisions.py
        |
        v
app/contradictions/divergence.py     project + classify; five classes, one per matrix row
        |
        v
app/contradictions/prompting.py      closed set, per-item evidence, untrusted envelope
        |
        v
contradiction-hunter                 ONE call, AgentCompletionPort, standard tier
        |
        v
app/contradictions/hunter.py         the gate: handles, per-item citations, forbidden fields
        |
        v
app/application/use_cases/hunt_contradictions.py
        |                            emission decided by the deterministic class
        v
ContradictionReport    divergences + ratings + SpecialistFinding records
```

#### It does not search, and cannot

`ContradictionHunter` holds a completion port and an agent definition. No retriever, no store,
no corpus access. The layering contract test forbids `app/contradictions` from importing
`app.retrieval` at all, so "the contradiction-hunter does not search for contradictions" is a
property of the layer rather than a rule in a docstring.

Divergence is found by **exact comparison** in the coupling index before the agent runs. The
agent's contribution is the two things the resolver cannot compute: whether the divergence
matters and to whom, and where the artifacts stand in time.

#### The five classes are deterministic buckets, not judgements

| class | produced by | reader's action |
|---|---|---|
| `material_contradiction` | `conflicts_with` relation | two peers disagree; a decision is needed |
| `stale_invalidation` | `invalidates_assumption` relation | the consumer is out of date |
| `unauthorized_override` | `unauthorized_override` relation | a governance defect: who was allowed to decide? |
| `legitimate_scoped_override` | `OverrideRecord` | **none.** Correct behaviour |
| `non_material_divergence` | `CouplingAnomaly` | scope review, not a conflict |

"Which class is this?" is answered by *where the item came from*. The class is set at
construction on a frozen `DivergenceItem`, and `DivergenceItem.__post_init__` refuses an item
whose class disagrees with its relationship - so even the code assembling the input cannot
mislabel a `conflicts_with` as a staleness finding, or give a legitimate override a relation
id it would look Class A through.

#### Why the agent cannot reclassify

The assessment schema has no field for `divergence_class`, `relationship`, `relation_id` or
`rule_id`, and `extra="forbid"` makes an attempt a rejected batch rather than a dropped key.
Two further details close the remaining routes:

- **The handle the model cites is local.** Items are `d1`, `d2`, ... and the prompt never
  renders a `relation_id` or a `rule_id`. So the model's output cannot name a Class A relation
  at all - not to relabel one, not to reattribute one, not to reference one.
- **A rejected batch costs the rating, not the divergence.** Both failure paths - provider
  error and schema violation - return every divergence unassessed. The resolver's conclusions
  do not depend on the call, so losing the call must not lose them.

#### The agent annotates; it cannot suppress

Emission is decided by the deterministic class, so a model rating a real contradiction
`immaterial` gets its rating recorded on the finding and does not remove the finding. That is
why "an unauthorized override remains a governance finding" needs no special case: it falls
out of a rule holding for all three emitted classes.

The two unemitted classes are the rows a reader would notice being wrong. A legitimate
override reported as a defect is visibly incorrect to anyone who reads the two documents, and
two consumers disagreeing reported as a contradiction points the reader at the wrong owner.
Both are already resolved deterministically; the agent tier does not restate them.

#### No numeric confidence, and a justified direction of travel

`Materiality` is three values and `TemporalPosition` five - closed enums an agent can defend,
rather than a number it cannot. `SpecialistFinding.evidence_position` carries it into the
domain, derived from whether the finding has evidence and whether the agent reached a
determination.

One asymmetry is deliberate: `immaterial` is the only judgement here that argues *against* a
divergence found by exact comparison, so the schema requires a reconciling exception or an
explicit rationale for it. Bare dismissal has no representation. `material`, conversely, must
name who is affected - "this matters" with no affected party is a label.

#### Citations are admissible per item

A real evidence id attached to an unrelated divergence supports an unrelated claim, which is a
fabricated citation assembled out of genuine parts. So the offered set is per handle, not one
flat pool, and the gate checks a citation against the item it was cited for.

Contextual evidence beyond the participating bindings has one narrow route, because the agent
definition asks whether an exception clause reconciles two artifacts and such a clause usually
sits in neither of them. The **caller** supplies the span and names the items it bears on; a
span naming an unknown item is refused. The retrieval, if any, is the caller's, and the agent
sees only its result.

#### A boundary recorded rather than silently crossed

`.claude/agents/contradiction-hunter.md` also describes contradictions no key comparison can
express - SOP versus observed behaviour, conflicting definitions, a procedure that cannot be
executed as written. **That half has no runtime path and no schema.** Requirement is the reason
rather than laziness: a corpus with no deterministic divergence must produce no contradiction
findings, and the cheapest guarantee of that is a schema with nowhere to put one. The
definition and the task contract both now name the boundary, so a cooperative model does not
spend a repair round on output that would be rejected.

Deliberately absent from this slice: the adversarial challenger, blind-spot hunter, evidence
arbiter, judge, graph projection, any UI, and any change to retrieval.

### 6.7 The adversarial challenger: an objection is a proof obligation

The third agentic stage, and the one with the most inverted incentive. A model told to
disagree will disagree, fluently, at whatever rate it is asked - so *producing objections* is
not evidence that this stage works. What makes it work is that the objections it cannot prove
do not survive, and that the deterministic layer it reviews cannot be talked down.

```text
ClassAFinding[] + ConsequenceHypothesis[]     finalized candidates
        |
        v
app/challenge/candidates.py      project; one rulebook per candidate, by origin
        |
        v
app/challenge/prompting.py       per-candidate evidence, per-candidate rulebook, envelope
        |
        v
adversarial-challenger           ONE call, AgentCompletionPort, standard tier
        |
        v
app/challenge/challenger.py      the gate: handles, admissibility, citations
        |
        v
app/application/use_cases/challenge_findings.py
        |
        v
ChallengeReport    ChallengeResult[] + FalsificationTest[] + refusals
```

#### Two rulebooks, chosen by producer

| candidate | what may be challenged |
|---|---|
| Class B hypothesis (`origin=agent`) | everything - mechanism, analogue, magnitude, whether the cited span supports the claim made or only a weaker nearby one |
| Class A finding (`origin=resolver`) | **only the four `ClassAGround` members.** A derivation is rule-wrong or input-wrong; the conclusion is not challengeable |

Which rulebook applies is read off `ChallengeTarget.origin`, never off anything the model
said, and the prompt states the applicable rulebook **against each candidate** rather than
once at the top - because the looser rulebook is the one that gets applied when the two sit
adjacent in one paragraph.

A Class A candidate's mechanism is deliberately **not rendered into the prompt**. It is
precisely what may not be challenged, so showing it would invite the single objection the
runtime must reject. What is shown instead is the inputs: assumption key, scope, effective
date, evidence - the things a challenge may actually dispute.

#### The four admissible grounds, and why there is no fifth

`disputed_span`, `disputed_key_or_scope`, `disputed_temporal_validity`,
`superseding_version` (checkpoint 8.1).

"Incorrect deterministic rule application" is **not** a fifth ground, and its absence is
load bearing. Every version of it that corpus evidence can support already reduces to
disputing an *input*: a precondition that did not hold is a temporal-validity dispute, two
keys that are not one variable is a key-or-scope dispute, a value the span does not carry is
a span dispute. What remains - "the rule ran correctly on correct inputs and I disagree with
the output" - is a claim about the *code*, not about the corpus. It cannot be evidence-backed,
and routing it through the challenge channel would let a model's opinion mark a derivation
wrong without touching anything checkable. A resolver defect is a bug report, not a
`ChallengeResult`.

#### Three outcomes, each with a price

| outcome | costs | enforced by |
|---|---|---|
| `refuted` | ≥1 admitted evidence id + what it contradicts | proposal validator, then the gate |
| `weakened` | ≥1 admitted evidence id + how the scope narrows | same |
| `no_refutation_found` | a falsification observation, and **no** evidence ids | same |

There is no fourth outcome, so "I have doubts" has no representation. The per-outcome rules
are duplicated from `ChallengeResult` onto the proposal schema on purpose: failing at the
boundary lets the runtime record *that the model produced a malformed objection*, whereas
failing at domain construction would surface from the wrong layer after the gate had already
decided to admit it.

The model supplies the falsification *observation*; the runtime mints the `TestId` and builds
the `FalsificationTest`. A model that could mint identifiers could point a challenge at a test
that does not exist.

#### An inadmissible objection is refused, not downgraded

This is the subtlest decision in the slice. A Class A `refuted` with no ground could be
converted to `no_refutation_found` - and must not be. That outcome asserts "I looked and found
nothing", which is not what happened: the model found something it was not permitted to say.
Converting would misreport the run *and* fabricate the falsification test that outcome owes,
since the model supplied no observation. So it is refused, counted by reason, and the candidate
is reported **unreviewed** rather than as having survived.

Four refusal reasons, kept apart because they mean different things and only one is about the
corpus: `invented_target`, `inadmissible_class_a_ground`, `class_a_ground_on_class_b`,
`no_admitted_citation`.

#### Citations are admissible per candidate

A real evidence id attached to a different candidate supports an unrelated claim - a
fabricated citation assembled out of genuine parts - so the offered set is per handle and an
existence check against one flat pool would admit it. When every cited id is dropped, the
`refuted`/`weakened` verdict is **uncited** and is refused rather than handed to the domain as
a record it would reject.

Contextual evidence has a route, and here it is not optional: the challenger's single most
valuable finding is *a newer source version supersedes the one the derivation used*, which is
by definition not in the candidate's own evidence set. The caller retrieves it and names the
candidates it is admissible against; a superseding span is flagged as such in the prompt.

#### What review cannot do

Nothing in `app/challenge` constructs, copies or re-provenances a `ClassAFinding`. A
`ChallengeResult` carries its own `produced_by` from the agent binding, so the challenger's
attribution and the resolver's cannot be confused. A test dumps every candidate before and
after a full run **on the refuted path** and compares.

A `refuted` verdict is therefore a finding *about* a candidate, not a deletion of it.
Suppression is the judge's decision over the arbitrated set; a challenger that could delete a
deterministic derivation would be the agreeable-model problem in its most direct form.

And surviving review is **not corroboration** (checkpoint 9.2): nobody finding contrary
evidence is strictly weaker than evidence of correctness, so there is no path from
`no_refutation_found` to a raised status. `ChallengeReport.survived` is named for exactly what
it is, and is reported apart from `unreviewed` - conflating those two is the one accounting
error that would make a failed stage look like a clean bill of health.

Deliberately absent from this slice: the blind-spot hunter, the evidence arbiter, the judge,
graph projection, any UI, and any change to retrieval.

### 6.8 The consequence judge: status, severity, and a computed confidence

The last agentic stage, the first place all five upstream stages meet, and the only output a
reader acts on directly. Its gate is therefore different in kind from the earlier ones: the
hunters' gate refuses claims the model could not support, the challenger's refuses objections
it could not prove, and this one refuses **statuses the evidence does not reach**.

```text
ClassAFinding[]  ConsequenceHypothesis[]  AssessedDivergence[]  ChallengeResult[]  gap ids
        \               |                        |                     |             /
         +--------------+------------------------+---------------------+------------+
                                        |
                        app/judgment/dossier.py      one dossier per candidate
                                        |             identity INHERITED, not minted
                                        v
                        app/judgment/prompting.py    per-candidate rulebook + evidence
                                        |
                                        v
                        consequence-judge            ONE call, standard tier
                                        |
                                        v
                        app/judgment/judge.py        the gate, then status_rules
                                        |
              +-------------------------+-------------------------+
              v                         v                         v
        Consequence            ArbitrationDecision         ConsequenceScore
        (the pipeline's)          (the judge's)          (the confidence function's)
```

Three records rather than one, because they have three different authors and a reader
auditing any of them needs to know which. All three carry the same ``consequence_id``.

#### The identity is inherited, and an orphan is refused

``app/challenge/candidates.py`` minted a ``consequence_id`` per candidate because
``ChallengeResult`` is keyed on it. The judge **reuses** that id on all three records, and
``dossiers_from`` raises when a ``ChallengeResult`` names a consequence no candidate carries.

That closes a failure that would otherwise be silent: mint a second id here and the upstream
refutation attaches to nothing, so the report renders a refuted finding as unchallenged.
``ArbitratedConsequence.__post_init__`` re-checks the same thing at the point the three
records are paired.

#### Severity is the judge's. Confidence is computed. They are never blended.

| | who decides | why |
|---|---|---|
| `severity` | the judge | a function of business impact the corpus does not contain (checkpoint 9.4) |
| `confidence` | `app/domain/services/confidence.py` | every input is a measurable property of the evidence |

`DispositionProposal` has no `confidence` field, so the judge is not asked and cannot be
believed. **No branch in the runtime moves a severity** - not for a low confidence, not for a
weak status. HIGH severity with LOW confidence is a legitimate and important output, and a
runtime that quietly reconciled them would destroy the one signal this stage adds. Tests pin
both directions.

The confidence function's *input type* is the enforcement of checkpoint 9.3:
`ConfidenceInputs` has no field for an agent, an agent count, a model, a vote or a level of
agreement, and `extra="forbid"` refuses one. It cannot count agents, so it cannot be talked
into treating agreement as corroboration. Corroboration is a saturating function of distinct
source versions **gated by strategy reach** - five versions reached by one strategy is one
strategy's view of the corpus, and scoring it as five-fold corroboration is how a lexical
retriever's blind spot becomes a confident number.

#### Status is proposed, then clamped downward only

`app/judgment/status_rules.py` applies the checkpoint 9.2 gates after the model answers:

| rule | effect |
|---|---|
| no admitted supporting evidence | Class B → `speculative`, Class A → `unresolved` |
| Class B with no historical analogue | → `speculative` (including from `disputed`) |
| a claimed analogue outside the admitted anchors | dropped upstream, so the row above then fires |
| an admitted refutation | → `rejected` |
| an admitted narrowing | at most `disputed` |
| `no_refutation_found` | **nothing** |

The last row is enforced by *absence*: `clamp` has no parameter carrying a clearance, so no
branch can read one, so none can raise anything on the strength of it. A test asserts two
otherwise identical candidates get the same status whether or not a challenger cleared them.

`clamp` never returns something stronger than it was given, asserted over the full
cross-product of statuses and candidate shapes rather than sampled cases. That
one-directional property is what makes proposing-then-clamping safe: the judge's disposition
is the ceiling, and code decides how far below it the evidence reaches. Every clamp is
recorded in the rationale, because a clamp and a cautious judge look identical otherwise and
mean different things - one is the corpus being thin, the other an opinion.

**A clamp can create an obligation the model was never asked about.** Clamping a HIGH-severity
finding to `unresolved` triggers the domain's requirement that it name what would settle it -
but the judge proposed something that owed nothing. Failing the batch there would discard a
whole arbitration because the model could not anticipate a gate, so the runtime derives the
action from the clamp *reason*. That is mechanical rather than invented: the reason is the
thing to fix. The judge's own actions always come first.

#### Disagreements are added to, never trimmed

`CandidateDossier.upstream_disagreements` collects every refutation, every narrowing with its
admissible ground, every rated divergence and every unanswered question - and the runtime
unions it with whatever the judge preserved. `_merge_disagreements` has no parameter for
trimming, because a trimming option is how "preserve material disagreement" quietly stops
holding once a report looks crowded. A test asserts that a judge preserving *nothing* still
delivers the upstream dissent to the reader.

#### An EvidenceGap is never a consequence

Type E is a claim about the **corpus**, not about the world (checkpoint 2), which is why it
can be neither an edge nor a consequence. Gaps enter the assembly as **ids** and leave as
`Consequence.evidence_gap_ids` - a finding pointing at a known hole. `dossiers_from` takes
`evidence_gap_ids` rather than gap records, so there is nothing to promote, and a test asserts
that adding gaps to an analysis adds no consequences.

#### The report surface only narrows

`in_default_view` can turn the judge's `True` into `False` and never the reverse: a `rejected`
finding is never foregrounded, and Class B reaches the default view only at HIGH or CRITICAL
severity **with** a historical analogue. Reviewers who receive many findings of which few
matter stop reading, and the system then has no value regardless of its precision.

#### Uncertainty is reported in four separate buckets

`arbitrated`, `undecided`, `clamped` and `open_questions`, plus `refused`. Kept apart because
"the judge rated it low", "the gates lowered it", "the judge never dispositioned it" and
"nobody could answer this" are four different facts, and collapsing them is exactly how an
uncertainty-aware report becomes a confident one. `JudgmentReport.unsupported_share` is the
honest headline for a thin corpus: a high value is the corpus telling the reader what it
cannot establish.

Deliberately absent from this slice: the blind-spot hunter, graph projection, the report
composer, any UI, and any change to retrieval.

## 7. Evidence architecture

```text
SourceDocument
   |
   +--> SourceVersion
           |
           +--> Chunk
                   |
                   +--> Embedding
                   |
                   +--> EvidenceItem
```

EvidenceItem is the stable reference used by agents.

Every consequence references EvidenceItem IDs.

### 7.1 Assumption store

```text
AssumptionKeyRegistry (CLOSED, versioned)
   |  resolution required for admission
   v
EvidenceItem
   |
   +--> AssumptionRecord
   |         key, scope, TypedValue, role, explicitness,
   |         holder_source_version_id, evidence_id, registry_version,
   |         [effective_from, effective_to)
   |
   |    (role = declares | consumes | describes  -- collision semantics depend on it)
   |    (explicitness = explicit | derived_from_text | implied  -- feeds confidence)
   |
   +--> Coupling Index:   (key, scope, as_of) -> {holders}
   +--> Outcome Index:    (key, scope, as_of) -> consistent
                                               | conflicts_with
                                               | legitimate_override
                                               | unauthorized_override
                                               | ambiguous_temporal

KeyRegistryGap   quarantined unregistered keys; volume is a product signal
KeyAlias         proposed | approved | rejected | superseded
                 only `approved` participates in Class A coupling
```

The coupling index is a materialized, deterministic projection of the assumption store.
It is rebuilt on ingestion, **never written by an agent**.

Two properties are load bearing:

- The index is keyed on the **triple** `(key, scope, as_of)`. Keying on `key` alone reports
  every legitimately scoped override as a contradiction, which floods the highest-trust
  output with false positives.
- Divergence between declarers resolves **three** ways, not one, and the third depends on
  `authority_class` resolved from the holder document. See `DOMAIN_CHECKPOINT.md` 5.4.

### 7.2 Temporal authority resolution

"Which version governed on date D" is deterministic and recurs throughout the system:
version selection, staleness detection, contradictory effective dates, and
retention-window reasoning all depend on it.

It is a single explicit domain service, not logic re-derived per agent:

```text
resolve_authority(key | source_id, as_of: date) -> SourceVersion | Ambiguous
```

`Ambiguous` is a real return value: overlapping or missing `effective_from` /
`effective_to` ranges are a finding about the corpus, not an error to be smoothed over by
picking the newest row.

## 8. Knowledge graph

The graph is not the only knowledge source.

It is built from:
- explicit metadata
- extracted entities
- validated relationships
- inferred consequence edges

Important distinction:
- observed edge: explicitly supported
- inferred edge: generated by reasoning and supported by evidence
- hypothesized edge: not yet validated

Every inferred edge stores its derivation and evidence.

### 8.1 The projector, as implemented

`app/graph/projection.py`. One pure function, `project(...)`, that takes admitted domain
records and returns a `ConsequenceGraph` plus everything that did not reach it.

```text
ResolvedBinding[]  AssumptionDelta[]  DerivedRelation[]  Consequence[]  EvidenceGap[]
       |                  |                   |                |             |
       +------------------+---------+---------+----------------+-------------+
                                    |
                        app/graph/projection.py
                                    |
                    ConsequenceGraph  +  SkippedRecord[]
```

#### Four properties, each structural rather than conventional

**No agent can author a node or an edge, because this layer cannot reach one.** `app/graph`
is a guarded layer permitted to import `app.domain` and `app.ingestion` and **nothing else** -
not `app.agents`, not any of the four agentic packages, not `app.retrieval`,
not `app.infrastructure`. Checkpoint 10.2 is therefore a property of the import graph, asserted
by the layering contract test and by parsing the module's imports.

**Nothing is invented from absence.** A record that cannot legally project is *reported* with
its reason, never approximated:

| skip reason | when |
|---|---|
| `no_arbitrated_consequence` | a derived relation nothing arbitrated - the three derived edges are consequence-bearing |
| `consequence_rejected` | the arbitration concluded the claim does not hold |
| `endpoint_not_in_supplied_bindings` | a relation's version ref has no document, which would dangle |
| `endpoints_identical` | one document disagreeing with itself is an anomaly, not a relationship |
| `change_carries_no_evidence` | an observed edge must cite a span, and none was supplied |
| `gap_partition_matches_no_node` | the gap still attaches to the analysis; it annotates no vertex |

A projection that silently dropped a contradiction and one that found none look identical
otherwise, which is why the refusals are part of the result rather than a log line.

**Type E never becomes an edge.** No relationship admits `Derivation.ABSENT`, so `GraphEdge`
is structurally incapable of carrying a gap. The projector's only use of an `EvidenceGap` is
to annotate the node its `partition` names, matched by **exact label** - interpreting a gap's
prose to decide which node it concerns would put a judgement in the one component that must
contain none.

**Two rebuilds of one store are byte-identical.** Node and edge ids are content-derived
(`node_id_for`, `edge_id_for` in `app/ingestion/identity.py`), every collection is sorted on a
total key, and `projected_at` is the caller's - it also pins `created_at`, because a default
clock read would make every rebuild differ in a field nobody set. A minted id would make every
rebuild produce a different graph and every stored edge dangle after one.

#### Identity recipes

`node_id_for(entity_type, natural_key)` includes the type, so an assumption key and an entity
sharing a label cannot collapse onto one node - which would silently merge two different kinds
of thing in the one visualization the product exists for.

`edge_id_for(relationship, source_node, target_node, record_id)` includes the record, which
does double duty. It keeps two facts about one pair distinct (a contradiction on one key, a
staleness on another), and it makes the projection idempotent: one `conflicts_with` relation
yields a Class A *finding per affected artifact*, so projecting per finding visits the relation
twice and both visits derive the same id.

#### Canonical ordering is the projector's job

`conflicts_with` between A and B is one fact. The projector swaps the endpoints into canonical
order before minting the id, so the entity's ordering validator never fires and two
projections of one store cannot differ. Directed relationships keep the resolver's
orientation - `unauthorized_override` runs narrower → broader, because the narrower artifact
is the one deviating.

#### Measured on the sample corpus

21 admitted bindings, 8 derived relations, no arbitrated consequences:

| | count |
|---|---|
| nodes | 22 (12 `source_document`, 9 `assumption_key`, 1 `change`) |
| edges | 21, all `binds_assumption` |
| skipped | 8, all `no_arbitrated_consequence` |

Five assumption keys have more than one holder: `refund.approval_threshold` (6),
`retention.window` (5), `sla.p1_response_time` (2), `auth.reset_token.ttl` (2),
`api.max_page_size` (2).

**`retention.window` is the demonstration.** Its five holders are a *Data Retention Policy*, an
*EU Data Protection Policy*, a *Customer Lifetime Value Metric Definition*, *Vendor Integration
Notes* and *EMEA Support Team Notes*. Lexical retrieval recovered **one** of them
(`docs/EVALUATION.md`, coupling recall), so a reader relying on text search sees the policy and
the graph shows all five. Nobody reading the retention policy would think to check the metric
definition - which is exactly the coupling the `assumption_key` node type exists to make
visible, and the reason the graph is worth projecting at all.

Note what is absent from that measurement and why: with no arbitrated consequences supplied,
**no derived edge projects**, and the eight relations the resolver found are reported as
skipped rather than drawn. The graph shows what the pipeline concluded, not everything it
computed.

### 8.2 Reviewer dispositions, as implemented

`app/judgment/dispositions.py` holds the rules; `app/application/use_cases/record_disposition.py`
is the path that records them; `InMemoryDispositionStore` is the store behind the
`DispositionStore` port.

`IDEA.md` 21.2 makes human disposition at review time the **primary** feedback signal, and the
reason shapes the design: outcome matching cannot be primary because attribution is ambiguous,
misses are unobservable, and a prevented consequence is counterfactually invisible - so a
system that works looks like one producing false positives. A reviewer's verdict has none of
those problems.

#### One verdict per `(analysis_id, consequence_id)`

Identity is the **pair**, not the finding. The same consequence reviewed in two analyses is
two verdicts, because a reviewer judges what they were shown - the evidence, the status and
the confidence of *that* run - and carrying a verdict forward silently would attribute an
opinion to a report the reviewer never saw.

#### Five values, and what each does

| value | foreground | reason required |
|---|---|---|
| `useful` | shown | no |
| `already_known` | **shown** | no |
| `not_material` | suppressed | **yes** |
| `wrong` | suppressed | **yes** |
| `unverifiable` | shown | no |

`FOREGROUND_SUPPRESSING` and `REASON_REQUIRED` both *read* `SUPPRESSING_DISPOSITIONS` rather
than restating it, so a change to the frozen set cannot leave this layer behind.

The reason requirement is the same asymmetry the judge's `immaterial` carries: the direction
that **subtracts** something from a reader's view must be justified. `unverifiable` is
deliberately excluded - "I could not tell" is a complete answer, and demanding prose for it
would push reviewers toward a value they can defend rather than the one that is true.

#### `already_known` does not suppress, and that is the point

It is the signal the product thesis rests on: a high rate means the system is mostly finding
what a diligent reviewer would find anyway. Suppressing it would destroy the measurement *and*
hide a correct finding - the reviewer already knew it, which is a statement about the reviewer
rather than about the finding. The frozen entity also refuses to let it request persistent
suppression, so this is not a rule the application layer could relax.

#### Two suppressions, and they are different

- **Foreground**, immediately: `wrong` and `not_material` remove the finding from the default
  view of *this* analysis. Computed on read by `foreground(...)`, which reads only the verdict -
  no confidence threshold, no severity, no age.
- **Future analyses**, only when the reviewer asks: the same verdict suppresses the finding for
  the same `(assumption_key, artifact_ref)` pair until the source is revised. `artifact_ref` is
  the affected source version id, which is content-derived - so a revised document has a
  different one and the suppression **expires by itself**. The reviewer said the finding is
  wrong about *that text*, not that they never want to hear about the document.

#### Suppression never deletes

No module in this slice constructs, copies or re-provenances a `Consequence`, a
`ConsequenceScore`, an `ArbitrationDecision` or a graph record - asserted by parsing both
modules for those constructors and for `model_copy`. A suppressed finding is still in the
analysis, still evidenced, still scored and still projected into the graph; it is simply not
shown first. A test serializes every consequence before and after a `wrong` verdict and
compares.

That is what makes the signal safe to give. A verdict recorded with no way back is a verdict
reviewers hesitate to give, and hesitant feedback is worse than none.

#### Reversible and auditable

`DispositionRecord` is an envelope over the frozen entity - the same pattern as `GapRecord` and
`FinalizedRetrieval` - carrying `superseded` verdicts and one `DispositionChange` per
transition with **both** values, the reviewer, the timestamp and the reason. "Changed to
useful" is not auditable without knowing what it was.

`change(...)` requires a reason **whatever the value**, which is stricter than `record(...)`: a
first verdict on `useful` needs no defence, but a change is the operation a reviewer performs
when they got it wrong, and a transition with no explanation is the one thing an audit trail
cannot use. The port has no `delete` and the implementation has none either.

#### Idempotency is not "ignore the second one"

Re-submitting the *same* verdict returns the existing record and appends nothing: a
double-click must not fabricate a transition. Equivalence compares what a reviewer actually
decided - the value, the note and the suppression request - and deliberately not the identity
or the timestamp, since a retry seconds later carries a new id and a new clock reading.

#### The signal, reported as rates

`DispositionSignals` reports `already_known_rate`, `useful_rate`, `wrong_rate` and `coverage`,
computed over the **reviewed** set. An analysis nobody has looked at reports `None` rather than
a 0% already-known rate that would read as a success. A corrected verdict moves the rate rather
than adding to both buckets, because only the current verdict is counted.

## 9. Persistence, as implemented

`app/infrastructure/database/`, with the schema owned by Alembic
(`migrations/versions/0001_initial_persistence_schema.py`). Twelve tables behind the seven
store ports; the in-memory implementations stay, and one conformance suite runs against both.

| port | tables |
|---|---|
| `SourceStore` | `source_documents`, `source_versions`, `chunks`, `evidence_items` |
| `AssumptionStore` | `assumption_records` |
| `RegistryGapStore` | `registry_gaps` |
| `AliasStore` | `key_aliases` |
| `CouplingIndexStore` | `coupling_sets`, `derived_relations`, `class_a_findings` |
| `DispositionStore` | `finding_dispositions` |
| `AnalysisStore` | `stored_analyses` |

### The mapping: JSONB payload plus projected columns

Every row carries the frozen domain entity verbatim in a `payload` JSONB column, plus the
handful of columns the ports actually query on - ids, tenant, assumption key, source version,
status. Reads are `Model.model_validate(row.payload)`.

This is a document mapping with relational indexes, chosen against normalization for two
reasons:

**Normalizing would create a second definition of every domain type.** The entities nest
`TypedValue`, `ScopeSelector`, `ModelProvenance`, `RunProvenance`, `PermissionScope` and a dozen
more frozen value objects, several carrying model validators that encode rules - an approved
alias names a human and a time; an interval is ordered; a blocked coupling strength is zero.
Spreading those across columns would put each rule in two places, and the two would drift.
Round-tripping through `model_validate` instead means **every read re-runs every validator**, so
a row the domain could not have produced cannot be served by the store either.

**The ports do not ask relational questions.** Every method in the contract is an id lookup, a
tenant enumeration, or a filter on one key. There is no join, no aggregate and no range scan in
it, so the columns are exactly what the contract needs. JSONB keeps `payload -> 'status'`
reachable if that changes, and a GIN index can be added per query when one has a reader.

What is given up is SQL-level queryability *inside* an entity. That is a real cost and it is
the one to revisit first if a reporting surface ever needs it.

**Projected columns are write-time projections, never read back into an entity.** A column
exists so a filter is indexable; the value served always comes from `payload`. So the worst a
stale column can do is make a *filter* wrong - a test corrupts one and asserts exactly that.
Where a port's predicate is a derived property (`blocks_class_a`, `participates_in_class_a`),
the column stores the underlying `status` and the query repeats the property's predicate, so
the entity keeps one definition of it.

**Ordinals, because SQL has no insertion order.** `versions_for` promises oldest first;
`coupling_sets` returns the tuple the resolver produced. An `ordinal` column assigned at write
time reproduces exactly what the in-memory implementation returns. Without it the two
implementations would satisfy the type and disagree about the answer.

### Transactions

One transaction per store call by default: the ports have no transaction concept and the
in-memory stores commit immediately, so that is the faithful mapping.
`Database.unit_of_work()` binds one session for its duration and the repositories join it
instead of opening their own - which is what makes the three coupling-projection writes atomic
*when the caller asks*, without putting a transaction in the port. Nesting is refused rather
than flattened, because a caller expecting an independent inner rollback would be wrong in a
way nothing reports.

No dialect-specific upsert. `put_*` is read-then-write, because the port semantics differ per
method - `put_evidence` keeps the first write, `put_alias` replaces, `put_gap` *accumulates a
counter* - and one upsert idiom would flatten those into one behaviour.

### Two envelopes need a codec

`DispositionRecord` and `StoredAnalysis` are frozen dataclasses wrapping Pydantic entities, so
they have no JSON round trip of their own. `app/infrastructure/database/codecs.py` supplies one.
Every decode goes through the domain constructor, and the payload carries an
`ENVELOPE_VERSION` that is verified on read: a payload written by a different envelope is
refused with a message naming both versions, because silently reading one whose meaning has
changed is how a review surface starts disagreeing with the analysis it claims to show.

`AnalysisResult.fingerprint()` is stored **verbatim** and never recomputed. It is what lets a
verdict be checked against the analysis that produced it, so a codec that regenerated it would
make a re-run look like the same analysis.

### Configuration

`CCE_DATABASE_URL`, with **no default**. `database_url()` raises `DatabaseNotConfigured` naming
the variable; `alembic.ini` carries no URL and `migrations/env.py` reads it from settings, so no
credential lives in a committed file. `.env.example` is a placeholder. Three contract tests
assert all three.

pgvector stores embeddings and object storage stores original artifacts - both still ahead.

## 10. Caching and idempotency

Use deterministic analysis IDs derived from:
- normalized change request
- corpus version
- analysis configuration version

Do not blindly cache final LLM output forever.

Cache:
- embeddings
- stable retrieval intermediates
- source parsing
- safe deterministic transformations

Consider caching agent outputs only with explicit prompt/model/version keys.

## 11. LLM gateway and provider strategy

CCE is provider-agnostic by construction. No provider SDK is a required dependency of
development, and no provider name appears in the domain or application layers.

```text
Client / API
   |
   v
Analysis Orchestrator          (application)
   |
   v
Agent Runtime                  (application)
   |
   v
LLM Gateway / ProviderPort     (domain interface)
   |
   +-- Groq adapter            (infrastructure) - default remote provider
   +-- Gemini adapter          (infrastructure) - optional
   +-- Mistral adapter         (infrastructure) - optional
   +-- Ollama adapter          (infrastructure) - optional, local
   |
   v
selected model                 (from configuration, never hard-coded)
```

### 11.1 Capability tiers

Tiers are declared by role, and configuration binds each tier to a provider and model.
Agents request a tier; they never name a model.

| tier | used by | characteristics |
|---|---|---|
| `reasoning` | consequence-judge, adversarial-challenger, consequence-hunter | strongest available |
| `standard` | specialist analysts, contradiction-hunter, report-composer | balanced |
| `fast` | assumption extraction, classification, routing, decomposition | cheap, high volume |

Assumption extraction runs on the `fast` tier over the whole corpus. Making it a
reasoning-tier call would make ingestion the dominant cost of the system.

### 11.2 Configuration shape

Per-agent binding is required so that provider-diversity experiments (section 4a) are a
configuration change rather than a code change:

```yaml
tiers:
  reasoning: { provider: groq, model: ${CCE_MODEL_REASONING} }
  standard:  { provider: groq, model: ${CCE_MODEL_STANDARD} }
  fast:      { provider: groq, model: ${CCE_MODEL_FAST} }

agents:
  change-decomposer:      { tier: fast }
  policy-analyst:         { tier: standard }
  consequence-hunter:     { tier: reasoning }
  adversarial-challenger: { tier: reasoning }
  consequence-judge:      { tier: reasoning }
  # explicit override wins over tier:
  # blind-spot-hunter:    { provider: gemini, model: ${CCE_MODEL_BLINDSPOT} }
```

Model identifiers come from environment/configuration only. Do not commit model names to
source: they are vendor inventory, they change, and a stale hard-coded identifier is an
outage.

### 11.3 Required gateway behavior

- uniform request/response contract across providers
- structured-output enforcement with schema validation and bounded repair retries
- token budget per call and per analysis, enforced before dispatch
- retries with exponential backoff and jitter, distinguishing retryable from terminal
- usage and cost accounting per call, attributed to agent and analysis
- safe caching keyed on (prompt version, schema version, provider, model, params, input hash)
- a deterministic fake adapter for tests, requiring no credentials or network

### 11.4 Embeddings

Embeddings sit behind their own port with the same rules. The default local embedding
implementation must require no API key, so that ingestion and retrieval are developable
and testable offline.

As built, three implementations satisfy `EmbeddingPort` and none imports a vendor SDK:

| implementation | what it is | used for |
|---|---|---|
| `HashingEmbedder` | feature hashing over a bag of words, L2-normalized, `hashlib`-based | the default; the whole test suite and the vector benchmark |
| `ReplayEmbedder` | vectors recorded from a real provider, keyed by `sha256(text)` | the seam a real provider fills; no fixture committed yet |
| `HttpEmbeddingProvider` | OpenAI-compatible `POST {base_url}/embeddings` over `httpx` | the transport seam. **Not enabled, and never called live** |
| `GeminiEmbeddingProvider` | `gemini-embedding-001` via `POST {base_url}/models/{model}:batchEmbedContents` | prepared for the planned experiment. **Not enabled, and never called live** |

Two properties of the default one are load-bearing rather than incidental:

**It is not semantic, and nothing may claim otherwise.** Feature hashing over tokens gives
two texts a high cosine only when they share literal words. It exercises the vector
machinery and provides a term-overlap signal; it knows nothing. Every measured vector figure
in `docs/EVALUATION.md` names the embedder for this reason.

**It hashes with `hashlib`, never the built-in `hash()`**, which is randomly salted per
process. An index built with `hash()` would rank differently on every run, and no benchmark
over it could be compared with its own history.

`ReplayEmbedder` **raises on a cache miss** rather than substituting another embedder or a
zero vector. Either fallback would put two embedders inside one index, where the cosine
between a recorded vector and a substituted one is meaningless — and the resulting recall
figure would read as a retrieval result rather than as a fixture defect.

#### 11.4.1 The adapter foundation

`app/infrastructure/embeddings/base.py` holds what an *adapter* owes, as distinct from what
a caller needs. `EmbeddingPort` is three members wide and says nothing about the three
obligations below, each of which a remote provider can violate in a way that is invisible
downstream. A subclass implements one method — `_embed_batch`, the wire call for one batch —
and inherits the rest, so everything that could go subtly wrong is implemented once.

| obligation | why it is not left to each adapter |
|---|---|
| **dimension validation**, per vector | `VectorRetriever` rejects a mis-sized vector at index construction, which is the right last line of defence but a late one. A mistyped model identifier, or a provider that truncates, is caught here with the index it came from |
| **normalized-vector validation**, where declared | some providers return unit vectors and some do not, and the ones that do occasionally stop. `VectorRetriever` computes true cosine so it survives either, but a *ranking* measured under one convention is not comparable with one measured under the other |
| **batching**, deterministic and order-preserving | `embed` takes a sequence and owes one vector per text in order. A remote provider has a request-size limit, so if the caller splits the call then every caller reimplements it and one will reassemble out of order — and the vector index pairs spans to vectors positionally |

Two further checks are cheap and worth naming: a **non-finite component** is refused (a NaN
compares false against everything and would silently poison every cosine it takes part in),
and an **empty request makes no call at all** (a provider billed per request should not be
paid for a question nobody asked).

`EmbeddingMetadata` carries `provider`, `model`, `dimensions` and `adapter_version`, and
derives the `name` the retriever folds into `scorer_version`. It sits on the **adapter**, not
on `EmbeddingPort`: retrieval needs a name and a width, and widening the domain contract to
carry vendor-shaped provenance would serve no caller. `VectorRetriever` is unchanged.

**Selection is configuration; enablement is a decision.** `EmbeddingRegistry` mirrors
`ProviderRegistry`: membership in `EMBEDDING_FACTORIES` means CCE knows how to build an
adapter, and using one additionally requires it in `CCE_EMBEDDING_ENABLED_PROVIDERS`, which
ships as `("hashing", "replay")` — the two that reach no network. A remote embedder sends
corpus text, including spans from documents under an ACL, to a third party; activating one
because a key appeared in the environment would widen the data-egress surface without anyone
deciding to (`docs/SECURITY.md`). Missing configuration raises `ProviderUnavailable` at
selection time, never mid-run.

Everything the remote adapter needs comes from configuration or the environment, and none of
it has a default: `CCE_MODEL_EMBEDDING`, `CCE_EMBEDDING_API_KEY`, `CCE_EMBEDDING_BASE_URL`,
`CCE_EMBEDDING_DIMENSIONS`. Deliberately absent from the adapter: retry and backoff (a
gateway concern embeddings do not yet share), caching, and per-provider special-casing — a
genuinely different wire format gets its own adapter rather than a flag.

#### 11.4.2 Two wire formats, one set of obligations

Gemini is the second adapter, and it is the case that shows why "a different wire format
gets its own adapter" was the right rule: it differs from the OpenAI shape in four places at
once.

| | OpenAI-compatible | Gemini |
|---|---|---|
| endpoint | `POST /embeddings` | `POST /models/{model}:batchEmbedContents` |
| auth | `Authorization: Bearer <key>` | `x-goog-api-key: <key>` |
| request | `{"model":…, "input": [text, …]}` | `{"requests": [{"content": {"parts": [{"text":…}]}, "outputDimensionality":…, "taskType":…}]}` |
| response | `{"data": [{"index": n, "embedding": […]}]}` | `{"embeddings": [{"values": […]}]}` |

A flag per difference would have made one adapter unreadable. What they share is shared as
code: `post_json` and `raise_for_status` in `http.py` do the transport wrapping and the
status-code translation for both, and both inherit every obligation in `EmbeddingAdapter`.
Error translation is the part an adapter must not get creatively wrong, so there is one of
it.

Three properties of the Gemini mapping are worth recording because they are decisions, not
details:

**The credential goes in a header, never the query string.** Gemini accepts `?key=`, which
is equivalent to the API and much worse operationally: query strings land in access logs,
proxy logs and error reports. A test asserts the URL carries no query at all.

**The response carries no index, so ordering is positional and unenforceable.** OpenAI's
shape lets the adapter sort by the provider's own `index`; `batchEmbedContents` returns
`embeddings[]` with nothing identifying which request each entry answers. The count is
checked — that catches a dropped or duplicated vector — but a *permuted* response would
produce a plausible wrong ranking rather than an error. Documented here because it cannot be
validated away.

**Output width is requested explicitly and then validated.** `gemini-embedding-001` is a
Matryoshka model: 3072 native, with 1536 and 768 as supported truncations. The adapter sends
`outputDimensionality` (without it the model returns 3072 and every vector would be the
wrong width), refuses an undocumented width before making a request, and validates the
returned width per vector. Google documents sub-3072 output as **not** unit length, so
`expects_unit_vectors` is false and the adapter does not normalize — cosine is
scale-invariant, so ranking is identical either way, and transforming a provider's numbers
inside a mapping adapter would make a recorded fixture something other than what the
provider returned.

## 12. Error handling

Agent error:
- record failure
- retry according to policy
- downgrade analysis scope if safe
- expose degraded status

Retrieval error:
- retry
- use alternate strategy
- mark corpus limitation

Judge error:
- fail closed for claims
- return unresolved state
- do not fabricate a final conclusion

Provider error:
- retry with backoff on rate limit and transient transport failure
- optionally fail over to another configured provider for the same tier
- record the provider actually used on every finding, since provider identity is an
  experimental variable
- if no provider is available, degrade to the deterministic Class A analysis and report
  the agent layer as unavailable rather than returning nothing

Schema-validation error:
- one bounded repair attempt
- then discard the finding and record a parse failure
- never coerce malformed output into a partially populated finding

## 13. Observability

Each analysis gets:
- correlation ID
- analysis ID
- trace ID

Each model invocation records:
- agent
- prompt version
- model
- latency
- input/output token counts if available
- cost estimate
- status

Do not log secrets or unrestricted source contents.

## 14. Security architecture

Authorization is enforced before evidence is returned to an agent.

```text
User -> API Auth -> Project Scope -> Retrieval ACL Filter -> Evidence
```

Agents never receive more source content than the requester is authorized to access.

### 14.1 Derived-content leakage

Filtering *evidence* by ACL does not prevent a *consequence statement* derived from
restricted evidence from carrying its content. Because analyses are expensive, the
temptation to cache and share them is strong, and a shared analysis is a cross-ACL
aggregation channel.

Decision for the MVP: **an analysis is scoped to the permission set it was computed
under.** Concretely:

- every analysis stores the `permission_scope_hash` it was computed with
- an analysis is served only to a requester whose effective scope hash matches
- a differing scope forces recomputation; it never triggers a filtered replay of stored
  findings

This is more expensive and it is the only version that is correct. Revisit only with a
per-finding ACL derived from the union of its cited evidence, which is strictly harder to
get right.

## 15. Deployment shape

MVP local:
- FastAPI
- Postgres + pgvector
- Redis
- worker process
- object storage emulator or local storage
- optional UI

Production:
- API service
- worker service
- Postgres
- object storage
- Redis
- observability stack
- secret manager
- external model provider

Use containers.

## 16. Scaling

Parallelize independent specialist agents.

Queue-based execution should support:
- concurrency limits
- per-tenant limits
- model rate limits
- retries
- dead-letter handling

Do not spawn unlimited agents.

## 17. Cost controls

Before expensive analysis:
- classify change complexity
- estimate evidence volume
- select specialist set
- set token budgets
- prioritize high-severity hypotheses
- stop early when evidence converges

A cheap baseline should be able to reject obviously small analyses.

## 18. Framework stance

Agent framework is an implementation detail.

Preferred dependency direction:

domain <- application <- infrastructure

The domain layer must not import LangChain/LangGraph/Anthropic SDK directly.

Adapters in infrastructure translate model/framework calls into domain interfaces.

## 19. UI data shape

Graph:
nodes:
- id
- label
- type
- owner
- source_count
- risk_score

edges:
- id
- source
- target
- relationship
- confidence
- severity
- evidence_ids
- inference_type

## 20. Recommended first vertical slice

Do NOT start with UI.

The first slice must prove the core mechanism, not the agent topology. Ordering is
deliberate: schemas precede prompts, and the deterministic path precedes the agent path.

Build:
1. ChangeSpec, EvidenceItem, AssumptionRecord, ConsequenceHypothesis (schemas first)
2. one ingestion format
3. assumption extraction at ingestion
4. coupling index + collision detection (deterministic)
5. temporal authority resolution
6. lexical + vector retrieval
7. invalidation resolver + coupling propagation (deterministic)
8. deterministic confidence scorer
9. change-decomposer producing an AssumptionDelta
10. three specialist agents over per-specialist evidence slices
11. evidence arbiter
12. report JSON
13. benchmark harness with negative controls and corpus ablation

Then add:
- consequence-hunter (Class B)
- challenger with citation obligation
- blind-spot partition
- graph
- UI

A useful checkpoint: steps 1-8 alone produce the knowledge-decay report
(`IDEA.md` 15.5) with no consequence analysis at all. If that output is not useful,
the corpus is not ready and no amount of agent work will fix it.
