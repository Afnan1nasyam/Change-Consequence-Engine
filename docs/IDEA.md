# Change Consequence Engine (CCE)
## Product / Research Idea Specification

## 1. Executive definition

Change Consequence Engine is a multi-agent evidence analysis platform for answering a question that conventional enterprise search, RAG, ticket systems, and explicit dependency maps handle poorly:

> Given a proposed change, what consequences could appear elsewhere in the organization even when those relationships are not explicitly documented?

The system ingests heterogeneous organizational knowledge, constructs evidence and relationship representations, generates consequence hypotheses from multiple specialist perspectives, independently challenges those hypotheses, and produces a traceable Change Consequence Graph.

The primary output is not a chat answer. It is an auditable impact package containing:
- proposed change interpretation
- directly affected entities
- inferred/latent consequences
- evidence supporting each consequence
- contradictions
- confidence and severity
- missing evidence
- unresolved questions
- historical analogues
- blind-spot findings
- recommended validation actions

## 2. Why the problem matters

Organizations routinely make changes through one system of record while the consequences live somewhere else.

Examples:
- a policy changes but SOPs remain stale
- an authorization threshold changes but training material remains outdated
- an API contract changes but an undocumented operational workflow depends on the old behavior
- a retention rule changes but analytics/reporting logic assumes a different historical window
- an approval workflow changes but the resulting delay invalidates an SLA or downstream process
- a terminology change makes old dashboards, tickets, runbooks, or automation rules semantically inconsistent

The difficulty is not only finding documents containing the same words. The difficulty is discovering weak, implicit, cross-domain relationships.

## 3. Core distinction

### Conventional impact analysis

Known dependency:
A -> B -> C

The system follows explicit references.

### CCE target

Latent consequence:
A -> [implicit relationship] -> D

The relationship may only be discoverable from:
- semantics
- workflow descriptions
- role responsibilities
- historical incidents
- configuration metadata
- timing assumptions
- terminology
- exceptions
- contradictory versions
- code/document relationships
- repeated operational behavior

CCE therefore treats consequence discovery as hypothesis generation plus evidence arbitration, not as simple graph traversal.

## 4. Example

Change request:

"Customer refunds above 5,000 now require manager approval instead of 10,000."

Potential findings:

Direct:
- refund approval policy changed
- authorization matrix changed
- refund workflow changed

Hidden:
- support response time may increase
- refund SLA may be affected
- training content may teach an obsolete threshold
- fraud monitoring baselines may shift
- finance reconciliation may see different timing
- escalation load may increase

Contradiction:
- policy says 5,000
- old support SOP says 10,000
- payment service configuration says 5,000

Historical analogue:
- a similar threshold change generated an increase in escalations

Blind spot:
- an independent agent identifies a reconciliation dependency not found by the main analysis

## 5. Product goals

Primary goals:
1. Detect consequences across heterogeneous information.
2. Surface latent relationships, not only explicit dependencies.
3. Provide evidence for every meaningful finding.
4. Make agent disagreement visible.
5. Distinguish likely impact from speculation.
6. Identify corpus blind spots and missing validation data.
7. Learn from predicted-vs-actual outcomes over time.
8. Produce actionable outputs for change owners.

Non-goals:
- automatic execution of production changes
- replacing change-approval committees
- autonomous legal/compliance decisions
- pretending to know facts absent from the corpus
- generic enterprise chat

## 6. Intended users

Initial:
- engineering change managers
- operations teams
- enterprise platform teams
- support operations
- internal audit/risk teams
- program/project managers handling cross-functional changes

Future:
- regulated operations
- manufacturing change management
- financial operations
- large internal knowledge environments

## 7. Core workflow

### Stage A — Change understanding

The Change Decomposer converts the request into a structured change specification:
- subject/entity
- old state
- new state
- scope
- effective date
- explicit exclusions
- affected business capability
- assumptions
- ambiguities

### Stage B — Evidence retrieval

Multiple retrieval strategies obtain candidate evidence:
- semantic retrieval
- lexical/keyword retrieval
- metadata filtering
- graph-neighborhood retrieval
- temporal retrieval
- source-type retrieval
- historical similarity retrieval

### Stage C — Independent specialist analysis

Specialists analyze the same change from different perspectives:
- policy
- process
- technology
- people/roles
- data/reporting
- risk/compliance
- history/incidents

They do not simply copy each other. Their perspectives are deliberately different.

### Stage D — Consequence hypothesis generation

A consequence hunter combines findings and explores second-/third-order hypotheses.

Each hypothesis must declare:
- affected entity
- relationship type
- mechanism
- evidence
- confidence
- severity
- uncertainty
- assumptions

### Stage E — Adversarial challenge

A challenger tries to falsify each important hypothesis.

It looks for:
- contrary evidence
- exceptions
- newer versions
- compensating controls
- alternate mechanisms
- evidence insufficiency

### Stage F — Independent blind-spot search

The Blind-Spot Agent receives the change and evidence environment but is instructed to search for consequences that the main pipeline has missed.

### Stage G — Arbitration

The Consequence Judge evaluates:
- evidence quality
- evidence independence
- recency
- source authority
- logical consistency
- number of independent supporting paths
- contradiction strength
- uncertainty

### Stage H — Graph construction

The resulting impact graph contains:
- source entities
- affected entities
- relationships
- evidence edges
- inferred edges
- confidence
- severity
- temporal context
- contradiction markers

### Stage I — Report generation

Output:
- executive summary
- high-risk consequences
- hidden consequences
- contradictions
- evidence table
- blind spots
- unresolved questions
- recommended validation steps

## 8. The multi-agent thesis

Multiple agents are justified because consequence discovery has competing epistemic tasks.

A single model tends to:
- collapse uncertainty
- anchor on the first plausible narrative
- miss evidence in unrelated domains
- overfit to a single retrieval strategy
- fail to challenge its own assumptions

CCE separates:
- interpretation
- domain analysis
- hypothesis generation
- evidence validation
- adversarial falsification
- blind-spot exploration
- arbitration

The goal is not "more agents = better."

The goal is controlled diversity of analysis followed by evidence-based arbitration.

## 9. Proposed agents

1. change-decomposer
2. policy-analyst
3. process-analyst
4. technology-analyst
5. people-analyst
6. data-analyst
7. risk-analyst
8. historical-analyst
9. consequence-hunter
10. contradiction-hunter
11. adversarial-challenger
12. blind-spot-hunter
13. evidence-arbiter
14. consequence-judge
15. report-composer

Do not automatically invoke all agents for every request. The orchestrator should select agents based on the change profile and available corpus.

## 10. Evidence and assumption model

### 10.1 Evidence model

Every source fragment becomes an EvidenceItem with:
- evidence_id
- source_id
- source_type
- title
- location
- content/span
- source_version
- created_at
- effective_from
- effective_to
- authority_score
- retrieval_score
- freshness_score
- metadata

Claims reference evidence IDs.

A claim without evidence is:
- hypothesis, if logically plausible
- unsupported, if presented as fact

### 10.2 Assumption model

Evidence is what a source *says*. An assumption is what a source *commits to*.

An AssumptionRecord is a typed, atomic proposition that an artifact binds:

- assumption_id
- key            (resolved in the **closed** AssumptionKeyRegistry)
- registry_version
- scope          (**required**; ScopeSelector, defaults to `global`)
- value          (typed: scalar / interval / duration / enum / boolean / reference)
- value_type, unit
- role           (declares | consumes | describes)
- explicitness   (explicit | derived_from_text | implied)
- holder_ref     (source_version_id)
- evidence_id    (the span the record was extracted from)
- effective_from / effective_to
- extractor_version
- span_verified / value_parsed / key_resolved
- extraction_quality  (**computed**, never model-supplied)

Three rules give this model its power:

1. The **coupling set** of a key is every artifact that binds it at a compatible scope.
   Coupling is derived from co-binding, not from cross-reference. Two documents that
   never mention each other are coupled if they bind the same key. The set is identified
   by the triple `(key, scope, as_of)`.

2. Divergence between two **declaring** artifacts resolves deterministically into one of
   three outcomes, decided by scope and by the holders' `authority_class`:
   same scope is a **contradiction**; a narrower scope with sufficient authority is a
   **legitimate override**; a narrower scope with insufficient authority is an
   **unauthorized override**, itself a Class A governance finding.

3. Only a **registered** key, and only an **approved** KeyAlias, may participate in Class A
   coupling. An unregistered key is quarantined as a KeyRegistryGap and never indexed.

Assumption extraction happens once per source version at ingestion time and is cached.
It is not repeated per analysis.

Full contract: `docs/DOMAIN_CHECKPOINT.md`, which is authoritative.

## 11. Consequence model

Each consequence contains:

consequence_id
change_id
title
description
category
affected_entity
mechanism
severity
status
consequence_class
origin
evidence_ids
contradicting_evidence_ids
assumption_keys
evidence_gap_ids
recommended_validation
agent_sources
merged_hypothesis_ids

Status values:
- supported
- likely
- disputed
- speculative
- rejected
- unresolved

`likelihood` is deliberately absent: it was undefined and is a probability by another name,
which 13 forbids until calibration has been separately tested.

`confidence` is deliberately absent from this list. It is computed downstream onto a
separate ConsequenceScore, never carried by anything a model authors (13).

## 12. Relationship taxonomy

The MVP taxonomy is deliberately small. A large taxonomy applied by many agents is
applied inconsistently, and a permissive catch-all edge absorbs most of the graph and
stops carrying information.

MVP edge types (seven), each with fixed semantics. `derivation` distinguishes
deterministic derivation (`derived`) from model inference (`inferred`) — they are separate
values, because collapsing them destroys the auditability of Class A.

| edge | meaning | derivation | direction |
|---|---|---|---|
| `directly_changes` | the ChangeSpec explicitly alters this entity | observed | directed |
| `binds_assumption` | artifact commits to a key/value; **a projection of the record, never authored** | extracted | directed |
| `invalidates_assumption` | the change makes a bound assumption false | **derived** | directed |
| `conflicts_with` | two declarers bind the same key at the **same scope** with divergent values | **derived** | symmetric |
| `unauthorized_override` | a narrower-scope declarer of **lower authority** contradicts a broader one | **derived** | directed (narrower → broader) |
| `creates_risk_for` | a behavioral/load effect, not assumption-mediated | inferred (LLM + analogue) | directed |
| `historically_analogous_to` | a prior change/incident with an observed outcome | **inferred** — the incident is observed, the analogy is a judgment | directed |

Deliberately excluded from the MVP: `potentially_impacts` (unbounded catch-all),
`depends_on`, `constrains`, `enables`, `operationally_precedes`,
`operationally_follows`, `compensates_for`, `measured_by`.

Ownership (`owned_by`), documentation (`documented_by`), and implementation
(`implemented_by`) are node metadata in the MVP, not edges.

The taxonomy is extensible, but a new edge type requires written semantics and an
admissible inference_type before any agent may emit it.

## 13. Confidence is not probability

The system should not claim calibrated probabilities until they are empirically calibrated.

Confidence is computed by **deterministic code over evidence features**, not asserted by
a model. A model-asserted confidence number is unauditable, untunable, and drifts
silently with prompt changes.

Factors and their direction:

| factor | source | direction |
|---|---|---|
| source authority | `authority_class` of holding artifact | + |
| source freshness | `effective_from`, version recency | + |
| independent corroboration | count of *retrieval-independent* supporting evidence | + |
| explicitness | assumption `explicitness` (`explicit` > `derived_from_text` > `implied`) | + |
| alias mediation | approved KeyAlias rather than a direct key match | - |
| contradiction | presence/strength of divergent binding | - |
| inference depth | hops from `directly_changes` | - |
| retrieval reliability | score agreement across strategies | + |

Two constraints on the implementation:

1. **Corroboration must be independent.** Multiple agents reading the same
   EvidenceBundle are not independent sources. Corroboration counts distinct
   `source_version_id`s reached by distinct retrieval strategies, never distinct agents.
2. The scoring function is versioned and its inputs are persisted, so any score can be
   recomputed and explained after the fact.

The function is a tunable weighted aggregation, deliberately simple until the feedback
loop provides data to fit it.

## 14. Severity

Initial severity:
- LOW
- MEDIUM
- HIGH
- CRITICAL

Severity measures potential operational/business consequence, not confidence.

A consequence can be:
- HIGH severity, LOW confidence
- LOW severity, HIGH confidence

These must not be conflated.

## 15. Core mechanism

The differentiating mechanism is:

> **Assumption-coupling analysis** - latent consequence discovery by invalidation of
> co-bound assumptions, with citation-obligated falsification.

A pipeline topology is not a mechanism. Specialist separation, critic models, candidate
diversification, and provenance-preserving RAG are all established patterns; assembling
them yields a systems contribution at best. The claim below is narrower and testable.

### 15.1 The reframing

A latent consequence exists because two or more artifacts **independently bind the same
unstated proposition**. They do not reference each other. They are coupled through a
hidden variable.

The refund case:
- the refund policy binds `refund.approval_threshold == 10000`
- the support training deck binds `refund.approval_threshold == 10000`
- the reconciliation runbook binds `refund.manager_approval_rate <= N/day`

No document cites another. All three are coupled. Conventional impact analysis follows
references and finds none of it.

So the unit of analysis is not the document and not the entity. It is the **assumption**.
A change invalidates assumptions; a consequence is an artifact that binds an invalidated
assumption.

### 15.2 The mechanism

| step | kind | output |
|---|---|---|
| 1. Extraction | LLM, once per source version, cached at ingestion | AssumptionRecords |
| 2. Coupling | deterministic | assumption key -> coupling set |
| 3. Change delta | structured | ChangeSpec as assumption deltas |
| 4. Propagation | deterministic | invalidated keys -> candidate affected artifacts |
| 5. Mechanism articulation | LLM, focused, per candidate | why invalidation harms this artifact |
| 6. Falsification | LLM, citation-obligated | contradicting evidence_id or `no_refutation_found` |

Steps 2, 3 and 4 contain no model call. Step 1 is amortized across every future analysis
of the same corpus. This directly serves principle 4 in `CLAUDE.md`: discovery of the
latent edge is computed, and the LLM is confined to extraction and explanation - the two
tasks it is actually reliable at.

### 15.3 What the reframing buys

- **Latent edges become computed and checkable** rather than model-asserted, so recall is
  measurable against a deterministic ground truth instead of a human opinion.
- **Contradiction detection becomes exact.** A divergent binding of the same key is a
  comparison, not an inference. It does not degrade as the corpus grows - unlike asking a
  model to notice `5,000 != 10,000` across a long context, which degrades exactly when
  the corpus gets large enough to be interesting.
- **Cost inverts favorably.** Analysis-time work is mostly deterministic traversal plus a
  bounded number of focused calls, instead of re-reading the corpus with seven
  specialists on every run.
- **Documentation gaps become a first-class output.** An assumption with exactly one
  holder is undocumented coupling; a key bound by code but by no policy is an unwritten
  rule. See 15.5.
- **Reproducibility becomes attainable.** The coupling graph is a pure function of the
  assumption store.

### 15.4 Honest scope limit

Assumption-coupling covers two consequence classes very differently.

**Class A - assumption-mediated** (documentation and configuration consistency): stale
thresholds, contradictory SOPs, misleading training material, drifted configuration,
invalidated definitions, broken temporal assumptions. Deterministically discoverable,
verifiable at delivery time, high trust. This is the class the mechanism owns.

**Class B - behavioral / quantitative** ("escalation volume will rise", "reconciliation
timing shifts"): causal and load-driven, not assumption-binding. No deterministic
derivation exists. These remain LLM hypotheses and **must** be anchored to a historical
analogue with an observed outcome, or be marked `speculative`.

Class B is where the product is most likely to be wrong and most likely to sound
impressive. Class A is where trust is earned. Do not let Class B output dominate the
report surface.

### 15.5 Corollary product: knowledge-decay detection

The assumption store yields the following with no change request as input:

- keys bound with divergent values (live contradictions)
- keys with a single holder (undocumented coupling, single point of knowledge)
- keys bound by code or configuration but by no policy or SOP (unwritten rules)
- keys whose holders have not been revised since the value last changed (stale docs)

This is a standing corpus-health report. It is cheaper than consequence analysis, requires
no prediction of the future, and is verifiable on delivery. Treat it as a first-class
deliverable, not a side effect.

### 15.6 Prior-art discipline

The broad problem has extensive prior art. The narrow claim is: *coupling inferred from
co-binding rather than from reference, a change expressed as an assumption delta, and
falsification carrying a citation obligation.* Any public originality claim must be
checked against current prior art before publication.

## 16. Production requirements

Must support:
- authentication boundary
- tenant/project isolation in data model
- source permissions
- configurable retention
- audit logs
- rate limiting
- retries
- idempotency
- request correlation IDs
- traceability
- model/provider abstraction
- cost accounting
- prompt/version tracking
- evidence provenance
- deterministic reproducibility controls where feasible

## 17. Failure behavior

If evidence is inadequate:
- do not hallucinate
- return unresolved findings
- explicitly list missing information
- recommend what source or validation would reduce uncertainty

If agents disagree:
- preserve disagreement
- run targeted challenge/retrieval
- let the judge downgrade or reject the claim
- never hide material disagreement

If retrieval returns no relevant evidence:
- state corpus insufficiency
- do not interpret absence of evidence as evidence of absence

## 18. Evaluation philosophy

Evaluation must compare CCE against simpler baselines.

Required baselines:
A. keyword search + analyst prompt
B. single-agent RAG
C. single-agent RAG + self-critique
D. multi-agent without adversarial challenge
E. full CCE

Metrics:
- precision of material consequences
- recall of known seeded consequences
- hidden-consequence discovery rate
- contradiction detection precision/recall
- evidence citation correctness
- unsupported-claim rate
- severity calibration quality
- blind-spot yield
- latency
- cost per analysis
- agent disagreement usefulness

## 19. Benchmark corpus

Build a synthetic but realistic benchmark.

Each scenario has:
- change request
- source corpus
- known direct dependencies
- known hidden dependencies
- deliberate stale documentation
- deliberate contradictions
- irrelevant distractors
- historical analogue
- expected unresolved questions
- expected assumption keys and coupling sets

Three construction rules exist to stop the benchmark from flattering the pipeline:

1. **Negative controls are mandatory, not optional, and target ~25% of scenarios.**
   These are changes whose correct answer is "little or nothing downstream". A system
   that always finds twelve consequences is useless, and the benchmark must be able to
   catch that.
2. **Scenario authorship must be separated from ontology authorship.** Scenarios are
   written before the relationship taxonomy and assumption key namespace are frozen,
   ideally by someone who has not read them. Otherwise recall measures only whether the
   pipeline found what was planted for it, in its own categories.
3. **Every scenario carries a corpus-ablation variant** (see `EVALUATION.md`).

Target initial benchmark:
30–50 scenarios across:
- support operations
- payments/refunds
- account lifecycle
- access management
- data retention
- SLA/process changes
- API contract changes
- reporting changes

## 20. Example benchmark scenario

Scenario:
"Reduce password reset token validity from 24 hours to 30 minutes."

Seed:
- auth policy
- support runbook
- email template
- mobile UI text
- incident ticket
- identity provider config
- analytics definitions

Expected findings:
- support ticket volume may rise
- email wording may become misleading
- mobile UI may promise longer recovery
- identity-provider configuration could disagree
- historical incident may show similar increase

Distractors:
- unrelated password complexity policy
- archived 2019 document
- unrelated account lockout policy

## 21. Feedback loop

### 21.1 Why outcome-matching cannot be the primary signal

Deriving ground truth from production incidents is close to unimplementable, for three
structural reasons:

- **Attribution is ambiguous.** An incident six weeks later may or may not belong to this
  change.
- **Misses are unobservable.** You only learn about consequences that caused visible pain,
  so production *recall* can never be measured - only precision.
- **Prevented consequences are counterfactually invisible.** If the system warns you and
  you act, the correct prediction looks like a false positive. The system is punished for
  working.

### 21.2 Primary signal: human disposition at review time

Capture, per finding, at the moment of review:

- `useful` - actionable and not already known
- `already_known` - correct, but the reviewer already had it
- `not_material` - correct, but not worth acting on
- `wrong` - the mechanism does not hold
- `unverifiable` - cannot be determined from available information

This is cheap, immediate, unambiguous, and sufficient to drive routing, retrieval tuning,
and the confidence function. `already_known` is the key signal for the product thesis: a
high rate means the system is mostly finding what a diligent reviewer would find anyway.

A disposition of `wrong` or `not_material` **suppresses that finding in future analyses**
for the same (assumption key, artifact) pair. The suppression is recorded and expires when
the source is revised. Without persistent suppression the same false positive is
re-delivered on every analysis and reviewers stop reading.

### 21.3 Secondary signal: observed outcomes

Where an outcome can be attributed with reasonable confidence, record it as
`correctly_predicted`, `partially_predicted`, `false_positive`, or `missed`. Treat this as
a weak, low-volume signal, never as the primary metric.

### 21.4 Governance

The feedback loop must not automatically rewrite production prompts, scoring weights, or
routing policy. All such changes go through the evaluation harness and versioning.

## 22. UX concept

Primary screen:
Change Analysis

Sections:
- Change summary
- Risk overview
- consequence graph
- high-confidence consequences
- hidden consequences
- contradictions
- blind spots
- source evidence
- unresolved questions
- validation checklist

Each graph node/edge is explorable back to source evidence.

## 23. Example API

POST /api/v1/changes/analyze

Input:
{
  "title": "...",
  "description": "...",
  "effective_date": "...",
  "scope": {...},
  "source_scope": {...}
}

Response:
{
  "analysis_id": "...",
  "status": "completed",
  "summary": {...},
  "consequences": [...],
  "contradictions": [...],
  "blind_spots": [...],
  "evidence": [...],
  "graph": {...},
  "unresolved_questions": [...]
}

## 24. Security boundary

The system can analyze sensitive organizational information, so:
- source permissions must propagate to retrieval
- users must not receive evidence they are not authorized to view
- all tool access should be scoped
- audit every source access in production
- never put raw secrets in prompts or logs
- redact credentials/tokens from ingestion

## 25. Development philosophy

Start as a vertically complete research prototype, then harden.

Never build a full enterprise platform before demonstrating:
- hidden consequence discovery
- contradiction handling
- evidence grounding
- adversarial improvement
- benchmark improvement over baseline

## 26. Future extensions

Potential:
- change proposal comparison
- scenario simulation
- what-if branching
- organizational knowledge decay detection
- stale-policy detection
- automatic test-plan generation from predicted consequences
- change validation assistant
- outcome feedback and calibrated prediction
- graph-based retrieval learning
- active evidence acquisition
