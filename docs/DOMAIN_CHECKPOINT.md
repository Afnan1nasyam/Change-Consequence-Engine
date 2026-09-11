# Domain Checkpoint — Finalized Contracts

Status: **DECIDED**. Authored 2026-09-07, before Phase 1.
Authority: where this document disagrees with another specification file, **this document
wins** and the other file is to be amended. Section 15 lists the amendments outstanding.

Purpose: freeze the domain semantics *before* Pydantic models exist, and before any
production prompt is written against them. A prompt instructed to "match the runtime
schema" against an unfrozen schema produces invented fields and fabricated citations.

---

## 1. The governing invariant

> **An LLM may propose a semantic relationship. It can never promote that relationship
> into the trusted deterministic layer by itself.**

Every rule below is a consequence of this. The two promotion ladders are the only routes
into the deterministic layer, and both pass through a human:

```
Unknown key                          Possible key equivalence
    |                                     |
    v                                     v
KeyRegistryGap                       KeyAlias proposal  (status=proposed)
    |                                     |
    v                                     v
Human review                         Human approval
    |                                     |
    v                                     v
Registry update (versioned)          Approved KeyAlias
    |                                     |
    v                                     v
Re-run deterministic resolver        Deterministic coupling
    |                                     |
    v                                     v
Possible Class A derivation          Possible Class A derivation
```

Neither ladder has a bypass. There is no confidence threshold, no auto-approval, no
"high-certainty" shortcut. A model's certainty about a relationship is not evidence that
the relationship exists.

---

## 2. Epistemic ontology — five types

The five types are distinct kinds of claim, never collapsed into one generic "assumption".
Each is a separate schema variant.

| | type | plain meaning | what would make it wrong |
|---|---|---|---|
| **A** | `ObservedFact` | a source says this | the span was misread |
| **B** | `AssumptionBinding` | a source *commits* to this proposition | the interpretation misread intent, or normalization chose the wrong key |
| **C** | `DerivedRelation` | the system computed this from ≥2 records by a named rule | the rule is wrong, or its inputs are wrong |
| **D** | `Hypothesis` | a model proposes this; it needs validation | the stated mechanism does not hold |
| **E** | `EvidenceGap` | the corpus cannot establish this | the search was inadequate (not: the world is otherwise) |

Type E is a claim **about the corpus**, not about the world. This distinction is load
bearing: it is why a gap can never become an edge.

### 2.1 Axis 1 — `derivation` (how the claim came to exist)

| value | meaning | producer |
|---|---|---|
| `observed` | explicitly stated in a source | extractor, mechanically verified |
| `extracted` | mechanically pulled from a source, incl. assumption binding | extractor + verifier |
| `derived` | computed by a **named deterministic rule** over records | resolver, no model |
| `inferred` | proposed by a model, mechanism articulated | LLM agent |
| `absent` | established by the *failure* of a search | retrieval, recorded |

`derived` and `inferred` are separate values. Collapsing them — as the current
`DATA_MODEL.md` does — makes one value mean two epistemically different things and
destroys the auditability of Class A.

### 2.2 Axis 2 — `support` (what backs the claim)

| value | meaning |
|---|---|
| `verified_span` | ≥1 evidence span, mechanically confirmed to contain the claim |
| `multi_span` | ≥2 spans from **distinct source versions** |
| `rule_over_records` | a rule id plus the record ids it consumed; **no evidence of its own** |
| `mechanism_only` | an articulated causal chain, evidence optional and possibly empty |
| `negative_provenance` | the executed QueryPlan that returned nothing |

### 2.3 The definition that matters

The two axes exist so that these three are never confused:

- **Not directly stated, but computed** — `derived` + `rule_over_records`. Legitimate,
  auditable, the highest-trust output in the system.
- **Not directly stated, model-proposed** — `inferred` + `mechanism_only`. Legitimate,
  useful, and structurally incapable of being labelled established.
- **Hallucination** — a claim whose **declared support fails mechanical verification**: it
  cites a span that does not contain it, cites a non-existent evidence id, or declares
  `verified_span` with no span.

> Hallucination is a **failed support claim**, not a missing one.

A single-axis model cannot express this, which is why the current `inference_type` field
lets "inferred" read as either "carefully derived" or "made up".

### 2.4 Evidence rules per type

| type | `evidence_ids` | additional mandatory provenance | may become a graph edge? |
|---|---|---|---|
| **A** `ObservedFact` | **≥1, required** | span offsets; containment verified | yes — node attributes, `directly_changes` |
| **B** `AssumptionBinding` | **exactly 1 primary, required** | span offsets, registry key + version, `extractor_version` | yes — `binds_assumption`, **as a projection only** |
| **C** `DerivedRelation` | **must be empty** | `rule_id`, `input_record_ids` (non-empty), `coupling_index_version` | yes — `invalidates_assumption`, `conflicts_with` |
| **D** `Hypothesis` | **may be empty** | mechanism text, `prompt_sha256`, `provider`, `model`; analogue id if above `speculative` | yes — `creates_risk_for` only, structurally marked |
| **E** `EvidenceGap` | **must be empty** | executed strategies, source types searched, `as_of`, what would resolve it | **never** |

Three rules deserve emphasis:

**C carries no evidence of its own.** Its evidence set is the *computed transitive union*
of its inputs' evidence, materialized on read. Demanding an `evidence_id` on a derived
relation would force the resolver to nominate an arbitrary input span and re-present it as
direct support — manufacturing false provenance, which is the failure this project exists
to prevent.

**D may legitimately have zero evidence.** A hypothesis with no evidence is not a defect;
it is a hypothesis. What is forbidden is a hypothesis *labelled as established* — see
§9.2.

**E must never be an edge.** Inventing a relationship out of silence is the structural
form of "absence of evidence as evidence of absence". A gap attaches to the analysis and
may *annotate* a node (for example "this key has a single holder"); it is never a
relationship between nodes.

---

## 3. The Binding — formal definition

A binding is admitted only as a fully verified 7-tuple with a validity interval:

```
Binding = (
    key,                       # resolved in AssumptionKeyRegistry @ registry_version
    scope,                     # ScopeSelector, lattice path, required
    value,                     # typed payload
    role,                      # declares | consumes | describes
    explicitness,              # explicit | derived_from_text | implied
    holder_source_version_id,  # the artifact that binds it
    evidence_id                # the span it was extracted from
)
valid over [effective_from, effective_to)
```

### 3.1 What a binding is bound to

A **registry-controlled `key` at a `scope`** — never to an entity. An entity join requires
cross-references between artifacts; a key join does not. That is the entire mechanism.

### 3.2 What may participate

**Source versions** participate. **Assumption keys** are the join medium. Nothing else.
Owners, teams, and services are node metadata, not coupling participants.

### 3.3 `role` — dependency direction

| role | meaning | example |
|---|---|---|
| `declares` | the artifact is the authority stating the value | a policy setting a threshold |
| `consumes` | the artifact depends on the value being what it says | a training deck teaching it |
| `describes` | the artifact mentions the value without depending on it | a retrospective narrating history |

`role` is separate from `explicitness` because they are orthogonal, and because
**collision semantics depend entirely on `role`**:

| composition | meaning | finding |
|---|---|---|
| `declares` vs `declares`, divergent | two authorities disagree | **contradiction** (`conflicts_with`) |
| `declares` vs `consumes`, divergent | the consumer is stale | **staleness** (`invalidates_assumption`) |
| `consumes` vs `consumes`, divergent | possibly different scopes, possibly both stale | **not a contradiction**; report as a coupling anomaly requiring scope review |
| any vs `describes` | historical mention | **no finding** |

The current `binding_kind: asserts | depends_on | implies` conflates these axes and
therefore leaves collision semantics undefined. It is replaced by `role` +
`explicitness`.

### 3.4 `explicitness`

| value | meaning | corroboration weight |
|---|---|---|
| `explicit` | the value appears verbatim in the span | highest |
| `derived_from_text` | unambiguously implied by the span's wording | middle |
| `implied` | required for the span to make sense, but unstated | lowest |

Ordering is `explicit > derived_from_text > implied` and feeds the confidence function.
The extractor emits this **enum**, never a number (§9.1).

### 3.5 What establishes a binding

Extraction *proposes*; verification *admits*. All three checks are mechanical, run after
extraction, and all must pass:

1. **Span verification** — the cited span exists and contains the value verbatim.
2. **Typed parse** — the value parses under its declared `value_type`, with `unit` present
   where required.
3. **Registry resolution** — `key` resolves in the `AssumptionKeyRegistry` at the pinned
   registry version.

A binding failing any check is **quarantined, never indexed**. Failure mode 3 produces a
`KeyRegistryGap` (§4).

### 3.5a `authority_class` — resolved, not stored

Authority is a property of the **holder document**, not of the binding, so it is not part
of the 7-tuple. It is resolved from `SourceDocument.authority_class` at comparison time and
recorded in the derivation of any finding that depended on it.

Ordering is a total order declared in configuration, for example:

```
regulatory > policy > standard > sop_runbook > config > training > wiki > ticket > chat
```

The ordering is versioned as `authority_model_version` and pinned onto every derivation
that used it, because changing the ordering changes findings.

### 3.6 `value_type`

`scalar` · `interval` · `duration` · `enum` · `boolean` · `reference`

`unit` is **required** for `scalar` and `duration`. Comparison is unit-aware; a unit
mismatch **blocks** coupling rather than weakening it (§5.6).

### 3.7 Negation and polarity

"Refunds above 10,000 require approval" and "refunds below 10,000 need no approval"
describe the same fact in opposite polarity. The normalizer must render both into one
canonical constraint form (`key`, `comparator`, `value`) or coupling silently fails.

MVP: the registry entry declares the canonical constraint form for each key, and the
extractor must emit that form. Polarity handling is a **registry-entry obligation**, not
inference.

---

## 4. Decision 1 — closed AssumptionKeyRegistry

**Decided: the registry is closed for the MVP.** Precision and trust of Class A take
priority over coverage.

### 4.1 Rules

1. The registry is **curated and versioned**. Every entry is authored deliberately.
2. **Only registered keys may participate in Class A coupling.**
3. Extraction **may propose** an unknown key. An unknown key is **quarantined as a
   `KeyRegistryGap`**.
4. Unknown keys **never silently enter the coupling index**. There is no fallback path.
5. **No clustering and no automatic key normalization in the MVP.** Normalization is
   registry-declared canonical form (§3.7), not learned.
6. The registry is **seeded with a small explicit vocabulary** covering the benchmark
   domains (`IDEA.md` §19): support operations, payments/refunds, account lifecycle,
   access management, data retention, SLA/process, API contracts, reporting.
7. **`KeyRegistryGap` volume is an evaluation and product signal**, reported per domain,
   telling us where the registry needs expansion. It is a first-class metric, not an error
   log.
8. Registry changes are **versioned and auditable**: who, when, why, and the resulting
   version.
9. A registry update must make the affected deterministic derivations **reproducible on the
   next analysis run** — see §4.3.

### 4.2 `AssumptionKeyRegistry` entry

```
key                     canonical dotted lowercase path
value_type              scalar | interval | duration | enum | boolean | reference
unit                    required when value_type in {scalar, duration}
canonical_form          the constraint shape extractors must emit (§3.7)
allowed_scopes          scope-lattice dimensions valid for this key
description             what the variable means, in one sentence
domain                  benchmark/product domain
introduced_in_version   registry version
deprecated_in_version   nullable
```

### 4.3 Registry versioning and reproducibility

An `AssumptionRecord` pins the `registry_version` under which it was admitted. A
`DerivedRelation` pins both `registry_version` and `coupling_index_version`.

Consequently:

- A registry update **does not retroactively mutate** stored derivations.
- Re-running the resolver at a newer registry version may admit previously quarantined
  bindings and therefore **produce new Class A derivations**. This is the intended growth
  path.
- Any past analysis remains explainable, because its registry version is recorded and its
  inputs are persisted.
- Two runs differing only in registry version must be **diffable** — that difference is
  precisely the value the registry expansion delivered.

---

## 5. Coupling — formal definition

### 5.1 CouplingSet identity

A CouplingSet is keyed on the **triple**:

```
(key, scope, as_of)
```

not on `key` alone. Membership is every admitted binding of that key whose scope is
compatible (§5.2) and whose validity interval contains `as_of` (§5.3).

Keying on `key` alone is the defect behind the current `has_collision := distinct_values >
1` rule, which would report a contradiction for every legitimately scoped override.

### 5.2 The scope lattice

`ScopeSelector` is a path over declared dimensions, partially ordered by containment:

```
global
  region:EU
    region:EU / tier:enterprise
  region:US
  tier:retail
```

- **Comparable**: two scopes are equal, or one **contains** the other.
- **Intersecting**: the scopes share at least one point (comparability implies
  intersection; siblings do not intersect).
- **Disjoint**: siblings such as `region:EU` and `region:US`.

Dimensions permitted for a key are declared by its registry entry (`allowed_scopes`), so a
scope is validated, not invented.

**Scope containment alone does not confer the right to override.** A narrower scope is
where an override *may* live; whether it is legitimate is decided by authority (§5.4.2).
Treating containment as sufficient would silently bless an unauthorized deviation, which is
one of the failure modes the product exists to surface.

### 5.3 Temporal validity

Every binding carries `[effective_from, effective_to)`, `effective_to` nullable meaning
open-ended. All coupling is evaluated **as of a date** — normally the ChangeSpec's
effective date.

Temporal authority resolution is a single deterministic domain service:

```
resolve_authority(key, scope, as_of) -> SourceVersion | Ambiguous
```

`Ambiguous` is a **real return value**, not an error to be smoothed away. Overlapping or
missing intervals among `declares` bindings at comparable scope are a **finding about the
corpus**, reported as such. Newest-wins is forbidden: it silently invents an authority the
corpus does not establish.

### 5.4 Divergence between declarers — three deterministic outcomes

Two `declares` bindings of the same key whose values differ are **not one finding type**.
Scope and authority decide which of three outcomes applies. All three are deterministic,
Class A, and require no model.

**Common preconditions** for any of the three:

1. same registry `key` (or an **approved** `KeyAlias`, §6),
2. **overlapping** validity intervals,
3. both `role = declares`,
4. values **unequal** under unit-aware typed comparison,
5. compatible units (mismatch blocks coupling entirely, §5.6).

#### 5.4.1 Same scope ⇒ `conflicts_with`

Scopes are **equal**. Two authorities declare different values for the same variable at the
same scope with overlapping validity.

Result: **`conflicts_with`**, symmetric, `derivation = derived`.

#### 5.4.2 Narrower scope with sufficient authority ⇒ legitimate override

One scope strictly **contains** the other, and:

```
narrower.holder.authority_class >= broader.holder.authority_class
```

This is a deliberate, authorized specialization. Result: the narrower binding **overrides**
the broader one for its scope. **No contradiction is emitted.** The override is recorded so
it can be reasoned about later (§5.5), and the narrower binding becomes the authoritative
value inside its scope for `resolve_authority`.

#### 5.4.3 Narrower scope with insufficient authority ⇒ `unauthorized_override`

One scope strictly contains the other, and:

```
narrower.holder.authority_class < broader.holder.authority_class
```

A lower-authority artifact is narrowing a higher-authority declaration. This is a
**governance and consistency defect**, not a legitimate specialization, and not a
contradiction between peers.

Result: **`unauthorized_override`**, a **directed** edge from the narrower binding's holder
to the broader binding's holder, `derivation = derived`, Class A.

Required behaviour:

- **both bindings and both scopes are preserved** — neither is discarded, suppressed, or
  rewritten;
- the **authority comparison is recorded in the derivation**: both `authority_class`
  values, the ordering used, and `authority_model_version`;
- **no LLM discovers, classifies, or rates this.** It is a governance rule evaluated by
  deterministic code. An agent may articulate why it matters; it may not decide that it
  happened.
- `resolve_authority` continues to return the **broader, higher-authority** value as
  authoritative, and flags the unauthorized narrowing.

#### 5.4.4 Summary

| key | scope | time | authority | outcome | edge |
|---|---|---|---|---|---|
| same | equal | overlapping | any | contradiction | `conflicts_with` (symmetric) |
| same | narrower ⊂ broader | overlapping | narrower ≥ broader | legitimate override | none |
| same | narrower ⊂ broader | overlapping | narrower < broader | governance defect | `unauthorized_override` (directed) |
| same | disjoint | any | any | not coupled for contradiction | none |
| same | any | non-overlapping | any | superseded, not conflicting | none |

### 5.5 Coupling-for-invalidation

A binding is coupled for invalidation to an `AssumptionDelta` iff **all** hold:

1. same registry `key` (or an approved `KeyAlias`),
2. scope **intersects** the delta's scope — inheritance counts, so a `global` change
   reaches narrower inheriting scopes,
3. the binding is valid at the delta's effective date,
4. `role ∈ {declares, consumes}` — `describes` never invalidates,
5. the binding's value is inconsistent with the delta's `new_value`.

Result: `invalidates_assumption`, `derivation = derived`, deterministic.

Where a narrower scope holds a **legitimate** override (§5.4.2), the finding is
**`scope_inheritance_affected`**: the override may still be intended, and requires
confirmation rather than being reported as broken. **The override itself is never labelled
contradictory** by an invalidation pass.

Where a narrower scope holds an **unauthorized** override (§5.4.3), the change produces two
independent findings that must not be merged: the pre-existing `unauthorized_override`
(which the change did not cause) and the invalidation or scope impact on the broader
binding. Conflating them would attribute a pre-existing governance defect to the proposed
change.

**The two relations are distinct and must not be merged.** Contradiction needs comparable
scope; invalidation needs only intersection.

### 5.6 Explicit vs inferred coupling

| | explicit coupling | inferred coupling |
|---|---|---|
| basis | both bindings resolve to the **same registry key** | two **different** keys are believed to name the same variable |
| producer | deterministic index lookup | model proposal |
| Class | **A** | **B** until an alias is approved |
| gate | none needed | **`KeyAlias`, human-approved** (§6) |

Inferred coupling is the single point at which invented dependencies could enter the
deterministic layer. It is therefore gated, not scored.

**Unit mismatch blocks coupling outright.** `refund.approval_threshold` (USD) and
`refund.fraud_review_threshold` (score 0–1) are never coupled regardless of lexical
similarity. This is a hard deterministic guard against normalizer over-merging.

### 5.7 Coupling strength

Deterministic, versioned, computed — never a judgment:

| input | direction |
|---|---|
| role composition (`declares`↔`consumes` strongest, `describes` weakest) | + |
| `explicitness` of both bindings | + |
| scope distance in the lattice (0 = identical) | − |
| temporal overlap fraction | + |
| holder `authority_class` | + |
| span verification result | gate |
| unit compatibility | **gate: mismatch ⇒ strength 0, coupling blocked** |
| alias-mediated rather than direct key match | − |

### 5.8 Coupling versus graph dependency

| | dependency | coupling |
|---|---|---|
| origin | asserted by a document author | computed by the system |
| requires a reference to exist | **yes** | **no — this is the point** |
| direction | inherently directed | symmetric at the key, asymmetric by `role` |
| discovery | graph traversal | index lookup on `(key, scope, as_of)` |
| absence from documentation | means no dependency exists | **is itself the finding** |

### 5.9 Worked examples

**E1 — stale training material (canonical Class A).**
Policy v3 `declares refund.approval_threshold = 10000 USD @global`. Training deck v7
`consumes` the same key and value. Delta sets `5000`. Same key, comparable scope,
overlapping intervals, `declares`→`consumes`: the deck is invalidated. Found with **zero**
model calls; the model is used only to articulate why it matters.

**E2 — cross-medium coupling, no cross-reference.**
IdP configuration `declares auth.reset_token.ttl = 24h`. The password-reset email template
`consumes` it, stating "valid for 24 hours". A YAML file and an HTML template, neither
referencing the other, coupled through one key. Deterministic once both are extracted.

**E3 — scope inheritance, not contradiction.**
`retention.window`: `global = 90d`, `region:EU = 30d`. Comparable scope, divergent values —
but the narrower legitimately overrides, so **no contradiction**. Change `global` to
`180d`: the analytics definition `consuming` `global` is **invalidated**; the EU binding is
reported `scope_inheritance_affected` for confirmation.

**E4 — unauthorized override (governance defect, deterministic).**
The data-retention policy (`authority_class = policy`) `declares retention.window = 90d
@global`. A team wiki page (`authority_class = wiki`) `declares retention.window = 30d
@region:EU`. Scopes are comparable and the wiki's is narrower, but
`wiki < policy`, so this is **not** a legitimate override: it emits
`unauthorized_override`, directed wiki → policy, with both authority classes and the
ordering version recorded. Both bindings are preserved. `resolve_authority` still returns
90d as authoritative for EU and flags the narrowing. No model was involved in reaching any
part of that conclusion.

Had the EU binding come from a regional compliance policy at `authority_class = regulatory`,
the same shape would be a **legitimate override** and emit nothing.

### 5.10 Counterexamples

**C1 — lexical neighbours, different variables.**
`refund.approval_threshold` (USD) versus `refund.fraud_review_threshold` (score 0–1).
Adjacent vocabulary, distinct registry keys, incompatible units. Coupling **blocked
deterministically** by §5.6.

**C2 — the undocumented load effect.**
"Manager approval volume will rise." No artifact `declares` a value for approval volume,
so there is no binding and no coupling: **Class B**.

C2 is the boundary rule made concrete: **if the reconciliation runbook says "expect ≤20
approvals/day", that is a binding and the same consequence becomes Class A.** Identical
subject matter; the class is decided purely by whether someone wrote the number down.

---

## 6. Decision 2 — human-approved KeyAlias

**Decided: aliases are human-approved.** No automatic threshold-based approval.

### 6.1 Rules

1. `KeyAlias` is a **first-class persisted object**.
2. The system **may propose** aliases.
3. A **proposed** alias is unapproved and **must not create Class A coupling**.
4. An unapproved alias **may appear in Class B analysis as a proposed relationship**,
   clearly marked as such.
5. **Only an explicitly human-approved alias may participate in deterministic Class A
   coupling.**
6. Approval is **auditable**.
7. **No automatic threshold-based approval.** A model's confidence in an equivalence is not
   evidence of the equivalence.
8. No workflow UI is required for the MVP: a persisted approval state plus an admin API
   endpoint is sufficient.
9. Alias proposals **feed the knowledge-decay / registry-gap review surface** planned for
   later phases (`IDEA.md` §15.5).

### 6.2 `KeyAlias`

```
alias_id
source_key              the key as encountered
target_key              the registered key it is believed to mean
proposal_evidence       evidence_ids supporting the equivalence claim
proposed_by             agent name + prompt_sha256 + provider + model, or a human id
proposed_at
approved_by             human id; null while unapproved
approved_at             null while unapproved
registry_version        the version the alias was approved against
status                  proposed | approved | rejected | superseded
rejection_reason        nullable
```

### 6.3 Effect of status

| status | Class A coupling | Class B analysis | in the graph |
|---|---|---|---|
| `proposed` | **no** | yes, marked as a proposed relationship | only as a `Hypothesis`-backed edge |
| `approved` | **yes** | n/a | yes, as a derived edge |
| `rejected` | no | **no** — a rejected alias is suppressed | no |
| `superseded` | no | no | no |

An alias approval, like a registry update, changes future runs and never retroactively
mutates stored derivations. `registry_version` on the alias makes the promotion point
auditable.

---

## 7. Class A and Class B

### 7.1 Definitions

**Class A — `class_a_assumption_mediated`.** A consequence for which there exists a
derivation chain from an `AssumptionDelta` to the affected artifact composed **only** of:
registry-resolved verified bindings, scope-compatible coupling (direct key match or
approved alias), and **named deterministic rules**. Discovered without a model. Evidence
backed by construction.

**Class B — `class_b_behavioral`.** A consequence requiring semantic or agentic hypothesis
generation across evidence. Causal, load-driven, or quantitative effects that no binding
expresses. No deterministic derivation exists.

### 7.2 The decision rule

> **A consequence is Class A iff the deterministic resolver produced it.**

Classification is **by producer, not by subject matter**. This is the whole defence of the
requirement that the agent layer must not rediscover Class A consequences merely because
they are easy to verbalize: *sounding derivable is not a derivation*.

Corollary: `consequence_class` is assigned **by the pipeline** and appears on **no
agent-authored schema**. No agent can self-declare Class A.

### 7.3 Examples

**Clear Class A** — training material teaching an obsolete threshold; IdP configuration
disagreeing with authentication policy; an analytics definition assuming a superseded
retention window; an SOP step referencing a removed approval tier; two policies declaring
divergent values for the same key at comparable scope.

**Clear Class B** — escalation volume rises; reconciliation timing shifts; customer
confusion increases; fraud baselines drift; support queue capacity strains.

### 7.4 Ambiguous cases

| archetype | disposition |
|---|---|
| **(i) derivable in principle, key not registered** | **Class B now**, plus a `KeyRegistryGap`. The intended growth path: registry expands, the consequence becomes Class A on re-run. |
| **(ii) partly derivable** — A proves the artifact is stale; how much that *matters* is judgment | **Split into two linked findings**: a Class A staleness finding (deterministic) and a Class B impact finding (hypothesis). **Never one blended finding** — blending launders a judgment into the trust-bearing tier. |
| **(iii) requires an unapproved `KeyAlias`** | **Class B**, carrying the alias proposal for review. Approval promotes on re-run. |

**Default rule: any ambiguity demotes to Class B.**

The asymmetry is deliberate. Misclassifying downward costs recall in the trusted tier.
Misclassifying upward destroys what the tier *means*, and Class A's entire value is that a
reviewer can trust it without re-deriving it.

### 7.5 Class B deduplication against Class A

Three structural mechanisms. Prompt wording will not hold this line.

1. **`consequence_class` is pipeline-assigned**, never agent-supplied (§7.2).
2. **Output-side merge.** Any Class B hypothesis whose `(assumption_key,
   affected_artifact)` pair already exists in the Class A set is **merged, not appended**,
   and re-attributed to the resolver. The model's contribution is demoted to *mechanism
   articulation* — which is what it is reliable at.
3. **A visible metric.** `class_b_rediscovery_rate` = share of Class B hypotheses
   suppressed as Class A duplicates. A high value means the hunter's prompt or evidence
   slice is misconfigured, **not** that the hunter is being productive.

Merging rather than dropping matters: the articulated mechanism is worth keeping even when
the discovery was not the model's.

---

## 8. ChallengeResult rules

Exactly three outcomes. There is no fourth, and an uncited objection is
**unrepresentable** in the schema rather than discouraged by prompt text.

| outcome | `evidence_ids` | also required |
|---|---|---|
| `refuted` | **≥1, span-verified, must actually contradict the claim** | what specifically it contradicts |
| `weakened` | **≥1, span-verified** (compensating control, exception, or newer source version) | how the claim's scope narrows |
| `no_refutation_found` | **must be empty** | **`falsification_test` mandatory** |

**`no_refutation_found` costs a falsification test.** This stops it being a lazy default
while equally refusing to punish it: on a well-evidenced consequence it is the correct
answer, and the test it yields feeds the validation checklist as a real deliverable.

### 8.1 Class A refutation admissibility

A challenger **cannot refute a Class A consequence on mechanism grounds.** A deterministic
derivation is only ever rule-wrong or input-wrong, so the admissible grounds are exactly:

- **disputed span** — the cited span does not say what the binding claims;
- **disputed key or scope** — the binding was filed under the wrong key or scope;
- **disputed temporal validity** — the binding was not in force at the effective date;
- **superseding version** — a newer source version changes the binding.

"I do not think that follows" is **inadmissible** against a derived relation, and is
rejected by the domain service, not by the prompt. Without this guard a model can talk
down a computed result, which would make the deterministic layer only as strong as the
most agreeable model in the pipeline.

---

## 9. Confidence, severity, corroboration

### 9.1 Confidence — computed only

**No LLM-authored schema contains a `confidence` field.** Agents emit *evidence positions*
(enums they can defend); confidence attaches downstream on a computed `ConsequenceScore`.

Removed from: `SpecialistFinding`, `ConsequenceHypothesis`, `AssumptionDelta`,
`ArbitrationDecision`, `ChallengeResult`, `GraphEdge`.

Inputs to the confidence function, per `IDEA.md` §13:

| factor | source | direction |
|---|---|---|
| source authority | holder `authority_class` | + |
| source freshness | `effective_from`, version recency | + |
| independent corroboration | distinct source versions (§9.3) | + |
| explicitness | binding `explicitness` ordering (§3.4) | + |
| contradiction | presence and strength of a divergent declaring binding | − |
| inference depth | derivation hops from `directly_changes` | − |
| retrieval reliability | score agreement across strategies | + |
| alias mediation | approved alias rather than direct key match | − |

`confidence_fn_version` is recorded, inputs are persisted, and any score is recomputable
and explainable after the fact.

**`extraction_confidence` is abolished.** It was an LLM-supplied float wearing a carve-out.
Replaced by:

- `explicitness` — an enum the extractor can justify (§3.4);
- `extraction_quality` — **computed** from `span_verified`, `value_parsed`,
  `key_resolved`.

The model emits categorical, checkable claims. Code emits numbers.

### 9.2 Status gating

`status ∈ {supported, likely, disputed, speculative, rejected, unresolved}`.

- `evidence_ids` empty ⇒ `status ∈ {speculative, unresolved}` **only**. Never `supported`
  or `likely`.
- Class B above `speculative` requires a non-null `historical_analogue_id` referencing an
  analogue with a **recorded observed outcome**. An analogue without an outcome does not
  qualify.
- `no_refutation_found` from the challenger is **not corroboration**. It means nobody found
  contrary evidence, which is strictly weaker than evidence of correctness, and must not
  raise status.

### 9.3 Corroboration — enforced structurally

Corroboration counts **distinct `(source_version_id, source_type)` pairs reached by
distinct retrieval strategies** within a coupling set.

It never counts agents. Several agents reading the same `EvidenceBundle` are **one**
source.

Enforcement is structural rather than policy: **the confidence scorer's input type does not
contain agent identity.** It cannot count agents because it cannot see them.

Also not corroboration: multiple chunks of one source version; the same claim reached twice
by one strategy.

### 9.4 Severity — the deliberate asymmetry

Severity **is** assigned by the judge. This asymmetry with confidence is intentional and is
recorded here so it is not "fixed" in the wrong direction later:

- **confidence** is a function of *evidence properties the system can measure* ⇒ computed.
- **severity** is a function of *business impact the corpus does not contain* ⇒ judgment.

Severity is therefore: constrained to `LOW | MEDIUM | HIGH | CRITICAL`, attributed to
prompt/provider/model, reviewer-overridable, and **never blended with confidence**. HIGH
severity with LOW confidence is a legitimate and important output.

---

## 10. Graph projection rules

1. **The graph is a pure projection.** It is rebuilt deterministically from the assumption
   store, derived relations, and arbitrated consequences. A writeable graph is a divergence
   bug.
2. **No agent writes an edge.** Agents produce findings; the projector produces edges.
3. **Edges carry no scores.** `confidence` and `severity` live on the consequence; an edge
   references `consequence_id`. Duplicated scores diverge.
4. **Node types**: `SourceDocument`, `AssumptionKey`, `Entity`, `Change`,
   `HistoricalEvent`. `AssumptionKey` is a node **because that is what makes hidden
   coupling visible** — two documents joined through a key neither mentions is the central
   visualization.
5. **Edge types**, with admissible derivation:

| edge | derivation | direction | source type |
|---|---|---|---|
| `directly_changes` | `observed` | directed | A |
| `binds_assumption` | `extracted` | directed | B, projection of the record |
| `invalidates_assumption` | `derived` | directed | C |
| `conflicts_with` | `derived` | **symmetric** | C |
| `unauthorized_override` | `derived` | **directed** (narrower → broader) | C |
| `creates_risk_for` | `inferred` | directed | D |
| `historically_analogous_to` | `inferred` | directed | D — the *incident* is observed, the *analogy* is a judgment |

The taxonomy is **seven** edges. It grew from six by the documented extension path: written
semantics (§5.4.3) and an admissible derivation, decided before any agent could emit a
finding that projects to it.

6. **Hypothesized edges are structurally distinct** in the payload so a client cannot
   render a `creates_risk_for` identically to an `invalidates_assumption`.
7. **Type E never projects to an edge** (§2.4).
8. Adding an edge type requires written semantics and an admissible derivation **before**
   any agent may emit a finding that would project to it.

---

## 11. Entity map

```
SourceDocument ──< SourceVersion ──< Chunk ──< EvidenceItem
                        │                          │
                        │                          ▼
                        └─────────────< AssumptionRecord            (type B)
                                              │ key, scope, value, role,
                                              │ explicitness, interval,
                                              │ registry_version
             AssumptionKeyRegistry ◄───────────┤  resolution REQUIRED
                  KeyRegistryGap ◄─────────────┤  on resolution failure
                        KeyAlias ◄─────────────┘  approved only (§6)
                                              │
                                              ▼
                                   CouplingSet (key, scope, as_of)  derived, never authored
                                              │
              ChangeRequest ─> ChangeSpec ─> AssumptionDelta
                                              │
                                              ▼
                                   DerivedRelation                  (type C)
                                              │ rule_id + input_record_ids
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
              Consequence(class_a)                    ConsequenceHypothesis(class_b, type D)
                       │                                             │
                       └──────────────┬──────────────────────────────┘
                                      ▼
                              ChallengeResult ──> FalsificationTest
                                      ▼
                              ArbitrationDecision      status + severity, NO confidence
                                      ▼
                              ConsequenceScore         computed; confidence_fn_version
                                      ▼
                              ConsequenceGraph         pure projection
                                      ▼
                              FindingDisposition       human signal; drives suppression

ObservedFact   (type A) ──> node attributes, directly_changes
EvidenceGap    (type E) ──> attaches to Analysis; may annotate a node; NEVER an edge
SpecialistFinding       ──> evidence position enum, no confidence
```

**New entities Phase 1 must carry:** `AssumptionKeyRegistry`, `KeyRegistryGap`,
`KeyAlias`, `ScopeSelector`, `DerivedRelation`, `EvidenceGap`, `ObservedFact`,
`ConsequenceScore`, `FalsificationTest`.

**Changed:** `AssumptionRecord` gains `scope`, `registry_version`, `span_verified`,
`value_parsed`, `key_resolved`, `extraction_quality`; `binding_kind` splits into `role` +
`explicitness`; `extraction_confidence` removed.

---

## 12. Invariants enforced by Pydantic

Intra-object, no I/O.

**Claim shape**
1. Claim types A–E are a **discriminated union**; each variant carries only its legal
   fields.
2. `evidence_ids` non-empty for `ObservedFact`, `AssumptionBinding`, `refuted`, `weakened`.
3. `evidence_ids` **empty** for `DerivedRelation`, `EvidenceGap`, `no_refutation_found`.
4. `DerivedRelation` requires `rule_id` and non-empty `input_record_ids`.
5. `EvidenceGap` requires non-empty negative provenance (strategies, source types,
   `as_of`).
6. `no_refutation_found` requires `falsification_test`.
7. `ChallengeResult.outcome` is one of exactly three values; per-outcome evidence
   requirements enforced by a model validator.

**Confidence and status**
8. **No `confidence` field exists on any agent-authored model.** Enforced by absence plus a
   contract test asserting it (`tests/contract/`).
9. `status ∉ {supported, likely}` when `evidence_ids` is empty.
10. `class_b` above `speculative` requires non-null `historical_analogue_id`.
11. `severity ∈ {LOW, MEDIUM, HIGH, CRITICAL}`; no numeric severity.
12. `consequence_class` absent from every agent-authored model.

**Assumption records**
13. `AssumptionRecord.evidence_id` required and non-empty.
14. `unit` required when `value_type ∈ {scalar, duration}`.
15. `effective_from < effective_to` when both present.
16. `key` matches the registry **shape** (dotted lowercase path); *existence* is a service
    check.
17. `scope` is a well-formed lattice path over declared dimensions.
18. `role` and `explicitness` both required; no combined `binding_kind` field exists.
19. `registry_version` required.

**Graph**
20. `relationship ∈` the seven MVP edge types (§10.5).
21. `derivation ∈` the five values; per-relationship admissibility enforced by a model
    validator (§10.5).
22. `GraphEdge` has **no** `confidence` and **no** `severity`; `consequence_id` required
    for consequence-bearing edges.

**Aliases and registry**
23. `KeyAlias.status = approved` requires `approved_by`, `approved_at`, and
    `registry_version`.
24. `KeyAlias.source_key != target_key`.
25. `KeyRegistryGap` requires the proposed key, the proposing evidence, and `as_of`.

---

## 13. Invariants enforced by deterministic domain services

Cross-object, requires the store.

**Admission**
1. **Span verification** — the cited evidence actually contains the claimed value. The
   anti-hallucination primitive; everything else rests on it.
2. **Registry resolution** — key exists at the pinned registry version; failure ⇒
   `KeyRegistryGap`, never silent admission.
3. **Typed parse and unit validation.**
4. **Quarantine** — a binding failing any admission check is never indexed.

**Coupling**
5. **Unit-aware typed value comparison**; mismatch **blocks** coupling.
6. **Scope lattice** — equality for contradiction, containment for override, intersection
   for invalidation.
6a. **Override legitimacy test** — `narrower.authority_class >= broader.authority_class`
   decides legitimate override versus `unauthorized_override` (§5.4.2, §5.4.3). Both
   authority classes, the ordering, and `authority_model_version` are recorded in the
   derivation. Never delegated to an agent.
7. **Temporal authority** — `resolve_authority(key, scope, as_of) -> SourceVersion |
   Ambiguous`; newest-wins forbidden.
8. **CouplingSet construction** on `(key, scope, as_of)`.
9. **Role-and-scope-aware collision detection** per the §3.3 matrix.
10. **Alias gate** — only `status = approved` aliases participate in Class A coupling.
11. **Coupling strength** — versioned, inputs persisted.

**Derivation and classification**
12. **Invalidation propagation**, recording `rule_id` and inputs on every
    `DerivedRelation`.
13. **Class A/B classification by producer**; agents cannot influence it.
14. **Class B suppression/merge against the Class A set** on `(assumption_key,
    affected_artifact)`.
15. **Class A refutation admissibility** — reject mechanism-only challenges to derived
    relations (§8.1).
16. **Transitive evidence materialization** for `DerivedRelation` on read.

**Scoring**
17. **Confidence computation** — versioned; scorer input type excludes agent identity.
18. **Corroboration counting** over distinct source versions and strategies.

**Projection and provenance**
19. **Graph projection** — rebuilt from the store; the graph is never authored.
20. **Provenance completeness** — every persisted finding carries `prompt_version`,
    `schema_version`, `provider`, `model`, `tier`, `confidence_fn_version`,
    `coupling_index_version`, `registry_version`, `retrieval_config_version`,
    `permission_scope_hash`.
21. **Reproducibility diff** — two runs differing only in registry or alias state must be
    diffable, and that diff is the value the promotion delivered.

---

## 14. Explicit non-goals for the MVP

- No key clustering, no learned normalization, no embedding-based key merging.
- No automatic or threshold-based alias approval.
- No `likelihood` field. It is a probability by another name and `IDEA.md` §13 forbids
  claiming probabilities before calibration.
- No `potentially_impacts` edge, and no catch-all edge of any kind.
- No writeable graph.
- No confidence supplied by any model, anywhere, for any reason.
- No agent discovery or classification of `unauthorized_override`. It is a governance rule,
  not a judgment.

---

## 15. Amendments outstanding in other documents

This document is authoritative; the following must be brought into line before Phase 1
freezes. See the consistency report accompanying this checkpoint.

| file | amendment |
|---|---|
| `docs/DATA_MODEL.md` | `AssumptionRecord`: add `scope`, `registry_version`, verification flags; split `binding_kind`; drop `extraction_confidence`. Add the nine new entities. Remove `confidence` from `SpecialistFinding`, `ConsequenceHypothesis`, `AssumptionDelta`, `ArbitrationDecision`, `GraphEdge`. Type `ChallengeResult.position` as the three outcomes. Add `derived` to inference types. Rename `ChangeSpec.assumptions`. |
| `docs/IDEA.md` | §10.2 replace `binding_kind` and add scope; §11 drop `likelihood`; §12 correct `historically_analogous_to` to `inferred` and note `binds_assumption` is a projection; §13 add alias-mediation factor |
| `docs/ARCHITECTURE.md` | §3.1 add registry resolution and quarantine to the ingestion pipeline; §3.2 add the alias gate; §7.1 key the coupling index on `(key, scope, as_of)` |
| `CLAUDE.md` | add the §1 governing invariant as a numbered principle |
| `.claude/agents/consequence-judge.md` | note that severity is judge-assigned by design while confidence is computed (§9.4) |
| `.claude/agents/adversarial-challenger.md` | add the Class A refutation admissibility constraint (§8.1) |
| `.claude/agents/contradiction-hunter.md` | add the role/scope matrix; two `consumes` bindings disagreeing is not a contradiction |
