# CCE Data Model

Authority: **`docs/DOMAIN_CHECKPOINT.md` is the authoritative domain contract.** This file
is the entity reference derived from it. Where the two disagree, the checkpoint wins and
this file is wrong.

Every field list below reflects the finalized ontology (checkpoint §2), the Binding
definition (§3), the closed registry (§4), coupling (§5), human-approved aliases (§6), and
the Class A/B rule (§7).

---

## Value objects

### ScopeSelector

A path over declared scope dimensions, partially ordered by containment
(checkpoint §5.2).

- `path` — ordered dimension/value pairs, e.g. `region:EU / tier:enterprise`; empty means `global`

Operations (domain service, not model logic): `equals`, `contains`, `intersects`,
`is_disjoint`, `distance`.

Permitted dimensions for a key come from that key's registry entry. A scope is validated,
never invented.

### TypedValue

- `value_type` — `scalar | interval | duration | enum | boolean | reference`
- `raw` — typed payload
- `unit` — **required** when `value_type ∈ {scalar, duration}`

Comparison is unit-aware. A unit mismatch **blocks** coupling rather than weakening it
(checkpoint §5.6).

### Derivation

`observed | extracted | derived | inferred | absent`

`derived` (deterministic rule) and `inferred` (model-proposed) are **separate values**.
Collapsing them destroys the auditability of Class A.

### Support

`verified_span | multi_span | rule_over_records | mechanism_only | negative_provenance`

### Status

`supported | likely | disputed | speculative | rejected | unresolved`

### Severity

`LOW | MEDIUM | HIGH | CRITICAL`

Judge-assigned by design (checkpoint §9.4). Never numeric, never blended with confidence.

### ConsequenceClass

`class_a_assumption_mediated | class_b_behavioral`

**Pipeline-assigned. Absent from every agent-authored model** (checkpoint §7.2).

### AuthorityClass

A total order declared in configuration and versioned as `authority_model_version`, e.g.

```
regulatory > policy > standard > sop_runbook > config > training > wiki > ticket > chat
```

Resolved from the holder document at comparison time; not stored on bindings
(checkpoint §3.5a).

---

## Identity and sources

### ChangeRequest
- `change_id`
- `tenant_id`
- `title`
- `description`
- `requested_by`
- `effective_date`
- `scope` — ScopeSelector
- `created_at`
- `source_refs`

### ChangeSpec
- `change_id`
- `subject_entities`
- `old_state`
- `new_state`
- `scope` — ScopeSelector
- `constraints`
- `analyst_caveats` — *renamed from `assumptions`*: caveats about the change request itself. **Not** AssumptionRecords; closer to type E. The old name invited miswiring to the coupling index.
- `ambiguities`
- `affected_domains`
- `assumption_deltas` — list of AssumptionDelta
- `version`

Carries no `confidence` and no `consequence_class`.

### SourceDocument
- `source_id`
- `tenant_id`
- `source_type`
- `title`
- `uri`
- `owner`
- `authority_class` — AuthorityClass; drives the override legitimacy test
- `created_at`
- `updated_at`
- `permissions`
- `current_version_id`

### SourceVersion
- `version_id`
- `source_id`
- `content_hash`
- `version_label`
- `effective_from`
- `effective_to`
- `created_at`

### Chunk
- `chunk_id`
- `source_version_id`
- `ordinal`
- `text`
- `char_start` / `char_end`

### EvidenceItem
- `evidence_id`
- `source_version_id`
- `chunk_id`
- `text_span`
- `char_start` / `char_end` — required for mechanical span verification
- `location` — the section trail the span came from, for a legible citation
- `metadata`
- `retrieval_scores`
- `authority_score`
- `freshness_score`

### Offset semantics

Every character offset in the system — on a `Chunk`, on an `EvidenceItem`, in a parser
section map — is **document-absolute**: it indexes into the canonical text of its
`SourceVersion`, never into the chunk that happens to contain it.

Two consequences, both deliberate:

- span verification is a single slice comparison against one string,
  `text[char_start:char_end] == text_span`, with no fuzzy matching anywhere;
- a chunker change cannot invalidate a citation an earlier analysis recorded, because an
  `evidence_id` is derived from `(version_id, char_start, char_end)` and not from the chunk.

The canonical text is produced by the parser and is byte-faithful apart from three
normalizations that would otherwise move offsets between environments: newline folding
(CRLF/CR to LF), Unicode NFC composition, and BOM removal. Whitespace is **not** collapsed
and case is **not** folded, because the span-verification invariant requires the stored text
to contain a cited value verbatim.

A chunk's span is trimmed to its non-whitespace content, so `text[start:end]` recovers the
stored `Chunk.text` exactly. Chunks do not overlap — an overlapping span would let one
sentence be counted as two corroborating sources (checkpoint §9.3).

For JSON sources the canonical text is a deterministic, key-sorted flat rendering
(`auth.reset_token.ttl_hours: 24`), and the content hash is taken over that rendering. So
reindenting or reordering a config does not create a new source version, and an evidence
span reads as a path and a value rather than as a fragment of raw JSON. The original bytes
are hashed separately as `raw_sha256`.

---

## Ingestion-time artifacts

These exist between extraction and admission. **None of them is a domain claim.** They are
listed here because they carry provenance a reviewer needs, not because anything downstream
may trust them.

### AssumptionProposal

What the extractor returns. A proposal, never a binding: admission produces a different
object, so there is no field an extractor can set that makes its output trusted.

- `proposed_key` — free text on purpose, so an unregistered key survives into a
  `KeyRegistryGap` rather than failing to parse
- `scope` — rendered scope path; defaults to `global`
- `value_quote` — the value **exactly as written in the document** (`10,000`, not `10000`)
- `value_type`, `unit`
- `role`, `explicitness`
- `quote` — the span the value was read from, copied character-for-character
- `chunk_ordinal` — which chunk the extractor was shown
- `effective_from` / `effective_to` — only when the text states them
- `ambiguity` — free text; where uncertainty goes

Structural refusals, enforced by the schema rather than by prompt text:

- **no offsets and no ids.** The model supplies a quote; code locates it. A model that could
  supply offsets could supply plausible ones, and there would be nothing left to verify.
- **no `confidence`, no `severity`, no `consequence_class`.** `extra="forbid"` means an
  extractor emitting one is rejected, not silently trimmed.
- **`value_quote` must appear inside `quote`.** A proposal citing a real sentence with an
  unrelated number attached cannot be constructed at all.

### QuarantinedProposal

A proposal that failed admission, kept with enough provenance to be reviewed.

- `failure` — one of the admission failure reasons, not a generic "rejected"
- `detail`
- `proposal`
- `source_version_id`, `chunk_id`
- `gap` — set only for `key_unregistered`; the quarantine and the gap are one event
- `proposed_by` — ModelProvenance for the call that produced it

It asserts nothing about the world, only that a model proposed something and code refused
it. Failures are recorded rather than dropped because their distribution is how we learn
whether the extractor, the registry, or the corpus is at fault (`docs/EVALUATION.md`).

### Admission failure reasons

`chunk_unknown` · `span_not_found` · `span_mismatch` · `value_not_in_span` ·
`value_parse_failed` · `key_unregistered` · `key_deprecated` · `alias_not_approved` ·
`registry_type_mismatch` · `unit_mismatch` · `scope_malformed` · `scope_not_permitted` ·
`interval_invalid` · `holder_metadata_missing` · `tenant_mismatch`

They are separate values because each implies a different reviewer action.
`key_unregistered` in particular is not an error: it is the signal that says where to grow
the registry.

---

## The registry and its gates

### AssumptionKeyRegistryEntry

The registry is **closed** (checkpoint §4). Only registered keys may participate in Class A
coupling.

- `key` — canonical dotted lowercase path
- `value_type`
- `unit` — required when `value_type ∈ {scalar, duration}`
- `canonical_form` — the constraint shape extractors must emit; carries polarity handling
- `allowed_scopes` — permitted scope dimensions
- `description`
- `domain`
- `introduced_in_version`
- `deprecated_in_version` — nullable

### KeyRegistryGap

Emitted when extraction proposes an unregistered key. The key is **quarantined, never
indexed**.

- `gap_id`
- `proposed_key`
- `proposed_value` — TypedValue, as extracted
- `evidence_id`
- `holder_source_version_id`
- `proposed_by` — agent name + `prompt_sha256` + `provider` + `model`
- `as_of`
- `registry_version` — the version it failed against
- `occurrence_count` — aggregated; **volume is an evaluation and product signal**, not an error log
- `status` — `open | registered | rejected`

### KeyAlias

Aliases are **human-approved** (checkpoint §6). No automatic or threshold-based approval.

- `alias_id`
- `source_key`
- `target_key`
- `proposal_evidence` — evidence_ids supporting the equivalence claim
- `proposed_by`
- `proposed_at`
- `approved_by` — null while unapproved
- `approved_at` — null while unapproved
- `registry_version`
- `status` — `proposed | approved | rejected | superseded`
- `rejection_reason` — nullable

Only `status = approved` participates in Class A coupling. `proposed` may appear in Class B
analysis as a marked proposed relationship.

---

## The five ontology types

### ObservedFact (type A)
- `fact_id`
- `statement`
- `evidence_ids` — **≥1, required**
- `derivation` — `observed`
- `support` — `verified_span | multi_span`

### AssumptionRecord (type B)

The core unit of the mechanism.

- `assumption_id`
- `tenant_id`
- `key` — resolved in the registry
- `registry_version` — **required**; pins the version admitted under
- `scope` — **ScopeSelector, required** (defaults to `global`)
- `value` — TypedValue
- `role` — `declares | consumes | describes`
- `explicitness` — `explicit | derived_from_text | implied`
- `holder_source_version_id`
- `evidence_id` — **required**
- `effective_from` / `effective_to`
- `extractor_version`
- `span_verified` / `value_parsed` / `key_resolved` — verification results
- `extraction_quality` — **computed** from the three flags above
- `created_at`

Changes from the previous draft: `scope` and `registry_version` added; `binding_kind` split
into `role` + `explicitness`; **`extraction_confidence` removed** — it was an LLM-supplied
number wearing a carve-out (checkpoint §9.1).

Invariants:
- no `evidence_id` ⇒ invalid, must not persist
- `role` and `explicitness` both required; no combined `binding_kind` field exists
- a record failing any admission check is quarantined, never indexed
- `extraction_quality` is never supplied by a model and never propagates directly into
  consequence confidence

### ResolvedBinding — the coupling read-model

Not a stored entity. An `AssumptionRecord` joined to the `authority_class` of the document
that holds it, assembled once by the application layer so the coupling engine is a **pure
function of its inputs**.

- `record` — the `AssumptionRecord`
- `authority_class` — resolved from `SourceDocument` at comparison time (checkpoint §3.5a)
- `source_id` / `source_title` — for the report surface

Authority is a property of the **holder document**, not of the binding, and it comes from
registration rather than content. Doing the join in one place keeps that security property
auditable: nothing in the coupling path reads authority from a file.

Invariants:
- a record that has not passed all three admission checks **cannot be constructed** as a
  `ResolvedBinding`, so a quarantined binding has no path into the index
- no `ModelProvenance` is reachable from it, so the coupling engine cannot see an agent and
  therefore cannot count one (checkpoint §9.3)

### DerivedRelation (type C)

Computed by a named deterministic rule. **Previously had no entity at all.**

- `relation_id`
- `rule_id` — the deterministic rule that produced it
- `rule_version`
- `input_record_ids` — **non-empty**
- `relationship` — `invalidates_assumption | conflicts_with | unauthorized_override`
- `subject_ref` / `object_ref`
- `derivation` — `derived`
- `support` — `rule_over_records`
- `evidence_ids` — **must be empty**
- `derivation_detail` — rule inputs; for override tests, both `authority_class` values and `authority_model_version`
- `coupling_index_version`
- `registry_version`

**Carries no evidence of its own.** Its evidence set is the computed transitive union of its
inputs, materialized on read. Requiring a direct `evidence_id` would force the resolver to
nominate an arbitrary input span and re-present it as direct support — fabricated
provenance.

### ClassAFinding — the resolver's output, before judgment

The reportable form of a `DerivedRelation`: one finding per **affected artifact**, carrying
the materialized evidence a reviewer needs and nothing a model supplied.

- `finding_id` / `tenant_id`
- `consequence_class` — `Literal[class_a_assumption_mediated]`, pipeline-assigned
- `origin` — `Literal[resolver]`
- `relation_id` / `relationship` / `rule_id` / `rule_version`
- `assumption_key` / `scope` / `as_of`
- `affected_source_version_id` / `affected_source_id` / `affected_artifact`
- `counterpart_source_version_id` / `counterpart_artifact` — nullable; absent only for a
  delta invalidation with no declaration to supersede
- `summary` / `observed_value` / `authoritative_value`
- `input_record_ids` — **non-empty**
- `delta_id` — nullable; the second derivation input when a change triggered the finding
- `materialized_evidence_ids` — **non-empty**, the computed union of the inputs' evidence
- `evidence_is_derived_union` — `Literal[True]`
- `strength` — `CouplingStrength`, nullable
- `merged_hypothesis_ids` — Class B hypotheses absorbed as mechanism articulation
- `registry_version` / `coupling_index_version` / `authority_model_version`

**Why this is not a `Consequence`.** A `Consequence` requires a `severity`, and §9.4 makes
severity *judgment* — a function of business impact the corpus does not contain — assigned
by the judge and attributed to a prompt/provider/model. The resolver has no model and no
business context, so having it stamp a severity would launder a guess into the trust-bearing
tier. The judge converts a `ClassAFinding` into a `Consequence` in a later phase by adding
exactly the field it is qualified to add. Recorded as an open review item against the
checkpoint's entity map in `docs/STATUS.md`.

Invariants:
- `consequence_class` and `origin` are `Literal`s, so no agent-authored payload can
  deserialize into one and no finding can be relabelled Class B
- carries **no** `severity` and **no** `confidence`; `extra="forbid"` rejects either key
- `relationship` must be one of the three derivable edges
- a derivation needs two inputs: either ≥2 distinct records, or one record plus a `delta_id`
- `materialized_evidence_ids` non-empty — a derivation nobody can check is not assertable

### ConsequenceHypothesis (type D)
- `hypothesis_id`
- `claim`
- `mechanism` — required
- `weakest_link` — the link in the chain the author considers least supported
- `affected_entities`
- `evidence_ids` — **may be empty**
- `evidence_gap_ids`
- `historical_analogue_id` — **required** to exceed `speculative`
- `derivation` — `inferred`
- `support` — `mechanism_only`
- `agent_sources`
- `prompt_sha256` / `provider` / `model`

Carries **no** `confidence`, **no** `severity`, **no** `consequence_class`.

### EvidenceGap (type E)

**Previously only a free-text `unknowns: list[str]`**, which could not distinguish
searched-and-absent from never-searched.

- `gap_id`
- `analysis_id`
- `question` — what could not be established
- `strategies_executed` — the QueryPlan that returned nothing
- `source_types_searched`
- `as_of`
- `would_be_resolved_by` — the source or check that would settle it
- `derivation` — `absent`
- `support` — `negative_provenance`
- `evidence_ids` — **must be empty**

**Never becomes a graph edge.** May annotate a node; is not a relationship.

---

## Coupling

### CouplingSet

A deterministic projection, **never written by an agent**.

- `key`
- `scope` — part of the identity
- `as_of` — part of the identity
- `holders` — list of `(source_version_id, assumption_id, value, role, authority_class)`
- `holder_count`
- `declarer_count`
- `distinct_declared_values`
- `outcome` — `consistent | conflicts_with | legitimate_override | unauthorized_override | ambiguous_temporal`
- `authority_model_version`
- `computed_at`
- `coupling_index_version`

Identity is the triple **`(key, scope, as_of)`**, not `key` alone. The previous
`has_collision := distinct_values > 1` rule is **removed**: it was scope-blind and
authority-blind, and would report every legitimate override as a contradiction.

`outcome` is decided by the deterministic three-way rule in checkpoint §5.4.

**Membership** is every admitted binding of `key` whose scope is **comparable** with
`scope` — equal, containing, or contained — and whose validity interval contains `as_of`.
Comparability rather than equality is what makes inheritance work: the set at `region:EU`
contains the `global` declaration that governs it. Siblings are excluded.

Note what `outcome` does **not** include: a stale consumer. The frozen outcome vocabulary
describes divergence *between declarers*, and a consumer disagreeing with its authority is
a separate finding (`invalidates_assumption`) rather than a property of the cell. A cell
whose declarers agree is `consistent` even when a dependent artifact is stale.

`outcome` precedence within one cell, strongest first: `conflicts_with`,
`unauthorized_override`, `ambiguous_temporal`, `legitimate_override`, `consistent`. A
relation is attributed to a cell only when **every** input record is a member of it, so one
defect appears in one place rather than in every cell that can see it.

### AuthorityResolution

The result of `resolve_authority(key, scope, as_of)`. A projection, not a stored entity.

- `key` / `scope` / `as_of`
- `outcome` — `resolved | ambiguous | no_declarer`
- `authoritative` — the governing `ResolvedBinding`; present **iff** `outcome = resolved`
- `candidates` — every declarer considered, each with `selected` and
  `unauthorized_narrowing_of`
- `ambiguity_reason` — `equal_authority_divergence | incomparable_values`; required iff ambiguous
- `unauthorized_narrowings` — narrower bindings that tried to override a higher authority
- `authority_model_version` / `coupling_index_version`

`ambiguous` and `no_declarer` are **real return values**, not errors. Newest-wins is
forbidden: the module contains no recency comparison between candidates, and the schema
refuses to name a winner unless the outcome is `resolved`.

### OverrideRecord

A **legitimate** narrower override (§5.4.2). Recorded, never reported.

- `key` / `narrower` / `broader` / `as_of` / `authority_model_version`

It emits no finding and no edge, because it is correct behaviour — but the invalidation pass
needs to know it exists, so a broader change reports `scope_inheritance_affected` for
confirmation rather than treating the override as broken. The schema refuses to represent
an override whose narrower side does not match or outrank the broader one; that shape is an
`unauthorized_override`.

### CouplingAnomaly

A reportable observation that is deliberately **not** a Class A finding.

- `kind` — `consumer_divergence | blocked_by_unit_mismatch | intra_document_divergence`
- `key` / `as_of` / `participants` (**≥2**) / `detail` / `coupling_index_version`

Two `consumes` bindings disagreeing is the canonical case: neither artifact is an authority,
so neither can be said to be wrong (§3.3). There is no edge for it in the seven-edge
taxonomy, and inventing one would recreate the `potentially_impacts` catch-all the taxonomy
exists to exclude.

### CouplingStrength

Deterministic, versioned, computed from evidence properties only (§5.7).

- `strength` — `[0, 1]`
- `blocked` — a failed gate; `strength` must then be exactly `0.0`
- `factor_inputs` — **non-empty**, so the score is recomputable and explainable
- `strength_fn_version`

Inputs: role composition (+), explicitness (+), scope distance (−), temporal overlap (+),
holder authority (+), alias mediation (−). **Gates:** span verification and unit
compatibility. A unit mismatch **blocks** coupling rather than weakening it — expressing it
as a low score would let a strong enough set of other factors carry an incomparable pair
over a threshold.

The function's signature carries no agent identity, agent count, model agreement or model
confidence, and none is reachable from `ResolvedBinding`. It cannot count agents because it
cannot see them (§9.3).

### AssumptionDelta
- `key`
- `scope` — ScopeSelector
- `old_value` / `new_value` — TypedValue
- `delta_kind` — `value_change | key_removed | key_added | scope_change`
- `asserted_by` — `change_request | decomposer_inference`
- `evidence_ids` — spans of the change request itself; the decomposer runs before retrieval
- `registry_version`

`confidence` **removed** — it was produced by the change-decomposer, an LLM.

### InvalidationReport

What one delta deterministically implies, with nothing merged.

- `delta` / `as_of`
- `impacts` — per binding: `outcome`, `reason`, and the `DerivedRelation` where one was emitted
- `pre_existing_unauthorized_overrides` — defects the change did **not** cause

`outcome` is `not_affected | invalidates_assumption | scope_inheritance_affected`.

`scope_inheritance_affected` exists so a **legitimate** narrower override is never reported
as broken by a broader change: it may still be intended, and it requires confirmation rather
than correction. The override itself is never labelled contradictory by an invalidation pass
(§5.5). Legitimacy is decided by `resolve_authority` at the binding's own scope, not by the
mere existence of an override record naming it — one artifact can legitimately outrank a
peer at `global` while remaining an unauthorized narrowing of a policy.

A pre-existing `unauthorized_override` is reported **separately and never merged** with the
change's impact. Folding them together would blame the change for a governance defect that
predates it.

The declaration the change *edits* — matched on exact scope, `declares` role, and a value
equal to `old_value` — is reported `not_affected`. A change to the retention policy does not
make the retention policy stale; it *is* the edit to it.

### KnowledgeDecayReport

The standing corpus-health deliverable (`IDEA.md` §15.5). Takes **no change request** as
input.

- `tenant_id` / `as_of`
- `entries` — `DecayEntry` rows
- `keys_examined` / `bindings_examined`
- `registry_version` / `coupling_index_version` / `authority_model_version`

Each `DecayEntry` carries `category`, `key`, `scope`, `summary`, `participants`,
`finding_id` (nullable), `rule_id` (nullable), `requires_human_decision`, and
`what_would_resolve` (nullable).

The report keeps three things apart, and the schema enforces the separation:

| bucket | categories | rule_id |
|---|---|---|
| **known inconsistency** — an artifact is wrong | `stale_consumer`, `unauthorized_override`, `contradictory_declarations` | **required** |
| **known ambiguity** — the corpus cannot decide | `ambiguous_authority`, `consumer_divergence` | **forbidden** |
| **missing knowledge** — nobody wrote it down | `registry_gap`, `single_holder`, `unwritten_rule` | **forbidden** |

An inconsistency row must cite the rule that derived it, or the report would become a second
source of truth rather than a projection of the resolver. A missing-knowledge row must cite
neither a rule nor a finding: **absence is recorded, never derived.** An ambiguity row must
state what human validation would resolve it, and `requires_human_decision` is mandatory —
handing an ambiguity to a reviewer as though it were a defect wastes their time, and handing
a defect over as an open question means it never gets fixed.

A symmetric contradiction produces **one** row even though it produces two findings: both
authors are told, but it is one inconsistency in the corpus.

---

## Findings and arbitration

### SpecialistFinding
- `finding_id`
- `agent`
- `claim`
- `category`
- `affected_entities`
- `evidence_ids`
- `evidence_position` — `supported | weakly_supported | disputed | unsupported | unresolved`
- `proposed_bindings` — candidate AssumptionRecords, subject to admission
- `evidence_gap_ids`
- `prompt_sha256` / `provider` / `model`

`confidence` **removed**; `unknowns: list[str]` replaced by `evidence_gap_ids`.

### ChallengeResult

Exactly three outcomes; an uncited objection is **unrepresentable**.

- `challenge_id`
- `consequence_id`
- `outcome` — `refuted | weakened | no_refutation_found`
- `contradicting_evidence_ids` — **≥1** for `refuted` and `weakened`; **empty** for `no_refutation_found`
- `contradicts_what` — required for `refuted`
- `scope_narrowing` — required for `weakened`
- `falsification_test_id` — **required** for `no_refutation_found`
- `class_a_ground` — required when challenging a Class A consequence: `disputed_span | disputed_key_or_scope | disputed_temporal_validity | superseding_version`
- `prompt_sha256` / `provider` / `model`

`position` (untyped) is **replaced** by `outcome`. `confidence` **removed**. Mechanism-only
challenges to a `DerivedRelation` are inadmissible (checkpoint §8.1).

### FalsificationTest
- `test_id`
- `consequence_id`
- `observation` — what would refute the claim
- `data_source_needed`
- `feeds_validation_checklist` — bool

### ArbitrationDecision
- `consequence_id`
- `status`
- `severity` — judge-assigned by design
- `rationale`
- `supporting_evidence_ids`
- `contradicting_evidence_ids`
- `preserved_disagreements`
- `validation_actions`
- `in_default_view` — bool
- `prompt_sha256` / `provider` / `model`

`confidence` **removed** — the judge is explicitly told not to emit one
(`.claude/agents/consequence-judge.md`), so the field contradicted its own agent contract.

### ConsequenceScore

Confidence lives here and **only** here. Computed, never supplied.

- `consequence_id`
- `confidence` — computed
- `confidence_fn_version`
- `factor_inputs` — persisted so any score is recomputable and explainable
- `corroboration_count` — distinct `(source_version_id, source_type)` reached by distinct strategies
- `corroborating_source_version_ids`
- `computed_at`

The scorer's input type **does not contain agent identity**. It cannot count agents because
it cannot see them (checkpoint §9.3).

### Consequence

The arbitrated entity.

- `consequence_id`
- `change_id`
- `title`
- `description`
- `category`
- `consequence_class` — **pipeline-assigned**
- `affected_entity`
- `mechanism`
- `origin` — `resolver | agent`; the basis of the Class A/B decision (checkpoint §7.2)
- `derived_relation_ids` — for Class A
- `hypothesis_ids` — for Class B
- `status`
- `severity`
- `evidence_ids`
- `contradicting_evidence_ids`
- `assumption_keys`
- `evidence_gap_ids`
- `recommended_validation`
- `agent_sources`
- `merged_hypothesis_ids` — Class B hypotheses merged into this Class A finding (checkpoint §7.5)

`likelihood` **removed**: undefined, and a probability by another name, which `IDEA.md` §13
forbids before calibration.

### FindingDisposition
- `disposition_id`
- `consequence_id`
- `analysis_id`
- `assumption_key` — nullable, for suppression scoping
- `artifact_ref` — nullable, for suppression scoping
- `disposition` — `useful | already_known | not_material | wrong | unverifiable`
- `note`
- `disposed_by`
- `disposed_at`
- `suppresses_until_source_revision` — bool

---

## Graph

The graph is a **pure projection** rebuilt from the store. It is never a second source of
truth, and no agent writes an edge.

### GraphNode
- `node_id`
- `entity_type` — `source_document | assumption_key | entity | change | historical_event`
- `label`
- `metadata` — includes `owner`, `documented_by`, `implemented_by`, `authority_class`
- `annotations` — e.g. single-holder key; may reference EvidenceGaps

`assumption_key` is a node type **because that is what makes hidden coupling visible**: two
documents joined through a key neither mentions is the central visualization.

### GraphEdge
- `edge_id`
- `source_node` / `target_node`
- `relationship` — the seven MVP types
- `derivation`
- `directionality` — `directed | symmetric`
- `consequence_id` — required for consequence-bearing edges
- `derived_relation_id` — for `derived` edges
- `evidence_ids`
- `is_hypothesized` — bool; hypothesized edges are structurally distinct so a client cannot render them identically to derived ones

`confidence` and `severity` **removed** — they live on the consequence and its score.
Duplicated scores diverge.

There is also no `produced_by` / model attribution field on an edge, and that absence is
the structural form of "no agent writes an edge": there is nowhere to record which model
authored one.

Edge invariants:
- `derivation` must be admissible for the `relationship` (table below)
- `directionality` is fixed by the relationship, not chosen per edge
- `consequence_id` required for consequence-bearing relationships
- `derived_relation_id` required when `derivation = derived`, forbidden otherwise
- `derivation ∈ {observed, extracted}` requires at least one `evidence_id`
- `derivation = derived` requires **empty** `evidence_ids` — a derived edge's evidence is
  the transitive union of its inputs, materialized on read, exactly as for `DerivedRelation`
- `is_hypothesized` is true iff `derivation = inferred`
- no self-loops, and a symmetric edge is stored once in canonical endpoint order so two
  projections of the same store cannot differ

### ConsequenceGraph

The materialized projection.

- `is_projection` — literally `true`; rebuilding is the only way one comes into existence
- `analysis_id`
- `nodes` / `edges`
- `registry_version`, `coupling_index_version` — required, so the rebuild is reproducible
- `projected_at`

Invariants: node and edge ids are unique, and every edge endpoint must be a node present
in the same projection. A dangling endpoint means the graph was assembled rather than
derived.

### Where each edge comes from (as implemented)

`app/graph/projection.py` is the only producer. Every edge is projected from exactly one
record, and the record it projects is named on the edge:

| edge | projected from | back-reference on the edge |
|---|---|---|
| `binds_assumption` | one admitted `AssumptionRecord` | `evidence_ids` = the binding's span |
| `directly_changes` | one `AssumptionDelta` on the `ChangeSpec` | `evidence_ids` = the change's spans |
| `invalidates_assumption` | a `DerivedRelation` + its arbitrated `Consequence` | `derived_relation_id`, `consequence_id` |
| `conflicts_with` | as above | as above |
| `unauthorized_override` | as above | as above |
| `creates_risk_for` | an arbitrated **Class B** `Consequence` | `consequence_id`, `evidence_ids` |
| `historically_analogous_to` | a Class B consequence's `historical_analogue_id` | `consequence_id` |

Two consequences of the frozen invariants, worth stating because they are load-bearing rather
than incidental:

- the three `derived` edges are **consequence-bearing**, so a relation the resolver derived but
  the pipeline never arbitrated cannot project - it is reported as skipped;
- a `rejected` consequence projects no edge, because drawing it would show a reader a
  relationship the arbitration concluded was false. Its `binds_assumption` edges survive: those
  are facts about what the documents say, and a rejected consequence does not un-bind a record.

Node and edge ids are **content-derived**, not minted (`node_id_for`, `edge_id_for` in
`app/ingestion/identity.py`), so rebuilding an unchanged store yields an identical graph and a
stored edge still resolves after a rebuild.

### Edge taxonomy

| edge | derivation | direction | ontology type |
|---|---|---|---|
| `directly_changes` | `observed` | directed | A |
| `binds_assumption` | `extracted` | directed | B — projection of the record, never authored |
| `invalidates_assumption` | `derived` | directed | C |
| `conflicts_with` | `derived` | **symmetric** | C |
| `unauthorized_override` | `derived` | **directed** (narrower → broader) | C |
| `creates_risk_for` | `inferred` | directed | D |
| `historically_analogous_to` | `inferred` | directed | D |

Seven edges. Grew from six via the documented extension path (checkpoint §5.4.3).

`historically_analogous_to` is `inferred`, **not** `observed`: the incident is observed, the
analogy is a judgment.

Excluded from the MVP: `potentially_impacts` (unbounded catch-all), `depends_on`,
`constrains`, `enables`, `operationally_precedes`, `operationally_follows`,
`compensates_for`, `measured_by`. Ownership, documentation and implementation are node
metadata.

---

## Provenance and reproducibility

Every persisted finding records enough to explain why two runs differ:

- `prompt_version` / `prompt_sha256`
- `schema_version`
- `provider` — the provider that actually served the call
- `model` — the identifier actually used
- `tier` — `reasoning | standard | fast`
- `confidence_fn_version`
- `coupling_index_version`
- `registry_version`
- `authority_model_version`
- `retrieval_config_version`
- `permission_scope_hash`

`provider` and `model` are per-finding because provider identity is an experimental
variable in the diversity study (`ARCHITECTURE.md` §4a), not merely an operational detail.

`registry_version` and `authority_model_version` are per-finding because a registry
expansion or an authority-ordering change legitimately produces new Class A derivations,
and two runs differing only in those must be **diffable**.

## Versioning rule

Prompts, schemas, routing policies, model configuration, retrieval configuration, the
assumption key registry, the authority ordering, alias approvals, and benchmark definitions
are versioned enough to explain why two analysis runs differ.

Registry and alias updates **never retroactively mutate stored derivations**. Re-running
the resolver at a newer version may admit previously quarantined bindings and produce new
Class A findings. That is the intended growth path.
