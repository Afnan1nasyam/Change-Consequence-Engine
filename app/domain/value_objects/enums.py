"""Domain enumerations.

Every value set here is frozen by ``docs/DOMAIN_CHECKPOINT.md``. Adding a member is a
specification change, not a code change: the checkpoint must be amended first, and
``scripts/check_spec_consistency.py`` asserts the two agree.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Epistemic axes (checkpoint 2.1, 2.2)
# ---------------------------------------------------------------------------


class Derivation(StrEnum):
    """How a claim came to exist.

    ``DERIVED`` and ``INFERRED`` are deliberately separate. Collapsing them - as an earlier
    draft did - makes one value mean both "computed by a named rule" and "proposed by a
    model", which destroys the auditability of Class A.
    """

    OBSERVED = "observed"
    EXTRACTED = "extracted"
    DERIVED = "derived"
    INFERRED = "inferred"
    ABSENT = "absent"


class Support(StrEnum):
    """What backs a claim.

    A claim whose declared support fails mechanical verification is a hallucination. A
    claim with *no* support declared as ``MECHANISM_ONLY`` is merely a hypothesis. The two
    axes exist so those are never confused (checkpoint 2.3).
    """

    VERIFIED_SPAN = "verified_span"
    MULTI_SPAN = "multi_span"
    RULE_OVER_RECORDS = "rule_over_records"
    MECHANISM_ONLY = "mechanism_only"
    NEGATIVE_PROVENANCE = "negative_provenance"


class ClaimType(StrEnum):
    """Discriminator for the five-part ontology (checkpoint 2)."""

    OBSERVED_FACT = "observed_fact"
    ASSUMPTION_BINDING = "assumption_binding"
    DERIVED_RELATION = "derived_relation"
    HYPOTHESIS = "hypothesis"
    EVIDENCE_GAP = "evidence_gap"


# ---------------------------------------------------------------------------
# Bindings (checkpoint 3.3, 3.4, 3.6)
# ---------------------------------------------------------------------------


class BindingRole(StrEnum):
    """Dependency direction. Collision semantics depend entirely on this."""

    DECLARES = "declares"
    CONSUMES = "consumes"
    DESCRIBES = "describes"


class Explicitness(StrEnum):
    """How directly the span states the value. Ordering feeds the confidence function."""

    EXPLICIT = "explicit"
    DERIVED_FROM_TEXT = "derived_from_text"
    IMPLIED = "implied"


#: Ordered strongest to weakest, for the confidence factor.
EXPLICITNESS_RANK: dict[Explicitness, int] = {
    Explicitness.EXPLICIT: 3,
    Explicitness.DERIVED_FROM_TEXT: 2,
    Explicitness.IMPLIED: 1,
}


class ValueType(StrEnum):
    SCALAR = "scalar"
    INTERVAL = "interval"
    DURATION = "duration"
    ENUM = "enum"
    BOOLEAN = "boolean"
    REFERENCE = "reference"


#: Value types for which ``unit`` is mandatory. A unitless scalar cannot be compared
#: safely, and an unsafe comparison is how false contradictions get manufactured.
UNIT_REQUIRED_TYPES: frozenset[ValueType] = frozenset({ValueType.SCALAR, ValueType.DURATION})


class AuthorityClass(StrEnum):
    """Holder authority, used by the override legitimacy test (checkpoint 3.5a, 5.4).

    Sourced from document *registration*, never from document content, so an injected file
    cannot claim policy authority (``docs/SECURITY.md``).
    """

    REGULATORY = "regulatory"
    POLICY = "policy"
    STANDARD = "standard"
    SOP_RUNBOOK = "sop_runbook"
    CONFIG = "config"
    TRAINING = "training"
    WIKI = "wiki"
    TICKET = "ticket"
    CHAT = "chat"


#: The total order. Versioned as ``authority_model_version`` and pinned onto every
#: derivation that used it, because changing the ordering changes findings.
AUTHORITY_RANK: dict[AuthorityClass, int] = {
    AuthorityClass.REGULATORY: 90,
    AuthorityClass.POLICY: 80,
    AuthorityClass.STANDARD: 70,
    AuthorityClass.SOP_RUNBOOK: 60,
    AuthorityClass.CONFIG: 50,
    AuthorityClass.TRAINING: 40,
    AuthorityClass.WIKI: 30,
    AuthorityClass.TICKET: 20,
    AuthorityClass.CHAT: 10,
}

AUTHORITY_MODEL_VERSION = "authority-1"


# ---------------------------------------------------------------------------
# Coupling (checkpoint 5.4)
# ---------------------------------------------------------------------------


class CouplingOutcome(StrEnum):
    """The deterministic three-way resolution of divergence between declarers.

    ``LEGITIMATE_OVERRIDE`` emits no finding. Reporting one as a contradiction is the
    fastest way to make the system untrusted, because the reader can see it is correct
    behaviour.
    """

    CONSISTENT = "consistent"
    CONFLICTS_WITH = "conflicts_with"
    LEGITIMATE_OVERRIDE = "legitimate_override"
    UNAUTHORIZED_OVERRIDE = "unauthorized_override"
    AMBIGUOUS_TEMPORAL = "ambiguous_temporal"


class DeltaKind(StrEnum):
    VALUE_CHANGE = "value_change"
    KEY_REMOVED = "key_removed"
    KEY_ADDED = "key_added"
    SCOPE_CHANGE = "scope_change"


class DeltaAssertedBy(StrEnum):
    CHANGE_REQUEST = "change_request"
    DECOMPOSER_INFERENCE = "decomposer_inference"


# ---------------------------------------------------------------------------
# Registry and aliases (checkpoint 4, 6)
# ---------------------------------------------------------------------------


class RegistryGapStatus(StrEnum):
    OPEN = "open"
    REGISTERED = "registered"
    REJECTED = "rejected"


class AliasStatus(StrEnum):
    """Only ``APPROVED`` participates in Class A coupling. No automatic approval exists."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


# ---------------------------------------------------------------------------
# Consequences (checkpoint 7, 9)
# ---------------------------------------------------------------------------


class ConsequenceClass(StrEnum):
    """Pipeline-assigned. Absent from every agent-authored model (checkpoint 7.2)."""

    CLASS_A_ASSUMPTION_MEDIATED = "class_a_assumption_mediated"
    CLASS_B_BEHAVIORAL = "class_b_behavioral"


class Origin(StrEnum):
    """What produced the consequence. This *is* the Class A/B decision (checkpoint 7.2).

    Classification is by producer, not by subject matter: sounding derivable is not a
    derivation.
    """

    RESOLVER = "resolver"
    AGENT = "agent"


class Status(StrEnum):
    SUPPORTED = "supported"
    LIKELY = "likely"
    DISPUTED = "disputed"
    SPECULATIVE = "speculative"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


#: Statuses that assert the claim holds, and therefore require evidence (checkpoint 9.2).
ASSERTIVE_STATUSES: frozenset[Status] = frozenset({Status.SUPPORTED, Status.LIKELY})

#: Statuses a Class B consequence may reach only with a historical analogue that has a
#: recorded observed outcome.
STATUSES_ABOVE_SPECULATIVE: frozenset[Status] = frozenset(
    {Status.SUPPORTED, Status.LIKELY, Status.DISPUTED}
)


class Severity(StrEnum):
    """Judge-assigned by design.

    The asymmetry with confidence is deliberate (checkpoint 9.4): confidence is a function
    of evidence properties the system can measure, so it is computed; severity is a
    function of business impact the corpus does not contain, so it is a judgment.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidencePosition(StrEnum):
    """What an agent may say instead of a confidence number."""

    SUPPORTED = "supported"
    WEAKLY_SUPPORTED = "weakly_supported"
    DISPUTED = "disputed"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Challenge (checkpoint 8)
# ---------------------------------------------------------------------------


class ChallengeOutcome(StrEnum):
    """Exactly three. An uncited objection is unrepresentable, not merely discouraged."""

    REFUTED = "refuted"
    WEAKENED = "weakened"
    NO_REFUTATION_FOUND = "no_refutation_found"


class ClassAGround(StrEnum):
    """The only admissible grounds for challenging a deterministic derivation.

    A derived relation is only ever rule-wrong or input-wrong, so "I do not think that
    follows" is inadmissible (checkpoint 8.1). Without this the deterministic layer would
    be only as strong as the most agreeable model in the pipeline.
    """

    DISPUTED_SPAN = "disputed_span"
    DISPUTED_KEY_OR_SCOPE = "disputed_key_or_scope"
    DISPUTED_TEMPORAL_VALIDITY = "disputed_temporal_validity"
    SUPERSEDING_VERSION = "superseding_version"


# ---------------------------------------------------------------------------
# Human feedback (checkpoint, IDEA 21.2)
# ---------------------------------------------------------------------------


class Disposition(StrEnum):
    USEFUL = "useful"
    ALREADY_KNOWN = "already_known"
    NOT_MATERIAL = "not_material"
    WRONG = "wrong"
    UNVERIFIABLE = "unverifiable"


#: Dispositions that suppress the finding for the same (assumption key, artifact) pair
#: until the source is revised. Without persistent suppression the same false positive is
#: re-delivered every analysis and reviewers stop reading.
SUPPRESSING_DISPOSITIONS: frozenset[Disposition] = frozenset(
    {Disposition.WRONG, Disposition.NOT_MATERIAL}
)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class SourceType(StrEnum):
    POLICY = "policy"
    SOP = "sop"
    RUNBOOK = "runbook"
    KNOWLEDGE_DOC = "knowledge_doc"
    TICKET = "ticket"
    INCIDENT = "incident"
    CHANGE_REQUEST = "change_request"
    CODE = "code"
    CONFIG = "config"
    TRAINING = "training"
    DASHBOARD = "dashboard"
    EMAIL_TEMPLATE = "email_template"
    UI_TEXT = "ui_text"
    ORG_METADATA = "org_metadata"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalStrategy(StrEnum):
    """A named way of reaching evidence.

    Only strategies that exist are listed. The value is persisted - corroboration counts
    distinct source versions reached by **distinct strategies** (checkpoint 9.3), and an
    :class:`~app.domain.entities.claims.EvidenceGap` records the strategies that were
    executed and returned nothing. A member naming an unimplemented retriever would let
    both of those claim work that never ran, so graph and hybrid strategies are added here
    when they are built and not before.

    ``VECTOR`` was added with ``app/retrieval/vectors.py`` and not one slice earlier. The
    two members are genuinely distinct for corroboration purposes: the vector index has its
    own preprocessing rather than sharing the lexical tokenizer, precisely so that two
    strategies agreeing is evidence rather than an artefact of a shared stage.

    ``HYBRID`` names the *fusion*, and it is deliberately not what a fused outcome reports
    in ``strategies_executed``: that lists the two strategies that actually ran, because
    checkpoint 9.3 counts evidence reached by **distinct** strategies and one ``hybrid``
    entry would collapse two into one. It appears on ``RetrievalResult.strategy``, where it
    correctly says which function produced *the score*.
    """

    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Value comparison (checkpoint 5.6, 13.5)
# ---------------------------------------------------------------------------


class ValueComparison(StrEnum):
    """The three-way result of a unit-aware typed comparison.

    ``INCOMPARABLE`` is not a weak ``DIFFERENT``. A unit mismatch **blocks** coupling
    outright rather than lowering its strength (checkpoint 5.6), because a comparison
    between a USD threshold and a 0-1 score is not a close call - it is meaningless, and
    reporting it as divergence is how a false contradiction gets manufactured.
    """

    EQUAL = "equal"
    DIFFERENT = "different"
    INCOMPARABLE = "incomparable"


# ---------------------------------------------------------------------------
# Temporal authority resolution (checkpoint 5.3, 13.7)
# ---------------------------------------------------------------------------


class AuthorityOutcome(StrEnum):
    """Result kind of ``resolve_authority``.

    ``AMBIGUOUS`` is a real return value, never an error smoothed away and never resolved
    by picking the newest row. Newest-wins silently invents an authority the corpus does
    not establish.
    """

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NO_DECLARER = "no_declarer"


class AmbiguityReason(StrEnum):
    """Why authority could not be established. Each maps to a different human action."""

    #: Two or more declarers share the top authority class, overlap in time, and disagree.
    EQUAL_AUTHORITY_DIVERGENCE = "equal_authority_divergence"
    #: Candidates at the governing scope cannot be compared at all (unit or type mismatch).
    INCOMPARABLE_VALUES = "incomparable_values"


# ---------------------------------------------------------------------------
# Invalidation propagation (checkpoint 5.5)
# ---------------------------------------------------------------------------


class InvalidationOutcome(StrEnum):
    """What a delta does to one binding.

    ``SCOPE_INHERITANCE_AFFECTED`` exists so that a *legitimate* narrower override is never
    reported as broken by a broader change: it may still be intended, and it requires
    confirmation rather than correction. The checkpoint is explicit that the override
    itself is never labelled contradictory by an invalidation pass (5.5).
    """

    NOT_AFFECTED = "not_affected"
    INVALIDATES_ASSUMPTION = "invalidates_assumption"
    SCOPE_INHERITANCE_AFFECTED = "scope_inheritance_affected"


# ---------------------------------------------------------------------------
# Coupling anomalies (checkpoint 3.3)
# ---------------------------------------------------------------------------


class AnomalyKind(StrEnum):
    """A reportable coupling observation that is deliberately **not** a Class A finding.

    Two ``consumes`` bindings disagreeing is the canonical case: neither artifact is an
    authority, so neither can be said to be wrong, and the checkpoint 3.3 matrix assigns it
    "not a contradiction; report as a coupling anomaly requiring scope review". There is no
    edge in the seven-edge taxonomy for it, and inventing one would be the catch-all the
    taxonomy exists to exclude.
    """

    CONSUMER_DIVERGENCE = "consumer_divergence"
    BLOCKED_BY_UNIT_MISMATCH = "blocked_by_unit_mismatch"
    INTRA_DOCUMENT_DIVERGENCE = "intra_document_divergence"


# ---------------------------------------------------------------------------
# Knowledge decay (IDEA 15.5)
# ---------------------------------------------------------------------------


class DecayCategory(StrEnum):
    """Sections of the standing corpus-health report.

    The report must keep three things apart, because collapsing them is how a corpus-health
    tool becomes untrustworthy: a **known inconsistency** (two artifacts disagree), a
    **known ambiguity** (the corpus cannot say which governs), and **missing knowledge**
    (nobody wrote it down). Absence is never reported as contradiction.
    """

    STALE_CONSUMER = "stale_consumer"
    UNAUTHORIZED_OVERRIDE = "unauthorized_override"
    CONTRADICTORY_DECLARATIONS = "contradictory_declarations"
    AMBIGUOUS_AUTHORITY = "ambiguous_authority"
    REGISTRY_GAP = "registry_gap"
    SINGLE_HOLDER = "single_holder"
    UNWRITTEN_RULE = "unwritten_rule"
    CONSUMER_DIVERGENCE = "consumer_divergence"


#: Categories that assert an artifact is *wrong*. Distinct from ambiguity and absence.
KNOWN_INCONSISTENCY: frozenset[DecayCategory] = frozenset(
    {
        DecayCategory.STALE_CONSUMER,
        DecayCategory.UNAUTHORIZED_OVERRIDE,
        DecayCategory.CONTRADICTORY_DECLARATIONS,
    }
)

#: Categories that assert the corpus cannot decide. Not a defect in any one artifact.
KNOWN_AMBIGUITY: frozenset[DecayCategory] = frozenset(
    {DecayCategory.AMBIGUOUS_AUTHORITY, DecayCategory.CONSUMER_DIVERGENCE}
)

#: Categories that assert nobody wrote something down. Never a contradiction.
MISSING_KNOWLEDGE: frozenset[DecayCategory] = frozenset(
    {
        DecayCategory.REGISTRY_GAP,
        DecayCategory.SINGLE_HOLDER,
        DecayCategory.UNWRITTEN_RULE,
    }
)
