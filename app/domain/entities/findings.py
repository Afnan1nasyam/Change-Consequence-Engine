"""Specialist findings, challenge, arbitration, scoring, consequences, dispositions.

Two invariants carry most of the weight in this module:

* **Confidence appears on exactly one entity, :class:`ConsequenceScore`, and is computed.**
  Every other model here is agent-authored, and ``extra="forbid"`` means an agent that
  emits a ``confidence`` key is rejected rather than silently ignored.
* **An uncited objection is unrepresentable.** :class:`ChallengeResult` enforces
  per-outcome evidence requirements, so ``refuted`` without a citation is a validation
  error, and ``no_refutation_found`` costs a falsification test.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.value_objects.enums import (
    ASSERTIVE_STATUSES,
    STATUSES_ABOVE_SPECULATIVE,
    SUPPRESSING_DISPOSITIONS,
    ChallengeOutcome,
    ClassAGround,
    ConsequenceClass,
    Disposition,
    EvidencePosition,
    Origin,
    Severity,
    Status,
)
from app.domain.value_objects.identifiers import (
    AnalysisId,
    AssumptionKey,
    ChallengeId,
    ChangeId,
    ConsequenceId,
    DispositionId,
    EventId,
    EvidenceId,
    FindingId,
    GapId,
    HypothesisId,
    RelationId,
    SourceVersionId,
    TestId,
    VersionLabel,
)
from app.domain.value_objects.provenance import ModelProvenance, RunProvenance

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
Statement = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


# ---------------------------------------------------------------------------
# Specialist findings
# ---------------------------------------------------------------------------


class SpecialistFinding(TimestampedModel):
    """One specialist's output over its own evidence slice.

    ``evidence_position`` replaces ``confidence``: an enum the agent can defend, rather
    than a number it cannot. ``evidence_gap_ids`` replaces the old free-text ``unknowns``,
    which could not distinguish searched-and-absent from never-searched.
    """

    finding_id: FindingId
    agent: ShortText
    claim: Statement
    category: ShortText
    affected_entities: tuple[ShortText, ...] = ()
    evidence_ids: tuple[EvidenceId, ...] = ()
    evidence_position: EvidencePosition
    assumption_keys: tuple[AssumptionKey, ...] = ()
    evidence_gap_ids: tuple[GapId, ...] = ()
    produced_by: ModelProvenance

    @model_validator(mode="after")
    def _position_requires_evidence(self) -> Self:
        """A supported position needs something supporting it."""
        needs_evidence = {EvidencePosition.SUPPORTED, EvidencePosition.WEAKLY_SUPPORTED}
        if self.evidence_position in needs_evidence and not self.evidence_ids:
            raise ValueError(
                f"finding {self.finding_id}: evidence_position="
                f"{self.evidence_position.value!r} requires at least one evidence id"
            )
        return self


# ---------------------------------------------------------------------------
# Challenge
# ---------------------------------------------------------------------------


class FalsificationTest(DomainModel):
    """What observation would refute the claim.

    The price of ``no_refutation_found``. It stops that outcome being a lazy default while
    equally refusing to punish it - and the test feeds the validation checklist as a real
    deliverable.
    """

    test_id: TestId
    consequence_id: ConsequenceId
    observation: Statement
    data_source_needed: ShortText | None = None
    feeds_validation_checklist: bool = True


class ChallengeResult(TimestampedModel):
    """Exactly three outcomes, with per-outcome evidence requirements.

    A model instructed merely to disagree produces plausible objections at whatever rate it
    is asked for them. Making the citation a *schema* requirement rather than a prompt
    request is what turns adversarial review into a proof obligation.
    """

    challenge_id: ChallengeId
    consequence_id: ConsequenceId
    outcome: ChallengeOutcome

    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()
    contradicts_what: Statement | None = None
    scope_narrowing: Statement | None = None
    falsification_test_id: TestId | None = None

    #: Required when challenging a Class A consequence. A deterministic derivation is only
    #: ever rule-wrong or input-wrong, so "I do not think that follows" is inadmissible.
    class_a_ground: ClassAGround | None = None

    produced_by: ModelProvenance

    @model_validator(mode="after")
    def _outcome_requirements(self) -> Self:
        if self.outcome is ChallengeOutcome.REFUTED:
            if not self.contradicting_evidence_ids:
                raise ValueError(
                    f"challenge {self.challenge_id}: outcome='refuted' requires at least one "
                    "contradicting evidence id. An uncited objection is not representable."
                )
            if not self.contradicts_what:
                raise ValueError(
                    f"challenge {self.challenge_id}: outcome='refuted' requires contradicts_what"
                )
        elif self.outcome is ChallengeOutcome.WEAKENED:
            if not self.contradicting_evidence_ids:
                raise ValueError(
                    f"challenge {self.challenge_id}: outcome='weakened' requires at least one "
                    "evidence id for the control, exception, or newer version"
                )
            if not self.scope_narrowing:
                raise ValueError(
                    f"challenge {self.challenge_id}: outcome='weakened' requires scope_narrowing"
                )
        else:  # NO_REFUTATION_FOUND
            if self.contradicting_evidence_ids:
                raise ValueError(
                    f"challenge {self.challenge_id}: outcome='no_refutation_found' must carry "
                    "no evidence ids"
                )
            if self.falsification_test_id is None:
                raise ValueError(
                    f"challenge {self.challenge_id}: outcome='no_refutation_found' requires a "
                    "falsification_test_id. It is a legitimate outcome, but it costs a test."
                )
            if self.contradicts_what or self.scope_narrowing:
                raise ValueError(
                    f"challenge {self.challenge_id}: outcome='no_refutation_found' must not "
                    "assert what it contradicts or narrows"
                )
        return self


# ---------------------------------------------------------------------------
# Arbitration and scoring
# ---------------------------------------------------------------------------


class ArbitrationDecision(TimestampedModel):
    """The judge's disposition: status and severity, never confidence.

    The asymmetry is deliberate. Confidence is a function of evidence properties the system
    can measure, so it is computed. Severity is a function of business impact the corpus
    does not contain, so it is a judgment and it is the judge's.
    """

    consequence_id: ConsequenceId
    status: Status
    severity: Severity
    rationale: Statement
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()
    preserved_disagreements: tuple[ShortText, ...] = ()
    validation_actions: tuple[ShortText, ...] = ()
    in_default_view: bool = False
    produced_by: ModelProvenance

    @model_validator(mode="after")
    def _assertive_status_requires_evidence(self) -> Self:
        if self.status in ASSERTIVE_STATUSES and not self.supporting_evidence_ids:
            raise ValueError(
                f"arbitration for {self.consequence_id}: status={self.status.value!r} "
                "requires supporting evidence. Absence of contrary evidence is not support."
            )
        return self

    @model_validator(mode="after")
    def _unresolved_high_risk_has_an_action(self) -> Self:
        if (
            self.status is Status.UNRESOLVED
            and self.severity in (Severity.HIGH, Severity.CRITICAL)
            and not self.validation_actions
        ):
            raise ValueError(
                f"arbitration for {self.consequence_id}: an unresolved "
                f"{self.severity.value} finding requires at least one validation action "
                "naming what would settle it"
            )
        return self


class ConsequenceScore(DomainModel):
    """Computed confidence. The only entity in the domain that carries one.

    ``corroborating_source_version_ids`` rather than a count of agreeing agents is the
    whole point: several agents reading the same evidence are one source. The scorer's
    input deliberately contains no agent identity, so it *cannot* count agents.
    """

    consequence_id: ConsequenceId
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_fn_version: VersionLabel
    factor_inputs: dict[str, float] = Field(default_factory=dict)
    corroborating_source_version_ids: tuple[SourceVersionId, ...] = ()
    corroborating_strategies: tuple[str, ...] = ()
    computed_at: dt.datetime

    @property
    def corroboration_count(self) -> int:
        """Distinct source versions, never agent votes."""
        return len(set(self.corroborating_source_version_ids))

    @model_validator(mode="after")
    def _factor_inputs_are_persisted(self) -> Self:
        """A score with no recorded inputs cannot be explained or recomputed."""
        if not self.factor_inputs:
            raise ValueError(
                f"score for {self.consequence_id}: factor_inputs must be persisted so the "
                "score can be recomputed and explained after the fact"
            )
        return self


# ---------------------------------------------------------------------------
# Consequence
# ---------------------------------------------------------------------------


class Consequence(TimestampedModel):
    """The arbitrated finding.

    ``consequence_class`` is **pipeline-assigned** and derived from ``origin``:
    Class A means the deterministic resolver produced it. Classification is by producer,
    not by subject matter - sounding derivable is not a derivation.

    Carries no ``confidence`` (see :class:`ConsequenceScore`) and no ``likelihood``, which
    was removed as an undefined probability by another name.
    """

    consequence_id: ConsequenceId
    change_id: ChangeId
    title: ShortText
    description: Statement
    category: ShortText

    consequence_class: ConsequenceClass
    origin: Origin
    affected_entity: ShortText
    mechanism: Statement

    derived_relation_ids: tuple[RelationId, ...] = ()
    hypothesis_ids: tuple[HypothesisId, ...] = ()
    merged_hypothesis_ids: tuple[HypothesisId, ...] = ()

    status: Status
    severity: Severity

    evidence_ids: tuple[EvidenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()
    assumption_keys: tuple[AssumptionKey, ...] = ()
    evidence_gap_ids: tuple[GapId, ...] = ()
    historical_analogue_id: EventId | None = None

    recommended_validation: tuple[ShortText, ...] = ()
    agent_sources: tuple[ShortText, ...] = ()
    run_provenance: RunProvenance

    @model_validator(mode="after")
    def _class_matches_origin(self) -> Self:
        """Class A iff the resolver produced it (checkpoint 7.2)."""
        expected = (
            ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED
            if self.origin is Origin.RESOLVER
            else ConsequenceClass.CLASS_B_BEHAVIORAL
        )
        if self.consequence_class is not expected:
            raise ValueError(
                f"consequence {self.consequence_id}: origin={self.origin.value!r} implies "
                f"class {expected.value!r}, got {self.consequence_class.value!r}. "
                "Classification is by producer, not by subject matter."
            )
        return self

    @model_validator(mode="after")
    def _class_a_is_derived(self) -> Self:
        if self.consequence_class is ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED:
            if not self.derived_relation_ids:
                raise ValueError(
                    f"consequence {self.consequence_id}: Class A requires at least one "
                    "derived_relation_id - it is defined by having been derived"
                )
            if not self.assumption_keys:
                raise ValueError(
                    f"consequence {self.consequence_id}: Class A is assumption-mediated and "
                    "requires at least one assumption key"
                )
        return self

    @model_validator(mode="after")
    def _class_b_is_hypothesised(self) -> Self:
        if self.consequence_class is ConsequenceClass.CLASS_B_BEHAVIORAL:
            if not self.hypothesis_ids:
                raise ValueError(
                    f"consequence {self.consequence_id}: Class B requires at least one "
                    "hypothesis_id"
                )
            if self.derived_relation_ids:
                raise ValueError(
                    f"consequence {self.consequence_id}: Class B must not claim a derived "
                    "relation; if it were derived it would be Class A"
                )
        return self

    @model_validator(mode="after")
    def _assertive_status_requires_evidence(self) -> Self:
        if self.status in ASSERTIVE_STATUSES and not self.evidence_ids:
            raise ValueError(
                f"consequence {self.consequence_id}: status={self.status.value!r} requires "
                "evidence. With none, the only honest statuses are speculative or unresolved."
            )
        return self

    @model_validator(mode="after")
    def _class_b_above_speculative_needs_an_analogue(self) -> Self:
        if (
            self.consequence_class is ConsequenceClass.CLASS_B_BEHAVIORAL
            and self.status in STATUSES_ABOVE_SPECULATIVE
            and self.historical_analogue_id is None
        ):
            raise ValueError(
                f"consequence {self.consequence_id}: a Class B finding at status="
                f"{self.status.value!r} requires a historical analogue with an observed "
                "outcome. Without one it caps at 'speculative'."
            )
        return self

    @model_validator(mode="after")
    def _merged_hypotheses_only_on_class_a(self) -> Self:
        """Class B hypotheses merge *into* a Class A finding, never the reverse."""
        if (
            self.merged_hypothesis_ids
            and self.consequence_class is not ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED
        ):
            raise ValueError(
                f"consequence {self.consequence_id}: merged_hypothesis_ids is only valid on a "
                "Class A finding that absorbed a rediscovered Class B hypothesis"
            )
        return self


# ---------------------------------------------------------------------------
# Human feedback
# ---------------------------------------------------------------------------


class FindingDisposition(TimestampedModel):
    """A reviewer's verdict on one finding.

    The primary feedback signal, in preference to incident matching: cheap, immediate,
    unambiguous. ``already_known`` is the key product signal - a high rate means the system
    is finding what a diligent reviewer would find anyway.
    """

    disposition_id: DispositionId
    consequence_id: ConsequenceId
    analysis_id: AnalysisId
    assumption_key: AssumptionKey | None = None
    artifact_ref: ShortText | None = None
    disposition: Disposition
    note: ShortText | None = None
    disposed_by: ShortText
    disposed_at: dt.datetime
    suppresses_until_source_revision: bool = False

    @model_validator(mode="after")
    def _suppression_needs_a_scope(self) -> Self:
        """Suppression must name what it suppresses, or it silences too much."""
        if self.suppresses_until_source_revision and not (self.assumption_key or self.artifact_ref):
            raise ValueError(
                f"disposition {self.disposition_id}: suppression requires an assumption_key "
                "or artifact_ref to scope it"
            )
        return self

    @model_validator(mode="after")
    def _only_negative_dispositions_suppress(self) -> Self:
        if (
            self.suppresses_until_source_revision
            and self.disposition not in SUPPRESSING_DISPOSITIONS
        ):
            raise ValueError(
                f"disposition {self.disposition_id}: only "
                f"{sorted(d.value for d in SUPPRESSING_DISPOSITIONS)} may suppress, got "
                f"{self.disposition.value!r}"
            )
        return self


class AnalysisReport(DomainModel):
    """Composed output. Introduces no new factual claims.

    ``suppressed_count`` is reported rather than hidden: alert load is a correctness
    constraint, and a reader who cannot see what was withheld cannot calibrate trust.
    """

    analysis_id: AnalysisId
    change_id: ChangeId
    executive_summary: Statement
    consequence_ids: tuple[ConsequenceId, ...] = ()
    default_view_consequence_ids: tuple[ConsequenceId, ...] = ()
    contradiction_relation_ids: tuple[RelationId, ...] = ()
    evidence_gap_ids: tuple[GapId, ...] = ()
    unresolved_questions: tuple[ShortText, ...] = ()
    validation_checklist: tuple[ShortText, ...] = ()
    suppressed_count: int = Field(default=0, ge=0)
    run_provenance: RunProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_view_is_a_subset(self) -> Self:
        stray = set(self.default_view_consequence_ids) - set(self.consequence_ids)
        if stray:
            raise ValueError(
                f"analysis {self.analysis_id}: default view references consequences absent "
                f"from the report: {sorted(stray)}"
            )
        return self
