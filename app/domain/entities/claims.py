"""The five-part epistemic ontology as a discriminated union.

Types A, C, D and E live here; type B (:class:`AssumptionRecord`) lives in
``assumption.py`` because it is also the core mechanism unit.

The evidence rules per type are the point of this module:

===== ==================== ==========================================================
type  evidence_ids         additional mandatory provenance
===== ==================== ==========================================================
A     >= 1 required        verified span offsets
B     exactly 1 required   registry key + version, extractor version
C     **must be empty**    rule_id + non-empty input_record_ids
D     may be empty         mechanism; analogue id to exceed `speculative`
E     **must be empty**    negative provenance: strategies searched, as_of
===== ==================== ==========================================================

C carrying no evidence of its own is not an oversight. Demanding an ``evidence_id`` on a
derived relation would force the resolver to nominate an arbitrary input span and
re-present it as direct support - manufacturing exactly the false provenance this project
exists to prevent. Its evidence set is the computed transitive union of its inputs.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.value_objects.enums import ClaimType, Derivation, Support
from app.domain.value_objects.identifiers import (
    AnalysisId,
    AssumptionId,
    AssumptionKey,
    EventId,
    EvidenceId,
    FactId,
    GapId,
    HypothesisId,
    RelationId,
)
from app.domain.value_objects.provenance import DerivationProvenance, ModelProvenance
from app.domain.value_objects.taxonomy import DERIVED_RELATIONSHIPS, Relationship

Statement = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]


# ---------------------------------------------------------------------------
# Type A - ObservedFact
# ---------------------------------------------------------------------------


class ObservedFact(TimestampedModel):
    """A source says this. Wrong only if the span was misread."""

    claim_type: Literal[ClaimType.OBSERVED_FACT] = ClaimType.OBSERVED_FACT
    derivation: Literal[Derivation.OBSERVED] = Derivation.OBSERVED

    fact_id: FactId
    statement: Statement
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    support: Support = Support.VERIFIED_SPAN

    @model_validator(mode="after")
    def _support_matches_evidence_count(self) -> Self:
        if self.support not in (Support.VERIFIED_SPAN, Support.MULTI_SPAN):
            raise ValueError(
                f"ObservedFact support must be verified_span or multi_span, "
                f"got {self.support.value!r}"
            )
        if self.support is Support.MULTI_SPAN and len(self.evidence_ids) < 2:
            raise ValueError("support='multi_span' requires at least two evidence ids")
        return self


# ---------------------------------------------------------------------------
# Type C - DerivedRelation
# ---------------------------------------------------------------------------


class DerivedRelation(TimestampedModel):
    """The system computed this from >= 2 records by a named rule.

    Wrong only if the rule is wrong or its inputs are wrong - which is why a challenger
    may not refute one on mechanism grounds.
    """

    claim_type: Literal[ClaimType.DERIVED_RELATION] = ClaimType.DERIVED_RELATION
    derivation: Literal[Derivation.DERIVED] = Derivation.DERIVED
    support: Literal[Support.RULE_OVER_RECORDS] = Support.RULE_OVER_RECORDS

    relation_id: RelationId
    relationship: Relationship
    subject_ref: ShortText
    object_ref: ShortText

    input_record_ids: tuple[AssumptionId, ...] = Field(min_length=1)
    provenance: DerivationProvenance

    #: Structurally empty. See the module docstring.
    evidence_ids: tuple[EvidenceId, ...] = ()

    @model_validator(mode="after")
    def _carries_no_direct_evidence(self) -> Self:
        if self.evidence_ids:
            raise ValueError(
                f"DerivedRelation {self.relation_id} must not carry evidence_ids "
                f"(got {len(self.evidence_ids)}). Its evidence is the computed transitive "
                "union of input_record_ids; a direct citation here would be fabricated "
                "provenance."
            )
        return self

    @model_validator(mode="after")
    def _relationship_is_derivable(self) -> Self:
        if self.relationship not in DERIVED_RELATIONSHIPS:
            raise ValueError(
                f"DerivedRelation cannot express {self.relationship.value!r}; "
                f"derivable relationships are "
                f"{sorted(r.value for r in DERIVED_RELATIONSHIPS)}"
            )
        return self

    @model_validator(mode="after")
    def _override_records_authority_comparison(self) -> Self:
        """An unauthorized-override finding must show its arithmetic.

        The checkpoint requires the authority comparison to be *recorded*, not merely
        applied, so the finding can be audited without re-running the resolver.
        """
        if self.relationship is Relationship.UNAUTHORIZED_OVERRIDE:
            if self.provenance.authority_model_version is None:
                raise ValueError(
                    "unauthorized_override requires provenance.authority_model_version"
                )
            required = {"narrower_authority_class", "broader_authority_class"}
            missing = required - set(self.provenance.detail)
            if missing:
                raise ValueError(
                    f"unauthorized_override requires provenance.detail keys {sorted(missing)}; "
                    "the authority comparison must be recorded, not just applied"
                )
        return self


# ---------------------------------------------------------------------------
# Type D - Hypothesis
# ---------------------------------------------------------------------------


class ConsequenceHypothesis(TimestampedModel):
    """A model proposes this; it needs validation.

    Carries no ``confidence``, no ``severity`` and no ``consequence_class``: all three are
    decided downstream, and ``consequence_class`` is pipeline-assigned so that no agent can
    self-declare Class A.

    ``evidence_ids`` may legitimately be empty. A hypothesis with no evidence is not a
    defect; a hypothesis *labelled as established* is.
    """

    claim_type: Literal[ClaimType.HYPOTHESIS] = ClaimType.HYPOTHESIS
    derivation: Literal[Derivation.INFERRED] = Derivation.INFERRED
    support: Literal[Support.MECHANISM_ONLY] = Support.MECHANISM_ONLY

    hypothesis_id: HypothesisId
    claim: Statement
    mechanism: Statement
    #: The link in the chain the author considers least supported. Required, because a
    #: mechanism with no acknowledged weak point is usually an unexamined one.
    weakest_link: ShortText

    affected_entities: tuple[ShortText, ...] = ()
    assumption_keys: tuple[AssumptionKey, ...] = ()
    evidence_ids: tuple[EvidenceId, ...] = ()
    evidence_gap_ids: tuple[GapId, ...] = ()

    #: Required for the consequence to exceed `speculative` (checkpoint 9.2).
    historical_analogue_id: EventId | None = None

    proposed_by: ModelProvenance


# ---------------------------------------------------------------------------
# Type E - EvidenceGap
# ---------------------------------------------------------------------------


class EvidenceGap(TimestampedModel):
    """The corpus cannot establish this.

    A claim **about the corpus**, not about the world - which is why it can never become a
    graph edge. Inventing a relationship out of silence is the structural form of treating
    absence of evidence as evidence of absence.

    The negative provenance is mandatory: without it, a gap is indistinguishable from
    nobody having looked.
    """

    claim_type: Literal[ClaimType.EVIDENCE_GAP] = ClaimType.EVIDENCE_GAP
    derivation: Literal[Derivation.ABSENT] = Derivation.ABSENT
    support: Literal[Support.NEGATIVE_PROVENANCE] = Support.NEGATIVE_PROVENANCE

    gap_id: GapId
    analysis_id: AnalysisId
    question: Statement

    strategies_executed: tuple[ShortText, ...] = Field(min_length=1)
    source_types_searched: tuple[ShortText, ...] = Field(min_length=1)
    as_of: dt.datetime
    would_be_resolved_by: Statement

    evidence_ids: tuple[EvidenceId, ...] = ()
    partition: ShortText | None = None

    @model_validator(mode="after")
    def _carries_no_evidence(self) -> Self:
        if self.evidence_ids:
            raise ValueError(
                f"EvidenceGap {self.gap_id} must not carry evidence_ids: a gap is the "
                "recorded absence of evidence"
            )
        return self


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------

#: Types A, C, D, E. Type B (AssumptionRecord) is imported by consumers that need it; it is
#: excluded here to avoid a circular import and because it is an indexed store record
#: rather than an analysis-time claim.
Claim = Annotated[
    ObservedFact | DerivedRelation | ConsequenceHypothesis | EvidenceGap,
    Field(discriminator="claim_type"),
]


class ClaimEnvelope(DomainModel):
    """Wrapper for round-tripping a heterogeneous claim list through the discriminator."""

    claim: Claim
