"""AssumptionRecord, CouplingSet, AssumptionDelta.

The AssumptionRecord is the core unit of the mechanism: a latent consequence exists because
two artifacts independently bind the same proposition without referencing each other.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.value_objects.enums import (
    AuthorityClass,
    BindingRole,
    ClaimType,
    CouplingOutcome,
    DeltaAssertedBy,
    DeltaKind,
    Derivation,
    Explicitness,
    Support,
)
from app.domain.value_objects.identifiers import (
    AssumptionId,
    AssumptionKey,
    DeltaId,
    EvidenceId,
    SourceVersionId,
    TenantId,
    VersionLabel,
)
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.typed_value import TypedValue


class AssumptionRecord(TimestampedModel):
    """Ontology type B: a typed atomic proposition an artifact binds.

    Two field pairs carry the design decisions that an earlier draft got wrong:

    * ``scope`` - without it, a legitimately scoped override compares as a contradiction.
    * ``role`` and ``explicitness`` - split from a single ``binding_kind``, because
      dependency direction and textual explicitness are orthogonal and only the first
      decides whether a divergence is a contradiction.

    There is deliberately **no** ``extraction_confidence``. It was an LLM-supplied number
    wearing a carve-out; the extractor now emits ``explicitness`` (an enum it can justify)
    and code computes ``extraction_quality`` from the verification flags.
    """

    # Literals, not defaults: a type B record that could declare itself a hypothesis - or
    # claim `derived` support it never had - would defeat the point of the ontology. The
    # axes are facts about *which variant this is*, so they are unwritable.
    claim_type: Literal[ClaimType.ASSUMPTION_BINDING] = ClaimType.ASSUMPTION_BINDING
    derivation: Literal[Derivation.EXTRACTED] = Derivation.EXTRACTED
    support: Literal[Support.VERIFIED_SPAN] = Support.VERIFIED_SPAN

    assumption_id: AssumptionId
    tenant_id: TenantId

    key: AssumptionKey
    registry_version: VersionLabel
    scope: ScopeSelector = Field(default_factory=ScopeSelector.global_scope)
    value: TypedValue

    role: BindingRole
    explicitness: Explicitness

    holder_source_version_id: SourceVersionId
    #: Required. A record without it is invalid and must not persist.
    evidence_id: EvidenceId

    effective_from: dt.datetime | None = None
    effective_to: dt.datetime | None = None

    extractor_version: VersionLabel

    # Admission results, set by the deterministic gate - never by the extractor.
    span_verified: bool = False
    value_parsed: bool = False
    key_resolved: bool = False

    @model_validator(mode="after")
    def _interval_ordered(self) -> Self:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise ValueError(
                f"assumption {self.assumption_id}: effective_from must precede effective_to"
            )
        return self

    @property
    def extraction_quality(self) -> float:
        """Computed from the verification flags. Never model-supplied.

        Does not propagate directly into consequence confidence; it gates *admission*.
        """
        checks = (self.span_verified, self.value_parsed, self.key_resolved)
        return sum(checks) / len(checks)

    @property
    def is_admitted(self) -> bool:
        """All three admission checks passed, so the record may enter the coupling index.

        A record failing any check is quarantined, never indexed.
        """
        return self.span_verified and self.value_parsed and self.key_resolved

    def is_in_force(self, as_of: dt.datetime) -> bool:
        if self.effective_from is not None and as_of < self.effective_from:
            return False
        return not (self.effective_to is not None and as_of >= self.effective_to)


class CouplingHolder(DomainModel):
    """One member of a coupling set, with the authority needed for the override test."""

    source_version_id: SourceVersionId
    assumption_id: AssumptionId
    value: TypedValue
    role: BindingRole
    authority_class: AuthorityClass
    scope: ScopeSelector = Field(default_factory=ScopeSelector.global_scope)


class CouplingSet(DomainModel):
    """A deterministic projection. Never written by an agent.

    Identity is the triple ``(key, scope, as_of)``. Keying on ``key`` alone was the defect
    behind the removed ``has_collision := distinct_values > 1`` rule, which was both
    scope-blind and authority-blind and would report every legitimate override as a
    contradiction.
    """

    key: AssumptionKey
    scope: ScopeSelector = Field(default_factory=ScopeSelector.global_scope)
    as_of: dt.datetime

    holders: tuple[CouplingHolder, ...] = ()
    outcome: CouplingOutcome = CouplingOutcome.CONSISTENT

    authority_model_version: VersionLabel
    coupling_index_version: VersionLabel
    computed_at: dt.datetime

    @property
    def holder_count(self) -> int:
        return len(self.holders)

    @property
    def declarers(self) -> tuple[CouplingHolder, ...]:
        return tuple(h for h in self.holders if h.role is BindingRole.DECLARES)

    @property
    def consumers(self) -> tuple[CouplingHolder, ...]:
        return tuple(h for h in self.holders if h.role is BindingRole.CONSUMES)

    @property
    def declarer_count(self) -> int:
        return len(self.declarers)

    @property
    def distinct_declared_values(self) -> int:
        return len({h.value.render() for h in self.declarers})

    @property
    def is_single_holder(self) -> bool:
        """Undocumented coupling: exactly one artifact knows this.

        Feeds the knowledge-decay report and the blind-spot partition.
        """
        return self.holder_count == 1

    @property
    def is_unwritten_rule(self) -> bool:
        """Bound by config or code but by no policy or SOP."""
        authorities = {h.authority_class for h in self.holders}
        governing = {
            AuthorityClass.REGULATORY,
            AuthorityClass.POLICY,
            AuthorityClass.STANDARD,
            AuthorityClass.SOP_RUNBOOK,
        }
        return bool(authorities) and not (authorities & governing)


class AssumptionDelta(DomainModel):
    """How a ChangeSpec is expressed against the assumption store.

    Carries **no** ``confidence``: it is produced by the change-decomposer, an LLM.
    ``evidence_ids`` cites spans of the change request itself, since the decomposer runs
    before retrieval and has no corpus evidence to cite.
    """

    delta_id: DeltaId
    key: AssumptionKey
    scope: ScopeSelector = Field(default_factory=ScopeSelector.global_scope)
    old_value: TypedValue | None = None
    new_value: TypedValue | None = None
    delta_kind: DeltaKind
    asserted_by: DeltaAssertedBy
    evidence_ids: tuple[EvidenceId, ...] = ()
    registry_version: VersionLabel
    note: Annotated[str, StringConstraints(max_length=512)] | None = None

    @model_validator(mode="after")
    def _values_match_delta_kind(self) -> Self:
        if self.delta_kind is DeltaKind.VALUE_CHANGE:
            if self.old_value is None or self.new_value is None:
                raise ValueError(
                    f"delta {self.delta_id}: value_change requires both old_value and new_value"
                )
            if self.old_value.render() == self.new_value.render():
                raise ValueError(
                    f"delta {self.delta_id}: value_change with identical values is not a change"
                )
        elif self.delta_kind is DeltaKind.KEY_ADDED:
            if self.new_value is None:
                raise ValueError(f"delta {self.delta_id}: key_added requires new_value")
            if self.old_value is not None:
                raise ValueError(f"delta {self.delta_id}: key_added must not carry old_value")
        elif self.delta_kind is DeltaKind.KEY_REMOVED:
            if self.old_value is None:
                raise ValueError(f"delta {self.delta_id}: key_removed requires old_value")
            if self.new_value is not None:
                raise ValueError(f"delta {self.delta_id}: key_removed must not carry new_value")
        return self
