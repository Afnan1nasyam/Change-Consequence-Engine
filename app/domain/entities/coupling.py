"""Coupling read-models: the resolved binding, authority resolution, and what they emit.

Nothing here is authored. Every object in this module is a deterministic projection of the
assumption store, and each carries the version labels needed to explain why it says what it
says.

The central type is :class:`ResolvedBinding`. It exists so the coupling engine can be a
**pure function of its inputs**: authority is a property of the holder *document*
(checkpoint 3.5a), not of the binding, so somebody has to join the two. Doing that join
once, in the application layer, keeps the collision matrix free of store access and makes
every rule in it testable by constructing values rather than by populating a database.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.assumption import AssumptionRecord
from app.domain.entities.base import DomainModel
from app.domain.value_objects.enums import (
    AUTHORITY_RANK,
    AmbiguityReason,
    AnomalyKind,
    AuthorityClass,
    AuthorityOutcome,
    BindingRole,
    Explicitness,
)
from app.domain.value_objects.identifiers import (
    AssumptionId,
    AssumptionKey,
    EvidenceId,
    SourceId,
    SourceVersionId,
    VersionLabel,
)
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.typed_value import TypedValue

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class ResolvedBinding(DomainModel):
    """An admitted binding joined to the authority of the document that holds it.

    ``authority_class`` is copied from :class:`SourceDocument` at join time, never from the
    binding and never from document content. A document that could assert its own authority
    could plant itself above a real policy, which is why the field is not part of the
    7-tuple in the first place (``docs/SECURITY.md``).
    """

    record: AssumptionRecord
    authority_class: AuthorityClass
    source_id: SourceId
    source_title: ShortText

    @model_validator(mode="after")
    def _record_is_admitted(self) -> Self:
        """Only admitted bindings may be coupled.

        The store already refuses unadmitted records, so this is a second gate on the path
        that matters most: a quarantined binding reaching the coupling engine would produce
        a Class A finding from unverified input, which is the one thing Class A promises
        cannot happen.
        """
        if not self.record.is_admitted:
            raise ValueError(
                f"binding {self.record.assumption_id} has not passed all three admission "
                "checks and must not enter the coupling index"
            )
        return self

    # -- delegation, so the collision matrix reads like the checkpoint ----

    @property
    def assumption_id(self) -> AssumptionId:
        return self.record.assumption_id

    @property
    def key(self) -> AssumptionKey:
        return self.record.key

    @property
    def scope(self) -> ScopeSelector:
        return self.record.scope

    @property
    def value(self) -> TypedValue:
        return self.record.value

    @property
    def role(self) -> BindingRole:
        return self.record.role

    @property
    def explicitness(self) -> Explicitness:
        return self.record.explicitness

    @property
    def source_version_id(self) -> SourceVersionId:
        return self.record.holder_source_version_id

    @property
    def evidence_id(self) -> EvidenceId:
        return self.record.evidence_id

    @property
    def interval(self) -> tuple[dt.datetime | None, dt.datetime | None]:
        return (self.record.effective_from, self.record.effective_to)

    @property
    def authority_rank(self) -> int:
        return AUTHORITY_RANK[self.authority_class]

    def is_in_force(self, as_of: dt.datetime) -> bool:
        return self.record.is_in_force(as_of)

    def outranks_or_matches(self, other: ResolvedBinding) -> bool:
        """The override legitimacy test: ``self.authority >= other.authority``.

        Checkpoint 5.4.2 / 5.4.3 and 13.6a. Deliberately ``>=``: a peer-authority
        narrowing is a legitimate specialization, not a governance defect.
        """
        return self.authority_rank >= other.authority_rank


class AuthorityCandidate(DomainModel):
    """One declarer considered by ``resolve_authority``, with why it was or was not chosen."""

    binding: ResolvedBinding
    #: True for the candidate that governs; false for every other considered candidate.
    selected: bool = False
    #: Set when the candidate was skipped because it narrows a higher authority.
    unauthorized_narrowing_of: AssumptionId | None = None


class AuthorityResolution(DomainModel):
    """The result of ``resolve_authority(key, scope, as_of)``.

    ``AMBIGUOUS`` is a first-class outcome, not an exception and not a hidden fallback.
    Newest-wins is forbidden: choosing the most recent row among equally authoritative
    declarers would invent an authority the corpus does not establish, and would do it
    invisibly (checkpoint 5.3).

    ``NO_DECLARER`` is likewise not an error. A key nothing declares is a *gap*, and a gap
    is a claim about the corpus rather than about the world.
    """

    key: AssumptionKey
    scope: ScopeSelector
    as_of: dt.datetime
    outcome: AuthorityOutcome

    #: The governing binding. Present iff ``outcome is RESOLVED``.
    authoritative: ResolvedBinding | None = None
    candidates: tuple[AuthorityCandidate, ...] = ()

    ambiguity_reason: AmbiguityReason | None = None
    #: Narrower bindings that tried to override a higher authority. Flagged, never applied,
    #: and never discarded (checkpoint 5.4.3).
    unauthorized_narrowings: tuple[ResolvedBinding, ...] = ()

    authority_model_version: VersionLabel
    coupling_index_version: VersionLabel

    @model_validator(mode="after")
    def _outcome_matches_payload(self) -> Self:
        if self.outcome is AuthorityOutcome.RESOLVED and self.authoritative is None:
            raise ValueError(f"authority for {self.key}@{self.scope}: resolved with no winner")
        if self.outcome is not AuthorityOutcome.RESOLVED and self.authoritative is not None:
            raise ValueError(
                f"authority for {self.key}@{self.scope}: outcome={self.outcome.value!r} "
                "must not name an authoritative binding. Reporting a winner for an "
                "ambiguous key is exactly the newest-wins failure the checkpoint forbids."
            )
        if self.outcome is AuthorityOutcome.AMBIGUOUS and self.ambiguity_reason is None:
            raise ValueError(
                f"authority for {self.key}@{self.scope}: ambiguity must state its reason, "
                "because each reason maps to a different human action"
            )
        if self.outcome is not AuthorityOutcome.AMBIGUOUS and self.ambiguity_reason is not None:
            raise ValueError(
                f"authority for {self.key}@{self.scope}: ambiguity_reason is only "
                f"meaningful when the outcome is ambiguous, got {self.outcome.value!r}"
            )
        return self

    @property
    def is_resolved(self) -> bool:
        return self.outcome is AuthorityOutcome.RESOLVED

    @property
    def authoritative_value(self) -> TypedValue | None:
        return self.authoritative.value if self.authoritative else None

    @property
    def tied_candidates(self) -> tuple[ResolvedBinding, ...]:
        """The candidates that could not be separated. Empty unless ambiguous."""
        if self.outcome is not AuthorityOutcome.AMBIGUOUS:
            return ()
        return tuple(c.binding for c in self.candidates if c.unauthorized_narrowing_of is None)


class OverrideRecord(DomainModel):
    """A **legitimate** narrower override (checkpoint 5.4.2).

    Recorded rather than reported. It emits no finding and no edge, because it is correct
    behaviour - but a later invalidation pass needs to know it exists, so that a change to
    the broader scope reports ``scope_inheritance_affected`` for confirmation instead of
    treating the override as broken (checkpoint 5.5).
    """

    key: AssumptionKey
    narrower: ResolvedBinding
    broader: ResolvedBinding
    as_of: dt.datetime
    authority_model_version: VersionLabel

    @model_validator(mode="after")
    def _is_actually_legitimate(self) -> Self:
        if not self.narrower.outranks_or_matches(self.broader):
            raise ValueError(
                f"override of {self.key}: narrower authority "
                f"{self.narrower.authority_class.value!r} does not match or outrank broader "
                f"{self.broader.authority_class.value!r}; this is an unauthorized_override, "
                "not a legitimate one"
            )
        return self


class CouplingAnomaly(DomainModel):
    """A reportable observation that is deliberately **not** a Class A finding.

    Two ``consumes`` bindings disagreeing is the canonical case. Neither artifact is an
    authority, so neither can be said to be wrong; the checkpoint 3.3 matrix calls it a
    coupling anomaly requiring scope review. There is no edge for it in the seven-edge
    taxonomy and inventing one would recreate the ``potentially_impacts`` catch-all the
    taxonomy exists to exclude.
    """

    kind: AnomalyKind
    key: AssumptionKey
    as_of: dt.datetime
    participants: tuple[AssumptionId, ...] = Field(min_length=2)
    detail: ShortText
    coupling_index_version: VersionLabel


class CouplingStrength(DomainModel):
    """A deterministic, versioned score over evidence properties only.

    The input list is the whole point. ``factor_inputs`` cannot contain agent identity,
    agent count, model agreement, or a model-supplied number, because the function that
    produces it never receives them (checkpoint 5.7, 9.3). Corroboration counts sources;
    several agents reading one span are one source.

    ``blocked`` means a unit mismatch: strength is exactly 0 and the pair must not be
    coupled at all. That is a gate, not a low score (checkpoint 5.6).
    """

    strength: float = Field(ge=0.0, le=1.0)
    blocked: bool = False
    factor_inputs: dict[str, float] = Field(default_factory=dict)
    strength_fn_version: VersionLabel

    @model_validator(mode="after")
    def _blocked_means_zero(self) -> Self:
        if self.blocked and self.strength != 0.0:
            raise ValueError(
                f"blocked coupling must score exactly 0.0, got {self.strength}. A unit "
                "mismatch blocks coupling; it does not weaken it."
            )
        return self

    @model_validator(mode="after")
    def _inputs_are_persisted(self) -> Self:
        if not self.factor_inputs:
            raise ValueError(
                "coupling strength must persist its factor inputs so the score can be "
                "recomputed and explained after the fact"
            )
        return self
