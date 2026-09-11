"""The closed assumption key registry and its two gates.

These three entities implement the governing invariant: *an LLM may propose a semantic
relationship, but it can never promote that relationship into the trusted deterministic
layer by itself.*

- :class:`AssumptionKeyRegistryEntry` - the curated, versioned vocabulary.
- :class:`KeyRegistryGap` - where an unregistered key goes. Quarantined, never indexed.
- :class:`KeyAlias` - a proposed key equivalence, inert until a human approves it.

Neither gate has a bypass. There is no confidence threshold that promotes a proposal.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.value_objects.enums import (
    UNIT_REQUIRED_TYPES,
    AliasStatus,
    RegistryGapStatus,
    ValueType,
)
from app.domain.value_objects.identifiers import (
    AliasId,
    AssumptionKey,
    EvidenceId,
    RegistryGapId,
    SourceVersionId,
    VersionLabel,
)
from app.domain.value_objects.provenance import ModelProvenance
from app.domain.value_objects.typed_value import TypedValue, Unit

Description = Annotated[str, StringConstraints(min_length=1, max_length=512)]
Domain = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class AssumptionKeyRegistryEntry(DomainModel):
    """One curated key. Only registered keys participate in Class A coupling.

    The registry is **closed** for the MVP: precision and trust of Class A take priority
    over coverage. ``canonical_form`` carries polarity handling - "above 10000 requires
    approval" and "below 10000 needs none" must render to one canonical constraint, or
    coupling silently fails, which the data model calls the worst failure mode in the
    system.
    """

    key: AssumptionKey
    value_type: ValueType
    unit: Unit | None = None
    canonical_form: Description
    allowed_scopes: tuple[str, ...] = ()
    description: Description
    domain: Domain
    introduced_in_version: VersionLabel
    deprecated_in_version: VersionLabel | None = None

    @model_validator(mode="after")
    def _unit_consistent_with_type(self) -> Self:
        if self.value_type in UNIT_REQUIRED_TYPES and not self.unit:
            raise ValueError(
                f"registry entry {self.key!r}: unit is required for "
                f"value_type={self.value_type.value!r}"
            )
        if self.value_type not in UNIT_REQUIRED_TYPES and self.unit:
            raise ValueError(
                f"registry entry {self.key!r}: unit must be omitted for "
                f"value_type={self.value_type.value!r}"
            )
        return self

    @property
    def is_active(self) -> bool:
        return self.deprecated_in_version is None

    def permits_dimension(self, dimension: str) -> bool:
        """Whether a scope dimension is declared for this key.

        An empty ``allowed_scopes`` means global-only: a scope is validated against the
        registry, never invented by an extractor.
        """
        return dimension in self.allowed_scopes


class KeyRegistryGap(TimestampedModel):
    """An unregistered key proposed by extraction. Quarantined, never indexed.

    ``occurrence_count`` exists because gap **volume is an evaluation and product signal**
    telling us where the registry needs expansion - not an error log to be drained.
    """

    gap_id: RegistryGapId
    proposed_key: str = Field(min_length=1, max_length=256)
    proposed_value: TypedValue
    evidence_id: EvidenceId
    holder_source_version_id: SourceVersionId
    proposed_by: ModelProvenance
    as_of: dt.datetime
    registry_version: VersionLabel
    occurrence_count: int = Field(default=1, ge=1)
    status: RegistryGapStatus = RegistryGapStatus.OPEN

    @property
    def blocks_class_a(self) -> bool:
        """An open gap can never contribute to a Class A derivation."""
        return self.status is not RegistryGapStatus.REGISTERED


class KeyAlias(DomainModel):
    """A proposed equivalence between two keys.

    Aliases are the single point at which invented dependencies could enter the
    deterministic layer, so they are **gated, not scored**. Only ``APPROVED`` participates
    in Class A coupling; ``PROPOSED`` may appear in Class B analysis, marked as proposed.
    """

    alias_id: AliasId
    source_key: str = Field(min_length=1, max_length=256)
    target_key: AssumptionKey
    proposal_evidence: tuple[EvidenceId, ...] = ()
    proposed_by: ModelProvenance | None = None
    proposed_at: dt.datetime
    approved_by: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None
    approved_at: dt.datetime | None = None
    registry_version: VersionLabel
    status: AliasStatus = AliasStatus.PROPOSED
    rejection_reason: Description | None = None

    @model_validator(mode="after")
    def _not_self_alias(self) -> Self:
        if self.source_key == self.target_key:
            raise ValueError(f"alias source_key and target_key are identical: {self.source_key!r}")
        return self

    @model_validator(mode="after")
    def _approval_is_auditable(self) -> Self:
        """An approved alias must name a human and a time. Approval must be auditable."""
        if self.status is AliasStatus.APPROVED:
            missing = [
                name
                for name, val in (
                    ("approved_by", self.approved_by),
                    ("approved_at", self.approved_at),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"alias {self.alias_id}: status='approved' requires {missing}. "
                    "Approval must be auditable; there is no automatic approval path."
                )
        else:
            if self.approved_by is not None or self.approved_at is not None:
                raise ValueError(
                    f"alias {self.alias_id}: approval fields set while status={self.status.value!r}"
                )
        return self

    @model_validator(mode="after")
    def _rejection_gives_a_reason(self) -> Self:
        if self.status is AliasStatus.REJECTED and not self.rejection_reason:
            raise ValueError(f"alias {self.alias_id}: status='rejected' requires a reason")
        return self

    @property
    def participates_in_class_a(self) -> bool:
        """The only question the resolver asks of an alias."""
        return self.status is AliasStatus.APPROVED

    @property
    def usable_in_class_b(self) -> bool:
        """A proposal may inform a Class B hypothesis, marked as proposed."""
        return self.status in (AliasStatus.PROPOSED, AliasStatus.APPROVED)
