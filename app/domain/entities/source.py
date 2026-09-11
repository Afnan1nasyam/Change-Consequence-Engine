"""Source documents, versions, chunks and evidence."""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.value_objects.enums import AuthorityClass, SourceType
from app.domain.value_objects.identifiers import (
    ChunkId,
    EvidenceId,
    SourceId,
    SourceVersionId,
    TenantId,
)
from app.domain.value_objects.provenance import Sha256

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class SourceDocument(TimestampedModel):
    """A registered artifact.

    ``authority_class`` comes from *registration*, never from document content. That is a
    security property, not a convenience: otherwise a planted file could claim policy
    authority and legitimately override a real policy (``docs/SECURITY.md``).
    """

    source_id: SourceId
    tenant_id: TenantId
    source_type: SourceType
    title: ShortText
    uri: ShortText
    owner: ShortText | None = None
    authority_class: AuthorityClass
    updated_at: dt.datetime | None = None
    permissions: tuple[str, ...] = ()
    current_version_id: SourceVersionId | None = None


class SourceVersion(TimestampedModel):
    """A content-addressed revision with an explicit validity interval."""

    version_id: SourceVersionId
    source_id: SourceId
    content_hash: Sha256
    version_label: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    effective_from: dt.datetime | None = None
    effective_to: dt.datetime | None = None

    @model_validator(mode="after")
    def _interval_ordered(self) -> Self:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise ValueError(
                f"effective_from ({self.effective_from.isoformat()}) must precede "
                f"effective_to ({self.effective_to.isoformat()})"
            )
        return self

    def is_in_force(self, as_of: dt.datetime) -> bool:
        """Whether this version governs at ``as_of``.

        Note the deliberate absence of newest-wins fallback logic: resolving *which*
        version is authoritative when intervals overlap or are missing is a deferred
        service that may legitimately answer ``Ambiguous``.
        """
        if self.effective_from is not None and as_of < self.effective_from:
            return False
        return not (self.effective_to is not None and as_of >= self.effective_to)


class Chunk(DomainModel):
    chunk_id: ChunkId
    source_version_id: SourceVersionId
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _offsets_ordered(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must exceed char_start ({self.char_start})"
            )
        return self


class EvidenceItem(DomainModel):
    """A citable span.

    ``char_start`` / ``char_end`` are required rather than optional because span
    verification - the anti-hallucination primitive everything else rests on - cannot run
    without offsets. An evidence item that cannot be verified is not evidence.
    """

    evidence_id: EvidenceId
    source_version_id: SourceVersionId
    chunk_id: ChunkId
    text_span: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    location: ShortText | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    authority_score: float | None = Field(default=None, ge=0.0, le=1.0)
    freshness_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _offsets_ordered(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must exceed char_start ({self.char_start})"
            )
        return self

    @property
    def retrieval_strategies(self) -> frozenset[str]:
        """Strategies that surfaced this span.

        Used by corroboration counting, which requires *distinct strategies* reaching
        *distinct source versions* - never a count of agreeing agents.
        """
        return frozenset(self.retrieval_scores)


class HistoricalEvent(TimestampedModel):
    """A prior change or incident, usable as a Class B analogue.

    ``observed_outcome`` is the field that matters: an analogue without a recorded outcome
    does not qualify as an anchor, and a Class B hypothesis without an anchor caps at
    ``speculative``.
    """

    event_id: Annotated[str, StringConstraints(pattern=r"^he_[0-9a-f]{32}$")]
    tenant_id: TenantId
    title: ShortText
    occurred_at: dt.datetime
    observed_outcome: str | None = None
    evidence_ids: tuple[EvidenceId, ...] = ()

    @property
    def qualifies_as_anchor(self) -> bool:
        return bool(self.observed_outcome) and bool(self.evidence_ids)
