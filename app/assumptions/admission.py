"""The admission gate: extraction proposes, verification admits.

This module is the trust boundary of the entire system. Above it, everything is a model's
opinion about a document. Below it, an :class:`AssumptionRecord` is something a reviewer can
be asked to rely on without re-reading the corpus.

Every check is deterministic and contains **no model call**. The gate never asks how
confident the extractor was, because that question has no admissible answer.

Order matters, and it follows ``docs/ARCHITECTURE.md`` 3.1:

1. **span** - the chunk exists, and the quoted span exists verbatim inside it;
2. **value in span** - the claimed value appears inside the span that is cited for it;
3. **typed parse** - the value parses under its declared type, with a unit where required;
4. **registry resolution** - the key resolves in the *closed* registry, or via an
   **approved** alias. Failure here, and only here, produces a ``KeyRegistryGap``;
5. **registry agreement** - the parsed value is the type and unit the entry declares;
6. **scope** - well formed, and every dimension is one the entry declares;
7. **temporal validity** - the interval is ordered, and inherits the holder version's
   interval when the text did not state one;
8. **holder metadata** - the version belongs to a registered document in the right tenant.

Parse precedes registry resolution deliberately: a ``KeyRegistryGap`` records the value that
was proposed, so it can only be written once that value is known to be well typed. An
unparseable value is an extraction failure, not a registry gap, and conflating the two would
pollute the signal that tells us where to grow the registry.

A failure produces a :class:`QuarantinedProposal` and nothing else. There is no partial
admission, no retry-with-lower-standards, and no confidence threshold that lets a failed
proposal through.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from app.assumptions.proposals import AssumptionProposal
from app.assumptions.registry import (
    AssumptionKeyRegistry,
    ResolutionFailure,
    scope_is_permitted,
)
from app.domain.entities.assumption import AssumptionRecord
from app.domain.entities.registry import KeyRegistryGap
from app.domain.entities.source import EvidenceItem, SourceDocument, SourceVersion
from app.domain.services.span_verification import (
    locate_quote,
    verify_offsets,
    verify_value_in_span,
)
from app.domain.services.typed_parse import parse_value, validate_against_entry
from app.domain.value_objects.provenance import ModelProvenance
from app.domain.value_objects.scope import ScopeSelector
from app.ingestion.chunking import ChunkWithContext
from app.ingestion.identity import (
    assumption_id_for,
    evidence_id_for,
    registry_gap_id_for,
)


class AdmissionFailure(StrEnum):
    """Why a proposal was quarantined.

    Each value is a distinct reviewer action, which is why they are not collapsed into one
    ``rejected``. ``KEY_UNREGISTERED`` in particular is not an error at all: it is the
    signal that the registry needs to grow.
    """

    CHUNK_UNKNOWN = "chunk_unknown"
    SPAN_NOT_FOUND = "span_not_found"
    SPAN_MISMATCH = "span_mismatch"
    VALUE_NOT_IN_SPAN = "value_not_in_span"
    VALUE_PARSE_FAILED = "value_parse_failed"
    KEY_UNREGISTERED = "key_unregistered"
    KEY_DEPRECATED = "key_deprecated"
    ALIAS_NOT_APPROVED = "alias_not_approved"
    REGISTRY_TYPE_MISMATCH = "registry_type_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    SCOPE_MALFORMED = "scope_malformed"
    SCOPE_NOT_PERMITTED = "scope_not_permitted"
    INTERVAL_INVALID = "interval_invalid"
    HOLDER_METADATA_MISSING = "holder_metadata_missing"
    TENANT_MISMATCH = "tenant_mismatch"


class QuarantinedProposal(BaseModel):
    """A proposal that failed admission, kept with enough provenance to be reviewed.

    Not a domain claim: it asserts nothing about the world, only that a model proposed
    something and code refused it. It is recorded rather than dropped because the failure
    distribution is how we learn whether the extractor, the registry, or the corpus is the
    problem (``docs/EVALUATION.md``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure: AdmissionFailure
    detail: str
    proposal: AssumptionProposal
    source_version_id: str
    chunk_id: str | None = None
    #: Set only for ``KEY_UNREGISTERED``: the quarantine and the gap are the same event.
    gap: KeyRegistryGap | None = None
    proposed_by: ModelProvenance

    @property
    def is_registry_gap(self) -> bool:
        return self.gap is not None


class AdmissionOutcome(BaseModel):
    """Everything one source version's proposals produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: tuple[AssumptionRecord, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    gaps: tuple[KeyRegistryGap, ...] = ()
    quarantined: tuple[QuarantinedProposal, ...] = ()
    registry_version: str = ""
    extractor_version: str = ""

    @property
    def admitted_count(self) -> int:
        return len(self.records)

    @property
    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.quarantined:
            counts[item.failure.value] = counts.get(item.failure.value, 0) + 1
        return counts


class _Admitted(NamedTuple):
    record: AssumptionRecord
    evidence: EvidenceItem


class AdmissionGate:
    """Deterministic verification of extraction proposals. No model call, ever."""

    def __init__(self, *, registry: AssumptionKeyRegistry, extractor_version: str) -> None:
        self._registry = registry
        self._extractor_version = extractor_version

    # -- public API -------------------------------------------------------

    def admit_all(
        self,
        proposals: Sequence[tuple[AssumptionProposal, ModelProvenance]],
        *,
        document: SourceDocument,
        version: SourceVersion,
        document_text: str,
        chunks: Sequence[ChunkWithContext],
    ) -> AdmissionOutcome:
        records: list[AssumptionRecord] = []
        evidence: list[EvidenceItem] = []
        gaps: list[KeyRegistryGap] = []
        quarantined: list[QuarantinedProposal] = []

        by_ordinal = {c.chunk.ordinal: c for c in chunks}

        for proposal, provenance in proposals:
            result = self.admit(
                proposal,
                provenance=provenance,
                document=document,
                version=version,
                document_text=document_text,
                chunk=by_ordinal.get(proposal.chunk_ordinal),
            )
            if isinstance(result, QuarantinedProposal):
                quarantined.append(result)
                if result.gap is not None:
                    gaps.append(result.gap)
            else:
                records.append(result.record)
                evidence.append(result.evidence)

        return AdmissionOutcome(
            records=_dedupe(records, key=lambda r: r.assumption_id),
            evidence=_dedupe(evidence, key=lambda e: e.evidence_id),
            gaps=_merge_gaps(gaps),
            quarantined=tuple(quarantined),
            registry_version=self._registry.version,
            extractor_version=self._extractor_version,
        )

    def admit(
        self,
        proposal: AssumptionProposal,
        *,
        provenance: ModelProvenance,
        document: SourceDocument,
        version: SourceVersion,
        document_text: str,
        chunk: ChunkWithContext | None,
    ) -> _Admitted | QuarantinedProposal:
        """Run every check against one proposal. Admit it, or quarantine it."""

        def reject(
            failure: AdmissionFailure, detail: str, *, gap: KeyRegistryGap | None = None
        ) -> QuarantinedProposal:
            return QuarantinedProposal(
                failure=failure,
                detail=detail,
                proposal=proposal,
                source_version_id=version.version_id,
                chunk_id=chunk.chunk.chunk_id if chunk else None,
                gap=gap,
                proposed_by=provenance,
            )

        # -- 8 (cheap, so first): holder metadata ------------------------
        if version.source_id != document.source_id:
            return reject(
                AdmissionFailure.HOLDER_METADATA_MISSING,
                f"version {version.version_id} belongs to {version.source_id}, "
                f"not to the supplied document {document.source_id}",
            )

        # -- 1: the chunk and the span -----------------------------------
        if chunk is None:
            return reject(
                AdmissionFailure.CHUNK_UNKNOWN,
                f"no chunk with ordinal {proposal.chunk_ordinal} in this version",
            )

        located = locate_quote(
            document_text,
            proposal.quote,
            search_start=chunk.chunk.char_start,
            search_end=chunk.chunk.char_end,
        )
        if located is None:
            return reject(
                AdmissionFailure.SPAN_NOT_FOUND,
                f"quote not found verbatim in chunk {proposal.chunk_ordinal}: "
                f"{proposal.quote[:120]!r}",
            )

        offsets_ok = verify_offsets(
            document_text,
            char_start=located.char_start,
            char_end=located.char_end,
            claimed=proposal.quote,
        )
        if not offsets_ok.ok:
            return reject(AdmissionFailure.SPAN_MISMATCH, offsets_ok.detail)

        # -- 2: the value must be inside the span it cites ---------------
        value_in_span = verify_value_in_span(located.text, rendered_value=proposal.value_quote)
        if not value_in_span.ok:
            return reject(AdmissionFailure.VALUE_NOT_IN_SPAN, value_in_span.detail)

        # -- 3: typed parse ----------------------------------------------
        parsed = parse_value(
            proposal.value_quote, value_type=proposal.value_type, unit=proposal.unit
        )
        if parsed.value is None:
            return reject(AdmissionFailure.VALUE_PARSE_FAILED, parsed.detail)

        evidence = EvidenceItem(
            evidence_id=evidence_id_for(
                version_id=version.version_id,
                char_start=located.char_start,
                char_end=located.char_end,
            ),
            source_version_id=version.version_id,
            chunk_id=chunk.chunk.chunk_id,
            text_span=located.text,
            char_start=located.char_start,
            char_end=located.char_end,
            location=chunk.section_label,
        )

        # -- 6a: scope must at least be well formed ----------------------
        try:
            scope = ScopeSelector.parse(proposal.scope)
        except ValueError as exc:
            return reject(AdmissionFailure.SCOPE_MALFORMED, f"scope {proposal.scope!r}: {exc}")

        # -- 4: registry resolution (the only gap-producing failure) -----
        resolution = self._registry.resolve(proposal.proposed_key)
        if resolution.entry is None:
            if resolution.failure is ResolutionFailure.UNREGISTERED:
                gap = KeyRegistryGap(
                    gap_id=registry_gap_id_for(
                        version_id=version.version_id,
                        proposed_key=proposal.proposed_key,
                        evidence_id=evidence.evidence_id,
                    ),
                    proposed_key=proposal.proposed_key,
                    proposed_value=parsed.value,
                    evidence_id=evidence.evidence_id,
                    holder_source_version_id=version.version_id,
                    proposed_by=provenance,
                    as_of=_as_datetime(version.effective_from) or version.created_at,
                    registry_version=self._registry.version,
                )
                return reject(AdmissionFailure.KEY_UNREGISTERED, resolution.detail, gap=gap)
            failure = (
                AdmissionFailure.ALIAS_NOT_APPROVED
                if resolution.failure is ResolutionFailure.ALIAS_NOT_APPROVED
                else AdmissionFailure.KEY_DEPRECATED
            )
            return reject(failure, resolution.detail)

        entry = resolution.entry

        # -- 5: the value must be what the registry says this key holds --
        agreed = validate_against_entry(parsed.value, entry)
        if agreed.value is None:
            failure = (
                AdmissionFailure.UNIT_MISMATCH
                if agreed.detail.startswith("registry declares unit")
                else AdmissionFailure.REGISTRY_TYPE_MISMATCH
            )
            return reject(failure, agreed.detail)

        # -- 6b: scope dimensions must be declared for this key ----------
        if not scope_is_permitted(entry, scope):
            return reject(
                AdmissionFailure.SCOPE_NOT_PERMITTED,
                f"key {entry.key!r} declares scopes {list(entry.allowed_scopes)}; "
                f"proposal uses {sorted(scope.dimensions)}",
            )

        # -- 7: temporal validity ----------------------------------------
        effective_from = _as_datetime(proposal.effective_from) or version.effective_from
        effective_to = _as_datetime(proposal.effective_to) or version.effective_to
        if (
            effective_from is not None
            and effective_to is not None
            and effective_from >= effective_to
        ):
            return reject(
                AdmissionFailure.INTERVAL_INVALID,
                f"effective_from {effective_from.isoformat()} does not precede "
                f"effective_to {effective_to.isoformat()}",
            )

        record = AssumptionRecord(
            assumption_id=assumption_id_for(
                version_id=version.version_id,
                key=entry.key,
                scope=scope.render(),
                role=proposal.role.value,
                evidence_id=evidence.evidence_id,
                extractor_version=self._extractor_version,
            ),
            tenant_id=document.tenant_id,
            key=entry.key,
            registry_version=self._registry.version,
            scope=scope,
            value=agreed.value,
            role=proposal.role,
            explicitness=proposal.explicitness,
            holder_source_version_id=version.version_id,
            evidence_id=evidence.evidence_id,
            effective_from=effective_from,
            effective_to=effective_to,
            extractor_version=self._extractor_version,
            # Set by the gate, never by the extractor. Reaching this line *is* the proof.
            span_verified=True,
            value_parsed=True,
            key_resolved=True,
        )
        return _Admitted(record, evidence)


def _as_datetime(value: dt.date | dt.datetime | None) -> dt.datetime | None:
    """Dates from a document become UTC midnight. Times are left alone."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC)


def _dedupe[T](items: Sequence[T], *, key: Callable[[T], str]) -> tuple[T, ...]:
    """Keep first occurrence per derived id.

    Two chunks quoting the same sentence produce the same evidence id, and the same binding
    proposed twice produces the same assumption id. Deduplicating on identity is what makes
    a re-ingest idempotent instead of doubling the store.
    """
    seen: set[str] = set()
    out: list[T] = []
    for item in items:
        identity = key(item)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(item)
    return tuple(out)


def _merge_gaps(gaps: Sequence[KeyRegistryGap]) -> tuple[KeyRegistryGap, ...]:
    """One gap per (version, key, span), with occurrences counted rather than repeated.

    Gap *volume* per domain is a product metric, so it must count distinct evidence of a
    missing key, not how many times one span was re-proposed.
    """
    merged: dict[str, KeyRegistryGap] = {}
    for gap in gaps:
        existing = merged.get(gap.gap_id)
        if existing is None:
            merged[gap.gap_id] = gap
        else:
            merged[gap.gap_id] = existing.model_copy(
                update={"occurrence_count": existing.occurrence_count + 1}
            )
    return tuple(merged[k] for k in sorted(merged))
