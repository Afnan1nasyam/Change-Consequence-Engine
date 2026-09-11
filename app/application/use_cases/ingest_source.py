"""Ingest one source document: parse, version, chunk, extract, admit.

This is the only place the ingestion stages are sequenced. Nothing in the API layer
performs ingestion logic; a route calls this and reports what it returns.

The caching decision lives here, because it is the one place that can see both the store and
the parser output. A version id is derived from the content hash, so an unchanged document
resolves to a version that is already present and extraction is skipped **on identity**
rather than by diffing text. That is what makes *re-ingesting an unchanged source performs
zero extraction calls* a property of the design rather than an optimization someone has to
remember.

Registration metadata - tenant, authority class, source type, permissions, effective
interval - is supplied by the caller and never read from document content. A document cannot
promote itself (``docs/SECURITY.md``).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.assumptions.admission import AdmissionGate, AdmissionOutcome
from app.assumptions.extraction import ExtractionOutcome
from app.assumptions.proposals import AssumptionProposal
from app.domain.entities.base import utc_now
from app.domain.entities.source import SourceDocument, SourceVersion
from app.domain.interfaces.repositories import (
    AssumptionStore,
    RegistryGapStore,
    SourceStore,
)
from app.domain.value_objects.enums import AuthorityClass, SourceType
from app.domain.value_objects.provenance import ModelProvenance
from app.ingestion.chunking import ChunkingConfig, ChunkWithContext, chunk_document
from app.ingestion.identity import (
    content_hash,
    source_id_for,
    version_id_for,
)
from app.ingestion.parsers import parse_payload

logger = logging.getLogger(__name__)


class SourceRegistration(BaseModel):
    """What the operator asserts about a document, independent of its content.

    ``authority_class`` is here and not in the parsed metadata for a security reason: if a
    document could declare its own authority, a planted wiki page could claim ``policy`` and
    legitimately override a real policy in the override test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    uri: str
    title: str = Field(min_length=1, max_length=512)
    source_type: SourceType
    authority_class: AuthorityClass
    suffix: str
    owner: str | None = None
    permissions: tuple[str, ...] = ()
    version_label: str = Field(default="v1", min_length=1, max_length=64)
    effective_from: dt.datetime | None = None
    effective_to: dt.datetime | None = None


class ExtractionPort(Protocol):
    """What ingestion needs from the extractor.

    Narrow on purpose: the use case knows that something proposes bindings for chunks and
    reports its own version. It does not know that a model is involved, which keeps the
    admission path testable without one - any object satisfying this Protocol works,
    including a fixture replaying recorded proposals.
    """

    @property
    def version(self) -> str: ...

    async def extract(
        self,
        chunks: Sequence[ChunkWithContext],
        *,
        correlation_id: str | None = None,
    ) -> ExtractionOutcome: ...


class IngestionReport(BaseModel):
    """The outcome of one ingestion, in the terms the acceptance criteria are written in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    version_id: str
    content_hash: str
    parser_name: str
    parser_version: str
    chunker_fingerprint: str
    chunk_count: int
    #: False when the version was already in the store: nothing was re-extracted.
    extracted: bool
    extraction_calls: int = 0
    admitted: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    quarantined: tuple[str, ...] = ()
    observed_instructions: tuple[str, ...] = ()
    secrets_redacted: tuple[str, ...] = ()
    delimiter_breakout_chunks: tuple[str, ...] = ()
    extraction_failures: tuple[str, ...] = ()

    @property
    def admitted_count(self) -> int:
        return len(self.admitted)

    @property
    def is_new_version(self) -> bool:
        return self.extracted


class IngestSourceUseCase:
    """Parse, version, chunk, extract and admit one document."""

    def __init__(
        self,
        *,
        sources: SourceStore,
        assumptions: AssumptionStore,
        gaps: RegistryGapStore,
        gate: AdmissionGate,
        extractor: ExtractionPort | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self._sources = sources
        self._assumptions = assumptions
        self._gaps = gaps
        self._gate = gate
        self._extractor = extractor
        self._chunking = chunking or ChunkingConfig()

    async def execute(
        self,
        payload: bytes,
        registration: SourceRegistration,
        *,
        correlation_id: str | None = None,
        force_extraction: bool = False,
    ) -> IngestionReport:
        parsed = parse_payload(
            payload, source_type=registration.source_type, suffix=registration.suffix
        )
        digest = content_hash(parsed.text)

        source_id = source_id_for(tenant_id=registration.tenant_id, uri=registration.uri)
        version_id = version_id_for(source_id=source_id, content_hash_hex=digest)

        document = self._register_document(registration, source_id=source_id, version_id=version_id)
        already_present = self._sources.has_version(version_id)

        version = SourceVersion(
            version_id=version_id,
            source_id=source_id,
            content_hash=digest,
            version_label=registration.version_label,
            effective_from=registration.effective_from,
            effective_to=registration.effective_to,
        )
        self._sources.put_version(version, text=parsed.text)

        chunks = chunk_document(parsed, version_id=version_id, config=self._chunking)
        self._sources.put_chunks(tuple(c.chunk for c in chunks))

        report_base = {
            "source_id": source_id,
            "version_id": version_id,
            "content_hash": digest,
            "parser_name": parsed.parser_name,
            "parser_version": parsed.parser_version,
            "chunker_fingerprint": self._chunking.fingerprint(),
            "chunk_count": len(chunks),
        }

        # -- the cache decision -------------------------------------------
        # Keyed on an explicit extraction marker, not on whether records exist. A document
        # that legitimately binds nothing produces no records, and on a real corpus most
        # documents bind nothing - keying on records would re-extract almost the whole
        # corpus on every ingestion and destroy the amortization this stage depends on.
        if self._extractor is None:
            return IngestionReport(**report_base, extracted=False)

        previous = self._sources.extracted_with(version_id)
        current = self._extractor.version
        if already_present and previous == current and not force_extraction:
            logger.info(
                "source version already extracted; skipping the model call",
                extra={
                    "version_id": version_id,
                    "source_id": source_id,
                    "extractor_version": current,
                },
            )
            return IngestionReport(
                **report_base,
                extracted=False,
                admitted=tuple(
                    r.assumption_id for r in self._assumptions.records_for_version(version_id)
                ),
            )

        outcome = await self._extractor.extract(chunks, correlation_id=correlation_id)
        proposals, provenances, telemetry = _unpack(outcome)

        admission = self._gate.admit_all(
            list(zip(proposals, provenances, strict=True)),
            document=document,
            version=version,
            document_text=parsed.text,
            chunks=chunks,
        )
        self._persist(admission)
        self._sources.mark_extracted(version_id, extractor_version=current)

        return IngestionReport(
            **report_base,
            extracted=True,
            extraction_calls=telemetry.calls,
            admitted=tuple(r.assumption_id for r in admission.records),
            gaps=tuple(g.gap_id for g in admission.gaps),
            quarantined=tuple(
                f"{q.failure.value}: {q.proposal.proposed_key}" for q in admission.quarantined
            ),
            observed_instructions=telemetry.observed_instructions,
            secrets_redacted=telemetry.secrets_redacted,
            delimiter_breakout_chunks=telemetry.breakout_chunks,
            extraction_failures=telemetry.failures,
        )

    # -- internals --------------------------------------------------------

    def _register_document(
        self, registration: SourceRegistration, *, source_id: str, version_id: str
    ) -> SourceDocument:
        existing = self._sources.get_document(registration.tenant_id, source_id)
        document = SourceDocument(
            source_id=source_id,
            tenant_id=registration.tenant_id,
            source_type=registration.source_type,
            title=registration.title,
            uri=registration.uri,
            owner=registration.owner,
            authority_class=registration.authority_class,
            updated_at=utc_now(),
            permissions=registration.permissions,
            current_version_id=version_id,
            created_at=existing.created_at if existing else utc_now(),
        )
        self._sources.put_document(document)
        return document

    def _persist(self, admission: AdmissionOutcome) -> None:
        """Evidence and records go to the trusted store; gaps go to quarantine.

        Quarantined proposals are deliberately not written to the assumption store, and the
        store would refuse them anyway: it accepts only records whose three admission flags
        are set.
        """
        self._sources.put_evidence(admission.evidence)
        self._assumptions.put_records(admission.records)
        for gap in admission.gaps:
            self._gaps.put_gap(gap)


class _Telemetry(NamedTuple):
    calls: int
    observed_instructions: tuple[str, ...]
    secrets_redacted: tuple[str, ...]
    breakout_chunks: tuple[str, ...]
    failures: tuple[str, ...]


def _unpack(
    outcome: ExtractionOutcome,
) -> tuple[list[AssumptionProposal], list[ModelProvenance], _Telemetry]:
    """Flatten an extraction outcome into proposals paired with their provenance.

    Each proposal keeps the provenance of the call that produced it, so a gap or a
    quarantine record can name the exact prompt, provider and model responsible.
    """
    proposals: list[AssumptionProposal] = []
    provenances: list[ModelProvenance] = []
    instructions: list[str] = []
    secrets: list[str] = []
    breakouts: list[str] = []
    failures: list[str] = []

    for extraction in outcome.extractions:
        for proposal in extraction.proposals:
            proposals.append(proposal)
            provenances.append(extraction.provenance)
        instructions.extend(extraction.observed_instructions)
        if extraction.secrets_redacted != "none":
            secrets.append(f"{extraction.chunk.chunk.chunk_id}: {extraction.secrets_redacted}")
        if extraction.attempted_delimiter_breakout:
            breakouts.append(extraction.chunk.chunk.chunk_id)
        if extraction.failure:
            failures.append(f"{extraction.chunk.chunk.chunk_id}: {extraction.failure}")

    return (
        proposals,
        provenances,
        _Telemetry(
            calls=outcome.calls_made,
            observed_instructions=tuple(instructions),
            secrets_redacted=tuple(secrets),
            breakout_chunks=tuple(breakouts),
            failures=tuple(failures),
        ),
    )
