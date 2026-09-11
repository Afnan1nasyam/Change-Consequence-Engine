"""Store ports for the ingestion path.

Protocols only. The domain says what a store must be able to answer; it does not know
whether the answer comes from Postgres, from object storage, or from a dictionary. That is
what lets Phase 2 be tested end to end with no database while the eventual SQLAlchemy
implementation satisfies the same contract.

**Tenant is a parameter, not a filter applied afterwards.** Every lookup that could cross a
tenant boundary takes ``tenant_id`` in its signature, so a caller cannot forget to scope a
query - the type checker will not let them (``docs/SECURITY.md`` 8, 9).

The canonical text of a version is stored through the same port as its metadata. Span
verification compares a citation against that exact text, so a store that could return the
metadata without the text would be able to hand out unverifiable evidence.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.entities.assumption import AssumptionRecord, CouplingSet
from app.domain.entities.claims import DerivedRelation
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.registry import KeyAlias, KeyRegistryGap
from app.domain.entities.source import Chunk, EvidenceItem, SourceDocument, SourceVersion


@runtime_checkable
class SourceStore(Protocol):
    """Documents, versions, canonical text, chunks and evidence."""

    def put_document(self, document: SourceDocument) -> None: ...

    def get_document(self, tenant_id: str, source_id: str) -> SourceDocument | None: ...

    def find_document_by_uri(self, tenant_id: str, uri: str) -> SourceDocument | None: ...

    def put_version(self, version: SourceVersion, *, text: str) -> None:
        """Store a version together with the canonical text its offsets index into."""
        ...

    def get_version(self, version_id: str) -> SourceVersion | None: ...

    def get_text(self, version_id: str) -> str | None: ...

    def versions_for(self, source_id: str) -> tuple[SourceVersion, ...]:
        """Every version of a document, oldest first. Source history is queryable."""
        ...

    def has_version(self, version_id: str) -> bool:
        """Cheap existence check."""
        ...

    def mark_extracted(self, version_id: str, *, extractor_version: str) -> None:
        """Record that extraction completed for this version under this extractor."""
        ...

    def extracted_with(self, version_id: str) -> str | None:
        """The extractor version this document was last extracted under, if any.

        This, not the presence of records, is what decides whether extraction can be
        skipped. A document that legitimately binds nothing produces no records, and keying
        the cache on records would re-extract it on every ingestion forever - which on a
        real corpus means re-extracting most of it, since most documents bind nothing.

        Returning the *version* rather than a boolean also makes a prompt or parser change
        re-extract, which is correct: those produce different records.
        """
        ...

    def put_chunks(self, chunks: tuple[Chunk, ...]) -> None: ...

    def chunks_for(self, version_id: str) -> tuple[Chunk, ...]: ...

    def put_evidence(self, items: tuple[EvidenceItem, ...]) -> None: ...

    def get_evidence(self, evidence_id: str) -> EvidenceItem | None: ...

    def evidence_for_tenant(self, tenant_id: str) -> tuple[EvidenceItem, ...]:
        """Every citable span owned by one tenant, ordered by ``evidence_id``.

        Retrieval needs to enumerate the corpus, and ``get_evidence`` cannot: a store that
        could only answer "is this id real?" would force a caller to guess ids. Tenant is a
        parameter for the same reason it is everywhere else - an unscoped enumeration of
        every span in the system is exactly the query that must be impossible to write by
        accident (``docs/SECURITY.md`` 8).

        A span has no tenant of its own; it inherits the one on the document holding its
        version, so an implementation resolves the join rather than reading a column.
        """
        ...


@runtime_checkable
class AssumptionStore(Protocol):
    """Admitted bindings only.

    There is deliberately no method to store a proposal or a quarantined item here. The
    trusted store cannot hold an unverified binding because there is no way to put one in.
    """

    def put_records(self, records: tuple[AssumptionRecord, ...]) -> None: ...

    def records_for_version(self, version_id: str) -> tuple[AssumptionRecord, ...]: ...

    def records_for_key(self, tenant_id: str, key: str) -> tuple[AssumptionRecord, ...]: ...

    def has_records_for_version(self, version_id: str) -> bool: ...

    def all_records(self, tenant_id: str) -> tuple[AssumptionRecord, ...]: ...


@runtime_checkable
class RegistryGapStore(Protocol):
    """Quarantined unknown keys.

    Gap volume per domain is a first-class product metric, not an error log, so the port
    exposes it as a query rather than leaving callers to count log lines.
    """

    def put_gap(self, gap: KeyRegistryGap) -> None: ...

    def get_gap(self, gap_id: str) -> KeyRegistryGap | None: ...

    def open_gaps(self) -> tuple[KeyRegistryGap, ...]: ...

    def gap_counts_by_key(self) -> dict[str, int]: ...


@runtime_checkable
class AliasStore(Protocol):
    """Proposed and approved key equivalences.

    ``approved`` is a separate query from ``all`` because the resolver must never
    accidentally iterate proposals: only an explicitly approved alias may participate in
    Class A coupling, and no threshold changes that.
    """

    def put_alias(self, alias: KeyAlias) -> None: ...

    def get_alias(self, alias_id: str) -> KeyAlias | None: ...

    def approved(self) -> tuple[KeyAlias, ...]: ...

    def all_aliases(self) -> tuple[KeyAlias, ...]: ...


@runtime_checkable
class CouplingIndexStore(Protocol):
    """The materialized coupling projection.

    Write-only from the resolver's side and read-only from everyone else's. There is
    deliberately no ``put_edge``, no ``update_outcome`` and no per-record mutator: the index
    is **rebuilt** from the assumption store rather than edited in place, so a stored cell
    can never drift from what the store implies. A writeable index would be a second source
    of truth, which checkpoint 10.1 calls a divergence bug.

    ``put_*`` replaces the tenant's previous projection wholesale for the same reason. A
    merge would leave a stale cell behind after a source was retracted, and that cell would
    keep producing a finding nobody could trace to a document.
    """

    def put_index(self, tenant_id: str, sets: tuple[CouplingSet, ...]) -> None: ...

    def put_relations(self, tenant_id: str, relations: tuple[DerivedRelation, ...]) -> None: ...

    def put_findings(self, tenant_id: str, findings: tuple[ClassAFinding, ...]) -> None: ...

    def coupling_sets(self, tenant_id: str) -> tuple[CouplingSet, ...]: ...

    def relations(self, tenant_id: str) -> tuple[DerivedRelation, ...]: ...

    def findings(self, tenant_id: str) -> tuple[ClassAFinding, ...]: ...

    def findings_for_key(self, tenant_id: str, key: str) -> tuple[ClassAFinding, ...]:
        """Findings on one assumption key. The query a reviewer works from."""
        ...
