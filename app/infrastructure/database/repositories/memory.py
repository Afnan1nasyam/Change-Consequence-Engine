"""In-memory store implementations.

These satisfy the ports in ``app/domain/interfaces/repositories.py`` and are the *only*
implementations in Phase 2. Postgres, SQLAlchemy models and Alembic revisions are
deliberately deferred: the schemas-before-tables rule means tables are written against a
frozen contract, and nothing in the Phase 2 acceptance criteria needs durability. Swapping
in a SQL implementation is a new class behind the same Protocol, with the same tests.

Tenant isolation is enforced here rather than assumed. ``get_document`` and
``find_document_by_uri`` take a tenant and will not return another tenant's row even when
handed a correct id, so a leak requires deleting a check rather than forgetting one
(``docs/SECURITY.md`` 8).
"""

from __future__ import annotations

from app.application.orchestration.analysis_store import (
    StoredAnalysis,
    TenantMismatch,
    UnknownStoredAnalysis,
)
from app.domain.entities.assumption import AssumptionRecord, CouplingSet
from app.domain.entities.claims import DerivedRelation
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.registry import KeyAlias, KeyRegistryGap
from app.domain.entities.source import Chunk, EvidenceItem, SourceDocument, SourceVersion
from app.judgment.dispositions import DispositionRecord


class InMemorySourceStore:
    def __init__(self) -> None:
        self._documents: dict[str, SourceDocument] = {}
        self._by_uri: dict[tuple[str, str], str] = {}
        self._versions: dict[str, SourceVersion] = {}
        self._texts: dict[str, str] = {}
        self._version_order: dict[str, list[str]] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._evidence: dict[str, EvidenceItem] = {}
        self._extracted: dict[str, str] = {}

    # -- documents --------------------------------------------------------

    def put_document(self, document: SourceDocument) -> None:
        self._documents[document.source_id] = document
        self._by_uri[(document.tenant_id, document.uri)] = document.source_id
        self._version_order.setdefault(document.source_id, [])

    def get_document(self, tenant_id: str, source_id: str) -> SourceDocument | None:
        document = self._documents.get(source_id)
        if document is None or document.tenant_id != tenant_id:
            return None
        return document

    def find_document_by_uri(self, tenant_id: str, uri: str) -> SourceDocument | None:
        source_id = self._by_uri.get((tenant_id, uri))
        return self._documents.get(source_id) if source_id else None

    # -- versions ---------------------------------------------------------

    def put_version(self, version: SourceVersion, *, text: str) -> None:
        new = version.version_id not in self._versions
        self._versions[version.version_id] = version
        self._texts[version.version_id] = text
        order = self._version_order.setdefault(version.source_id, [])
        if new:
            order.append(version.version_id)

    def get_version(self, version_id: str) -> SourceVersion | None:
        return self._versions.get(version_id)

    def get_text(self, version_id: str) -> str | None:
        return self._texts.get(version_id)

    def versions_for(self, source_id: str) -> tuple[SourceVersion, ...]:
        return tuple(self._versions[v] for v in self._version_order.get(source_id, []))

    def has_version(self, version_id: str) -> bool:
        return version_id in self._versions

    def mark_extracted(self, version_id: str, *, extractor_version: str) -> None:
        self._extracted[version_id] = extractor_version

    def extracted_with(self, version_id: str) -> str | None:
        return self._extracted.get(version_id)

    # -- chunks and evidence ---------------------------------------------

    def put_chunks(self, chunks: tuple[Chunk, ...]) -> None:
        for chunk in chunks:
            bucket = self._chunks.setdefault(chunk.source_version_id, [])
            if all(existing.chunk_id != chunk.chunk_id for existing in bucket):
                bucket.append(chunk)

    def chunks_for(self, version_id: str) -> tuple[Chunk, ...]:
        return tuple(sorted(self._chunks.get(version_id, []), key=lambda c: c.ordinal))

    def put_evidence(self, items: tuple[EvidenceItem, ...]) -> None:
        for item in items:
            self._evidence.setdefault(item.evidence_id, item)

    def get_evidence(self, evidence_id: str) -> EvidenceItem | None:
        return self._evidence.get(evidence_id)

    def evidence_for_tenant(self, tenant_id: str) -> tuple[EvidenceItem, ...]:
        """Resolved through version -> document, because a span carries no tenant itself.

        Sorted by ``evidence_id``: retrieval builds an index from this, and an index built
        from an unordered enumeration would make a tie break depend on insertion order.
        """
        return tuple(
            sorted(
                (
                    item
                    for item in self._evidence.values()
                    if self._tenant_of_version(item.source_version_id) == tenant_id
                ),
                key=lambda item: item.evidence_id,
            )
        )

    def _tenant_of_version(self, version_id: str) -> str | None:
        version = self._versions.get(version_id)
        if version is None:
            return None
        document = self._documents.get(version.source_id)
        return document.tenant_id if document else None


class InMemoryAssumptionStore:
    def __init__(self) -> None:
        self._records: dict[str, AssumptionRecord] = {}
        self._by_version: dict[str, list[str]] = {}

    def put_records(self, records: tuple[AssumptionRecord, ...]) -> None:
        for record in records:
            if not record.is_admitted:
                raise ValueError(
                    f"refusing to store {record.assumption_id}: the assumption store holds "
                    "admitted bindings only, and this record has not passed all three "
                    "admission checks"
                )
            if record.assumption_id in self._records:
                continue
            self._records[record.assumption_id] = record
            self._by_version.setdefault(record.holder_source_version_id, []).append(
                record.assumption_id
            )

    def records_for_version(self, version_id: str) -> tuple[AssumptionRecord, ...]:
        return tuple(self._records[i] for i in self._by_version.get(version_id, []))

    def records_for_key(self, tenant_id: str, key: str) -> tuple[AssumptionRecord, ...]:
        return tuple(r for r in self._records.values() if r.key == key and r.tenant_id == tenant_id)

    def has_records_for_version(self, version_id: str) -> bool:
        return bool(self._by_version.get(version_id))

    def all_records(self, tenant_id: str) -> tuple[AssumptionRecord, ...]:
        return tuple(r for r in self._records.values() if r.tenant_id == tenant_id)


class InMemoryRegistryGapStore:
    def __init__(self) -> None:
        self._gaps: dict[str, KeyRegistryGap] = {}

    def put_gap(self, gap: KeyRegistryGap) -> None:
        existing = self._gaps.get(gap.gap_id)
        if existing is None:
            self._gaps[gap.gap_id] = gap
            return
        # The same unknown key from the same span, seen again: count it, do not duplicate.
        self._gaps[gap.gap_id] = existing.model_copy(
            update={"occurrence_count": existing.occurrence_count + gap.occurrence_count}
        )

    def get_gap(self, gap_id: str) -> KeyRegistryGap | None:
        return self._gaps.get(gap_id)

    def open_gaps(self) -> tuple[KeyRegistryGap, ...]:
        return tuple(g for g in self._gaps.values() if g.blocks_class_a)

    def gap_counts_by_key(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for gap in self._gaps.values():
            counts[gap.proposed_key] = counts.get(gap.proposed_key, 0) + gap.occurrence_count
        return counts


class InMemoryAliasStore:
    def __init__(self) -> None:
        self._aliases: dict[str, KeyAlias] = {}

    def put_alias(self, alias: KeyAlias) -> None:
        self._aliases[alias.alias_id] = alias

    def get_alias(self, alias_id: str) -> KeyAlias | None:
        return self._aliases.get(alias_id)

    def approved(self) -> tuple[KeyAlias, ...]:
        return tuple(a for a in self._aliases.values() if a.participates_in_class_a)

    def all_aliases(self) -> tuple[KeyAlias, ...]:
        return tuple(self._aliases.values())


class InMemoryCouplingIndexStore:
    """The Phase 3 coupling projection, in memory.

    ``put_*`` **replaces** the tenant's slice rather than merging into it. That is not a
    simplification of a real implementation - it is the behaviour the contract requires. A
    merge would leave a stale cell behind after a source was retracted, and that cell would
    go on producing a finding that no longer follows from the store.
    """

    def __init__(self) -> None:
        self._sets: dict[str, tuple[CouplingSet, ...]] = {}
        self._relations: dict[str, tuple[DerivedRelation, ...]] = {}
        self._findings: dict[str, tuple[ClassAFinding, ...]] = {}

    def put_index(self, tenant_id: str, sets: tuple[CouplingSet, ...]) -> None:
        self._sets[tenant_id] = sets

    def put_relations(self, tenant_id: str, relations: tuple[DerivedRelation, ...]) -> None:
        self._relations[tenant_id] = relations

    def put_findings(self, tenant_id: str, findings: tuple[ClassAFinding, ...]) -> None:
        self._findings[tenant_id] = findings

    def coupling_sets(self, tenant_id: str) -> tuple[CouplingSet, ...]:
        return self._sets.get(tenant_id, ())

    def relations(self, tenant_id: str) -> tuple[DerivedRelation, ...]:
        return self._relations.get(tenant_id, ())

    def findings(self, tenant_id: str) -> tuple[ClassAFinding, ...]:
        return self._findings.get(tenant_id, ())

    def findings_for_key(self, tenant_id: str, key: str) -> tuple[ClassAFinding, ...]:
        return tuple(f for f in self.findings(tenant_id) if f.assumption_key == key)


class InMemoryDispositionStore:
    """Reviewer verdicts, in memory, keyed on ``(analysis_id, consequence_id)``.

    ``put`` replaces the current verdict for that pair. It does not merge and it does not
    delete: the :class:`DispositionRecord` handed in already carries its own superseded
    verdicts and its transition history, so replacing the entry keeps the audit trail intact
    while making the *current* answer unambiguous.

    A secondary index on ``consequence_id`` exists for one reason: a submission naming a
    consequence that belongs to a different analysis is a wiring defect rather than a typo,
    and without this lookup the two are indistinguishable.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], DispositionRecord] = {}

    def put(self, record: DispositionRecord) -> None:
        self._records[record.key] = record

    def get(self, analysis_id: str, consequence_id: str) -> DispositionRecord | None:
        return self._records.get((analysis_id, consequence_id))

    def for_analysis(self, analysis_id: str) -> tuple[DispositionRecord, ...]:
        """Sorted by consequence id, so a report over them is stable between reads."""
        return tuple(
            sorted(
                (r for key, r in self._records.items() if key[0] == analysis_id),
                key=lambda r: r.current.consequence_id,
            )
        )

    def find_consequence(self, consequence_id: str) -> tuple[DispositionRecord, ...]:
        return tuple(
            sorted(
                (r for key, r in self._records.items() if key[1] == consequence_id),
                key=lambda r: r.current.analysis_id,
            )
        )


class InMemoryAnalysisStore:
    """Stored analyses, in memory, keyed on ``(tenant_id, analysis_id)``.

    ``put`` replaces the entry for that pair. It does not merge: a snapshot is a record of one
    finished run, and merging two would produce a review surface that never existed.

    A cross-tenant read **raises** rather than returning ``None``. The distinction matters:
    "not found" is an ordinary miss a UI can render, while a request for another tenant's
    analysis is an isolation defect, and returning the same answer for both would hide the one
    that needs investigating (``docs/SECURITY.md`` 8, 9).
    """

    def __init__(self) -> None:
        self._by_id: dict[str, StoredAnalysis] = {}

    def put(self, analysis: StoredAnalysis) -> None:
        self._by_id[analysis.analysis_id] = analysis

    def get(self, tenant_id: str, analysis_id: str) -> StoredAnalysis | None:
        found = self._by_id.get(analysis_id)
        if found is None:
            return None
        if found.tenant_id != tenant_id:
            raise TenantMismatch(
                f"analysis {analysis_id} belongs to another tenant; refusing to serve it to "
                f"{tenant_id!r}. A cross-tenant read is an isolation defect, not a miss."
            )
        return found

    def require(self, tenant_id: str, analysis_id: str) -> StoredAnalysis:
        found = self.get(tenant_id, analysis_id)
        if found is None:
            raise UnknownStoredAnalysis(f"no stored analysis {analysis_id} for tenant {tenant_id}")
        return found

    def for_tenant(self, tenant_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                analysis_id
                for analysis_id, stored in self._by_id.items()
                if stored.tenant_id == tenant_id
            )
        )
