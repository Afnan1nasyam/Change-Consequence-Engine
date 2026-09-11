"""PostgreSQL-backed implementations of all seven store ports.

Same protocols, same semantics, same tests. Each class below satisfies the matching Protocol
in ``app/domain/interfaces/repositories.py`` (or, for the two application-layer ports, in
``app/judgment/dispositions.py`` and ``app/application/orchestration/analysis_store.py``), and
``tests/integration/test_sql_repositories.py`` runs one conformance suite against both these
and the in-memory implementations.

Four things are done the same way everywhere, and each is a decision
-------------------------------------------------------------------

**One transaction per call, unless the caller opened a unit of work.** The ports have no
transaction concept and the in-memory stores commit immediately, so that is the faithful
mapping. ``Database.unit_of_work()`` lets a caller group the three coupling-projection writes
when it wants them atomic.

**No dialect-specific upsert.** ``put_*`` is read-then-write rather than
``ON CONFLICT DO UPDATE``, because the port semantics differ per method - ``put_evidence``
keeps the first write, ``put_alias`` replaces, ``put_gap`` *accumulates a counter* - and a
single upsert idiom would flatten those into one. Writing them out means each reads like the
contract it implements, and the same code runs on any dialect.

**Every query has an explicit ORDER BY.** Several ports promise an order (oldest first, sorted
by id); SQL guarantees none without one. Where the port promises insertion order, a stored
``ordinal`` reproduces it.

**Reads go through the entity constructor.** ``Model.model_validate(row.payload)`` re-runs
every validator, so a row that the domain could not have produced cannot be served. A stale
projected column can therefore only make a *filter* wrong, never a returned value.

What is deliberately absent: any method that mutates a stored aggregate in place. The coupling
index is rebuilt, not edited; a disposition's history travels inside its own record; an
analysis snapshot is write-once per id. There is no ``update_*`` here to call by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select

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
from app.domain.value_objects.enums import AliasStatus, RegistryGapStatus
from app.infrastructure.database.codecs import (
    decode_analysis,
    decode_disposition,
    encode_analysis,
    encode_disposition,
)
from app.infrastructure.database.models.tables import (
    AssumptionRecordRow,
    ChunkRow,
    ClassAFindingRow,
    CouplingSetRow,
    DerivedRelationRow,
    EvidenceItemRow,
    FindingDispositionRow,
    KeyAliasRow,
    RegistryGapRow,
    SourceDocumentRow,
    SourceVersionRow,
    StoredAnalysisRow,
)
from app.infrastructure.database.session import Database
from app.judgment.dispositions import DispositionRecord


class _SqlStore:
    """Holds the session source. Deliberately the only shared state."""

    def __init__(self, database: Database) -> None:
        self._db = database


def _dump(entity: Any) -> dict[str, Any]:
    """JSON-mode dump, which is what makes the payload portable across drivers.

    ``mode="json"`` rather than the default: a Python-mode dump keeps ``datetime`` and enum
    objects, which a JSON column would then have to coerce per driver. Doing it once here means
    what is stored is exactly what is read.
    """
    dumped: dict[str, Any] = entity.model_dump(mode="json")
    return dumped


# ---------------------------------------------------------------------------
# SourceStore
# ---------------------------------------------------------------------------


class SqlSourceStore(_SqlStore):
    """Documents, versions, canonical text, chunks and evidence."""

    # -- documents --------------------------------------------------------

    def put_document(self, document: SourceDocument) -> None:
        with self._db.session() as session:
            row = session.get(SourceDocumentRow, document.source_id)
            if row is None:
                session.add(
                    SourceDocumentRow(
                        source_id=document.source_id,
                        tenant_id=document.tenant_id,
                        uri=document.uri,
                        payload=_dump(document),
                    )
                )
                return
            row.tenant_id = document.tenant_id
            row.uri = document.uri
            row.payload = _dump(document)

    def get_document(self, tenant_id: str, source_id: str) -> SourceDocument | None:
        """Tenant is part of the predicate, not a check after the fetch.

        So a leak requires editing the query rather than forgetting a branch
        (``docs/SECURITY.md`` 8).
        """
        with self._db.session() as session:
            row = session.scalars(
                select(SourceDocumentRow).where(
                    SourceDocumentRow.source_id == source_id,
                    SourceDocumentRow.tenant_id == tenant_id,
                )
            ).one_or_none()
            return SourceDocument.model_validate(row.payload) if row else None

    def find_document_by_uri(self, tenant_id: str, uri: str) -> SourceDocument | None:
        with self._db.session() as session:
            row = session.scalars(
                select(SourceDocumentRow).where(
                    SourceDocumentRow.tenant_id == tenant_id,
                    SourceDocumentRow.uri == uri,
                )
            ).one_or_none()
            return SourceDocument.model_validate(row.payload) if row else None

    # -- versions ---------------------------------------------------------

    def put_version(self, version: SourceVersion, *, text: str) -> None:
        """Metadata and canonical text in one write, because the port stores them together."""
        with self._db.session() as session:
            row = session.get(SourceVersionRow, version.version_id)
            if row is None:
                session.add(
                    SourceVersionRow(
                        version_id=version.version_id,
                        source_id=version.source_id,
                        # Appended, so `versions_for` returns what it returned in memory.
                        ordinal=_next_ordinal(
                            session,
                            SourceVersionRow,
                            SourceVersionRow.source_id == version.source_id,
                        ),
                        canonical_text=text,
                        payload=_dump(version),
                    )
                )
                return
            # A re-put updates in place and keeps the ordinal: the version has not moved in
            # the document's history, its content has been restated.
            row.payload = _dump(version)
            row.canonical_text = text

    def get_version(self, version_id: str) -> SourceVersion | None:
        with self._db.session() as session:
            row = session.get(SourceVersionRow, version_id)
            return SourceVersion.model_validate(row.payload) if row else None

    def get_text(self, version_id: str) -> str | None:
        with self._db.session() as session:
            return session.scalars(
                select(SourceVersionRow.canonical_text).where(
                    SourceVersionRow.version_id == version_id
                )
            ).one_or_none()

    def versions_for(self, source_id: str) -> tuple[SourceVersion, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(SourceVersionRow)
                .where(SourceVersionRow.source_id == source_id)
                .order_by(SourceVersionRow.ordinal)
            ).all()
            return tuple(SourceVersion.model_validate(row.payload) for row in rows)

    def has_version(self, version_id: str) -> bool:
        with self._db.session() as session:
            return (
                session.scalars(
                    select(SourceVersionRow.version_id).where(
                        SourceVersionRow.version_id == version_id
                    )
                ).one_or_none()
                is not None
            )

    def mark_extracted(self, version_id: str, *, extractor_version: str) -> None:
        with self._db.session() as session:
            row = session.get(SourceVersionRow, version_id)
            if row is None:
                raise KeyError(
                    f"cannot mark {version_id} extracted: no such version. Extraction is "
                    "recorded against a stored version, so this is a wiring defect rather "
                    "than a missing row to create."
                )
            row.extractor_version = extractor_version

    def extracted_with(self, version_id: str) -> str | None:
        with self._db.session() as session:
            return session.scalars(
                select(SourceVersionRow.extractor_version).where(
                    SourceVersionRow.version_id == version_id
                )
            ).one_or_none()

    # -- chunks and evidence ---------------------------------------------

    def put_chunks(self, chunks: tuple[Chunk, ...]) -> None:
        """First write wins per chunk id, matching the in-memory behaviour."""
        with self._db.session() as session:
            known = set(
                session.scalars(
                    select(ChunkRow.chunk_id).where(
                        ChunkRow.chunk_id.in_([c.chunk_id for c in chunks])
                    )
                ).all()
            )
            for chunk in chunks:
                if chunk.chunk_id in known:
                    continue
                known.add(chunk.chunk_id)
                session.add(
                    ChunkRow(
                        chunk_id=chunk.chunk_id,
                        source_version_id=chunk.source_version_id,
                        ordinal=chunk.ordinal,
                        payload=_dump(chunk),
                    )
                )

    def chunks_for(self, version_id: str) -> tuple[Chunk, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(ChunkRow)
                .where(ChunkRow.source_version_id == version_id)
                .order_by(ChunkRow.ordinal)
            ).all()
            return tuple(Chunk.model_validate(row.payload) for row in rows)

    def put_evidence(self, items: tuple[EvidenceItem, ...]) -> None:
        """First write wins, as the in-memory ``setdefault`` does.

        Evidence ids are content-derived, so a second write with the same id is the same span;
        replacing it would be a no-op with a write cost.
        """
        with self._db.session() as session:
            known = set(
                session.scalars(
                    select(EvidenceItemRow.evidence_id).where(
                        EvidenceItemRow.evidence_id.in_([i.evidence_id for i in items])
                    )
                ).all()
            )
            for item in items:
                if item.evidence_id in known:
                    continue
                known.add(item.evidence_id)
                session.add(
                    EvidenceItemRow(
                        evidence_id=item.evidence_id,
                        source_version_id=item.source_version_id,
                        payload=_dump(item),
                    )
                )

    def get_evidence(self, evidence_id: str) -> EvidenceItem | None:
        with self._db.session() as session:
            row = session.get(EvidenceItemRow, evidence_id)
            return EvidenceItem.model_validate(row.payload) if row else None

    def evidence_for_tenant(self, tenant_id: str) -> tuple[EvidenceItem, ...]:
        """Resolved through version -> document, because a span carries no tenant itself.

        A real join rather than a denormalized ``tenant_id`` on the span: duplicating the
        tenant onto evidence would create a second answer to "who owns this span", and the two
        could disagree after a document moved. Sorted by ``evidence_id`` because retrieval
        builds an index from this and a tie-break must not depend on physical row order.
        """
        with self._db.session() as session:
            rows = session.scalars(
                select(EvidenceItemRow)
                .join(
                    SourceVersionRow,
                    SourceVersionRow.version_id == EvidenceItemRow.source_version_id,
                )
                .join(
                    SourceDocumentRow,
                    SourceDocumentRow.source_id == SourceVersionRow.source_id,
                )
                .where(SourceDocumentRow.tenant_id == tenant_id)
                .order_by(EvidenceItemRow.evidence_id)
            ).all()
            return tuple(EvidenceItem.model_validate(row.payload) for row in rows)


# ---------------------------------------------------------------------------
# AssumptionStore
# ---------------------------------------------------------------------------


class SqlAssumptionStore(_SqlStore):
    """Admitted bindings only. The refusal below is the store's half of that guarantee."""

    def put_records(self, records: tuple[AssumptionRecord, ...]) -> None:
        with self._db.session() as session:
            ordinal = _next_ordinal(session, AssumptionRecordRow)
            known = set(
                session.scalars(
                    select(AssumptionRecordRow.assumption_id).where(
                        AssumptionRecordRow.assumption_id.in_([r.assumption_id for r in records])
                    )
                ).all()
            )
            for record in records:
                if not record.is_admitted:
                    raise ValueError(
                        f"refusing to store {record.assumption_id}: the assumption store holds "
                        "admitted bindings only, and this record has not passed all three "
                        "admission checks"
                    )
                if record.assumption_id in known:
                    continue
                known.add(record.assumption_id)
                session.add(
                    AssumptionRecordRow(
                        assumption_id=record.assumption_id,
                        tenant_id=record.tenant_id,
                        assumption_key=record.key,
                        holder_source_version_id=record.holder_source_version_id,
                        ordinal=ordinal,
                        payload=_dump(record),
                    )
                )
                ordinal += 1

    def records_for_version(self, version_id: str) -> tuple[AssumptionRecord, ...]:
        return self._query(AssumptionRecordRow.holder_source_version_id == version_id)

    def records_for_key(self, tenant_id: str, key: str) -> tuple[AssumptionRecord, ...]:
        return self._query(
            AssumptionRecordRow.tenant_id == tenant_id,
            AssumptionRecordRow.assumption_key == key,
        )

    def has_records_for_version(self, version_id: str) -> bool:
        with self._db.session() as session:
            found = session.scalars(
                select(AssumptionRecordRow.assumption_id)
                .where(AssumptionRecordRow.holder_source_version_id == version_id)
                .limit(1)
            ).one_or_none()
            return found is not None

    def all_records(self, tenant_id: str) -> tuple[AssumptionRecord, ...]:
        return self._query(AssumptionRecordRow.tenant_id == tenant_id)

    def _query(self, *where: Any) -> tuple[AssumptionRecord, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(AssumptionRecordRow).where(*where).order_by(AssumptionRecordRow.ordinal)
            ).all()
            return tuple(AssumptionRecord.model_validate(row.payload) for row in rows)


# ---------------------------------------------------------------------------
# RegistryGapStore
# ---------------------------------------------------------------------------


class SqlRegistryGapStore(_SqlStore):
    """Quarantined unknown keys, with volume as a first-class query."""

    def put_gap(self, gap: KeyRegistryGap) -> None:
        """Accumulates rather than replaces. Gap volume is the product signal here.

        The same unknown key from the same span seen again is one gap with a higher count, not
        a second gap - which is why this is not a plain upsert.
        """
        with self._db.session() as session:
            row = session.get(RegistryGapRow, gap.gap_id)
            if row is None:
                session.add(
                    RegistryGapRow(
                        gap_id=gap.gap_id,
                        proposed_key=gap.proposed_key,
                        status=gap.status.value,
                        occurrence_count=gap.occurrence_count,
                        payload=_dump(gap),
                    )
                )
                return
            existing = KeyRegistryGap.model_validate(row.payload)
            merged = existing.model_copy(
                update={"occurrence_count": existing.occurrence_count + gap.occurrence_count}
            )
            row.occurrence_count = merged.occurrence_count
            row.status = merged.status.value
            row.payload = _dump(merged)

    def get_gap(self, gap_id: str) -> KeyRegistryGap | None:
        with self._db.session() as session:
            row = session.get(RegistryGapRow, gap_id)
            return KeyRegistryGap.model_validate(row.payload) if row else None

    def open_gaps(self) -> tuple[KeyRegistryGap, ...]:
        """Everything that blocks a Class A derivation - i.e. anything not registered.

        The predicate is expressed over ``status`` because that is the field
        ``KeyRegistryGap.blocks_class_a`` reads. Storing the derived flag would give the
        property a second implementation that could disagree with it.
        """
        with self._db.session() as session:
            rows = session.scalars(
                select(RegistryGapRow)
                .where(RegistryGapRow.status != RegistryGapStatus.REGISTERED.value)
                .order_by(RegistryGapRow.gap_id)
            ).all()
            return tuple(KeyRegistryGap.model_validate(row.payload) for row in rows)

    def gap_counts_by_key(self) -> dict[str, int]:
        with self._db.session() as session:
            rows = session.execute(
                select(
                    RegistryGapRow.proposed_key,
                    func.sum(RegistryGapRow.occurrence_count),
                )
                .group_by(RegistryGapRow.proposed_key)
                .order_by(RegistryGapRow.proposed_key)
            ).all()
            return {key: int(total or 0) for key, total in rows}


# ---------------------------------------------------------------------------
# AliasStore
# ---------------------------------------------------------------------------


class SqlAliasStore(_SqlStore):
    """Proposed and approved key equivalences. ``approved`` is a separate query on purpose."""

    def put_alias(self, alias: KeyAlias) -> None:
        with self._db.session() as session:
            row = session.get(KeyAliasRow, alias.alias_id)
            if row is None:
                session.add(
                    KeyAliasRow(
                        alias_id=alias.alias_id,
                        source_key=alias.source_key,
                        target_key=alias.target_key,
                        status=alias.status.value,
                        payload=_dump(alias),
                    )
                )
                return
            row.source_key = alias.source_key
            row.target_key = alias.target_key
            row.status = alias.status.value
            row.payload = _dump(alias)

    def get_alias(self, alias_id: str) -> KeyAlias | None:
        with self._db.session() as session:
            row = session.get(KeyAliasRow, alias_id)
            return KeyAlias.model_validate(row.payload) if row else None

    def approved(self) -> tuple[KeyAlias, ...]:
        """Only ``APPROVED`` participates in Class A coupling, and no threshold changes that."""
        return self._query(KeyAliasRow.status == AliasStatus.APPROVED.value)

    def all_aliases(self) -> tuple[KeyAlias, ...]:
        return self._query()

    def _query(self, *where: Any) -> tuple[KeyAlias, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(KeyAliasRow).where(*where).order_by(KeyAliasRow.alias_id)
            ).all()
            return tuple(KeyAlias.model_validate(row.payload) for row in rows)


# ---------------------------------------------------------------------------
# CouplingIndexStore
# ---------------------------------------------------------------------------


class SqlCouplingIndexStore(_SqlStore):
    """The materialized coupling projection, replaced wholesale per tenant.

    ``put_*`` deletes the tenant's slice and re-inserts. That is the contract, not an
    optimisation shortcut: a merge would leave a stale cell behind after a source was
    retracted, and that cell would keep producing a finding nobody could trace to a document
    (checkpoint 10.1).

    A caller that wants the three writes to land together opens
    ``Database.unit_of_work()``; otherwise each is its own transaction, exactly as in memory.
    """

    def put_index(self, tenant_id: str, sets: tuple[CouplingSet, ...]) -> None:
        with self._db.session() as session:
            session.execute(delete(CouplingSetRow).where(CouplingSetRow.tenant_id == tenant_id))
            session.flush()
            for ordinal, item in enumerate(sets):
                session.add(
                    CouplingSetRow(
                        tenant_id=tenant_id,
                        ordinal=ordinal,
                        assumption_key=item.key,
                        scope_rendered=item.scope.render(),
                        payload=_dump(item),
                    )
                )

    def put_relations(self, tenant_id: str, relations: tuple[DerivedRelation, ...]) -> None:
        with self._db.session() as session:
            session.execute(
                delete(DerivedRelationRow).where(DerivedRelationRow.tenant_id == tenant_id)
            )
            session.flush()
            for ordinal, relation in enumerate(relations):
                session.add(
                    DerivedRelationRow(
                        tenant_id=tenant_id,
                        relation_id=relation.relation_id,
                        ordinal=ordinal,
                        relationship=relation.relationship.value,
                        payload=_dump(relation),
                    )
                )

    def put_findings(self, tenant_id: str, findings: tuple[ClassAFinding, ...]) -> None:
        with self._db.session() as session:
            session.execute(delete(ClassAFindingRow).where(ClassAFindingRow.tenant_id == tenant_id))
            session.flush()
            for ordinal, finding in enumerate(findings):
                session.add(
                    ClassAFindingRow(
                        tenant_id=tenant_id,
                        finding_id=finding.finding_id,
                        ordinal=ordinal,
                        assumption_key=finding.assumption_key,
                        rule_id=finding.rule_id,
                        payload=_dump(finding),
                    )
                )

    def coupling_sets(self, tenant_id: str) -> tuple[CouplingSet, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(CouplingSetRow)
                .where(CouplingSetRow.tenant_id == tenant_id)
                .order_by(CouplingSetRow.ordinal)
            ).all()
            return tuple(CouplingSet.model_validate(row.payload) for row in rows)

    def relations(self, tenant_id: str) -> tuple[DerivedRelation, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(DerivedRelationRow)
                .where(DerivedRelationRow.tenant_id == tenant_id)
                .order_by(DerivedRelationRow.ordinal)
            ).all()
            return tuple(DerivedRelation.model_validate(row.payload) for row in rows)

    def findings(self, tenant_id: str) -> tuple[ClassAFinding, ...]:
        return self._findings(ClassAFindingRow.tenant_id == tenant_id)

    def findings_for_key(self, tenant_id: str, key: str) -> tuple[ClassAFinding, ...]:
        return self._findings(
            ClassAFindingRow.tenant_id == tenant_id,
            ClassAFindingRow.assumption_key == key,
        )

    def _findings(self, *where: Any) -> tuple[ClassAFinding, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(ClassAFindingRow).where(*where).order_by(ClassAFindingRow.ordinal)
            ).all()
            return tuple(ClassAFinding.model_validate(row.payload) for row in rows)


# ---------------------------------------------------------------------------
# DispositionStore
# ---------------------------------------------------------------------------


class SqlDispositionStore(_SqlStore):
    """Reviewer verdicts, keyed on ``(analysis_id, consequence_id)``.

    ``put`` replaces the row. It does not merge and it does not delete: the
    :class:`DispositionRecord` handed in already carries its superseded verdicts and its
    transition history, so replacing keeps the audit trail intact while making the *current*
    answer unambiguous.
    """

    def put(self, record: DispositionRecord) -> None:
        analysis_id, consequence_id = record.key
        scope = record.suppression_scope
        with self._db.session() as session:
            row = session.get(FindingDispositionRow, (analysis_id, consequence_id))
            payload = encode_disposition(record)
            if row is None:
                session.add(
                    FindingDispositionRow(
                        analysis_id=analysis_id,
                        consequence_id=consequence_id,
                        disposition=record.disposition.value,
                        suppression_key=scope[0] if scope else None,
                        suppression_artifact_ref=scope[1] if scope else None,
                        payload=payload,
                    )
                )
                return
            row.disposition = record.disposition.value
            row.suppression_key = scope[0] if scope else None
            row.suppression_artifact_ref = scope[1] if scope else None
            row.payload = payload

    def get(self, analysis_id: str, consequence_id: str) -> DispositionRecord | None:
        with self._db.session() as session:
            row = session.get(FindingDispositionRow, (analysis_id, consequence_id))
            return decode_disposition(row.payload) if row else None

    def for_analysis(self, analysis_id: str) -> tuple[DispositionRecord, ...]:
        with self._db.session() as session:
            rows = session.scalars(
                select(FindingDispositionRow)
                .where(FindingDispositionRow.analysis_id == analysis_id)
                .order_by(FindingDispositionRow.consequence_id)
            ).all()
            return tuple(decode_disposition(row.payload) for row in rows)

    def find_consequence(self, consequence_id: str) -> tuple[DispositionRecord, ...]:
        """Every analysis that has ruled on this consequence.

        Exists so that a verdict naming a consequence belonging to a *different* analysis is
        distinguishable from a typo, which is a wiring defect worth reporting differently.
        """
        with self._db.session() as session:
            rows = session.scalars(
                select(FindingDispositionRow)
                .where(FindingDispositionRow.consequence_id == consequence_id)
                .order_by(FindingDispositionRow.analysis_id)
            ).all()
            return tuple(decode_disposition(row.payload) for row in rows)


# ---------------------------------------------------------------------------
# AnalysisStore
# ---------------------------------------------------------------------------


class SqlAnalysisStore(_SqlStore):
    """Stored analyses, addressable by ``(tenant_id, analysis_id)``.

    A cross-tenant read **raises** rather than returning ``None``, matching the in-memory
    store: "not found" is an ordinary miss a UI can render, while a request for another
    tenant's analysis is an isolation defect, and one answer for both would hide the one that
    needs investigating.

    The row is fetched by id and the tenant compared afterwards *on purpose* here - the
    opposite of ``SqlSourceStore.get_document``. Putting the tenant in the predicate would
    turn a cross-tenant request into an indistinguishable miss, which is precisely the
    behaviour this port rejects.
    """

    def put(self, analysis: StoredAnalysis) -> None:
        from app.application.orchestration.analysis import StageStatus

        succeeded = not any(outcome.status is StageStatus.FAILED for outcome in analysis.stages)
        with self._db.session() as session:
            row = session.get(StoredAnalysisRow, analysis.analysis_id)
            payload = encode_analysis(analysis)
            if row is None:
                session.add(
                    StoredAnalysisRow(
                        analysis_id=analysis.analysis_id,
                        tenant_id=analysis.tenant_id,
                        run_id=analysis.run_id,
                        pipeline_version=analysis.pipeline_version,
                        snapshot_version=analysis.snapshot_version,
                        stored_at=analysis.stored_at,
                        succeeded=succeeded,
                        payload=payload,
                    )
                )
                return
            row.tenant_id = analysis.tenant_id
            row.run_id = analysis.run_id
            row.pipeline_version = analysis.pipeline_version
            row.snapshot_version = analysis.snapshot_version
            row.stored_at = analysis.stored_at
            row.succeeded = succeeded
            row.payload = payload

    def get(self, tenant_id: str, analysis_id: str) -> StoredAnalysis | None:
        with self._db.session() as session:
            row = session.get(StoredAnalysisRow, analysis_id)
            if row is None:
                return None
            if row.tenant_id != tenant_id:
                raise TenantMismatch(
                    f"analysis {analysis_id} belongs to another tenant; refusing to serve it "
                    f"to {tenant_id!r}. A cross-tenant read is an isolation defect, not a miss."
                )
            return decode_analysis(row.payload)

    def require(self, tenant_id: str, analysis_id: str) -> StoredAnalysis:
        found = self.get(tenant_id, analysis_id)
        if found is None:
            raise UnknownStoredAnalysis(f"no stored analysis {analysis_id} for tenant {tenant_id}")
        return found

    def for_tenant(self, tenant_id: str) -> tuple[str, ...]:
        with self._db.session() as session:
            return tuple(
                session.scalars(
                    select(StoredAnalysisRow.analysis_id)
                    .where(StoredAnalysisRow.tenant_id == tenant_id)
                    .order_by(StoredAnalysisRow.analysis_id)
                ).all()
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SqlStores:
    """All seven stores over one :class:`Database`. The composition seam.

    One handle means one connection pool and one place a unit of work can span several stores,
    which is the only reason to construct them together rather than one at a time.
    """

    database: Database
    sources: SqlSourceStore
    assumptions: SqlAssumptionStore
    gaps: SqlRegistryGapStore
    aliases: SqlAliasStore
    coupling: SqlCouplingIndexStore
    dispositions: SqlDispositionStore
    analyses: SqlAnalysisStore

    @classmethod
    def over(cls, database: Database) -> SqlStores:
        return cls(
            database=database,
            sources=SqlSourceStore(database),
            assumptions=SqlAssumptionStore(database),
            gaps=SqlRegistryGapStore(database),
            aliases=SqlAliasStore(database),
            coupling=SqlCouplingIndexStore(database),
            dispositions=SqlDispositionStore(database),
            analyses=SqlAnalysisStore(database),
        )

    @classmethod
    def configured(cls) -> SqlStores:
        """From ``CCE_DATABASE_URL``. Raises ``DatabaseNotConfigured`` when it is unset."""
        return cls.over(Database.configured())


def _next_ordinal(session: Any, row_type: Any, *where: Any) -> int:
    """The next insertion ordinal for a scope. ``MAX(ordinal) + 1``, or 0 when empty.

    Not a sequence: the ordinal is scoped (per document, per store) and it has to survive a
    delete-and-replace of a *different* scope untouched, which a global sequence would not
    give. Read inside the caller's transaction, so two concurrent appends to the same document
    serialize on the unique constraint rather than silently sharing an ordinal.
    """
    statement = select(func.max(row_type.ordinal))
    if where:
        statement = statement.where(*where)
    current = session.scalars(statement).one_or_none()
    return 0 if current is None else int(current) + 1
