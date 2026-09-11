"""The relational schema. One table per stored aggregate, payload as JSONB.

The mapping strategy, and why it is this one
--------------------------------------------

Every row carries the frozen domain entity verbatim in a ``payload`` JSONB column, plus the
handful of **projected columns** the ports actually query on - ids, tenant, assumption key,
source version. Reads are ``Model.model_validate(row.payload)``.

This is a document mapping with relational indexes, not a normalized schema, and that is a
deliberate choice against the obvious alternative:

**Normalizing would create a second definition of every domain type.** The entities nest
``TypedValue``, ``ScopeSelector``, ``ModelProvenance``, ``RunProvenance``, ``PermissionScope``
and a dozen more frozen value objects, several of them with model validators that encode rules
(an approved alias names a human and a time; an interval is ordered; a blocked coupling
strength is zero). Spreading those across columns would put the same rules in two places - the
Pydantic validator and a CHECK constraint - and the two would drift. Round-tripping through
``model_validate`` instead means **every read re-runs every validator**, so a row that could not
be constructed by the domain cannot be served by the store either.

**The ports do not ask relational questions.** Every method in
``app/domain/interfaces/repositories.py`` is an id lookup, a tenant enumeration, or a filter on
one key. There is no join, no aggregate and no range scan in the contract, so the columns below
are exactly what the contract needs and nothing is indexed speculatively.

What is given up is queryability *inside* an entity from SQL. JSONB keeps that reachable
(``payload -> 'status'``) and a GIN index can be added per query when one is actually needed;
until then an index nobody queries is a write cost with no reader.

**Projected columns are write-time projections, never read back into an entity.** A column like
``status`` exists so a filter is indexable; the value served to a caller always comes from
``payload``. So the two cannot disagree in a way a reader could observe - the worst case is a
stale filter, which the ``put_*`` paths make impossible by recomputing on every write.

Portability
-----------

``JSONB`` carries a ``JSON`` variant for SQLite so the *same* repository code can be exercised
without a Postgres server (``tests/integration/test_sql_repositories.py``). The production
target is Postgres; nothing here uses a Postgres-only construct beyond the column type, and the
repositories deliberately avoid dialect-specific upsert so behaviour is identical on both.

Ordinals
--------

Several ports promise insertion order (``versions_for`` is "oldest first";
``coupling_sets`` returns the tuple as written). SQL has no insertion order, so it is stored:
an ``ordinal`` column assigned at write time reproduces exactly what the in-memory
implementation returns. Without it the two implementations would satisfy the type and disagree
about the answer.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base

#: JSONB on Postgres, JSON elsewhere. The variant is what lets the repository tests run with
#: no server while the production dialect keeps the indexable, binary representation.
Payload = JSONB().with_variant(JSON(), "sqlite")

#: Every identifier in this system is a prefixed 32-hex-character string. 64 is generous and
#: uniform, which matters more than tightness: a per-table width would be one more thing that
#: can disagree with `app/domain/value_objects/identifiers.py`.
ID_LEN = 64

#: An assumption key or a document URI. Both are bounded in the domain at 256 / 1024.
KEY_LEN = 256
URI_LEN = 1024


class _Row(Base):
    """Shared column conventions. Abstract, so it creates no table of its own."""

    __abstract__ = True

    payload: Mapped[dict[str, Any]] = mapped_column(Payload, nullable=False)
    #: When this row was written. Operational metadata, never part of the entity.
    written_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(tz=dt.UTC)
    )


# ---------------------------------------------------------------------------
# SourceStore
# ---------------------------------------------------------------------------


class SourceDocumentRow(_Row):
    __tablename__ = "source_documents"

    source_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False, index=True)
    uri: Mapped[str] = mapped_column(String(URI_LEN), nullable=False)

    __table_args__ = (
        # `find_document_by_uri` is a tenant-scoped lookup and ingestion relies on it being
        # unique: two documents at one uri for one tenant would make re-ingestion ambiguous.
        UniqueConstraint("tenant_id", "uri", name="uq_source_documents_tenant_uri"),
    )


class SourceVersionRow(_Row):
    __tablename__ = "source_versions"

    version_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String(ID_LEN), ForeignKey("source_documents.source_id"), nullable=False, index=True
    )
    #: Insertion order within one document. `versions_for` promises oldest first.
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The canonical text the offsets index into. Stored through the same port as the
    #: metadata, because a store that could serve one without the other could hand out
    #: unverifiable evidence.
    canonical_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: The extractor version this version was last extracted under. Nullable: never
    #: extracted is a distinct state from extracted by an unknown extractor.
    extractor_version: Mapped[str | None] = mapped_column(String(ID_LEN), nullable=True)

    __table_args__ = (UniqueConstraint("source_id", "ordinal", name="uq_source_versions_ordinal"),)


class ChunkRow(_Row):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    source_version_id: Mapped[str] = mapped_column(
        String(ID_LEN), ForeignKey("source_versions.version_id"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class EvidenceItemRow(_Row):
    __tablename__ = "evidence_items"

    evidence_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    source_version_id: Mapped[str] = mapped_column(
        String(ID_LEN), ForeignKey("source_versions.version_id"), nullable=False, index=True
    )


# ---------------------------------------------------------------------------
# AssumptionStore
# ---------------------------------------------------------------------------


class AssumptionRecordRow(_Row):
    __tablename__ = "assumption_records"

    assumption_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    assumption_key: Mapped[str] = mapped_column(String(KEY_LEN), nullable=False)
    holder_source_version_id: Mapped[str] = mapped_column(
        String(ID_LEN), ForeignKey("source_versions.version_id"), nullable=False, index=True
    )
    #: Global insertion order. `records_for_version` and `all_records` both promise the order
    #: they were written in, and one counter reproduces both.
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        # `records_for_key` is the resolver's hot path: one key, one tenant.
        Index("ix_assumption_records_tenant_key", "tenant_id", "assumption_key"),
    )


# ---------------------------------------------------------------------------
# RegistryGapStore
# ---------------------------------------------------------------------------


class RegistryGapRow(_Row):
    __tablename__ = "registry_gaps"

    gap_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    proposed_key: Mapped[str] = mapped_column(String(KEY_LEN), nullable=False, index=True)
    #: The underlying enum value, not the derived `blocks_class_a` flag. Filtering on the
    #: field the property reads keeps one definition of "blocks", in the entity.
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False)


# ---------------------------------------------------------------------------
# AliasStore
# ---------------------------------------------------------------------------


class KeyAliasRow(_Row):
    __tablename__ = "key_aliases"

    alias_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    source_key: Mapped[str] = mapped_column(String(KEY_LEN), nullable=False, index=True)
    target_key: Mapped[str] = mapped_column(String(KEY_LEN), nullable=False)
    #: As above: `participates_in_class_a` is `status is APPROVED`, and the entity owns that.
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)


# ---------------------------------------------------------------------------
# CouplingIndexStore - a projection, replaced wholesale per tenant
# ---------------------------------------------------------------------------


class CouplingSetRow(_Row):
    """No natural single-column key: identity is ``(key, scope, as_of)``.

    A surrogate ``(tenant_id, ordinal)`` is used instead of hashing that triple, because the
    port replaces the whole tenant slice on every write - so the ordinal *is* the identity for
    as long as the row exists, and it also preserves the order the resolver produced.
    """

    __tablename__ = "coupling_sets"

    tenant_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    assumption_key: Mapped[str] = mapped_column(String(KEY_LEN), nullable=False)
    #: ``ScopeSelector.render()``, the canonical string form. Stored rendered rather than
    #: structured because the only question asked of it here is equality.
    scope_rendered: Mapped[str] = mapped_column(String(KEY_LEN), nullable=False)


class DerivedRelationRow(_Row):
    __tablename__ = "derived_relations"

    tenant_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    relation_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    relationship: Mapped[str] = mapped_column(String(64), nullable=False)


class ClassAFindingRow(_Row):
    __tablename__ = "class_a_findings"

    tenant_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    finding_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    #: `findings_for_key` is the query a reviewer works from.
    assumption_key: Mapped[str] = mapped_column(String(KEY_LEN), nullable=False)
    rule_id: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (Index("ix_class_a_findings_tenant_key", "tenant_id", "assumption_key"),)


# ---------------------------------------------------------------------------
# DispositionStore
# ---------------------------------------------------------------------------


class FindingDispositionRow(_Row):
    """One reviewer verdict per ``(analysis_id, consequence_id)``, with its audit trail.

    The history and the superseded verdicts live inside ``payload`` rather than in a child
    table. They are only ever read as part of the record - there is no query in the port for
    "every transition across all findings" - and a child table would let a row exist whose
    history disagreed with its current verdict.
    """

    __tablename__ = "finding_dispositions"

    analysis_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    consequence_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    disposition: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    #: The two halves of a persistent suppression's scope, projected so a future analysis can
    #: find the standing verdicts for a key without loading every row. Null when the reviewer
    #: did not ask for persistent suppression.
    suppression_key: Mapped[str | None] = mapped_column(String(KEY_LEN), nullable=True)
    suppression_artifact_ref: Mapped[str | None] = mapped_column(String(ID_LEN), nullable=True)

    __table_args__ = (
        Index("ix_finding_dispositions_consequence", "consequence_id"),
        Index("ix_finding_dispositions_suppression", "suppression_key", "suppression_artifact_ref"),
    )


# ---------------------------------------------------------------------------
# AnalysisStore
# ---------------------------------------------------------------------------


class StoredAnalysisRow(_Row):
    """The review snapshot of one finished run.

    ``tenant_id`` is a real column and every read is scoped by it, which is what makes the
    cross-tenant refusal a property of the query rather than of a caller remembering to check.
    """

    __tablename__ = "stored_analyses"

    analysis_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_version: Mapped[str] = mapped_column(String(64), nullable=False)
    stored_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Whether the run reported any failed stage. Projected because "show me the analyses that
    #: completed cleanly" is the first question an operator asks and it should not need a
    #: JSON traversal.
    succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False)


#: Every table, in dependency order. Used by the migration-parity contract test so a new model
#: that nobody migrated is a failure rather than a surprise.
ALL_TABLES = (
    SourceDocumentRow,
    SourceVersionRow,
    ChunkRow,
    EvidenceItemRow,
    AssumptionRecordRow,
    RegistryGapRow,
    KeyAliasRow,
    CouplingSetRow,
    DerivedRelationRow,
    ClassAFindingRow,
    FindingDispositionRow,
    StoredAnalysisRow,
)
