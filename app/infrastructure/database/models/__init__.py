"""ORM model registry.

Importing this package registers every table on ``Base.metadata``, which is what Alembic's
``env.py`` relies on for autogenerate. Re-exported here so a caller never has to know which
module a table lives in.
"""

from __future__ import annotations

from app.infrastructure.database.models.tables import (
    ALL_TABLES,
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

__all__ = [
    "ALL_TABLES",
    "AssumptionRecordRow",
    "ChunkRow",
    "ClassAFindingRow",
    "CouplingSetRow",
    "DerivedRelationRow",
    "EvidenceItemRow",
    "FindingDispositionRow",
    "KeyAliasRow",
    "RegistryGapRow",
    "SourceDocumentRow",
    "SourceVersionRow",
    "StoredAnalysisRow",
]
