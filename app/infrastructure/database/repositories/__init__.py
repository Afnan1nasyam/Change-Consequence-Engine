"""Store implementations: in-memory for tests, SQL for production.

Both sets satisfy the same Protocols, and one conformance suite
(``tests/integration/test_sql_repositories.py``) runs against both - so "the SQL store behaves
like the in-memory one" is asserted rather than assumed.
"""

from __future__ import annotations

from app.infrastructure.database.repositories.memory import (
    InMemoryAliasStore,
    InMemoryAnalysisStore,
    InMemoryAssumptionStore,
    InMemoryCouplingIndexStore,
    InMemoryDispositionStore,
    InMemoryRegistryGapStore,
    InMemorySourceStore,
)
from app.infrastructure.database.repositories.sql import (
    SqlAliasStore,
    SqlAnalysisStore,
    SqlAssumptionStore,
    SqlCouplingIndexStore,
    SqlDispositionStore,
    SqlRegistryGapStore,
    SqlSourceStore,
    SqlStores,
)

__all__ = [
    "InMemoryAliasStore",
    "InMemoryAnalysisStore",
    "InMemoryAssumptionStore",
    "InMemoryCouplingIndexStore",
    "InMemoryDispositionStore",
    "InMemoryRegistryGapStore",
    "InMemorySourceStore",
    "SqlAliasStore",
    "SqlAnalysisStore",
    "SqlAssumptionStore",
    "SqlCouplingIndexStore",
    "SqlDispositionStore",
    "SqlRegistryGapStore",
    "SqlSourceStore",
    "SqlStores",
]
