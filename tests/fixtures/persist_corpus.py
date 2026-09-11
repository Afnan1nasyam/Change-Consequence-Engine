"""Write a complete analysis to a database, as a separate process would.

Run as ``python -m tests.fixtures.persist_corpus <database-url>``. It migrates the schema,
ingests the real sample corpus through the real ingestion path, resolves the coupling
projection, runs the whole twelve-stage pipeline, records two reviewer verdicts, and prints a
JSON manifest of what it wrote.

It exists to be invoked in a **subprocess**, because that is the only honest way to test
durability across a restart: a second engine in the same process shares an import graph, a
settings cache and any accidental in-memory state, and a test that reads back through it can
pass while nothing was ever written to disk. A separate interpreter shares nothing but the
database.

The assembly is imported from ``tests.integration.test_analysis_pipeline`` rather than
duplicated. A second copy of the wiring would let the persisted pipeline drift from the one
the evaluation suite measures, and then "it survives a restart" would be true of something
else.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
from typing import Any

from app.application.use_cases.record_disposition import RecordDispositionUseCase
from app.application.use_cases.resolve_coupling import ResolveCouplingUseCase
from app.domain.value_objects.enums import Disposition
from app.domain.value_objects.identifiers import new_id
from app.evaluation.corpus import CorpusStores, load_corpus
from app.infrastructure.database.repositories.sql import (
    SqlAliasStore,
    SqlAnalysisStore,
    SqlAssumptionStore,
    SqlCouplingIndexStore,
    SqlDispositionStore,
    SqlRegistryGapStore,
    SqlSourceStore,
)
from app.infrastructure.database.session import Database

AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 9, 11, 12, 0, tzinfo=dt.UTC)
REVIEWER = "privacy-office/reviewer-1"


def sql_corpus_stores(database: Database) -> CorpusStores:
    """The four ingestion stores, SQL-backed. Tenant and registry version are overwritten."""
    return CorpusStores(
        tenant_id="",
        registry_version="",
        sources=SqlSourceStore(database),
        assumptions=SqlAssumptionStore(database),
        gaps=SqlRegistryGapStore(database),
        aliases=SqlAliasStore(database),
    )


def write_everything(database: Database) -> dict[str, Any]:
    """Ingest, resolve, analyse, review. Returns a manifest the reader asserts against."""
    from tests.integration import test_analysis_pipeline as wiring

    corpus = asyncio.run(load_corpus(into=sql_corpus_stores(database)))
    assert not corpus.extraction_failures, corpus.extraction_failures

    coupling_store = SqlCouplingIndexStore(database)
    coupling = ResolveCouplingUseCase(
        sources=corpus.sources,
        assumptions=corpus.assumptions,
        gaps=corpus.gaps,
        index_store=coupling_store,
    )
    # One transaction for the three projection writes, which is the case
    # `Database.unit_of_work` exists for.
    with database.unit_of_work():
        analysis_of_corpus = coupling.execute(
            tenant_id=corpus.tenant_id,
            as_of=AS_OF,
            registry_version=corpus.registry_version,
        )

    analyses = SqlAnalysisStore(database)
    dispositions = SqlDispositionStore(database)
    use_case, _provider = wiring.pipeline(corpus, analyses=analyses, dispositions=dispositions)
    change = wiring.retention_change()
    analysis_id = new_id("analysis")
    result = asyncio.run(
        use_case.execute(
            change=change,
            query=wiring.query_for(corpus, "personal data retention window after account closure"),
            tenant_id=corpus.tenant_id,
            registry_version=corpus.registry_version,
            as_of=AS_OF,
            analysis_id=analysis_id,
            run_id=new_id("analysis"),
            now=NOW,
        )
    )

    reviews = RecordDispositionUseCase(store=dispositions, analyses=analyses)
    stored = analyses.require(corpus.tenant_id, analysis_id)
    # Two targets in distinct suppression scopes, so one verdict does not cover both.
    by_scope: dict[tuple[str, str], Any] = {}
    for finding in stored.findings:
        version = finding.artifact_version_id
        if version is None:
            continue
        for key in finding.consequence.assumption_keys:
            by_scope.setdefault((key, version), finding)
    scopes = sorted(by_scope)
    rejected, known = by_scope[scopes[0]], by_scope[scopes[1]]

    reviews.record_reviewed(
        tenant_id=corpus.tenant_id,
        analysis_id=analysis_id,
        consequence_id=rejected.consequence_id,
        disposition=Disposition.WRONG,
        disposed_by=REVIEWER,
        disposed_at=NOW,
        note="The declaring deck was reissued before this change.",
        suppress_until_source_revision=True,
    )
    reviews.record_reviewed(
        tenant_id=corpus.tenant_id,
        analysis_id=analysis_id,
        consequence_id=known.consequence_id,
        disposition=Disposition.ALREADY_KNOWN,
        disposed_by=REVIEWER,
        disposed_at=NOW,
    )
    # And a change of mind, so the audit trail has something in it to survive.
    reviews.change(
        analysis_id=analysis_id,
        consequences=stored.consequences,
        consequence_id=known.consequence_id,
        disposition=Disposition.USEFUL,
        changed_by=REVIEWER,
        changed_at=NOW + dt.timedelta(hours=1),
        reason="Reviewed again with the runbook to hand; this was new after all.",
    )

    assert result.report is not None
    return {
        "tenant_id": corpus.tenant_id,
        "registry_version": corpus.registry_version,
        "analysis_id": analysis_id,
        "run_id": result.run_id,
        "change_id": change.change_id,
        "fingerprint": result.fingerprint(),
        "documents": len(corpus.reports),
        "assumption_records": len(corpus.assumptions.all_records(corpus.tenant_id)),
        "open_gaps": len(corpus.gaps.open_gaps()),
        "coupling_sets": len(analysis_of_corpus.index.sets),
        "relations": len(analysis_of_corpus.index.relations),
        "standing_findings": len(analysis_of_corpus.findings),
        "class_a": len(result.class_a),
        "consequences": sorted(c.consequence_id for c in result.consequences),
        "evidence_ids": sorted(result.retrieval.evidence_ids),
        "foregrounded": sorted(f.consequence_id for f in result.report.high_priority),
        "suppressed_consequence_id": rejected.consequence_id,
        "suppressed_key": scopes[0][0],
        "suppressed_artifact_version_id": scopes[0][1],
        "changed_consequence_id": known.consequence_id,
        "graph_edges": len(result.graph.graph.edges) if result.graph else 0,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m tests.fixtures.persist_corpus <database-url>", file=sys.stderr)
        return 2

    from tests.fixtures.databases import migrate, pointed_at

    url = argv[1]
    with pointed_at(url):
        migrate(url)
        database = Database.from_url(url)
        try:
            manifest = write_everything(database)
        finally:
            database.dispose()
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main(sys.argv))
