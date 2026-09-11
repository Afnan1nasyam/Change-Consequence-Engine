"""Does the state survive a process restart? Written by one interpreter, read by another.

This is the question the persistence milestone exists to answer, and it is the one a test in a
single process cannot honestly answer. A second engine in the same interpreter shares an import
graph, a settings cache and any accidental module-level state, so a read-back can succeed while
nothing ever reached disk.

So the writer is a **subprocess**: ``python -m tests.fixtures.persist_corpus <url>`` migrates
the schema, ingests the real sample corpus through the real ingestion path, resolves the
coupling projection, runs the full twelve-stage pipeline, records two reviewer verdicts and one
reversal, then exits. Everything below runs afterwards, in this process, against nothing but
the database file it left behind.

What is verified, in the order the requirement asks for it: the analysis snapshot, its
findings, the reviewer dispositions and their audit trail, suppression expiry, the evidence a
citation resolves to, and every other port's state.

Substrate: SQLite by default so this runs in CI with no infrastructure; the same tests run
against Postgres when ``CCE_TEST_DATABASE_URL`` is set. See ``tests/fixtures/databases.py`` for
what each substrate does and does not prove.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.application.orchestration.analysis import StageStatus
from app.application.use_cases.record_disposition import RecordDispositionUseCase
from app.domain.value_objects.enums import Disposition
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
from tests.fixtures.databases import configured_url, is_postgres, sqlite_url

pytestmark = pytest.mark.integration

OTHER_TENANT = "tn_" + "9" * 32


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Database, dict[str, Any]]]:
    """Run the writer in a separate interpreter, then hand back a fresh handle and its manifest.

    Module-scoped because the write is the expensive part (a full corpus ingestion and a full
    analysis) and every test below reads the same result. Each test still opens its own
    repositories, so nothing is shared between them but the bytes on disk.
    """
    url = configured_url() or sqlite_url(tmp_path_factory.mktemp("restart") / "cce.db")
    completed = subprocess.run(
        [sys.executable, "-m", "tests.fixtures.persist_corpus", url],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    manifest = json.loads(completed.stdout.strip().splitlines()[-1])

    database = Database.from_url(url)
    try:
        yield database, manifest
    finally:
        database.dispose()


class TestTheAnalysisSnapshot:
    def test_the_analysis_is_retrievable_by_id_after_a_restart(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        database, manifest = written
        store = SqlAnalysisStore(database)

        assert store.for_tenant(manifest["tenant_id"]) == (manifest["analysis_id"],)
        stored = store.require(manifest["tenant_id"], manifest["analysis_id"])
        assert stored.analysis_id == manifest["analysis_id"]
        assert stored.run_id == manifest["run_id"]
        assert stored.change.change_id == manifest["change_id"]
        assert stored.pipeline_version == manifest["fingerprint"]["pipeline_version"]

    def test_the_fingerprint_survives_verbatim(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """The comparison that lets a verdict be checked against the analysis that produced it.

        Verbatim, not recomputed: a codec that regenerated it would make a re-run look like
        the same analysis, which is precisely the confusion the fingerprint exists to prevent.
        """
        database, manifest = written
        stored = SqlAnalysisStore(database).require(manifest["tenant_id"], manifest["analysis_id"])

        assert stored.fingerprint == manifest["fingerprint"]

    def test_every_stage_outcome_survives_with_its_status_and_count(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """`empty` and `failed` are different facts, and they have to stay different."""
        database, manifest = written
        stored = SqlAnalysisStore(database).require(manifest["tenant_id"], manifest["analysis_id"])

        rebuilt = {
            outcome.stage.value: [outcome.status.value, outcome.produced]
            for outcome in stored.stages
        }
        assert rebuilt == manifest["fingerprint"]["stages"]
        assert all(isinstance(outcome.status, StageStatus) for outcome in stored.stages)

    def test_every_finding_survives_with_its_decision_and_score(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        database, manifest = written
        stored = SqlAnalysisStore(database).require(manifest["tenant_id"], manifest["analysis_id"])

        assert sorted(stored.consequence_ids) == manifest["consequences"]
        for finding in stored.findings:
            assert finding.decision is not None
            assert finding.score is not None
            assert finding.decision.consequence_id == finding.consequence_id
            assert finding.score.consequence_id == finding.consequence_id
            # The computed confidence, not a re-derivation of it.
            assert 0.0 <= finding.score.confidence <= 1.0
        assert sorted(stored.foregrounded) == manifest["foregrounded"]

    def test_the_graph_projection_survives(self, written: tuple[Database, dict[str, Any]]) -> None:
        database, manifest = written
        stored = SqlAnalysisStore(database).require(manifest["tenant_id"], manifest["analysis_id"])

        assert stored.graph is not None
        assert stored.graph.is_projection is True
        assert len(stored.graph.edges) == manifest["graph_edges"]

    def test_the_evidence_a_citation_resolves_to_survives(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """A reviewer who can read a claim and not its evidence is not reviewing."""
        database, manifest = written
        stored = SqlAnalysisStore(database).require(manifest["tenant_id"], manifest["analysis_id"])

        resolved = stored.evidence_by_id
        assert set(manifest["evidence_ids"]) <= set(resolved)
        for finding in stored.findings:
            for evidence_id in finding.consequence.evidence_ids:
                assert evidence_id in resolved
                assert resolved[evidence_id].text_span

    def test_a_cross_tenant_read_raises_rather_than_returning_nothing(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """An isolation defect must not be indistinguishable from an ordinary miss."""
        from app.application.orchestration.analysis_store import TenantMismatch

        database, manifest = written
        store = SqlAnalysisStore(database)

        with pytest.raises(TenantMismatch, match="another tenant"):
            store.get(OTHER_TENANT, manifest["analysis_id"])
        assert store.for_tenant(OTHER_TENANT) == ()
        # A genuine miss is still a miss.
        assert store.get(manifest["tenant_id"], "an_" + "0" * 32) is None


class TestReviewerState:
    def test_both_verdicts_survive_with_their_reviewer_and_timestamp(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        database, manifest = written
        records = SqlDispositionStore(database).for_analysis(manifest["analysis_id"])

        assert len(records) == 2
        for record in records:
            assert record.current.disposed_by == "privacy-office/reviewer-1"
            assert record.current.disposed_at is not None
            assert record.current.analysis_id == manifest["analysis_id"]

    def test_the_audit_trail_survives_the_restart(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """The reversal, both of its values, who made it and why.

        "Changed to useful" is not auditable without knowing what it was, which is the whole
        reason the history is stored rather than only the current verdict.
        """
        database, manifest = written
        record = SqlDispositionStore(database).get(
            manifest["analysis_id"], manifest["changed_consequence_id"]
        )

        assert record is not None
        assert record.disposition is Disposition.USEFUL
        assert record.was_changed
        assert record.original is Disposition.ALREADY_KNOWN
        [change] = record.history
        assert change.from_disposition is Disposition.ALREADY_KNOWN
        assert change.to_disposition is Disposition.USEFUL
        assert change.reason
        assert change.changed_by == "privacy-office/reviewer-1"
        # And the withdrawn verdict is kept as it was recorded.
        assert len(record.superseded) == 1
        assert record.superseded[0].disposition is Disposition.ALREADY_KNOWN

    def test_the_suppressing_verdict_survives_with_its_scope(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        database, manifest = written
        record = SqlDispositionStore(database).get(
            manifest["analysis_id"], manifest["suppressed_consequence_id"]
        )

        assert record is not None
        assert record.disposition is Disposition.WRONG
        assert record.suppresses_foreground
        assert record.suppression_scope == (
            manifest["suppressed_key"],
            manifest["suppressed_artifact_version_id"],
        )

    def test_suppression_is_readable_by_a_new_process_and_expires_on_a_revision(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """Both halves of the rule, across a restart.

        The suppression applies to the version the reviewer was looking at and to no other:
        they said the finding was wrong about *that* text, and a revised document is different
        text. Expiry is the same ``suppression_has_expired`` the in-memory path uses - there is
        no second rule on the SQL side.
        """
        database, manifest = written
        reviews = RecordDispositionUseCase(
            store=SqlDispositionStore(database), analyses=SqlAnalysisStore(database)
        )
        tenant_id = manifest["tenant_id"]

        standing = reviews.suppressions(tenant_id=tenant_id)
        assert len(standing) == 1
        assert standing[0].scope == (
            manifest["suppressed_key"],
            manifest["suppressed_artifact_version_id"],
        )

        assert reviews.is_suppressed(
            tenant_id=tenant_id,
            assumption_key=manifest["suppressed_key"],
            artifact_version_id=manifest["suppressed_artifact_version_id"],
        )
        # A revised source version has a different content-derived id, so the suppression
        # lifts without anybody withdrawing it.
        assert not reviews.is_suppressed(
            tenant_id=tenant_id,
            assumption_key=manifest["suppressed_key"],
            artifact_version_id="sv_" + "e" * 32,
        )
        # And a Class B finding, which has no single source version, is never caught by one.
        assert not reviews.is_suppressed(
            tenant_id=tenant_id,
            assumption_key=manifest["suppressed_key"],
            artifact_version_id=None,
        )

    def test_the_already_known_signal_is_measurable_after_a_restart(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """The verdict that suppresses nothing still has to be countable, or the thesis is not.

        The reversal moved this finding to ``useful``, so the *current* already-known count is
        zero - and the history still records that it was already known once, which is the
        distinction a rate over current verdicts would lose.
        """
        database, manifest = written
        reviews = RecordDispositionUseCase(
            store=SqlDispositionStore(database), analyses=SqlAnalysisStore(database)
        )

        verdicts = reviews.verdicts(tenant_id=manifest["tenant_id"])
        assert {v.disposition for v in verdicts} == {Disposition.WRONG, Disposition.USEFUL}
        assert all(v.artifact_version_id for v in verdicts)

        signals = reviews.signals_reviewed(
            tenant_id=manifest["tenant_id"], analysis_id=manifest["analysis_id"]
        )
        assert signals.reviewed == 2
        record = SqlDispositionStore(database).get(
            manifest["analysis_id"], manifest["changed_consequence_id"]
        )
        assert record is not None and record.original is Disposition.ALREADY_KNOWN

    def test_the_foreground_a_reviewer_now_sees_is_reconstructible(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """The judge's default view, narrowed by the verdicts, both read from the database."""
        database, manifest = written
        reviews = RecordDispositionUseCase(
            store=SqlDispositionStore(database), analyses=SqlAnalysisStore(database)
        )

        view = reviews.foreground_reviewed(
            tenant_id=manifest["tenant_id"], analysis_id=manifest["analysis_id"]
        )
        shown = {c.consequence_id for c in view.shown}
        assert manifest["suppressed_consequence_id"] not in shown
        assert shown < set(manifest["foregrounded"])


class TestTheOtherFivePorts:
    def test_the_ingested_corpus_survives(self, written: tuple[Database, dict[str, Any]]) -> None:
        """Documents, versions, canonical text, chunks and citable spans."""
        database, manifest = written
        sources = SqlSourceStore(database)

        spans = sources.evidence_for_tenant(manifest["tenant_id"])
        assert spans
        assert [s.evidence_id for s in spans] == sorted(s.evidence_id for s in spans)
        assert set(manifest["evidence_ids"]) <= {s.evidence_id for s in spans}

        # Follow one span back to its text, which is the join span verification needs.
        span = next(s for s in spans if s.evidence_id == manifest["evidence_ids"][0])
        version = sources.get_version(span.source_version_id)
        assert version is not None
        text = sources.get_text(span.source_version_id)
        assert text is not None
        assert text[span.char_start : span.char_end] == span.text_span

        document = sources.get_document(manifest["tenant_id"], version.source_id)
        assert document is not None
        assert sources.versions_for(version.source_id)[0].version_id == version.version_id
        assert sources.chunks_for(version.version_id)

    def test_the_extraction_marker_survives_so_re_ingestion_stays_free(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """The cache key for skipping extraction. Losing it means re-extracting the corpus."""
        database, manifest = written
        sources = SqlSourceStore(database)
        span = sources.evidence_for_tenant(manifest["tenant_id"])[0]

        assert sources.extracted_with(span.source_version_id) is not None

    def test_the_admitted_assumption_records_survive(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        database, manifest = written
        store = SqlAssumptionStore(database)

        records = store.all_records(manifest["tenant_id"])
        assert len(records) == manifest["assumption_records"]
        assert all(record.is_admitted for record in records)
        assert store.records_for_key(manifest["tenant_id"], "retention.window")
        assert store.has_records_for_version(records[0].holder_source_version_id)

    def test_the_quarantined_keys_and_their_volume_survive(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """Gap volume is a product metric, so the count has to survive, not just the row."""
        database, manifest = written
        store = SqlRegistryGapStore(database)

        gaps = store.open_gaps()
        assert len(gaps) == manifest["open_gaps"]
        counts = store.gap_counts_by_key()
        assert counts
        assert sum(counts.values()) >= len(gaps)
        assert store.get_gap(gaps[0].gap_id) == gaps[0]

    def test_the_alias_store_survives_and_is_empty_as_the_corpus_leaves_it(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """An empty table read back is still a read. It must not raise."""
        database, _manifest = written
        store = SqlAliasStore(database)

        assert store.all_aliases() == ()
        assert store.approved() == ()

    def test_the_coupling_projection_survives_with_its_order(
        self, written: tuple[Database, dict[str, Any]]
    ) -> None:
        """Written in one unit of work, read back whole.

        The order matters: the resolver produces the tuple and the index is a projection of
        it, so a store that returned the same rows in a different order would make the
        projection depend on physical row layout.
        """
        database, manifest = written
        store = SqlCouplingIndexStore(database)
        tenant_id = manifest["tenant_id"]

        assert len(store.coupling_sets(tenant_id)) == manifest["coupling_sets"]
        assert len(store.relations(tenant_id)) == manifest["relations"]
        findings = store.findings(tenant_id)
        assert len(findings) == manifest["standing_findings"]
        assert store.findings_for_key(tenant_id, "retention.window")
        assert all(
            f.assumption_key == "retention.window"
            for f in store.findings_for_key(tenant_id, "retention.window")
        )
        # And another tenant's slice is empty rather than everyone's.
        assert store.coupling_sets(OTHER_TENANT) == ()
        assert store.findings(OTHER_TENANT) == ()


class TestSubstrate:
    def test_the_tests_report_which_substrate_they_ran_against(self) -> None:
        """Not an assertion about behaviour - a record in the output of what was proved.

        SQLite proves the SQL, the mapping, the codecs and durability. It proves nothing about
        JSONB indexing or Postgres isolation. A reader of a green run needs to know which of
        those two they got.
        """
        url = configured_url()
        if url is None:
            pytest.skip(
                "running on the SQLite substrate; set CCE_TEST_DATABASE_URL to a Postgres "
                "URL to run this suite against the production dialect"
            )
        assert is_postgres(), f"CCE_TEST_DATABASE_URL is not a Postgres URL: {url!r}"
