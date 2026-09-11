"""The composition root, exercised through the path production uses.

Until this existed, the PostgreSQL stores were implemented, migrated and tested and *nothing
constructed them*. So the test that matters is not "can the stores be built" - the persistence
suite already proves that - it is: **does `Application.configured()` produce an object graph
that runs the real CCE analysis path against the real stores.**

Which means `configured()`, not `build()`. The overrides on `configured()` are there so the
shipped function is the tested one; a test that reached for `build()` to avoid needing a
credential would be testing a different function from the one that ships.

Substrate: SQLite by default so this runs in CI with no infrastructure, Postgres when
`CCE_TEST_DATABASE_URL` is set - the same rule and the same caveats as the rest of the
persistence suites (`tests/fixtures/databases.py`). Provider: the cooperative fake, through
the real `LlmGateway`, so there are no credentials and no network.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.application.orchestration.analysis import (
    ANALYSIS_PIPELINE_VERSION,
    AnalysisStage,
)
from app.composition import REQUIRED_AGENTS, Application, CompositionError, main, readiness
from app.config.settings import Settings, get_settings
from app.domain.entities.registry import KeyAlias
from app.domain.value_objects.enums import AliasStatus, Disposition
from app.domain.value_objects.identifiers import new_id
from app.evaluation.corpus import CorpusStores, load_corpus
from app.infrastructure.database.repositories import memory as mem
from app.infrastructure.database.repositories.sql import (
    SqlAnalysisStore,
    SqlSourceStore,
)
from app.infrastructure.llm.providers.fake import FakeProvider
from app.infrastructure.llm.registry import ProviderRegistry
from tests.conftest import make_settings
from tests.fixtures.databases import configured_url, migrate, pointed_at, sqlite_url
from tests.fixtures.fake_agents import cooperative_responder
from tests.integration import test_analysis_pipeline as wiring

pytestmark = pytest.mark.integration

AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 9, 11, 12, 0, tzinfo=dt.UTC)


def app_settings(**over: object) -> Settings:
    """Test settings that keep the fake provider and the offline embedder."""
    return make_settings(embedding_provider="hashing", **over)


def _alias(tag: str, source_key: str, *, approved: bool) -> KeyAlias:
    """One alias pointing at a registered key. ``retention.window`` is in the real registry."""
    return KeyAlias(
        alias_id=f"al_{tag * 32}",
        source_key=source_key,
        target_key="retention.window",
        proposed_at=AS_OF,
        approved_by="registry-owner" if approved else None,
        approved_at=AS_OF if approved else None,
        registry_version="registry-1",
        status=AliasStatus.APPROVED if approved else AliasStatus.PROPOSED,
    )


def fake_providers(settings: Settings) -> ProviderRegistry:
    """The real registry over a cooperative fake. No credentials, no network."""
    return ProviderRegistry(
        settings, overrides={"fake": FakeProvider(responder=cooperative_responder())}
    )


@pytest.fixture(scope="module")
def url(tmp_path_factory: pytest.TempPathFactory) -> str:
    target = configured_url() or sqlite_url(tmp_path_factory.mktemp("composed") / "cce.db")
    migrate(target)
    return target


@pytest.fixture
def application(url: str) -> Iterator[Application]:
    """One application, constructed exactly as production constructs it.

    ``pointed_at`` sets ``CCE_DATABASE_URL`` and clears the settings cache, which is how the
    configured path is reached without a real server in the environment.
    """
    with pointed_at(url):
        settings = app_settings()
        app = Application.configured(settings, providers=fake_providers(settings))
        try:
            yield app
        finally:
            asyncio.run(app.aclose())


class TestConstruction:
    def test_the_configured_application_holds_the_sql_stores(
        self, application: Application
    ) -> None:
        """Through the existing ``SqlStores`` seam - not a second wiring of the same classes."""
        assert application.is_persistent
        assert application.database is not None
        assert isinstance(application.sources, SqlSourceStore)
        assert isinstance(application.analyses, SqlAnalysisStore)
        # And every store shares one handle, so a unit of work can span them.
        assert application.sources._db is application.database

    def test_every_use_case_the_analysis_path_needs_is_wired(
        self, application: Application
    ) -> None:
        assert application.ingest is not None
        assert application.coupling is not None
        assert application.retrieval is not None
        assert application.analyze is not None
        assert application.reviews is not None
        # The registry is loaded and closed, and the agents are all present.
        assert application.registry.keys
        assert set(REQUIRED_AGENTS) <= set(application.definitions)

    def test_the_alias_store_reaches_the_registry_and_only_approvals_resolve(self) -> None:
        """The wiring, and the gate ``CLAUDE.md`` 20 rests on.

        Forgetting to pass the alias store at all is the plausible bug here - the registry
        would load, every test would pass, and an approved equivalence would silently stop
        participating in Class A coupling. So this asserts the approved one *does* resolve.

        The proposed one must not, and note where that is enforced: ``AssumptionKeyRegistry``
        indexes by ``participates_in_class_a`` itself, so calling ``approved()`` here is
        defence in depth rather than the only gate. Asserting the behaviour rather than the
        call is the difference between testing the rule and testing this file's wording.
        """
        settings = app_settings()
        aliases = mem.InMemoryAliasStore()
        aliases.put_alias(_alias("a", "legacy.retention_days", approved=True))
        aliases.put_alias(_alias("b", "legacy.retention_period", approved=False))

        app = Application.build(
            sources=mem.InMemorySourceStore(),
            assumptions=mem.InMemoryAssumptionStore(),
            gaps=mem.InMemoryRegistryGapStore(),
            aliases=aliases,
            coupling_index=mem.InMemoryCouplingIndexStore(),
            dispositions=mem.InMemoryDispositionStore(),
            analyses=mem.InMemoryAnalysisStore(),
            settings=settings,
            providers=fake_providers(settings),
        )
        try:
            approved = app.registry.resolve("legacy.retention_days")
            assert approved.resolved
            assert approved.via_alias is not None
            assert approved.entry is not None
            assert approved.entry.key == "retention.window"

            proposed = app.registry.resolve("legacy.retention_period")
            assert not proposed.resolved
            assert proposed.via_alias is None
        finally:
            asyncio.run(app.aclose())

    def test_an_unset_database_url_fails_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming the variable, rather than on the first query."""
        from app.infrastructure.database.session import DatabaseNotConfigured

        monkeypatch.delenv("CCE_DATABASE_URL", raising=False)
        get_settings.cache_clear()
        try:
            with pytest.raises(DatabaseNotConfigured, match="CCE_DATABASE_URL"):
                Application.configured(app_settings(database_url=None))
        finally:
            get_settings.cache_clear()

    def test_a_missing_agent_definition_fails_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An agent renamed in ``.claude/agents`` must not fail four stages into a run."""
        monkeypatch.setattr(
            "app.composition._read_definitions", lambda: {"assumption-extractor": object()}
        )

        with pytest.raises(CompositionError, match="consequence-hunter"):
            Application.build(
                sources=mem.InMemorySourceStore(),
                assumptions=mem.InMemoryAssumptionStore(),
                gaps=mem.InMemoryRegistryGapStore(),
                aliases=mem.InMemoryAliasStore(),
                coupling_index=mem.InMemoryCouplingIndexStore(),
                dispositions=mem.InMemoryDispositionStore(),
                analyses=mem.InMemoryAnalysisStore(),
                settings=app_settings(),
            )


class TestTheInMemoryCompositionStaysAvailable:
    def test_the_same_graph_builds_over_the_in_memory_stores(self) -> None:
        """The requirement that keeps the unit suites working, and the reason the fields are
        typed as ports rather than as the SQL classes."""
        settings = app_settings()
        app = Application.build(
            sources=mem.InMemorySourceStore(),
            assumptions=mem.InMemoryAssumptionStore(),
            gaps=mem.InMemoryRegistryGapStore(),
            aliases=mem.InMemoryAliasStore(),
            coupling_index=mem.InMemoryCouplingIndexStore(),
            dispositions=mem.InMemoryDispositionStore(),
            analyses=mem.InMemoryAnalysisStore(),
            settings=settings,
            providers=fake_providers(settings),
        )
        try:
            assert not app.is_persistent
            assert app.database is None
            assert app.analyze is not None
            assert readiness(app)["database"] == "not configured"
        finally:
            asyncio.run(app.aclose())


class TestTheAnalysisPathRuns:
    def test_the_configured_application_ingests_resolves_analyses_and_reviews(
        self, application: Application
    ) -> None:
        """The whole point, end to end, over the stores the composition root opened.

        Ingestion through ``application.ingest`` would need the live extractor, so the corpus
        is loaded through the existing replay loader pointed at *these* stores - the same
        substitution the rest of the suite makes, and the only one: parsers, chunking,
        identity, span verification, the closed registry and the admission gate all run for
        real against PostgreSQL-backed storage.
        """
        corpus = asyncio.run(
            load_corpus(
                into=CorpusStores(
                    tenant_id="",
                    registry_version="",
                    sources=application.sources,
                    assumptions=application.assumptions,
                    gaps=application.gaps,
                    aliases=application.aliases,
                )
            )
        )
        assert not corpus.extraction_failures, corpus.extraction_failures
        assert corpus.assumptions.all_records(corpus.tenant_id)

        # The coupling projection, written through the application's own store.
        assert application.database is not None
        with application.database.unit_of_work():
            standing = application.coupling.execute(
                tenant_id=corpus.tenant_id,
                as_of=AS_OF,
                registry_version=corpus.registry_version,
            )
        assert standing.findings
        assert application.coupling_index.findings(corpus.tenant_id)

        # And the twelve-stage analysis, through the wired use case.
        analysis_id = new_id("analysis")
        result = asyncio.run(
            application.analyze.execute(
                change=wiring.retention_change(),
                query=wiring.query_for(corpus, "personal data retention window"),
                tenant_id=corpus.tenant_id,
                registry_version=corpus.registry_version,
                as_of=AS_OF,
                analysis_id=analysis_id,
                run_id=new_id("analysis"),
                now=NOW,
            )
        )

        assert [s.stage for s in result.stages] == list(AnalysisStage)
        assert result.failed_stages == ()
        assert result.pipeline_version == ANALYSIS_PIPELINE_VERSION
        assert result.class_a
        assert result.consequences
        assert result.material_consequences
        assert result.report is not None
        assert result.report.high_priority

        # The orchestrator persisted its own snapshot through the SQL analysis store, so the
        # review path can find it - which is the wiring this composition exists to complete.
        assert application.analyses.for_tenant(corpus.tenant_id) == (analysis_id,)
        stored = application.analyses.require(corpus.tenant_id, analysis_id)
        assert stored.fingerprint == result.fingerprint()

        target = next(f for f in stored.findings if f.artifact_version_id is not None)
        outcome = application.reviews.record_reviewed(
            tenant_id=corpus.tenant_id,
            analysis_id=analysis_id,
            consequence_id=target.consequence_id,
            disposition=Disposition.WRONG,
            disposed_by="privacy-office/reviewer-1",
            disposed_at=NOW,
            note="The declaring deck was reissued before this change.",
            suppress_until_source_revision=True,
        )
        assert outcome.created
        assert application.reviews.is_suppressed(
            tenant_id=corpus.tenant_id,
            assumption_key=target.consequence.assumption_keys[0],
            artifact_version_id=target.artifact_version_id,
        )

    def test_readiness_reports_the_database_as_reachable_and_migrated(
        self, application: Application
    ) -> None:
        report = readiness(application)

        # The dialect actually in use, not the intended one: `postgresql` when
        # CCE_TEST_DATABASE_URL names a server, `sqlite` on the default substrate.
        assert report["persistence"] == application.persistence
        assert report["persistence"] in {"postgresql", "sqlite"}
        assert report["database"] == "reachable"
        assert report["migrated"] is True
        assert "stored_analyses" in report["tables"]  # type: ignore[operator]
        assert report["registered_keys"]


class TestTheEntryPoint:
    def test_the_entry_point_reports_readiness_and_exits_zero(
        self, url: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pointed_at(url):
            code = main([])

        assert code == 0
        out = capsys.readouterr().out
        assert "database: reachable" in out
        assert "migrated: True" in out

    def test_the_entry_point_exits_non_zero_with_no_database_configured(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A configuration mistake has to be loud here, not on the first analysis."""
        monkeypatch.delenv("CCE_DATABASE_URL", raising=False)
        get_settings.cache_clear()
        try:
            code = main([])
        finally:
            get_settings.cache_clear()

        assert code == 1
        assert "CCE_DATABASE_URL" in capsys.readouterr().err

    def test_the_json_form_is_machine_readable(
        self, url: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        with pointed_at(url):
            assert main(["--json"]) == 0

        report = json.loads(capsys.readouterr().out)
        assert report["database"] == "reachable"
        assert sorted(report["agents"]) == report["agents"]


def test_the_composition_root_adds_no_framework() -> None:
    """No web framework, no DI container, no worker - asserted rather than promised.

    The module is the place those would be added by reflex, so the constraint is checked
    where it would be broken.
    """
    source = Path("app/composition.py").read_text(encoding="utf-8")
    for forbidden in ("fastapi", "celery", "dependency_injector", "punq", "kubernetes", "redis"):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source
