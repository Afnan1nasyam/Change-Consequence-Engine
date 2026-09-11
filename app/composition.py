"""The composition root: one place that turns configuration into a working application.

Every piece of this system was built to be constructed by its caller - ports injected, no
module-level singletons, no service locator. That was the right design and it left one thing
undone: nothing actually *did* the constructing outside a test. The PostgreSQL stores existed
and no entry point ever opened them.

This module is that entry point, and it is deliberately a function returning a frozen
dataclass rather than a container. There is no registration, no lifetime scope, no resolution
by type. Wiring an object graph is a hundred lines of constructor calls, and the value of
writing them out is that the graph is *readable* - a reader can see what depends on what
without learning a framework's rules first.

Two ways in
-----------

:meth:`Application.configured` is the production path. It reads settings, opens the database
from ``CCE_DATABASE_URL``, builds the seven SQL stores through the existing
``SqlStores.configured()`` seam, and wires the ingestion and analysis use cases over them.

:meth:`Application.build` takes the seven stores as parameters, so the in-memory composition
stays first-class: a test hands it ``InMemorySourceStore()`` and friends and gets the same
object graph with no database. That is why the store fields below are typed as the **ports**
rather than as the SQL classes - the graph does not know or care which it got, which is the
property the ports existed for.

What this module does not do
----------------------------

It adds no framework, no worker, no route and no new abstraction. It does not decide *when* an
analysis runs, who asked for it, or what happens to the result - those are a caller's
decisions, and the caller does not exist yet. ``app/api/deps.py`` keeps its own per-process
gateway cache for the health routes; the overlap is six lines of constructor calls, and
merging them would trade that duplication for a composition root that imports the web layer.
The two have different lifetimes on purpose: those caches are process-wide singletons serving
requests, this is one owned graph a caller closes.

Running it
----------

``python -m app.composition`` constructs everything and reports what it got, including whether
the database is reachable and migrated. It is a readiness check, not a server: the point is
that a configuration mistake fails here, with a message naming the setting, rather than on the
first analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Final

from app.agents.runtime.agent_loader import AgentDefinition, load_agent_definitions
from app.application.orchestration.analysis import AnalyzeChangeUseCase
from app.application.orchestration.analysis_store import AnalysisStore
from app.application.orchestration.retrieval_service import RetrievalService
from app.application.use_cases.arbitrate_evidence import ArbitrateEvidenceUseCase
from app.application.use_cases.challenge_findings import ChallengeFindingsUseCase
from app.application.use_cases.compose_report import ComposeReportUseCase
from app.application.use_cases.consult_specialists import ConsultSpecialistsUseCase
from app.application.use_cases.hunt_blind_spots import HuntBlindSpotsUseCase
from app.application.use_cases.hunt_consequences import HuntConsequencesUseCase
from app.application.use_cases.hunt_contradictions import HuntContradictionsUseCase
from app.application.use_cases.ingest_source import IngestSourceUseCase
from app.application.use_cases.judge_consequences import JudgeConsequencesUseCase
from app.application.use_cases.record_disposition import RecordDispositionUseCase
from app.application.use_cases.resolve_coupling import ResolveCouplingUseCase
from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.assumptions.admission import AdmissionGate
from app.assumptions.extraction import AssumptionExtractor
from app.assumptions.registry import AssumptionKeyRegistry
from app.blindspots.hunter import BlindSpotHunter
from app.challenge.challenger import AdversarialChallenger
from app.config.agent_bindings import AgentBindings
from app.config.settings import Settings, get_settings
from app.consequences.hunter import ConsequenceHunter
from app.contradictions.hunter import ContradictionHunter
from app.domain.interfaces.llm import EmbeddingPort, ProviderUnavailable
from app.domain.interfaces.repositories import (
    AliasStore,
    AssumptionStore,
    CouplingIndexStore,
    RegistryGapStore,
    SourceStore,
)
from app.infrastructure.database.repositories.sql import SqlStores
from app.infrastructure.database.session import Database
from app.infrastructure.embeddings.registry import EmbeddingRegistry
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.registry import ProviderRegistry
from app.infrastructure.observability.logging import get_logger
from app.infrastructure.observability.usage import InMemoryUsageLedger
from app.judgment.arbitration import EvidenceArbiter
from app.judgment.dispositions import DispositionStore
from app.judgment.judge import ConsequenceJudge
from app.reporting.composer import ReportComposer
from app.specialists.analyst import SpecialistAnalyst
from app.specialists.roster import SPECIALIST_AGENTS

logger = get_logger(__name__)

#: The agents the analysis path invokes, in pipeline order. Named so a missing definition
#: fails at construction with the agent's name rather than mid-analysis.
REQUIRED_AGENTS: Final = (
    "assumption-extractor",
    "consequence-hunter",
    "contradiction-hunter",
    "blind-spot-hunter",
    "adversarial-challenger",
    "evidence-arbiter",
    "consequence-judge",
    "report-composer",
)

#: The domain analysts. **Not** in ``REQUIRED_AGENTS``: the fan-out reports a specialist with
#: no definition as un-selected with its own reason, so a deployment may legitimately run a
#: subset and a renamed ``.md`` costs one domain rather than the analysis.
OPTIONAL_AGENTS: Final = SPECIALIST_AGENTS


class CompositionError(RuntimeError):
    """The application could not be constructed. Always names what was missing."""


@dataclass(frozen=True)
class Application:
    """One wired application. Frozen, because rewiring a live graph is never the answer.

    The store fields are the **ports**, not the SQL classes: this graph works the same over
    the in-memory implementations, which is what keeps the test composition first-class
    instead of a parallel wiring that can drift.

    ``database`` is ``None`` for the in-memory composition. Optional rather than absent, so a
    caller that needs a unit of work or a connection can ask for one and get an honest answer.
    """

    settings: Settings

    database: Database | None
    sources: SourceStore
    assumptions: AssumptionStore
    gaps: RegistryGapStore
    aliases: AliasStore
    coupling_index: CouplingIndexStore
    dispositions: DispositionStore
    analyses: AnalysisStore

    registry: AssumptionKeyRegistry
    definitions: dict[str, AgentDefinition]
    providers: ProviderRegistry
    embeddings: EmbeddingRegistry
    gateway: LlmGateway

    ingest: IngestSourceUseCase
    coupling: ResolveCouplingUseCase
    retrieval: RetrievalService
    analyze: AnalyzeChangeUseCase
    reviews: RecordDispositionUseCase

    # -- construction -----------------------------------------------------

    @classmethod
    def configured(
        cls,
        settings: Settings | None = None,
        *,
        providers: ProviderRegistry | None = None,
        embedder: EmbeddingPort | None = None,
    ) -> Application:
        """The production path: PostgreSQL stores from ``CCE_DATABASE_URL``.

        Raises ``DatabaseNotConfigured`` when the URL is unset - at construction, naming the
        variable, rather than on the first query.

        ``providers`` and ``embedder`` forward to :meth:`build`. They are here so *this* path
        - the one production uses - is the one the end-to-end test exercises, with a fake
        provider standing in for a credential. A test that had to call ``build`` instead would
        be testing a different function from the one that ships.
        """
        resolved = settings or get_settings()
        database = Database.configured()
        stores = SqlStores.over(database)
        return cls.build(
            settings=resolved,
            database=database,
            sources=stores.sources,
            assumptions=stores.assumptions,
            gaps=stores.gaps,
            aliases=stores.aliases,
            coupling_index=stores.coupling,
            dispositions=stores.dispositions,
            analyses=stores.analyses,
            providers=providers,
            embedder=embedder,
        )

    @classmethod
    def build(
        cls,
        *,
        sources: SourceStore,
        assumptions: AssumptionStore,
        gaps: RegistryGapStore,
        aliases: AliasStore,
        coupling_index: CouplingIndexStore,
        dispositions: DispositionStore,
        analyses: AnalysisStore,
        settings: Settings | None = None,
        database: Database | None = None,
        gateway: LlmGateway | None = None,
        providers: ProviderRegistry | None = None,
        embedder: EmbeddingPort | None = None,
    ) -> Application:
        """Wire the graph over whichever stores were handed in.

        Every override exists for one reason and no other: ``gateway`` and ``providers`` so a
        test can inject a fake provider without credentials, ``embedder`` so a recorded
        fixture can be pinned. There is no override that changes analysis semantics.
        """
        resolved = settings or get_settings()
        registry = _load_registry(resolved, aliases)
        definitions = _load_definitions()

        provider_registry = providers or ProviderRegistry(resolved)
        embeddings = EmbeddingRegistry(
            resolved, overrides={resolved.embedding_provider: embedder} if embedder else None
        )
        llm = gateway or LlmGateway(
            settings=resolved,
            registry=provider_registry,
            bindings=AgentBindings.load(resolved.agents_config_path),
            ledger=InMemoryUsageLedger(),
        )

        extractor = AssumptionExtractor(
            completions=llm,
            definition=definitions["assumption-extractor"],
            registry=registry,
        )
        ingest = IngestSourceUseCase(
            sources=sources,
            assumptions=assumptions,
            gaps=gaps,
            extractor=extractor,
            # The gate pins the extractor version onto every admitted record, so it is built
            # from the extractor rather than from configuration - a prompt change then
            # re-extracts, which is the behaviour `extracted_with` exists for.
            gate=AdmissionGate(registry=registry, extractor_version=extractor.version),
        )

        coupling = ResolveCouplingUseCase(
            sources=sources,
            assumptions=assumptions,
            gaps=gaps,
            index_store=coupling_index,
        )
        retrieval = RetrievalService(
            retrieval=RetrieveEvidenceUseCase(sources=sources, assumptions=assumptions),
            default_strategy=resolved.retrieval_strategy,
            embedder=_resolve_embedder(embeddings, resolved),
        )
        reviews = RecordDispositionUseCase(store=dispositions, analyses=analyses)

        analyze = AnalyzeChangeUseCase(
            coupling=coupling,
            retrieval=retrieval,
            specialists=ConsultSpecialistsUseCase(
                analysts={
                    agent: SpecialistAnalyst(completions=llm, definition=definitions[agent])
                    for agent in OPTIONAL_AGENTS
                    if agent in definitions
                },
                retrieval=retrieval,
            ),
            class_b=HuntConsequencesUseCase(
                hunter=ConsequenceHunter(
                    completions=llm, definition=definitions["consequence-hunter"]
                )
            ),
            contradictions=HuntContradictionsUseCase(
                hunter=ContradictionHunter(
                    completions=llm, definition=definitions["contradiction-hunter"]
                )
            ),
            blind_spots=HuntBlindSpotsUseCase(
                hunter=BlindSpotHunter(
                    completions=llm, definition=definitions["blind-spot-hunter"]
                ),
                retrieval=retrieval,
            ),
            challenge=ChallengeFindingsUseCase(
                challenger=AdversarialChallenger(
                    completions=llm, definition=definitions["adversarial-challenger"]
                )
            ),
            arbitration=ArbitrateEvidenceUseCase(
                arbiter=EvidenceArbiter(completions=llm, definition=definitions["evidence-arbiter"])
            ),
            judgment=JudgeConsequencesUseCase(
                judge=ConsequenceJudge(completions=llm, definition=definitions["consequence-judge"])
            ),
            report=ComposeReportUseCase(
                composer=ReportComposer(completions=llm, definition=definitions["report-composer"])
            ),
            analyses=analyses,
            reviews=reviews,
        )

        logger.info(
            "application_composed",
            extra={
                "environment": resolved.environment,
                "persistence": database.engine.dialect.name if database else "in-memory",
                "retrieval_strategy": retrieval.default_strategy.value,
                "registry_version": registry.version,
                "agents": len(definitions),
                "specialists": [a for a in OPTIONAL_AGENTS if a in definitions],
            },
        )
        return cls(
            settings=resolved,
            database=database,
            sources=sources,
            assumptions=assumptions,
            gaps=gaps,
            aliases=aliases,
            coupling_index=coupling_index,
            dispositions=dispositions,
            analyses=analyses,
            registry=registry,
            definitions=definitions,
            providers=provider_registry,
            embeddings=embeddings,
            gateway=llm,
            ingest=ingest,
            coupling=coupling,
            retrieval=retrieval,
            analyze=analyze,
            reviews=reviews,
        )

    # -- lifecycle --------------------------------------------------------

    @property
    def is_persistent(self) -> bool:
        """Whether this graph is backed by a database rather than by process memory."""
        return self.database is not None

    @property
    def persistence(self) -> str:
        """The dialect actually in use, read off the engine.

        Not the string ``"postgres"`` whenever a URL happens to be set: the production target
        is Postgres and the test substrate is SQLite, and an operator reading a readiness
        report needs to be told which one they got rather than which one was intended.
        """
        if self.database is None:
            return "in-memory"
        return self.database.engine.dialect.name

    async def aclose(self) -> None:
        """Release the sockets and the connection pool. Idempotent.

        Providers and embedders hold HTTP clients; the database holds a pool. Leaking either
        is the ordinary cost of a reload, so closing is a method rather than a comment.
        """
        await self.providers.aclose()
        await self.embeddings.aclose()
        if self.database is not None:
            self.database.dispose()


# ---------------------------------------------------------------------------
# Construction-time checks
# ---------------------------------------------------------------------------


def _load_registry(settings: Settings, aliases: AliasStore) -> AssumptionKeyRegistry:
    """The closed key registry, with the **approved** aliases only.

    ``approved()`` rather than ``all_aliases()``: only an explicitly approved equivalence may
    participate in Class A coupling, and there is no threshold that bypasses that gate
    (``CLAUDE.md`` 20). Loading proposals here would be the one way to smuggle an invented
    dependency into the deterministic layer.
    """
    try:
        return AssumptionKeyRegistry.load(settings.assumption_keys_path, aliases=aliases.approved())
    except Exception as exc:
        raise CompositionError(
            f"could not load the assumption key registry from {settings.assumption_keys_path}: "
            f"{exc}"
        ) from exc


def _load_definitions() -> dict[str, AgentDefinition]:
    """Every agent definition, with the ones the analysis path needs asserted present.

    Checked here rather than discovered at dispatch: an agent whose ``.md`` was renamed should
    fail at construction naming the agent, not four stages into an analysis.
    """
    definitions = _read_definitions()
    missing = [name for name in REQUIRED_AGENTS if name not in definitions]
    if missing:
        raise CompositionError(
            f"agent definitions missing for {missing}. The analysis path dispatches by name, "
            "so a renamed or deleted .md would otherwise fail partway through a run."
        )
    return definitions


def _read_definitions() -> dict[str, AgentDefinition]:
    try:
        return load_agent_definitions()
    except Exception as exc:
        raise CompositionError(f"could not load agent definitions: {exc}") from exc


def _resolve_embedder(embeddings: EmbeddingRegistry, settings: Settings) -> EmbeddingPort | None:
    """The configured embedder, or ``None`` when it is not enabled.

    ``None`` rather than a substitute. ``RetrievalService`` already refuses a vector or hybrid
    request it cannot serve, with a message saying so - falling back to lexical would give a
    caller who asked for semantic retrieval no way to know they did not get it.
    """
    try:
        return embeddings.default()
    except ProviderUnavailable as exc:
        logger.warning(
            "embedder_unavailable",
            extra={"provider": settings.embedding_provider, "detail": str(exc)},
        )
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def readiness(app: Application) -> dict[str, object]:
    """What was constructed, and whether the database is reachable and migrated.

    A readiness *report* rather than a boolean: "not ready" is not actionable, and the three
    things that go wrong here - no URL, no server, no migration - need different fixes.
    """
    report: dict[str, object] = {
        "environment": app.settings.environment,
        "persistence": app.persistence,
        "retrieval_strategy": app.retrieval.default_strategy.value,
        "retrieval_available": [s.value for s in app.retrieval.available_strategies()],
        "registry_version": app.registry.version,
        "registered_keys": len(app.registry.keys),
        "agents": sorted(app.definitions),
        "specialists": [a for a in OPTIONAL_AGENTS if a in app.definitions],
        "llm_providers": list(app.providers.enabled),
        "embedding_providers": list(app.embeddings.enabled),
    }
    if app.database is None:
        report["database"] = "not configured"
        return report

    from sqlalchemy import inspect

    try:
        tables = sorted(inspect(app.database.engine).get_table_names())
    except Exception as exc:
        report["database"] = "unreachable"
        report["database_error"] = str(exc)
        return report

    report["database"] = "reachable"
    report["tables"] = tables
    report["migrated"] = "alembic_version" in tables
    return report


def main(argv: list[str] | None = None) -> int:
    """Construct the application and report readiness. Exits non-zero when it cannot.

    The whole point: a configuration mistake fails here, naming the setting, instead of on the
    first analysis somebody runs.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.composition",
        description="Construct the CCE application from configuration and report readiness.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the readiness report as one JSON object"
    )
    args = parser.parse_args(argv)

    try:
        app = Application.configured()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        report = readiness(app)
    finally:
        asyncio.run(app.aclose())

    if args.json:
        import json

        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0 if report.get("database") == "reachable" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(main())
