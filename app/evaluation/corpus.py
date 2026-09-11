"""Load the sample corpus into stores, with extraction replayed rather than called.

One loader, used by the demonstration scripts and by the coupling and evaluation tests.
Duplicating it would let the script and the tests drift, and a green test suite against a
corpus the script no longer produces would be worse than having no script.

The substitution is narrow and deliberate: the **only** thing replaced is the model call.
Parsers, chunking, identity, span verification, typed parse, the closed registry and the
admission gate all run for real, which is what lets the deterministic layer be exercised
with no credentials and no network. If a recorded proposal stops matching the corpus text,
span verification rejects it and the load reports an extraction failure rather than quietly
admitting a stale binding.

Gold labels are **not** loaded here. They live in ``data/benchmarks/`` and are read only by
tests and by the benchmark harness beside this module
(``app/evaluation/retrieval_benchmark.py``); a label reachable from a production-adjacent
loader is a label the system can accidentally read. That the harness may read one is not a
weakening: ``tests/contract/test_layering.py`` restricts every layer on the ingestion and
retrieval path to importing only its peers, so nothing in the system under test can import
``app.evaluation`` at all.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.application.use_cases.ingest_source import (
    IngestionReport,
    IngestSourceUseCase,
    SourceRegistration,
)
from app.assumptions.admission import AdmissionGate
from app.assumptions.registry import AssumptionKeyRegistry
from app.config.settings import REPO_ROOT
from app.domain.interfaces.repositories import (
    AliasStore,
    AssumptionStore,
    RegistryGapStore,
    SourceStore,
)
from app.domain.value_objects.enums import AuthorityClass, SourceType
from app.domain.value_objects.provenance import ModelProvenance
from app.evaluation.replay import ReplayExtractor, load_recordings
from app.infrastructure.database.repositories.memory import (
    InMemoryAliasStore,
    InMemoryAssumptionStore,
    InMemoryRegistryGapStore,
    InMemorySourceStore,
)

CORPUS_DIR = REPO_ROOT / "data" / "sample_corpus"
RECORDINGS_PATH = REPO_ROOT / "data" / "fixtures" / "extraction_proposals.json"
REGISTRY_PATH = REPO_ROOT / "app" / "config" / "assumption_keys.yaml"

#: Provenance stamped onto replayed proposals. The provider is named ``replay`` rather than
#: a real vendor so a finding can never be mistaken for one a live model produced.
REPLAY_PROVENANCE = ModelProvenance(
    agent="assumption-extractor",
    prompt_version=1,
    prompt_sha256="0" * 64,
    provider="replay",
    model="recorded-proposals",
    tier="fast",
)


@dataclass
class CorpusStores:
    """The stores after a corpus load, plus what the load itself reported."""

    tenant_id: str
    registry_version: str
    # The **ports**, not the in-memory classes. Widened so the same loader can ingest into a
    # SQL-backed store, which is how the persistence layer is exercised through the real
    # ingestion path rather than through a fixture that writes rows directly.
    sources: SourceStore
    assumptions: AssumptionStore
    gaps: RegistryGapStore
    aliases: AliasStore
    reports: dict[str, IngestionReport] = field(default_factory=dict)

    @property
    def extraction_failures(self) -> dict[str, tuple[str, ...]]:
        """Recorded proposals that no longer match the corpus text.

        Non-empty means the fixtures and the documents have drifted apart. Callers should
        treat it as a hard failure: every downstream Class A result rests on spans that
        verified.
        """
        return {
            path: report.extraction_failures
            for path, report in self.reports.items()
            if report.extraction_failures
        }

    @property
    def admitted_count(self) -> int:
        return len(self.assumptions.all_records(self.tenant_id))


def load_manifest() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        (CORPUS_DIR / "manifest.yaml").read_text(encoding="utf-8")
    )
    return loaded


def registration_for(tenant_id: str, entry: dict[str, Any]) -> SourceRegistration:
    """Build a registration from one manifest entry.

    ``authority_class`` comes from here - operator-supplied registration metadata - and
    never from the document. That is the property that makes the injected corpus file
    powerless: it demands ``regulatory`` in its own text and is registered as ``wiki``.

    ``permissions`` is read the same way and defaults to unlabelled, which is how every
    manifest entry currently stands. A test that needs to show a document being filtered out
    can label it in a manifest copy; a document cannot label itself.
    """
    relative = Path(entry["path"])
    effective_from = entry.get("effective_from")
    return SourceRegistration(
        tenant_id=tenant_id,
        uri=f"corpus://{relative.as_posix()}",
        title=entry["title"],
        source_type=SourceType(entry["source_type"]),
        authority_class=AuthorityClass(entry["authority_class"]),
        suffix=relative.suffix,
        owner=entry.get("owner"),
        permissions=tuple(entry.get("permissions", ())),
        version_label=str(entry.get("version_label", "v1")),
        effective_from=(
            dt.datetime.combine(effective_from, dt.time(tzinfo=dt.UTC))
            if isinstance(effective_from, dt.date)
            else None
        ),
    )


async def load_corpus(
    *,
    registry: AssumptionKeyRegistry | None = None,
    manifest: dict[str, Any] | None = None,
    passes: int = 1,
    into: CorpusStores | None = None,
) -> CorpusStores:
    """Ingest every manifest source into fresh in-memory stores.

    ``passes > 1`` re-ingests the same bytes, which must cost zero extraction calls. The
    coupling tests use a single pass; the ingestion demonstration uses two to show that
    re-ingestion is free.

    ``into`` supplies the stores instead of constructing in-memory ones, so the persistence
    tests can run this exact path against a database. Its ``tenant_id`` and
    ``registry_version`` are overwritten from the manifest and the registry, because those
    are properties of what is being ingested rather than of where it lands.
    """
    resolved_registry = (
        registry if registry is not None else AssumptionKeyRegistry.load(REGISTRY_PATH)
    )
    resolved_manifest = manifest if manifest is not None else load_manifest()
    tenant_id = str(resolved_manifest["tenant_id"])

    stores = (
        CorpusStores(
            tenant_id=tenant_id,
            registry_version=resolved_registry.version,
            sources=InMemorySourceStore(),
            assumptions=InMemoryAssumptionStore(),
            gaps=InMemoryRegistryGapStore(),
            aliases=InMemoryAliasStore(),
        )
        if into is None
        else CorpusStores(
            tenant_id=tenant_id,
            registry_version=resolved_registry.version,
            sources=into.sources,
            assumptions=into.assumptions,
            gaps=into.gaps,
            aliases=into.aliases,
        )
    )

    recordings = load_recordings(RECORDINGS_PATH)

    for _ in range(max(1, passes)):
        for entry in resolved_manifest["sources"]:
            relative = str(entry["path"])
            extractor = ReplayExtractor(recorded=recordings[relative], provenance=REPLAY_PROVENANCE)
            # The gate pins the extractor version onto every admitted record, so it is
            # constructed per extractor rather than shared.
            use_case = IngestSourceUseCase(
                sources=stores.sources,
                assumptions=stores.assumptions,
                gaps=stores.gaps,
                extractor=extractor,
                gate=AdmissionGate(registry=resolved_registry, extractor_version=extractor.version),
            )
            stores.reports[relative] = await use_case.execute(
                (CORPUS_DIR / relative).read_bytes(),
                registration_for(tenant_id, entry),
            )

    return stores
