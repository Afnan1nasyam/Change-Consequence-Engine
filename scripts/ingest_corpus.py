"""Ingest the sample corpus and print what the deterministic layer produced.

    python -m scripts.ingest_corpus
    python -m scripts.ingest_corpus --twice        # prove re-ingestion costs nothing
    python -m scripts.ingest_corpus --extractor llm

Everything except the model call is real: the parsers, the chunker, the identity recipes,
the closed registry, the admission gate and the stores. The default ``replay`` extractor
substitutes recorded proposals for the call, which is what lets this run with no credentials
and no network while still exercising every deterministic check.

``--extractor llm`` uses the real extractor through the gateway. With the fake provider
configured it will report extraction failures rather than proposals, which is the correct and
visible behaviour for output that does not satisfy the schema.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import yaml

from app.agents.runtime.agent_loader import load_agent_definitions
from app.application.use_cases.ingest_source import (
    IngestionReport,
    IngestSourceUseCase,
    SourceRegistration,
)
from app.assumptions.admission import AdmissionGate
from app.assumptions.extraction import AssumptionExtractor, extractor_version
from app.assumptions.registry import AssumptionKeyRegistry
from app.config.settings import REPO_ROOT, Settings
from app.domain.value_objects.enums import AuthorityClass, SourceType
from app.domain.value_objects.provenance import ModelProvenance
from app.evaluation.replay import ReplayExtractor, load_recordings
from app.infrastructure.database.repositories.memory import (
    InMemoryAliasStore,
    InMemoryAssumptionStore,
    InMemoryRegistryGapStore,
    InMemorySourceStore,
)

CORPUS = REPO_ROOT / "data" / "sample_corpus"
RECORDINGS = REPO_ROOT / "data" / "fixtures" / "extraction_proposals.json"
REGISTRY = REPO_ROOT / "app" / "config" / "assumption_keys.yaml"


def _registration(tenant_id: str, entry: dict[str, Any]) -> SourceRegistration:
    path = Path(entry["path"])
    effective_from = entry.get("effective_from")
    return SourceRegistration(
        tenant_id=tenant_id,
        uri=f"corpus://{path.as_posix()}",
        title=entry["title"],
        source_type=SourceType(entry["source_type"]),
        authority_class=AuthorityClass(entry["authority_class"]),
        suffix=path.suffix,
        owner=entry.get("owner"),
        version_label=str(entry.get("version_label", "v1")),
        effective_from=(
            dt.datetime.combine(effective_from, dt.time(tzinfo=dt.UTC))
            if isinstance(effective_from, dt.date)
            else None
        ),
    )


async def run(*, extractor_kind: str, passes: int) -> int:
    manifest = yaml.safe_load((CORPUS / "manifest.yaml").read_text(encoding="utf-8"))
    tenant_id = manifest["tenant_id"]
    registry = AssumptionKeyRegistry.load(REGISTRY)
    definitions = load_agent_definitions()
    definition = definitions["assumption-extractor"]

    sources = InMemorySourceStore()
    assumptions = InMemoryAssumptionStore()
    gaps = InMemoryRegistryGapStore()
    aliases = InMemoryAliasStore()

    recordings = load_recordings(RECORDINGS)
    settings = Settings()

    if extractor_kind == "llm":
        from app.config.agent_bindings import AgentBindings
        from app.infrastructure.llm.gateway import LlmGateway
        from app.infrastructure.llm.registry import ProviderRegistry
        from app.infrastructure.observability.usage import InMemoryUsageLedger

        gateway = LlmGateway(
            settings=settings,
            registry=ProviderRegistry(settings),
            bindings=AgentBindings.load(settings.agents_config_path),
            ledger=InMemoryUsageLedger(),
        )

    print("=" * 100)
    print(
        f"registry {registry.version}: {len(registry)} keys across {len(registry.domains)} domains"
    )
    print(f"extractor: {extractor_kind}")
    print("=" * 100)

    reports: list[tuple[str, IngestionReport]] = []
    missing_recordings: list[str] = []

    for pass_index in range(passes):
        first_pass = pass_index == 0
        if not first_pass:
            print("\n" + "-" * 100)
            print("second pass over identical bytes")
            print("-" * 100)

        for entry in manifest["sources"]:
            relative = entry["path"]
            payload = (CORPUS / relative).read_bytes()
            registration = _registration(tenant_id, entry)

            if extractor_kind == "llm":
                extractor: Any = AssumptionExtractor(
                    completions=gateway,
                    definition=definition,
                    registry=registry,
                )
            else:
                recorded = recordings.get(relative)
                if recorded is None:
                    missing_recordings.append(relative)
                    continue
                extractor = ReplayExtractor(
                    recorded=recorded,
                    provenance=ModelProvenance(
                        agent="assumption-extractor",
                        prompt_version=definition.version,
                        prompt_sha256=definition.prompt_sha256,
                        provider="replay",
                        model="recorded",
                        tier="fast",
                    ),
                    version_label=extractor_version(definition),
                )

            use_case = IngestSourceUseCase(
                sources=sources,
                assumptions=assumptions,
                gaps=gaps,
                gate=AdmissionGate(registry=registry, extractor_version=extractor.version),
                extractor=extractor,
            )
            report = await use_case.execute(payload, registration)
            if first_pass:
                reports.append((relative, report))
            _print_report(relative, report)

    _print_summary(
        sources=sources,
        assumptions=assumptions,
        gaps=gaps,
        aliases=aliases,
        tenant_id=tenant_id,
        reports=reports,
    )

    if missing_recordings:
        print(f"\nno recorded proposals for: {missing_recordings}")
        return 1
    return 0


def _print_report(relative: str, report: IngestionReport) -> None:
    flag = "extracted" if report.extracted else "cached   "
    print(
        f"\n[{flag}] {relative}"
        f"\n    version   {report.version_id}  ({report.parser_name}, {report.chunk_count} chunks)"
        f"\n    admitted  {report.admitted_count}"
    )
    if report.gaps:
        print(f"    gaps      {len(report.gaps)}")
    if report.quarantined:
        for item in report.quarantined:
            print(f"    QUARANTINE {item}")
    if report.observed_instructions:
        print(f"    injection attempts recorded: {len(report.observed_instructions)}")
    if report.secrets_redacted:
        for item in report.secrets_redacted:
            print(f"    redacted  {item}")
    if report.delimiter_breakout_chunks:
        print(f"    delimiter breakout attempts: {len(report.delimiter_breakout_chunks)}")
    if report.extraction_failures:
        for item in report.extraction_failures:
            print(f"    EXTRACTION FAILURE {item}")


def _print_summary(
    *,
    sources: InMemorySourceStore,
    assumptions: InMemoryAssumptionStore,
    gaps: InMemoryRegistryGapStore,
    aliases: InMemoryAliasStore,
    tenant_id: str,
    reports: list[tuple[str, IngestionReport]],
) -> None:
    records = assumptions.all_records(tenant_id)
    print("\n" + "=" * 100)
    print("ASSUMPTION STORE")
    print("=" * 100)
    for record in sorted(records, key=lambda r: (r.key, r.scope.render(), r.role.value)):
        evidence = sources.get_evidence(record.evidence_id)
        span = evidence.text_span if evidence else "<missing>"
        print(
            f"{record.key:<32} {record.scope.render():<12} {record.role.value:<10} "
            f"{record.value.render():<12} {record.holder_source_version_id}"
        )
        print(f"    span verified against source: {span[:88]!r}")

    print("\n" + "=" * 100)
    print("REGISTRY GAPS (quarantined, never indexed)")
    print("=" * 100)
    for gap in gaps.open_gaps():
        print(
            f"{gap.proposed_key:<32} {gap.proposed_value.render():<12} "
            f"x{gap.occurrence_count}  from {gap.holder_source_version_id}"
        )
    if not gaps.open_gaps():
        print("(none)")

    print("\n" + "=" * 100)
    print("ALIASES")
    print("=" * 100)
    print(f"approved: {len(aliases.approved())}  total: {len(aliases.all_aliases())}")

    total_admitted = sum(r.admitted_count for _, r in reports)
    total_calls = sum(r.extraction_calls for _, r in reports)
    print("\n" + "=" * 100)
    print(
        f"{len(reports)} sources, {total_admitted} admitted bindings, "
        f"{len(gaps.open_gaps())} gaps, {total_calls} live extraction calls"
    )
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", choices=("replay", "llm"), default="replay")
    parser.add_argument(
        "--twice",
        action="store_true",
        help="ingest every source twice to show the second pass skips extraction",
    )
    args = parser.parse_args()
    return asyncio.run(run(extractor_kind=args.extractor, passes=2 if args.twice else 1))


if __name__ == "__main__":
    sys.exit(main())
