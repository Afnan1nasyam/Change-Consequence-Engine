"""The ingestion path end to end, and the sample corpus scored against its gold labels.

Everything here is real except the model call, which is replayed from
``data/fixtures/extraction_proposals.json``. That substitution is deliberate: it holds
extraction fixed so that any change in these results is a change in the deterministic
layer, which is the only thing these tests are about.

The Phase 2 acceptance criteria are asserted by name, so a regression names itself.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.application.use_cases.ingest_source import (
    IngestionReport,
    IngestSourceUseCase,
    SourceRegistration,
)
from app.assumptions.admission import AdmissionGate
from app.assumptions.extraction import ExtractionOutcome
from app.assumptions.registry import AssumptionKeyRegistry
from app.config.settings import REPO_ROOT
from app.domain.services.span_verification import verify_offsets
from app.domain.value_objects.enums import AuthorityClass, BindingRole, SourceType
from app.domain.value_objects.provenance import ModelProvenance
from app.evaluation.replay import ReplayExtractor, load_recordings
from app.infrastructure.database.repositories.memory import (
    InMemoryAliasStore,
    InMemoryAssumptionStore,
    InMemoryRegistryGapStore,
    InMemorySourceStore,
)
from app.ingestion.chunking import ChunkWithContext

CORPUS = REPO_ROOT / "data" / "sample_corpus"
RECORDINGS = REPO_ROOT / "data" / "fixtures" / "extraction_proposals.json"
REGISTRY_PATH = REPO_ROOT / "app" / "config" / "assumption_keys.yaml"
EXPECTATIONS = REPO_ROOT / "data" / "benchmarks" / "ingestion_expectations.yaml"

TENANT_A = "tn_" + "0" * 31 + "1"
TENANT_B = "tn_" + "0" * 31 + "2"

POLICY = b"""# Refund Approval Policy

## Approval limits

Refunds of 10,000 USD or more require manager approval before release.
"""

PROPOSALS: dict[str, Any] = {
    "bindings": [
        {
            "proposed_key": "refund.approval_threshold",
            "scope": "global",
            "value_quote": "10,000",
            "value_type": "scalar",
            "unit": "USD",
            "role": "declares",
            "explicitness": "explicit",
            "quote": "Refunds of 10,000 USD or more require manager approval before release.",
        }
    ],
    "observed_instructions": [],
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Harness:
    """Real stores, real gate, replayed extraction."""

    def __init__(self, registry: AssumptionKeyRegistry) -> None:
        self.sources = InMemorySourceStore()
        self.assumptions = InMemoryAssumptionStore()
        self.gaps = InMemoryRegistryGapStore()
        self.aliases = InMemoryAliasStore()
        self.registry = registry
        self.extractions = 0
        self.extractor_version = "extractor-1+test"

    def use_case(self, recorded: dict[str, Any]) -> IngestSourceUseCase:
        from app.evaluation.replay import RecordedExtraction

        harness = self

        class CountingReplay(ReplayExtractor):
            async def extract(
                self,
                chunks: Sequence[ChunkWithContext],
                *,
                correlation_id: str | None = None,
            ) -> ExtractionOutcome:
                harness.extractions += 1
                return await super().extract(chunks, correlation_id=correlation_id)

        extractor = CountingReplay(
            recorded=RecordedExtraction.model_validate(recorded),
            provenance=ModelProvenance(
                agent="assumption-extractor",
                prompt_version=1,
                prompt_sha256="a" * 64,
                provider="replay",
                model="recorded",
                tier="fast",
            ),
            version_label=self.extractor_version,
        )
        return IngestSourceUseCase(
            sources=self.sources,
            assumptions=self.assumptions,
            gaps=self.gaps,
            gate=AdmissionGate(registry=self.registry, extractor_version=extractor.version),
            extractor=extractor,
        )


@pytest.fixture(scope="module")
def registry() -> AssumptionKeyRegistry:
    return AssumptionKeyRegistry.load(REGISTRY_PATH)


@pytest.fixture
def harness(registry: AssumptionKeyRegistry) -> Harness:
    return Harness(registry)


def registration(**over: Any) -> SourceRegistration:
    payload: dict[str, Any] = {
        "tenant_id": TENANT_A,
        "uri": "corpus://policy/refund-approval-policy.md",
        "title": "Refund Approval Policy",
        "source_type": SourceType.POLICY,
        "authority_class": AuthorityClass.POLICY,
        "suffix": ".md",
        "version_label": "v3",
        "effective_from": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
    }
    payload.update(over)
    return SourceRegistration(**payload)


# ---------------------------------------------------------------------------
# Acceptance: a corpus ingests
# ---------------------------------------------------------------------------


async def test_a_document_ingests_into_the_full_chain(harness: Harness) -> None:
    report = await harness.use_case(PROPOSALS).execute(POLICY, registration())

    assert report.chunk_count > 0
    assert report.extracted
    assert report.admitted_count == 1

    document = harness.sources.get_document(TENANT_A, report.source_id)
    assert document is not None
    version = harness.sources.get_version(report.version_id)
    assert version is not None
    assert harness.sources.chunks_for(report.version_id)

    record = harness.assumptions.records_for_version(report.version_id)[0]
    evidence = harness.sources.get_evidence(record.evidence_id)
    assert evidence is not None
    assert record.is_admitted


async def test_authority_comes_from_registration_not_content(harness: Harness) -> None:
    """The document says nothing about its authority, and could not be believed if it did."""
    report = await harness.use_case(PROPOSALS).execute(
        POLICY, registration(authority_class=AuthorityClass.WIKI)
    )
    document = harness.sources.get_document(TENANT_A, report.source_id)
    assert document is not None
    assert document.authority_class is AuthorityClass.WIKI


# ---------------------------------------------------------------------------
# Acceptance: evidence IDs remain stable, spans verify
# ---------------------------------------------------------------------------


async def test_evidence_ids_are_stable_across_reingestion(harness: Harness) -> None:
    first = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    ids_before = {r.evidence_id for r in harness.assumptions.records_for_version(first.version_id)}

    fresh = Harness(harness.registry)
    second = await fresh.use_case(PROPOSALS).execute(POLICY, registration())
    ids_after = {r.evidence_id for r in fresh.assumptions.records_for_version(second.version_id)}

    assert ids_before == ids_after
    assert first.version_id == second.version_id


async def test_every_stored_span_verifies_against_the_stored_text(
    harness: Harness,
) -> None:
    """The anti-hallucination primitive, asserted over what actually got persisted."""
    report = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    text = harness.sources.get_text(report.version_id)
    assert text is not None

    for record in harness.assumptions.records_for_version(report.version_id):
        evidence = harness.sources.get_evidence(record.evidence_id)
        assert evidence is not None
        assert verify_offsets(
            text,
            char_start=evidence.char_start,
            char_end=evidence.char_end,
            claimed=evidence.text_span,
        ).ok


# ---------------------------------------------------------------------------
# Acceptance: re-ingesting an unchanged source performs zero extraction calls
# ---------------------------------------------------------------------------


async def test_reingesting_unchanged_bytes_performs_no_extraction(
    harness: Harness,
) -> None:
    first = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    assert harness.extractions == 1
    assert first.extracted

    second = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    assert harness.extractions == 1, "extraction must be skipped on identity"
    assert not second.extracted
    assert second.version_id == first.version_id
    assert second.admitted_count == first.admitted_count


async def test_reformatting_json_does_not_re_extract(
    registry: AssumptionKeyRegistry,
) -> None:
    """The JSON parser's canonical rendering is what makes this true."""
    harness = Harness(registry)
    proposals = {
        "bindings": [
            {
                "proposed_key": "api.max_page_size",
                "scope": "global",
                "value_quote": "100",
                "value_type": "scalar",
                "unit": "items",
                "role": "declares",
                "explicitness": "explicit",
                "quote": "api.max_page_size: 100",
            }
        ],
        "observed_instructions": [],
    }
    reg = registration(uri="corpus://config/idp.json", suffix=".json")

    first = await harness.use_case(proposals).execute(
        b'{"api": {"max_page_size": 100, "default_page_size": 25}}', reg
    )
    second = await harness.use_case(proposals).execute(
        b'{\n  "api": {\n    "default_page_size": 25,\n    "max_page_size": 100\n  }\n}', reg
    )

    assert first.version_id == second.version_id
    assert harness.extractions == 1


async def test_an_edited_document_creates_a_new_version_and_re_extracts(
    harness: Harness,
) -> None:
    first = await harness.use_case(PROPOSALS).execute(POLICY, registration())

    edited = POLICY.replace(b"10,000", b"5,000")
    edited_proposals = {
        "bindings": [
            {
                **PROPOSALS["bindings"][0],
                "value_quote": "5,000",
                "quote": "Refunds of 5,000 USD or more require manager approval before release.",
            }
        ],
        "observed_instructions": [],
    }
    second = await harness.use_case(edited_proposals).execute(edited, registration())

    assert second.version_id != first.version_id
    assert second.extracted
    assert harness.extractions == 2


# ---------------------------------------------------------------------------
# Acceptance: source versions are queryable
# ---------------------------------------------------------------------------


async def test_version_history_is_queryable_in_order(harness: Harness) -> None:
    first = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    edited = POLICY.replace(b"10,000", b"5,000")
    second = await harness.use_case({"bindings": [], "observed_instructions": []}).execute(
        edited, registration(version_label="v4")
    )

    versions = harness.sources.versions_for(first.source_id)
    assert [v.version_id for v in versions] == [first.version_id, second.version_id]
    assert [v.version_label for v in versions] == ["v3", "v4"]


async def test_text_stored_per_version_is_independent(harness: Harness) -> None:
    first = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    second = await harness.use_case({"bindings": [], "observed_instructions": []}).execute(
        POLICY.replace(b"10,000", b"5,000"), registration()
    )
    assert harness.sources.get_text(first.version_id) != harness.sources.get_text(second.version_id)


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------


async def test_two_tenants_ingesting_one_uri_stay_separate(harness: Harness) -> None:
    a = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    b = await harness.use_case(PROPOSALS).execute(POLICY, registration(tenant_id=TENANT_B))

    assert a.source_id != b.source_id
    assert a.version_id != b.version_id


async def test_a_document_is_not_readable_from_another_tenant(harness: Harness) -> None:
    """A leak must require deleting a check, not forgetting one."""
    report = await harness.use_case(PROPOSALS).execute(POLICY, registration())

    assert harness.sources.get_document(TENANT_A, report.source_id) is not None
    assert harness.sources.get_document(TENANT_B, report.source_id) is None
    assert harness.sources.find_document_by_uri(TENANT_B, registration().uri) is None


async def test_records_are_scoped_by_tenant(harness: Harness) -> None:
    await harness.use_case(PROPOSALS).execute(POLICY, registration())
    await harness.use_case(PROPOSALS).execute(POLICY, registration(tenant_id=TENANT_B))

    assert len(harness.assumptions.all_records(TENANT_A)) == 1
    assert len(harness.assumptions.all_records(TENANT_B)) == 1
    assert not harness.assumptions.records_for_key(TENANT_A, "retention.window")


# ---------------------------------------------------------------------------
# The trusted store refuses what did not pass the gate
# ---------------------------------------------------------------------------


async def test_the_assumption_store_refuses_an_unadmitted_record(
    harness: Harness,
) -> None:
    report = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    record = harness.assumptions.records_for_version(report.version_id)[0]
    unverified = record.model_copy(update={"span_verified": False})

    with pytest.raises(ValueError, match="admitted bindings only"):
        harness.assumptions.put_records((unverified,))


async def test_a_quarantined_proposal_never_reaches_the_assumption_store(
    harness: Harness,
) -> None:
    unknown = {
        "bindings": [{**PROPOSALS["bindings"][0], "proposed_key": "refund.escalation_ceiling"}],
        "observed_instructions": [],
    }
    report = await harness.use_case(unknown).execute(POLICY, registration())

    assert report.admitted_count == 0
    assert len(report.gaps) == 1
    assert not harness.assumptions.records_for_version(report.version_id)
    assert harness.gaps.open_gaps()


async def test_gap_volume_is_queryable_per_key(harness: Harness) -> None:
    unknown = {
        "bindings": [{**PROPOSALS["bindings"][0], "proposed_key": "refund.escalation_ceiling"}],
        "observed_instructions": [],
    }
    await harness.use_case(unknown).execute(POLICY, registration())
    assert harness.gaps.gap_counts_by_key()["refund.escalation_ceiling"] >= 1


# ---------------------------------------------------------------------------
# The sample corpus, scored against its gold labels
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def expectations() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(EXPECTATIONS.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((CORPUS / "manifest.yaml").read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def corpus(
    registry: AssumptionKeyRegistry, manifest: dict[str, Any]
) -> tuple[Harness, dict[str, IngestionReport]]:
    """Ingest the whole corpus once, then assert against the result."""
    return asyncio.run(_ingest_corpus(registry, manifest))


async def _ingest_corpus(
    registry: AssumptionKeyRegistry, manifest: dict[str, Any]
) -> tuple[Harness, dict[str, IngestionReport]]:
    harness = Harness(registry)
    recordings = load_recordings(RECORDINGS)
    reports: dict[str, IngestionReport] = {}

    for entry in manifest["sources"]:
        relative = entry["path"]
        recorded = recordings[relative]
        effective_from = entry.get("effective_from")
        reg = SourceRegistration(
            tenant_id=manifest["tenant_id"],
            uri=f"corpus://{relative}",
            title=entry["title"],
            source_type=SourceType(entry["source_type"]),
            authority_class=AuthorityClass(entry["authority_class"]),
            suffix=Path(relative).suffix,
            owner=entry.get("owner"),
            version_label=str(entry.get("version_label", "v1")),
            effective_from=(
                dt.datetime.combine(effective_from, dt.time(tzinfo=dt.UTC))
                if isinstance(effective_from, dt.date)
                else None
            ),
        )
        reports[relative] = await harness.use_case(recorded.model_dump()).execute(
            (CORPUS / relative).read_bytes(), reg
        )

    return harness, reports


def test_the_corpus_ingests_without_extraction_failures(
    corpus: tuple[Harness, dict[str, IngestionReport]],
) -> None:
    _harness, reports = corpus
    failures = {path: r.extraction_failures for path, r in reports.items() if r.extraction_failures}
    assert not failures, f"recorded proposals that no longer match the corpus: {failures}"


def test_every_expected_binding_is_admitted(
    corpus: tuple[Harness, dict[str, IngestionReport]],
    expectations: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    harness, _reports = corpus
    tenant = manifest["tenant_id"]

    actual = {
        (
            record.key,
            record.scope.render(),
            record.role.value,
            record.value.render(),
        )
        for record in harness.assumptions.all_records(tenant)
    }

    missing = [
        tuple(row[:4])
        for row in expectations["admitted"]
        if (row[0], row[1], row[2], row[3]) not in actual
    ]
    assert not missing, f"expected bindings not admitted: {missing}"


def test_every_expected_gap_is_quarantined(
    corpus: tuple[Harness, dict[str, IngestionReport]], expectations: dict[str, Any]
) -> None:
    harness, _reports = corpus
    quarantined = {gap.proposed_key for gap in harness.gaps.open_gaps()}

    for expected in expectations["expected_gaps"]:
        assert expected["key"] in quarantined, expected["key"]


def test_a_quarantined_key_is_absent_from_the_trusted_store(
    corpus: tuple[Harness, dict[str, IngestionReport]],
    expectations: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Never indexed. There is no fallback path."""
    harness, _reports = corpus
    stored_keys = {r.key for r in harness.assumptions.all_records(manifest["tenant_id"])}

    for expected in expectations["expected_gaps"]:
        assert expected["key"] not in stored_keys


def test_an_irrelevant_document_binds_nothing(
    corpus: tuple[Harness, dict[str, IngestionReport]], expectations: dict[str, Any]
) -> None:
    """The negative control. A system that finds something everywhere finds nothing."""
    _harness, reports = corpus
    for path in expectations["expected_silent"]:
        assert reports[path].admitted_count == 0, path
        assert not reports[path].gaps, path


def test_role_distinguishes_declaration_from_repetition(
    corpus: tuple[Harness, dict[str, IngestionReport]], manifest: dict[str, Any]
) -> None:
    """The axis that decides contradiction versus staleness later on."""
    harness, _reports = corpus
    threshold = harness.assumptions.records_for_key(
        manifest["tenant_id"], "refund.approval_threshold"
    )
    roles = {r.role for r in threshold}
    assert BindingRole.DECLARES in roles
    assert BindingRole.CONSUMES in roles
    assert BindingRole.DESCRIBES in roles


def test_the_corpus_seeds_a_same_scope_divergence_between_declarers(
    corpus: tuple[Harness, dict[str, IngestionReport]], manifest: dict[str, Any]
) -> None:
    """Phase 2 does not resolve it; the corpus must be able to produce it."""
    harness, _reports = corpus
    declarers = [
        r
        for r in harness.assumptions.records_for_key(
            manifest["tenant_id"], "refund.approval_threshold"
        )
        if r.role is BindingRole.DECLARES and r.scope.is_global
    ]
    values = {r.value.render() for r in declarers}
    assert len(values) > 1, f"expected divergent global declarers, got {values}"


def test_the_corpus_seeds_a_scoped_override_pair(
    corpus: tuple[Harness, dict[str, IngestionReport]], manifest: dict[str, Any]
) -> None:
    harness, _reports = corpus
    retention = harness.assumptions.records_for_key(manifest["tenant_id"], "retention.window")
    scopes = {r.scope.render() for r in retention}
    assert "global" in scopes
    assert "region:eu" in scopes


# ---------------------------------------------------------------------------
# Injection, on the real corpus
# ---------------------------------------------------------------------------


def test_the_injected_document_cannot_fabricate_authority(
    corpus: tuple[Harness, dict[str, IngestionReport]],
    expectations: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    harness, reports = corpus
    spec = expectations["injection_expectations"]
    report = reports[spec["path"]]

    document = harness.sources.get_document(manifest["tenant_id"], report.source_id)
    assert document is not None
    assert document.authority_class.value == spec["authority_class_must_be"]


def test_the_injected_document_cannot_create_a_registry_key(
    corpus: tuple[Harness, dict[str, IngestionReport]], registry: AssumptionKeyRegistry
) -> None:
    """The load-bearing outcome: the demand is inert because the registry is closed."""
    harness, _reports = corpus
    assert not registry.contains("vendor.master_override")
    assert "vendor.master_override" in {g.proposed_key for g in harness.gaps.open_gaps()}


def test_the_injected_document_cannot_approve_an_alias(
    corpus: tuple[Harness, dict[str, IngestionReport]],
) -> None:
    harness, _reports = corpus
    assert harness.aliases.approved() == ()
    assert harness.aliases.all_aliases() == ()


def test_the_injected_documents_own_bindings_are_recorded_but_powerless(
    corpus: tuple[Harness, dict[str, IngestionReport]],
    expectations: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """The documented boundary, asserted so it is not mistaken for a defect.

    A span-verified binding on a *registered* key is admitted even from a hostile document -
    the text really does say it. What the document cannot do is make it authoritative: it
    holds `wiki` authority, so at coupling time it can only ever be an unauthorized
    override of the real policy, never a replacement for it.
    """
    harness, reports = corpus
    spec = expectations["injection_expectations"]
    version_id = reports[spec["path"]].version_id

    records = harness.assumptions.records_for_version(version_id)
    admitted = {(r.key, r.scope.render(), r.role.value, r.value.render()) for r in records}
    for row in spec["admitted_despite_injection"]:
        assert (row[0], row[1], row[2], row[3]) in admitted

    document = harness.sources.get_document(manifest["tenant_id"], reports[spec["path"]].source_id)
    assert document is not None
    assert document.authority_class is AuthorityClass.WIKI


def test_injected_instructions_are_recorded_as_content(
    corpus: tuple[Harness, dict[str, IngestionReport]], expectations: dict[str, Any]
) -> None:
    _harness, reports = corpus
    spec = expectations["injection_expectations"]
    report = reports[spec["path"]]
    assert len(report.observed_instructions) >= spec["observed_instruction_count_at_least"]


def test_no_admitted_record_anywhere_carries_a_confidence_number(
    corpus: tuple[Harness, dict[str, IngestionReport]], manifest: dict[str, Any]
) -> None:
    harness, _reports = corpus
    for record in harness.assumptions.all_records(manifest["tenant_id"]):
        dumped = record.model_dump()
        assert "confidence" not in dumped
        assert "extraction_confidence" not in dumped


# ---------------------------------------------------------------------------
# The cache marker
# ---------------------------------------------------------------------------


async def test_a_document_that_binds_nothing_is_still_only_extracted_once(
    harness: Harness,
) -> None:
    """The negative control must be cached like anything else.

    Keying the cache on "does this version have records" re-extracted every silent document
    on every ingestion. On a real corpus most documents bind nothing, so that quietly
    re-extracts most of the corpus and destroys the amortization this stage depends on.
    """
    empty: dict[str, Any] = {"bindings": [], "observed_instructions": []}

    first = await harness.use_case(empty).execute(POLICY, registration())
    assert first.extracted
    assert first.admitted_count == 0
    assert harness.extractions == 1

    second = await harness.use_case(empty).execute(POLICY, registration())
    assert not second.extracted
    assert harness.extractions == 1


async def test_the_store_records_which_extractor_version_ran(harness: Harness) -> None:
    report = await harness.use_case(PROPOSALS).execute(POLICY, registration())
    assert harness.sources.extracted_with(report.version_id) == "extractor-1+test"


async def test_a_new_extractor_version_re_extracts_an_unchanged_document(
    registry: AssumptionKeyRegistry,
) -> None:
    """A prompt or parser change produces different records, so the cache must miss."""
    harness = Harness(registry)
    await harness.use_case(PROPOSALS).execute(POLICY, registration())
    assert harness.extractions == 1

    harness.extractor_version = "extractor-2+test"
    report = await harness.use_case(PROPOSALS).execute(POLICY, registration())

    assert report.extracted
    assert harness.extractions == 2
    assert harness.sources.extracted_with(report.version_id) == "extractor-2+test"


async def test_force_extraction_overrides_the_marker(harness: Harness) -> None:
    await harness.use_case(PROPOSALS).execute(POLICY, registration())
    report = await harness.use_case(PROPOSALS).execute(
        POLICY, registration(), force_extraction=True
    )
    assert report.extracted
    assert harness.extractions == 2
