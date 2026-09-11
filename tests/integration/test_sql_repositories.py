"""One conformance suite, run against both store implementations.

The risk in adding a second implementation of seven ports is not that it fails - a failure is
loud. It is that it *differs*: returns the same rows in another order, replaces where the other
skips, accumulates where the other overwrites. Every one of those satisfies the type and breaks
a caller.

So most of what follows is parametrized over the in-memory store and the SQL store and asserts
the same thing of both. Where a port's contract is deliberately asymmetric (a cross-tenant
analysis read *raises*; an unadmitted record is *refused*) that is asserted of both too.

A few tests are SQL-only, and they are the ones about being SQL: the transaction boundary, the
unit of work, and the projected columns.

Substrate: SQLite by default, Postgres when ``CCE_TEST_DATABASE_URL`` is set. See
``tests/fixtures/databases.py``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from app.application.orchestration.analysis import AnalysisStage, StageOutcome, StageStatus
from app.application.orchestration.analysis_store import (
    StoredAnalysis,
    TenantMismatch,
    UnknownStoredAnalysis,
)
from app.domain.entities.assumption import AssumptionRecord
from app.domain.entities.findings import FindingDisposition
from app.domain.entities.registry import KeyAlias, KeyRegistryGap
from app.domain.entities.source import Chunk, EvidenceItem, SourceDocument, SourceVersion
from app.domain.value_objects.enums import (
    AliasStatus,
    AuthorityClass,
    BindingRole,
    Disposition,
    Explicitness,
    RegistryGapStatus,
    SourceType,
    ValueType,
)
from app.domain.value_objects.provenance import ModelProvenance
from app.domain.value_objects.typed_value import TypedValue
from app.infrastructure.database.repositories import memory as mem
from app.infrastructure.database.repositories import sql
from app.infrastructure.database.session import Database
from app.judgment.dispositions import DispositionChange, DispositionRecord
from tests.fixtures.databases import disposable_database

pytestmark = pytest.mark.integration

TENANT = "tn_" + "0" * 31 + "1"
OTHER_TENANT = "tn_" + "9" * 32
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 9, 11, 12, 0, tzinfo=dt.UTC)

#: Which implementations every shared test runs against.
IMPLEMENTATIONS = ["memory", "sql"]


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Database]:
    with disposable_database(tmp_path_factory.mktemp("repos"), name="conformance") as handle:
        yield handle


@pytest.fixture(autouse=True)
def _clean(database: Database) -> Iterator[None]:
    """Truncate between tests, so ordering assertions are about writes and not about history."""
    yield
    from app.infrastructure.database.base import Base

    with database.session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())


@dataclass(frozen=True)
class _Bundle:
    """The seven stores of one implementation, built together.

    Typed ``Any`` deliberately: the point of the suite is that both implementations satisfy the
    same *runtime* contract, and annotating the Protocols here would make the test pass or fail
    on a structural-typing question instead of on behaviour.
    """

    sources: Any
    assumptions: Any
    gaps: Any
    aliases: Any
    coupling: Any
    dispositions: Any
    analyses: Any


@pytest.fixture
def stores(request: pytest.FixtureRequest, database: Database) -> _Bundle:
    """Both implementations behind one name, selected by the ``implementation`` parameter."""
    which = getattr(request, "param", "sql")
    if which == "memory":
        return _Bundle(
            sources=mem.InMemorySourceStore(),
            assumptions=mem.InMemoryAssumptionStore(),
            gaps=mem.InMemoryRegistryGapStore(),
            aliases=mem.InMemoryAliasStore(),
            coupling=mem.InMemoryCouplingIndexStore(),
            dispositions=mem.InMemoryDispositionStore(),
            analyses=mem.InMemoryAnalysisStore(),
        )
    return _Bundle(
        sources=sql.SqlSourceStore(database),
        assumptions=sql.SqlAssumptionStore(database),
        gaps=sql.SqlRegistryGapStore(database),
        aliases=sql.SqlAliasStore(database),
        coupling=sql.SqlCouplingIndexStore(database),
        dispositions=sql.SqlDispositionStore(database),
        analyses=sql.SqlAnalysisStore(database),
    )


both = pytest.mark.parametrize("stores", IMPLEMENTATIONS, indirect=True)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _hex(prefix: str, tag: str) -> str:
    """A well-formed identifier from a one-character tag.

    Tags must be hex characters: every id in this system is validated against
    ``^<prefix>_[0-9a-f]{32}$``, so ``"9"`` produces a value the domain refuses - which is a
    fixture bug that reads like a store bug.
    """
    assert len(tag) == 1 and tag in "0123456789abcdef", f"tag must be one hex char: {tag!r}"
    return f"{prefix}_{tag * 32}"


def document(tag: str, *, tenant_id: str = TENANT) -> SourceDocument:
    return SourceDocument(
        source_id=_hex("src", tag),
        tenant_id=tenant_id,
        uri=f"corpus://{tag}.md",
        title=f"Document {tag}",
        source_type=SourceType.POLICY,
        authority_class=AuthorityClass.POLICY,
    )


def version(tag: str, *, source_tag: str | None = None) -> SourceVersion:
    return SourceVersion(
        version_id=_hex("sv", tag),
        source_id=_hex("src", source_tag or tag),
        content_hash="a" * 64,
        version_label=f"v-{tag}",
    )


def evidence(tag: str, *, version_tag: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=_hex("ev", tag),
        source_version_id=_hex("sv", version_tag),
        chunk_id=_hex("ch", tag),
        text_span=f"span {tag} records a retention window of 90 days.",
        char_start=0,
        char_end=48,
        location=f"Document {tag} > Retention",
    )


def chunk(tag: str, *, version_tag: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=_hex("ch", tag),
        source_version_id=_hex("sv", version_tag),
        ordinal=ordinal,
        text=f"chunk {tag}",
        char_start=0,
        char_end=8,
    )


def provenance() -> ModelProvenance:
    return ModelProvenance(
        agent="assumption-extractor",
        prompt_version=1,
        prompt_sha256="0" * 64,
        provider="replay",
        model="recorded-proposals",
        tier="fast",
    )


def record(
    tag: str,
    *,
    key: str = "retention.window",
    version_tag: str = "a",
    tenant_id: str = TENANT,
    admitted: bool = True,
) -> AssumptionRecord:
    return AssumptionRecord(
        assumption_id=_hex("as", tag),
        tenant_id=tenant_id,
        key=key,
        registry_version="registry-1",
        value=TypedValue(value_type=ValueType.DURATION, raw=90, unit="d"),
        role=BindingRole.DECLARES,
        explicitness=Explicitness.EXPLICIT,
        holder_source_version_id=_hex("sv", version_tag),
        evidence_id=_hex("ev", tag),
        extractor_version="extractor-1",
        span_verified=admitted,
        value_parsed=admitted,
        key_resolved=admitted,
    )


def gap(
    tag: str,
    *,
    key: str = "unknown.key",
    count: int = 1,
    status: RegistryGapStatus = RegistryGapStatus.OPEN,
) -> KeyRegistryGap:
    return KeyRegistryGap(
        gap_id=_hex("rg", tag),
        proposed_key=key,
        proposed_value=TypedValue(value_type=ValueType.SCALAR, raw=1, unit="count"),
        evidence_id=_hex("ev", tag),
        holder_source_version_id=_hex("sv", "a"),
        proposed_by=provenance(),
        as_of=AS_OF,
        registry_version="registry-1",
        occurrence_count=count,
        status=status,
    )


def alias(tag: str, *, status: AliasStatus = AliasStatus.PROPOSED) -> KeyAlias:
    approved = status is AliasStatus.APPROVED
    return KeyAlias(
        alias_id=_hex("al", tag),
        source_key=f"legacy.{tag}",
        target_key="retention.window",
        proposed_at=AS_OF,
        approved_by="registry-owner" if approved else None,
        approved_at=AS_OF if approved else None,
        registry_version="registry-1",
        status=status,
        rejection_reason="not equivalent" if status is AliasStatus.REJECTED else None,
    )


def disposition(
    tag: str,
    *,
    analysis_id: str,
    consequence_id: str,
    value: Disposition = Disposition.USEFUL,
    suppress: bool = False,
) -> DispositionRecord:
    return DispositionRecord(
        current=FindingDisposition(
            disposition_id=_hex("dp", tag),
            consequence_id=consequence_id,
            analysis_id=analysis_id,
            assumption_key="retention.window" if suppress else None,
            artifact_ref=_hex("sv", "a") if suppress else None,
            disposition=value,
            note="because the deck was reissued" if suppress else None,
            disposed_by="privacy-office/reviewer-1",
            disposed_at=NOW,
            suppresses_until_source_revision=suppress,
        )
    )


def analysis(tag: str, *, tenant_id: str = TENANT) -> StoredAnalysis:
    from app.domain.entities.change import ChangeSpec

    return StoredAnalysis(
        analysis_id=_hex("an", tag),
        tenant_id=tenant_id,
        run_id=_hex("an", "f"),
        change=ChangeSpec(
            change_id=_hex("chg", tag),
            subject_entities=("personal data retention window",),
            old_state="90 days",
            new_state="30 days",
            version="change-1",
        ),
        run_provenance=None,
        stages=(
            StageOutcome(stage=AnalysisStage.COUPLING, status=StageStatus.COMPLETED, produced=3),
            StageOutcome(stage=AnalysisStage.CLASS_B, status=StageStatus.FAILED, detail="boom"),
        ),
        findings=(),
        evidence=(evidence("a", version_tag="a"),),
        evidence_gap_ids=("eg_" + "1" * 32,),
        open_questions=("who owns the reconciliation step?",),
        fingerprint={"pipeline_version": "analysis-pipeline-1", "stages": {}},
        stored_at=NOW,
        pipeline_version="analysis-pipeline-1",
    )


def seed_documents(stores: _Bundle, *tags: str, tenant_id: str = TENANT) -> None:
    """A document plus one version per tag, which most reads need to exist first."""
    for tag in tags:
        stores.sources.put_document(document(tag, tenant_id=tenant_id))
        stores.sources.put_version(
            version(tag), text=f"span {tag} records a retention window of 90 days."
        )


# ---------------------------------------------------------------------------
# SourceStore
# ---------------------------------------------------------------------------


@both
class TestSourceStore:
    def test_a_document_round_trips_and_is_findable_by_uri(self, stores: _Bundle) -> None:
        stored = document("a")
        stores.sources.put_document(stored)

        assert stores.sources.get_document(TENANT, stored.source_id) == stored
        assert stores.sources.find_document_by_uri(TENANT, stored.uri) == stored

    def test_a_document_is_invisible_to_another_tenant(self, stores: _Bundle) -> None:
        """A correct id from the wrong tenant is a miss, not a read."""
        stored = document("a")
        stores.sources.put_document(stored)

        assert stores.sources.get_document(OTHER_TENANT, stored.source_id) is None
        assert stores.sources.find_document_by_uri(OTHER_TENANT, stored.uri) is None

    def test_a_version_carries_its_canonical_text(self, stores: _Bundle) -> None:
        """Stored together, because a store that could serve one without the other could hand
        out unverifiable evidence."""
        stored = version("a")
        stores.sources.put_document(document("a"))
        stores.sources.put_version(stored, text="span a records a retention window of 90 days.")

        assert stores.sources.get_version(_hex("sv", "a")) == stored
        assert stores.sources.get_text(_hex("sv", "a")) == (
            "span a records a retention window of 90 days."
        )
        assert stores.sources.has_version(_hex("sv", "a"))
        assert not stores.sources.has_version(_hex("sv", "9"))

    def test_versions_come_back_oldest_first(self, stores: _Bundle) -> None:
        """Insertion order, which SQL does not give without being told to store it."""
        stores.sources.put_document(document("a"))
        for tag in ("c", "b", "d"):
            stores.sources.put_version(version(tag, source_tag="a"), text=f"text {tag}")

        assert [v.version_label for v in stores.sources.versions_for(_hex("src", "a"))] == [
            "v-c",
            "v-b",
            "v-d",
        ]

    def test_re_putting_a_version_updates_it_without_moving_it_in_history(
        self, stores: _Bundle
    ) -> None:
        stores.sources.put_document(document("a"))
        stores.sources.put_version(version("b", source_tag="a"), text="first")
        stores.sources.put_version(version("c", source_tag="a"), text="second")
        stores.sources.put_version(version("b", source_tag="a"), text="restated")

        history = stores.sources.versions_for(_hex("src", "a"))
        assert [v.version_label for v in history] == ["v-b", "v-c"]
        assert stores.sources.get_text(_hex("sv", "b")) == "restated"

    def test_the_extraction_marker_is_the_extractor_version_not_a_boolean(
        self, stores: _Bundle
    ) -> None:
        """Returning the version is what makes a prompt change re-extract."""
        seed_documents(stores, "a")

        assert stores.sources.extracted_with(_hex("sv", "a")) is None
        stores.sources.mark_extracted(_hex("sv", "a"), extractor_version="extractor-2")
        assert stores.sources.extracted_with(_hex("sv", "a")) == "extractor-2"

    def test_chunks_come_back_in_ordinal_order_and_are_written_once(self, stores: _Bundle) -> None:
        seed_documents(stores, "a")
        stores.sources.put_chunks(
            (
                chunk("c", version_tag="a", ordinal=2),
                chunk("b", version_tag="a", ordinal=1),
            )
        )
        # A second write of the same chunk id must not duplicate it.
        stores.sources.put_chunks((chunk("b", version_tag="a", ordinal=1),))

        chunks = stores.sources.chunks_for(_hex("sv", "a"))
        assert [c.ordinal for c in chunks] == [1, 2]

    def test_evidence_keeps_the_first_write(self, stores: _Bundle) -> None:
        """Evidence ids are content-derived, so a second write with the same id is the same
        span; replacing it would be a write with no reader."""
        seed_documents(stores, "a")
        first = evidence("b", version_tag="a")
        stores.sources.put_evidence((first,))
        stores.sources.put_evidence((first.model_copy(update={"location": "elsewhere"}),))

        assert stores.sources.get_evidence(first.evidence_id) == first

    def test_evidence_for_a_tenant_resolves_through_the_version_and_sorts_by_id(
        self, stores: _Bundle
    ) -> None:
        """A span carries no tenant of its own, so this is a join and not a column read."""
        seed_documents(stores, "a")
        seed_documents(stores, "9", tenant_id=OTHER_TENANT)
        stores.sources.put_evidence(
            (evidence("c", version_tag="a"), evidence("b", version_tag="a"))
        )
        stores.sources.put_evidence((evidence("8", version_tag="9"),))

        ours = stores.sources.evidence_for_tenant(TENANT)
        assert [e.evidence_id for e in ours] == sorted(e.evidence_id for e in ours)
        assert {e.evidence_id for e in ours} == {_hex("ev", "b"), _hex("ev", "c")}
        assert {e.evidence_id for e in stores.sources.evidence_for_tenant(OTHER_TENANT)} == {
            _hex("ev", "8")
        }


# ---------------------------------------------------------------------------
# AssumptionStore
# ---------------------------------------------------------------------------


@both
class TestAssumptionStore:
    def test_records_round_trip_and_are_queryable_three_ways(self, stores: _Bundle) -> None:
        seed_documents(stores, "a")
        stored = record("b", version_tag="a")
        stores.assumptions.put_records((stored,))

        assert stores.assumptions.records_for_version(_hex("sv", "a")) == (stored,)
        assert stores.assumptions.records_for_key(TENANT, "retention.window") == (stored,)
        assert stores.assumptions.all_records(TENANT) == (stored,)
        assert stores.assumptions.has_records_for_version(_hex("sv", "a"))

    def test_an_unadmitted_record_is_refused(self, stores: _Bundle) -> None:
        """The store holds admitted bindings only, and there is no path to put one in."""
        seed_documents(stores, "a")

        with pytest.raises(ValueError, match="admitted bindings only"):
            stores.assumptions.put_records((record("b", version_tag="a", admitted=False),))
        assert stores.assumptions.all_records(TENANT) == ()

    def test_a_re_put_record_is_skipped_not_duplicated(self, stores: _Bundle) -> None:
        seed_documents(stores, "a")
        stored = record("b", version_tag="a")
        stores.assumptions.put_records((stored,))
        stores.assumptions.put_records((stored,))

        assert stores.assumptions.all_records(TENANT) == (stored,)

    def test_records_keep_the_order_they_were_written_in(self, stores: _Bundle) -> None:
        seed_documents(stores, "a")
        stores.assumptions.put_records((record("c", version_tag="a"),))
        stores.assumptions.put_records((record("b", version_tag="a"),))

        assert [r.assumption_id for r in stores.assumptions.all_records(TENANT)] == [
            _hex("as", "c"),
            _hex("as", "b"),
        ]

    def test_another_tenants_records_are_not_returned(self, stores: _Bundle) -> None:
        seed_documents(stores, "a")
        ours = record("b", version_tag="a")
        theirs = record("c", version_tag="a", tenant_id=OTHER_TENANT)
        stores.assumptions.put_records((ours, theirs))

        assert stores.assumptions.all_records(TENANT) == (ours,)
        assert stores.assumptions.records_for_key(OTHER_TENANT, "retention.window") == (theirs,)


# ---------------------------------------------------------------------------
# RegistryGapStore and AliasStore
# ---------------------------------------------------------------------------


@both
class TestRegistryGapStore:
    def test_the_same_gap_seen_again_accumulates_rather_than_duplicating(
        self, stores: _Bundle
    ) -> None:
        """Gap volume is a product signal, so the second sighting is a count and not a row."""
        stores.gaps.put_gap(gap("a", count=1))
        stores.gaps.put_gap(gap("a", count=2))

        stored = stores.gaps.get_gap(_hex("rg", "a"))
        assert stored is not None
        assert stored.occurrence_count == 3
        assert stores.gaps.gap_counts_by_key() == {"unknown.key": 3}

    def test_only_unregistered_gaps_block_class_a(self, stores: _Bundle) -> None:
        stores.gaps.put_gap(gap("a"))
        stores.gaps.put_gap(gap("b", key="other.key", status=RegistryGapStatus.REGISTERED))

        assert {g.gap_id for g in stores.gaps.open_gaps()} == {_hex("rg", "a")}
        # The registered one is still stored, and still counted.
        assert stores.gaps.get_gap(_hex("rg", "b")) is not None
        assert set(stores.gaps.gap_counts_by_key()) == {"unknown.key", "other.key"}


@both
class TestAliasStore:
    def test_only_approved_aliases_participate_in_class_a(self, stores: _Bundle) -> None:
        stores.aliases.put_alias(alias("a"))
        stores.aliases.put_alias(alias("b", status=AliasStatus.APPROVED))
        stores.aliases.put_alias(alias("c", status=AliasStatus.REJECTED))

        assert {a.alias_id for a in stores.aliases.approved()} == {_hex("al", "b")}
        assert len(stores.aliases.all_aliases()) == 3

    def test_putting_an_alias_replaces_it(self, stores: _Bundle) -> None:
        """Unlike a gap: an approval is a new state for the same alias, not another sighting."""
        stores.aliases.put_alias(alias("a"))
        stores.aliases.put_alias(alias("a", status=AliasStatus.APPROVED))

        stored = stores.aliases.get_alias(_hex("al", "a"))
        assert stored is not None
        assert stored.status is AliasStatus.APPROVED
        assert stored.approved_by == "registry-owner"
        assert len(stores.aliases.all_aliases()) == 1


# ---------------------------------------------------------------------------
# CouplingIndexStore
# ---------------------------------------------------------------------------


@both
class TestCouplingIndexStore:
    def test_a_tenants_slice_is_replaced_wholesale_not_merged(self, stores: _Bundle) -> None:
        """The contract, not a shortcut: a merge would leave a stale cell producing a finding
        nobody could trace to a document."""
        from app.domain.entities.class_a import ClassAFinding

        first = _findings(("retention.window", "access.review_interval"))
        stores.coupling.put_findings(TENANT, first)
        assert len(stores.coupling.findings(TENANT)) == 2

        second = _findings(("retention.window",))
        stores.coupling.put_findings(TENANT, second)

        surviving = stores.coupling.findings(TENANT)
        assert len(surviving) == 1
        assert isinstance(surviving[0], ClassAFinding)
        assert surviving[0].assumption_key == "retention.window"

    def test_findings_are_queryable_by_key_and_scoped_by_tenant(self, stores: _Bundle) -> None:
        stores.coupling.put_findings(TENANT, _findings(("retention.window", "api.max_page_size")))

        assert len(stores.coupling.findings_for_key(TENANT, "retention.window")) == 1
        assert stores.coupling.findings_for_key(TENANT, "nothing.here") == ()
        assert stores.coupling.findings(OTHER_TENANT) == ()

    def test_an_empty_projection_reads_back_as_empty_rather_than_raising(
        self, stores: _Bundle
    ) -> None:
        assert stores.coupling.coupling_sets(TENANT) == ()
        assert stores.coupling.relations(TENANT) == ()
        assert stores.coupling.findings(TENANT) == ()


def _findings(keys: tuple[str, ...]) -> tuple[Any, ...]:
    """Minimal Class A findings. Built here because only the key and the id are read."""
    from app.domain.entities.class_a import ClassAFinding
    from app.domain.value_objects.scope import ScopeSelector
    from app.domain.value_objects.taxonomy import Relationship

    return tuple(
        ClassAFinding(
            finding_id=_hex("ca", chr(ord("a") + index)),
            tenant_id=TENANT,
            relation_id=_hex("dr", chr(ord("a") + index)),
            relationship=Relationship.CONFLICTS_WITH,
            rule_id="same_scope_contradiction",
            rule_version="class-a-1",
            assumption_key=key,
            scope=ScopeSelector.global_scope(),
            as_of=AS_OF,
            affected_source_version_id=_hex("sv", "a"),
            affected_source_id=_hex("src", "a"),
            affected_artifact=f"Document about {key}",
            summary=f"Two artifacts disagree about {key}.",
            input_record_ids=(_hex("as", "a"), _hex("as", "b")),
            materialized_evidence_ids=(_hex("ev", "a"),),
            registry_version="registry-1",
            coupling_index_version="coupling-1",
            authority_model_version="authority-1",
        )
        for index, key in enumerate(keys)
    )


# ---------------------------------------------------------------------------
# DispositionStore
# ---------------------------------------------------------------------------


@both
class TestDispositionStore:
    def test_a_verdict_round_trips_with_its_history_and_superseded_values(
        self, stores: _Bundle
    ) -> None:
        analysis_id, consequence_id = _hex("an", "a"), _hex("cq", "a")
        original = disposition("a", analysis_id=analysis_id, consequence_id=consequence_id)
        changed = DispositionRecord(
            current=original.current.model_copy(
                update={"disposition": Disposition.NOT_MATERIAL, "note": "out of scope"}
            ),
            history=(
                DispositionChange(
                    from_disposition=Disposition.USEFUL,
                    to_disposition=Disposition.NOT_MATERIAL,
                    changed_at=NOW,
                    changed_by="privacy-office/reviewer-1",
                    reason="out of scope after all",
                ),
            ),
            superseded=(original.current,),
        )
        stores.dispositions.put(changed)

        read = stores.dispositions.get(analysis_id, consequence_id)
        assert read == changed
        assert read is not None
        assert read.original is Disposition.USEFUL
        assert read.disposition is Disposition.NOT_MATERIAL
        assert read.suppresses_foreground

    def test_put_replaces_the_verdict_for_one_pair(self, stores: _Bundle) -> None:
        analysis_id, consequence_id = _hex("an", "a"), _hex("cq", "a")
        stores.dispositions.put(
            disposition("a", analysis_id=analysis_id, consequence_id=consequence_id)
        )
        stores.dispositions.put(
            disposition(
                "a",
                analysis_id=analysis_id,
                consequence_id=consequence_id,
                value=Disposition.UNVERIFIABLE,
            )
        )

        assert len(stores.dispositions.for_analysis(analysis_id)) == 1
        read = stores.dispositions.get(analysis_id, consequence_id)
        assert read is not None and read.disposition is Disposition.UNVERIFIABLE

    def test_an_analysis_reads_its_verdicts_sorted_by_consequence(self, stores: _Bundle) -> None:
        analysis_id = _hex("an", "a")
        for tag in ("c", "b"):
            stores.dispositions.put(
                disposition(tag, analysis_id=analysis_id, consequence_id=_hex("cq", tag))
            )

        ids = [r.current.consequence_id for r in stores.dispositions.for_analysis(analysis_id)]
        assert ids == sorted(ids)

    def test_a_consequence_reads_every_analysis_that_ruled_on_it(self, stores: _Bundle) -> None:
        """The lookup that makes a cross-analysis submission distinguishable from a typo."""
        consequence_id = _hex("cq", "a")
        for tag in ("c", "b"):
            stores.dispositions.put(
                disposition(tag, analysis_id=_hex("an", tag), consequence_id=consequence_id)
            )

        found = stores.dispositions.find_consequence(consequence_id)
        assert [r.current.analysis_id for r in found] == sorted(
            r.current.analysis_id for r in found
        )
        assert len(found) == 2

    def test_a_suppression_scope_survives_the_round_trip(self, stores: _Bundle) -> None:
        analysis_id, consequence_id = _hex("an", "a"), _hex("cq", "a")
        stores.dispositions.put(
            disposition(
                "a",
                analysis_id=analysis_id,
                consequence_id=consequence_id,
                value=Disposition.WRONG,
                suppress=True,
            )
        )

        read = stores.dispositions.get(analysis_id, consequence_id)
        assert read is not None
        assert read.suppression_scope == ("retention.window", _hex("sv", "a"))


# ---------------------------------------------------------------------------
# AnalysisStore
# ---------------------------------------------------------------------------


@both
class TestAnalysisStore:
    def test_a_snapshot_round_trips_whole(self, stores: _Bundle) -> None:
        stored = analysis("a")
        stores.analyses.put(stored)

        read = stores.analyses.require(TENANT, stored.analysis_id)
        assert read == stored

    def test_a_cross_tenant_read_raises_and_a_real_miss_returns_none(self, stores: _Bundle) -> None:
        stored = analysis("a")
        stores.analyses.put(stored)

        with pytest.raises(TenantMismatch, match="another tenant"):
            stores.analyses.get(OTHER_TENANT, stored.analysis_id)
        assert stores.analyses.get(TENANT, _hex("an", "9")) is None
        with pytest.raises(UnknownStoredAnalysis):
            stores.analyses.require(TENANT, _hex("an", "9"))

    def test_a_tenants_analyses_are_listed_sorted_and_scoped(self, stores: _Bundle) -> None:
        stores.analyses.put(analysis("c"))
        stores.analyses.put(analysis("b"))
        stores.analyses.put(analysis("9", tenant_id=OTHER_TENANT))

        assert stores.analyses.for_tenant(TENANT) == (_hex("an", "b"), _hex("an", "c"))
        assert stores.analyses.for_tenant(OTHER_TENANT) == (_hex("an", "9"),)

    def test_put_replaces_the_snapshot_for_one_id(self, stores: _Bundle) -> None:
        stores.analyses.put(analysis("a"))
        stores.analyses.put(analysis("a"))

        assert stores.analyses.for_tenant(TENANT) == (_hex("an", "a"),)

    def test_a_failed_stage_survives_as_failed(self, stores: _Bundle) -> None:
        """`empty` and `failed` are different facts; a codec that lost the difference would
        make an outage look like a clean run that found nothing."""
        stores.analyses.put(analysis("a"))

        read = stores.analyses.require(TENANT, _hex("an", "a"))
        statuses = {outcome.stage: outcome.status for outcome in read.stages}
        assert statuses[AnalysisStage.CLASS_B] is StageStatus.FAILED
        assert statuses[AnalysisStage.COUPLING] is StageStatus.COMPLETED
        failed = next(o for o in read.stages if o.status is StageStatus.FAILED)
        assert failed.detail == "boom"


# ---------------------------------------------------------------------------
# SQL-only: the things that are about being SQL
# ---------------------------------------------------------------------------


class TestTransactionBoundary:
    def test_each_call_is_its_own_transaction_by_default(self, database: Database) -> None:
        """Which is the faithful mapping: the ports have no transaction and the in-memory
        stores commit immediately."""
        aliases = sql.SqlAliasStore(database)
        aliases.put_alias(alias("a"))

        # A second handle on the same database sees it, so it was committed and not merely
        # pending in a session.
        other = sql.SqlAliasStore(Database.from_url(database.engine.url.render_as_string(False)))
        try:
            assert other.get_alias(_hex("al", "a")) is not None
        finally:
            other._db.dispose()

    def test_a_unit_of_work_makes_several_calls_atomic(self, database: Database) -> None:
        """The case it exists for: three coupling writes that must land together."""
        coupling = sql.SqlCouplingIndexStore(database)

        with pytest.raises(RuntimeError, match="deliberate"), database.unit_of_work():
            coupling.put_findings(TENANT, _findings(("retention.window",)))
            raise RuntimeError("a deliberate failure mid-projection")

        assert coupling.findings(TENANT) == ()

    def test_a_nested_unit_of_work_is_refused(self, database: Database) -> None:
        """Rather than silently flattened: a caller relying on an independent inner rollback
        would be wrong in a way nothing reports."""
        with database.unit_of_work():
            opened = database.unit_of_work()
            with pytest.raises(RuntimeError, match="already open"):
                opened.__enter__()

    def test_marking_an_unknown_version_extracted_is_refused(self, database: Database) -> None:
        """A wiring defect, not a row to create. The in-memory store cannot tell the
        difference, which is a gap in the in-memory store rather than in this one."""
        sources = sql.SqlSourceStore(database)

        with pytest.raises(KeyError, match="no such version"):
            sources.mark_extracted(_hex("sv", "9"), extractor_version="extractor-1")


class TestProjectedColumns:
    def test_a_projected_column_is_never_the_value_that_is_served(self, database: Database) -> None:
        """The payload is the source of truth; a column exists so a filter is indexable.

        Corrupting the column changes which rows a filter finds and cannot change what a
        found row says - which is the property that makes the denormalization safe.
        """
        from app.infrastructure.database.models.tables import KeyAliasRow

        aliases = sql.SqlAliasStore(database)
        aliases.put_alias(alias("a", status=AliasStatus.APPROVED))

        with database.session() as session:
            row = session.get(KeyAliasRow, _hex("al", "a"))
            assert row is not None
            row.status = AliasStatus.PROPOSED.value

        served = aliases.get_alias(_hex("al", "a"))
        assert served is not None
        assert served.status is AliasStatus.APPROVED  # from the payload
        assert aliases.approved() == ()  # the filter followed the column

    def test_the_suppression_scope_is_projected_for_the_next_analysis_to_find(
        self, database: Database
    ) -> None:
        from app.infrastructure.database.models.tables import FindingDispositionRow

        dispositions = sql.SqlDispositionStore(database)
        dispositions.put(
            disposition(
                "a",
                analysis_id=_hex("an", "a"),
                consequence_id=_hex("cq", "a"),
                value=Disposition.WRONG,
                suppress=True,
            )
        )

        with database.session() as session:
            row = session.get(FindingDispositionRow, (_hex("an", "a"), _hex("cq", "a")))
            assert row is not None
            assert row.suppression_key == "retention.window"
            assert row.suppression_artifact_ref == _hex("sv", "a")
            assert row.disposition == Disposition.WRONG.value

    def test_a_non_suppressing_verdict_projects_no_scope(self, database: Database) -> None:
        from app.infrastructure.database.models.tables import FindingDispositionRow

        dispositions = sql.SqlDispositionStore(database)
        dispositions.put(
            disposition(
                "a",
                analysis_id=_hex("an", "a"),
                consequence_id=_hex("cq", "a"),
                value=Disposition.ALREADY_KNOWN,
            )
        )

        with database.session() as session:
            row = session.get(FindingDispositionRow, (_hex("an", "a"), _hex("cq", "a")))
            assert row is not None
            assert row.suppression_key is None
            assert row.suppression_artifact_ref is None


class TestConfiguration:
    def test_an_unset_database_url_refuses_at_the_point_of_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not at import, and not with a default carrying credentials."""
        from app.config.settings import get_settings
        from app.infrastructure.database.session import DatabaseNotConfigured, database_url

        monkeypatch.delenv("CCE_DATABASE_URL", raising=False)
        get_settings.cache_clear()
        try:
            with pytest.raises(DatabaseNotConfigured, match="CCE_DATABASE_URL"):
                database_url()
        finally:
            get_settings.cache_clear()

    def test_the_settings_default_carries_no_credential(self) -> None:
        """A default of `postgres://cce:cce@...` is a credential in source."""
        from app.config.settings import Settings

        assert Settings.model_fields["database_url"].default is None

    def test_the_env_example_is_a_placeholder_not_a_password(self) -> None:
        from app.config.settings import REPO_ROOT

        example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        line = next(entry for entry in example.splitlines() if entry.startswith("CCE_DATABASE_URL"))
        assert "PASSWORD" in line or line.endswith("=")
