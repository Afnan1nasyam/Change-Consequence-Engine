"""Stored analyses: what a reviewer retrieves, scoped to their tenant, unchangeable after.

Two finished features were inert without this store. The disposition path had to be *handed*
the consequences a reviewer saw, so a caller passing the wrong list would have had its verdict
accepted. And persistent suppression was recorded, scoped and expiry-tested with nothing
reading it, so the next analysis re-reported a finding a reviewer had already rejected -
exactly the failure ``IDEA.md`` 21.2 says makes reviewers stop reading.

The tests carrying the most weight:

* :func:`TestTenantIsolation.test_another_tenant_gets_a_refusal_not_a_miss`
* :func:`TestDispositionValidation.test_a_verdict_on_a_finding_from_another_analysis_is_refused`
* :func:`TestSuppressionAcrossAnalyses.test_a_suppression_survives_a_new_analysis_of_the_same_version`
* :func:`TestSuppressionAcrossAnalyses.test_a_revised_source_version_lifts_the_suppression`
* :func:`TestRoundTrip.test_the_fingerprint_is_stored_verbatim`
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
import pathlib
from typing import Any

import pytest

from app.application.orchestration.analysis import (
    AnalysisStage,
    StageOutcome,
    StageStatus,
)
from app.application.orchestration.analysis_store import (
    ANALYSIS_SNAPSHOT_VERSION,
    AnalysisStore,
    ReviewedFinding,
    StoredAnalysis,
    TenantMismatch,
    UnknownStoredAnalysis,
    snapshot,
)
from app.application.use_cases.record_disposition import RecordDispositionUseCase
from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import (
    ArbitrationDecision,
    Consequence,
    ConsequenceScore,
)
from app.domain.entities.source import EvidenceItem
from app.domain.value_objects.enums import (
    ConsequenceClass,
    Disposition,
    Origin,
    Severity,
    Status,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import ModelProvenance, RunProvenance
from app.infrastructure.database.repositories.memory import (
    InMemoryAnalysisStore,
    InMemoryDispositionStore,
)
from app.judgment.dispositions import (
    CrossAnalysisConsequence,
    UnknownAnalysis,
    UnknownConsequence,
)

TENANT = "tn_" + "0" * 31 + "1"
OTHER_TENANT = "tn_" + "0" * 31 + "2"
ANALYSIS = new_id("analysis")
LATER_ANALYSIS = new_id("analysis")
KEY = "retention.window"
VERSION = "sv_" + "1" * 32
REVISED = "sv_" + "2" * 32
EVIDENCE = "ev_" + "a" * 32
REVIEWER = "privacy-office/reviewer-1"
AT = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)
LATER = dt.datetime(2026, 9, 12, 9, 0, tzinfo=dt.UTC)
WHY = "The deck was reissued before the change; the mechanism does not hold."


# ---------------------------------------------------------------------------
# Builders - a stored analysis assembled directly, and one via ``snapshot``
# ---------------------------------------------------------------------------


def change() -> ChangeSpec:
    return ChangeSpec(
        change_id=new_id("change"),
        subject_entities=("personal data retention window",),
        old_state="90 days",
        new_state="30 days",
        version="change-1",
    )


def consequence(**over: Any) -> Consequence:
    payload: dict[str, Any] = {
        "consequence_id": new_id("consequence"),
        "change_id": new_id("change"),
        "title": "The deck teaches the old retention window",
        "description": "The deck states 90 days where the policy now states 30.",
        "category": "stale_consumer",
        "consequence_class": ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED,
        "origin": Origin.RESOLVER,
        "affected_entity": "Refund Handling Onboarding Deck",
        "mechanism": "rule stale_consumer over two bindings of retention.window",
        "derived_relation_ids": (new_id("relation"),),
        "assumption_keys": (KEY,),
        "status": Status.LIKELY,
        "severity": Severity.HIGH,
        "evidence_ids": (EVIDENCE,),
        "run_provenance": RunProvenance(schema_version="schema-1"),
    }
    payload.update(over)
    return Consequence(**payload)


def decision(consequence_id: str, **over: Any) -> ArbitrationDecision:
    payload: dict[str, Any] = {
        "consequence_id": consequence_id,
        "status": Status.LIKELY,
        "severity": Severity.HIGH,
        "rationale": "The deck contradicts the governing policy and is in active use.",
        "supporting_evidence_ids": (EVIDENCE,),
        "in_default_view": True,
        "produced_by": ModelProvenance(
            agent="consequence-judge",
            prompt_version=4,
            prompt_sha256="0" * 64,
            provider="fake",
            model="fake-model",
            tier="standard",
        ),
    }
    payload.update(over)
    return ArbitrationDecision(**payload)


def score(consequence_id: str) -> ConsequenceScore:
    return ConsequenceScore(
        consequence_id=consequence_id,
        confidence=0.42,
        confidence_fn_version="confidence-1",
        factor_inputs={"source_authority": 0.8},
        computed_at=AT,
    )


def span() -> EvidenceItem:
    return EvidenceItem(
        evidence_id=EVIDENCE,
        source_version_id=VERSION,
        chunk_id="ch_" + "a" * 32,
        text_span="Personal data is retained for 90 days after the account is closed.",
        char_start=0,
        char_end=66,
    )


def stored(
    *,
    analysis_id: str = ANALYSIS,
    tenant_id: str = TENANT,
    findings: tuple[ReviewedFinding, ...] | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> StoredAnalysis:
    """A snapshot assembled directly, for the store's own behaviour."""
    if findings is None:
        subject = consequence()
        findings = (
            ReviewedFinding(
                consequence=subject,
                decision=decision(subject.consequence_id),
                score=score(subject.consequence_id),
                artifact_version_id=VERSION,
                in_default_view=True,
            ),
        )
    return StoredAnalysis(
        analysis_id=analysis_id,
        tenant_id=tenant_id,
        run_id=new_id("analysis"),
        change=change(),
        run_provenance=RunProvenance(schema_version="schema-1", registry_version="registry-1"),
        stages=(
            StageOutcome(stage=AnalysisStage.COUPLING, status=StageStatus.COMPLETED, produced=21),
            StageOutcome(stage=AnalysisStage.CLASS_A, status=StageStatus.COMPLETED, produced=9),
        ),
        findings=findings,
        evidence=(span(),),
        evidence_gap_ids=("eg_" + "1" * 32,),
        open_questions=("What is the current deck revision?",),
        fingerprint=fingerprint if fingerprint is not None else {"consequences": [["likely"]]},
        stored_at=AT,
        pipeline_version="analysis-pipeline-1",
    )


def wired(
    analyses: InMemoryAnalysisStore | None = None,
) -> tuple[RecordDispositionUseCase, InMemoryAnalysisStore, InMemoryDispositionStore]:
    analysis_store = analyses or InMemoryAnalysisStore()
    disposition_store = InMemoryDispositionStore()
    return (
        RecordDispositionUseCase(store=disposition_store, analyses=analysis_store),
        analysis_store,
        disposition_store,
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_an_analysis_is_retrievable_by_id(self) -> None:
        store = InMemoryAnalysisStore()
        record = stored()
        store.put(record)

        assert store.get(TENANT, ANALYSIS) is record
        assert store.require(TENANT, ANALYSIS) is record
        assert store.for_tenant(TENANT) == (ANALYSIS,)

    def test_the_review_surface_survives_the_round_trip(self) -> None:
        """Everything a reviewer needs to judge the finding, and the judge's own view."""
        store = InMemoryAnalysisStore()
        record = stored()
        store.put(record)

        read = store.require(TENANT, ANALYSIS)
        finding = read.findings[0]

        assert read.change == record.change
        assert read.run_provenance == record.run_provenance
        assert read.stages == record.stages
        assert finding.consequence.status is Status.LIKELY
        assert finding.decision is not None and finding.decision.rationale
        assert finding.score is not None and finding.score.confidence == 0.42
        assert finding.in_default_view
        assert read.foregrounded == (finding.consequence_id,)
        assert read.evidence_by_id[EVIDENCE].text_span

    def test_the_fingerprint_is_stored_verbatim(self) -> None:
        """Stored, never recomputed - a recomputed value could drift from its analysis.

        It is what lets a verdict be checked against the analysis that produced it: a re-run
        yielding a different fingerprint is a different analysis, and a verdict from one does
        not apply to the other.
        """
        store = InMemoryAnalysisStore()
        expected = {"consequences": [["likely", "HIGH", "stale_consumer", "Deck"]]}
        store.put(stored(fingerprint=expected))

        assert store.require(TENANT, ANALYSIS).fingerprint == expected

    def test_the_snapshot_version_is_recorded(self) -> None:
        """A snapshot read back under a different shape is not what the reviewer saw."""
        assert stored().snapshot_version == ANALYSIS_SNAPSHOT_VERSION

    def test_a_missing_analysis_reads_as_none_and_requires_as_a_refusal(self) -> None:
        """Both, because the two callers differ: a listing wants ``None``, a verdict does not."""
        store = InMemoryAnalysisStore()

        assert store.get(TENANT, ANALYSIS) is None
        with pytest.raises(UnknownStoredAnalysis, match="no stored analysis"):
            store.require(TENANT, ANALYSIS)

    def test_finding_lookup_answers_per_consequence(self) -> None:
        store = InMemoryAnalysisStore()
        record = stored()
        store.put(record)
        read = store.require(TENANT, ANALYSIS)
        consequence_id = read.findings[0].consequence_id

        assert read.finding(consequence_id) is read.findings[0]
        assert read.finding(new_id("consequence")) is None
        assert read.artifact_version_of(consequence_id) == VERSION
        assert read.artifact_version_of(new_id("consequence")) is None

    def test_the_store_offers_no_way_to_edit_a_stored_finding(self) -> None:
        """A verdict refers to an analysis, so the analysis must not change under it.

        An editable snapshot would make "what did the reviewer see?" unanswerable, which is
        the one question it exists to answer.
        """
        methods = {
            name for name, _ in inspect.getmembers(InMemoryAnalysisStore, inspect.isfunction)
        }
        assert methods == {"__init__", "put", "get", "require", "for_tenant"}
        assert isinstance(InMemoryAnalysisStore(), AnalysisStore)


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_another_tenant_gets_a_refusal_not_a_miss(self) -> None:
        """A cross-tenant read is an isolation defect. Returning ``None`` would hide it.

        The distinction is the whole point: "not found" is an ordinary miss a UI renders,
        while a request for another tenant's analysis needs investigating.
        """
        store = InMemoryAnalysisStore()
        store.put(stored(tenant_id=TENANT))

        with pytest.raises(TenantMismatch, match="another tenant"):
            store.get(OTHER_TENANT, ANALYSIS)
        with pytest.raises(TenantMismatch):
            store.require(OTHER_TENANT, ANALYSIS)

    def test_a_tenant_lists_only_its_own_analyses(self) -> None:
        store = InMemoryAnalysisStore()
        store.put(stored(analysis_id=ANALYSIS, tenant_id=TENANT))
        store.put(stored(analysis_id=LATER_ANALYSIS, tenant_id=OTHER_TENANT))

        assert store.for_tenant(TENANT) == (ANALYSIS,)
        assert store.for_tenant(OTHER_TENANT) == (LATER_ANALYSIS,)

    def test_tenant_is_a_parameter_on_every_read(self) -> None:
        """Matching the ingestion ports: a caller cannot forget to scope a lookup."""
        for name in ("get", "require", "for_tenant"):
            parameters = set(inspect.signature(getattr(AnalysisStore, name)).parameters)
            assert "tenant_id" in parameters, name

    def test_a_verdict_cannot_be_recorded_against_another_tenants_analysis(self) -> None:
        subject, analyses, _dispositions = wired()
        analyses.put(stored(tenant_id=TENANT))
        consequence_id = analyses.require(TENANT, ANALYSIS).findings[0].consequence_id

        with pytest.raises(TenantMismatch):
            subject.record_reviewed(
                tenant_id=OTHER_TENANT,
                analysis_id=ANALYSIS,
                consequence_id=consequence_id,
                disposition=Disposition.USEFUL,
                disposed_by=REVIEWER,
                disposed_at=AT,
            )


# ---------------------------------------------------------------------------
# Disposition validation against the stored analysis
# ---------------------------------------------------------------------------


class TestDispositionValidation:
    def test_a_verdict_against_the_correct_analysis_is_accepted(self) -> None:
        subject, analyses, dispositions = wired()
        analyses.put(stored())
        consequence_id = analyses.require(TENANT, ANALYSIS).findings[0].consequence_id

        outcome = subject.record_reviewed(
            tenant_id=TENANT,
            analysis_id=ANALYSIS,
            consequence_id=consequence_id,
            disposition=Disposition.USEFUL,
            disposed_by=REVIEWER,
            disposed_at=AT,
        )

        assert outcome.created
        assert dispositions.get(ANALYSIS, consequence_id) is outcome.record

    def test_a_verdict_on_a_finding_from_another_analysis_is_refused(self) -> None:
        """The failure the store exists to catch.

        Without it a caller could pass any list, and a reviewer's "this is wrong" would land
        on the right finding in the wrong run - attributed, auditable, and about a report they
        never saw.
        """
        subject, analyses, _dispositions = wired()
        mine = stored(analysis_id=ANALYSIS)
        theirs = stored(analysis_id=LATER_ANALYSIS)
        analyses.put(mine)
        analyses.put(theirs)

        # Either refusal is correct, and which one depends on whether the *other* analysis
        # already has a verdict on that consequence: with one it is a cross-analysis defect,
        # without one the finding is simply not in this analysis. Both are refusals.
        with pytest.raises((CrossAnalysisConsequence, UnknownConsequence)):
            subject.record_reviewed(
                tenant_id=TENANT,
                analysis_id=ANALYSIS,
                consequence_id=theirs.findings[0].consequence_id,
                disposition=Disposition.USEFUL,
                disposed_by=REVIEWER,
                disposed_at=AT,
            )

    def test_a_verdict_against_an_unknown_analysis_is_refused(self) -> None:
        subject, _analyses, _dispositions = wired()

        with pytest.raises(UnknownStoredAnalysis, match="no stored analysis"):
            subject.record_reviewed(
                tenant_id=TENANT,
                analysis_id=ANALYSIS,
                consequence_id=new_id("consequence"),
                disposition=Disposition.USEFUL,
                disposed_by=REVIEWER,
                disposed_at=AT,
            )

    def test_a_use_case_with_no_store_says_so_rather_than_guessing(self) -> None:
        subject = RecordDispositionUseCase(store=InMemoryDispositionStore())

        with pytest.raises(UnknownAnalysis, match="without an analysis store"):
            subject.record_reviewed(
                tenant_id=TENANT,
                analysis_id=ANALYSIS,
                consequence_id=new_id("consequence"),
                disposition=Disposition.USEFUL,
                disposed_by=REVIEWER,
                disposed_at=AT,
            )

    def test_the_artifact_reference_comes_from_the_stored_finding(self) -> None:
        """Not from the caller. It is what makes a suppression expire on revision."""
        subject, analyses, _dispositions = wired()
        analyses.put(stored())
        consequence_id = analyses.require(TENANT, ANALYSIS).findings[0].consequence_id

        outcome = subject.record_reviewed(
            tenant_id=TENANT,
            analysis_id=ANALYSIS,
            consequence_id=consequence_id,
            disposition=Disposition.WRONG,
            disposed_by=REVIEWER,
            disposed_at=AT,
            note=WHY,
            suppress_until_source_revision=True,
        )

        assert outcome.record.current.artifact_ref == VERSION
        assert outcome.record.suppression_scope == (KEY, VERSION)

    def test_the_reconstructed_view_applies_the_verdicts_since(self) -> None:
        """Requirement 3: reconstruct exactly what the reviewer should now see."""
        subject, analyses, _dispositions = wired()
        analyses.put(stored())
        consequence_id = analyses.require(TENANT, ANALYSIS).findings[0].consequence_id

        before = subject.foreground_reviewed(tenant_id=TENANT, analysis_id=ANALYSIS)
        assert len(before.shown) == 1

        subject.record_reviewed(
            tenant_id=TENANT,
            analysis_id=ANALYSIS,
            consequence_id=consequence_id,
            disposition=Disposition.WRONG,
            disposed_by=REVIEWER,
            disposed_at=AT,
            note=WHY,
        )

        after = subject.foreground_reviewed(tenant_id=TENANT, analysis_id=ANALYSIS)
        assert after.shown == ()
        assert after.suppressed_count == 1

    def test_the_signals_read_the_stored_analysis(self) -> None:
        subject, analyses, _dispositions = wired()
        analyses.put(stored())
        consequence_id = analyses.require(TENANT, ANALYSIS).findings[0].consequence_id

        subject.record_reviewed(
            tenant_id=TENANT,
            analysis_id=ANALYSIS,
            consequence_id=consequence_id,
            disposition=Disposition.ALREADY_KNOWN,
            disposed_by=REVIEWER,
            disposed_at=AT,
        )

        measured = subject.signals_reviewed(tenant_id=TENANT, analysis_id=ANALYSIS)
        assert measured.reviewed == 1
        assert measured.already_known_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Suppression, read by a later analysis
# ---------------------------------------------------------------------------


class TestSuppressionAcrossAnalyses:
    def _reject(self) -> tuple[RecordDispositionUseCase, InMemoryAnalysisStore]:
        subject, analyses, _dispositions = wired()
        analyses.put(stored())
        consequence_id = analyses.require(TENANT, ANALYSIS).findings[0].consequence_id
        subject.record_reviewed(
            tenant_id=TENANT,
            analysis_id=ANALYSIS,
            consequence_id=consequence_id,
            disposition=Disposition.WRONG,
            disposed_by=REVIEWER,
            disposed_at=AT,
            note=WHY,
            suppress_until_source_revision=True,
        )
        return subject, analyses

    def test_a_standing_suppression_is_readable(self) -> None:
        """Requirement 7: readable by the next analysis flow, with no rule redefined."""
        subject, _analyses = self._reject()

        found = subject.suppressions(tenant_id=TENANT)

        assert len(found) == 1
        assert found[0].scope == (KEY, VERSION)
        assert found[0].disposition is Disposition.WRONG
        assert found[0].disposed_by == REVIEWER
        assert found[0].disposed_at == AT

    def test_a_suppression_survives_a_new_analysis_of_the_same_version(self) -> None:
        """The failure this closes: re-delivering a finding a reviewer already rejected.

        A second analysis over an unrevised document asks the same question and gets the same
        answer, so the reviewer is not asked twice.
        """
        subject, _analyses = self._reject()

        assert subject.is_suppressed(
            tenant_id=TENANT, assumption_key=KEY, artifact_version_id=VERSION
        )

    def test_a_revised_source_version_lifts_the_suppression(self) -> None:
        """The reviewer said the finding was wrong about *that* text.

        A revised document is different text, and the version id is content-derived - so the
        suppression expires by itself rather than needing anybody to remember to clear it.
        """
        subject, _analyses = self._reject()

        assert not subject.is_suppressed(
            tenant_id=TENANT, assumption_key=KEY, artifact_version_id=REVISED
        )

    def test_a_suppression_does_not_cover_a_different_key(self) -> None:
        subject, _analyses = self._reject()

        assert not subject.is_suppressed(
            tenant_id=TENANT,
            assumption_key="refund.approval_threshold",
            artifact_version_id=VERSION,
        )

    def test_a_suppression_does_not_cross_tenants(self) -> None:
        subject, _analyses = self._reject()

        assert subject.suppressions(tenant_id=OTHER_TENANT) == ()
        assert not subject.is_suppressed(
            tenant_id=OTHER_TENANT, assumption_key=KEY, artifact_version_id=VERSION
        )

    def test_a_verdict_that_did_not_ask_to_suppress_is_not_standing(self) -> None:
        """The reviewer said it was wrong for this run, not that they never want it again."""
        subject, analyses, _dispositions = wired()
        analyses.put(stored())
        consequence_id = analyses.require(TENANT, ANALYSIS).findings[0].consequence_id
        subject.record_reviewed(
            tenant_id=TENANT,
            analysis_id=ANALYSIS,
            consequence_id=consequence_id,
            disposition=Disposition.WRONG,
            disposed_by=REVIEWER,
            disposed_at=AT,
            note=WHY,
        )

        assert subject.suppressions(tenant_id=TENANT) == ()
        assert not subject.is_suppressed(
            tenant_id=TENANT, assumption_key=KEY, artifact_version_id=VERSION
        )

    def test_reversing_the_verdict_lifts_the_standing_suppression(self) -> None:
        """Reversibility reaches across analyses too."""
        subject, analyses = self._reject()
        stored_analysis = analyses.require(TENANT, ANALYSIS)
        consequence_id = stored_analysis.findings[0].consequence_id

        subject.change(
            analysis_id=ANALYSIS,
            consequences=stored_analysis.consequences,
            consequence_id=consequence_id,
            disposition=Disposition.USEFUL,
            changed_by="privacy-office/reviewer-2",
            changed_at=LATER,
            reason="The earlier verdict was mine and it was wrong.",
        )

        assert subject.suppressions(tenant_id=TENANT) == ()
        assert not subject.is_suppressed(
            tenant_id=TENANT, assumption_key=KEY, artifact_version_id=VERSION
        )

    def test_a_class_b_finding_is_never_silently_suppressed(self) -> None:
        """A hypothesis has no single source version, so no document revision refutes it.

        Passing ``None`` matches nothing rather than matching everything, which is the safe
        direction: the alternative would let a verdict about a document silence a behavioural
        claim nobody reviewed.
        """
        subject, _analyses = self._reject()

        assert not subject.is_suppressed(
            tenant_id=TENANT, assumption_key=KEY, artifact_version_id=None
        )

    def test_reading_suppressions_without_an_analysis_store_is_refused(self) -> None:
        """Which artifact a verdict was about is a property of the analysis it names."""
        subject = RecordDispositionUseCase(store=InMemoryDispositionStore())

        with pytest.raises(UnknownAnalysis, match="needs an analysis store"):
            subject.suppressions(tenant_id=TENANT)

    def test_the_expiry_rule_has_one_implementation(self) -> None:
        """The lookup delegates. It does not restate when a suppression expires.

        Parsed rather than grepped: a substring search would match the docstrings explaining
        the rule, which is a test that reads correctly and asserts nothing.
        """
        source = pathlib.Path("app/application/use_cases/record_disposition.py").read_text(
            encoding="utf-8"
        )
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "suppression_has_expired" in called


# ---------------------------------------------------------------------------
# The snapshot projection
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_the_snapshot_takes_only_review_relevant_state(self) -> None:
        """Not a serialized ``AnalysisResult``.

        Storing the live working objects would make the store a second copy of the pipeline's
        intermediate state, able to disagree with it. The snapshot's fields are the review
        surface and nothing else.
        """
        fields = set(StoredAnalysis.__dataclass_fields__)

        assert fields == {
            "analysis_id",
            "tenant_id",
            "run_id",
            "change",
            "run_provenance",
            "stages",
            "findings",
            "graph",
            "evidence",
            # The precedent a raised Class B status rested on. Review-relevant by the same
            # test as the rest: a reviewer asked why a finding is not `speculative` has to be
            # able to read the analogue without re-running the analysis.
            "historical_events",
            "evidence_gap_ids",
            "open_questions",
            "fingerprint",
            "stored_at",
            "pipeline_version",
            "snapshot_version",
        }
        assert not {"coupling", "retrieval", "class_b", "contradictions", "challenge"} & fields

    def test_the_snapshot_signature_derives_the_tenant(self) -> None:
        """From the retrieval query - the boundary the analysis actually ran under."""
        parameters = set(inspect.signature(snapshot).parameters)
        assert parameters == {"result", "stored_at", "tenant_id"}

    def test_nothing_in_the_store_constructs_a_consequence(self) -> None:
        """The snapshot holds findings by reference and never rebuilds one."""
        tree = ast.parse(
            pathlib.Path("app/application/orchestration/analysis_store.py").read_text(
                encoding="utf-8"
            )
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert (
            not {
                "Consequence",
                "ArbitrationDecision",
                "ConsequenceScore",
                "ConsequenceGraph",
            }
            & called
        )
        assert "model_copy" not in attributes

    def test_a_reviewed_finding_bundles_the_three_records(self) -> None:
        """Answering "what did the reviewer see for this consequence?" without zipping lists."""
        subject = consequence()
        finding = ReviewedFinding(
            consequence=subject,
            decision=decision(subject.consequence_id),
            score=score(subject.consequence_id),
        )

        assert finding.consequence_id == subject.consequence_id
        assert finding.confidence == 0.42
        assert finding.artifact_version_id is None
        assert not finding.in_default_view

    def test_a_finding_with_no_score_reports_no_confidence(self) -> None:
        finding = ReviewedFinding(consequence=consequence())
        assert finding.confidence is None
