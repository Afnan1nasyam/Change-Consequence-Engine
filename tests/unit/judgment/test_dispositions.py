"""Reviewer verdicts: reversible, auditable, and unable to touch the finding they judge.

``IDEA.md`` 21.2 makes human disposition the **primary** feedback signal, and the reason is
worth restating because it shapes every test here: outcome matching cannot be primary, since
attribution is ambiguous, misses are unobservable, and a prevented consequence is
counterfactually invisible - so a system that works looks like one producing false positives.

That makes two properties load-bearing:

* **suppression must not delete.** A reviewer marking a finding wrong hides it from the
  foreground; the consequence, its evidence, its score and its graph edges are untouched, so a
  reviewer who was wrong loses nothing by saying so.
* **``already_known`` must not suppress.** It is the signal the product thesis rests on - a
  high rate means the system is finding what a diligent reviewer would find anyway - and
  suppressing it would destroy the measurement *and* hide a correct finding.

The tests carrying the most weight:

* :func:`TestSuppression.test_a_suppressing_verdict_changes_nothing_about_the_consequence`
* :func:`TestSuppression.test_suppression_is_reversible`
* :func:`TestAlreadyKnown.test_already_known_does_not_suppress`
* :func:`TestRejections.test_a_consequence_from_another_analysis_is_refused_distinctly`
* :func:`TestIdempotency.test_resubmitting_the_same_verdict_records_no_transition`
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
import pathlib
from typing import Any

import pytest
from pydantic import ValidationError

from app.application.use_cases.record_disposition import RecordDispositionUseCase
from app.domain.entities.findings import Consequence, FindingDisposition
from app.domain.value_objects.enums import (
    SUPPRESSING_DISPOSITIONS,
    ConsequenceClass,
    Disposition,
    Origin,
    Severity,
    Status,
)
from app.domain.value_objects.identifiers import id_kind, new_id
from app.domain.value_objects.provenance import RunProvenance
from app.infrastructure.database.repositories.memory import InMemoryDispositionStore
from app.judgment.dispositions import (
    FOREGROUND_SUPPRESSING,
    REASON_REQUIRED,
    CrossAnalysisConsequence,
    DispositionRecord,
    UnknownAnalysis,
    UnknownConsequence,
    foreground,
    requires_reason,
    signals,
    suppresses_foreground,
    suppression_has_expired,
)

ANALYSIS = new_id("analysis")
OTHER_ANALYSIS = new_id("analysis")
REVIEWER = "privacy-office/reviewer-1"
AT = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)
LATER = dt.datetime(2026, 9, 11, 9, 0, tzinfo=dt.UTC)
EVIDENCE = "ev_" + "a" * 32
KEY = "retention.window"
VERSION = "sv_" + "1" * 32
REVISED_VERSION = "sv_" + "2" * 32


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def consequence(**over: Any) -> Consequence:
    payload: dict[str, Any] = {
        "consequence_id": new_id("consequence"),
        "change_id": new_id("change"),
        "title": "The training deck teaches the old retention window",
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


def use_case() -> tuple[RecordDispositionUseCase, InMemoryDispositionStore]:
    store = InMemoryDispositionStore()
    return RecordDispositionUseCase(store=store), store


def record(
    subject: RecordDispositionUseCase,
    consequences: tuple[Consequence, ...],
    *,
    disposition: Disposition = Disposition.USEFUL,
    consequence_id: str | None = None,
    analysis_id: str = ANALYSIS,
    note: str | None = None,
    at: dt.datetime = AT,
    by: str = REVIEWER,
    suppress: bool = False,
    artifact_ref: str | None = None,
) -> Any:
    return subject.record(
        analysis_id=analysis_id,
        consequences=consequences,
        consequence_id=consequence_id or consequences[0].consequence_id,
        disposition=disposition,
        disposed_by=by,
        disposed_at=at,
        note=note if note is not None else _default_note(disposition),
        suppress_until_source_revision=suppress,
        artifact_ref=artifact_ref,
    )


def _default_note(disposition: Disposition) -> str | None:
    """A reason for the two verdicts that require one, and nothing for the rest."""
    if disposition in REASON_REQUIRED:
        return "The deck was reissued before the change; the mechanism does not hold."
    return None


# ---------------------------------------------------------------------------
# Creating a verdict
# ---------------------------------------------------------------------------


class TestCreate:
    def test_a_verdict_is_recorded_against_the_pair(self) -> None:
        subject, store = use_case()
        finding = consequence()

        outcome = record(subject, (finding,))

        assert outcome.created
        assert not outcome.changed
        assert outcome.record.key == (ANALYSIS, finding.consequence_id)
        assert store.get(ANALYSIS, finding.consequence_id) is outcome.record

    def test_the_verdict_carries_the_reviewer_and_the_time(self) -> None:
        subject, _store = use_case()
        outcome = record(subject, (consequence(),))

        current = outcome.record.current
        assert current.disposed_by == REVIEWER
        assert current.disposed_at == AT
        assert id_kind(current.disposition_id) == "disposition"

    def test_the_verdict_inherits_the_findings_key_and_artifact(self) -> None:
        """So a future-analysis suppression has something to be scoped by."""
        subject, _store = use_case()
        finding = consequence()

        current = record(subject, (finding,)).record.current

        assert current.assumption_key == KEY
        assert current.artifact_ref == finding.affected_entity

    @pytest.mark.parametrize("disposition", sorted(Disposition))
    def test_every_frozen_value_can_be_recorded(self, disposition: Disposition) -> None:
        """Five values, and no sixth. A value nothing can record is dead contract."""
        subject, _store = use_case()
        outcome = record(subject, (consequence(),), disposition=disposition)

        assert outcome.record.disposition is disposition

    def test_the_five_values_are_exactly_the_frozen_set(self) -> None:
        assert {d.value for d in Disposition} == {
            "useful",
            "already_known",
            "not_material",
            "wrong",
            "unverifiable",
        }

    def test_two_analyses_may_hold_different_verdicts_on_one_finding(self) -> None:
        """Identity is the pair. A reviewer judges what they were shown.

        The same consequence reviewed in two runs is two verdicts: the evidence, the status
        and the confidence the reviewer saw were different, and carrying one forward would
        attribute an opinion to a report they never read.
        """
        subject, store = use_case()
        finding = consequence()

        record(subject, (finding,), analysis_id=ANALYSIS, disposition=Disposition.USEFUL)
        record(
            subject,
            (finding,),
            analysis_id=OTHER_ANALYSIS,
            disposition=Disposition.ALREADY_KNOWN,
        )

        mine = store.get(ANALYSIS, finding.consequence_id)
        theirs = store.get(OTHER_ANALYSIS, finding.consequence_id)
        assert mine is not None and theirs is not None
        assert mine.disposition is Disposition.USEFUL
        assert theirs.disposition is Disposition.ALREADY_KNOWN


# ---------------------------------------------------------------------------
# Changing one
# ---------------------------------------------------------------------------


class TestChange:
    def test_a_different_verdict_supersedes_and_records_the_transition(self) -> None:
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.WRONG)

        outcome = subject.change(
            analysis_id=ANALYSIS,
            consequences=(finding,),
            consequence_id=finding.consequence_id,
            disposition=Disposition.USEFUL,
            changed_by="privacy-office/reviewer-2",
            changed_at=LATER,
            reason="Re-read the deck; the earlier verdict was mine and it was wrong.",
        )

        assert outcome.changed
        assert not outcome.created
        assert outcome.record.disposition is Disposition.USEFUL
        assert outcome.record.original is Disposition.WRONG

    def test_the_transition_records_both_values_the_reviewer_and_the_reason(self) -> None:
        """ "Changed to useful" is not auditable without knowing what it was."""
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.WRONG)

        outcome = subject.change(
            analysis_id=ANALYSIS,
            consequences=(finding,),
            consequence_id=finding.consequence_id,
            disposition=Disposition.USEFUL,
            changed_by="privacy-office/reviewer-2",
            changed_at=LATER,
            reason="Corrected after re-reading the source.",
        )

        transition = outcome.record.history[-1]
        assert transition.from_disposition is Disposition.WRONG
        assert transition.to_disposition is Disposition.USEFUL
        assert transition.changed_at == LATER
        assert transition.changed_by == "privacy-office/reviewer-2"
        assert transition.reason == "Corrected after re-reading the source."

    def test_the_superseded_verdict_survives_as_it_was_recorded(self) -> None:
        """The withdrawn reviewer, timestamp and note all outlive the withdrawal."""
        subject, _store = use_case()
        finding = consequence()
        first = record(
            subject, (finding,), disposition=Disposition.WRONG, by="reviewer-a"
        ).record.current

        outcome = subject.change(
            analysis_id=ANALYSIS,
            consequences=(finding,),
            consequence_id=finding.consequence_id,
            disposition=Disposition.USEFUL,
            changed_by="reviewer-b",
            changed_at=LATER,
            reason="Corrected.",
        )

        assert outcome.record.superseded == (first,)
        assert outcome.record.superseded[0].disposed_by == "reviewer-a"
        assert outcome.record.superseded[0].note

    def test_a_change_requires_a_reason_whatever_the_value(self) -> None:
        """Stricter than recording, deliberately.

        A first verdict on ``useful`` needs no defence. A *change* is the operation a
        reviewer performs when they got it wrong, and a transition with no explanation is
        the one thing an audit trail cannot use.
        """
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.USEFUL)

        with pytest.raises(UnknownConsequence, match="requires a reason"):
            subject.change(
                analysis_id=ANALYSIS,
                consequences=(finding,),
                consequence_id=finding.consequence_id,
                disposition=Disposition.ALREADY_KNOWN,
                changed_by=REVIEWER,
                changed_at=LATER,
                reason="   ",
            )

    def test_changing_a_verdict_that_was_never_recorded_is_refused(self) -> None:
        subject, _store = use_case()
        finding = consequence()

        with pytest.raises(UnknownConsequence, match="no verdict on record"):
            subject.change(
                analysis_id=ANALYSIS,
                consequences=(finding,),
                consequence_id=finding.consequence_id,
                disposition=Disposition.USEFUL,
                changed_by=REVIEWER,
                changed_at=LATER,
                reason="Correcting something that does not exist.",
            )

    def test_several_changes_accumulate_in_order(self) -> None:
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.USEFUL)

        for target in (Disposition.WRONG, Disposition.UNVERIFIABLE, Disposition.USEFUL):
            outcome = subject.change(
                analysis_id=ANALYSIS,
                consequences=(finding,),
                consequence_id=finding.consequence_id,
                disposition=target,
                changed_by=REVIEWER,
                changed_at=LATER,
                reason=f"Moving to {target.value}.",
            )

        assert [t.to_disposition for t in outcome.record.history] == [
            Disposition.WRONG,
            Disposition.UNVERIFIABLE,
            Disposition.USEFUL,
        ]
        assert outcome.record.original is Disposition.USEFUL
        assert len(outcome.record.superseded) == 3


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------


class TestSuppression:
    @pytest.mark.parametrize("disposition", sorted(SUPPRESSING_DISPOSITIONS))
    def test_wrong_and_not_material_suppress_the_foreground(self, disposition: Disposition) -> None:
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=disposition)

        view = subject.foreground(analysis_id=ANALYSIS, consequences=(finding,))

        assert view.shown == ()
        assert view.suppressed == (finding,)
        assert view.suppressed_by[finding.consequence_id] is disposition
        assert view.suppressed_count == 1

    @pytest.mark.parametrize(
        "disposition",
        [Disposition.USEFUL, Disposition.ALREADY_KNOWN, Disposition.UNVERIFIABLE],
    )
    def test_the_other_three_leave_the_finding_shown(self, disposition: Disposition) -> None:
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=disposition)

        view = subject.foreground(analysis_id=ANALYSIS, consequences=(finding,))

        assert view.shown == (finding,)
        assert view.suppressed == ()

    def test_a_suppressing_verdict_changes_nothing_about_the_consequence(self) -> None:
        """Requirement 6, and the property that makes the signal safe to give.

        Serialized before and after: the consequence, its evidence, its provenance and its
        status are all identical. Suppression is a view decision computed on read.
        """
        subject, _store = use_case()
        finding = consequence()
        before = finding.model_dump_json()

        record(subject, (finding,), disposition=Disposition.WRONG)
        view = subject.foreground(analysis_id=ANALYSIS, consequences=(finding,))

        assert finding.model_dump_json() == before
        assert view.suppressed[0] is finding
        assert view.suppressed[0].evidence_ids == (EVIDENCE,)
        assert view.suppressed[0].status is Status.LIKELY

    def test_suppression_is_reversible(self) -> None:
        """A reviewer who was wrong loses nothing by saying so."""
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.WRONG)

        assert subject.foreground(analysis_id=ANALYSIS, consequences=(finding,)).shown == ()

        outcome = subject.change(
            analysis_id=ANALYSIS,
            consequences=(finding,),
            consequence_id=finding.consequence_id,
            disposition=Disposition.USEFUL,
            changed_by=REVIEWER,
            changed_at=LATER,
            reason="The earlier verdict was mine and it was wrong.",
        )

        restored = subject.foreground(analysis_id=ANALYSIS, consequences=(finding,))
        assert restored.shown == (finding,)
        assert restored.suppressed == ()
        assert outcome.record.history[-1].restored

    def test_the_store_offers_no_way_to_delete_a_verdict(self) -> None:
        """A verdict is reversed by recording another, which leaves a trail.

        Removing one would erase the reviewer's judgement and the audit of it together, so
        the port has no ``delete`` and the in-memory implementation has none either.
        """
        assert not hasattr(InMemoryDispositionStore, "delete")
        assert not hasattr(InMemoryDispositionStore, "remove")
        methods = {
            name for name, _ in inspect.getmembers(InMemoryDispositionStore, inspect.isfunction)
        }
        assert methods == {"__init__", "put", "get", "for_analysis", "find_consequence"}

    def test_suppression_needs_a_reason(self) -> None:
        """The direction that subtracts is the one that must be justified."""
        subject, _store = use_case()

        with pytest.raises(UnknownConsequence, match="requires a note"):
            record(subject, (consequence(),), disposition=Disposition.WRONG, note="")

    def test_the_two_reason_requiring_values_are_the_two_suppressing_ones(self) -> None:
        assert REASON_REQUIRED == SUPPRESSING_DISPOSITIONS
        assert FOREGROUND_SUPPRESSING == SUPPRESSING_DISPOSITIONS
        assert Disposition.UNVERIFIABLE not in REASON_REQUIRED

    def test_only_a_foregrounded_finding_counts_as_suppressed(self) -> None:
        """A finding the judge never foregrounded was not hidden by a reviewer."""
        shown, hidden = consequence(), consequence()
        subject, _store = use_case()
        record(
            subject,
            (shown, hidden),
            consequence_id=hidden.consequence_id,
            disposition=Disposition.WRONG,
        )

        view = subject.foreground(
            analysis_id=ANALYSIS,
            consequences=(shown, hidden),
            foregrounded=(shown.consequence_id,),
        )

        assert view.shown == (shown,)
        assert view.suppressed == ()


# ---------------------------------------------------------------------------
# already_known
# ---------------------------------------------------------------------------


class TestAlreadyKnown:
    def test_already_known_does_not_suppress(self) -> None:
        """The signal the product thesis rests on, and it must stay visible.

        Suppressing it would destroy the measurement *and* hide a correct finding: the
        reviewer already knew it, which is a statement about the reviewer rather than about
        the finding.
        """
        assert not suppresses_foreground(Disposition.ALREADY_KNOWN)
        assert Disposition.ALREADY_KNOWN not in SUPPRESSING_DISPOSITIONS

        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.ALREADY_KNOWN)

        view = subject.foreground(analysis_id=ANALYSIS, consequences=(finding,))
        assert view.shown == (finding,)

    def test_already_known_cannot_request_persistent_suppression(self) -> None:
        """The frozen entity refuses it, so this is not a rule this layer could relax."""
        with pytest.raises(ValidationError, match="may suppress"):
            FindingDisposition(
                disposition_id=new_id("disposition"),
                consequence_id=new_id("consequence"),
                analysis_id=ANALYSIS,
                assumption_key=KEY,
                disposition=Disposition.ALREADY_KNOWN,
                disposed_by=REVIEWER,
                disposed_at=AT,
                suppresses_until_source_revision=True,
            )

    def test_the_already_known_rate_is_reported(self) -> None:
        """A system with excellent precision and a 90% already-known rate is not working."""
        findings = tuple(consequence() for _ in range(4))
        subject, _store = use_case()
        for finding, disposition in zip(
            findings,
            (
                Disposition.ALREADY_KNOWN,
                Disposition.ALREADY_KNOWN,
                Disposition.ALREADY_KNOWN,
                Disposition.USEFUL,
            ),
            strict=True,
        ):
            record(
                subject, findings, consequence_id=finding.consequence_id, disposition=disposition
            )

        measured = subject.signals(analysis_id=ANALYSIS, consequences=findings)

        assert measured.reviewed == 4
        assert measured.already_known_rate == pytest.approx(0.75)
        assert measured.useful_rate == pytest.approx(0.25)
        assert measured.coverage == pytest.approx(1.0)

    def test_an_unreviewed_analysis_reports_no_rate_rather_than_zero(self) -> None:
        """Otherwise an analysis nobody looked at reads as a 0% already-known success."""
        subject, _store = use_case()
        measured = subject.signals(analysis_id=ANALYSIS, consequences=(consequence(),))

        assert measured.reviewed == 0
        assert measured.unreviewed == 1
        assert measured.already_known_rate is None
        assert measured.coverage == pytest.approx(0.0)

    def test_a_corrected_verdict_moves_the_rate_rather_than_adding_to_both(self) -> None:
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.ALREADY_KNOWN)
        subject.change(
            analysis_id=ANALYSIS,
            consequences=(finding,),
            consequence_id=finding.consequence_id,
            disposition=Disposition.USEFUL,
            changed_by=REVIEWER,
            changed_at=LATER,
            reason="On reflection the reviewer did not already have it.",
        )

        measured = subject.signals(analysis_id=ANALYSIS, consequences=(finding,))

        assert measured.reviewed == 1
        assert measured.already_known_rate == pytest.approx(0.0)
        assert measured.useful_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Persistent suppression, and its expiry
# ---------------------------------------------------------------------------


class TestPersistentSuppression:
    def test_a_suppression_is_scoped_by_key_and_artifact(self) -> None:
        """Suppression must name what it suppresses, or it silences too much."""
        subject, _store = use_case()
        finding = consequence()

        outcome = record(
            subject,
            (finding,),
            disposition=Disposition.WRONG,
            suppress=True,
            artifact_ref=VERSION,
        )

        assert outcome.record.suppression_scope == (KEY, VERSION)
        assert outcome.record.current.suppresses_until_source_revision

    def test_an_unsuppressed_verdict_has_no_scope(self) -> None:
        subject, _store = use_case()
        outcome = record(subject, (consequence(),), disposition=Disposition.WRONG)

        assert outcome.record.suppression_scope is None

    def test_a_revised_source_expires_the_suppression(self) -> None:
        """The reviewer said the finding is wrong about *that* text, not about the document.

        ``artifact_ref`` is the source version id, which is content-derived - so a revised
        document has a different one and the suppression lifts by itself.
        """
        subject, _store = use_case()
        finding = consequence()
        outcome = record(
            subject,
            (finding,),
            disposition=Disposition.WRONG,
            suppress=True,
            artifact_ref=VERSION,
        )

        assert not suppression_has_expired(outcome.record, artifact_ref=VERSION)
        assert suppression_has_expired(outcome.record, artifact_ref=REVISED_VERSION)

    def test_a_record_that_never_suppressed_has_nothing_to_expire(self) -> None:
        """``True`` here would read as "the suppression lifted", which never happened."""
        subject, _store = use_case()
        outcome = record(subject, (consequence(),), disposition=Disposition.USEFUL)

        assert not suppression_has_expired(outcome.record, artifact_ref=REVISED_VERSION)

    def test_reversing_a_suppressing_verdict_drops_the_persistent_scope(self) -> None:
        subject, _store = use_case()
        finding = consequence()
        record(
            subject,
            (finding,),
            disposition=Disposition.WRONG,
            suppress=True,
            artifact_ref=VERSION,
        )

        outcome = subject.change(
            analysis_id=ANALYSIS,
            consequences=(finding,),
            consequence_id=finding.consequence_id,
            disposition=Disposition.USEFUL,
            changed_by=REVIEWER,
            changed_at=LATER,
            reason="Withdrawn.",
        )

        assert outcome.record.suppression_scope is None
        assert not outcome.record.current.suppresses_until_source_revision


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


class TestRejections:
    def test_an_unknown_analysis_is_refused(self) -> None:
        """An analysis with no consequences has nothing to review, so the verdict has no
        subject - and reporting "unknown consequence" would send the caller looking in the
        wrong place."""
        subject, _store = use_case()

        with pytest.raises(UnknownAnalysis, match="nothing to review"):
            subject.record(
                analysis_id=ANALYSIS,
                consequences=(),
                consequence_id=new_id("consequence"),
                disposition=Disposition.USEFUL,
                disposed_by=REVIEWER,
                disposed_at=AT,
            )

    def test_an_unknown_consequence_is_refused(self) -> None:
        subject, _store = use_case()

        with pytest.raises(UnknownConsequence, match="contains no consequence"):
            record(subject, (consequence(),), consequence_id=new_id("consequence"))

    def test_a_consequence_from_another_analysis_is_refused_distinctly(self) -> None:
        """The dangerous one: without the store's index this looks like a typo.

        A verdict landing on the right finding in the wrong run would be attributed,
        auditable, and about a report the reviewer never saw.
        """
        subject, _store = use_case()
        theirs = consequence()
        mine = consequence()
        record(subject, (theirs,), analysis_id=OTHER_ANALYSIS)

        with pytest.raises(CrossAnalysisConsequence, match="belongs to analysis"):
            record(
                subject,
                (mine,),
                analysis_id=ANALYSIS,
                consequence_id=theirs.consequence_id,
            )

    def test_the_three_rejections_are_distinct_types(self) -> None:
        """Collapsing them would hide the one that indicates a wiring defect."""
        assert issubclass(UnknownAnalysis, Exception)
        assert not issubclass(UnknownAnalysis, CrossAnalysisConsequence)
        assert not issubclass(UnknownConsequence, CrossAnalysisConsequence)
        assert not issubclass(CrossAnalysisConsequence, UnknownConsequence)

    def test_a_verdict_is_validated_against_the_analysis_it_was_given(self) -> None:
        """Not against every analysis that ever ran. The reviewer judged what they saw."""
        subject, _store = use_case()
        shown = consequence()
        unshown = consequence()

        record(subject, (shown,))
        with pytest.raises(UnknownConsequence):
            record(subject, (shown,), consequence_id=unshown.consequence_id)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_resubmitting_the_same_verdict_records_no_transition(self) -> None:
        """A double-click must not fabricate a transition in the audit trail."""
        subject, _store = use_case()
        finding = consequence()

        first = record(subject, (finding,), disposition=Disposition.USEFUL)
        second = record(subject, (finding,), disposition=Disposition.USEFUL)

        assert first.created
        assert second.unchanged
        assert not second.changed
        assert second.record.history == ()
        assert second.record.superseded == ()

    def test_a_retry_at_a_later_clock_reading_is_still_a_no_op(self) -> None:
        """The identity and the timestamp are not what a reviewer decided."""
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.USEFUL, at=AT)

        second = record(subject, (finding,), disposition=Disposition.USEFUL, at=LATER)

        assert second.unchanged
        assert second.record.current.disposed_at == AT

    def test_a_changed_note_is_a_change_not_a_duplicate(self) -> None:
        """The note is part of what the reviewer decided."""
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.WRONG, note="First reason.")

        second = record(subject, (finding,), disposition=Disposition.WRONG, note="A better reason.")

        assert second.changed
        assert second.record.history[-1].from_disposition is Disposition.WRONG
        assert second.record.history[-1].to_disposition is Disposition.WRONG

    def test_a_different_verdict_through_record_is_still_audited(self) -> None:
        """``record`` is not a back door around the trail."""
        subject, _store = use_case()
        finding = consequence()
        record(subject, (finding,), disposition=Disposition.USEFUL)

        second = record(subject, (finding,), disposition=Disposition.WRONG)

        assert second.changed
        assert second.record.history[-1].suppressed
        assert second.record.superseded


# ---------------------------------------------------------------------------
# Shape and isolation
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_no_module_here_constructs_a_consequence_or_a_score(self) -> None:
        """Requirement 6, as a property of the code rather than of the tests.

        Parsed rather than grepped: a substring search would match the docstrings explaining
        the rule, which is a test that reads correctly and asserts nothing.
        """
        for path in (
            "app/judgment/dispositions.py",
            "app/application/use_cases/record_disposition.py",
        ):
            tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
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
                    "ConsequenceScore",
                    "ArbitrationDecision",
                    "GraphEdge",
                    "GraphNode",
                    "ConsequenceGraph",
                }
                & called
            ), path
            assert "model_copy" not in attributes, path

    def test_the_rules_module_reads_the_frozen_set_rather_than_restating_it(self) -> None:
        """So a change to `SUPPRESSING_DISPOSITIONS` cannot leave this layer behind."""
        source = pathlib.Path("app/judgment/dispositions.py").read_text(encoding="utf-8")
        assert "FOREGROUND_SUPPRESSING: Final = SUPPRESSING_DISPOSITIONS" in source

    def test_suppression_reads_only_the_verdict(self) -> None:
        """No confidence threshold, no severity, no age. Only what the reviewer said."""
        parameters = set(inspect.signature(suppresses_foreground).parameters)
        assert parameters == {"disposition"}
        assert set(inspect.signature(requires_reason).parameters) == {"disposition"}

    def test_the_use_case_holds_only_its_two_stores(self) -> None:
        """The disposition store, and optionally the analysis store it validates against."""
        parameters = set(inspect.signature(RecordDispositionUseCase.__init__).parameters)
        assert parameters == {"self", "store", "analyses"}

    def test_the_foreground_helper_returns_the_consequences_it_was_given(self) -> None:
        """Identity, not equality: nothing was copied on the way through."""
        first, second = consequence(), consequence()
        view = foreground((first, second), records=())

        assert view.shown[0] is first
        assert view.shown[1] is second

    def test_signals_count_the_current_verdict_only(self) -> None:
        findings = (consequence(),)
        history = DispositionRecord(
            current=FindingDisposition(
                disposition_id=new_id("disposition"),
                consequence_id=findings[0].consequence_id,
                analysis_id=ANALYSIS,
                disposition=Disposition.USEFUL,
                disposed_by=REVIEWER,
                disposed_at=AT,
            )
        )
        measured = signals(findings, records=(history,))

        assert measured.counts[Disposition.USEFUL] == 1
        assert sum(measured.counts.values()) == 1
