"""The four analogue admission rules, unit-level.

The integration suite proves an admitted anchor raises a Class B status end to end. This file
proves the gate itself, because the interesting cases are the refusals and each one is a
one-line branch an end-to-end test can only reach by luck.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from app.ingestion.identity import event_id_for
from app.specialists.analogues import (
    ANALOGUE_AGENT,
    REFUSED_NOT_IN_PAST,
    REFUSED_UNCITED,
    REFUSED_WRONG_AGENT,
    AnalogueOutcome,
    admit,
)
from app.specialists.proposals import AnalogueProposal

AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
TENANT = "tn_" + "1" * 32

#: Two spans the analyst's slice really held, and one it did not.
FIRST = "ev_" + "a" * 32
SECOND = "ev_" + "b" * 32
OFFERED = frozenset({FIRST, SECOND})
OUTSIDE = "ev_" + "c" * 32


def proposal(**over: Any) -> AnalogueProposal:
    payload: dict[str, Any] = {
        "title": "Refund threshold reduction caused an approval backlog",
        "occurred_at": dt.datetime(2024, 6, 3, tzinfo=dt.UTC),
        "observed_outcome": "Queue depth tripled for three weeks; turnaround 4h to 19h.",
        "cited_evidence_ids": (FIRST, SECOND),
        "differs_how": "That change halved the threshold; this one halves it again.",
    }
    payload.update(over)
    return AnalogueProposal(**payload)


def admitting(*proposals: AnalogueProposal, agent: str = ANALOGUE_AGENT) -> AnalogueOutcome:
    return admit(
        proposals,
        agent=agent,
        tenant_id=TENANT,
        offered_evidence_ids=OFFERED,
        as_of=AS_OF,
    )


class TestTheIdIsDerivedNeverProposed:
    def test_the_schema_has_no_id_field_to_invent_one_in(self) -> None:
        """The safeguard is structural: there is nowhere to put an ``he_...``."""
        assert "event_id" not in AnalogueProposal.model_fields
        assert not [f for f in AnalogueProposal.model_fields if f.endswith("_id")]

    def test_the_id_comes_from_the_spans_and_the_tenant(self) -> None:
        (event,) = admitting(proposal()).events

        assert event.event_id == event_id_for(tenant_id=TENANT, evidence_ids=[FIRST, SECOND])
        assert event.tenant_id == TENANT
        assert set(event.evidence_ids) == OFFERED

    def test_the_title_does_not_participate_in_identity(self) -> None:
        """Two readings of one incident are one precedent, so the model cannot split it in
        two by renaming it - nor corroborate itself by describing it twice."""
        first = admitting(proposal())
        renamed = admitting(proposal(title="Another description of the same incident"))

        assert first.events[0].event_id == renamed.events[0].event_id

    def test_the_same_spans_twice_yield_one_event(self) -> None:
        assert len(admitting(proposal(), proposal(title="Restated")).events) == 1

    def test_a_second_tenant_cannot_collide_onto_the_same_precedent(self) -> None:
        mine = admitting(proposal())
        theirs = admit(
            (proposal(),),
            agent=ANALOGUE_AGENT,
            tenant_id="tn_" + "2" * 32,
            offered_evidence_ids=OFFERED,
            as_of=AS_OF,
        )

        assert mine.events[0].event_id != theirs.events[0].event_id

    def test_dropping_a_span_changes_the_id(self) -> None:
        """An id derived from spans is only valid for the spans that produced it, so an
        anchor cannot be carried over to a differently evidenced claim."""
        both = admitting(proposal())
        one = admitting(proposal(cited_evidence_ids=(FIRST,)))

        assert both.events[0].event_id != one.events[0].event_id


class TestCitationsAreCheckedAgainstTheSlice:
    def test_a_span_outside_the_slice_is_dropped_and_reported(self) -> None:
        outcome = admitting(proposal(cited_evidence_ids=(FIRST, SECOND, OUTSIDE)))

        (event,) = outcome.events
        assert set(event.evidence_ids) == OFFERED
        assert outcome.fabricated_evidence_ids == (OUTSIDE,)

    def test_an_event_resting_only_on_outside_spans_is_refused(self) -> None:
        """Not recorded-but-unqualified: with no admitted span there is nothing to be
        traceable to, and an untraceable event record is worse than none."""
        outcome = admitting(proposal(cited_evidence_ids=(OUTSIDE,)))

        assert outcome.events == ()
        assert outcome.refused == (
            ("Refund threshold reduction caused an approval backlog", REFUSED_UNCITED),
        )
        assert outcome.anchor_ids == frozenset()

    def test_an_uncited_event_is_refused(self) -> None:
        outcome = admitting(proposal(cited_evidence_ids=()))

        assert outcome.events == ()
        assert outcome.refused[0][1] == REFUSED_UNCITED


class TestOnlyARecordedOutcomeAnchors:
    def test_an_outcome_less_precedent_is_kept_and_cannot_anchor(self) -> None:
        outcome = admitting(proposal(observed_outcome=None))

        (event,) = outcome.events
        assert event.qualifies_as_anchor is False
        assert outcome.anchors == ()
        assert outcome.anchor_ids == frozenset()
        # Kept and reported - a real precedent worth reading, structurally unable to raise.
        assert outcome.recorded_without_outcome == (event,)
        assert outcome.refused == ()

    def test_a_measured_outcome_anchors(self) -> None:
        outcome = admitting(proposal())

        (event,) = outcome.events
        assert event.qualifies_as_anchor is True
        assert outcome.anchor_ids == frozenset({event.event_id})
        assert outcome.by_id(event.event_id) == event
        assert outcome.recorded_without_outcome == ()

    def test_by_id_refuses_an_id_it_did_not_admit(self) -> None:
        assert admitting(proposal()).by_id("he_" + "9" * 32) is None


class TestTheRemainingRefusals:
    def test_an_analogue_in_the_future_is_refused(self) -> None:
        """It has not happened, so there is no outcome to have observed."""
        outcome = admitting(proposal(occurred_at=AS_OF + dt.timedelta(days=1)))

        assert outcome.events == ()
        assert outcome.refused[0][1] == REFUSED_NOT_IN_PAST

    def test_an_analogue_at_the_analysis_instant_is_refused(self) -> None:
        assert admitting(proposal(occurred_at=AS_OF)).refused[0][1] == REFUSED_NOT_IN_PAST

    def test_any_specialist_but_the_historical_analyst_is_refused_outright(self) -> None:
        """One agent carries the anchor obligation, so one agent may discharge it."""
        for agent in ("policy-analyst", "risk-analyst", "consequence-hunter"):
            outcome = admitting(proposal(), agent=agent)

            assert outcome.events == ()
            assert outcome.anchor_ids == frozenset()
            assert outcome.refused == ((proposal().title, REFUSED_WRONG_AGENT),)

    def test_every_refusal_names_the_proposal_and_the_reason(self) -> None:
        """A dropped analogue a reader cannot account for reads like a lost one."""
        outcome = admitting(
            proposal(title="Uncited", cited_evidence_ids=()),
            proposal(title="Future", occurred_at=AS_OF + dt.timedelta(days=30)),
        )

        assert outcome.events == ()
        assert dict(outcome.refused) == {
            "Uncited": REFUSED_UNCITED,
            "Future": REFUSED_NOT_IN_PAST,
        }

    def test_nothing_proposed_is_nothing_refused(self) -> None:
        outcome = admitting()

        assert outcome.events == ()
        assert outcome.refused == ()
        assert outcome.anchor_ids == frozenset()
