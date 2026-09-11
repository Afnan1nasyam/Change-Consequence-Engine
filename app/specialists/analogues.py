"""Turn the historical analyst's proposals into admitted :class:`HistoricalEvent` records.

The Class B story had one piece missing. A behavioural claim cannot exceed ``speculative``
without a historical analogue carrying an observed outcome (checkpoint 9.2 rule 2), the
historical analyst's definition makes it "the anchor for Class B claims" - and
``historical_analogue_id`` is an ``EventId``, which nothing in the system could produce. So the
gate was unreachable in the raising direction and, worse, **open in the wrong one**: the
consequence-hunter's proposal has always had a ``historical_analogue_id`` field, so a model
emitting a well-formed invented ``he_...`` would have lifted a Class B finding above
``speculative`` with nothing behind it.

This module closes both halves, and neither needed a new domain model or a new table:
:class:`~app.domain.entities.source.HistoricalEvent` already exists with exactly the right
shape, including ``qualifies_as_anchor``.

Four admission rules, all deterministic
---------------------------------------

**The id is derived, never proposed.** :func:`app.ingestion.identity.event_id_for` hashes the
tenant and the cited spans, so the model has no field to put an id in and no way to influence
the one it gets. Two analyses citing the same spans agree on the id - which is why the anchor
resolves by re-derivation rather than by lookup, and why no store was needed.

**Citations are checked against the historical analyst's own slice.** Its slice came from the
same retriever under the same ``PermissionScope``, so an event can only rest on spans the
requester was authorized to read. An id from outside the slice is dropped exactly like a
fabricated one.

**An event with no admitted citation is refused outright.** Not recorded-but-unqualified:
without a span there is nothing to be traceable to, and an untraceable event record is worse
than none.

**Only a recorded observed outcome anchors.** An event with citations but no measured outcome
is *kept* - it is a real precedent worth showing a reader - and ``qualifies_as_anchor`` is
``False``, so it cannot lift a status. That is the analyst's own instruction ("an analogue with
no recorded outcome does not qualify as an anchor - say so") expressed as a property of the
record rather than a request in a prompt.

A future ``occurred_at`` is refused as well. An analogue that has not happened yet is a
misreading, not a precedent.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Final

from app.domain.entities.source import HistoricalEvent
from app.ingestion.identity import event_id_for
from app.specialists.proposals import AnalogueProposal

logger = logging.getLogger(__name__)

#: Bumped when an admission rule changes, because that changes which analogues can anchor.
ANALOGUE_ADMISSION_VERSION: Final = "historical-analogue-1"

#: The agent whose definition asks for analogues. Every other specialist's ``historical_events``
#: are refused - one agent has the anchor obligation, so one agent may discharge it.
ANALOGUE_AGENT: Final = "historical-analyst"

REFUSED_UNCITED: Final = "no admitted citation, so nothing to be traceable to"
REFUSED_NOT_IN_PAST: Final = "occurred_at is not before the analysis as_of"
REFUSED_WRONG_AGENT: Final = "only the historical-analyst may propose an analogue"


@dataclass(frozen=True)
class AnalogueOutcome:
    """The admitted events, and everything that did not survive.

    ``anchors`` is the subset a Class B hypothesis may cite. Separating it from ``events``
    rather than filtering at the call site keeps one definition of "qualifies": the entity's.
    """

    events: tuple[HistoricalEvent, ...] = ()
    #: ``(title, reason)`` per refusal. Titles rather than ids, because a refused proposal
    #: never got an id - which is the point.
    refused: tuple[tuple[str, str], ...] = ()
    #: Cited ids that were not in the analyst's slice.
    fabricated_evidence_ids: tuple[str, ...] = ()
    admission_version: str = ANALOGUE_ADMISSION_VERSION

    @property
    def anchors(self) -> tuple[HistoricalEvent, ...]:
        """Events that may raise a Class B status. The entity decides, not this module."""
        return tuple(event for event in self.events if event.qualifies_as_anchor)

    @property
    def anchor_ids(self) -> frozenset[str]:
        """The closed set a hunter's ``historical_analogue_id`` is checked against."""
        return frozenset(event.event_id for event in self.anchors)

    @property
    def recorded_without_outcome(self) -> tuple[HistoricalEvent, ...]:
        """Real precedents that cannot anchor. Shown to a reader, never used to raise."""
        return tuple(event for event in self.events if not event.qualifies_as_anchor)

    def by_id(self, event_id: str) -> HistoricalEvent | None:
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None


def admit(
    proposals: tuple[AnalogueProposal, ...],
    *,
    agent: str,
    tenant_id: str,
    offered_evidence_ids: frozenset[str],
    as_of: dt.datetime,
) -> AnalogueOutcome:
    """Admit what the rules allow. Every step can only weaken what a proposal claims.

    There is no branch here that adds evidence, mints an id from anything but the spans, or
    turns an outcome-less precedent into an anchor.
    """
    if agent != ANALOGUE_AGENT:
        if proposals:
            logger.warning(
                "a specialist other than the historical analyst proposed an analogue; refused",
                extra={"agent": agent, "count": len(proposals)},
            )
        return AnalogueOutcome(refused=tuple((p.title, REFUSED_WRONG_AGENT) for p in proposals))

    events: list[HistoricalEvent] = []
    refused: list[tuple[str, str]] = []
    fabricated: list[str] = []
    seen: set[str] = set()

    for proposal in proposals:
        kept = tuple(sorted({i for i in proposal.cited_evidence_ids if i in offered_evidence_ids}))
        fabricated.extend(i for i in proposal.cited_evidence_ids if i not in offered_evidence_ids)

        if not kept:
            refused.append((proposal.title, REFUSED_UNCITED))
            continue
        if proposal.occurred_at >= as_of:
            refused.append((proposal.title, REFUSED_NOT_IN_PAST))
            continue

        event_id = event_id_for(tenant_id=tenant_id, evidence_ids=kept)
        if event_id in seen:
            # The same spans twice is the same event described twice, not two precedents.
            continue
        seen.add(event_id)
        events.append(
            HistoricalEvent(
                event_id=event_id,
                tenant_id=tenant_id,
                title=proposal.title,
                occurred_at=proposal.occurred_at,
                # Empty stays empty. A precedent with no measured outcome is recorded and
                # cannot anchor, which the entity's `qualifies_as_anchor` decides.
                observed_outcome=proposal.observed_outcome,
                evidence_ids=kept,
            )
        )

    if fabricated:
        logger.warning(
            "historical analyst cited spans outside its slice for an analogue; dropped",
            extra={"agent": agent, "count": len(fabricated)},
        )
    if refused:
        logger.info(
            "historical analogues refused",
            extra={"agent": agent, "reasons": [reason for _title, reason in refused]},
        )

    return AnalogueOutcome(
        events=tuple(events),
        refused=tuple(refused),
        fabricated_evidence_ids=tuple(sorted(set(fabricated))),
    )
