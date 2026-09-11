"""The specialist fan-out: slice the authorized evidence, invoke who has something to read.

The application path for the seven analysts. Three deterministic steps around one round of
model calls, and the deterministic parts are where the guarantees live.

**1. Slice, through the same retriever.** One query per specialist, each a ``model_copy`` of
the *caller's* ``RetrievalQuery`` changing only ``source_types`` and ``top_k``. Tenant and
``PermissionScope`` are untouched and cannot be re-decided here, so a specialist sees a subset
of what the requester was authorized to see and never more. Reading the corpus any other way -
the coupling index, the source store - would hand an agent spans the requester is not cleared
for, which is an isolation defect dressed as a feature.

**2. Select on evidence, not on intent.** A specialist is invoked iff its slice is non-empty
(:func:`app.specialists.roster.select`). Every exclusion is reported with its reason, because
"did not run" and "ran and found nothing" are different facts.

**3. Suppress duplicates, deterministically.** Two rules, both after the calls and neither
trusting a model to have cooperated:

* *cross-specialist* - two analysts reporting the same claim on the same citations is one
  observation. Kept in roster order, so the answer does not depend on which call returned
  first.
* *Class A restatement* - an observation on a key the resolver already decided, resting only
  on spans that finding already materialized, is a restatement of a deterministic result at a
  trust level the specialist did not earn. Suppressed, and counted.

Where the findings go
---------------------

Into the **existing** Class B stage, not into a second reasoning path. The consequence-hunter
already exists to combine "independent evidence from multiple domains"; the fan-out is what
makes that literal by producing the domains. Specialist observations reach the hunter as
attributed context alongside the union of the slices' spans, so a hypothesis it proposes can
cite a span only a specialist retrieved - and from there the hypothesis is challenged,
arbitrated and judged like any other.

Nothing here assigns a class, a status, a severity or a number, and nothing here writes to a
store. A specialist finding is an observation; everything downstream of it is unchanged.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Final

from app.application.orchestration.retrieval_service import (
    RetrievalRequest,
    RetrievalService,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.findings import SpecialistFinding
from app.domain.entities.source import HistoricalEvent
from app.domain.interfaces.retrieval import MAX_TOP_K, RetrievalOutcome, RetrievalQuery
from app.specialists.analyst import SpecialistAnalyst, SpecialistOutcome
from app.specialists.proposals import ObservedBinding
from app.specialists.roster import (
    MAX_SLICE_SPANS,
    ROSTER,
    SLICE_TOP_K,
    FanOut,
    SliceSpan,
    SpecialistSlice,
    select,
)

logger = logging.getLogger(__name__)

#: Class A claims shown to each specialist as a do-not-restate list. Bounded: the list is a
#: courtesy to a cooperative model, and the deterministic suppression runs regardless.
MAX_KNOWN_CLAIMS: Final = 12

#: Why a finding did not survive. Reported per finding rather than as a count, because a
#: suppressed observation a reader cannot account for reads like a lost one.
SUPPRESSED_DUPLICATE: Final = "another specialist reported the same claim on the same evidence"
SUPPRESSED_RESTATES_CLASS_A: Final = "restates a deterministic Class A finding from the same spans"


@dataclass(frozen=True)
class SpecialistReport:
    """The fan-out, end to end: who ran, what they found, and what was suppressed."""

    fan_out: FanOut = field(default_factory=FanOut)
    #: One outcome per invoked specialist, in roster order. Includes failures.
    outcomes: tuple[SpecialistOutcome, ...] = ()
    #: Findings that survived both suppression rules. What a report would show.
    retained: tuple[SpecialistFinding, ...] = ()
    #: ``(finding, reason)`` for everything suppressed.
    suppressed: tuple[tuple[SpecialistFinding, str], ...] = ()
    #: Every span any invoked specialist could cite. The Class B stage's extra evidence.
    slice_outcomes: tuple[RetrievalOutcome, ...] = ()

    @property
    def invoked(self) -> tuple[str, ...]:
        return tuple(outcome.agent for outcome in self.outcomes)

    @property
    def failed(self) -> tuple[str, ...]:
        """Specialists whose call failed. One analyst's outage, never the stage's."""
        return tuple(o.agent for o in self.outcomes if not o.succeeded)

    @property
    def succeeded(self) -> bool:
        """Whether every invoked specialist returned. An empty fan-out trivially succeeds."""
        return not self.failed

    @property
    def produced(self) -> int:
        return len(self.retained)

    @property
    def agents_with_findings(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.agent for f in self.retained))

    @property
    def historical_events(self) -> tuple[HistoricalEvent, ...]:
        """Prior cases the historical analyst evidenced. Ids derived, never proposed."""
        return tuple(e for o in self.outcomes for e in o.analogues.events)

    @property
    def anchors(self) -> tuple[HistoricalEvent, ...]:
        """The subset carrying a recorded outcome. The only events that may raise a status."""
        return tuple(e for e in self.historical_events if e.qualifies_as_anchor)

    @property
    def anchor_ids(self) -> frozenset[str]:
        return frozenset(e.event_id for e in self.anchors)

    @property
    def refused_analogues(self) -> tuple[tuple[str, str], ...]:
        """``(title, reason)`` per refused analogue. Reported, never silently dropped."""
        return tuple(item for o in self.outcomes for item in o.analogues.refused)

    @property
    def observed_bindings(self) -> tuple[ObservedBinding, ...]:
        """Propositions the slices commit to. Proposals for registry review, never admitted."""
        return tuple(b for o in self.outcomes for b in o.observed_bindings)

    @property
    def open_questions(self) -> tuple[str, ...]:
        collected: list[str] = []
        for outcome in self.outcomes:
            for question in outcome.unresolved_questions:
                if question not in collected:
                    collected.append(question)
        return tuple(collected)

    @property
    def fabricated_evidence_ids(self) -> tuple[str, ...]:
        return tuple(sorted({i for o in self.outcomes for i in o.fabricated_evidence_ids}))

    @property
    def evidence_ids(self) -> frozenset[str]:
        """Ids the retained findings cite. A subset of what the requester was authorized to see."""
        return frozenset(i for f in self.retained for i in f.evidence_ids)

    def findings_by(self, agent: str) -> tuple[SpecialistFinding, ...]:
        return tuple(f for f in self.retained if f.agent == agent)


class ConsultSpecialistsUseCase:
    """Run the fan-out for one change. Holds the analysts and the retriever, nothing else.

    ``analysts`` is keyed by agent name, so a deployment can enable a subset by supplying a
    subset - a specialist with no analyst is reported un-selected rather than missing.
    """

    def __init__(
        self,
        *,
        analysts: dict[str, SpecialistAnalyst],
        retrieval: RetrievalService,
    ) -> None:
        unknown = sorted(set(analysts) - {s.agent for s in ROSTER})
        if unknown:
            raise ValueError(
                f"{unknown} are not on the specialist roster; this fan-out dispatches the "
                "seven analysts only"
            )
        self._analysts = analysts
        self._retrieval = retrieval

    async def execute(
        self,
        *,
        change: ChangeSpec,
        query: RetrievalQuery,
        class_a: tuple[ClassAFinding, ...] = (),
        as_of: dt.datetime | None = None,
        correlation_id: str | None = None,
    ) -> SpecialistReport:
        """Slice, select, invoke, suppress."""
        slices: list[SpecialistSlice] = []
        outcomes_by_agent: dict[str, RetrievalOutcome] = {}
        for specialist in ROSTER:
            finalized = await self._retrieval.retrieve(
                RetrievalRequest(query=slice_query(query, specialist.source_types))
            )
            outcomes_by_agent[specialist.agent] = finalized.outcome
            slices.append(_slice_from(specialist, finalized.outcome))

        fan_out = select(slices, available=tuple(self._analysts))
        if not fan_out.selected:
            logger.info(
                "specialist_fan_out_empty",
                extra={"change_id": change.change_id, "reasons": fan_out.reasons},
            )
            return SpecialistReport(fan_out=fan_out)

        known = tuple(f.summary for f in class_a)[:MAX_KNOWN_CLAIMS]
        outcomes: list[SpecialistOutcome] = []
        for chosen in fan_out.selected:
            analyst = self._analysts[chosen.agent]
            outcomes.append(
                await analyst.observe(
                    change=change,
                    slice_=chosen,
                    known_claims=known,
                    # The event id recipe includes the tenant, so two tenants holding the
                    # same span cannot collide onto one precedent.
                    tenant_id=query.tenant_id,
                    as_of=as_of,
                    correlation_id=correlation_id,
                )
            )

        retained, suppressed = suppress(
            tuple(f for o in outcomes for f in o.findings), class_a=class_a
        )

        logger.info(
            "specialist_fan_out_complete",
            extra={
                "change_id": change.change_id,
                "invoked": [o.agent for o in outcomes],
                "failed": [o.agent for o in outcomes if not o.succeeded],
                "retained": len(retained),
                "suppressed": len(suppressed),
                "anchors": len([e for o in outcomes for e in o.analogues.anchors]),
                "shared_spans": len(fan_out.shared_evidence_ids),
            },
        )
        return SpecialistReport(
            fan_out=fan_out,
            outcomes=tuple(outcomes),
            retained=retained,
            suppressed=suppressed,
            slice_outcomes=tuple(outcomes_by_agent[chosen.agent] for chosen in fan_out.selected),
        )


def slice_query(query: RetrievalQuery, source_types: frozenset[object]) -> RetrievalQuery:
    """One specialist's query: the caller's, narrowed to its source types.

    Derived by copy so nothing about the authorization boundary can be re-decided - only
    ``source_types`` and ``top_k`` change. A slice query that widened the scope would be a
    privilege escalation with a plausible motive.
    """
    from app.domain.value_objects.enums import SourceType

    ordered = tuple(
        sorted((t for t in source_types if isinstance(t, SourceType)), key=lambda t: t.value)
    )
    return query.model_copy(update={"source_types": ordered, "top_k": min(MAX_TOP_K, SLICE_TOP_K)})


def _slice_from(specialist: object, outcome: RetrievalOutcome) -> SpecialistSlice:
    """Project a retrieval outcome into what one specialist is shown.

    Truncated from the top of the ranking, so the spans dropped are the ones retrieval ranked
    least relevant - and the count is recorded, because a hypothesis that would have rested on
    span 13 cannot be proposed and the honest place for that is a reported limit.
    """
    from app.specialists.roster import Specialist

    assert isinstance(specialist, Specialist)  # narrows for the type checker
    results = outcome.results[:MAX_SLICE_SPANS]
    return SpecialistSlice(
        specialist=specialist,
        spans=tuple(
            SliceSpan(
                evidence_id=result.candidate.evidence_id,
                text=result.candidate.text,
                source_type=result.candidate.source_type.value,
                authority_class=result.candidate.authority_class.value,
                source_title=result.candidate.document.title,
                assumption_keys=tuple(result.candidate.assumption_keys),
            )
            for result in results
        ),
        truncated=max(0, len(outcome.results) - MAX_SLICE_SPANS),
    )


def suppress(
    findings: tuple[SpecialistFinding, ...], *, class_a: tuple[ClassAFinding, ...]
) -> tuple[tuple[SpecialistFinding, ...], tuple[tuple[SpecialistFinding, str], ...]]:
    """The two deterministic suppression rules. Runs after the calls, trusts no cooperation.

    Order is the input order, which the fan-out fixes to roster order - so which of two
    duplicate observations survives does not depend on which call returned first.
    """
    class_a_scope = {
        (finding.assumption_key, frozenset(finding.materialized_evidence_ids))
        for finding in class_a
    }
    class_a_keys = {finding.assumption_key for finding in class_a}

    retained: list[SpecialistFinding] = []
    suppressed: list[tuple[SpecialistFinding, str]] = []
    seen: set[tuple[str, frozenset[str]]] = set()

    for finding in findings:
        identity = (_normalise(finding.claim), frozenset(finding.evidence_ids))
        if identity in seen:
            suppressed.append((finding, SUPPRESSED_DUPLICATE))
            continue
        if _restates_class_a(finding, keys=class_a_keys, scopes=class_a_scope):
            suppressed.append((finding, SUPPRESSED_RESTATES_CLASS_A))
            continue
        seen.add(identity)
        retained.append(finding)

    return tuple(retained), tuple(suppressed)


def _restates_class_a(
    finding: SpecialistFinding,
    *,
    keys: set[str],
    scopes: set[tuple[str, frozenset[str]]],
) -> bool:
    """Whether this observation re-reports a deterministic finding from the same spans.

    Both conditions, and the second is the one that matters: an observation on a decided key
    that cites *other* evidence is new information about that key and survives. Only one
    resting entirely on spans the resolver already materialized is a restatement - the same
    conclusion at a trust level the specialist did not earn.
    """
    if not finding.evidence_ids:
        return False
    if not (set(finding.assumption_keys) & keys):
        return False
    cited = frozenset(finding.evidence_ids)
    return any(key in finding.assumption_keys and cited <= spans for key, spans in scopes)


def _normalise(claim: str) -> str:
    """Claim text for duplicate comparison. Whitespace and case only.

    Deliberately not a similarity measure: two specialists phrasing one observation slightly
    differently are two observations as far as this rule is concerned, and collapsing them on
    a threshold would silently drop a finding nobody could account for.
    """
    return " ".join(claim.lower().split())
