"""Assemble the hunter's input, invoke it, then suppress Class A duplicates.

The application path for the first agentic stage. Three inputs go in - a ChangeSpec, the
finalized retrieval evidence, and the Class A findings the resolver already derived - and
retained Class B hypotheses come out.

**The suppression step is deterministic and runs after the model, not instead of it.** The
prompt shows the hunter the Class A set and asks it not to re-derive them; this module does
not trust that. It calls the existing :func:`~app.assumptions.class_a.merge_class_b`, which
suppresses a hypothesis whose ``(assumption_key, affected_source_version_id)`` pair is
already in the Class A set (checkpoint 7.5.2). The rule is reused rather than reimplemented -
a second copy would drift, and this one is already mutation-tested.

Suppression **merges rather than drops**: the absorbing finding records the hypothesis id, so
the model's articulated mechanism stays available to a reader. What must not survive is a
second, independent finding claiming the same thing at a trust level the model did not earn.

**Evidence comes from the finalized retrieval outcome, whichever strategy produced it.** The
hunter is handed ids and text, and it cannot tell whether lexical, vector or hybrid retrieval
found them - which is correct: a strategy is how evidence was reached, not what it says.

What this module does not do: it does not score, rank, judge, challenge or project a graph,
and it does not decide ``consequence_class`` - that is pipeline-assigned and every hypothesis
here is Class B by *producer* (an agent), which is the definition (``CLAUDE.md`` 22).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.assumptions.class_a import merge_class_b
from app.consequences.hunter import ConsequenceHunter, HuntOutcome
from app.consequences.prompting import (
    AdmissibleAnalogue,
    EvidenceForHunting,
    SpecialistObservation,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding, ClassBMergeOutcome
from app.domain.interfaces.retrieval import RetrievalOutcome

logger = logging.getLogger(__name__)

#: Evidence handed to one hunter call. Bounded because retrieval is unbounded relative to a
#: prompt: `top_k` can be 100, and a hunter shown 100 spans is being asked to summarize a
#: corpus rather than reason about a change (``CLAUDE.md``, cost discipline).
MAX_EVIDENCE_SPANS: Final = 24


@dataclass(frozen=True)
class ClassBOutcome:
    """The Class B stage, end to end.

    Both the retained hypotheses and everything that explains the number: what the model
    produced, what the deterministic rule suppressed, and why.
    """

    #: Hypotheses that survived suppression. What a report would show.
    retained: tuple[ConsequenceHypothesis, ...] = ()
    #: Hypotheses absorbed into an existing Class A finding.
    suppressed: tuple[ConsequenceHypothesis, ...] = ()
    #: The deterministic merge result, including which finding absorbed what.
    merge: ClassBMergeOutcome | None = None
    #: The Class A set after absorption, each finding recording what it absorbed.
    class_a: tuple[ClassAFinding, ...] = ()
    #: The raw hunter outcome, so a figure can always be decomposed without re-running.
    hunt: HuntOutcome | None = None
    #: Spans offered to the model, after truncation.
    evidence_offered: int = 0

    @property
    def succeeded(self) -> bool:
        return self.hunt is not None and self.hunt.succeeded

    @property
    def rediscovery_rate(self) -> float | None:
        """Share of proposals that duplicated an existing Class A finding.

        The metric ``docs/EVALUATION.md`` wires as ``class_b_rediscovery_rate``. A **high**
        value is a warning that the hunter's evidence slice is misconfigured, not evidence
        that it is being productive.
        """
        if self.merge is None or self.merge.considered == 0:
            return None
        return len(self.merge.suppressed_hypothesis_ids) / self.merge.considered


class HuntConsequencesUseCase:
    """Run the Class B stage for one change.

    Holds the hunter. It does not hold a retriever: evidence arrives already finalized, so
    this use case cannot widen what the requester was authorized to see - every filter and
    the authorization decision were applied before the outcome it is handed.
    """

    def __init__(self, *, hunter: ConsequenceHunter) -> None:
        self._hunter = hunter

    async def execute(
        self,
        *,
        change: ChangeSpec,
        evidence: RetrievalOutcome,
        class_a: tuple[ClassAFinding, ...] = (),
        specialists: tuple[SpecialistObservation, ...] = (),
        slice_evidence: Sequence[RetrievalOutcome] = (),
        analogues: tuple[AdmissibleAnalogue, ...] = (),
        correlation_id: str | None = None,
    ) -> ClassBOutcome:
        """Assemble, invoke, suppress.

        A failed call yields an outcome with no hypotheses and the failure recorded, never a
        partial set: "found nothing" and "did not run" are different facts about an analysis
        and a reader must be able to tell them apart.

        ``specialists`` and ``slice_evidence`` come from the domain fan-out: the verified
        observations, and the outcomes their slices were drawn from. The slices widen what the
        hunter may *cite* - a specialist retrieves source types the main query did not reach -
        and every one of them came from the same retriever under the same permission scope, so
        the offered set stays a subset of what the requester was authorized to see.

        ``analogues`` is the closed set of historical anchors, each with an id the runtime
        derived from evidence spans. A claimed anchor outside it is dropped, so a Class B
        hypothesis can only exceed ``speculative`` on a precedent somebody evidenced - which is
        checkpoint 9.2 rule 2 becoming reachable rather than changing.
        """
        offered = evidence_for_hunting(evidence, extra=slice_evidence)

        outcome = await self._hunter.hunt(
            change=change,
            evidence=offered,
            class_a=class_a,
            specialists=specialists,
            analogues=analogues,
            correlation_id=correlation_id,
        )

        if not outcome.succeeded:
            return ClassBOutcome(hunt=outcome, class_a=class_a, evidence_offered=len(offered))

        merge, absorbed = merge_class_b(
            outcome.hypotheses,
            class_a,
            affected_artifact_of=outcome.affected_artifact_of,
        )

        suppressed_ids = set(merge.suppressed_hypothesis_ids)
        retained = tuple(h for h in outcome.hypotheses if h.hypothesis_id not in suppressed_ids)
        suppressed = tuple(h for h in outcome.hypotheses if h.hypothesis_id in suppressed_ids)

        if suppressed:
            logger.info(
                "suppressed Class B hypotheses that duplicate an existing Class A finding",
                extra={
                    "change_id": change.change_id,
                    "suppressed": len(suppressed),
                    "considered": merge.considered,
                },
            )

        return ClassBOutcome(
            retained=retained,
            suppressed=suppressed,
            merge=merge,
            class_a=absorbed,
            hunt=outcome,
            evidence_offered=len(offered),
        )


def evidence_for_hunting(
    outcome: RetrievalOutcome,
    *,
    extra: Sequence[RetrievalOutcome] = (),
    limit: int = MAX_EVIDENCE_SPANS,
) -> tuple[EvidenceForHunting, ...]:
    """Project a retrieval outcome into what the hunter is shown.

    Truncated at ``limit`` **from the top of the ranking**, so the spans dropped are the ones
    retrieval ranked least relevant. That is a cost decision with a consequence worth naming:
    a hypothesis that would have rested on span 25 cannot be proposed, and the honest place
    for that is a recorded limit rather than an unbounded prompt.

    The projection is deliberately narrow - id, text, source type, authority. No title, no
    uri, no offsets: the model needs an id it can cite and text it can read, and a match must
    be attributable to words inside the citation.

    ``extra`` holds the specialists' slice outcomes. The main result ranking comes first and
    the slices follow, deduplicated by evidence id - so the spans retrieval ranked most
    relevant to the change keep their places and the domain slices add to the tail rather than
    displacing them.
    """
    projected: dict[str, EvidenceForHunting] = {}
    for source in (outcome, *extra):
        for result in source.results:
            if len(projected) >= limit:
                return tuple(projected.values())
            projected.setdefault(
                result.evidence_id,
                EvidenceForHunting(
                    evidence_id=result.evidence_id,
                    text=result.candidate.text,
                    source_type=result.candidate.source_type.value,
                    authority_class=result.candidate.authority_class.value,
                ),
            )
    return tuple(projected.values())
