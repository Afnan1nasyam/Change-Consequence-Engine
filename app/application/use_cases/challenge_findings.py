"""Put the finalized candidates in front of the challenger, and report what survived review.

The application seam for the adversarial challenger. It projects the finalized Class A
findings and Class B hypotheses into challengeable targets, hands them over with whatever
evidence the caller supplied, and reports the verdicts.

**It does not search, and neither does the agent.** Every citable span arrives as an argument.
That matters more here than for the other agentic stages, because the challenger's most
valuable finding - *a newer source version supersedes the one the derivation used* - is by
definition not in the candidate's own evidence set. So there is a route for it
(``extra_evidence`` / ``context``), and the route is the caller's: the retrieval, if any, is
performed upstream and the challenger sees only its result.

**What review can and cannot change.**

======================== ============================================================
verdict                  effect on the candidate
======================== ============================================================
``refuted``              recorded against it; the candidate is **not** deleted
``weakened``             recorded against it, with how the scope narrows
``no_refutation_found``  recorded, and yields a ``FalsificationTest``
refused / unchallenged   nothing recorded; the candidate is flagged as unreviewed
======================== ============================================================

A ``refuted`` verdict is a *finding about* the candidate, not a deletion of it. Suppression is
the judge's decision over the arbitrated set, and a challenger that could delete a
deterministic derivation would be the agreeable-model problem in its most direct form. So this
layer emits ``ChallengeResult`` records beside the candidates and changes none of them - a
test dumps every candidate before and after a full run and compares.

**``no_refutation_found`` is not corroboration.** It means nobody found contrary evidence,
which is strictly weaker than evidence of correctness, and it must never raise a status
(checkpoint 9.2). :attr:`ChallengeReport.survived` is named for what it is - review survived,
not correctness established.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from app.challenge.candidates import ChallengeTarget, targets_from
from app.challenge.challenger import (
    AdversarialChallenger,
    ChallengedCandidate,
    ChallengeOutcomeSet,
)
from app.challenge.prompting import EvidenceForChallenge
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.findings import ChallengeResult, FalsificationTest
from app.domain.entities.source import EvidenceItem

#: Cap on the spans one call may carry. Retrieval feeds reasoning and reasoning is priced per
#: token (``CLAUDE.md``, cost discipline). More candidates than this need splitting into
#: several calls, not a larger prompt.
MAX_EVIDENCE_SPANS: Final = 32

#: Truncation width for a span rendered into the prompt. Long enough to judge whether it
#: contradicts the claim, short enough that thirty fit.
SPAN_CHARS: Final = 600


@dataclass(frozen=True)
class ChallengeReport:
    """The candidates, the verdicts, and the deliverables review produced.

    All three, because they answer different questions. ``targets`` is what was put up for
    review; ``outcome`` is what the challenger said and what admitting it revealed;
    ``results`` and ``tests`` are the domain records a reader and the validation checklist
    consume.
    """

    targets: tuple[ChallengeTarget, ...]
    outcome: ChallengeOutcomeSet
    results: tuple[ChallengeResult, ...]
    tests: tuple[FalsificationTest, ...]

    @property
    def refuted(self) -> tuple[ChallengedCandidate, ...]:
        return self.outcome.refuted

    @property
    def weakened(self) -> tuple[ChallengedCandidate, ...]:
        return self.outcome.weakened

    @property
    def survived(self) -> tuple[ChallengedCandidate, ...]:
        """Candidates the challenger reviewed and could not refute.

        Named for what it is. Surviving review is **not** corroboration: nobody finding
        contrary evidence is strictly weaker than evidence of correctness (checkpoint 9.2).
        """
        return self.outcome.unrefuted

    @property
    def unreviewed(self) -> tuple[ChallengedCandidate, ...]:
        """Candidates with no admitted verdict - ignored, or refused by the gate.

        Reported separately from ``survived`` because conflating them is the one accounting
        error that would flatter the pipeline: a stage that failed would otherwise look like
        a stage that found nothing wrong.
        """
        return self.outcome.unchallenged

    @property
    def false_positives_removed(self) -> int:
        """The metric the challenger is actually for.

        Refuted plus weakened. A challenger's value is how many false positives it removes,
        not how many objections it raises - so the count that matters is the one that
        survived the citation obligation, which is exactly what ``results`` contains.
        """
        return len(self.refuted) + len(self.weakened)

    @property
    def objection_refusal_rate(self) -> float | None:
        """Share of the model's verdicts the gate would not admit.

        A calibration signal about the *model*, not the corpus. A high value means it is
        producing objections it cannot support, which is the failure mode this whole stage is
        designed around. ``None`` when it returned nothing to judge.
        """
        attempted = len(self.results) + len(self.outcome.refused)
        return len(self.outcome.refused) / attempted if attempted else None


class ChallengeFindingsUseCase:
    """One adversarial-challenger run over one change's finalized candidates.

    Holds the challenger and nothing else. No store and no retriever: candidates and evidence
    are passed in, so this class cannot widen what the agent sees.
    """

    def __init__(self, *, challenger: AdversarialChallenger) -> None:
        self._challenger = challenger

    async def execute(
        self,
        *,
        change: ChangeSpec,
        class_a: Iterable[ClassAFinding] = (),
        hypotheses: Iterable[ConsequenceHypothesis] = (),
        evidence: Mapping[str, EvidenceItem] | None = None,
        extra_evidence: Mapping[str, tuple[str, ...]] | None = None,
        context: tuple[EvidenceForChallenge, ...] = (),
        correlation_id: str | None = None,
    ) -> ChallengeReport:
        """Project, challenge, and report.

        ``evidence`` maps evidence id to the stored span, for rendering. ``extra_evidence``
        widens a candidate's *citable* set, keyed by ``finding_id`` or ``hypothesis_id`` -
        the route for a superseding version or an exception clause, which by definition is
        not in the candidate's own evidence. ``context`` is the same route for a span already
        projected by the caller, carrying the candidates it is admissible against.

        A context span naming an unknown candidate is refused: a citation admitted against a
        candidate that does not exist is the fabricated-provenance case in a different
        costume, and it is a caller bug rather than a model one.
        """
        targets = targets_from(
            class_a=class_a, hypotheses=hypotheses, extra_evidence=extra_evidence
        )
        _check_context(context, targets)

        outcome = await self._challenger.challenge(
            change=change,
            targets=targets,
            evidence=evidence_for_challenge(targets, evidence=evidence or {}) + context,
            correlation_id=correlation_id,
        )

        return ChallengeReport(
            targets=targets,
            outcome=outcome,
            results=tuple(c.result for c in outcome.challenged if c.result is not None),
            tests=outcome.falsification_tests,
        )


def _check_context(
    context: tuple[EvidenceForChallenge, ...],
    targets: tuple[ChallengeTarget, ...],
) -> None:
    """Every contextual span must name at least one real candidate it bears on."""
    handles = {target.target_id for target in targets}
    for span in context:
        unknown = set(span.target_ids) - handles
        if unknown or not span.target_ids:
            raise ValueError(
                f"contextual span {span.evidence_id} is admissible against "
                f"{sorted(unknown) or 'no candidate'}, which is not in the candidate set "
                f"{sorted(handles)}. Contextual evidence must be attached to a candidate "
                "that exists."
            )


def evidence_for_challenge(
    targets: tuple[ChallengeTarget, ...],
    *,
    evidence: Mapping[str, EvidenceItem],
    limit: int = MAX_EVIDENCE_SPANS,
) -> tuple[EvidenceForChallenge, ...]:
    """Project the spans behind the candidate set, recording which candidates each serves.

    One span may be admissible against several candidates - the policy sentence a Class A
    finding rests on is often the same one a Class B hypothesis cites - so ``target_ids`` is a
    tuple and the span is rendered once. Deduplicating is not only cheaper: the same text
    twice in one prompt invites the model to read it as two independent sources, and
    corroboration counts sources, never repetitions (``CLAUDE.md`` 16).

    A span the caller did not supply is simply absent. The candidate is still challengeable,
    with less to cite; the alternative would be this layer reaching into a store to fill the
    gap, which is the search neither this layer nor the agent may perform.
    """
    serves: dict[str, list[str]] = {}
    for target in targets:
        for evidence_id in target.evidence_ids:
            serves.setdefault(evidence_id, []).append(target.target_id)

    spans: list[EvidenceForChallenge] = []
    for evidence_id in sorted(serves):
        stored = evidence.get(evidence_id)
        if stored is None:
            continue
        spans.append(
            EvidenceForChallenge(
                evidence_id=evidence_id,
                text=stored.text_span[:SPAN_CHARS],
                source_title=stored.location or "(unlabelled span)",
                authority_class=stored.metadata.get("authority_class", "unknown"),
                target_ids=tuple(serves[evidence_id]),
            )
        )
        if len(spans) >= limit:
            break

    return tuple(spans)
