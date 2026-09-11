"""Assemble the candidate set, measure its evidence, and ask the arbiter to position it.

The application seam for the evidence arbiter. It reuses three things the judgment path
already owns rather than growing its own copies:

* :func:`~app.judgment.dossier.dossiers_from` for candidate resolution, so a candidate has
  one identity and one handle across both stages;
* :func:`~app.application.use_cases.judge_consequences.measure_confidence_inputs` for the
  deterministic factor measurement, so the corroboration count the arbiter's ceiling uses is
  the same number the score is computed from;
* :func:`~app.application.use_cases.judge_consequences.evidence_for_judgment` for the span
  projection, so both stages see identical text under identical ids.

What it produces is the per-candidate :class:`~app.judgment.arbitration.EvidenceRuling` and -
the part the pipeline consumes - ``extra_disagreements``, in exactly the shape
``dossiers_from`` accepts. That is how a disagreement the arbiter recorded reaches the judge
without this module knowing anything about status.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.application.use_cases.judge_consequences import (
    evidence_for_judgment,
    measure_confidence_inputs,
)
from app.challenge.candidates import ChallengeTarget
from app.challenge.challenger import ChallengedCandidate
from app.contradictions.hunter import AssessedDivergence
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.coupling import ResolvedBinding
from app.domain.entities.source import EvidenceItem
from app.domain.interfaces.retrieval import RetrievalOutcome
from app.domain.value_objects.enums import EvidencePosition
from app.judgment.arbitration import (
    ArbitrationOutcome,
    EvidenceArbiter,
    EvidenceRuling,
)
from app.judgment.confidence_inputs import AssembledInputs
from app.judgment.dossier import CandidateDossier, dossiers_from


@dataclass(frozen=True)
class ArbitrationReport:
    """Everything the arbitration produced, and the measurement it was graded against."""

    dossiers: tuple[CandidateDossier, ...]
    outcome: ArbitrationOutcome
    measured: dict[str, AssembledInputs] = field(default_factory=dict)

    @property
    def rulings(self) -> tuple[EvidenceRuling, ...]:
        return self.outcome.rulings

    @property
    def succeeded(self) -> bool:
        return self.outcome.succeeded

    @property
    def by_consequence(self) -> dict[str, EvidenceRuling]:
        return self.outcome.by_consequence

    @property
    def positions(self) -> dict[str, EvidencePosition]:
        return {r.consequence_id: r.position for r in self.rulings}

    @property
    def extra_disagreements(self) -> dict[str, tuple[str, ...]]:
        """Per-consequence disagreements, in the shape ``dossiers_from`` consumes."""
        return self.outcome.preserved_disagreements

    @property
    def carried_by_evidence(self) -> tuple[EvidenceRuling, ...]:
        """Rulings the evidence carried to an assertive position after the ceiling."""
        return tuple(
            r
            for r in self.rulings
            if r.position in (EvidencePosition.SUPPORTED, EvidencePosition.WEAKLY_SUPPORTED)
        )

    @property
    def clamped_share(self) -> float | None:
        """Share of rulings the deterministic ceiling had to lower.

        A **high** value is not a defect in the arbiter: it is the distance between what the
        model would have asserted and what the evidence supports, which is the measurement
        this stage exists to produce.
        """
        if not self.rulings:
            return None
        return len(self.outcome.clamped) / len(self.rulings)


class ArbitrateEvidenceUseCase:
    """One evidence-arbiter run over one change's candidate set. Holds the arbiter only."""

    def __init__(self, *, arbiter: EvidenceArbiter) -> None:
        self._arbiter = arbiter

    async def execute(
        self,
        *,
        change: ChangeSpec,
        targets: Iterable[ChallengeTarget] = (),
        challenged: Iterable[ChallengedCandidate] = (),
        class_a: Iterable[ClassAFinding] = (),
        hypotheses: Iterable[ConsequenceHypothesis] = (),
        divergences: Iterable[AssessedDivergence] = (),
        evidence_gap_ids: Iterable[str] = (),
        hunt_questions: Iterable[str] = (),
        evidence: Mapping[str, EvidenceItem] | None = None,
        outcomes: Sequence[RetrievalOutcome] = (),
        bindings: Mapping[str, ResolvedBinding] | None = None,
        now: dt.datetime | None = None,
        correlation_id: str | None = None,
    ) -> ArbitrationReport:
        """Assemble, measure, arbitrate.

        The candidate inputs mirror :class:`JudgeConsequencesUseCase.execute` deliberately:
        both stages must resolve the *same* candidate set, and passing them the same
        arguments is a stronger guarantee of that than any assertion downstream.
        """
        as_of = now or dt.datetime.now(tz=dt.UTC)
        dossiers = dossiers_from(
            targets=targets,
            challenged=challenged,
            class_a=class_a,
            hypotheses=hypotheses,
            divergences=divergences,
            evidence_gap_ids=evidence_gap_ids,
            hunt_questions=hunt_questions,
        )
        measured = measure_confidence_inputs(
            dossiers, outcomes=outcomes, bindings=bindings or {}, as_of=as_of
        )
        outcome = await self._arbiter.arbitrate(
            change=change,
            dossiers=dossiers,
            evidence=evidence_for_judgment(dossiers, evidence=evidence or {}),
            measured=measured,
            now=as_of,
            correlation_id=correlation_id,
        )
        return ArbitrationReport(dossiers=dossiers, outcome=outcome, measured=measured)
