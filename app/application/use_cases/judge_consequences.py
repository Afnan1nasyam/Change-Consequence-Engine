"""Assemble the whole pipeline's output, arbitrate it, and report what a reader sees.

The application seam for the consequence judge, and the first place in this system where all
five upstream stages meet. It does three things:

1. assemble one dossier per candidate from the resolver, the consequence-hunter, the
   contradiction-hunter, the adversarial-challenger and the EvidenceGap producer;
2. run one judge call and apply the frozen status gates;
3. report the ``Consequence`` / ``ArbitrationDecision`` / ``ConsequenceScore`` triples plus
   everything the run could not settle.

**It does not search, and it does not score.** Evidence arrives as an argument; confidence
comes from :mod:`app.domain.services.confidence`, measured by
:mod:`app.judgment.confidence_inputs` over the retrieval outcomes the caller supplies. This
layer therefore never produces a number itself, and could not if it wanted to:
``ConfidenceInputs`` has no field for a confidence.

Pass ``outcomes`` - one :class:`RetrievalOutcome` per executed strategy - and the factors are
measured from it per candidate. Pass ``confidence_inputs`` to supply them directly, which is
what a caller that measured them elsewhere does. Pass neither and every score is an honest
zero: the inputs were not measured, and a zero says so where a missing score would say
nothing at all.

**An EvidenceGap never becomes a consequence.** Gaps enter as ids and leave as
``Consequence.evidence_gap_ids``: a reference from a finding to a known hole in the corpus.
Type E is a claim about the *corpus*, not about the world (checkpoint 2), which is why it can
be neither an edge nor a consequence. There is no parameter here that turns one into a
candidate, and a test asserts that adding gaps adds no consequences.

**Uncertainty is a first-class output, so nothing here trims it** (``CLAUDE.md`` 7). The
report carries the undecided candidates, the clamped statuses, the refused dispositions and
every unanswered question from every stage - and each is reported *separately*, because "the
judge found it immaterial", "the gates lowered it", "the judge never dispositioned it" and
"nobody could answer this" are four different facts and collapsing them is how an
uncertainty-aware report becomes a confident one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.challenge.candidates import ChallengeTarget
from app.challenge.challenger import ChallengedCandidate
from app.contradictions.hunter import AssessedDivergence
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.coupling import ResolvedBinding
from app.domain.entities.findings import (
    ArbitrationDecision,
    Consequence,
    ConsequenceScore,
)
from app.domain.entities.source import EvidenceItem
from app.domain.interfaces.retrieval import RetrievalOutcome
from app.domain.services.confidence import ConfidenceInputs
from app.domain.value_objects.provenance import RunProvenance
from app.judgment.confidence_inputs import AssembledInputs, assemble
from app.judgment.dossier import CandidateDossier, dossiers_from
from app.judgment.judge import (
    ArbitratedConsequence,
    ConsequenceJudge,
    JudgmentOutcome,
)
from app.judgment.prompting import EvidenceForJudgment
from app.judgment.status_rules import ClampReason

#: Cap on the spans one call may carry. Retrieval feeds reasoning and reasoning is priced per
#: token (``CLAUDE.md``, cost discipline).
MAX_EVIDENCE_SPANS: Final = 40

#: Truncation width for a span rendered into the prompt.
SPAN_CHARS: Final = 600


@dataclass(frozen=True)
class JudgmentReport:
    """Everything the arbitration produced, and everything it could not settle."""

    dossiers: tuple[CandidateDossier, ...]
    outcome: JudgmentOutcome
    consequences: tuple[Consequence, ...]
    decisions: tuple[ArbitrationDecision, ...]
    scores: tuple[ConsequenceScore, ...]
    #: Every unanswered question from every stage, verbatim. Not deduplicated against the
    #: gaps: a question the corpus could not answer and a recorded gap are different
    #: artefacts, and only one of them has negative provenance behind it.
    open_questions: tuple[str, ...] = ()
    #: What the confidence measurement could reach, per consequence. Reported because a
    #: factor measured as absent and a factor nothing could measure are different facts, and
    #: the score alone cannot tell them apart.
    confidence_coverage: dict[str, AssembledInputs] = field(default_factory=dict)

    @property
    def unmeasured_confidence_factors(self) -> tuple[str, ...]:
        """Factors no candidate could measure, across the run.

        The honest companion to a confidence number. On a corpus where nothing states an
        effective date every score loses freshness, and a reader comparing two analyses needs
        to know that is the corpus rather than the findings.
        """
        missing: list[str] = []
        for assembled in self.confidence_coverage.values():
            for factor in assembled.coverage.unmeasured:
                if factor not in missing:
                    missing.append(factor)
        return tuple(missing)

    @property
    def default_view(self) -> tuple[ArbitratedConsequence, ...]:
        """What a reviewer sees first. Narrowed from the judge's proposal by the surface rule."""
        return self.outcome.default_view

    @property
    def undecided(self) -> tuple[CandidateDossier, ...]:
        """Candidates the judge did not disposition, or whose disposition was refused.

        Not the same as `rejected`, and reported apart from it. A finding nobody arbitrated
        has not been cleared.
        """
        return self.outcome.undecided

    @property
    def clamped(self) -> tuple[ArbitratedConsequence, ...]:
        return self.outcome.clamped

    @property
    def preserved_disagreements(self) -> tuple[str, ...]:
        """Every disagreement that survived into a decision, across all candidates."""
        collected: list[str] = []
        for decision in self.decisions:
            for item in decision.preserved_disagreements:
                if item not in collected:
                    collected.append(item)
        return tuple(collected)

    @property
    def evidence_gap_ids(self) -> tuple[str, ...]:
        """The gaps referenced by the arbitrated set. Never consequences themselves."""
        collected: set[str] = set()
        for consequence in self.consequences:
            collected.update(consequence.evidence_gap_ids)
        return tuple(sorted(collected))

    @property
    def unsupported_share(self) -> float | None:
        """Share of arbitrated findings the evidence could not carry to an assertive status.

        The honest headline for a thin corpus. A high value is not a defect in the judge - it
        is the corpus telling the reader what it cannot establish, and hiding it would make
        every other number in the report less trustworthy.
        """
        if not self.outcome.arbitrated:
            return None
        lowered = sum(
            1
            for a in self.outcome.arbitrated
            if ClampReason.NO_SUPPORTING_EVIDENCE in a.clamp.reasons
        )
        return lowered / len(self.outcome.arbitrated)


class JudgeConsequencesUseCase:
    """One consequence-judge run over one change's finalized pipeline output.

    Holds the judge and nothing else: no store, no retriever, and no confidence function of
    its own.
    """

    def __init__(self, *, judge: ConsequenceJudge) -> None:
        self._judge = judge

    async def execute(
        self,
        *,
        change: ChangeSpec,
        run_provenance: RunProvenance,
        targets: Iterable[ChallengeTarget] = (),
        challenged: Iterable[ChallengedCandidate] = (),
        class_a: Iterable[ClassAFinding] = (),
        hypotheses: Iterable[ConsequenceHypothesis] = (),
        divergences: Iterable[AssessedDivergence] = (),
        evidence_gap_ids: Iterable[str] = (),
        hunt_questions: Iterable[str] = (),
        evidence: Mapping[str, EvidenceItem] | None = None,
        context: tuple[EvidenceForJudgment, ...] = (),
        outcomes: Sequence[RetrievalOutcome] = (),
        bindings: Mapping[str, ResolvedBinding] | None = None,
        confidence_inputs: Mapping[str, ConfidenceInputs] | None = None,
        extra_disagreements: Mapping[str, tuple[str, ...]] | None = None,
        now: dt.datetime | None = None,
        correlation_id: str | None = None,
    ) -> JudgmentReport:
        """Assemble, arbitrate, and report.

        ``targets`` and ``challenged`` come from the challenge slice and carry the
        ``consequence_id`` the rest of the pipeline is keyed on; ``class_a`` and
        ``hypotheses`` cover candidates the challenger never saw, because a review stage that
        failed must not delete the findings it failed to review.

        ``outcomes`` is one :class:`RetrievalOutcome` per executed strategy. Where supplied,
        the confidence factors are measured from it per candidate - authority, freshness,
        corroborating versions, strategy reach - by
        :func:`app.judgment.confidence_inputs.assemble`. ``confidence_inputs`` overrides that
        for a caller that measured them elsewhere. With neither, every score is an honest
        zero: the inputs were not measured, and a zero says so where a missing score would
        say nothing at all.
        """
        dossiers = dossiers_from(
            targets=targets,
            challenged=challenged,
            class_a=class_a,
            hypotheses=hypotheses,
            divergences=divergences,
            evidence_gap_ids=evidence_gap_ids,
            hunt_questions=hunt_questions,
            # The evidence arbiter's residual disagreements arrive here. They travel into
            # every dossier and out through the decision; nothing in the status rules reads
            # them, so an arbiter cannot raise a status by disagreeing eloquently.
            extra_disagreements=extra_disagreements,
        )
        _check_context(context, dossiers)

        measured = measure_confidence_inputs(
            dossiers,
            outcomes=outcomes,
            bindings=bindings or {},
            as_of=now or dt.datetime.now(tz=dt.UTC),
        )
        # A caller that measured the factors elsewhere wins over the assembly, so a future
        # store-backed measurement can be dropped in without touching this layer.
        supplied = dict(confidence_inputs or {})
        resolved = {
            consequence_id: supplied.get(consequence_id, assembled.inputs)
            for consequence_id, assembled in measured.items()
        }
        resolved.update({key: value for key, value in supplied.items() if key not in resolved})

        outcome = await self._judge.arbitrate(
            change=change,
            dossiers=dossiers,
            evidence=evidence_for_judgment(dossiers, evidence=evidence or {}) + context,
            run_provenance=run_provenance,
            confidence_inputs=resolved,
            now=now,
            correlation_id=correlation_id,
        )

        return JudgmentReport(
            dossiers=dossiers,
            outcome=outcome,
            consequences=outcome.consequences,
            decisions=outcome.decisions,
            scores=outcome.scores,
            open_questions=_open_questions(hunt_questions, outcome),
            confidence_coverage=measured,
        )


def measure_confidence_inputs(
    dossiers: tuple[CandidateDossier, ...],
    *,
    outcomes: Sequence[RetrievalOutcome],
    bindings: Mapping[str, ResolvedBinding],
    as_of: dt.datetime,
) -> dict[str, AssembledInputs]:
    """Measure the confidence factors for every candidate, keyed on ``consequence_id``.

    One assembly per candidate, because the factors are properties of *that candidate's*
    supporting evidence: two findings citing different spans of the same corpus have
    different authority and different corroboration.

    Each candidate's own input records and divergences are passed through, so explicitness is
    measured over the bindings that actually underlie the claim and a finding that *reports* a
    contradiction is not penalized for the contradiction it reports.
    """
    if not outcomes:
        # Nothing to measure from. Returning nothing rather than an empty measurement, so the
        # judge scores an honest zero and `unmeasured_confidence_factors` stays silent about
        # factors nobody attempted rather than listing all of them.
        return {}
    tenant_id = _sole_tenant(outcomes, dossiers)
    return {
        dossier.consequence_id: assemble(
            supporting_evidence_ids=dossier.evidence_ids,
            outcomes=outcomes,
            consequence_class=dossier.consequence_class,
            tenant_id=tenant_id,
            as_of=dossier.as_of or as_of,
            bindings=bindings,
            own_input_record_ids=_own_records(dossier),
            divergences=dossier.divergences,
        )
        for dossier in dossiers
    }


def _sole_tenant(
    outcomes: Sequence[RetrievalOutcome], dossiers: tuple[CandidateDossier, ...]
) -> str:
    """The tenant every outcome and every Class A candidate agree on.

    Read off the *queries*, because that is the tenant the retrievers narrowed to and
    therefore the only one whose spans are in these results. Disagreement raises: confidence
    computed across two tenants would be an isolation defect surfacing as a slightly
    different number, which is the worst possible place for one.
    """
    tenants = {outcome.query.tenant_id for outcome in outcomes}
    tenants.update(d.finding.tenant_id for d in dossiers if d.finding is not None)
    if len(tenants) != 1:
        raise ValueError(
            f"confidence inputs span {len(tenants)} tenants ({sorted(tenants)}). Retrieval "
            "outcomes and Class A findings must all belong to one tenant before anything is "
            "scored over them."
        )
    return tenants.pop()


def _own_records(dossier: CandidateDossier) -> tuple[str, ...]:
    """The bindings underlying a Class A finding. A hypothesis has none of its own."""
    if dossier.finding is not None:
        return tuple(dossier.finding.input_record_ids)
    return ()


def _check_context(
    context: tuple[EvidenceForJudgment, ...],
    dossiers: tuple[CandidateDossier, ...],
) -> None:
    """Every contextual span must name at least one real candidate it bears on."""
    handles = {d.handle for d in dossiers}
    for span in context:
        unknown = set(span.handles) - handles
        if unknown or not span.handles:
            raise ValueError(
                f"contextual span {span.evidence_id} is admissible for "
                f"{sorted(unknown) or 'no candidate'}, which is not in the candidate set "
                f"{sorted(handles)}. Contextual evidence must be attached to a candidate "
                "that exists."
            )


def _open_questions(hunt_questions: Iterable[str], outcome: JudgmentOutcome) -> tuple[str, ...]:
    """Every stage's unanswered questions, in order, deduplicated.

    Collected rather than left on each stage's outcome because a reader asking "what does
    this analysis not know?" should not have to visit four objects to find out.
    """
    collected: list[str] = []
    for question in (*hunt_questions, *outcome.unanswered_questions):
        if question not in collected:
            collected.append(question)
    return tuple(collected)


def evidence_for_judgment(
    dossiers: tuple[CandidateDossier, ...],
    *,
    evidence: Mapping[str, EvidenceItem],
    limit: int = MAX_EVIDENCE_SPANS,
) -> tuple[EvidenceForJudgment, ...]:
    """Project the spans behind the candidate set, recording which candidates each serves.

    One span may be admissible for several candidates - the policy sentence a Class A finding
    rests on is often the one a Class B hypothesis cites - so ``handles`` is a tuple and the
    span is rendered once. Deduplicating is not only cheaper: the same text twice in one
    prompt invites the model to read it as two independent sources, and corroboration counts
    sources, never repetitions (``CLAUDE.md`` 16).
    """
    serves: dict[str, list[str]] = {}
    for dossier in dossiers:
        for evidence_id in sorted(dossier.citable_evidence_ids):
            serves.setdefault(evidence_id, []).append(dossier.handle)

    spans: list[EvidenceForJudgment] = []
    for evidence_id in sorted(serves):
        stored = evidence.get(evidence_id)
        if stored is None:
            # Not fetched. Filling the gap from a store would be a search this layer does not
            # perform, and the candidate is still arbitrable with less to cite.
            continue
        spans.append(
            EvidenceForJudgment(
                evidence_id=evidence_id,
                text=stored.text_span[:SPAN_CHARS],
                source_title=stored.location or "(unlabelled span)",
                authority_class=stored.metadata.get("authority_class", "unknown"),
                handles=tuple(serves[evidence_id]),
            )
        )
        if len(spans) >= limit:
            break

    return tuple(spans)
