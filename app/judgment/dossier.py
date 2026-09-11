"""One dossier per candidate: everything five upstream stages concluded about it.

The judge is the first stage that sees the whole pipeline at once, and the assembly is where
most of its correctness lives. Five inputs converge here:

============================= =========================================================
from                          contributes
============================= =========================================================
Class A resolver              the derived finding, its rule, its materialized evidence
consequence-hunter            the Class B hypotheses and their analogues
contradiction-hunter          rated divergences touching the candidate's assumption key
adversarial-challenger        the verdict, its ground, and its contradicting evidence
EvidenceGap producer          gap **ids**, as references - never as candidates
============================= =========================================================

**The identity is inherited, never minted.** A candidate that reached the challenger already
has a ``consequence_id`` (``app/challenge/candidates.py`` mints it once, because
``ChallengeResult`` is keyed on it). This module reuses that id, and
:func:`dossiers_from` refuses a ``ChallengeResult`` whose ``consequence_id`` matches no
candidate. An orphaned verdict is a proof obligation attached to nothing, and the failure
would otherwise be silent - the report would simply not mention that a finding had been
refuted.

**An EvidenceGap is never a candidate.** Type E is a claim about the *corpus*, not about the
world (checkpoint 2), which is why it can never become an edge and why it cannot become a
consequence either. Gaps arrive here and leave as ``evidence_gap_ids`` on the
:class:`~app.domain.entities.findings.Consequence` - a reference from a finding to a known
hole in the corpus. There is no constructor in this module that takes a gap and produces a
candidate, and a test asserts that adding gaps to an analysis adds no consequences.

**Class A arrives already classified, and cannot be reclassified here.** The dossier's
``consequence_class`` is derived from which slot the candidate occupies - ``finding`` or
``hypothesis`` - exactly as in the challenge layer, and the same ``__post_init__`` refusals
apply. Classification is by producer (``CLAUDE.md`` 22); a consequence does not become Class A
by being easy to explain.

**Disagreement is collected, not resolved.** Everything the upstream stages disputed is
gathered into :attr:`CandidateDossier.upstream_disagreements` and travels into the
arbitration whether the model mentions it or not. Averaging away a disagreement to produce a
cleaner report is the failure this field exists to prevent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

from app.challenge.candidates import ChallengeTarget
from app.challenge.challenger import ChallengedCandidate
from app.contradictions.hunter import AssessedDivergence
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.findings import ChallengeResult
from app.domain.value_objects.enums import (
    ChallengeOutcome,
    ConsequenceClass,
    Origin,
)

#: Bumped when this assembly changes, because it changes what the judge was shown and
#: therefore what its disposition refers to.
DOSSIER_VERSION: Final = "judgment-dossier-1"

#: The local handle the model addresses a candidate by. Short, and deliberately not a domain
#: identifier: the judge must not be able to name a ``consequence_id``, a ``relation_id`` or a
#: ``challenge_id``, because naming one is the first step to reattributing it.
HANDLE_PREFIX: Final = "j"


class DossierAssemblyError(ValueError):
    """The upstream outputs cannot be assembled as written.

    A ``ValueError`` because nothing was attempted and no model was involved. Raised rather
    than skipped in every case: each one it covers would otherwise drop an upstream
    conclusion, and a refutation nobody was told about is worse than a crash.
    """


@dataclass(frozen=True)
class CandidateDossier:
    """One candidate, and everything the pipeline concluded about it.

    Frozen. The candidate is held **by reference** to a frozen Pydantic model and is never
    constructed, copied or re-provenanced here - which is what makes "Class A provenance is
    unchanged" a property of the layer rather than a rule.
    """

    #: The local handle the model cites.
    handle: str
    #: **Inherited** from the challenge layer where one exists. Never minted here.
    consequence_id: str
    consequence_class: ConsequenceClass
    origin: Origin

    claim: str
    mechanism: str
    #: The closed set of evidence ids the judge may cite for this candidate.
    evidence_ids: tuple[str, ...]

    finding: ClassAFinding | None = None
    hypothesis: ConsequenceHypothesis | None = None

    #: The challenger's admitted verdicts on this candidate. Plural: a candidate may be
    #: challenged more than once across runs, and the judge sees all of them.
    challenges: tuple[ChallengeResult, ...] = ()
    #: Rated divergences touching this candidate's assumption key. Context for severity;
    #: the judge does not re-decide whether a divergence occurred.
    divergences: tuple[AssessedDivergence, ...] = ()
    #: References to known holes in the corpus. Never candidates.
    evidence_gap_ids: tuple[str, ...] = ()

    assumption_keys: tuple[str, ...] = ()
    affected_entity: str = ""
    historical_analogue_id: str | None = None
    rule_id: str | None = None
    as_of: dt.datetime | None = None

    #: Everything upstream disputed, gathered verbatim. Travels into the arbitration whether
    #: the judge mentions it or not (checkpoint 9: material disagreement is preserved, never
    #: resolved by averaging).
    upstream_disagreements: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if (self.finding is None) == (self.hypothesis is None):
            raise DossierAssemblyError(
                f"{self.handle}: a dossier is exactly one candidate, got "
                f"finding={'set' if self.finding else 'None'} and "
                f"hypothesis={'set' if self.hypothesis else 'None'}"
            )
        expected_origin = Origin.RESOLVER if self.finding is not None else Origin.AGENT
        if self.origin is not expected_origin:
            raise DossierAssemblyError(
                f"{self.handle}: a {'finding' if self.finding else 'hypothesis'} implies "
                f"origin={expected_origin.value!r}, got {self.origin.value!r}. "
                "Classification is by producer, not by subject matter."
            )
        expected_class = (
            ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED
            if self.origin is Origin.RESOLVER
            else ConsequenceClass.CLASS_B_BEHAVIORAL
        )
        if self.consequence_class is not expected_class:
            raise DossierAssemblyError(
                f"{self.handle}: origin={self.origin.value!r} implies class "
                f"{expected_class.value!r}, got {self.consequence_class.value!r}"
            )

    @property
    def is_class_a(self) -> bool:
        return self.origin is Origin.RESOLVER

    @property
    def was_refuted(self) -> bool:
        """Whether any admitted challenge refuted this candidate."""
        return any(c.outcome is ChallengeOutcome.REFUTED for c in self.challenges)

    @property
    def was_weakened(self) -> bool:
        return any(c.outcome is ChallengeOutcome.WEAKENED for c in self.challenges)

    @property
    def survived_challenge(self) -> bool:
        """Whether a challenger looked and found nothing.

        Reported, and deliberately **not** used to raise anything. Nobody finding contrary
        evidence is strictly weaker than evidence of correctness (checkpoint 9.2), so this
        property has no consumer in the status rules - only in the rationale a reader sees.
        """
        return any(c.outcome is ChallengeOutcome.NO_REFUTATION_FOUND for c in self.challenges)

    @property
    def challenge_evidence_ids(self) -> tuple[str, ...]:
        """Evidence the challenger cited against this candidate.

        Citable by the judge, because a disposition that could not reference the refutation
        it is acting on would be unauditable. Kept apart from ``evidence_ids`` so *support*
        and *contradiction* cannot be confused for one another.
        """
        found: list[str] = []
        for challenge in self.challenges:
            found.extend(challenge.contradicting_evidence_ids)
        return tuple(sorted(set(found)))

    @property
    def citable_evidence_ids(self) -> frozenset[str]:
        """Everything the judge may cite for this candidate, supporting or contradicting."""
        return frozenset(self.evidence_ids) | frozenset(self.challenge_evidence_ids)


def dossiers_from(
    *,
    targets: Iterable[ChallengeTarget] = (),
    challenged: Iterable[ChallengedCandidate] = (),
    class_a: Iterable[ClassAFinding] = (),
    hypotheses: Iterable[ConsequenceHypothesis] = (),
    divergences: Iterable[AssessedDivergence] = (),
    evidence_gap_ids: Iterable[str] = (),
    hunt_questions: Iterable[str] = (),
    extra_disagreements: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[CandidateDossier, ...]:
    """Assemble one dossier per candidate from the whole upstream pipeline.

    Candidates come from ``targets`` where the challenger ran, and from ``class_a`` /
    ``hypotheses`` where it did not - a stage that failed must not delete the findings it
    failed to review. Where both are supplied, the ``ChallengeTarget`` wins, because it
    carries the ``consequence_id`` the rest of the pipeline already keyed on.

    ``evidence_gap_ids`` and ``hunt_questions`` attach to **every** dossier as context. They
    are properties of the analysis rather than of one finding, and hiding them from a
    candidate because they were not obviously about it is how an uncertainty-aware report
    becomes a confident one.
    """
    resolved = _resolve_candidates(targets=targets, class_a=class_a, hypotheses=hypotheses)
    verdicts = _verdicts_by_consequence(challenged)
    _refuse_orphans(verdicts, known={c.consequence_id for c in resolved})

    gap_ids = tuple(sorted(set(evidence_gap_ids)))
    questions = tuple(hunt_questions)
    disagreements = extra_disagreements or {}

    dossiers: list[CandidateDossier] = []
    for index, candidate in enumerate(resolved, start=1):
        challenges = verdicts.get(candidate.consequence_id, ())
        relevant = tuple(
            d for d in divergences if d.item.assumption_key in candidate.assumption_keys
        )
        dossiers.append(
            CandidateDossier(
                handle=f"{HANDLE_PREFIX}{index}",
                consequence_id=candidate.consequence_id,
                consequence_class=candidate.consequence_class,
                origin=candidate.origin,
                claim=candidate.claim,
                mechanism=candidate.mechanism,
                evidence_ids=candidate.evidence_ids,
                finding=candidate.finding,
                hypothesis=candidate.hypothesis,
                challenges=challenges,
                divergences=relevant,
                evidence_gap_ids=gap_ids,
                assumption_keys=candidate.assumption_keys,
                affected_entity=candidate.affected_entity,
                historical_analogue_id=candidate.historical_analogue_id,
                rule_id=candidate.rule_id,
                as_of=candidate.as_of,
                upstream_disagreements=_disagreements(
                    challenges=challenges,
                    divergences=relevant,
                    questions=questions,
                    supplied=disagreements.get(candidate.consequence_id, ()),
                ),
            )
        )
    return tuple(dossiers)


def by_handle(dossiers: Iterable[CandidateDossier]) -> dict[str, CandidateDossier]:
    """Index dossiers by the handle the model cites. Refuses a duplicate handle."""
    indexed: dict[str, CandidateDossier] = {}
    for dossier in dossiers:
        if dossier.handle in indexed:
            raise DossierAssemblyError(
                f"duplicate dossier handle {dossier.handle!r}; a handle must address one "
                "candidate or a disposition cannot be attributed"
            )
        indexed[dossier.handle] = dossier
    return indexed


# ---------------------------------------------------------------------------
# Assembly internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """A candidate with its identity settled, before the dossier is built around it."""

    consequence_id: str
    consequence_class: ConsequenceClass
    origin: Origin
    claim: str
    mechanism: str
    evidence_ids: tuple[str, ...]
    assumption_keys: tuple[str, ...]
    affected_entity: str
    finding: ClassAFinding | None = None
    hypothesis: ConsequenceHypothesis | None = None
    historical_analogue_id: str | None = None
    rule_id: str | None = None
    as_of: dt.datetime | None = None


def _resolve_candidates(
    *,
    targets: Iterable[ChallengeTarget],
    class_a: Iterable[ClassAFinding],
    hypotheses: Iterable[ConsequenceHypothesis],
) -> tuple[_Candidate, ...]:
    """Candidates from the challenge layer first, then anything it never saw.

    A finding present in both is taken from the ``ChallengeTarget``, so its established
    ``consequence_id`` survives. A finding present only in ``class_a`` gets one minted here -
    the challenger never ran on it, so nothing downstream is keyed on it yet.
    """
    resolved: list[_Candidate] = []
    seen_findings: set[str] = set()
    seen_hypotheses: set[str] = set()

    for target in targets:
        if target.finding is not None:
            seen_findings.add(target.finding.finding_id)
            resolved.append(_from_finding(target.finding, consequence_id=target.consequence_id))
        else:
            assert target.hypothesis is not None  # guaranteed by ChallengeTarget
            seen_hypotheses.add(target.hypothesis.hypothesis_id)
            resolved.append(
                _from_hypothesis(target.hypothesis, consequence_id=target.consequence_id)
            )

    for finding in sorted(class_a, key=lambda f: f.finding_id):
        if finding.finding_id not in seen_findings:
            resolved.append(_from_finding(finding, consequence_id=None))
    for hypothesis in sorted(hypotheses, key=lambda h: h.hypothesis_id):
        if hypothesis.hypothesis_id not in seen_hypotheses:
            resolved.append(_from_hypothesis(hypothesis, consequence_id=None))

    return tuple(resolved)


def _from_finding(finding: ClassAFinding, *, consequence_id: str | None) -> _Candidate:
    from app.domain.value_objects.identifiers import new_id

    return _Candidate(
        consequence_id=consequence_id or new_id("consequence"),
        consequence_class=ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED,
        origin=Origin.RESOLVER,
        claim=finding.summary,
        # The derivation, stated as the mechanism. Not a model's account of one: a Class A
        # mechanism is the rule that fired over the records it fired on.
        mechanism=(
            f"rule {finding.rule_id} over {len(finding.input_record_ids)} bindings of "
            f"{finding.assumption_key} at {finding.scope.render()}"
        ),
        evidence_ids=finding.materialized_evidence_ids,
        assumption_keys=(finding.assumption_key,),
        affected_entity=finding.affected_artifact,
        finding=finding,
        rule_id=finding.rule_id,
        as_of=finding.as_of,
    )


def _from_hypothesis(
    hypothesis: ConsequenceHypothesis, *, consequence_id: str | None
) -> _Candidate:
    from app.domain.value_objects.identifiers import new_id

    return _Candidate(
        consequence_id=consequence_id or new_id("consequence"),
        consequence_class=ConsequenceClass.CLASS_B_BEHAVIORAL,
        origin=Origin.AGENT,
        claim=hypothesis.claim,
        mechanism=hypothesis.mechanism,
        evidence_ids=hypothesis.evidence_ids,
        assumption_keys=tuple(hypothesis.assumption_keys),
        affected_entity=(
            hypothesis.affected_entities[0] if hypothesis.affected_entities else "(unnamed)"
        ),
        hypothesis=hypothesis,
        historical_analogue_id=hypothesis.historical_analogue_id,
    )


def _verdicts_by_consequence(
    challenged: Iterable[ChallengedCandidate],
) -> dict[str, tuple[ChallengeResult, ...]]:
    grouped: dict[str, list[ChallengeResult]] = {}
    for candidate in challenged:
        if candidate.result is not None:
            grouped.setdefault(candidate.result.consequence_id, []).append(candidate.result)
    return {key: tuple(value) for key, value in grouped.items()}


def _refuse_orphans(
    verdicts: Mapping[str, tuple[ChallengeResult, ...]], *, known: set[str]
) -> None:
    """A verdict keyed on a consequence no candidate carries is refused.

    The concrete failure this prevents: the challenge layer mints a ``consequence_id`` and the
    judge mints a second one for the same finding, so the refutation silently attaches to
    nothing and the report shows a refuted finding as unchallenged. Raising here makes the
    id-inheritance requirement enforced rather than documented.
    """
    orphans = sorted(set(verdicts) - known)
    if orphans:
        raise DossierAssemblyError(
            f"{len(orphans)} challenge verdict(s) name a consequence no candidate carries: "
            f"{orphans}. A verdict keyed on an unknown consequence is a proof obligation "
            "attached to nothing - the candidate's established consequence_id must be reused "
            "rather than minted a second time."
        )


def _disagreements(
    *,
    challenges: tuple[ChallengeResult, ...],
    divergences: tuple[AssessedDivergence, ...],
    questions: tuple[str, ...],
    supplied: tuple[str, ...],
) -> tuple[str, ...]:
    """Gather everything upstream disputed about this candidate, verbatim.

    Collected by the runtime rather than left to the judge to remember, because the whole
    point is that a disagreement survives a stage that would prefer a cleaner report. What
    goes in: every refutation and narrowing, every divergence the contradiction-hunter rated,
    and the questions the hunters could not answer.
    """
    collected: list[str] = []

    for challenge in challenges:
        if challenge.outcome is ChallengeOutcome.REFUTED and challenge.contradicts_what:
            ground = f" [{challenge.class_a_ground.value}]" if challenge.class_a_ground else ""
            collected.append(f"challenger refuted{ground}: {challenge.contradicts_what}")
        elif challenge.outcome is ChallengeOutcome.WEAKENED and challenge.scope_narrowing:
            ground = f" [{challenge.class_a_ground.value}]" if challenge.class_a_ground else ""
            collected.append(f"challenger narrowed{ground}: {challenge.scope_narrowing}")

    for divergence in divergences:
        materiality = divergence.materiality.value if divergence.materiality else "unrated"
        collected.append(
            f"{divergence.divergence_class.value} on {divergence.item.assumption_key} "
            f"(rated {materiality}): {divergence.item.detail}"
        )

    collected.extend(f"unanswered: {question}" for question in questions)
    collected.extend(supplied)

    # Truncated to the field's width rather than dropped: a disagreement a reader cannot see
    # in full is still better than one they are never told about.
    return tuple(item[:512] for item in collected)
