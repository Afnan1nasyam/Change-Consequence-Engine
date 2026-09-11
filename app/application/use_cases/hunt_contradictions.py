"""Rate the deterministic divergence set, and project what survives into specialist findings.

The application seam for the contradiction-hunter. It does three things and deliberately not
a fourth:

1. project the resolver's collisions into the closed, already-classified divergence set;
2. assemble the evidence spans behind those divergences - **supplied, never fetched**;
3. run one agent call and turn what it returns into
   :class:`~app.domain.entities.findings.SpecialistFinding` records.

The fourth thing it does not do is search. Evidence arrives as an argument: either from the
participating bindings (which the resolver already named) or from a finalized retrieval the
caller performed. This layer never calls a retriever, and the agent it invokes holds none.
That is what "it may retrieve supporting evidence only when the contract requires it" looks
like when the requirement is structural: the retrieval, if any, is the caller's, and the
agent sees only its result.

**Which findings are emitted, and which are not.**

============================= =========================================================
class                         emitted as a ``SpecialistFinding``?
============================= =========================================================
``material_contradiction``    yes - a contradiction
``stale_invalidation``        yes - the consumer needs updating
``unauthorized_override``     yes - a governance defect, whatever the rated materiality
``legitimate_scoped_override`` **no** - correct behaviour is not a finding
``non_material_divergence``   **no** - already reported as a ``CouplingAnomaly``
============================= =========================================================

The bottom two rows are the ones a reader would notice being wrong. A legitimate override
reported as a defect is visibly incorrect to anyone who reads the two documents, and two
consumers disagreeing reported as a contradiction points the reader at the wrong owner - both
are already resolved deterministically, and neither needs the agent tier to say so again.

**The agent cannot suppress a finding, only annotate one.** Emission is decided by the
deterministic class. A model rating a real contradiction ``immaterial`` gets its rating
recorded on the finding and does not remove the finding, because Class A means the resolver
produced it and the resolver does not withdraw. This is why "unauthorized_override remains a
governance finding" needs no special case here: it falls out of a rule that holds for all
three emitted classes.

**Confidence has no representation.** ``SpecialistFinding.evidence_position`` is the enum an
agent may defend instead of a number, and it is derived here from two facts - whether the
finding carries evidence, and whether the agent reached a determination. Neither is a model
-supplied score.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from app.assumptions.collisions import KeyCollisions
from app.contradictions.divergence import (
    CLASS_A_CLASSES,
    DivergenceItem,
    divergences_from,
)
from app.contradictions.hunter import AssessedDivergence, ContradictionHunter, ContradictionOutcome
from app.contradictions.prompting import EvidenceForContradiction
from app.contradictions.proposals import Materiality
from app.domain.entities.change import ChangeSpec
from app.domain.entities.coupling import ResolvedBinding
from app.domain.entities.findings import SpecialistFinding
from app.domain.entities.source import EvidenceItem
from app.domain.value_objects.enums import EvidencePosition
from app.domain.value_objects.identifiers import new_id

#: Divergence classes that reach a reader as an agent-tier finding.
#:
#: **Emission follows derivation**: exactly the classes a ``DerivedRelation`` stands behind.
#: Defined as that set rather than assembled from the three names, because the two are the
#: same set and writing it as a union would make a definition look like a coincidence. The
#: two absent classes are absent because the deterministic layer already says everything
#: true about them - a legitimate override is correct behaviour, and a consumer divergence
#: is already reported as a ``CouplingAnomaly``.
REPORTED_CLASSES: Final = CLASS_A_CLASSES

#: Cap on the evidence spans one call may carry. Retrieval feeds reasoning and reasoning is
#: priced per token (``CLAUDE.md``, cost discipline). A divergence set large enough to exceed
#: this needs splitting by key, not a larger prompt.
MAX_EVIDENCE_SPANS: Final = 32

#: Truncation width for a span rendered into the prompt. Long enough to read the claim,
#: short enough that thirty of them fit.
SPAN_CHARS: Final = 600


@dataclass(frozen=True)
class ContradictionReport:
    """The deterministic divergence set, its ratings, and the findings that follow.

    All three, because they answer different questions. ``divergences`` is what the resolver
    concluded; ``outcome`` is what the agent added and what admitting it revealed;
    ``findings`` is what a reader sees. Collapsing them would lose the ability to say that
    an analysis found divergences the agent failed to rate.
    """

    divergences: tuple[DivergenceItem, ...]
    outcome: ContradictionOutcome
    findings: tuple[SpecialistFinding, ...]

    @property
    def contradictions(self) -> tuple[AssessedDivergence, ...]:
        return self.outcome.contradictions

    @property
    def governance_defects(self) -> tuple[AssessedDivergence, ...]:
        return self.outcome.governance_defects

    @property
    def suppressed(self) -> tuple[AssessedDivergence, ...]:
        """Divergences correctly kept out of the findings list.

        Named rather than merely absent: "the resolver found six divergences and two of them
        are correct behaviour" is a more useful statement than "there are four findings".
        """
        return tuple(a for a in self.outcome.assessed if a.divergence_class not in REPORTED_CLASSES)

    @property
    def rated_immaterial(self) -> tuple[AssessedDivergence, ...]:
        """Emitted findings the agent considered immaterial.

        They are still findings. This is the visible form of the agent being unable to
        suppress a deterministic conclusion.
        """
        return tuple(
            a
            for a in self.outcome.assessed
            if a.divergence_class in REPORTED_CLASSES and a.materiality is Materiality.IMMATERIAL
        )


class HuntContradictionsUseCase:
    """One contradiction-hunter run over one change's deterministic collisions.

    Holds the hunter and nothing else. In particular it holds no store and no retriever: the
    bindings and any contextual evidence are passed in by the caller, so this class cannot
    widen the corpus the agent sees.
    """

    def __init__(self, *, hunter: ContradictionHunter) -> None:
        self._hunter = hunter

    async def execute(
        self,
        *,
        change: ChangeSpec,
        collisions: Iterable[KeyCollisions],
        bindings: Mapping[str, ResolvedBinding],
        evidence: Mapping[str, EvidenceItem] | None = None,
        context: tuple[EvidenceForContradiction, ...] = (),
        correlation_id: str | None = None,
    ) -> ContradictionReport:
        """Project, rate, and report.

        ``evidence`` maps evidence id to the stored span, for the spans behind the
        participating bindings. A span the caller did not supply is simply not shown - the
        divergence is still assessed, with less to cite. That is the honest degradation: the
        alternative would be this layer reaching into a store to fill the gap, which is the
        search this agent must not perform.

        ``context`` is the narrow, contract-required route for evidence *beyond* the
        participating bindings. The agent definition asks whether an exception clause
        reconciles two artifacts, and such a clause usually sits in neither of them - so
        without this the question is unanswerable. The caller performed the retrieval and
        names, per span, which divergences it bears on; a span naming an unknown item is
        refused, because a citation admitted against a divergence that does not exist is
        exactly the fabricated-provenance case in a different costume.
        """
        divergences = divergences_from(collisions, bindings=bindings)
        _check_context(context, divergences)

        outcome = await self._hunter.assess(
            change=change,
            divergences=divergences,
            evidence=evidence_for_contradictions(
                divergences, bindings=bindings, evidence=evidence or {}
            )
            + context,
            correlation_id=correlation_id,
        )

        return ContradictionReport(
            divergences=divergences,
            outcome=outcome,
            findings=specialist_findings(outcome),
        )


def _check_context(
    context: tuple[EvidenceForContradiction, ...],
    divergences: tuple[DivergenceItem, ...],
) -> None:
    """Every contextual span must name at least one real divergence it bears on.

    A caller bug rather than a model one, so it raises rather than being dropped and
    counted: silently ignoring it would leave the agent unable to cite a span it was shown,
    and the only visible symptom would be an assessment that looks unsupported.
    """
    handles = {item.item_id for item in divergences}
    for span in context:
        unknown = set(span.item_ids) - handles
        if unknown or not span.item_ids:
            raise ValueError(
                f"contextual span {span.evidence_id} is admissible for "
                f"{sorted(unknown) or 'no divergence'}, which is not in the divergence set "
                f"{sorted(handles)}. Contextual evidence must be attached to a divergence "
                "that exists."
            )


def evidence_for_contradictions(
    divergences: tuple[DivergenceItem, ...],
    *,
    bindings: Mapping[str, ResolvedBinding],
    evidence: Mapping[str, EvidenceItem],
    limit: int = MAX_EVIDENCE_SPANS,
) -> tuple[EvidenceForContradiction, ...]:
    """Project the spans behind the divergence set, recording which item each serves.

    One span may be admissible for several divergences - the same policy sentence can
    participate in a contradiction and a staleness finding - so ``item_ids`` is a tuple and
    the span is rendered once. Deduplicating is not only cheaper: the same text twice in one
    prompt invites the model to treat it as two independent sources, and corroboration counts
    sources, never repetitions (``CLAUDE.md`` 16).
    """
    serves: dict[str, list[str]] = {}
    for item in divergences:
        for evidence_id in item.evidence_ids:
            serves.setdefault(evidence_id, []).append(item.item_id)

    holder_of = {binding.evidence_id: binding for binding in bindings.values()}

    spans: list[EvidenceForContradiction] = []
    for evidence_id in sorted(serves):
        stored = evidence.get(evidence_id)
        if stored is None:
            # Not fetched. See the use case docstring: filling this in from a store would be
            # the search this agent must not perform.
            continue
        binding = holder_of.get(evidence_id)
        spans.append(
            EvidenceForContradiction(
                evidence_id=evidence_id,
                text=stored.text_span[:SPAN_CHARS],
                source_title=binding.source_title if binding else "(unknown source)",
                authority_class=binding.authority_class.value if binding else "unknown",
                item_ids=tuple(serves[evidence_id]),
            )
        )
        if len(spans) >= limit:
            break

    return tuple(spans)


def specialist_findings(outcome: ContradictionOutcome) -> tuple[SpecialistFinding, ...]:
    """Turn the reportable divergences into agent-tier findings.

    Emission is decided by :data:`REPORTED_CLASSES` - the **deterministic** class - so the
    model's rating changes what a finding *says* and never whether it exists. The evidence
    on the finding is the divergence's own, from the participating bindings, so a finding is
    evidenced even when the model cited nothing and cannot be better evidenced because the
    model cited more.
    """
    if outcome.provenance is None:
        raise ValueError(
            "a contradiction outcome without provenance cannot produce findings; every "
            "agent-tier claim must name the prompt and binding that produced it"
        )

    findings: list[SpecialistFinding] = []
    for assessed in outcome.assessed:
        if assessed.divergence_class not in REPORTED_CLASSES:
            continue
        findings.append(
            SpecialistFinding(
                finding_id=new_id("finding"),
                agent=outcome.provenance.agent,
                claim=_claim(assessed),
                # The five-way classification reaches the reader here, unchanged from what
                # the resolver decided.
                category=assessed.divergence_class.value,
                affected_entities=(assessed.item.detail[:512],),
                evidence_ids=assessed.item.evidence_ids,
                evidence_position=_position(assessed),
                assumption_keys=(assessed.item.assumption_key,),
                produced_by=outcome.provenance,
            )
        )
    return tuple(findings)


def _claim(assessed: AssessedDivergence) -> str:
    """The finding's sentence: the resolver's conclusion first, the rating second.

    Order matters for a reader. The deterministic fact is what they can rely on; the
    materiality is an annotation on it, and putting the annotation first would invite the
    two to be read as one claim of equal standing.
    """
    item = assessed.item
    head = (
        f"{item.divergence_class.value} on `{item.assumption_key}` at {item.scope} "
        f"as of {item.as_of.date().isoformat()}: {item.detail}"
    )
    if assessed.assessment is None:
        return f"{head} Not assessed for materiality this run."

    rating = assessed.assessment
    parts = [f"Rated {rating.materiality.value}, {rating.temporal_position.value}."]
    if rating.who_is_affected:
        parts.append(f"Affects: {rating.who_is_affected}")
    if rating.reconciling_exception:
        parts.append(f"Reconciling exception: {rating.reconciling_exception}")
    if rating.authoritative_reading:
        parts.append(f"Read as governing: {rating.authoritative_reading}")
    return f"{head} " + " ".join(parts)


def _position(assessed: AssessedDivergence) -> EvidencePosition:
    """What the agent may say instead of a confidence number.

    Derived from two facts and no score: whether the finding carries evidence, and whether
    the agent reached a determination. ``UNRESOLVED`` for an item nobody rated or one rated
    ``undetermined`` - both are cases where the corpus, not the finding, is the limit.
    """
    if not assessed.item.evidence_ids:
        return EvidencePosition.UNRESOLVED
    materiality = assessed.materiality
    if materiality is None or materiality is Materiality.UNDETERMINED:
        return EvidencePosition.UNRESOLVED
    return EvidencePosition.SUPPORTED
