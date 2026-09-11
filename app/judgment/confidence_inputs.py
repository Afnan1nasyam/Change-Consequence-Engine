"""Measure the confidence factors from pipeline data. Deterministic, and it counts no agents.

:mod:`app.domain.services.confidence` has existed with nothing measuring its inputs, so every
score was an honest zero. This module is the measurement, and it changes neither the factor
list nor the weights: checkpoint 9.1 fixes the factors, the function fixes the arithmetic, and
this fixes only *where each number comes from*.

============================= ==========================================================
factor                        measured from
============================= ==========================================================
``source_authority``          holder ``authority_class`` of the supporting candidates
``source_freshness``          ``SourceVersion.effective_from`` against the analysis date
``explicitness``              ``AssumptionRecord.explicitness`` of the candidate's bindings
``corroborating_source_version_ids`` distinct source versions among the supporting spans
``corroborating_strategies``  the strategies whose outcome actually returned one
``retrieval_agreement``       share of supporting spans every executed strategy reached
``contradiction_strength``    the strongest *other* divergence on the same assumption key
``inference_depth``           1 for a derivation, 2 for a model-inferred hypothesis
``alias_mediated``            caller-supplied; nothing records it yet (see ``STATUS.md``)
============================= ==========================================================

Four decisions carry the weight here.

**Agent identity cannot enter, structurally.** :func:`assemble` has no parameter for an agent,
an agent count, a model, or a level of agreement between agents - and
:class:`~app.domain.services.confidence.ConfidenceInputs` has no field one would fit. So the
enforcement of checkpoint 9.3 is that the *data does not exist at this boundary*, not that
somebody remembered not to pass it. Corroboration counts sources; several agents reading one
span are one source, and neither this function nor the one it feeds can tell how many agents
there were.

**Only evidence the requester actually received is measured.** The supporting spans are
matched against the :class:`~app.domain.interfaces.retrieval.RetrievalOutcome` results, which
the retrievers already narrowed by tenant and by permission *before* scoring. An id no outcome
returned contributes nothing - so a span the requester may not read cannot raise their
confidence, and a narrower scope can only ever produce a lower or equal number. This module
touches no store, which is what makes that a property rather than a promise.

**A candidate's own divergence does not penalize it.** For a Class A finding that *reports* a
contradiction, the divergence is the evidence, not a reason to doubt the claim. So
``contradiction_strength`` is measured from divergences whose participants are **not** the
candidate's own input records. Without that exclusion every contradiction finding would be
penalized for the thing it found.

**Score-level agreement across strategies is not measurable, and pretending otherwise would
be worse than the alternative.** A BM25 score and a cosine similarity are not on a common
scale; normalizing them would invent a comparison the scorers do not support. So
``retrieval_agreement`` is *reach* agreement - the share of supporting spans that every
executed strategy returned - which is a real, deterministic reading of "the strategies agree"
and is honest about what it measures.

**What could not be measured is reported rather than defaulted silently.**
:class:`FactorCoverage` records how much of each factor the supplied data could reach, because
a 0.0 that means "measured as absent" and a 0.0 that means "nothing to measure" are different
facts about an analysis and a reader comparing two runs needs to tell them apart.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from app.contradictions.divergence import DivergenceClass
from app.contradictions.hunter import AssessedDivergence
from app.domain.entities.coupling import ResolvedBinding
from app.domain.interfaces.retrieval import EvidenceCandidate, RetrievalOutcome
from app.domain.services.confidence import ConfidenceInputs
from app.domain.value_objects.enums import (
    AUTHORITY_RANK,
    EXPLICITNESS_RANK,
    ConsequenceClass,
)

#: Bumped when a factor's *source* changes, which is separate from the function version:
#: measuring authority differently changes the number without changing the arithmetic, and
#: two runs differing only in that must still be diffable.
CONFIDENCE_INPUTS_VERSION: Final = "confidence-inputs-1"

#: Age at which a source version scores zero freshness. Five years, matching the coupling
#: scorer's horizon so the two do not disagree about what "old" means. Clamped rather than
#: unbounded: a corpus of mostly ancient documents should not make every finding equally
#: fresh by comparison.
FRESHNESS_HORIZON: Final = dt.timedelta(days=365 * 5)

#: Derivation hops from ``directly_changes`` (checkpoint 9.1, inference depth, negative).
#:
#: A Class A finding is one hop: the resolver derived it from bindings by a named rule. A
#: Class B hypothesis is two: it is inferred *beyond* the evidence layer, and the extra hop is
#: exactly the thing that makes it a hypothesis. Counted from the producer rather than parsed
#: out of a mechanism string, because a chain rendered as prose is not a measurement.
INFERENCE_DEPTH: Final[dict[ConsequenceClass, int]] = {
    ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED: 1,
    ConsequenceClass.CLASS_B_BEHAVIORAL: 2,
}

#: How much each divergence class counts against a *different* claim on the same key.
#:
#: The checkpoint's factor is "presence and strength of a divergent declaring binding". A
#: material contradiction is that case exactly. A stale consumer and an unauthorized override
#: both involve a divergent binding that does **not** govern, so they weigh less. The two
#: remaining classes are not contradictions at all - the checkpoint says so in as many words -
#: and weigh nothing.
CONTRADICTION_WEIGHTS: Final[dict[DivergenceClass, float]] = {
    DivergenceClass.MATERIAL_CONTRADICTION: 1.0,
    DivergenceClass.STALE_INVALIDATION: 0.5,
    DivergenceClass.UNAUTHORIZED_OVERRIDE: 0.5,
    DivergenceClass.LEGITIMATE_SCOPED_OVERRIDE: 0.0,
    DivergenceClass.NON_MATERIAL_DIVERGENCE: 0.0,
}

_MAX_AUTHORITY: Final = max(AUTHORITY_RANK.values())
_MAX_EXPLICITNESS: Final = max(EXPLICITNESS_RANK.values())

#: Precision for the reported factors, so two runs over identical inputs compare equal rather
#: than differing in float noise.
_PRECISION: Final = 6


class ConfidenceAssemblyError(ValueError):
    """The supplied pipeline data cannot be measured as written.

    A ``ValueError`` because nothing was attempted and no model was involved. Raised rather
    than skipped for a tenant mismatch: silently dropping a cross-tenant candidate would turn
    an isolation defect into a slightly lower confidence number, which is the worst possible
    place for it to surface.
    """


@dataclass(frozen=True)
class FactorCoverage:
    """How much of each factor the supplied data could reach.

    Reported because a factor measured as absent and a factor nothing could measure are
    different facts, and a bare 0.0 cannot tell them apart. On a corpus where no source states
    an effective date, ``source_freshness`` is 0.0 for every finding - and a reader needs to
    know that is the corpus rather than the findings.
    """

    supporting_ids: int
    #: Supporting ids that some executed strategy actually returned. The rest were not in the
    #: requester's authorized result set and contributed nothing.
    matched_candidates: int
    distinct_versions: int
    strategies_executed: int
    versions_dated: int
    bindings_supplied: int

    @property
    def unmatched_ids(self) -> int:
        """Cited ids no strategy returned for this requester."""
        return max(0, self.supporting_ids - self.matched_candidates)

    @property
    def unmeasured(self) -> tuple[str, ...]:
        """Factors the supplied data could not reach at all. Ordered, for stable reporting."""
        missing: list[str] = []
        if self.matched_candidates == 0:
            missing.extend(("source_authority", "source_freshness", "corroboration"))
        elif self.versions_dated == 0:
            missing.append("source_freshness")
        if self.bindings_supplied == 0:
            missing.append("explicitness")
        if self.strategies_executed < 2:
            # One strategy cannot agree with itself, so there is nothing to measure - which
            # is different from two strategies measured as disagreeing.
            missing.append("retrieval_agreement")
        return tuple(missing)

    @property
    def fully_measured(self) -> bool:
        return not self.unmeasured


@dataclass(frozen=True)
class AssembledInputs:
    """The measured factors, and what the measurement could not reach."""

    inputs: ConfidenceInputs
    coverage: FactorCoverage
    assembler_version: str = CONFIDENCE_INPUTS_VERSION


def assemble(
    *,
    supporting_evidence_ids: Sequence[str],
    outcomes: Sequence[RetrievalOutcome],
    consequence_class: ConsequenceClass,
    tenant_id: str,
    as_of: dt.datetime,
    bindings: Mapping[str, ResolvedBinding] | None = None,
    own_input_record_ids: Sequence[str] = (),
    divergences: Iterable[AssessedDivergence] = (),
    alias_mediated: bool = False,
) -> AssembledInputs:
    """Measure the confidence factors for one consequence.

    ``supporting_evidence_ids`` are the ids the judge's disposition *admitted* - already
    filtered to what was offered for that candidate. ``outcomes`` is one
    :class:`RetrievalOutcome` per executed strategy, and it is the only source of candidates:
    an id no outcome returned is counted as unmatched and contributes nothing, because the
    retrievers narrowed by tenant and permission before scoring and this function must not
    reach past that.

    ``own_input_record_ids`` are the candidate's own bindings, used twice: to scope
    ``explicitness`` to the bindings that actually underlie this claim, and to exclude the
    candidate's own divergence from ``contradiction_strength`` - a finding that *reports* a
    contradiction must not be penalized for the thing it found.

    Note what this signature does not accept: no agent, no agent count, no model, no measure
    of agreement between agents, and no confidence. Checkpoint 9.3 is enforced by the data not
    existing here.
    """
    reach = _reach(outcomes, tenant_id=tenant_id)
    matched = {
        evidence_id: reach[evidence_id]
        for evidence_id in dict.fromkeys(supporting_evidence_ids)
        if evidence_id in reach
    }

    candidates = tuple(candidate for candidate, _ in matched.values())
    versions = _distinct_versions(candidates)
    strategies = _strategies_that_returned(matched)
    executed = tuple(
        dict.fromkeys(
            strategy.value for outcome in outcomes for strategy in outcome.strategies_executed
        )
    )
    own_bindings = _own_bindings(bindings or {}, own_input_record_ids)

    coverage = FactorCoverage(
        supporting_ids=len(dict.fromkeys(supporting_evidence_ids)),
        matched_candidates=len(matched),
        distinct_versions=len(versions),
        strategies_executed=len(executed),
        versions_dated=sum(1 for version in versions.values() if version is not None),
        bindings_supplied=len(own_bindings),
    )

    inputs = ConfidenceInputs(
        source_authority=_authority(candidates),
        source_freshness=_freshness(versions, as_of=as_of),
        explicitness=_explicitness(own_bindings),
        corroborating_source_version_ids=tuple(sorted(versions)),
        corroborating_strategies=strategies,
        retrieval_agreement=_agreement(matched, executed=executed),
        contradiction_strength=_contradiction(divergences, own=own_input_record_ids),
        inference_depth=INFERENCE_DEPTH[consequence_class],
        alias_mediated=alias_mediated,
    )
    return AssembledInputs(inputs=inputs, coverage=coverage)


# ---------------------------------------------------------------------------
# Reach: what the requester actually received, per strategy
# ---------------------------------------------------------------------------


def _reach(
    outcomes: Sequence[RetrievalOutcome], *, tenant_id: str
) -> dict[str, tuple[EvidenceCandidate, frozenset[str]]]:
    """Index every returned span by id, with the strategies that returned it.

    The candidates come from the outcomes and nowhere else, so this cannot see a span the
    retrievers filtered out. A cross-tenant candidate raises: the retriever already re-checks
    tenant on every candidate, and a second check here costs nothing and turns an isolation
    defect into a loud failure rather than a marginally different score.
    """
    found: dict[str, tuple[EvidenceCandidate, set[str]]] = {}
    for outcome in outcomes:
        strategies = {strategy.value for strategy in outcome.strategies_executed}
        for result in outcome.results:
            candidate = result.candidate
            if candidate.tenant_id != tenant_id:
                raise ConfidenceAssemblyError(
                    f"candidate {candidate.evidence_id} belongs to tenant "
                    f"{candidate.tenant_id} but is being scored for {tenant_id}. Confidence "
                    "must never be computed across a tenant boundary."
                )
            existing = found.get(candidate.evidence_id)
            if existing is None:
                found[candidate.evidence_id] = (candidate, set(strategies))
            else:
                existing[1].update(strategies)
    return {
        evidence_id: (candidate, frozenset(strategies))
        for evidence_id, (candidate, strategies) in found.items()
    }


def _distinct_versions(
    candidates: Iterable[EvidenceCandidate],
) -> dict[str, dt.datetime | None]:
    """Distinct supporting source versions, mapped to their effective date.

    Keyed on ``source_version_id`` alone. Checkpoint 9.3 counts distinct
    ``(source_version_id, source_type)`` pairs, and a version belongs to exactly one document
    with exactly one source type - so the version id already determines the pair and adding
    the type would only make the key longer.

    Two spans from one document collapse to one entry, which is the whole point: a statement
    repeated twice in one policy is one source, not two.
    """
    versions: dict[str, dt.datetime | None] = {}
    for candidate in candidates:
        versions[candidate.source_version_id] = candidate.version.effective_from
    return versions


def _strategies_that_returned(
    matched: Mapping[str, tuple[EvidenceCandidate, frozenset[str]]],
) -> tuple[str, ...]:
    """Strategies that returned at least one *supporting* span.

    Not the strategies that ran. A strategy that executed and returned nothing relevant has
    not corroborated anything, and counting it would let an empty second strategy widen the
    corroboration gate for free.
    """
    reached: set[str] = set()
    for _candidate, strategies in matched.values():
        reached.update(strategies)
    return tuple(sorted(reached))


# ---------------------------------------------------------------------------
# The factors
# ---------------------------------------------------------------------------


def _authority(candidates: Sequence[EvidenceCandidate]) -> float:
    """Mean holder authority over the supporting spans, normalized.

    Mean rather than max, matching ``coupling_strength.compute``: a claim supported by a
    policy *and* a chat message is weaker than one supported by the policy alone, and taking
    the maximum would let one strong citation carry a pile of weak ones.
    """
    if not candidates:
        return 0.0
    total = sum(AUTHORITY_RANK[candidate.authority_class] for candidate in candidates)
    return round(total / (len(candidates) * _MAX_AUTHORITY), _PRECISION)


def _freshness(versions: Mapping[str, dt.datetime | None], *, as_of: dt.datetime) -> float:
    """Mean recency over distinct supporting versions, against the horizon.

    Distinct *versions*, not spans: two quotations from one document describe one document's
    age. A version with no effective date is excluded from the mean rather than scored zero -
    scoring it would report an undated document as stale, which is a claim the corpus does not
    support. When no version is dated the factor is 0.0 and
    :attr:`FactorCoverage.unmeasured` says so.

    A version effective *after* ``as_of`` scores full freshness rather than negative age: it
    is not yet in force, which the temporal filters own, and it is certainly not stale.
    """
    dated = [effective for effective in versions.values() if effective is not None]
    if not dated:
        return 0.0
    scores = []
    for effective in dated:
        age = as_of - effective
        if age <= dt.timedelta(0):
            scores.append(1.0)
        else:
            scores.append(max(0.0, 1.0 - age / FRESHNESS_HORIZON))
    return round(sum(scores) / len(scores), _PRECISION)


def _explicitness(bindings: Sequence[ResolvedBinding]) -> float:
    """Mean binding explicitness, normalized.

    An explicit statement outranks a value the extractor had to infer from surrounding text
    (checkpoint 3.4). Measured over the candidate's *own* bindings, so an unrelated explicit
    binding elsewhere in the corpus cannot raise it.
    """
    if not bindings:
        return 0.0
    total = sum(EXPLICITNESS_RANK[binding.explicitness] for binding in bindings)
    return round(total / (len(bindings) * _MAX_EXPLICITNESS), _PRECISION)


def _agreement(
    matched: Mapping[str, tuple[EvidenceCandidate, frozenset[str]]],
    *,
    executed: Sequence[str],
) -> float:
    """Share of supporting spans that **every** executed strategy returned.

    Reach agreement rather than score agreement, and deliberately so: a BM25 score and a
    cosine similarity are not on a common scale, so comparing them needs a normalization
    neither scorer supports and any number produced that way would describe the
    normalization. Whether two independent strategies both found a span is a real, checkable
    fact about retrieval reliability.

    Zero when fewer than two strategies ran, because one strategy cannot agree with itself -
    and :attr:`FactorCoverage.unmeasured` reports that rather than letting the zero read as
    disagreement.
    """
    if len(executed) < 2 or not matched:
        return 0.0
    required = set(executed)
    unanimous = sum(1 for _candidate, strategies in matched.values() if required <= strategies)
    return round(unanimous / len(matched), _PRECISION)


def _contradiction(divergences: Iterable[AssessedDivergence], *, own: Sequence[str]) -> float:
    """The strongest divergence weighing against this claim, excluding its own.

    The exclusion is the important part. A Class A finding that reports a contradiction has
    that contradiction as its *evidence*; penalizing it for the divergence it found would
    invert the factor. So a divergence whose participants are the candidate's own input
    records is skipped, and only a divergence about the same key from elsewhere counts.
    """
    own_records = set(own)
    strengths = [
        CONTRADICTION_WEIGHTS[divergence.divergence_class]
        for divergence in divergences
        if not own_records or not set(divergence.item.participants) <= own_records
    ]
    return max(strengths, default=0.0)


def _own_bindings(
    bindings: Mapping[str, ResolvedBinding], own: Sequence[str]
) -> tuple[ResolvedBinding, ...]:
    """The candidate's own bindings, in a stable order.

    Falls back to every supplied binding when the caller named none, so a caller that has the
    bindings but not the mapping still gets an explicitness measurement rather than a silent
    zero.
    """
    if own:
        return tuple(bindings[record_id] for record_id in own if record_id in bindings)
    return tuple(bindings[key] for key in sorted(bindings))
