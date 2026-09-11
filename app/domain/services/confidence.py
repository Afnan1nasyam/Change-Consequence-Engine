"""The deterministic confidence function. The only place a confidence number is produced.

Checkpoint 9.1 fixes the eight factors and their directions; checkpoint 13.17 requires the
function to be versioned and its input type to **exclude agent identity**. This module fixes
the arithmetic and pins it behind ``CONFIDENCE_FN_VERSION``.

**What the signature makes impossible.** :func:`compute` takes a :class:`ConfidenceInputs`
and nothing else, and that type has no field for an agent name, an agent count, a model, a
vote, a level of agreement between agents, or a model-supplied number. So the function cannot
read any of them - the enforcement is structural rather than a rule somebody has to remember.
Corroboration counts **distinct source versions reached by distinct retrieval strategies**
(checkpoint 9.3); several agents reading one span are one source, and this type cannot even
express "several agents".

**Why a function at all, rather than the judge's number.** A model asked for a confidence
value produces a fluent one, and a fluent number is indistinguishable from a calibrated one
at the point a reader sees it. Severity is the judge's because business impact is not in the
corpus; confidence is not, because every one of its inputs is a measurable property of the
evidence (checkpoint 9.4). That asymmetry is recorded in three places so it is not "fixed" in
the wrong direction, and this is the third.

**The weights are not tuned.** They encode the orderings the checkpoint states and nothing
more; there is no calibration data yet. That is exactly why the version label exists, and why
``docs/IDEA.md`` 13 forbids claiming a probability: this number is an ordering aid, not a
likelihood, and the absence of a ``likelihood`` field anywhere in the domain is deliberate.

**Two inputs subtract, and neither is a veto.** A live contradiction and inference depth both
reduce confidence without zeroing it, because a contradicted claim is still a claim and a
third-order effect is still an effect. The gates in this system are elsewhere - status gating
refuses an unsupported claim outright, and a unit mismatch blocks coupling entirely - and
putting a gate here as well would let one weak factor silently delete a finding.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.domain.entities.findings import ConsequenceScore
from app.domain.value_objects.rules import CONFIDENCE_FN_VERSION

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Reference = Annotated[str, StringConstraints(min_length=1, max_length=120)]

#: Weights over the additive factors. Sum to 1.0 so the result needs no rescaling.
#:
#: Corroboration is weighted highest because it is the one factor that cannot be faked by a
#: single confident document, and it is the factor the whole assumption-coupling mechanism
#: exists to make measurable.
_WEIGHTS: Final[dict[str, float]] = {
    "source_authority": 0.25,
    "corroboration": 0.30,
    "explicitness": 0.20,
    "source_freshness": 0.15,
    "retrieval_agreement": 0.10,
}

#: Corroboration saturates here. Three independent source versions is a strong signal; the
#: fourth adds little, and a linear scale would let a widely-copied statement outscore a
#: genuinely triangulated one.
_CORROBORATION_SATURATION: Final = 3

#: A live contradicting declaration scales confidence down by at most this much. Not a veto:
#: a contradicted claim is still a claim, and the contradiction is reported separately as its
#: own finding.
_CONTRADICTION_FLOOR: Final = 0.5

#: Each derivation hop away from ``directly_changes`` costs this share of what remains.
#: Compounding rather than linear, because uncertainty compounds along a chain.
_DEPTH_DECAY: Final = 0.15

#: Depth beyond this is treated as equally distant. Past a few hops the difference between
#: five and six is not something this function can claim to measure.
_MAX_DEPTH: Final = 4

#: Alias mediation penalty (checkpoint 5.7, 9.1). An approved alias is trusted enough to
#: produce Class A, but a direct key match is still the stronger signal.
_ALIAS_FACTOR: Final = 0.85

#: Precision for the reported number. Six places is far more than the function's accuracy
#: warrants; it exists so two runs over identical inputs compare equal rather than differing
#: in float noise.
_PRECISION: Final = 6


class ConfidenceInputs(BaseModel):
    """Measurable properties of the evidence behind one consequence. Nothing else.

    Every field is a property of *documents and spans*. There is deliberately no field for
    which agent reported the finding, how many agents agreed, which model ran, or what any
    of them thought the confidence was - and ``extra="forbid"`` makes an attempt to add one
    a validation error rather than an ignored key.

    That is the structural form of checkpoint 9.3: this type cannot count agents, so the
    function cannot be talked into treating agreement as corroboration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Holder authority of the supporting sources, normalized over ``AUTHORITY_RANK``.
    source_authority: Unit = 0.0
    #: Recency of the supporting versions, normalized by the caller against its horizon.
    source_freshness: Unit = 0.0
    #: Binding explicitness, normalized over ``EXPLICITNESS_RANK``. An explicit statement
    #: outranks a value the extractor had to infer from text.
    explicitness: Unit = 0.0

    #: The distinct source versions supporting the claim. **Versions, not agents**, and
    #: counted as a set: the same version cited twice is one source.
    corroborating_source_version_ids: tuple[Reference, ...] = ()
    #: The distinct retrieval strategies that reached them. Two strategies agreeing on one
    #: version is weaker than two versions, and this is what keeps them apart.
    corroborating_strategies: tuple[Reference, ...] = ()
    #: Agreement between retrieval strategies' scores, where the caller can measure it.
    retrieval_agreement: Unit = 0.0

    #: Strength of a live divergent declaring binding. Subtracts.
    contradiction_strength: Unit = 0.0
    #: Derivation hops from ``directly_changes``. Subtracts.
    inference_depth: int = Field(default=0, ge=0)
    #: Whether the coupling was reached through an approved ``KeyAlias`` rather than a direct
    #: key match. Subtracts.
    alias_mediated: bool = False

    @model_validator(mode="after")
    def _corroboration_needs_a_strategy_to_have_reached_it(self) -> Self:
        """A source version nothing retrieved cannot corroborate anything.

        Checkpoint 9.3 counts distinct ``(source_version, strategy)`` reach, so a caller
        claiming versions with no strategy has not measured corroboration - it has asserted
        it. Refused here rather than scored, because the alternative is a number whose
        largest input was never observed.
        """
        if self.corroborating_source_version_ids and not self.corroborating_strategies:
            raise ValueError(
                "corroborating source versions were supplied with no retrieval strategy that "
                "reached them. Corroboration counts distinct source versions reached by "
                "distinct strategies; a version nothing retrieved corroborates nothing."
            )
        return self

    @property
    def corroboration_count(self) -> int:
        """Distinct source versions. Never a count of agreeing agents."""
        return len(set(self.corroborating_source_version_ids))

    @property
    def strategy_count(self) -> int:
        return len(set(self.corroborating_strategies))


def compute(
    inputs: ConfidenceInputs,
    *,
    consequence_id: str,
    computed_at: dt.datetime,
) -> ConsequenceScore:
    """Score one consequence from measurable evidence properties.

    Returns the domain :class:`ConsequenceScore`, which persists ``factor_inputs`` so the
    number can be recomputed and explained after the fact - a score whose inputs were
    discarded is an opinion with a decimal point.
    """
    corroboration = _corroboration(inputs)

    factors = {
        "source_authority": inputs.source_authority,
        "corroboration": corroboration,
        "explicitness": inputs.explicitness,
        "source_freshness": inputs.source_freshness,
        "retrieval_agreement": inputs.retrieval_agreement,
    }
    base = sum(_WEIGHTS[name] * value for name, value in factors.items())

    contradiction_multiplier = 1.0 - (1.0 - _CONTRADICTION_FLOOR) * inputs.contradiction_strength
    depth_multiplier = (1.0 - _DEPTH_DECAY) ** min(inputs.inference_depth, _MAX_DEPTH)
    alias_multiplier = _ALIAS_FACTOR if inputs.alias_mediated else 1.0

    confidence = base * contradiction_multiplier * depth_multiplier * alias_multiplier

    return ConsequenceScore(
        consequence_id=consequence_id,
        confidence=round(min(1.0, max(0.0, confidence)), _PRECISION),
        confidence_fn_version=CONFIDENCE_FN_VERSION,
        # Every input, including the ones that only subtract, so the score is explainable
        # rather than merely reproducible.
        factor_inputs={
            **{name: round(value, _PRECISION) for name, value in factors.items()},
            "contradiction_multiplier": round(contradiction_multiplier, _PRECISION),
            "depth_multiplier": round(depth_multiplier, _PRECISION),
            "alias_multiplier": alias_multiplier,
            "corroborating_versions": float(inputs.corroboration_count),
            "corroborating_strategies": float(inputs.strategy_count),
        },
        corroborating_source_version_ids=inputs.corroborating_source_version_ids,
        corroborating_strategies=inputs.corroborating_strategies,
        computed_at=computed_at,
    )


def _corroboration(inputs: ConfidenceInputs) -> float:
    """Corroboration as a saturating function of distinct versions, gated by strategy reach.

    The ``min`` against the strategy count is the load-bearing part. Five source versions all
    reached by one strategy is one strategy's view of the corpus, and treating it as
    five-fold corroboration is how a lexical retriever's blind spot becomes a confident
    number (checkpoint 9.3).
    """
    versions = inputs.corroboration_count
    if versions == 0:
        return 0.0
    effective = min(versions, max(1, inputs.strategy_count) * _CORROBORATION_SATURATION)
    return min(1.0, effective / _CORROBORATION_SATURATION)
