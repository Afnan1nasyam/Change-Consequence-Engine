"""Hybrid retrieval: reciprocal rank fusion over the lexical and vector rankings.

**A pure function of two outcomes.** :func:`fuse` takes the
:class:`~app.domain.interfaces.retrieval.RetrievalOutcome` each strategy already produced and
returns a fused one. It does not construct a retriever, does not touch a store, does not
embed anything, and neither underlying retriever knows it exists. That is what makes every
guarantee below inherited rather than re-implemented: tenant isolation, authorization,
source-type, scope and temporal filtering, and provenance validation all happened before a
result reached this module, and fusion can only *reorder and merge* what the two strategies
were each already allowed to return.

Why reciprocal rank fusion
--------------------------

The two rankings are scored on **incomparable scales**. BM25 is an unbounded sum over term
weights; cosine is bounded in ``[-1, 1]``. Any fusion that adds or weights the *scores*
first has to normalize them, and every normalization is a decision nobody has measured:
min-max normalization is unstable when a strategy returns one result (the single result
becomes 1.0 regardless of how weak it was), and z-scores need a distribution that a 21-span
corpus does not have.

RRF sidesteps the whole problem by discarding the scores and fusing the **ranks**:

    score(d) = Σ over strategies s that returned d of 1 / (k + rank_s(d))

Three properties earn it the default:

* **Scale-free.** It never compares a BM25 score with a cosine, so there is no
  normalization to get wrong and no hidden tuning.
* **One parameter, and a conventional value.** ``k`` damps how much the very top of each
  list dominates. The 60 below is the value from the original RRF paper, and it is
  **untuned against this corpus** - like ``k1=1.2``/``b=0.75`` before it, any other number
  here would be a guess wearing a decimal point. The fingerprint travels with every
  outcome so a later tuning pass is attributable.
* **Agreement is rewarded structurally.** A span both strategies rank sums two
  reciprocals, so corroborated evidence rises above evidence only one strategy found -
  which is the behaviour that makes a second strategy worth having at all.

What fusion may and may not do
------------------------------

May: merge, reorder, and truncate to ``top_k``.

May **not**: introduce a span neither strategy returned. That is asserted rather than
assumed - the fused result set is a subset of the union of the two inputs, and a fusion that
could add a candidate would be a retrieval strategy of its own, running outside every filter
the two inputs passed.

**Strategy provenance is preserved, not flattened.** :class:`HybridResult` records the rank
each strategy gave a span, so ``lexical-only``, ``vector-only`` and ``both`` are readable off
the outcome. That is not a debugging convenience: checkpoint 9.3 counts corroboration across
*distinct strategies*, and a fused list that forgot which strategy found what could not
support that count.

Deliberately absent: reranking, query expansion, weighted or learned fusion, and any third
strategy. Each is separately measurable and none is justified by a number yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.domain.interfaces.retrieval import (
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalResult,
    RetrievalTrace,
)
from app.domain.value_objects.enums import RetrievalStrategy

#: The fusion function itself. Bumped when the arithmetic changes; ``k`` travels in the
#: fingerprint instead, so retuning it is visible without pretending the function changed.
HYBRID_FUSER: Final = "hybrid-rrf-1"

#: Fused scores are rounded before ranking, to the same precision and for the same reason as
#: the two strategies below: reciprocals of small integers produce exact ties constantly, and
#: without a rounding step plus a total tie break the order could differ between platforms.
SCORE_PRECISION: Final = 6

#: The strategies this module fuses, in the order their versions are reported.
FUSED_STRATEGIES: Final = (RetrievalStrategy.LEXICAL, RetrievalStrategy.VECTOR)

#: Filter buckets that must be identical across the two inputs. Both retrievers apply the
#: same filters in the same fixed order over the same candidate set, so a disagreement here
#: means one of them filtered differently and the fusion would be over two different corpora.
SHARED_TRACE_BUCKETS: Final = (
    "considered",
    "excluded_by_tenant",
    "excluded_by_permission",
    "excluded_by_source_type",
    "excluded_by_as_of",
    "excluded_by_scope",
    "excluded_unscoped",
)


class HybridConfig(BaseModel):
    """Fusion parameters. One, and it is untuned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The RRF damping constant. Higher means the top of each input list matters less.
    #: 60 is the value from the original paper and is **not** tuned against this corpus.
    k: int = Field(default=60, gt=0)

    def fingerprint(self, *, lexical: str, vector: str) -> str:
        """The fused scorer identity: the fuser, its parameter, and both inputs.

        Both input fingerprints are inside it because a fused ranking is only reproducible
        if the two rankings that produced it are. Changing the embedder or retuning BM25
        changes this string, which is what stops a benchmark figure being reprinted under a
        scorer that did not produce it.
        """
        return f"{HYBRID_FUSER}:k={self.k}:[{lexical}]+[{vector}]"


@dataclass(frozen=True)
class HybridResult:
    """One fused span, with the rank each strategy gave it.

    ``contributions`` is the whole point of the class: it maps a strategy to the rank *that
    strategy* assigned, so the fused list can still answer "which strategy found this?".
    A span absent from a strategy's ranking simply has no entry.
    """

    candidate: EvidenceCandidate
    score: float
    rank: int
    contributions: Mapping[RetrievalStrategy, int]

    @property
    def evidence_id(self) -> str:
        return self.candidate.evidence_id

    @property
    def found_by(self) -> frozenset[RetrievalStrategy]:
        return frozenset(self.contributions)

    @property
    def by_both(self) -> bool:
        return len(self.contributions) == len(FUSED_STRATEGIES)

    @property
    def lexical_only(self) -> bool:
        return set(self.contributions) == {RetrievalStrategy.LEXICAL}

    @property
    def vector_only(self) -> bool:
        return set(self.contributions) == {RetrievalStrategy.VECTOR}

    def rank_in(self, strategy: RetrievalStrategy) -> int | None:
        return self.contributions.get(strategy)


@dataclass(frozen=True)
class HybridOutcome:
    """The fused ranking, the standard contract, and both inputs.

    Three things travel together because each answers a different question, and collapsing
    them would lose one:

    ``outcome``
        a plain :class:`RetrievalOutcome`, so every existing consumer works unchanged and
        the fused ranking is subject to the same validators as any other.
    ``results``
        the same ranking with per-strategy provenance attached.
    ``lexical`` / ``vector``
        the inputs, unmodified, so a fused figure can always be decomposed back into the two
        it came from without re-running anything.
    """

    outcome: RetrievalOutcome
    results: tuple[HybridResult, ...]
    lexical: RetrievalOutcome
    vector: RetrievalOutcome

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(result.evidence_id for result in self.results)

    @property
    def is_empty(self) -> bool:
        return not self.results

    @property
    def scorer_version(self) -> str:
        return self.outcome.scorer_version

    # -- the counts the slice exists to produce ---------------------------

    @property
    def by_both(self) -> tuple[HybridResult, ...]:
        return tuple(result for result in self.results if result.by_both)

    @property
    def lexical_only(self) -> tuple[HybridResult, ...]:
        return tuple(result for result in self.results if result.lexical_only)

    @property
    def vector_only(self) -> tuple[HybridResult, ...]:
        return tuple(result for result in self.results if result.vector_only)

    @property
    def agreement(self) -> float | None:
        """Share of the fused list both strategies found. ``None`` when it is empty.

        Read as a diagnostic, not a quality score. Near 1.0 means the two strategies are
        returning the same evidence and the second one is adding little; near 0.0 means they
        disagree, which is either complementary coverage or one of them being wrong, and
        only the labelled benchmark can tell those apart.
        """
        if not self.results:
            return None
        return len(self.by_both) / len(self.results)


def reciprocal_rank(rank: int, *, k: int) -> float:
    """``1 / (k + rank)``. The whole of RRF's per-strategy contribution."""
    if rank < 1:
        raise ValueError(f"rank must be 1-based, got {rank}")
    return 1.0 / (k + rank)


def fuse(
    lexical: RetrievalOutcome,
    vector: RetrievalOutcome,
    *,
    config: HybridConfig | None = None,
) -> HybridOutcome:
    """Fuse two rankings of the same query into one.

    Both inputs must answer the **same query**. That is checked rather than trusted: fusing
    two different queries, or the same query under two permission scopes, would produce a
    ranking attributable to neither and - in the scope case - could surface a span the
    requester may not read. The check is the cheap version of the guarantee the two
    retrievers already enforce separately.
    """
    settings = config or HybridConfig()
    _require_same_question(lexical, vector)

    strategies = tuple(_sole_strategy(outcome) for outcome in (lexical, vector))
    if set(strategies) != set(FUSED_STRATEGIES):
        # Two rankings from one strategy is that strategy twice, not a hybrid. Fusing them
        # would overwrite one's ranks with the other's and report corroboration that never
        # happened - which is the failure mode with a consequence, since checkpoint 9.3
        # counts evidence reached by *distinct* strategies.
        raise ValueError(
            f"hybrid fusion combines {[s.value for s in FUSED_STRATEGIES]}, got "
            f"{[s.value for s in strategies]}. Two rankings from one strategy would report "
            "corroboration that did not happen."
        )

    contributions: dict[str, dict[RetrievalStrategy, int]] = {}
    candidates: dict[str, EvidenceCandidate] = {}
    scores: dict[str, float] = {}

    for outcome, strategy in zip((lexical, vector), strategies, strict=True):
        for result in outcome.results:
            evidence_id = result.evidence_id
            candidates.setdefault(evidence_id, result.candidate)
            contributions.setdefault(evidence_id, {})[strategy] = result.rank
            scores[evidence_id] = scores.get(evidence_id, 0.0) + reciprocal_rank(
                result.rank, k=settings.k
            )

    # Descending fused score, then ascending evidence_id. The second key is what makes the
    # order total: reciprocals of small integers tie exactly and often, so this is
    # load-bearing rather than defensive.
    ordered = sorted(
        ((round(score, SCORE_PRECISION), evidence_id) for evidence_id, score in scores.items()),
        key=lambda row: (-row[0], row[1]),
    )
    selected = ordered[: lexical.query.top_k]

    results = tuple(
        HybridResult(
            candidate=candidates[evidence_id],
            score=score,
            rank=rank,
            contributions=dict(sorted(contributions[evidence_id].items())),
        )
        for rank, (score, evidence_id) in enumerate(selected, start=1)
    )

    scorer_version = settings.fingerprint(
        lexical=lexical.scorer_version, vector=vector.scorer_version
    )
    outcome = RetrievalOutcome(
        query=lexical.query,
        results=tuple(
            RetrievalResult(
                candidate=result.candidate,
                # The fused ranking is what this result carries, so the strategy that
                # produced *the score* is hybrid. Which strategies found the span is in
                # `contributions`, where it can be read without being confused for this.
                strategy=RetrievalStrategy.HYBRID,
                score=result.score,
                rank=result.rank,
            )
            for result in results
        ),
        # Both strategies genuinely ran, and naming them - rather than naming `hybrid` -
        # is what keeps corroboration honest: checkpoint 9.3 counts evidence reached by
        # distinct strategies, and a single `hybrid` entry would collapse two into one.
        strategies_executed=FUSED_STRATEGIES,
        permission_scope_hash=lexical.permission_scope_hash,
        trace=_fuse_traces(lexical, vector, matched=len(ordered), returned=len(selected)),
        scorer_version=scorer_version,
    )

    return HybridOutcome(outcome=outcome, results=results, lexical=lexical, vector=vector)


def _require_same_question(lexical: RetrievalOutcome, vector: RetrievalOutcome) -> None:
    if lexical.query != vector.query:
        raise ValueError(
            "hybrid fusion requires two rankings of the same query; the inputs differ, so "
            "the fused ranking would be attributable to neither"
        )
    if lexical.permission_scope_hash != vector.permission_scope_hash:
        raise ValueError(
            "hybrid fusion requires both rankings to have been computed under the same "
            "permission scope; fusing across scopes could surface a span the requester may "
            "not read"
        )
    if lexical.retrieval_config_version != vector.retrieval_config_version:
        raise ValueError(
            "hybrid fusion requires both rankings to have been filtered by the same "
            f"retrieval configuration, got {lexical.retrieval_config_version!r} and "
            f"{vector.retrieval_config_version!r}"
        )


def _sole_strategy(outcome: RetrievalOutcome) -> RetrievalStrategy:
    """The one strategy an input ran, refusing anything already fused.

    Fusing a fused outcome would double-count the ranks inside it and produce a score no
    formula describes, so an input naming more than one strategy is rejected here rather
    than silently flattened.
    """
    executed = outcome.strategies_executed
    if len(executed) != 1:
        raise ValueError(
            f"hybrid fusion takes single-strategy inputs, got {[s.value for s in executed]}. "
            "Fusing an already-fused ranking would double-count its ranks."
        )
    strategy = executed[0]
    if strategy not in FUSED_STRATEGIES:
        raise ValueError(
            f"hybrid fusion combines {[s.value for s in FUSED_STRATEGIES]}, not {strategy.value!r}"
        )
    return strategy


def _fuse_traces(
    lexical: RetrievalOutcome,
    vector: RetrievalOutcome,
    *,
    matched: int,
    returned: int,
) -> RetrievalTrace:
    """One trace for the fused run, taking the filter buckets from the agreeing inputs.

    The filter buckets are **asserted equal** rather than summed. Both retrievers narrow the
    same candidate set to the same tenant and the same authorized slice, then apply the same
    filters in the same fixed order - so they must agree, and summing would double-count
    every excluded span and make the trace's arithmetic meaningless. A disagreement means one
    of them filtered differently, which is a defect worth failing on rather than averaging.

    ``excluded_zero_score`` is the only bucket that legitimately differs, because it is the
    one decided by *scoring*: it becomes the count of candidates that survived every filter
    and were scored by neither strategy.
    """
    shared: dict[str, int] = {}
    for bucket in SHARED_TRACE_BUCKETS:
        left = getattr(lexical.trace, bucket)
        right = getattr(vector.trace, bucket)
        if left != right:
            raise ValueError(
                f"the two strategies disagree on {bucket} ({left} vs {right}); they filter "
                "the same candidate set in the same order, so a difference means one of "
                "them filtered differently and the fusion would span two corpora"
            )
        shared[bucket] = left

    filtered = sum(value for bucket, value in shared.items() if bucket != "considered")
    survived = shared["considered"] - filtered

    return RetrievalTrace(
        **shared,
        # Candidates no strategy scored. Derived from the survivors rather than from either
        # input's own zero-score count, which describes one strategy's scoring and not the
        # union's.
        excluded_zero_score=max(0, survived - matched),
        matched=matched,
        returned=returned,
    )


def strategy_split(
    outcomes: Sequence[HybridOutcome],
) -> dict[str, int]:
    """Pool the lexical-only / vector-only / both counts over several fused runs.

    The aggregate the benchmark reports. Counted over *returned* results, so it answers
    "what did each strategy contribute to what a reader actually saw" rather than the
    different question of what each could have contributed at unbounded depth.
    """
    split = {"both": 0, "lexical_only": 0, "vector_only": 0, "returned": 0}
    for hybrid in outcomes:
        split["both"] += len(hybrid.by_both)
        split["lexical_only"] += len(hybrid.lexical_only)
        split["vector_only"] += len(hybrid.vector_only)
        split["returned"] += len(hybrid.results)
    return split
