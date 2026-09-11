"""Hybrid fusion: the arithmetic, the invariants, and what fusion must not do.

:func:`~app.retrieval.hybrid.fuse` is a pure function of two outcomes, so almost everything
here is testable by constructing rankings - no store, no embedder, no corpus. That is the
point of the design: fusion inherits every guarantee rather than re-implementing one.

The tests carrying the most weight, each guarding something that would be invisible
downstream:

* :func:`TestFusionCannotWiden.test_the_fused_set_is_a_subset_of_the_union` - a fusion that
  could add a candidate would be a retrieval strategy running outside every filter its
  inputs passed.
* :func:`TestSameQuestion.test_two_scopes_cannot_be_fused` - fusing across permission scopes
  could surface a span the requester may not read.
* :func:`TestTraceFusion.test_disagreeing_filter_buckets_are_refused` - the two strategies
  filter the same candidate set in the same order, so a disagreement means the fusion spans
  two corpora.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from app.domain.interfaces.retrieval import (
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalResult,
    RetrievalTrace,
)
from app.domain.value_objects.enums import RetrievalStrategy, SourceType
from app.domain.value_objects.permissions import PermissionScope
from app.retrieval.hybrid import (
    FUSED_STRATEGIES,
    HYBRID_FUSER,
    SCORE_PRECISION,
    HybridConfig,
    fuse,
    reciprocal_rank,
    strategy_split,
)
from tests.fixtures.retrieval import OPEN_SCOPE, TENANT, candidate

REFUND = "Refunds of 10,000 USD or more require manager approval before release."
RETENTION = "Personal data is retained for 90 days after the account is closed."
EU = "For EU customers we keep personal data for 30 days after closure."
TTL = "auth.reset_token.ttl_hours: 24"

LEXICAL_VERSION = "lexical-bm25-1:tokens-3:k1=1.2/b=0.75"
VECTOR_VERSION = "vector-cosine-1:hashing-1:d=256"


def spans() -> dict[str, EvidenceCandidate]:
    """Four stable candidates, keyed by a short label."""
    return {
        "refund": candidate(REFUND, n=1),
        "retention": candidate(RETENTION, n=2),
        "eu": candidate(EU, n=3, source_type=SourceType.KNOWLEDGE_DOC, scopes=("region:EU",)),
        "ttl": candidate(TTL, n=4, source_type=SourceType.CONFIG),
    }


def query(**over: Any) -> RetrievalQuery:
    over.setdefault("permission_scope", OPEN_SCOPE)
    over.setdefault("tenant_id", TENANT)
    over.setdefault("text", "refunds personal data")
    return RetrievalQuery(**over)


def trace(*, matched: int, returned: int, **over: int) -> RetrievalTrace:
    fields: dict[str, int] = {"considered": 4, "matched": matched, "returned": returned}
    fields.update(over)
    return RetrievalTrace(**fields)


def outcome(
    strategy: RetrievalStrategy,
    ranked: list[EvidenceCandidate],
    *,
    retrieval_query: RetrievalQuery | None = None,
    scorer: str | None = None,
    zero_score: int | None = None,
    **trace_over: int,
) -> RetrievalOutcome:
    """One strategy's ranking, with descending synthetic scores.

    The scores are arbitrary and deliberately on different scales for the two strategies -
    fusion must not read them, and a test that gave both the same scale could not tell.
    """
    resolved = retrieval_query if retrieval_query is not None else query()
    # Two deliberately different scales - an unbounded BM25-shaped sum and a bounded
    # cosine-shaped similarity - and both strictly positive, as the contract requires.
    # Fusion must not read either, and a test giving both the same scale could not tell.
    top = 12.5 if strategy is RetrievalStrategy.LEXICAL else 0.91
    remaining = 4 - len(ranked) if zero_score is None else zero_score
    return RetrievalOutcome(
        query=resolved,
        results=tuple(
            RetrievalResult(
                candidate=span,
                strategy=strategy,
                score=(top - index) if strategy is RetrievalStrategy.LEXICAL else top / (index + 1),
                rank=index + 1,
            )
            for index, span in enumerate(ranked)
        ),
        strategies_executed=(strategy,),
        permission_scope_hash=resolved.permission_scope_hash,
        trace=trace(
            matched=len(ranked),
            returned=len(ranked),
            excluded_zero_score=max(0, remaining),
            **trace_over,
        ),
        scorer_version=scorer
        or (LEXICAL_VERSION if strategy is RetrievalStrategy.LEXICAL else VECTOR_VERSION),
    )


def lexical(ranked: list[EvidenceCandidate], **over: Any) -> RetrievalOutcome:
    return outcome(RetrievalStrategy.LEXICAL, ranked, **over)


def vector(ranked: list[EvidenceCandidate], **over: Any) -> RetrievalOutcome:
    return outcome(RetrievalStrategy.VECTOR, ranked, **over)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


class TestReciprocalRank:
    def test_the_contribution_falls_as_rank_grows(self) -> None:
        assert reciprocal_rank(1, k=60) > reciprocal_rank(2, k=60) > reciprocal_rank(10, k=60)

    def test_k_damps_how_much_the_top_dominates(self) -> None:
        """The one parameter, and what it does. A small ``k`` makes rank 1 overwhelming; a
        large one flattens the list."""
        sharp = reciprocal_rank(1, k=1) / reciprocal_rank(10, k=1)
        flat = reciprocal_rank(1, k=1000) / reciprocal_rank(10, k=1000)
        assert sharp > flat

    def test_a_contribution_is_always_positive(self) -> None:
        """Which is why a fused score satisfies ``RetrievalResult.score > 0`` for free."""
        assert reciprocal_rank(100, k=60) > 0.0

    def test_a_zero_or_negative_rank_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            reciprocal_rank(0, k=60)


class TestFusedScoring:
    def test_a_span_both_strategies_rank_sums_two_reciprocals(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))

        expected = round(2 * reciprocal_rank(1, k=60), SCORE_PRECISION)
        assert hybrid.results[0].score == expected

    def test_agreement_outranks_a_single_strategys_first_place(self) -> None:
        """The behaviour that makes a second strategy worth having: corroborated evidence
        rises above evidence only one strategy found."""
        s = spans()
        hybrid = fuse(
            lexical([s["retention"], s["refund"]]),
            vector([s["refund"], s["eu"]]),
        )
        assert hybrid.results[0].evidence_id == s["refund"].evidence_id
        assert hybrid.results[0].by_both

    def test_the_input_scores_are_never_read(self) -> None:
        """RRF fuses ranks, not scores - which is what lets it combine an unbounded BM25
        sum with a bounded cosine without normalizing either.

        Multiplying one side's scores by a thousand must change nothing.
        """
        s = spans()
        ranked = [s["refund"], s["retention"]]
        baseline = fuse(lexical(ranked), vector(list(reversed(ranked))))

        inflated = lexical(ranked)
        inflated = inflated.model_copy(
            update={
                "results": tuple(
                    result.model_copy(update={"score": result.score * 1000.0})
                    for result in inflated.results
                )
            }
        )
        assert fuse(inflated, vector(list(reversed(ranked)))).evidence_ids == baseline.evidence_ids

    def test_scores_are_rounded_before_ranking(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["retention"]]))
        assert all(r.score == round(r.score, SCORE_PRECISION) for r in hybrid.results)

    def test_k_is_configurable_and_travels_in_the_fingerprint(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]), config=HybridConfig(k=10))

        assert "k=10" in hybrid.scorer_version
        assert hybrid.results[0].score == round(2 * reciprocal_rank(1, k=10), SCORE_PRECISION)

    def test_the_default_k_is_the_conventional_untuned_one(self) -> None:
        """60 is the value from the original RRF paper, and it is not tuned here."""
        assert HybridConfig().k == 60


class TestDeterminism:
    def test_two_fusions_of_the_same_inputs_are_identical(self) -> None:
        s = spans()
        left, right = lexical([s["refund"], s["eu"]]), vector([s["retention"], s["refund"]])
        assert fuse(left, right).evidence_ids == fuse(left, right).evidence_ids

    def test_an_exact_tie_is_broken_by_evidence_id(self) -> None:
        """Reciprocals of small integers tie exactly and often, so the tie break is
        load-bearing here rather than defensive."""
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["retention"]]))

        assert hybrid.results[0].score == hybrid.results[1].score
        assert hybrid.evidence_ids == tuple(sorted(hybrid.evidence_ids))

    def test_ranks_are_dense_and_ascending(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"], s["eu"]]), vector([s["retention"], s["ttl"]]))
        assert [r.rank for r in hybrid.results] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# What fusion may not do
# ---------------------------------------------------------------------------


class TestFusionCannotWiden:
    def test_the_fused_set_is_a_subset_of_the_union(self) -> None:
        """A fusion that could add a candidate would be a retrieval strategy of its own,
        running outside every filter the two inputs passed."""
        s = spans()
        left, right = lexical([s["refund"]]), vector([s["retention"], s["eu"]])
        hybrid = fuse(left, right)

        union = set(left.evidence_ids) | set(right.evidence_ids)
        assert set(hybrid.evidence_ids) <= union

    def test_a_span_neither_strategy_returned_is_absent(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))
        assert s["ttl"].evidence_id not in hybrid.evidence_ids

    def test_two_empty_rankings_fuse_to_nothing(self) -> None:
        """Not an error, and not a fallback to either input: nothing matched."""
        hybrid = fuse(lexical([]), vector([]))
        assert hybrid.is_empty
        assert hybrid.outcome.is_empty

    def test_one_empty_ranking_yields_the_other(self) -> None:
        s = spans()
        hybrid = fuse(lexical([]), vector([s["refund"]]))

        assert hybrid.evidence_ids == (s["refund"].evidence_id,)
        assert hybrid.results[0].vector_only

    def test_the_fused_list_is_truncated_to_top_k(self) -> None:
        s = spans()
        limited = query(top_k=2)
        hybrid = fuse(
            lexical([s["refund"], s["retention"]], retrieval_query=limited),
            vector([s["eu"], s["ttl"]], retrieval_query=limited),
        )
        assert len(hybrid.results) == 2
        assert hybrid.outcome.trace.truncated

    def test_a_candidate_is_carried_by_reference_not_rebuilt(self) -> None:
        """So a citation taken from a fused result is the one the admission gate verified."""
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))
        assert hybrid.results[0].candidate is s["refund"]


class TestSameQuestion:
    def test_two_different_queries_cannot_be_fused(self) -> None:
        s = spans()
        with pytest.raises(ValueError, match="same query"):
            fuse(
                lexical([s["refund"]], retrieval_query=query(text="refunds")),
                vector([s["refund"]], retrieval_query=query(text="retention")),
            )

    def test_two_scopes_cannot_be_fused(self) -> None:
        """Fusing across permission scopes could surface a span the requester may not
        read - the one failure here with a security consequence."""
        s = spans()
        granted = query(permission_scope=PermissionScope.of("privacy-office"))
        with pytest.raises(ValueError, match="same query"):
            fuse(
                lexical([s["refund"]], retrieval_query=query()),
                vector([s["refund"]], retrieval_query=granted),
            )

    def test_an_already_fused_input_is_refused(self) -> None:
        """Fusing a fused ranking would double-count the ranks inside it and produce a
        score no formula describes."""
        s = spans()
        once = fuse(lexical([s["refund"]]), vector([s["refund"]]))
        with pytest.raises(ValueError, match="double-count"):
            fuse(once.outcome, vector([s["refund"]]))

    def test_two_rankings_from_the_same_strategy_are_refused(self) -> None:
        """Hybrid means lexical *and* vector. Two lexical runs are one strategy twice, and
        fusing them would report corroboration that did not happen."""
        s = spans()
        with pytest.raises(ValueError, match="corroboration that did not happen"):
            fuse(lexical([s["refund"]]), lexical([s["retention"]]))

    def test_two_vector_rankings_are_refused_too(self) -> None:
        s = spans()
        with pytest.raises(ValueError, match="corroboration that did not happen"):
            fuse(vector([s["refund"]]), vector([s["retention"]]))


# ---------------------------------------------------------------------------
# Strategy provenance
# ---------------------------------------------------------------------------


class TestStrategyProvenance:
    def test_each_result_records_the_rank_each_strategy_gave_it(self) -> None:
        s = spans()
        hybrid = fuse(
            lexical([s["retention"], s["refund"]]),
            vector([s["refund"], s["eu"]]),
        )
        fused = {r.evidence_id: r for r in hybrid.results}
        refund = fused[s["refund"].evidence_id]

        assert refund.rank_in(RetrievalStrategy.LEXICAL) == 2
        assert refund.rank_in(RetrievalStrategy.VECTOR) == 1

    def test_the_three_categories_partition_the_fused_list(self) -> None:
        s = spans()
        hybrid = fuse(
            lexical([s["refund"], s["retention"]]),
            vector([s["refund"], s["eu"]]),
        )
        assert len(hybrid.by_both) == 1
        assert len(hybrid.lexical_only) == 1
        assert len(hybrid.vector_only) == 1
        assert len(hybrid.by_both) + len(hybrid.lexical_only) + len(hybrid.vector_only) == len(
            hybrid.results
        )

    def test_lexical_only_names_the_lexical_strategy_and_nothing_else(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["retention"]]), vector([s["refund"]]))
        only = hybrid.lexical_only[0]

        assert only.found_by == {RetrievalStrategy.LEXICAL}
        assert only.rank_in(RetrievalStrategy.VECTOR) is None
        assert not only.by_both

    def test_agreement_is_the_share_found_by_both(self) -> None:
        s = spans()
        hybrid = fuse(
            lexical([s["refund"], s["retention"]]),
            vector([s["refund"], s["eu"]]),
        )
        assert hybrid.agreement == pytest.approx(1 / 3)

    def test_agreement_over_an_empty_list_is_unknown_not_zero(self) -> None:
        assert fuse(lexical([]), vector([])).agreement is None

    def test_both_inputs_are_kept_for_decomposition(self) -> None:
        """So a fused figure can always be taken apart into the two it came from without
        re-running anything."""
        s = spans()
        left, right = lexical([s["refund"]]), vector([s["retention"]])
        hybrid = fuse(left, right)

        assert hybrid.lexical is left
        assert hybrid.vector is right

    def test_the_split_aggregates_over_several_runs(self) -> None:
        s = spans()
        runs = [
            fuse(lexical([s["refund"]]), vector([s["refund"]])),
            fuse(lexical([s["retention"]]), vector([s["eu"]])),
        ]
        assert strategy_split(runs) == {
            "both": 1,
            "lexical_only": 1,
            "vector_only": 1,
            "returned": 3,
        }


# ---------------------------------------------------------------------------
# The contract it hands downstream
# ---------------------------------------------------------------------------


class TestTheStandardOutcome:
    def test_it_produces_a_valid_retrieval_outcome(self) -> None:
        """So every existing consumer works unchanged and the fused ranking is subject to
        the same validators as any other."""
        s = spans()
        hybrid = fuse(lexical([s["refund"], s["eu"]]), vector([s["refund"], s["ttl"]]))

        assert isinstance(hybrid.outcome, RetrievalOutcome)
        assert hybrid.outcome.evidence_ids == hybrid.evidence_ids

    def test_every_result_names_the_hybrid_strategy(self) -> None:
        """Because the fused *score* is what the result carries. Which strategies found the
        span lives in ``contributions``, where it cannot be confused for this."""
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))
        assert all(r.strategy is RetrievalStrategy.HYBRID for r in hybrid.outcome.results)

    def test_strategies_executed_names_both_inputs_not_hybrid(self) -> None:
        """Checkpoint 9.3 counts evidence reached by *distinct* strategies. A single
        ``hybrid`` entry would collapse two into one and understate corroboration."""
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))

        assert hybrid.outcome.strategies_executed == FUSED_STRATEGIES
        assert RetrievalStrategy.HYBRID not in hybrid.outcome.strategies_executed

    def test_the_scorer_version_contains_the_fuser_and_both_inputs(self) -> None:
        """A fused ranking is only reproducible if the two that produced it are."""
        s = spans()
        version = fuse(lexical([s["refund"]]), vector([s["refund"]])).scorer_version

        assert HYBRID_FUSER in version
        assert LEXICAL_VERSION in version
        assert VECTOR_VERSION in version

    def test_the_permission_scope_hash_is_carried_through(self) -> None:
        s = spans()
        scoped = query(permission_scope=PermissionScope.of("privacy-office"))
        hybrid = fuse(
            lexical([s["refund"]], retrieval_query=scoped),
            vector([s["refund"]], retrieval_query=scoped),
        )
        assert hybrid.outcome.permission_scope_hash == scoped.permission_scope_hash

    def test_the_filter_configuration_is_inherited_unchanged(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))
        assert hybrid.outcome.retrieval_config_version == "retrieval-1"


class TestTraceFusion:
    def test_the_shared_filter_buckets_are_carried_not_summed(self) -> None:
        """Summing would double-count every excluded span and make the arithmetic
        meaningless: both strategies excluded the same ones."""
        s = spans()
        hybrid = fuse(
            lexical([s["refund"]], excluded_by_permission=1, excluded_by_as_of=1),
            vector([s["refund"]], excluded_by_permission=1, excluded_by_as_of=1),
        )
        assert hybrid.outcome.trace.excluded_by_permission == 1
        assert hybrid.outcome.trace.excluded_by_as_of == 1

    def test_disagreeing_filter_buckets_are_refused(self) -> None:
        """The invariant that proves the two retrievers filter identically. A difference
        means one of them filtered differently and the fusion spans two corpora."""
        s = spans()
        with pytest.raises(ValueError, match="disagree on excluded_by_permission"):
            fuse(
                lexical([s["refund"]], excluded_by_permission=1),
                vector([s["refund"]], excluded_by_permission=0),
            )

    def test_matched_is_the_size_of_the_union(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"], s["eu"]]), vector([s["refund"], s["ttl"]]))
        assert hybrid.outcome.trace.matched == 3

    def test_zero_score_counts_the_candidates_neither_strategy_scored(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["retention"]]))

        # 4 considered, none filtered, 2 in the union -> 2 scored by neither.
        assert hybrid.outcome.trace.excluded_zero_score == 2

    def test_the_buckets_still_sum_to_considered(self) -> None:
        """What makes the trace usable as negative provenance for an ``EvidenceGap``."""
        s = spans()
        hybrid = fuse(
            lexical([s["refund"]], excluded_by_as_of=1),
            vector([s["refund"], s["eu"]], excluded_by_as_of=1),
        )
        fused = hybrid.outcome.trace
        assert fused.excluded + fused.matched == fused.considered


class TestFilteringIsInherited:
    """Fusion applies no filter of its own, and does not need to.

    Every filter ran twice - once per strategy - before :func:`fuse` saw anything. These
    assert the consequence: a span excluded by either strategy's filters is simply not in
    the input, so it cannot appear in the output.
    """

    def test_an_excluded_span_cannot_be_reintroduced(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))

        for excluded in ("retention", "eu", "ttl"):
            assert s[excluded].evidence_id not in hybrid.evidence_ids

    def test_a_withheld_span_stays_withheld(self) -> None:
        """Authorization is applied inside each retriever, before fusion exists. A span
        neither returned cannot be fused into the result."""
        restricted = query(permission_scope=OPEN_SCOPE)
        hybrid = fuse(
            lexical([], retrieval_query=restricted, excluded_by_permission=2),
            vector([], retrieval_query=restricted, excluded_by_permission=2),
        )
        assert hybrid.is_empty
        assert hybrid.outcome.trace.excluded_by_permission == 2

    def test_every_fused_candidate_still_carries_its_registration_authority(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"], s["eu"]]), vector([s["eu"], s["refund"]]))
        assert all(
            r.candidate.authority_class is r.candidate.document.authority_class
            for r in hybrid.results
        )

    def test_offsets_still_verify_on_a_fused_result(self) -> None:
        s = spans()
        hybrid = fuse(lexical([s["refund"]]), vector([s["refund"]]))
        item = hybrid.results[0].candidate.evidence
        assert item.text_span[item.char_start : item.char_end] == item.text_span


class TestConfigMismatch:
    def test_two_filter_configurations_cannot_be_fused(self) -> None:
        s = spans()
        stale = lexical([s["refund"]]).model_copy(
            update={"retrieval_config_version": "retrieval-0"}
        )
        with pytest.raises(ValueError, match=r"retrieval configuration"):
            fuse(stale, vector([s["refund"]]))


def test_neither_retriever_was_modified() -> None:
    """The fingerprints of both inputs are unchanged by this slice.

    Stated as a test because the whole value of a fused figure rests on the two baselines
    it decomposes into still being the frozen ones.
    """
    from app.retrieval.lexical import LexicalConfig
    from app.retrieval.vectors import VectorConfig

    assert LexicalConfig().fingerprint() == LEXICAL_VERSION
    assert VectorConfig(embedder="hashing-1", dimensions=256).fingerprint() == VECTOR_VERSION


def test_a_scoped_query_is_not_something_fusion_can_widen() -> None:
    """A temporal filter, seen from fusion: the excluded span is absent from both inputs."""
    s = spans()
    dated = query(as_of=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    hybrid = fuse(
        lexical([s["refund"]], retrieval_query=dated, excluded_by_as_of=1),
        vector([s["refund"]], retrieval_query=dated, excluded_by_as_of=1),
    )
    assert hybrid.evidence_ids == (s["refund"].evidence_id,)
    assert hybrid.outcome.trace.excluded_by_as_of == 1
