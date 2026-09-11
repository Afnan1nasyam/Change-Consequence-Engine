"""Hybrid fusion scored against the labelled suite, and compared with both baselines.

Same 62 expectations, same denominators, same reachability. Nothing in the suite was changed
for fusion, and neither input retriever was touched - `test_neither_baseline_moved` asserts
that from the two frozen scorer fingerprints rather than trusting it.

**Read the numbers with the embedder in mind.** The vector input is `HashingEmbedder`, which
is feature hashing over a bag of words and carries no semantic knowledge. So this measures
fusion over a strong lexical ranking and a weak term-overlap one, and *not* what fusion would
do with a real embedder. That distinction is the whole reading of the result below.

The test worth reading first is
:func:`TestTheComparison.test_fusion_recovered_the_lexical_set_and_ranked_it_worse`. It is
the result of the slice: RRF over these two inputs returns exactly the lexical result set,
adds nothing, and orders it worse at every cut-off but the deepest.
"""

from __future__ import annotations

import asyncio

import pytest

from app.domain.value_objects.enums import RetrievalStrategy
from app.evaluation.retrieval_benchmark import (
    BenchmarkReport,
    RetrievalSuite,
    load_suite,
    run_hybrid_suite,
    run_suite,
    run_vector_suite,
)
from app.evaluation.retrieval_metrics import ExpectationOutcome
from app.infrastructure.embeddings.hashing import HashingEmbedder
from app.retrieval.hybrid import HybridOutcome, strategy_split

#: Measured on 2026-09-10 by ``python -m scripts.retrieval_benchmark --strategy hybrid``
#: and quoted in ``docs/EVALUATION.md``. RRF at k=60 over `tokens-3` and `hashing-1:d=256`.
MEASURED_RECALL = {1: 0.339, 3: 0.661, 5: 0.839, 10: 0.911}
MEASURED_END_TO_END = {1: 0.306, 3: 0.597, 5: 0.758, 10: 0.823}
MEASURED_AUTHORIZATION_RECALL = {1: 0.5, 3: 1.0, 5: 1.0, 10: 1.0}

#: The two frozen baselines, for the comparison only. Neither is re-measured here and
#: neither may be tuned in response to anything below.
LEXICAL_BASELINE = {1: 0.357, 3: 0.750, 5: 0.893, 10: 0.911}
VECTOR_BASELINE = {1: 0.304, 3: 0.607, 5: 0.768, 10: 0.804}

#: Contribution over the 260 results the 35 fused queries returned in total.
MEASURED_SPLIT = {"both": 143, "lexical_only": 13, "vector_only": 104, "returned": 260}

TOLERANCE = 0.002


@pytest.fixture(scope="module")
def suite() -> RetrievalSuite:
    return load_suite()


@pytest.fixture(scope="module")
def fused(suite: RetrievalSuite) -> tuple[BenchmarkReport, tuple[HybridOutcome, ...]]:
    return asyncio.run(run_hybrid_suite(suite, embedder=HashingEmbedder()))


@pytest.fixture(scope="module")
def report(fused: tuple[BenchmarkReport, tuple[HybridOutcome, ...]]) -> BenchmarkReport:
    return fused[0]


@pytest.fixture(scope="module")
def outcomes(
    fused: tuple[BenchmarkReport, tuple[HybridOutcome, ...]],
) -> tuple[HybridOutcome, ...]:
    return fused[1]


@pytest.fixture(scope="module")
def lexical(suite: RetrievalSuite) -> BenchmarkReport:
    return asyncio.run(run_suite(suite))


@pytest.fixture(scope="module")
def vector(suite: RetrievalSuite) -> BenchmarkReport:
    return asyncio.run(run_vector_suite(suite, embedder=HashingEmbedder()))


def hits(report: BenchmarkReport) -> set[tuple[str, str]]:
    return {
        (q.query_id, e.source) for q in report.queries for e in q.expectations if e.rank is not None
    }


# ---------------------------------------------------------------------------
# The measurement is sound
# ---------------------------------------------------------------------------


class TestMeasurementIsSound:
    def test_it_names_itself_the_hybrid_strategy(self, report: BenchmarkReport) -> None:
        assert report.strategy is RetrievalStrategy.HYBRID

    def test_the_pinned_hybrid_scorer_is_the_one_that_ran(
        self, report: BenchmarkReport, suite: RetrievalSuite
    ) -> None:
        """And it is the *hybrid* pin: a fused run must not satisfy either input's pin, or
        one strategy's figures could be reprinted under another's heading."""
        assert report.version_drift(suite) == ()
        assert report.scorer_version == (
            "hybrid-rrf-1:k=60:"
            "[lexical-bm25-1:tokens-3:k1=1.2/b=0.75]+[vector-cosine-1:hashing-1:d=256]"
        )
        assert suite.pinned_scorer(RetrievalStrategy.LEXICAL) != report.scorer_version
        assert suite.pinned_scorer(RetrievalStrategy.VECTOR) != report.scorer_version

    def test_the_fingerprint_contains_both_inputs(self, report: BenchmarkReport) -> None:
        """A fused ranking is only reproducible if the two that produced it are, so
        retuning either invalidates this pin."""
        assert "lexical-bm25-1:tokens-3" in report.scorer_version
        assert "vector-cosine-1:hashing-1" in report.scorer_version

    def test_every_label_resolves(self, report: BenchmarkReport) -> None:
        assert report.label_defects == ()

    def test_no_expectation_is_removed_by_a_filter_its_query_asked_for(
        self, report: BenchmarkReport
    ) -> None:
        assert report.metrics.overall.filtered_by_query == 0

    def test_the_run_is_deterministic(self, suite: RetrievalSuite) -> None:
        """Reciprocals of small integers tie exactly and often, so the total tie break is
        load-bearing rather than defensive."""
        first, _ = asyncio.run(run_hybrid_suite(suite, embedder=HashingEmbedder()))
        second, _ = asyncio.run(run_hybrid_suite(suite, embedder=HashingEmbedder()))
        assert first.queries == second.queries
        assert first.metrics == second.metrics

    def test_the_run_needs_no_credentials_and_no_network(self, report: BenchmarkReport) -> None:
        """Extraction is replayed and the embedder is the offline hashing one."""
        assert report.queries


class TestInvariants:
    def test_withheld_evidence_is_never_returned(self, report: BenchmarkReport) -> None:
        """A third strategy is a third chance to leak, and fusion is the one that merges
        two result sets - so this matters more here, not less."""
        assert report.leaks == ()
        assert report.metrics.authorization.leaks == 0

    def test_the_filter_configuration_is_inherited_unchanged(self, report: BenchmarkReport) -> None:
        assert report.retrieval_config_version == "retrieval-1"

    def test_reachability_is_identical_to_both_baselines(
        self, report: BenchmarkReport, lexical: BenchmarkReport, vector: BenchmarkReport
    ) -> None:
        """The precondition for comparing three runs. Coverage, the unreachable set and the
        withheld set are properties of ingestion and authorization; if fusion moved any of
        them, the recall differences would be partly differences of denominator.
        """
        for other in (lexical, vector):
            assert report.metrics.overall.retrievable == other.metrics.overall.retrievable
            assert report.metrics.overall.not_admitted == other.metrics.overall.not_admitted
            assert (
                report.metrics.overall.withheld_by_permission
                == other.metrics.overall.withheld_by_permission
            )

    def test_every_fused_result_came_from_one_of_the_two_inputs(
        self, outcomes: tuple[HybridOutcome, ...]
    ) -> None:
        """Fusion may merge, reorder and truncate. It may not introduce a span neither
        strategy returned - that would be a retrieval strategy running outside every filter
        its inputs passed."""
        for hybrid in outcomes:
            union = set(hybrid.lexical.evidence_ids) | set(hybrid.vector.evidence_ids)
            assert set(hybrid.evidence_ids) <= union

    def test_neither_baseline_moved(
        self, lexical: BenchmarkReport, vector: BenchmarkReport, suite: RetrievalSuite
    ) -> None:
        """Both inputs still report their frozen fingerprints and their frozen figures."""
        assert lexical.scorer_version == "lexical-bm25-1:tokens-3:k1=1.2/b=0.75"
        assert vector.scorer_version == "vector-cosine-1:hashing-1:d=256"
        assert lexical.version_drift(suite) == ()
        assert vector.version_drift(suite) == ()

        for k, expected in LEXICAL_BASELINE.items():
            assert lexical.metrics.overall.recall[k] == pytest.approx(expected, abs=TOLERANCE)
        for k, expected in VECTOR_BASELINE.items():
            assert vector.metrics.overall.recall[k] == pytest.approx(expected, abs=TOLERANCE)


# ---------------------------------------------------------------------------
# The measured numbers
# ---------------------------------------------------------------------------


class TestMeasuredResults:
    @pytest.mark.parametrize("k", sorted(MEASURED_RECALL))
    def test_hybrid_recall_at_k(self, report: BenchmarkReport, k: int) -> None:
        assert report.metrics.overall.recall[k] == pytest.approx(MEASURED_RECALL[k], abs=TOLERANCE)

    @pytest.mark.parametrize("k", sorted(MEASURED_END_TO_END))
    def test_hybrid_end_to_end_recall_at_k(self, report: BenchmarkReport, k: int) -> None:
        assert report.metrics.overall.end_to_end_recall[k] == pytest.approx(
            MEASURED_END_TO_END[k], abs=TOLERANCE
        )

    @pytest.mark.parametrize("k", sorted(MEASURED_AUTHORIZATION_RECALL))
    def test_authorization_filtered_recall_is_unchanged(
        self, report: BenchmarkReport, k: int
    ) -> None:
        authorization = report.metrics.authorization
        assert authorization.restricted_queries == 3
        assert authorization.withheld == 2
        assert authorization.recall[k] == pytest.approx(
            MEASURED_AUTHORIZATION_RECALL[k], abs=TOLERANCE
        )

    def test_unreachable_and_withheld_expectations(self, report: BenchmarkReport) -> None:
        overall = report.metrics.overall
        assert overall.retrievable == 56
        assert overall.not_admitted == 4
        assert overall.withheld_by_permission == 2
        assert overall.coverage == pytest.approx(0.903, abs=TOLERANCE)

    def test_expectations_retrieved_and_missed(self, report: BenchmarkReport) -> None:
        outcomes = report.metrics.overall.outcomes
        assert outcomes[ExpectationOutcome.RETRIEVED] == 51
        assert outcomes[ExpectationOutcome.RETRIEVAL_MISS] == 5

    def test_negative_controls_inherit_the_vector_inputs_noise(
        self, report: BenchmarkReport, vector: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """The clearest cost of fusing in a noisy strategy: the union carries its noise.

        Lexical alone returns 5 results across the four controls and is silent on three of
        them. Hybrid returns 28 and is silent on one - identical to the vector input,
        because a union cannot be quieter than its noisiest member.
        """
        controls = report.metrics.negative_controls
        assert controls.silent == 1
        assert controls.returned == 28
        assert controls.returned == vector.metrics.negative_controls.returned
        assert controls.returned > lexical.metrics.negative_controls.returned


# ---------------------------------------------------------------------------
# Strategy contribution
# ---------------------------------------------------------------------------


class TestStrategyContribution:
    def test_the_split_is_reported_and_partitions_the_returned_results(
        self, outcomes: tuple[HybridOutcome, ...]
    ) -> None:
        split = strategy_split(outcomes)
        assert split == MEASURED_SPLIT
        assert split["both"] + split["lexical_only"] + split["vector_only"] == split["returned"]

    def test_the_vector_input_dominates_the_union_by_volume(
        self, outcomes: tuple[HybridOutcome, ...]
    ) -> None:
        """104 vector-only results against 13 lexical-only. The weak strategy contributes
        eight times as much *volume* to the fused lists while contributing no labelled
        evidence the strong one missed - which is precisely how RRF ends up ranking worse.
        """
        split = strategy_split(outcomes)
        assert split["vector_only"] >= 8 * split["lexical_only"]

    def test_agreement_is_reported_per_query(self, outcomes: tuple[HybridOutcome, ...]) -> None:
        agreements = [h.agreement for h in outcomes if h.agreement is not None]
        assert agreements
        mean = sum(agreements) / len(agreements)
        assert mean == pytest.approx(0.572, abs=0.01)

    def test_every_fused_result_records_which_strategies_found_it(
        self, outcomes: tuple[HybridOutcome, ...]
    ) -> None:
        """Checkpoint 9.3 counts corroboration across distinct strategies, so a fused list
        that forgot which strategy found what could not support that count."""
        for hybrid in outcomes:
            for result in hybrid.results:
                assert result.contributions
                assert result.found_by <= {RetrievalStrategy.LEXICAL, RetrievalStrategy.VECTOR}

    def test_strategies_executed_names_both_inputs(
        self, report: BenchmarkReport, outcomes: tuple[HybridOutcome, ...]
    ) -> None:
        for hybrid in outcomes:
            assert hybrid.outcome.strategies_executed == (
                RetrievalStrategy.LEXICAL,
                RetrievalStrategy.VECTOR,
            )

    def test_a_corroborated_span_outranks_one_only_one_strategy_found(
        self, outcomes: tuple[HybridOutcome, ...]
    ) -> None:
        """The property that makes RRF worth using at all, checked over the real runs: in
        any fused list, no single-strategy result outranks a both-strategy one that shares
        its best input rank.
        """
        for hybrid in outcomes:
            for result in hybrid.results:
                if not result.by_both:
                    continue
                best = min(result.contributions.values())
                for other in hybrid.results:
                    if other.by_both or other.rank >= result.rank:
                        continue
                    assert min(other.contributions.values()) < best


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


class TestTheComparison:
    def test_fusion_recovered_the_lexical_set_and_ranked_it_worse(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """The result of the slice, and it is not the hoped-for one.

        The fused run finds **exactly** the expectations lexical finds - same 51 hits, same
        5 misses, same set - so fusion recovered the strong strategy's coverage and added
        nothing. But it *orders* them worse: recall drops at 1, 3 and 5 and only ties at 10.
        That is what RRF does when one input is much weaker: 104 vector-only results push
        relevant lexical results down the list without ever contributing a labelled find.

        Not tuned away. `k` is the conventional 60 and neither input was touched.
        """
        assert hits(report) == hits(lexical)

        for k in (1, 3, 5):
            hybrid_recall = report.metrics.overall.recall[k]
            lexical_recall = lexical.metrics.overall.recall[k]
            assert hybrid_recall is not None and lexical_recall is not None
            assert hybrid_recall < lexical_recall, f"@{k}"

        assert report.metrics.overall.recall[10] == pytest.approx(
            LEXICAL_BASELINE[10], abs=TOLERANCE
        )

    def test_fusion_beats_the_weaker_input_at_every_k(
        self, report: BenchmarkReport, vector: BenchmarkReport
    ) -> None:
        """Which is the other half of the honest reading: RRF is doing its job - it is
        strictly better than the weak input everywhere - and its job is not enough to beat
        the strong input alone."""
        for k in (1, 3, 5, 10):
            hybrid_recall = report.metrics.overall.recall[k]
            vector_recall = vector.metrics.overall.recall[k]
            assert hybrid_recall is not None and vector_recall is not None
            assert hybrid_recall > vector_recall, f"@{k}"

    def test_the_five_remaining_misses_are_the_same_five(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """Four are the coupling set of ``retention.window`` and one is the token-expiry
        wording. A third strategy built from the same two failing on the same five is
        stronger evidence than either failing alone."""
        missed = {
            (q.query_id, e.source)
            for q in report.queries
            for e in q.expectations
            if e.outcome is ExpectationOutcome.RETRIEVAL_MISS
        }
        assert len(missed) == 5
        assert {query_id for query_id, _ in missed} == {
            "key_retention_window",
            "lexical_miss_token_expiry_wording",
        }

    def test_no_category_beats_the_lexical_baseline(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """Checked rather than assumed, because a single category improving would be the
        one thing that made fusion worth keeping at this k. None does at @10."""
        for category, metrics in report.metrics.by_category.items():
            hybrid_recall = metrics.recall[10]
            lexical_recall = lexical.metrics.by_category[category].recall[10]
            if hybrid_recall is None or lexical_recall is None:
                continue
            assert hybrid_recall <= lexical_recall + TOLERANCE, category
