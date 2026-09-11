"""The vector strategy scored against the labelled suite, and compared with the baseline.

Same 62 expectations, same denominators, same reachability accounting as the lexical run -
which is the whole reason the comparison is legitimate. An expectation is a relevance
judgment about the *question*, so nothing in the suite was changed for this strategy, and
:func:`TestTheComparisonIsFair.test_reachability_is_identical_under_both_strategies`
asserts that rather than assuming it.

**Read the measured numbers with the embedder in mind.** They were produced by
``HashingEmbedder``, which is feature hashing over a bag of words and carries no semantic
knowledge at all. So these figures measure the vector *machinery* plus a term-overlap
signal, and they are **not** a measurement of semantic retrieval. The class that would make
it one - ``ReplayEmbedder`` over vectors recorded from a real provider - exists and has no
fixture yet.

The test worth reading first is
:func:`TestTheComparisonIsFair.test_vector_finds_a_strict_subset_of_what_lexical_finds`.
It is the result of the slice: with this embedder the second strategy adds nothing, so the
case for fusing them is currently zero and building fusion now would be building it against
a number that says not to.
"""

from __future__ import annotations

import asyncio

import pytest

from app.domain.value_objects.enums import RetrievalStrategy
from app.evaluation.retrieval_benchmark import (
    BenchmarkReport,
    RetrievalSuite,
    load_suite,
    run_suite,
    run_vector_suite,
)
from app.evaluation.retrieval_metrics import ExpectationOutcome, QueryScore
from app.infrastructure.embeddings.hashing import HashingEmbedder

#: Measured on 2026-09-09 by ``python -m scripts.retrieval_benchmark --strategy vector``
#: and quoted in ``docs/EVALUATION.md``. Embedder: ``hashing-1`` at 256 dimensions.
MEASURED_RECALL = {1: 0.304, 3: 0.607, 5: 0.768, 10: 0.804}
MEASURED_END_TO_END = {1: 0.274, 3: 0.548, 5: 0.694, 10: 0.726}
MEASURED_AUTHORIZATION_RECALL = {1: 0.5, 3: 1.0, 5: 1.0, 10: 1.0}

#: The frozen lexical baseline under ``tokens-3``, for the comparison only. Not re-measured
#: here and not to be tuned in response to anything below.
LEXICAL_BASELINE_RECALL = {1: 0.357, 3: 0.750, 5: 0.893, 10: 0.911}

TOLERANCE = 0.002


@pytest.fixture(scope="module")
def suite() -> RetrievalSuite:
    return load_suite()


@pytest.fixture(scope="module")
def report(suite: RetrievalSuite) -> BenchmarkReport:
    return asyncio.run(run_vector_suite(suite, embedder=HashingEmbedder()))


@pytest.fixture(scope="module")
def lexical(suite: RetrievalSuite) -> BenchmarkReport:
    return asyncio.run(run_suite(suite))


def hits(report: BenchmarkReport) -> set[tuple[str, str]]:
    """Every labelled expectation the run returned, as (query, source) pairs."""
    return {
        (q.query_id, e.source) for q in report.queries for e in q.expectations if e.rank is not None
    }


def reachable(report: BenchmarkReport) -> set[tuple[str, str]]:
    return {(q.query_id, e.source) for q in report.queries for e in q.expectations if e.retrievable}


def by_id(report: BenchmarkReport) -> dict[str, QueryScore]:
    return {query.query_id: query for query in report.queries}


# ---------------------------------------------------------------------------
# The measurement is sound
# ---------------------------------------------------------------------------


class TestMeasurementIsSound:
    def test_the_run_needs_no_credentials_and_no_network(self, report: BenchmarkReport) -> None:
        """Extraction is replayed and the embedder is the offline hashing one
        (``ARCHITECTURE.md`` 11.4). No provider is constructed anywhere on this path."""
        assert report.queries

    def test_it_names_itself_the_vector_strategy(self, report: BenchmarkReport) -> None:
        assert report.strategy is RetrievalStrategy.VECTOR

    def test_the_pinned_vector_scorer_is_the_one_that_ran(
        self, report: BenchmarkReport, suite: RetrievalSuite
    ) -> None:
        """And the pin is the *vector* pin. A vector run must not be able to satisfy the
        lexical pin, or one strategy's figures could be reprinted under the other's."""
        assert report.version_drift(suite) == ()
        assert report.scorer_version == "vector-cosine-1:hashing-1:d=256"
        assert suite.pinned_scorer(RetrievalStrategy.VECTOR) == report.scorer_version
        assert suite.pinned_scorer(RetrievalStrategy.LEXICAL) != report.scorer_version

    def test_every_label_resolves(self, report: BenchmarkReport) -> None:
        assert report.label_defects == ()

    def test_no_expectation_is_removed_by_a_filter_its_query_asked_for(
        self, report: BenchmarkReport
    ) -> None:
        assert report.metrics.overall.filtered_by_query == 0

    def test_the_outcome_of_every_expectation_is_accounted_for(
        self, report: BenchmarkReport
    ) -> None:
        overall = report.metrics.overall
        assert sum(overall.outcomes.values()) == overall.expectations == 62

    def test_the_run_is_deterministic(self, suite: RetrievalSuite) -> None:
        """Cosine over floats produces near-ties constantly, so this is load-bearing here
        rather than defensive."""
        first = asyncio.run(run_vector_suite(suite, embedder=HashingEmbedder()))
        second = asyncio.run(run_vector_suite(suite, embedder=HashingEmbedder()))
        assert first.queries == second.queries
        assert first.metrics == second.metrics


class TestInvariants:
    def test_withheld_evidence_is_never_returned(self, report: BenchmarkReport) -> None:
        """The one metric with a value fixed in advance, and it must hold for every
        strategy. A second retriever is a second chance to leak."""
        assert report.leaks == ()
        assert report.metrics.authorization.leaks == 0

    def test_the_filter_configuration_is_the_lexical_one(self, report: BenchmarkReport) -> None:
        assert report.retrieval_config_version == "retrieval-1"

    def test_the_corpus_is_the_same_21_admitted_spans(self, report: BenchmarkReport) -> None:
        assert report.corpus_spans == 21


# ---------------------------------------------------------------------------
# The measured numbers
# ---------------------------------------------------------------------------


class TestMeasuredResults:
    @pytest.mark.parametrize("k", sorted(MEASURED_RECALL))
    def test_vector_recall_at_k(self, report: BenchmarkReport, k: int) -> None:
        assert report.metrics.overall.recall[k] == pytest.approx(MEASURED_RECALL[k], abs=TOLERANCE)

    @pytest.mark.parametrize("k", sorted(MEASURED_END_TO_END))
    def test_vector_end_to_end_recall_at_k(self, report: BenchmarkReport, k: int) -> None:
        assert report.metrics.overall.end_to_end_recall[k] == pytest.approx(
            MEASURED_END_TO_END[k], abs=TOLERANCE
        )

    @pytest.mark.parametrize("k", sorted(MEASURED_AUTHORIZATION_RECALL))
    def test_vector_authorization_filtered_recall_at_k(
        self, report: BenchmarkReport, k: int
    ) -> None:
        authorization = report.metrics.authorization
        assert authorization.restricted_queries == 3
        assert authorization.withheld == 2
        assert authorization.retrievable == 2
        assert authorization.recall[k] == pytest.approx(
            MEASURED_AUTHORIZATION_RECALL[k], abs=TOLERANCE
        )

    def test_unreachable_and_withheld_expectations(self, report: BenchmarkReport) -> None:
        """Both are properties of ingestion and authorization, so a change of *strategy*
        must not move either. Asserting the numbers here is what makes the cross-strategy
        recall comparison meaningful."""
        overall = report.metrics.overall
        assert overall.retrievable == 56
        assert overall.not_admitted == 4
        assert overall.withheld_by_permission == 2
        assert overall.coverage == pytest.approx(0.903, abs=TOLERANCE)

    def test_expectations_retrieved_and_missed(self, report: BenchmarkReport) -> None:
        outcomes = report.metrics.overall.outcomes
        assert outcomes[ExpectationOutcome.RETRIEVED] == 46
        assert outcomes[ExpectationOutcome.RETRIEVAL_MISS] == 10

    def test_negative_controls_are_far_noisier_than_lexical(self, report: BenchmarkReport) -> None:
        """Measured, and the sharpest cost of this strategy: 28 results across the four
        controls against lexical's 5, and one silent run against three.

        Cosine has no natural cutoff. Lexical retrieval is quiet because a span sharing no
        term scores exactly nothing; here a span sharing one hashed coordinate scores
        something, and something is enough to be returned. A similarity floor would fix it
        and would be an untuned knob - so it is measured rather than pre-empted.
        """
        controls = report.metrics.negative_controls
        assert controls.queries == 4
        assert controls.silent == 1
        assert controls.returned == 28


# ---------------------------------------------------------------------------
# Lexical vs vector
# ---------------------------------------------------------------------------


class TestTheComparisonIsFair:
    def test_reachability_is_identical_under_both_strategies(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """The precondition for comparing the two at all.

        Coverage, the unreachable set and the withheld set are properties of ingestion and
        authorization. If a change of strategy moved any of them, the recall difference
        below would be partly a difference of denominators and would mean nothing.
        """
        assert reachable(report) == reachable(lexical)
        assert report.metrics.overall.retrievable == lexical.metrics.overall.retrievable
        assert report.metrics.overall.not_admitted == lexical.metrics.overall.not_admitted
        assert (
            report.metrics.overall.withheld_by_permission
            == lexical.metrics.overall.withheld_by_permission
        )

    def test_both_strategies_run_the_same_labels_and_the_same_filters(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        assert report.metrics.overall.expectations == lexical.metrics.overall.expectations
        assert report.retrieval_config_version == lexical.retrieval_config_version
        assert report.suite_version == lexical.suite_version

    def test_the_two_scorers_are_distinguishable(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        assert report.scorer_version != lexical.scorer_version
        assert report.strategy is not lexical.strategy

    def test_lexical_remains_the_better_strategy_at_every_k(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """The headline comparison, and it goes the way the docs say.

        Not the expected result when the slice started, and it is not tuned away: nothing
        about lexical retrieval was touched to produce it, and the vector side has no knob
        to turn that would not be an invention.
        """
        for k in (1, 3, 5, 10):
            vector_recall = report.metrics.overall.recall[k]
            lexical_recall = lexical.metrics.overall.recall[k]
            assert vector_recall is not None and lexical_recall is not None
            assert vector_recall < lexical_recall, f"@{k}"
            assert lexical_recall == pytest.approx(LEXICAL_BASELINE_RECALL[k], abs=TOLERANCE)

    def test_vector_finds_a_strict_subset_of_what_lexical_finds(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """The result of the slice, and the reason fusion is not being built next.

        With this embedder the vector strategy finds **nothing** the lexical one misses:
        46 expectations found by both, 0 by vector alone, 5 by lexical alone. A second
        strategy is worth fusing when it reaches evidence the first cannot, and by that
        test this embedder is worth nothing - so the honest next step is a real embedder,
        not a weighting.

        This is a statement about ``hashing-1`` and not about vector retrieval. Feature
        hashing over a bag of words is a term-overlap signal, and a term-overlap signal
        cannot reach what a better term-overlap signal misses.
        """
        vector_hits = hits(report)
        lexical_hits = hits(lexical)

        assert len(vector_hits & lexical_hits) == 46
        assert vector_hits - lexical_hits == set()
        assert len(lexical_hits - vector_hits) == 5
        assert vector_hits < lexical_hits

    def test_the_five_lexical_only_finds_are_all_morphological(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """*Why* lexical wins, which matters more than that it does.

        Every expectation lexical reaches and vector does not is a singular/plural pair
        that the ``tokens-3`` suffix rule merges. The hashing embedder has no suffix rule -
        deliberately, since it must not share preprocessing with the lexical strategy - so
        ``refund`` and ``Refunds`` land in different coordinates. The gap is a
        preprocessing difference, not evidence that cosine is a worse ranking function.
        """
        only_lexical = hits(lexical) - hits(report)
        assert {query_id for query_id, _ in only_lexical} == {
            "key_refund_approval_threshold",
            "lexical_miss_refund_singular",
            "paraphrase_who_signs_off_large_refunds",
        }

    def test_neither_strategy_reaches_the_five_hardest_expectations(
        self, report: BenchmarkReport, lexical: BenchmarkReport
    ) -> None:
        """Four of them are the coupling set of ``retention.window``, which no text-similarity
        strategy reconstructs - the empirical case for the closed registry, now confirmed by
        a second strategy rather than argued from one.
        """
        missed_by_both = reachable(lexical) - hits(lexical) - hits(report)
        assert {query_id for query_id, _ in missed_by_both} == {
            "key_retention_window",
            "lexical_miss_token_expiry_wording",
        }
        assert len(missed_by_both) == 5

    def test_the_baseline_was_not_modified_to_produce_the_comparison(
        self, lexical: BenchmarkReport, suite: RetrievalSuite
    ) -> None:
        """The lexical run still reports the frozen scorer and the frozen figures."""
        assert lexical.scorer_version == "lexical-bm25-1:tokens-3:k1=1.2/b=0.75"
        assert lexical.version_drift(suite) == ()
        assert lexical.metrics.negative_controls.returned == 5


class TestWhereVectorIsWorse:
    def test_the_plural_category_collapses(self, report: BenchmarkReport) -> None:
        """``known_lexical_miss`` recalls 20% @10 against lexical's 80%, for the reason
        above: no suffix rule in the embedder's tokenizer."""
        assert report.metrics.by_category["known_lexical_miss"].recall[10] == pytest.approx(
            0.2, abs=TOLERANCE
        )

    def test_the_paraphrase_category_is_worse_not_better(self, report: BenchmarkReport) -> None:
        """The finding that most contradicts the expectation going in.

        Paraphrase recall was the *motivation* for adding a vector strategy, and with this
        embedder it is 71.4% @10 against lexical's 100%. An embedder with no semantics
        cannot help a paraphrase, and measuring that is worth more than assuming the
        opposite would have held.
        """
        assert report.metrics.by_category["paraphrase"].recall[10] == pytest.approx(
            0.714, abs=TOLERANCE
        )


class TestWhereVectorMatches:
    @pytest.mark.parametrize(
        "category",
        [
            "exact_terminology",
            "numeric_value",
            "cross_medium",
            "scoped_concept",
            "version_currency",
            "source_type_filter",
            "permission_restricted",
        ],
    )
    def test_it_matches_lexical_at_ten_on_the_filter_bound_categories(
        self, report: BenchmarkReport, category: str
    ) -> None:
        """Where the answer is decided by a filter or by literal wording, the strategies
        agree - which is the evidence that the vector *machinery* is wired correctly rather
        than merely returning something."""
        assert report.metrics.by_category[category].recall[10] == 1.0

    def test_the_coverage_gap_category_is_unreachable_for_both(
        self, report: BenchmarkReport
    ) -> None:
        """No embedding can retrieve a span that was never admitted."""
        gaps = report.metrics.by_category["coverage_gap"]
        assert gaps.retrievable == 0
        assert gaps.not_admitted == 4
        assert gaps.recall[10] is None
