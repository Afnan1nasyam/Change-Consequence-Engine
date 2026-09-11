"""The recall arithmetic, over hand-built outcomes. No corpus, no store, no retriever.

The metrics are the part of a benchmark most easily wrong in a way nobody notices: a
denominator that quietly includes unreachable evidence understates the retriever, and one
that quietly excludes a miss flatters it. Every case below fixes a denominator.

The tests carrying the most weight are the ones asserting what is **not** counted:
:class:`TestDenominators` and :func:`TestAuthorization.test_a_withheld_expectation_is_not_a_miss`.
"""

from __future__ import annotations

import pytest

from app.domain.interfaces.retrieval import RetrievalTrace
from app.evaluation.retrieval_metrics import (
    FILTERED,
    K_VALUES,
    RETRIEVABLE,
    ExpectationOutcome,
    ExpectationResult,
    QueryScore,
    ratio,
    score,
    score_authorization,
    score_negative_controls,
    score_recall,
)

EVIDENCE = "ev_" + "a" * 32


def trace(**over: int) -> RetrievalTrace:
    """A trace with the fields a metric reads and zeros everywhere else."""
    fields: dict[str, int] = {"considered": 21, "matched": 0, "returned": 0}
    fields.update(over)
    return RetrievalTrace(**fields)


def expectation(
    outcome: ExpectationOutcome,
    *,
    rank: int | None = None,
    query_id: str = "q",
    source: str = "policy/a.md",
    evidence_id: str | None = EVIDENCE,
) -> ExpectationResult:
    return ExpectationResult(
        query_id=query_id,
        source=source,
        quote="a quote",
        outcome=outcome,
        evidence_id=evidence_id,
        rank=rank,
    )


def hit(rank: int, **over: object) -> ExpectationResult:
    return expectation(ExpectationOutcome.RETRIEVED, rank=rank, **over)  # type: ignore[arg-type]


def miss(**over: object) -> ExpectationResult:
    return expectation(ExpectationOutcome.RETRIEVAL_MISS, **over)  # type: ignore[arg-type]


def query(
    *expectations: ExpectationResult,
    query_id: str = "q",
    category: str = "cat",
    returned: tuple[str, ...] = (),
    permission_excluded: int = 0,
) -> QueryScore:
    return QueryScore(
        query_id=query_id,
        category=category,
        returned=returned,
        trace=trace(excluded_by_permission=permission_excluded),
        expectations=expectations,
    )


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------


class TestRatio:
    def test_a_ratio_over_nothing_is_unknown_rather_than_zero(self) -> None:
        """The single most important line in the module.

        ``0.0`` would read as "the retriever found none of it" for a category where there
        was nothing to find, which is the opposite of the truth.
        """
        assert ratio(0, 0) is None

    def test_a_ratio_over_something_is_a_fraction(self) -> None:
        assert ratio(1, 4) == 0.25


class TestExpectationInvariants:
    def test_a_rank_is_present_exactly_for_a_retrieved_expectation(self) -> None:
        with pytest.raises(ValueError, match="rank is present exactly"):
            expectation(ExpectationOutcome.RETRIEVAL_MISS, rank=3)

    def test_a_retrieved_expectation_without_a_rank_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="rank is present exactly"):
            expectation(ExpectationOutcome.RETRIEVED)

    def test_a_rank_is_one_based(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            expectation(ExpectationOutcome.RETRIEVED, rank=0)

    @pytest.mark.parametrize("outcome", sorted(RETRIEVABLE))
    def test_claiming_the_retriever_held_a_span_requires_naming_it(
        self, outcome: ExpectationOutcome
    ) -> None:
        """A miss without an ``evidence_id`` is indistinguishable from unreachable text."""
        rank = 1 if outcome is ExpectationOutcome.RETRIEVED else None
        with pytest.raises(ValueError, match="must name one"):
            expectation(outcome, rank=rank, evidence_id=None)

    def test_an_unreachable_expectation_need_not_name_a_span(self) -> None:
        assert expectation(ExpectationOutcome.NOT_ADMITTED, evidence_id=None).evidence_id is None

    @pytest.mark.parametrize("k,found", [(1, False), (2, True), (10, True)])
    def test_found_within_is_inclusive_of_k(self, k: int, found: bool) -> None:
        assert hit(2).found_within(k) is found

    def test_a_miss_is_found_within_no_k(self) -> None:
        assert not any(miss().found_within(k) for k in K_VALUES)


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_at_k_counts_the_expectations_ranked_within_k(self) -> None:
        metrics = score_recall([query(hit(1), hit(3), hit(9), miss())])

        assert metrics.hits == {1: 1, 3: 2, 5: 2, 10: 3}
        assert metrics.recall == {1: 0.25, 3: 0.5, 5: 0.5, 10: 0.75}

    def test_recall_is_monotonic_in_k(self) -> None:
        metrics = score_recall([query(hit(1), hit(4), hit(10), miss(), miss())])
        values = [metrics.recall[k] for k in K_VALUES]
        assert values == sorted(values)  # type: ignore[type-var]

    def test_an_expectation_ranked_beyond_the_largest_k_is_a_miss_at_every_k(self) -> None:
        """The suite runs at ``MAX_TOP_K``, so a rank of 11 is a real observation.

        It is reported as reachable-and-not-found rather than as absent, which is the
        distinction that tells "ranked too low" apart from "shares no term".
        """
        metrics = score_recall([query(hit(11))])
        assert metrics.retrievable == 1
        assert set(metrics.hits.values()) == {0}

    def test_pooling_is_across_queries(self) -> None:
        metrics = score_recall([query(hit(1), query_id="a"), query(miss(), query_id="b")])
        assert metrics.queries == 2
        assert metrics.recall[1] == 0.5


class TestDenominators:
    """What is excluded from the recall denominator, and why. The core of the module."""

    def test_unadmitted_evidence_is_not_a_retrieval_miss(self) -> None:
        """No span exists, so no strategy could have returned it. Not retrieval's failure."""
        metrics = score_recall(
            [query(hit(1), expectation(ExpectationOutcome.NOT_ADMITTED, evidence_id=None))]
        )
        assert metrics.retrievable == 1
        assert metrics.recall[1] == 1.0
        assert metrics.not_admitted == 1

    def test_evidence_withheld_by_authorization_is_not_a_retrieval_miss(self) -> None:
        metrics = score_recall(
            [query(hit(1), expectation(ExpectationOutcome.EXCLUDED_BY_PERMISSION))]
        )
        assert metrics.retrievable == 1
        assert metrics.recall[1] == 1.0
        assert metrics.withheld_by_permission == 1

    @pytest.mark.parametrize("outcome", sorted(FILTERED))
    def test_evidence_removed_by_a_query_filter_is_not_a_retrieval_miss(
        self, outcome: ExpectationOutcome
    ) -> None:
        """It also means the label and the query disagree: a defect in the suite."""
        metrics = score_recall([query(hit(1), expectation(outcome))])
        assert metrics.retrievable == 1
        assert metrics.filtered_by_query == 1

    def test_a_label_defect_is_excluded_from_every_denominator(self) -> None:
        """A broken label is not evidence about the corpus or about the retriever."""
        metrics = score_recall(
            [
                query(
                    hit(1),
                    expectation(ExpectationOutcome.LABEL_DEFECT, evidence_id=None),
                )
            ]
        )
        assert metrics.expectations == 2
        assert metrics.labelled == 1
        assert metrics.label_defects == 1
        assert metrics.coverage == 1.0

    def test_coverage_is_the_reachable_share_of_the_labelled_evidence(self) -> None:
        metrics = score_recall(
            [
                query(
                    hit(1),
                    miss(),
                    expectation(ExpectationOutcome.NOT_ADMITTED, evidence_id=None),
                    expectation(ExpectationOutcome.EXCLUDED_BY_PERMISSION),
                )
            ]
        )
        assert metrics.coverage == 0.5

    def test_coverage_is_unknown_when_nothing_was_labelled(self) -> None:
        assert score_recall([query()]).coverage is None

    def test_end_to_end_recall_is_never_above_recall(self) -> None:
        """It divides the same hits by a larger denominator, so it cannot be higher.

        Reporting only the higher figure blames nobody for the ingestion boundary;
        reporting only the lower one blames the retriever for it. Both are printed.
        """
        metrics = score_recall(
            [query(hit(1), expectation(ExpectationOutcome.NOT_ADMITTED, evidence_id=None))]
        )
        assert metrics.recall[1] == 1.0
        assert metrics.end_to_end_recall[1] == 0.5

    def test_the_outcome_counts_partition_the_expectations(self) -> None:
        metrics = score_recall(
            [
                query(
                    hit(1),
                    miss(),
                    expectation(ExpectationOutcome.NOT_ADMITTED, evidence_id=None),
                    expectation(ExpectationOutcome.EXCLUDED_BY_PERMISSION),
                    expectation(ExpectationOutcome.LABEL_DEFECT, evidence_id=None),
                )
            ]
        )
        assert sum(metrics.outcomes.values()) == metrics.expectations == 5


class TestMacroRecall:
    def test_the_mean_per_query_differs_from_the_pooled_figure(self) -> None:
        """Which is why both are reported.

        One query carrying four labels dominates a pooled figure that three single-label
        queries barely move.
        """
        queries = [
            query(hit(1), query_id="one"),
            query(miss(), miss(), miss(), miss(), query_id="many"),
        ]
        metrics = score_recall(queries)

        assert metrics.recall[1] == 0.2
        assert metrics.macro_recall[1] == 0.5

    def test_a_query_with_nothing_reachable_is_left_out_of_the_mean(self) -> None:
        """Counting it as a zero would charge the retriever for the ingestion boundary."""
        queries = [
            query(hit(1), query_id="reachable"),
            query(
                expectation(ExpectationOutcome.NOT_ADMITTED, evidence_id=None),
                query_id="unreachable",
            ),
        ]
        assert score_recall(queries).macro_recall[1] == 1.0

    def test_the_mean_is_unknown_when_no_query_has_anything_reachable(self) -> None:
        assert score_recall([query()]).macro_recall[1] is None


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


class TestAuthorization:
    def test_only_queries_where_authorization_removed_something_are_scored(self) -> None:
        """Read from the trace, not from the scope.

        A scope that happens to read everything is not a restricted run, and averaging it
        in would dilute the figure that measures restriction.
        """
        queries = [
            query(hit(1), query_id="open"),
            query(miss(), query_id="restricted", permission_excluded=2),
        ]
        metrics = score_authorization(queries)

        assert metrics.restricted_queries == 1
        assert metrics.recall[1] == 0.0

    def test_a_withheld_expectation_is_not_a_miss(self) -> None:
        """It is counted, kept out of the denominator, and named as withheld."""
        metrics = score_authorization(
            [
                query(
                    hit(1),
                    expectation(ExpectationOutcome.EXCLUDED_BY_PERMISSION),
                    permission_excluded=2,
                )
            ]
        )
        assert metrics.expectations == 2
        assert metrics.retrievable == 1
        assert metrics.withheld == 1
        assert metrics.recall[1] == 1.0

    def test_returning_withheld_evidence_is_a_leak(self) -> None:
        leaky = query(
            expectation(ExpectationOutcome.EXCLUDED_BY_PERMISSION),
            returned=(EVIDENCE,),
            permission_excluded=1,
        )
        assert len(leaky.leaked) == 1
        assert score_authorization([leaky]).leaks == 1

    def test_withheld_evidence_that_stays_withheld_is_not_a_leak(self) -> None:
        clean = query(
            expectation(ExpectationOutcome.EXCLUDED_BY_PERMISSION),
            returned=("ev_" + "b" * 32,),
            permission_excluded=1,
        )
        assert clean.leaked == ()
        assert score_authorization([clean]).leaks == 0

    def test_leaks_are_counted_over_every_query_not_only_the_restricted_ones(self) -> None:
        """Scoping the check to the queries expected to restrict would mean trusting the
        suite to have named the risky cases correctly."""
        unexpected = query(
            expectation(ExpectationOutcome.EXCLUDED_BY_PERMISSION),
            returned=(EVIDENCE,),
            permission_excluded=0,
        )
        metrics = score_authorization([unexpected])
        assert metrics.restricted_queries == 0
        assert metrics.leaks == 1


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


class TestNegativeControls:
    def test_a_query_with_no_expectations_is_a_negative_control(self) -> None:
        assert query().is_negative_control
        assert not query(hit(1)).is_negative_control

    def test_silence_is_counted_separately_from_noise(self) -> None:
        metrics = score_negative_controls(
            [
                query(query_id="quiet"),
                query(query_id="noisy", returned=(EVIDENCE, "ev_" + "c" * 32)),
                query(hit(1), query_id="not-a-control"),
            ]
        )
        assert metrics.queries == 2
        assert metrics.silent == 1
        assert metrics.returned == 2
        assert metrics.noise_per_query == 1.0

    def test_noise_is_unknown_when_there_are_no_controls(self) -> None:
        assert score_negative_controls([query(hit(1))]).noise_per_query is None


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------


class TestScore:
    def test_categories_partition_the_queries(self) -> None:
        metrics = score(
            [
                query(hit(1), query_id="a", category="exact"),
                query(miss(), query_id="b", category="paraphrase"),
                query(hit(2), query_id="c", category="paraphrase"),
            ]
        )
        assert sorted(metrics.by_category) == ["exact", "paraphrase"]
        assert sum(m.queries for m in metrics.by_category.values()) == metrics.overall.queries
        assert (
            sum(m.expectations for m in metrics.by_category.values())
            == metrics.overall.expectations
        )

    def test_a_category_is_scored_on_its_own_expectations(self) -> None:
        metrics = score(
            [
                query(hit(1), query_id="a", category="exact"),
                query(miss(), query_id="b", category="paraphrase"),
            ]
        )
        assert metrics.by_category["exact"].recall[1] == 1.0
        assert metrics.by_category["paraphrase"].recall[1] == 0.0

    def test_scoring_is_a_pure_function_of_its_input(self) -> None:
        queries = [query(hit(1), miss(), permission_excluded=1)]
        assert score(queries) == score(queries)

    def test_the_cut_offs_are_the_ones_the_benchmark_reports(self) -> None:
        assert K_VALUES == (1, 3, 5, 10)
        assert tuple(score([query(hit(1))]).overall.recall) == K_VALUES
