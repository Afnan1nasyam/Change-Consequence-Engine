"""The retrieval benchmark, scored over the real corpus, with no model call.

Unit tests prove the recall arithmetic and the suite schema. This runs the labelled queries
through the real stores against the real lexical retriever and asserts three different
kinds of thing, which are worth keeping apart in the reader's mind:

**Correctness of the measurement.** Every label resolves; no expectation is charged to a
filter the query did not ask for; the diagnostic that classifies a withheld or filtered
expectation agrees with the retriever's own trace. If these fail, no number below means
anything.

**Properties that must hold whatever the numbers are.** Withheld evidence is never
returned. The run is deterministic. The pinned scorer is the one that ran.

**The measured numbers themselves**, pinned as a regression guard. These are *observations*,
not targets: several of them are poor, and they are asserted exactly so that a change which
moves them cannot land silently. A failure here is an instruction to re-measure and update
``docs/EVALUATION.md``, not necessarily a defect.

The test worth reading first is
:func:`TestMeasurementIsSound.test_the_exclusion_diagnostic_agrees_with_the_retrievers_trace`.
The benchmark decides *why* an expectation did not arrive using its own copy of the filter
order, because the retriever's chain is deliberately private; that test is what stops the
copy drifting away from the original and silently reclassifying a miss.
"""

from __future__ import annotations

import asyncio

import pytest

from app.evaluation.corpus import load_corpus
from app.evaluation.retrieval_benchmark import (
    BenchmarkReport,
    RetrievalSuite,
    load_suite,
    run_suite,
)
from app.evaluation.retrieval_metrics import ExpectationOutcome, QueryScore

#: Measured on 2026-09-09 by ``python -m scripts.retrieval_benchmark`` and quoted in
#: ``docs/EVALUATION.md``. Recall is over the *reachable* labels; end-to-end recall divides
#: the same hits by every label an analyst wrote. Current scorer: ``tokens-3``.
MEASURED_RECALL = {1: 0.357, 3: 0.750, 5: 0.893, 10: 0.911}
MEASURED_END_TO_END = {1: 0.323, 3: 0.677, 5: 0.806, 10: 0.823}
MEASURED_AUTHORIZATION_RECALL = {1: 0.5, 3: 1.0, 5: 1.0, 10: 1.0}

#: Recall under each tokenizer the suite has scored, kept so the direction of every change
#: is on the record rather than only in a document:
#:
#: ============  =====  =====  =====  =====  ==========================================
#: scorer          @1     @3     @5    @10   change
#: ============  =====  =====  =====  =====  ==========================================
#: ``tokens-1``  33.9%  62.5%  78.6%  80.4%  as the benchmark first found the retriever
#: ``tokens-2``  35.7%  67.9%  80.4%  80.4%  stopword list (issue 32)
#: ``tokens-3``  35.7%  75.0%  89.3%  91.1%  guarded plural rule (issue 26)
#: ============  =====  =====  =====  =====  ==========================================
#:
#: Only the immediately preceding scorer is asserted, and only as an inequality: a live run
#: can show the direction of the last change, and chaining every historical scorer would
#: pin numbers no current code can produce.
RECALL_BEFORE_PLURAL_RULE = {1: 0.357, 3: 0.679, 5: 0.804, 10: 0.804}

#: Expectation outcomes under ``tokens-2``, before the plural rule. Six labelled spans that
#: were reachable and missed are now retrieved.
OUTCOMES_BEFORE_PLURAL_RULE = {"retrieved": 45, "retrieval_miss": 11}

TOLERANCE = 0.002


@pytest.fixture(scope="module")
def suite() -> RetrievalSuite:
    return load_suite()


@pytest.fixture(scope="module")
def report(suite: RetrievalSuite) -> BenchmarkReport:
    return asyncio.run(run_suite(suite))


def by_id(report: BenchmarkReport) -> dict[str, QueryScore]:
    return {query.query_id: query for query in report.queries}


def ranks(query: QueryScore) -> dict[str, int | None]:
    return {expectation.source: expectation.rank for expectation in query.expectations}


# ---------------------------------------------------------------------------
# The measurement is sound
# ---------------------------------------------------------------------------


class TestMeasurementIsSound:
    def test_the_run_makes_no_model_call_and_needs_no_credentials(self) -> None:
        """Extraction is replayed; nothing on the retrieval path can reach a provider."""
        stores = asyncio.run(load_corpus())
        assert not stores.extraction_failures, stores.extraction_failures
        assert sum(r.extraction_calls for r in stores.reports.values()) == 0

    def test_every_label_resolves_to_a_span_or_to_a_known_gap(
        self, report: BenchmarkReport
    ) -> None:
        """No label defects: every quote is present exactly once in the document it names.

        This is what makes the rest of the file readable. A stale quote would otherwise be
        counted as unreachable evidence and blamed on the admission boundary.
        """
        assert report.label_defects == (), [
            (d.query_id, d.source, d.why) for d in report.label_defects
        ]

    def test_no_expectation_is_removed_by_a_filter_its_query_asked_for(
        self, report: BenchmarkReport
    ) -> None:
        """A non-zero count here means a label and a query disagree - the suite is wrong.

        Authorization is deliberately not one of these: withholding relevant evidence from
        a requester who may not read it is correct, and the label stays as written.
        """
        assert report.metrics.overall.filtered_by_query == 0

    def test_the_outcome_of_every_expectation_is_accounted_for(
        self, report: BenchmarkReport
    ) -> None:
        overall = report.metrics.overall
        assert sum(overall.outcomes.values()) == overall.expectations
        assert (
            overall.retrievable
            + overall.not_admitted
            + overall.withheld_by_permission
            + overall.filtered_by_query
            + overall.label_defects
            == overall.expectations
        )

    def test_the_exclusion_diagnostic_agrees_with_the_retrievers_trace(
        self, report: BenchmarkReport
    ) -> None:
        """If the benchmark says a filter removed a labelled span, the trace must agree.

        The benchmark classifies an unreturned expectation with its own copy of the filter
        order, because the retriever's chain is private on purpose - it enforces
        authorization by never iterating an unreadable span, and a per-candidate entry
        point into it would weaken that. This is the guard against the copy drifting: for
        every bucket, an expectation charged to it requires the real trace to have charged
        at least that many spans to the same bucket.
        """
        buckets = {
            ExpectationOutcome.EXCLUDED_BY_PERMISSION: "excluded_by_permission",
            ExpectationOutcome.EXCLUDED_BY_SOURCE_TYPE: "excluded_by_source_type",
            ExpectationOutcome.EXCLUDED_BY_AS_OF: "excluded_by_as_of",
            ExpectationOutcome.EXCLUDED_BY_SCOPE: "excluded_by_scope",
            ExpectationOutcome.EXCLUDED_UNSCOPED: "excluded_unscoped",
        }
        for query in report.queries:
            for outcome, field in buckets.items():
                charged = sum(1 for e in query.expectations if e.outcome is outcome)
                if charged:
                    assert charged <= getattr(query.trace, field), (
                        f"{query.query_id}: the benchmark charged {charged} labelled span(s) "
                        f"to {field}, the retriever charged {getattr(query.trace, field)}"
                    )

    def test_a_retrieved_expectation_is_at_the_rank_the_outcome_reported(
        self, report: BenchmarkReport
    ) -> None:
        """The rank is read back from the returned list, so a hit cannot be mislabelled."""
        for query in report.queries:
            for expectation in query.expectations:
                if expectation.rank is None:
                    continue
                assert expectation.evidence_id is not None
                assert query.returned[expectation.rank - 1] == expectation.evidence_id

    def test_a_missed_expectation_is_absent_from_the_returned_list(
        self, report: BenchmarkReport
    ) -> None:
        for query in report.queries:
            returned = set(query.returned)
            for expectation in query.expectations:
                if expectation.outcome is ExpectationOutcome.RETRIEVAL_MISS:
                    assert expectation.evidence_id not in returned


# ---------------------------------------------------------------------------
# Properties that hold whatever the numbers are
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_withheld_evidence_is_never_returned(self, report: BenchmarkReport) -> None:
        """The one metric with a value fixed in advance. A leak is a security defect."""
        assert report.leaks == ()
        assert report.metrics.authorization.leaks == 0

    def test_the_run_is_deterministic(self, suite: RetrievalSuite) -> None:
        """Same corpus, same suite, same report - including the order of every ranking."""
        first = asyncio.run(run_suite(suite))
        second = asyncio.run(run_suite(suite))
        assert first.queries == second.queries
        assert first.metrics == second.metrics

    def test_the_pinned_scorer_is_the_one_that_ran(
        self, report: BenchmarkReport, suite: RetrievalSuite
    ) -> None:
        """A recall figure belongs to one scorer and one filter set. If either moves, the
        numbers below must be re-measured rather than reprinted."""
        assert report.version_drift(suite) == ()
        assert report.scorer_version == "lexical-bm25-1:tokens-3:k1=1.2/b=0.75"
        assert report.retrieval_config_version == "retrieval-1"

    def test_recall_never_exceeds_the_end_to_end_figure_at_any_k(
        self, report: BenchmarkReport
    ) -> None:
        overall = report.metrics.overall
        for k, value in overall.recall.items():
            end_to_end = overall.end_to_end_recall[k]
            assert value is not None and end_to_end is not None
            assert end_to_end <= value

    def test_the_benchmark_reads_a_corpus_of_admitted_spans_only(
        self, report: BenchmarkReport
    ) -> None:
        assert report.corpus_spans == 21


# ---------------------------------------------------------------------------
# The measured numbers, pinned
# ---------------------------------------------------------------------------


class TestMeasuredResults:
    def test_the_suite_size(self, report: BenchmarkReport) -> None:
        overall = report.metrics.overall
        assert overall.queries == 35
        assert overall.expectations == 62
        assert overall.labelled == 62

    def test_coverage_of_the_expected_evidence(self, report: BenchmarkReport) -> None:
        """Four labelled spans are unreachable by any strategy: no admitted binding covers
        them (``docs/STATUS.md`` 25). Two more are correctly withheld by authorization."""
        overall = report.metrics.overall
        assert overall.retrievable == 56
        assert overall.not_admitted == 4
        assert overall.withheld_by_permission == 2
        assert overall.coverage == pytest.approx(0.903, abs=TOLERANCE)

    @pytest.mark.parametrize("k", sorted(MEASURED_RECALL))
    def test_recall_at_k(self, report: BenchmarkReport, k: int) -> None:
        assert report.metrics.overall.recall[k] == pytest.approx(MEASURED_RECALL[k], abs=TOLERANCE)

    @pytest.mark.parametrize("k", sorted(MEASURED_END_TO_END))
    def test_end_to_end_recall_at_k(self, report: BenchmarkReport, k: int) -> None:
        assert report.metrics.overall.end_to_end_recall[k] == pytest.approx(
            MEASURED_END_TO_END[k], abs=TOLERANCE
        )

    @pytest.mark.parametrize("k", sorted(MEASURED_AUTHORIZATION_RECALL))
    def test_authorization_filtered_recall_at_k(self, report: BenchmarkReport, k: int) -> None:
        authorization = report.metrics.authorization
        assert authorization.restricted_queries == 3
        assert authorization.withheld == 2
        assert authorization.recall[k] == pytest.approx(
            MEASURED_AUTHORIZATION_RECALL[k], abs=TOLERANCE
        )

    def test_most_negative_controls_are_now_silent(self, report: BenchmarkReport) -> None:
        """The measurement the stopword list was made for: 24 results became 5.

        Three of four controls are silent. The survivor is
        ``negative_pack_personal_items``, and it survives for a reason no stopword list can
        address: it shares the *content* words "personal" and "items" with the retention
        spans and the page-size span. What is left is content-word ambiguity, which is a
        precision problem for a later strategy rather than an unfinished stopword list.
        """
        controls = report.metrics.negative_controls
        assert controls.queries == 4
        assert controls.silent == 3
        assert controls.returned == 5

    def test_the_plural_rule_found_evidence_that_was_previously_missed(
        self, report: BenchmarkReport
    ) -> None:
        """The shape of this change, and it is a different shape from the stopword list.

        The stopword list moved evidence *up* without changing what came back at all. The
        plural rule changes what comes back: six labelled spans that were reachable and
        missed are now retrieved, so recall rises at 3, 5 and 10 and the miss count falls.
        Both are asserted, because "recall went up" is compatible with having traded
        something away and the outcome counts are what rule that out.
        """
        recall = report.metrics.overall.recall
        for k in (3, 5, 10):
            before = RECALL_BEFORE_PLURAL_RULE[k]
            after = recall[k]
            assert after is not None and after > before, f"@{k}: {before} -> {after}"

        outcomes = report.metrics.overall.outcomes
        assert outcomes[ExpectationOutcome.RETRIEVED] == 51
        assert outcomes[ExpectationOutcome.RETRIEVAL_MISS] == 5
        assert (
            outcomes[ExpectationOutcome.RETRIEVED] - OUTCOMES_BEFORE_PLURAL_RULE["retrieved"]
            == OUTCOMES_BEFORE_PLURAL_RULE["retrieval_miss"]
            - outcomes[ExpectationOutcome.RETRIEVAL_MISS]
            == 6
        )

    def test_the_plural_rule_cost_nothing_in_reachability_or_noise(
        self, report: BenchmarkReport
    ) -> None:
        """What must not have moved, and did not.

        Coverage, the unreachable count and the withheld count are properties of ingestion
        and authorization, so a tokenizer change touching any of them would mean something
        had gone wrong. The negative controls are the precision-adjacent check: they are
        byte-identical to the stopword-era measurement, so widening what matches did not
        buy recall by making the retriever noisier.
        """
        overall = report.metrics.overall
        assert overall.retrievable == 56
        assert overall.not_admitted == 4
        assert overall.withheld_by_permission == 2
        assert overall.coverage == pytest.approx(0.903, abs=TOLERANCE)

        controls = report.metrics.negative_controls
        assert controls.silent == 3
        assert controls.returned == 5


class TestWhereRetrievalFails:
    """The misses that carry information, asserted individually.

    Each of these is a *documented* limitation of the retriever being priced. They are
    asserted as misses so that fixing one is a visible, deliberate change with a
    before-figure attached.
    """

    def test_what_still_misses_is_a_genuine_vocabulary_gap(self, report: BenchmarkReport) -> None:
        """The remaining misses, and a correction to what this file used to claim.

        Under ``tokens-2`` the suite recorded "who has to sign off on a large refund" as
        semantically unreachable, and the singular/plural miss as a separate defect. Both
        readings were partly wrong: the paraphrase failed because the spans write "Refunds"
        and the query writes "refund", so the plural rule alone recovered every expectation
        of both queries.

        What remains is narrower and genuinely semantic. Five reachable expectations are
        still missed - four in ``key_retention_window`` and one in
        ``lexical_miss_token_expiry_wording``, where the runbook says "the link is valid for
        24 hours" and shares no word *form* with "token expiry" under any inflection. That
        is the vector case, and it is smaller than it looked.
        """
        missed = [
            (q.query_id, e.source)
            for q in report.queries
            for e in q.expectations
            if e.outcome is ExpectationOutcome.RETRIEVAL_MISS
        ]
        assert len(missed) == 5
        assert {query_id for query_id, _ in missed} == {
            "key_retention_window",
            "lexical_miss_token_expiry_wording",
        }

    def test_typing_an_assumption_key_does_not_reconstruct_its_coupling_set(
        self, report: BenchmarkReport
    ) -> None:
        """The most important negative result in the suite.

        ``retention.window`` is bound by five artifacts. Only one of them spells the key
        out - the injected wiki page - and the prose that binds it writes "retained for 90
        days", sharing no term with the key. So lexical retrieval returns the least
        trustworthy member of the coupling set and misses the governing policy.

        This is why coupling is resolved through the closed registry and never through text
        similarity, and why Class A recall does not depend on retrieval at all.
        """
        query = by_id(report)["key_retention_window"]
        found = {e.source for e in query.expectations if e.rank is not None}
        assert found == {"wiki/vendor-integration-notes.md"}
        assert len(query.expectations) == 5

    def test_unadmitted_text_is_unreachable_and_reported_as_such(
        self, report: BenchmarkReport
    ) -> None:
        """A material policy exception, a delegation window, an unregistered escalation
        ceiling and an unregistered config value. All four are stored, none retrievable."""
        gaps = report.metrics.by_category["coverage_gap"]
        assert gaps.labelled == 4
        assert gaps.not_admitted == 4
        assert gaps.retrievable == 0
        assert gaps.coverage == 0.0
        assert gaps.recall[10] is None, "recall over nothing must be unknown, not zero"


class TestTokenizerEffects:
    """What the two tokenizer changes did, query by query.

    Pinned individually because the headline recall numbers understate both: the stopword
    list's value is mostly in results that *stopped* coming back (``docs/STATUS.md`` 32),
    which a recall suite cannot see at all, and the plural rule moved rankings in both
    directions (issue 26).
    """

    def test_a_control_answered_only_by_function_words_is_now_silent(
        self, report: BenchmarkReport
    ) -> None:
        """ "the coffee machine is coming with us" returned 13 admitted spans under
        ``tokens-1``, on ``the``, ``is`` and ``with``. It now returns none."""
        assert by_id(report)["negative_coffee_machine"].returned == ()

    def test_the_surviving_control_survives_on_content_words(self, report: BenchmarkReport) -> None:
        """ "pack your personal items into the crate" still returns 5 spans, down from 11.

        It shares "personal" with three retention spans and "items" with the page-size
        span. Both are content words, so this is the floor for a stopword list rather than
        evidence of an incomplete one - and it is why the remaining figure is reported
        instead of being tuned away.
        """
        assert len(by_id(report)["negative_pack_personal_items"].returned) == 5

    def test_noise_stayed_low_on_the_paraphrase_queries(self, report: BenchmarkReport) -> None:
        """The stopword list cut function-word overlap; the plural rule did not undo it.

        ``paraphrase_password_reset_link_validity`` went 15 spans to 3 under ``tokens-2``
        and is still 3. ``paraphrase_who_signs_off_large_refunds`` went 7 to 2 and is now 7
        again - but for the opposite reason: it is returning the spans that *answer* it,
        both of which are labelled and were missed before.
        """
        queries = by_id(report)
        assert len(queries["paraphrase_password_reset_link_validity"].returned) == 3
        assert len(queries["numeric_retention_ninety_days"].returned) == 8

        signs_off = queries["paraphrase_who_signs_off_large_refunds"]
        assert len(signs_off.returned) == 7
        assert all(e.rank is not None for e in signs_off.expectations)

    def test_the_wiki_page_outranks_the_governing_policy_again(
        self, report: BenchmarkReport
    ) -> None:
        """The one clear regression from the plural rule, and it undoes a stopword-era gain.

        Asked "how long do we keep customer data once an account is closed": under
        ``tokens-1`` the EMEA wiki page - an unauthorized override at `wiki` authority - was
        2nd and the governing policy 3rd; the stopword list made it policy 1st, wiki 4th;
        the plural rule reverses that to wiki 1st, policy 2nd. The wiki span writes
        "customers", which now matches the query's "customer", so it gains a term the
        policy does not have.

        Every expectation is still returned, so no recall figure moves - this is purely
        about the order an agent reads them in. Retrieval does not know about authority and
        must not; the fix belongs in how a consumer orders evidence, not in the tokenizer.
        Pinned so the trade stays visible rather than being quietly re-won and re-lost.
        """
        ranked = ranks(by_id(report)["paraphrase_how_long_we_keep_customer_data"])
        assert ranked["wiki/emea-support-notes.md"] == 1
        assert ranked["policy/data-retention-policy.md"] == 2
        assert ranked["policy/eu-data-protection-policy.md"] == 3

    def test_the_plural_rule_promoted_the_authoritative_refund_spans(
        self, report: BenchmarkReport
    ) -> None:
        """The same effect pulling the other way, on the queries the change was aimed at.

        On ``numeric_refund_threshold_with_separator`` the wiki page fell from 2nd to 4th
        while the governing policy rose from 3rd to 2nd and the training deck from 5th to
        3rd, because "Refunds" now matches "refund". Authority-blind either way, but the
        aggregate is not one-sided and reporting only the regression above would be as
        selective as reporting only this.
        """
        ranked = ranks(by_id(report)["numeric_refund_threshold_with_separator"])
        assert ranked["policy/refund-approval-policy.md"] == 2
        assert ranked["training/refund-handling-deck.md"] == 3
        assert ranked["wiki/emea-support-notes.md"] == 4

    def test_the_two_equal_authority_peers_swapped_places(self, report: BenchmarkReport) -> None:
        """Recorded because it is the one ranking that arguably got worse.

        Under ``tokens-1`` the 15-minute runbook was 1st and the 30-minute one 2nd; they
        are now reversed. Both are returned either way, so no recall figure moves, and
        neither order is *correct* - they are peers at the same authority class and
        `resolve_authority` answers Ambiguous for exactly this pair. Reported rather than
        presented as an improvement.
        """
        ranked = ranks(by_id(report)["as_of_p1_target_2026_01"])
        assert ranked["sop/incident-response-runbook.md"] == 1
        assert ranked["sop/password-reset-runbook.md"] == 2


class TestWhereRetrievalWorks:
    def test_exact_terminology_is_found_at_the_top(self, report: BenchmarkReport) -> None:
        exact = report.metrics.by_category["exact_terminology"]
        assert exact.recall[1] == pytest.approx(0.8, abs=TOLERANCE)
        assert exact.recall[3] == 1.0

    def test_a_digit_group_separator_does_not_change_the_ranking(
        self, report: BenchmarkReport
    ) -> None:
        """The corpus writes 10,000 and a change request writes 10000. The tokenizer drops
        the separator, so the two queries must be indistinguishable - a difference here
        would be a tokenizer defect, not a recall result."""
        queries = by_id(report)
        separated = queries["numeric_refund_threshold_with_separator"]
        plain = queries["numeric_refund_threshold_without_separator"]

        assert separated.returned == plain.returned
        assert ranks(separated) == ranks(plain)

    def test_a_cross_medium_pair_is_recovered_by_a_shared_number(
        self, report: BenchmarkReport
    ) -> None:
        """A JSON config and a prose runbook, joined by nothing but the value 24."""
        assert report.metrics.by_category["cross_medium"].recall[3] == 1.0

    def test_a_scoped_query_keeps_the_global_declaration(self, report: BenchmarkReport) -> None:
        """Scope is intersection, not equality: a global policy governs EU customers too."""
        query = by_id(report)["scoped_eu_retention"]
        assert all(e.rank is not None for e in query.expectations)

    def test_a_disjoint_scope_excludes_the_sibling_region(self, report: BenchmarkReport) -> None:
        """The EU spans are irrelevant to a US question, so they are not labelled; the
        exclusion is visible in the trace instead."""
        query = by_id(report)["scoped_us_retention"]
        assert query.trace.excluded_by_scope == 2
        assert all(e.rank is not None for e in query.expectations)

    def test_a_stale_version_is_excluded_at_the_earlier_date_and_returned_at_the_later(
        self, report: BenchmarkReport
    ) -> None:
        """The same question at two dates. At 2025-05-01 the incident runbook does not yet
        govern, so only the 15-minute target comes back; at 2026-01-01 both peers do."""
        queries = by_id(report)
        early = queries["as_of_p1_target_2025_05"]
        late = queries["as_of_p1_target_2026_01"]

        assert early.trace.excluded_by_as_of == 7
        assert len(early.expectations) == 1
        assert all(e.rank is not None for e in early.expectations)
        assert len(late.expectations) == 2
        assert all(e.rank is not None for e in late.expectations)

    def test_authorization_does_not_change_the_ranking_of_readable_evidence(
        self, report: BenchmarkReport
    ) -> None:
        """The substantive authorization result, and it was not a foregone conclusion.

        Corpus statistics are recomputed over the authorized slice, so a restricted
        requester scores every span differently by construction. Measured on the identical
        query with and without the grant, the evidence they are entitled to comes back at
        the same ranks: the different statistics did not reorder it.
        """
        queries = by_id(report)
        granted = queries["exact_manager_approval_before_release"]
        restricted = queries["restricted_refund_threshold_least_privilege"]

        assert restricted.trace.excluded_by_permission == 2
        assert granted.trace.excluded_by_permission == 0
        assert (
            ranks(restricted)
            == ranks(granted)
            == {
                "policy/refund-approval-policy.md": 1,
                "standard/payments-operations-standard.md": 2,
            }
        )

    def test_the_same_label_is_a_hit_or_a_withholding_depending_only_on_the_grant(
        self, report: BenchmarkReport
    ) -> None:
        """One relevance judgment, two requesters. The label does not change."""
        queries = by_id(report)
        withheld = queries["restricted_retention_least_privilege"].expectations[0]
        granted = queries["restricted_retention_privacy_reader"].expectations[0]

        assert withheld.quote == granted.quote
        assert withheld.evidence_id == granted.evidence_id
        assert withheld.outcome is ExpectationOutcome.EXCLUDED_BY_PERMISSION
        assert granted.outcome is ExpectationOutcome.RETRIEVED

    def test_a_foreign_tenant_reaches_nothing(self, report: BenchmarkReport) -> None:
        """Isolation is upstream of the retriever: the store enumeration is per tenant, so
        a foreign tenant's index is empty and nothing is even considered."""
        query = by_id(report)["foreign_tenant_refund_threshold"]
        assert query.returned == ()
        assert query.trace.considered == 0
