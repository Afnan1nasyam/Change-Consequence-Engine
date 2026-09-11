"""Lexical retrieval: the contract, the filters, the ranking, and what must not happen.

Deliberately negative-heavy. The tests that carry the most weight here are the ones
asserting that a span is **not** returned - a wrong tenant, a disjoint scope, a version out
of force, a query sharing no term - because a retriever that returns everything looks
healthy in every positive test and destroys precision downstream.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.domain.interfaces.retrieval import (
    MAX_TOP_K,
    RETRIEVAL_CONFIG_VERSION,
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalPort,
    RetrievalQuery,
    RetrievalResult,
    RetrievalTrace,
)
from app.domain.value_objects.enums import RetrievalStrategy, SourceType
from app.domain.value_objects.scope import ScopeSelector
from app.retrieval.lexical import LexicalConfig, LexicalRetriever
from tests.fixtures.ids import make_id
from tests.fixtures.retrieval import AS_OF, OPEN_SCOPE, OTHER_TENANT, TENANT, candidate

REFUND = "Refunds of 10,000 USD or more require manager approval before release."
RETENTION = "Personal data is retained for 90 days after the account is closed."
EU_RETENTION = "For EU customers we keep personal data for 30 days after closure."
TTL = "auth.reset_token.ttl_hours: 24"


def corpus() -> tuple[EvidenceCandidate, ...]:
    """Four spans that differ in every dimension a filter can see."""
    return (
        candidate(REFUND, n=1, source_type=SourceType.POLICY),
        candidate(RETENTION, n=2, source_type=SourceType.POLICY),
        candidate(
            EU_RETENTION,
            n=3,
            source_type=SourceType.KNOWLEDGE_DOC,
            scopes=("region:EU",),
            assumption_keys=("retention.window",),
        ),
        candidate(TTL, n=4, source_type=SourceType.CONFIG),
    )


def query(text: str, **over: object) -> RetrievalQuery:
    """A query under the open scope unless a test supplies its own."""
    over.setdefault("permission_scope", OPEN_SCOPE)
    return RetrievalQuery(tenant_id=TENANT, text=text, **over)


# ---------------------------------------------------------------------------
# The port
# ---------------------------------------------------------------------------


class TestPort:
    def test_the_retriever_satisfies_the_port(self) -> None:
        assert isinstance(LexicalRetriever(corpus()), RetrievalPort)

    def test_the_outcome_names_the_strategy_that_ran(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("refunds"))
        assert outcome.results
        assert outcome.strategies_executed == (RetrievalStrategy.LEXICAL,)
        assert all(r.strategy is RetrievalStrategy.LEXICAL for r in outcome.results)

    def test_the_outcome_pins_both_versions(self) -> None:
        """Configuration and scorer are versioned apart: retuning is not a filter change."""
        retriever = LexicalRetriever(corpus())
        outcome = retriever.retrieve(query("refunds"))
        assert outcome.retrieval_config_version == RETRIEVAL_CONFIG_VERSION
        assert outcome.scorer_version == retriever.scorer_version
        assert "bm25" in outcome.scorer_version

    def test_retuning_a_parameter_changes_the_scorer_version(self) -> None:
        loose = LexicalRetriever(corpus(), config=LexicalConfig(k1=2.0))
        assert loose.scorer_version != LexicalRetriever(corpus()).scorer_version


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_every_result_carries_its_evidence_id_and_offsets(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("refunds manager approval"))
        assert outcome.results
        for result in outcome.results:
            evidence = result.candidate.evidence
            assert result.evidence_id == evidence.evidence_id
            assert evidence.char_end > evidence.char_start
            assert evidence.text_span == result.candidate.text

    def test_every_result_carries_its_source_provenance(self) -> None:
        """The document a span came from, and the authority it carries, travel with it."""
        outcome = LexicalRetriever(corpus()).retrieve(query("refunds"))
        result = outcome.results[0]
        assert result.source_id == result.candidate.document.source_id
        assert result.source_version_id == result.candidate.version.version_id
        assert result.source_type is result.candidate.document.source_type
        assert result.candidate.authority_class is result.candidate.document.authority_class

    def test_the_stored_evidence_item_is_returned_unmodified(self) -> None:
        """Retrieval must not rewrite a citation. The object it hands back is the stored one."""
        spans = corpus()
        outcome = LexicalRetriever(spans).retrieve(query("refunds"))
        original = {c.evidence_id: c.evidence for c in spans}
        assert outcome.results[0].candidate.evidence == original[outcome.results[0].evidence_id]

    def test_a_candidate_whose_evidence_belongs_to_another_version_is_rejected(self) -> None:
        """A citation assembled out of genuine parts is still a fabricated citation."""
        good = candidate(REFUND, n=1)
        with pytest.raises(ValidationError, match="was joined to"):
            EvidenceCandidate(
                evidence=good.evidence,
                document=good.document,
                version=candidate(REFUND, n=7).version,
            )

    def test_a_candidate_whose_version_belongs_to_another_document_is_rejected(self) -> None:
        good = candidate(REFUND, n=1)
        with pytest.raises(ValidationError, match="belongs to"):
            EvidenceCandidate(
                evidence=good.evidence,
                document=candidate(REFUND, n=7).document,
                version=good.version,
            )


# ---------------------------------------------------------------------------
# Ranking and determinism
# ---------------------------------------------------------------------------


class TestRanking:
    def test_a_matching_span_is_returned(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("refunds approval"))
        assert REFUND in [r.candidate.text for r in outcome.results]

    def test_results_are_ordered_by_descending_score(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("personal data retained days"))
        scores = [r.score for r in outcome.results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_dense_and_start_at_one(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("personal data"))
        assert [r.rank for r in outcome.results] == list(range(1, len(outcome.results) + 1))

    def test_top_k_truncates_and_the_trace_says_so(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("personal data", top_k=1))
        assert outcome.trace.returned == 1
        assert outcome.trace.matched == 2
        assert outcome.trace.truncated

    def test_an_untruncated_result_set_is_not_marked_truncated(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("personal data", top_k=10))
        assert not outcome.trace.truncated

    def test_the_top_k_slice_is_the_highest_scoring_prefix(self) -> None:
        full = LexicalRetriever(corpus()).retrieve(query("personal data", top_k=10))
        clipped = LexicalRetriever(corpus()).retrieve(query("personal data", top_k=1))
        assert clipped.evidence_ids == full.evidence_ids[:1]

    def test_two_runs_of_one_query_are_identical(self) -> None:
        retriever = LexicalRetriever(corpus())
        first = retriever.retrieve(query("personal data days"))
        second = retriever.retrieve(query("personal data days"))
        assert first == second

    def test_candidate_order_cannot_change_the_ranking(self) -> None:
        """Store iteration order must not decide what an analysis saw."""
        forward = LexicalRetriever(corpus()).retrieve(query("personal data days"))
        reverse = LexicalRetriever(tuple(reversed(corpus()))).retrieve(query("personal data days"))
        assert forward.evidence_ids == reverse.evidence_ids
        assert [r.score for r in forward.results] == [r.score for r in reverse.results]

    def test_equal_scores_are_broken_by_evidence_id(self) -> None:
        """Two spans quoting the same words must not swap places between runs."""
        twins = (
            candidate("retention window is ninety days", n=5),
            candidate("retention window is ninety days", n=2),
        )
        outcome = LexicalRetriever(twins).retrieve(query("retention window"))
        assert len({r.score for r in outcome.results}) == 1
        assert list(outcome.evidence_ids) == sorted(outcome.evidence_ids)

    def test_a_longer_span_does_not_outrank_a_precise_one_by_length_alone(self) -> None:
        precise = candidate("retention window", n=1)
        padded = candidate("retention window " + "unrelated filler wording " * 20, n=2)
        outcome = LexicalRetriever((padded, precise)).retrieve(query("retention window"))
        assert outcome.results[0].evidence_id == precise.evidence_id


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestSourceTypeFilter:
    def test_an_omitted_filter_admits_every_source_type(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("personal data"))
        assert outcome.trace.excluded_by_source_type == 0

    def test_a_filter_admits_only_the_named_types(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(
            query("personal data", source_types=(SourceType.POLICY,))
        )
        assert [r.source_type for r in outcome.results] == [SourceType.POLICY]
        assert outcome.trace.excluded_by_source_type == 2

    def test_several_types_may_be_named(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(
            query(
                "personal data 24",
                source_types=(SourceType.KNOWLEDGE_DOC, SourceType.CONFIG),
            )
        )
        assert {r.source_type for r in outcome.results} == {
            SourceType.KNOWLEDGE_DOC,
            SourceType.CONFIG,
        }

    def test_a_type_no_document_has_returns_nothing(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(
            query("personal data", source_types=(SourceType.EMAIL_TEMPLATE,))
        )
        assert outcome.is_empty
        assert outcome.trace.excluded_by_source_type == 4

    def test_filtering_does_not_change_the_score_of_a_surviving_span(self) -> None:
        """Filters select; they do not re-weight. Otherwise two runs are incomparable."""
        unfiltered = LexicalRetriever(corpus()).retrieve(query("personal data"))
        filtered = LexicalRetriever(corpus()).retrieve(
            query("personal data", source_types=(SourceType.POLICY,))
        )
        by_id = {r.evidence_id: r.score for r in unfiltered.results}
        assert all(by_id[r.evidence_id] == r.score for r in filtered.results)


class TestScopeFilter:
    def test_a_broader_declaration_is_relevant_to_a_narrower_question(self) -> None:
        """``global`` reaches ``region:EU``: intersection, not equality (checkpoint 5.5)."""
        outcome = LexicalRetriever(corpus()).retrieve(
            query("personal data", scope=ScopeSelector.parse("region:EU"))
        )
        assert {r.candidate.text for r in outcome.results} == {RETENTION, EU_RETENTION}

    def test_a_narrower_declaration_is_relevant_to_a_broader_question(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(
            query("personal data", scope=ScopeSelector.global_scope())
        )
        assert {r.candidate.text for r in outcome.results} == {RETENTION, EU_RETENTION}

    def test_a_disjoint_scope_is_excluded(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(
            query("personal data", scope=ScopeSelector.parse("region:US"))
        )
        assert {r.candidate.text for r in outcome.results} == {RETENTION}
        assert outcome.trace.excluded_by_scope == 1

    def test_a_span_binding_nothing_is_excluded_and_counted_apart(self) -> None:
        """It is a coverage signal, not a mismatch: no scope filter can ever reach it."""
        unbound = candidate(RETENTION, n=9, scopes=(), assumption_keys=())
        outcome = LexicalRetriever((unbound,)).retrieve(
            query("personal data", scope=ScopeSelector.global_scope())
        )
        assert outcome.is_empty
        assert outcome.trace.excluded_unscoped == 1
        assert outcome.trace.excluded_by_scope == 0

    def test_an_unbound_span_is_retrievable_when_no_scope_is_asked_for(self) -> None:
        unbound = candidate(RETENTION, n=9, scopes=(), assumption_keys=())
        outcome = LexicalRetriever((unbound,)).retrieve(query("personal data"))
        assert outcome.evidence_ids == (unbound.evidence_id,)

    def test_any_bound_scope_of_a_span_can_satisfy_the_filter(self) -> None:
        multi = candidate(RETENTION, n=9, scopes=("region:EU", "region:US"))
        for region in ("region:EU", "region:US"):
            outcome = LexicalRetriever((multi,)).retrieve(
                query("personal data", scope=ScopeSelector.parse(region))
            )
            assert outcome.evidence_ids == (multi.evidence_id,)


class TestTemporalFilter:
    def test_omitting_as_of_ignores_validity_entirely(self) -> None:
        expired = candidate(
            RETENTION,
            n=9,
            effective_from=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            effective_to=dt.datetime(2021, 1, 1, tzinfo=dt.UTC),
        )
        outcome = LexicalRetriever((expired,)).retrieve(query("personal data"))
        assert outcome.evidence_ids == (expired.evidence_id,)

    def test_a_version_not_yet_in_force_is_excluded(self) -> None:
        future = candidate(RETENTION, n=9, effective_from=dt.datetime(2027, 1, 1, tzinfo=dt.UTC))
        outcome = LexicalRetriever((future,)).retrieve(query("personal data", as_of=AS_OF))
        assert outcome.is_empty
        assert outcome.trace.excluded_by_as_of == 1

    def test_a_superseded_version_is_excluded(self) -> None:
        past = candidate(
            RETENTION,
            n=9,
            effective_from=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            effective_to=dt.datetime(2021, 1, 1, tzinfo=dt.UTC),
        )
        outcome = LexicalRetriever((past,)).retrieve(query("personal data", as_of=AS_OF))
        assert outcome.is_empty
        assert outcome.trace.excluded_by_as_of == 1

    def test_validity_is_half_open_at_the_upper_bound(self) -> None:
        """``[from, to)`` - the instant it expires, it no longer governs."""
        span = candidate(
            RETENTION,
            n=9,
            effective_from=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            effective_to=AS_OF,
        )
        assert LexicalRetriever((span,)).retrieve(query("personal data", as_of=AS_OF)).is_empty

    def test_validity_is_inclusive_at_the_lower_bound(self) -> None:
        span = candidate(RETENTION, n=9, effective_from=AS_OF)
        outcome = LexicalRetriever((span,)).retrieve(query("personal data", as_of=AS_OF))
        assert outcome.evidence_ids == (span.evidence_id,)

    def test_an_open_ended_version_is_always_in_force(self) -> None:
        span = candidate(RETENTION, n=9, effective_from=dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
        outcome = LexicalRetriever((span,)).retrieve(query("personal data", as_of=AS_OF))
        assert outcome.evidence_ids == (span.evidence_id,)


class TestTenantIsolation:
    def test_another_tenants_span_is_never_returned(self) -> None:
        theirs = candidate(REFUND, n=9, tenant_id=OTHER_TENANT)
        outcome = LexicalRetriever((theirs, *corpus())).retrieve(query("refunds"))
        assert theirs.evidence_id not in outcome.evidence_ids
        assert outcome.trace.excluded_by_tenant == 1

    def test_a_tenant_with_no_corpus_gets_an_empty_outcome(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(
            RetrievalQuery(tenant_id=OTHER_TENANT, text="refunds", permission_scope=OPEN_SCOPE)
        )
        assert outcome.is_empty
        assert outcome.trace.excluded_by_tenant == 4

    def test_another_tenants_corpus_cannot_change_a_score(self) -> None:
        """Document frequency is per tenant, so a ranking never depends on who else is here."""
        alone = LexicalRetriever(corpus()).retrieve(query("personal data"))
        crowded = LexicalRetriever(
            (
                *corpus(),
                *(candidate(RETENTION, n=100 + i, tenant_id=OTHER_TENANT) for i in range(5)),
            )
        ).retrieve(query("personal data"))
        assert [r.score for r in alone.results] == [r.score for r in crowded.results]


# ---------------------------------------------------------------------------
# Nothing found, and the accounting for it
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_a_query_sharing_no_term_returns_nothing(self) -> None:
        outcome = LexicalRetriever(corpus()).retrieve(query("bicycle maintenance schedule"))
        assert outcome.is_empty
        assert outcome.trace.excluded_zero_score == 4

    def test_a_singular_now_reaches_its_plural(self) -> None:
        """Was the recorded defect, and is now the regression test for its fix.

        ``refund`` reached nothing while the policy wrote ``Refunds``. The benchmark priced
        the miss - ``known_lexical_miss`` recalled 20% flat across every *k* - and the
        guarded plural rule closed it (``docs/STATUS.md`` 26). Both spellings must reach
        the same span, in the same order, or the rule is asymmetric.
        """
        singular = LexicalRetriever(corpus()).retrieve(query("refund"))
        plural = LexicalRetriever(corpus()).retrieve(query("refunds"))

        assert singular.results
        assert singular.evidence_ids == plural.evidence_ids
        assert [r.score for r in singular.results] == [r.score for r in plural.results]

    def test_a_guarded_word_still_matches_only_itself(self) -> None:
        """The rule must not have made ``access``-shaped words reachable by accident.

        A query for a word that is not in any span returns nothing, exactly as before -
        the suffix rule widens what *matches*, and must never widen what *exists*.
        """
        outcome = LexicalRetriever(corpus()).retrieve(query("analysis access status"))
        assert outcome.is_empty
        assert outcome.trace.excluded_zero_score == 4

    def test_a_query_of_only_function_words_returns_nothing(self) -> None:
        """The stopword rule, seen from the retriever (``docs/STATUS.md`` 32).

        Every span here contains "of", "for" or "the", so before the list this query
        ranked the whole corpus by function-word overlap. It is now an empty search, and
        the trace says the words matched nothing rather than that a filter intervened.
        """
        outcome = LexicalRetriever(corpus()).retrieve(query("what is it about the ones we have"))
        assert outcome.is_empty
        assert outcome.trace.excluded_zero_score == 4
        assert outcome.trace.excluded_by_source_type == 0

    def test_a_function_word_cannot_carry_a_span_into_the_results(self) -> None:
        """A span sharing *only* function words with the query is not a weak result.

        The retention span contains "is", "for", "the" and "after"; the query contains
        three of them plus one content term that the span does not have. Only the span
        holding the content term may come back.
        """
        outcome = LexicalRetriever(corpus()).retrieve(query("is the ttl after all"))
        assert outcome.evidence_ids == (candidate(TTL, n=4).evidence_id,)

    def test_a_punctuation_only_query_returns_nothing(self) -> None:
        """No usable term is an empty search, not an error - and it says so in the trace."""
        outcome = LexicalRetriever(corpus()).retrieve(query("!!! ???"))
        assert outcome.is_empty
        assert outcome.trace.excluded_zero_score == 4

    def test_an_empty_index_returns_an_empty_outcome(self) -> None:
        outcome = LexicalRetriever(()).retrieve(query("refunds"))
        assert outcome.is_empty
        assert outcome.trace.considered == 0

    def test_nothing_matched_and_everything_filtered_are_distinguishable(self) -> None:
        """The point of the trace: one empty list, two very different corpus observations."""
        nothing_matched = LexicalRetriever(corpus()).retrieve(query("bicycle"))
        all_filtered = LexicalRetriever(corpus()).retrieve(
            query("refunds", source_types=(SourceType.EMAIL_TEMPLATE,))
        )
        assert nothing_matched.trace.excluded_zero_score == 4
        assert nothing_matched.trace.excluded_by_source_type == 0
        assert all_filtered.trace.excluded_by_source_type == 4
        assert all_filtered.trace.excluded_zero_score == 0


class TestTraceAccounting:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"source_types": (SourceType.POLICY,)},
            {"scope": ScopeSelector.parse("region:US")},
            {"as_of": AS_OF},
            {"top_k": 1},
        ],
    )
    def test_every_candidate_is_counted_exactly_once(self, kwargs: dict[str, object]) -> None:
        spans = (*corpus(), candidate(REFUND, n=9, tenant_id=OTHER_TENANT))
        outcome = LexicalRetriever(spans).retrieve(query("personal data refunds", **kwargs))
        assert outcome.trace.excluded + outcome.trace.matched == outcome.trace.considered
        assert outcome.trace.considered == len(spans)

    def test_duplicate_candidates_are_indexed_once(self) -> None:
        """The same span offered twice must not inflate its own term statistics."""
        span = candidate(REFUND, n=1)
        retriever = LexicalRetriever((span, span, span))
        assert retriever.size == 1
        outcome = retriever.retrieve(query("refunds"))
        assert outcome.evidence_ids == (span.evidence_id,)


# ---------------------------------------------------------------------------
# The contract refuses invalid shapes
# ---------------------------------------------------------------------------


class TestQueryValidation:
    def test_empty_query_text_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalQuery(tenant_id=TENANT, text="", permission_scope=OPEN_SCOPE)

    def test_whitespace_only_query_text_is_rejected(self) -> None:
        """Stripped by the model config, so it fails the same minimum-length check."""
        with pytest.raises(ValidationError):
            RetrievalQuery(tenant_id=TENANT, text="   ", permission_scope=OPEN_SCOPE)

    @pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
    def test_top_k_outside_its_bounds_is_rejected(self, top_k: int) -> None:
        with pytest.raises(ValidationError):
            RetrievalQuery(
                tenant_id=TENANT, text="refunds", top_k=top_k, permission_scope=OPEN_SCOPE
            )

    def test_a_malformed_tenant_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalQuery(tenant_id="tenant-1", text="refunds", permission_scope=OPEN_SCOPE)

    def test_an_unknown_field_is_rejected(self) -> None:
        """``extra="forbid"``: a misspelled filter must fail, never be silently ignored."""
        with pytest.raises(ValidationError):
            RetrievalQuery(
                tenant_id=TENANT,
                text="refunds",
                permission_scope=OPEN_SCOPE,
                source_type=SourceType.POLICY,  # type: ignore[call-arg]
            )

    def test_a_query_cannot_carry_a_confidence_number(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalQuery(
                tenant_id=TENANT,
                text="refunds",
                permission_scope=OPEN_SCOPE,
                confidence=0.9,  # type: ignore[call-arg]
            )


class TestResultValidation:
    def test_a_result_cannot_carry_a_zero_score(self) -> None:
        """A span sharing no term with the query is not a weak result; it is not a result."""
        with pytest.raises(ValidationError):
            RetrievalResult(
                candidate=candidate(REFUND, n=1),
                strategy=RetrievalStrategy.LEXICAL,
                score=0.0,
                rank=1,
            )

    def test_a_result_cannot_carry_rank_zero(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalResult(
                candidate=candidate(REFUND, n=1),
                strategy=RetrievalStrategy.LEXICAL,
                score=1.0,
                rank=0,
            )

    def test_an_outcome_with_sparse_ranks_is_rejected(self) -> None:
        result = RetrievalResult(
            candidate=candidate(REFUND, n=1),
            strategy=RetrievalStrategy.LEXICAL,
            score=1.0,
            rank=2,
        )
        with pytest.raises(ValidationError, match="dense and ascending"):
            RetrievalOutcome(
                query=query("refunds"),
                results=(result,),
                trace=RetrievalTrace(considered=1, matched=1, returned=1),
                permission_scope_hash=OPEN_SCOPE.scope_hash,
                scorer_version="lexical-bm25-1",
            )

    def test_an_outcome_whose_trace_disagrees_with_its_results_is_rejected(self) -> None:
        result = RetrievalResult(
            candidate=candidate(REFUND, n=1),
            strategy=RetrievalStrategy.LEXICAL,
            score=1.0,
            rank=1,
        )
        with pytest.raises(ValidationError, match="returned but"):
            RetrievalOutcome(
                query=query("refunds"),
                results=(result,),
                trace=RetrievalTrace(considered=1, matched=1, returned=0),
                permission_scope_hash=OPEN_SCOPE.scope_hash,
                scorer_version="lexical-bm25-1",
            )

    def test_a_negative_trace_count_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalTrace(considered=-1, matched=0, returned=0)


def test_the_strategy_vocabulary_names_only_what_exists() -> None:
    """Graph retrieval is unimplemented. A member for it would let an ``EvidenceGap`` claim
    a search that never ran, and would let corroboration count a strategy that cannot be
    executed (checkpoint 9.3).

    ``vector`` joined with ``app/retrieval/vectors.py`` and ``hybrid`` with
    ``app/retrieval/hybrid.py``. The assertion is deliberately not just a wider literal:
    every member must be *claimed by something that exists*, so adding a member without an
    implementation fails here.
    """
    from app.retrieval.hybrid import FUSED_STRATEGIES, HybridConfig
    from app.retrieval.vectors import VectorConfig, VectorRetriever

    assert {s.value for s in RetrievalStrategy} == {"lexical", "vector", "hybrid"}

    # The two single-strategy members are each claimed by a retriever.
    retrievers = {
        LexicalRetriever(()).strategy,
        VectorRetriever((), config=VectorConfig(embedder="any", dimensions=8)).strategy,
    }
    assert retrievers == set(FUSED_STRATEGIES)

    # `hybrid` is claimed by the fuser rather than by a retriever, because fusion is a
    # function over two rankings and not a third index. Its config existing is the
    # implementation; `FUSED_STRATEGIES` is what it fuses.
    assert HybridConfig().k > 0
    assert retrievers | {RetrievalStrategy.HYBRID} == set(RetrievalStrategy)


def test_a_result_never_reaches_an_unrelated_tenants_id() -> None:
    """Belt and braces around the isolation tests above, phrased as a property."""
    spans = (*corpus(), candidate(REFUND, n=9, tenant_id=OTHER_TENANT))
    outcome = LexicalRetriever(spans).retrieve(query("refunds personal data 24"))
    assert all(r.candidate.tenant_id == TENANT for r in outcome.results)
    assert make_id("tenant", 2) == OTHER_TENANT
