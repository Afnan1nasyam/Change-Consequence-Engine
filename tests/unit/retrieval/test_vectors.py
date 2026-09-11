"""Vector retrieval: the contract, the filters, the ranking, and what must not happen.

Deliberately a mirror of ``test_lexical.py``. Every structural property asserted there is
asserted here against the same fixture corpus, because none of those properties was about
BM25: tenant isolation, authorization before exposure, filter order, the total tie break,
and provenance that still verifies. A second strategy that quietly relaxed one of them
would be a security or reproducibility regression wearing the clothes of a feature.

The tests carrying the most weight:

* :func:`TestAuthorization.test_an_unreadable_span_is_never_returned` and its sibling
  asserting the withheld count is query-independent - the same pair that guards lexical.
* :func:`TestDeterminism.test_an_exact_tie_is_broken_by_evidence_id` - cosine produces
  near-ties constantly, so the tie break is load-bearing here rather than defensive.
* :func:`TestIndexIntegrity.test_a_vector_of_the_wrong_width_is_refused` - mixing embedders
  in one index makes every cosine in it meaningless, and silently is the worst way for that
  to happen.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from app.domain.interfaces.retrieval import (
    MAX_TOP_K,
    RETRIEVAL_CONFIG_VERSION,
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalPort,
    RetrievalQuery,
)
from app.domain.value_objects.enums import RetrievalStrategy, SourceType
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.infrastructure.embeddings.hashing import HashingEmbedder
from app.retrieval.vectors import (
    SCORE_PRECISION,
    EmbeddedCandidate,
    VectorConfig,
    VectorQuery,
    VectorRetriever,
    cosine,
)
from tests.fixtures.retrieval import AS_OF, OPEN_SCOPE, OTHER_TENANT, TENANT, candidate

REFUND = "Refunds of 10,000 USD or more require manager approval before release."
RETENTION = "Personal data is retained for 90 days after the account is closed."
EU_RETENTION = "For EU customers we keep personal data for 30 days after closure."
TTL = "auth.reset_token.ttl_hours: 24"

EMBEDDER = HashingEmbedder(dimensions=64)
CONFIG = VectorConfig(embedder=EMBEDDER.name, dimensions=EMBEDDER.dimensions)


def corpus() -> tuple[EvidenceCandidate, ...]:
    """The same four spans the lexical suite uses, differing in every filterable dimension."""
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
        candidate(
            TTL,
            n=4,
            source_type=SourceType.CONFIG,
            effective_from=dt.datetime(2026, 6, 1, tzinfo=dt.UTC),
            assumption_keys=("auth.reset_token.ttl",),
        ),
    )


def embed(candidates: tuple[EvidenceCandidate, ...]) -> tuple[EmbeddedCandidate, ...]:
    return tuple(
        EmbeddedCandidate(candidate=c, vector=tuple(EMBEDDER.embed_one(c.text))) for c in candidates
    )


def retriever(candidates: tuple[EvidenceCandidate, ...] | None = None) -> VectorRetriever:
    return VectorRetriever(embed(candidates if candidates is not None else corpus()), config=CONFIG)


def query(text: str, **over: Any) -> VectorQuery:
    over.setdefault("permission_scope", OPEN_SCOPE)
    over.setdefault("tenant_id", TENANT)
    return VectorQuery(
        query=RetrievalQuery(text=text, **over),
        vector=tuple(EMBEDDER.embed_one(text)),
    )


def ask(text: str, **over: Any) -> RetrievalOutcome:
    """One query against a freshly built index over the whole fixture corpus."""
    return retriever().retrieve(query(text, **over))


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------


class TestSurface:
    def test_the_strategy_is_named_vector(self) -> None:
        assert retriever().strategy is RetrievalStrategy.VECTOR

    def test_the_scorer_version_names_the_similarity_and_the_embedder(self) -> None:
        """A recall figure belongs to one embedder. Two indexes built with different
        providers must never look like the same scorer."""
        version = retriever().scorer_version
        assert "vector-cosine-1" in version
        assert EMBEDDER.name in version
        assert "d=64" in version

    def test_a_different_embedder_is_a_different_scorer(self) -> None:
        wide = VectorConfig(embedder=EMBEDDER.name, dimensions=128)
        assert wide.fingerprint() != CONFIG.fingerprint()

    def test_the_filter_configuration_is_shared_with_lexical(self) -> None:
        """The filters are identical, so the config version legitimately does not change -
        only the scorer does. That is what makes a filtered vector run comparable with a
        filtered lexical one."""
        outcome = retriever().retrieve(query("refunds"))
        assert outcome.retrieval_config_version == RETRIEVAL_CONFIG_VERSION

    def test_the_retriever_does_not_actually_satisfy_the_retrieval_port(self) -> None:
        """Stated rather than assumed, and the mechanism is worth knowing.

        ``RetrievalPort`` is ``runtime_checkable``, so ``isinstance`` passes on attribute
        *names* alone and cannot see that ``retrieve`` takes a different argument type -
        which is exactly the kind of false comfort a structural check gives. The real
        assertion is behavioural: handed a bare ``RetrievalQuery``, this retriever fails.

        The port is synchronous because a lexical scan is computation; this strategy needs
        the query embedded first, which is I/O, so the port-shaped surface is the async use
        case above (``docs/ARCHITECTURE.md`` 6.1).
        """
        assert isinstance(retriever(), RetrievalPort), "structural check passes on names"

        bare = RetrievalQuery(tenant_id=TENANT, text="refunds", permission_scope=OPEN_SCOPE)
        with pytest.raises(AttributeError):
            retriever().retrieve(bare)  # type: ignore[arg-type]


class TestIndexIntegrity:
    def test_a_vector_of_the_wrong_width_is_refused(self) -> None:
        """Mixing embedders in one index makes every cosine in it meaningless."""
        span = corpus()[0]
        with pytest.raises(ValueError, match="Mixing embedders"):
            VectorRetriever(
                (EmbeddedCandidate(candidate=span, vector=(0.1, 0.2, 0.3)),), config=CONFIG
            )

    def test_a_query_vector_of_the_wrong_width_is_refused(self) -> None:
        wrong = VectorQuery(query=query("refunds").query, vector=(0.1, 0.2))
        with pytest.raises(ValueError, match="different embedder"):
            retriever().retrieve(wrong)

    def test_duplicate_spans_are_indexed_once(self) -> None:
        """The same span offered twice must not occupy two result ranks."""
        span = corpus()[0]
        index = VectorRetriever(embed((span, span, span)), config=CONFIG)
        assert index.size == 1
        assert index.retrieve(query("refunds")).evidence_ids == (span.evidence_id,)

    def test_an_empty_index_returns_an_empty_outcome(self) -> None:
        outcome = VectorRetriever((), config=CONFIG).retrieve(query("refunds"))
        assert outcome.is_empty
        assert outcome.trace.considered == 0


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------


class TestCosine:
    def test_a_vector_is_maximally_similar_to_itself(self) -> None:
        vector = EMBEDDER.embed_one(REFUND)
        assert cosine(vector, vector) == pytest.approx(1.0)

    def test_a_zero_vector_matches_nothing(self) -> None:
        """No direction means no signal, which is "no match" rather than an arbitrary
        placement in the ranking."""
        assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposed_vectors_score_negative(self) -> None:
        """Which the retriever treats as no result at all, not as a weak one."""
        assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_similarity_is_symmetric(self) -> None:
        left = EMBEDDER.embed_one(REFUND)
        right = EMBEDDER.embed_one(RETENTION)
        assert cosine(left, right) == cosine(right, left)

    def test_length_does_not_matter_only_direction(self) -> None:
        assert cosine([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_mismatched_widths_are_a_programming_error(self) -> None:
        with pytest.raises(ValueError):
            cosine([1.0, 0.0], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_the_most_similar_span_ranks_first(self) -> None:
        outcome = ask("Refunds of 10,000 USD or more require manager approval before release.")
        assert outcome.results[0].candidate.text == REFUND

    def test_scores_descend(self) -> None:
        outcome = ask("personal data retained after the account is closed")
        scores = [result.score for result in outcome.results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_dense_and_start_at_one(self) -> None:
        outcome = ask("personal data")
        assert [r.rank for r in outcome.results] == list(range(1, len(outcome.results) + 1))

    def test_every_result_carries_the_vector_strategy(self) -> None:
        outcome = ask("personal data")
        assert all(r.strategy is RetrievalStrategy.VECTOR for r in outcome.results)

    def test_a_result_never_carries_a_non_positive_score(self) -> None:
        """The contract refuses it, and the retriever must not try: an orthogonal span is
        not a weak result, it is not a result."""
        outcome = ask("personal data")
        assert all(r.score > 0.0 for r in outcome.results)

    def test_top_k_truncates_and_the_trace_says_so(self) -> None:
        outcome = ask("personal data retained", top_k=1)
        assert len(outcome.results) == 1
        assert outcome.trace.truncated
        assert outcome.trace.matched > 1

    def test_a_semantically_closer_span_outranks_a_more_distant_one(self) -> None:
        """The ordering claim, on this embedder: a query drawn from one span's wording must
        place that span above a span about an unrelated subject.

        With the hashing embedder "closer" means "shares more terms" - it carries no
        semantics (see its module docstring). This asserts the *ranking mechanism* works,
        not that the retriever understands anything.
        """
        outcome = ask("reset token ttl hours")
        ordered = [r.candidate.text for r in outcome.results]
        assert ordered[0] == TTL

    def test_similarity_is_pairwise_so_the_corpus_cannot_change_a_score(self) -> None:
        """The structural difference from BM25, and the reason authorization needs no
        statistics recomputed: a span's score depends on the query and on itself only.
        """
        one = VectorRetriever(embed(corpus()[:2]), config=CONFIG).retrieve(query("personal data"))
        all_four = retriever().retrieve(query("personal data"))

        first = {r.evidence_id: r.score for r in one.results}
        second = {r.evidence_id: r.score for r in all_four.results}
        shared = set(first) & set(second)
        assert shared
        assert all(first[e] == second[e] for e in shared)


class TestDeterminism:
    def test_two_runs_of_one_query_are_identical(self) -> None:
        """One index, asked twice. Built once, because the fixture stamps a fresh
        ``created_at`` per call and comparing two *different* corpora would test nothing.
        """
        index = retriever()
        first = index.retrieve(query("personal data retained"))
        second = index.retrieve(query("personal data retained"))

        assert first.results == second.results
        assert first.evidence_ids == second.evidence_ids
        assert first.trace == second.trace

    def test_an_exact_tie_is_broken_by_evidence_id(self) -> None:
        """Two spans with identical text score identically, and the order must still be
        total. Cosine near-ties are routine, so this is the property that makes a vector
        ranking reproducible at all.
        """
        spans = (candidate(REFUND, n=7), candidate(REFUND, n=8))
        outcome = VectorRetriever(embed(spans), config=CONFIG).retrieve(query("refunds"))

        assert len(outcome.results) == 2
        assert outcome.results[0].score == outcome.results[1].score
        assert outcome.evidence_ids == tuple(sorted(outcome.evidence_ids))

    def test_index_construction_order_cannot_change_the_ranking(self) -> None:
        spans = corpus()
        forward = VectorRetriever(embed(spans), config=CONFIG).retrieve(query("personal data"))
        reverse = VectorRetriever(embed(tuple(reversed(spans))), config=CONFIG).retrieve(
            query("personal data")
        )
        assert forward.evidence_ids == reverse.evidence_ids

    def test_scores_are_rounded_before_ranking(self) -> None:
        """So two spans differing only in floating-point noise tie and are broken by id
        rather than by whichever way the last bit fell on this platform."""
        outcome = ask("personal data")
        assert all(r.score == round(r.score, SCORE_PRECISION) for r in outcome.results)


# ---------------------------------------------------------------------------
# Isolation and authorization
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_another_tenants_span_is_never_returned(self) -> None:
        spans = (*corpus(), candidate(REFUND, n=9, tenant_id=OTHER_TENANT))
        outcome = VectorRetriever(embed(spans), config=CONFIG).retrieve(query("refunds"))
        assert all(r.candidate.tenant_id == TENANT for r in outcome.results)

    def test_a_foreign_tenants_spans_are_charged_to_the_tenant_bucket(self) -> None:
        spans = (*corpus(), candidate(REFUND, n=9, tenant_id=OTHER_TENANT))
        outcome = VectorRetriever(embed(spans), config=CONFIG).retrieve(query("refunds"))
        assert outcome.trace.excluded_by_tenant == 1

    def test_a_tenant_with_no_spans_reaches_nothing(self) -> None:
        outcome = retriever().retrieve(query("refunds", tenant_id=OTHER_TENANT))
        assert outcome.is_empty
        assert outcome.trace.excluded_by_tenant == 4


#: A label put on one fixture document, and a grant that matches it.
RESTRICTED = "privacy-office"


def restricted_corpus() -> tuple[EvidenceCandidate, ...]:
    """The same corpus with the retention span's document labelled and re-titled.

    The distinct title matters: every fixture document shares a default title, so a leak
    test searching the outcome for "Refund Approval Policy" would find one of the *allowed*
    documents and fail for the wrong reason.
    """
    spans = list(corpus())
    document = spans[1].document.model_copy(
        update={"permissions": (RESTRICTED,), "title": "Restricted Retention Schedule"}
    )
    spans[1] = spans[1].model_copy(update={"document": document})
    return tuple(spans)


class TestAuthorization:
    def test_an_unreadable_span_is_never_returned(self) -> None:
        index = VectorRetriever(embed(restricted_corpus()), config=CONFIG)
        outcome = index.retrieve(query("personal data retained for 90 days"))

        assert RETENTION not in [r.candidate.text for r in outcome.results]
        assert outcome.trace.excluded_by_permission == 1

    def test_a_grant_makes_the_same_span_reachable(self) -> None:
        index = VectorRetriever(embed(restricted_corpus()), config=CONFIG)
        outcome = index.retrieve(
            query(
                "personal data retained for 90 days",
                permission_scope=PermissionScope.of(RESTRICTED),
            )
        )
        assert RETENTION in [r.candidate.text for r in outcome.results]
        assert outcome.trace.excluded_by_permission == 0

    def test_the_withheld_count_does_not_depend_on_the_query(self) -> None:
        """The oracle test. ``excluded_by_permission`` says how much of the tenant's corpus
        is closed to the requester, never whether any of it matched their words - otherwise
        the field answers "how many restricted documents are about my search terms?".
        """
        index = VectorRetriever(embed(restricted_corpus()), config=CONFIG)
        matching = index.retrieve(query("personal data retained for 90 days"))
        unrelated = index.retrieve(query("bicycle maintenance schedule"))

        assert matching.trace.excluded_by_permission == unrelated.trace.excluded_by_permission == 1

    def test_nothing_about_a_denied_document_escapes_but_a_count(self) -> None:
        """Serialize the whole outcome and look for every field the denied document has."""
        index = VectorRetriever(embed(restricted_corpus()), config=CONFIG)
        outcome = index.retrieve(query("personal data retained for 90 days"))
        denied = restricted_corpus()[1]
        blob = outcome.model_dump_json()

        for leaked in (
            denied.evidence_id,
            denied.source_id,
            denied.source_version_id,
            denied.chunk_id,
            denied.document.uri,
            denied.document.title,
            RESTRICTED,
        ):
            assert leaked not in blob, leaked

    def test_the_scope_hash_is_stamped_and_distinguishes_two_requesters(self) -> None:
        index = VectorRetriever(embed(restricted_corpus()), config=CONFIG)
        without = index.retrieve(query("personal data"))
        with_grant = index.retrieve(
            query("personal data", permission_scope=PermissionScope.of(RESTRICTED))
        )

        assert without.permission_scope_hash == without.query.permission_scope_hash
        assert without.permission_scope_hash != with_grant.permission_scope_hash

    def test_a_grant_does_not_cross_a_tenant_boundary(self) -> None:
        """Holding the label a foreign tenant's document carries does not make it readable."""
        foreign = candidate(RETENTION, n=9, tenant_id=OTHER_TENANT, permissions=(RESTRICTED,))
        index = VectorRetriever(embed((*corpus(), foreign)), config=CONFIG)
        outcome = index.retrieve(
            query("personal data", permission_scope=PermissionScope.of(RESTRICTED))
        )

        assert foreign.evidence_id not in outcome.evidence_ids
        assert outcome.trace.excluded_by_tenant == 1
        assert outcome.trace.excluded_by_permission == 0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestSourceTypeFilter:
    def test_only_the_named_types_come_back(self) -> None:
        outcome = ask("personal data", source_types=(SourceType.POLICY,))
        assert all(r.source_type is SourceType.POLICY for r in outcome.results)
        assert outcome.trace.excluded_by_source_type == 2

    def test_an_empty_allow_list_means_every_type(self) -> None:
        """A caller cannot silence retrieval with an empty allow-list."""
        outcome = ask("personal data", source_types=())
        assert outcome.trace.excluded_by_source_type == 0

    def test_a_type_nobody_has_returns_nothing(self) -> None:
        outcome = ask("personal data", source_types=(SourceType.EMAIL_TEMPLATE,))
        assert outcome.is_empty
        assert outcome.trace.excluded_by_source_type == 4

    def test_the_filter_selects_without_re_weighting(self) -> None:
        """A surviving span keeps the score it had unfiltered, so a filtered run is
        comparable with an unfiltered one."""
        unfiltered = ask("personal data")
        filtered = ask("personal data", source_types=(SourceType.POLICY,))

        before = {r.evidence_id: r.score for r in unfiltered.results}
        after = {r.evidence_id: r.score for r in filtered.results}
        assert after
        assert all(before[e] == after[e] for e in after)


class TestScopeFilter:
    def test_a_scoped_query_keeps_a_global_declaration(self) -> None:
        """Intersection, not equality: a global span is relevant to a regional question."""
        outcome = ask("personal data", scope=ScopeSelector.parse("region:EU"))
        texts = [r.candidate.text for r in outcome.results]
        assert EU_RETENTION in texts
        assert RETENTION in texts

    def test_a_disjoint_sibling_scope_is_excluded(self) -> None:
        outcome = ask("personal data", scope=ScopeSelector.parse("region:US"))
        assert EU_RETENTION not in [r.candidate.text for r in outcome.results]
        assert outcome.trace.excluded_by_scope == 1

    def test_a_span_that_binds_nothing_is_counted_apart(self) -> None:
        """A coverage signal, not a mismatch: no scope filter can reach an unscoped span."""
        unscoped = candidate(REFUND, n=5, scopes=(), assumption_keys=())
        index = VectorRetriever(embed((unscoped,)), config=CONFIG)
        outcome = index.retrieve(query("refunds", scope=ScopeSelector.parse("region:EU")))

        assert outcome.is_empty
        assert outcome.trace.excluded_unscoped == 1
        assert outcome.trace.excluded_by_scope == 0


class TestTemporalFilter:
    def test_a_version_not_yet_in_force_is_excluded(self) -> None:
        outcome = ask("reset token ttl", as_of=AS_OF)
        assert TTL not in [r.candidate.text for r in outcome.results]
        assert outcome.trace.excluded_by_as_of == 1

    def test_the_same_span_returns_once_its_version_governs(self) -> None:
        outcome = ask("reset token ttl", as_of=dt.datetime(2026, 7, 1, tzinfo=dt.UTC))
        assert TTL in [r.candidate.text for r in outcome.results]
        assert outcome.trace.excluded_by_as_of == 0

    def test_no_as_of_applies_no_temporal_filter(self) -> None:
        outcome = ask("reset token ttl")
        assert outcome.trace.excluded_by_as_of == 0


class TestFilterOrder:
    def test_a_candidate_is_charged_to_exactly_one_bucket(self) -> None:
        """The trace buckets must sum to what was considered, or the trace cannot be used
        as negative provenance for an ``EvidenceGap``."""
        spans = (*corpus(), candidate(REFUND, n=9, tenant_id=OTHER_TENANT))
        outcome = VectorRetriever(embed(spans), config=CONFIG).retrieve(
            query(
                "personal data refunds",
                source_types=(SourceType.POLICY,),
                as_of=AS_OF,
                scope=ScopeSelector.parse("region:EU"),
            )
        )
        assert outcome.trace.excluded + outcome.trace.matched == outcome.trace.considered
        assert outcome.trace.considered == len(spans)

    def test_the_filter_order_matches_the_lexical_retrievers(self) -> None:
        """Source type before as-of, so a span failing both is charged to source type. The
        two strategies must exclude the same spans for the same reasons, or a filtered
        comparison between them is meaningless.
        """
        outcome = ask("reset token ttl", source_types=(SourceType.POLICY,), as_of=AS_OF)
        assert outcome.trace.excluded_by_source_type == 2
        assert outcome.trace.excluded_by_as_of == 0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_a_result_carries_the_stored_evidence_by_reference(self) -> None:
        """Nothing is flattened into a retrieval-local shape, so a citation taken from a
        vector result is the same citation the admission gate verified."""
        span = corpus()[0]
        outcome = VectorRetriever(embed((span,)), config=CONFIG).retrieve(query("refunds"))
        result = outcome.results[0]

        assert result.candidate.evidence is span.evidence
        assert result.candidate.version is span.version
        assert result.candidate.document is span.document

    def test_offsets_still_recover_the_quoted_text(self) -> None:
        span = corpus()[0]
        outcome = VectorRetriever(embed((span,)), config=CONFIG).retrieve(query("refunds"))
        item = outcome.results[0].candidate.evidence
        assert item.text_span[item.char_start : item.char_end] == item.text_span

    def test_a_mis_joined_candidate_cannot_be_indexed(self) -> None:
        """The domain contract refuses it at construction, and the vector path adds no way
        round - a citation assembled out of genuine parts is still fabricated.

        Constructed rather than ``model_copy``-ed, because ``model_copy`` skips validators
        and would make this test pass without asserting anything.
        """
        one, two = corpus()[0], corpus()[1]
        with pytest.raises(ValueError, match="cites version"):
            EvidenceCandidate(
                evidence=one.evidence,
                document=two.document,
                version=two.version,
                scopes=one.scopes,
                assumption_keys=one.assumption_keys,
            )

    def test_the_authority_comes_from_registration_not_from_similarity(self) -> None:
        outcome = ask("refunds")
        assert all(
            r.candidate.authority_class is r.candidate.document.authority_class
            for r in outcome.results
        )


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_nothing_matched_and_everything_filtered_are_distinguishable(self) -> None:
        """One empty list, two different observations about the corpus."""
        all_filtered = ask("refunds", source_types=(SourceType.EMAIL_TEMPLATE,))
        assert all_filtered.is_empty
        assert all_filtered.trace.excluded_by_source_type == 4
        assert all_filtered.trace.excluded_zero_score == 0

    def test_a_query_with_no_terms_reaches_nothing(self) -> None:
        """An empty embedding has no direction, so it matches nothing rather than
        everything - which is the failure mode that would make a vector retriever return
        the whole corpus for punctuation."""
        outcome = retriever().retrieve(query("!!! ???"))
        assert outcome.is_empty
        assert outcome.trace.excluded_zero_score == 4

    def test_an_unrelated_query_is_quiet_with_this_embedder(self) -> None:
        """Measured, and the *reason* matters more than the result.

        "bicycle maintenance schedule" scores exactly 0.0 against all four spans, so the
        retriever returns nothing - the same answer lexical retrieval gives. That is **not**
        a property of vector retrieval; it is a property of *this* embedder. Feature hashing
        over a bag of words is sparse and signed, so a text sharing no token with a span
        lands in no shared coordinate and the dot product is exactly zero.

        A real dense embedding model has no zeros: it would put almost every span at a
        positive cosine to almost any text, and this strategy would then return ``top_k``
        results for a question the corpus cannot answer. So the quietness measured here
        must **not** be reported as evidence that vector retrieval is quiet. Whether a
        similarity floor is needed is a question a recorded real provider answers, and the
        floor is deliberately not invented in advance (``docs/EVALUATION.md``).
        """
        outcome = ask("bicycle maintenance schedule")
        assert outcome.is_empty
        assert outcome.trace.excluded_zero_score == 4


class TestQueryValidation:
    def test_the_permission_scope_has_no_default(self) -> None:
        with pytest.raises(ValidationError, match="permission_scope"):
            RetrievalQuery(tenant_id=TENANT, text="refunds")  # type: ignore[call-arg]

    def test_top_k_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalQuery(
                tenant_id=TENANT,
                text="refunds",
                permission_scope=OPEN_SCOPE,
                top_k=MAX_TOP_K + 1,
            )


# ---------------------------------------------------------------------------
# The embedders
# ---------------------------------------------------------------------------


class TestHashingEmbedder:
    def test_it_is_deterministic_across_calls(self) -> None:
        assert EMBEDDER.embed_one(REFUND) == EMBEDDER.embed_one(REFUND)

    def test_it_is_deterministic_across_processes(self) -> None:
        """Uses ``hashlib``, never the built-in ``hash()``, which is salted per process.

        Pinned by asserting a fixed coordinate of a fixed text: if someone swaps in
        ``hash()``, this fails on the second run in a different interpreter rather than
        producing a benchmark nobody can reproduce.
        """
        vector = HashingEmbedder(dimensions=8).embed_one("refund")
        assert vector == pytest.approx([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_every_vector_has_the_declared_width(self) -> None:
        vectors = asyncio.run(EMBEDDER.embed([REFUND, RETENTION, ""]))
        assert all(len(v) == EMBEDDER.dimensions for v in vectors)

    def test_vectors_are_unit_length(self) -> None:
        vector = EMBEDDER.embed_one(REFUND)
        assert cosine(vector, vector) == pytest.approx(1.0)
        assert sum(c * c for c in vector) == pytest.approx(1.0)

    def test_a_text_with_no_terms_embeds_to_zero(self) -> None:
        assert EMBEDDER.embed_one("!!! ???") == [0.0] * EMBEDDER.dimensions

    def test_it_does_not_use_the_lexical_tokenizer(self) -> None:
        """Two strategies that share a preprocessing stage are closer to one strategy than
        to two, and checkpoint 9.3 counts corroboration across *distinct* strategies.

        The lexical tokenizer drops ``the`` and stems ``refunds`` to ``refund``; this one
        does neither, so the two indexes cannot agree merely because they normalized alike.
        """
        from app.infrastructure.embeddings.hashing import terms
        from app.retrieval.tokens import tokenize

        assert terms("the refunds") == ("the", "refunds")
        assert tokenize("the refunds") == ("refund",)

    def test_an_absurd_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            HashingEmbedder(dimensions=1)

    def test_it_needs_no_credentials_and_no_network(self) -> None:
        """Stated as a test so it cannot quietly stop being true
        (``ARCHITECTURE.md`` 11.4)."""
        assert asyncio.run(HashingEmbedder().embed(["anything"]))
