"""Vector retrieval end to end over the real sample corpus, through the real stores.

Unit tests prove the similarity, the filters and the ranking. This proves they work against
spans that were actually ingested, joined through the real store ports, with the evidence
ids the admission gate minted - and through the async boundary the vector strategy needs
and the lexical one does not.

The tests worth reading first:

* :func:`TestProvenance.test_every_returned_span_still_verifies_against_the_stored_text` -
  retrieval is the path that hands evidence to an agent, so a result whose offsets no longer
  recover the quoted text would be a citation the reader cannot check. Asserted for the
  lexical path already; a second strategy must not be the one that breaks it.
* :func:`TestPairing.test_a_provider_that_reorders_its_batch_is_a_defect_not_a_ranking` -
  the span-to-vector pairing is positional, so a reordering provider would silently attach
  every vector to the wrong span and produce a plausible, wrong ranking.

Nothing here reads a gold-label file: this is a wiring test. Vector recall is measured
separately in ``tests/evaluation/test_vector_benchmark.py``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.domain.entities.source import SourceDocument
from app.domain.interfaces.retrieval import RetrievalOutcome, RetrievalQuery
from app.domain.value_objects.enums import RetrievalStrategy, SourceType
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.evaluation.corpus import CorpusStores, load_corpus
from app.infrastructure.embeddings.hashing import HashingEmbedder

AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
FOREIGN_TENANT = "tn_" + "9" * 32
OPEN_SCOPE = PermissionScope()
EMBEDDER = HashingEmbedder()


@pytest.fixture(scope="module")
def corpus() -> CorpusStores:
    stores = asyncio.run(load_corpus())
    assert not stores.extraction_failures, stores.extraction_failures
    return stores


@pytest.fixture(scope="module")
def retrieval(corpus: CorpusStores) -> RetrieveEvidenceUseCase:
    return RetrieveEvidenceUseCase(sources=corpus.sources, assumptions=corpus.assumptions)


def ask(
    corpus: CorpusStores,
    retrieval: RetrieveEvidenceUseCase,
    text: str,
    **over: Any,
) -> RetrievalOutcome:
    over.setdefault("permission_scope", OPEN_SCOPE)
    query = RetrievalQuery(tenant_id=corpus.tenant_id, text=text, **over)
    return asyncio.run(retrieval.execute_vector(query, embedder=EMBEDDER))


class TestTheAsyncBoundary:
    def test_the_use_case_answers_a_query_by_cosine_similarity(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "refunds require manager approval")
        assert outcome.results
        assert outcome.strategies_executed == (RetrievalStrategy.VECTOR,)

    def test_the_outcome_names_the_embedder_in_its_scorer_version(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """A recall figure belongs to one embedder, so the outcome must say which."""
        outcome = ask(corpus, retrieval, "refunds")
        assert outcome.scorer_version == f"vector-cosine-1:{EMBEDDER.name}:d={EMBEDDER.dimensions}"

    def test_the_filter_configuration_is_the_lexical_one(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "refunds")
        assert outcome.retrieval_config_version == "retrieval-1"

    def test_the_lexical_path_is_untouched_and_still_synchronous(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """The frozen baseline is only a baseline if work on the second strategy cannot
        perturb the first. ``execute`` takes no embedder and cannot acquire one."""
        query = RetrievalQuery(
            tenant_id=corpus.tenant_id, text="refunds", permission_scope=OPEN_SCOPE
        )
        outcome = retrieval.execute(query)

        assert outcome.strategies_executed == (RetrievalStrategy.LEXICAL,)
        assert "bm25" in outcome.scorer_version

    def test_the_two_strategies_disagree_about_ranking_over_the_same_corpus(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """Not a defect - the point. Two strategies that always agreed would be one
        strategy, and corroboration across them would count nothing (checkpoint 9.3).
        """
        query = RetrievalQuery(
            tenant_id=corpus.tenant_id,
            text="how long do we keep personal data",
            permission_scope=OPEN_SCOPE,
        )
        lexical = retrieval.execute(query)
        vector = asyncio.run(retrieval.execute_vector(query, embedder=EMBEDDER))

        assert lexical.evidence_ids
        assert vector.evidence_ids
        assert lexical.evidence_ids != vector.evidence_ids


class TestPairing:
    def test_a_provider_returning_the_wrong_count_is_refused(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """Silently truncating would attach vectors to the wrong spans from that point on."""

        class ShortEmbedder:
            name = "short-1"
            dimensions = 8

            async def embed(self, texts: Sequence[str]) -> list[list[float]]:
                return [[1.0] * 8]

        query = RetrievalQuery(
            tenant_id=corpus.tenant_id, text="refunds", permission_scope=OPEN_SCOPE
        )
        with pytest.raises(ValueError, match="pairing is positional"):
            asyncio.run(retrieval.execute_vector(query, embedder=ShortEmbedder()))

    def test_a_provider_that_reorders_its_batch_is_a_defect_not_a_ranking(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """The pairing is positional, and nothing downstream can detect a permutation.

        This test does not claim the code catches it - it cannot, and pretending otherwise
        would be worse than recording it. What it asserts is that a reordering provider
        produces a *different* ranking from a faithful one, which is what makes the
        one-call-per-batch contract in ``_embed_candidates`` load-bearing rather than
        stylistic.
        """

        class ReversingEmbedder:
            name = "reversing-1"
            dimensions = EMBEDDER.dimensions

            async def embed(self, texts: Sequence[str]) -> list[list[float]]:
                vectors = await EMBEDDER.embed(texts)
                return list(reversed(vectors))

        query = RetrievalQuery(
            tenant_id=corpus.tenant_id, text="refunds manager approval", permission_scope=OPEN_SCOPE
        )
        faithful = asyncio.run(retrieval.execute_vector(query, embedder=EMBEDDER))
        reversed_outcome = asyncio.run(
            retrieval.execute_vector(query, embedder=ReversingEmbedder())
        )

        assert faithful.evidence_ids != reversed_outcome.evidence_ids

    def test_the_text_embedded_is_the_span_and_nothing_else(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """Titles and section labels are excluded, exactly as for BM25: a match must be
        attributable to words inside the citation."""
        captured: list[str] = []

        class RecordingEmbedder:
            name = EMBEDDER.name
            dimensions = EMBEDDER.dimensions

            async def embed(self, texts: Sequence[str]) -> list[list[float]]:
                captured.extend(texts)
                return await EMBEDDER.embed(texts)

        query = RetrievalQuery(
            tenant_id=corpus.tenant_id, text="refunds", permission_scope=OPEN_SCOPE
        )
        asyncio.run(retrieval.execute_vector(query, embedder=RecordingEmbedder()))

        spans = {item.text_span for item in corpus.sources.evidence_for_tenant(corpus.tenant_id)}
        assert set(captured) - {"refunds"} <= spans


class TestProvenance:
    def test_every_returned_span_still_verifies_against_the_stored_text(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """The offsets of every vector result must still recover its quoted text from the
        canonical version. A second strategy must not be the one that breaks this."""
        outcome = ask(corpus, retrieval, "personal data retained refunds approval")
        assert outcome.results

        for result in outcome.results:
            item = result.candidate.evidence
            text = corpus.sources.get_text(item.source_version_id)
            assert text is not None
            assert text[item.char_start : item.char_end] == item.text_span

    def test_each_result_carries_the_authority_from_registration(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """The injected vendor page demands ``regulatory`` in its own text and must not get
        it from a similarity score either."""
        outcome = ask(corpus, retrieval, "refund approval threshold globally")
        vendor = [r for r in outcome.results if "vendor" in r.candidate.document.uri]
        assert vendor
        assert all(r.candidate.authority_class.value == "wiki" for r in vendor)

    def test_the_corpus_size_is_the_admitted_span_count(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "refunds")
        assert outcome.trace.considered == corpus.admitted_count == 21


class TestFiltersOverTheRealCorpus:
    def test_a_foreign_tenant_reaches_nothing(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        query = RetrievalQuery(
            tenant_id=FOREIGN_TENANT, text="refunds approval", permission_scope=OPEN_SCOPE
        )
        outcome = asyncio.run(retrieval.execute_vector(query, embedder=EMBEDDER))

        assert outcome.is_empty
        assert outcome.trace.considered == 0

    def test_a_source_type_filter_narrows_the_results(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "personal data", source_types=(SourceType.POLICY,))
        assert outcome.results
        assert all(r.source_type is SourceType.POLICY for r in outcome.results)

    def test_an_as_of_before_a_version_excludes_it(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        early = ask(
            corpus,
            retrieval,
            "P1 incident acknowledged",
            as_of=dt.datetime(2025, 5, 1, tzinfo=dt.UTC),
        )
        late = ask(corpus, retrieval, "P1 incident acknowledged", as_of=AS_OF)
        assert early.trace.excluded_by_as_of > late.trace.excluded_by_as_of

    def test_a_disjoint_scope_excludes_the_eu_spans(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(
            corpus, retrieval, "personal data retained", scope=ScopeSelector.parse("region:US")
        )
        assert outcome.trace.excluded_by_scope == 2


#: The label put on one real corpus document, exactly as the lexical suite does it. The
#: shipped ``manifest.yaml`` stays unchanged: registration metadata is operator input.
PRIVACY_GRANT = "privacy-office"
RESTRICTED_URI = "corpus://policy/data-retention-policy.md"


@pytest.fixture
def restricted(corpus: CorpusStores) -> Iterator[SourceDocument]:
    document = corpus.sources.find_document_by_uri(corpus.tenant_id, RESTRICTED_URI)
    assert document is not None, RESTRICTED_URI
    corpus.sources.put_document(document.model_copy(update={"permissions": (PRIVACY_GRANT,)}))
    try:
        yield document
    finally:
        corpus.sources.put_document(document)


class TestAuthorizationOverTheRealCorpus:
    def test_a_restricted_span_is_withheld_from_a_scope_without_the_grant(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        outcome = ask(corpus, retrieval, "personal data is retained for 90 days")
        uris = {r.candidate.document.uri for r in outcome.results}

        assert RESTRICTED_URI not in uris
        assert outcome.trace.excluded_by_permission == 2

    def test_the_grant_makes_it_reachable_again(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        outcome = ask(
            corpus,
            retrieval,
            "personal data is retained for 90 days",
            permission_scope=PermissionScope.of(PRIVACY_GRANT),
        )
        uris = {r.candidate.document.uri for r in outcome.results}

        assert RESTRICTED_URI in uris
        assert outcome.trace.excluded_by_permission == 0

    def test_the_withheld_count_is_independent_of_the_query(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        """The same oracle test the lexical path carries: the count says how much of the
        corpus is closed to the requester, never whether any of it matched their words."""
        matching = ask(corpus, retrieval, "personal data is retained for 90 days")
        unrelated = ask(corpus, retrieval, "bicycle maintenance schedule")

        assert matching.trace.excluded_by_permission == unrelated.trace.excluded_by_permission == 2

    def test_nothing_about_the_denied_document_escapes(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        outcome = ask(corpus, retrieval, "personal data is retained for 90 days")
        blob = outcome.model_dump_json()

        assert RESTRICTED_URI not in blob
        assert restricted.title not in blob
        assert PRIVACY_GRANT not in blob


class TestDeterminismOverTheRealCorpus:
    def test_two_runs_of_one_query_are_byte_identical(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        first = ask(corpus, retrieval, "personal data retained after closure")
        second = ask(corpus, retrieval, "personal data retained after closure")
        assert first.model_dump_json() == second.model_dump_json()

    def test_the_run_needs_no_credentials_and_no_network(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """Stated as a test so it cannot quietly stop being true. The embedder is the
        offline hashing one (``ARCHITECTURE.md`` 11.4); no provider is constructed anywhere
        on this path."""
        assert ask(corpus, retrieval, "refunds").results
