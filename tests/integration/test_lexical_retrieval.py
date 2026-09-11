"""Lexical retrieval end to end over the real sample corpus, through the real stores.

Unit tests prove the filters and the ranking. This proves they work against spans that were
actually ingested, joined through the actual store ports, with the evidence ids the
admission gate minted - which is the only place a join defect can show itself.

The test worth reading first is
:func:`test_every_returned_span_still_verifies_against_the_stored_text`. Retrieval is the
path that hands evidence to an agent, so a result whose offsets no longer recover the quoted
text would be a citation the reader cannot check, and every guarantee above it rests on that
slice comparison holding.

Nothing here reads a gold-label file: this is a wiring test, not the retrieval benchmark.
Recall is measured separately, against labelled queries, in
``tests/evaluation/test_retrieval_benchmark.py``. Keeping the two apart matters: a wiring
failure and a recall regression need different fixes, and a file that asserted both would
report them as the same event.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Iterator

import pytest

from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.domain.entities.source import SourceDocument
from app.domain.interfaces.retrieval import RetrievalOutcome, RetrievalQuery
from app.domain.value_objects.enums import AuthorityClass, SourceType
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.evaluation.corpus import CorpusStores, load_corpus

#: Well after every version in the manifest comes into force.
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
FOREIGN_TENANT = "tn_" + "9" * 32

#: No document in the sample corpus carries permission labels, so the least-privileged
#: scope reads all of it. That is a property of the corpus, asserted in TestAuthorization
#: rather than assumed here.
OPEN_SCOPE = PermissionScope()


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
    **over: object,
) -> RetrievalOutcome:
    over.setdefault("permission_scope", OPEN_SCOPE)
    return retrieval.execute(RetrievalQuery(tenant_id=corpus.tenant_id, text=text, **over))


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


class TestCandidateJoin:
    def test_every_stored_span_becomes_a_candidate(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        built = retrieval.authorized_candidates(corpus.tenant_id, OPEN_SCOPE)
        stored = corpus.sources.evidence_for_tenant(corpus.tenant_id)
        assert built.unresolvable == ()
        assert len(built.candidates) == len(stored) == corpus.admitted_count

    def test_each_candidate_carries_the_authority_from_registration(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """Not from the document text. The injected corpus file demands ``regulatory``."""
        built = retrieval.authorized_candidates(corpus.tenant_id, OPEN_SCOPE)
        vendor = [c for c in built.candidates if "vendor" in c.document.uri]
        assert vendor
        assert all(c.authority_class is AuthorityClass.WIKI for c in vendor)

    def test_each_candidate_carries_the_keys_and_scopes_its_span_binds(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        built = retrieval.authorized_candidates(corpus.tenant_id, OPEN_SCOPE)
        assert all(c.assumption_keys for c in built.candidates)
        eu = [c for c in built.candidates if ScopeSelector.parse("region:EU") in c.scopes]
        assert {key for c in eu for key in c.assumption_keys} == {"retention.window"}

    def test_the_candidate_list_is_reproducible(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        first = retrieval.authorized_candidates(corpus.tenant_id, OPEN_SCOPE)
        second = retrieval.authorized_candidates(corpus.tenant_id, OPEN_SCOPE)
        assert first == second


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestRetrieval:
    def test_the_refund_threshold_question_reaches_the_policy_that_declares_it(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "refunds requiring manager approval 10,000 USD")
        uris = [r.candidate.document.uri for r in outcome.results]
        assert "corpus://policy/refund-approval-policy.md" in uris

    def test_a_threshold_written_with_a_separator_is_found_without_one(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """The corpus writes ``10,000``; a change request writes ``10000``."""
        with_separator = ask(corpus, retrieval, "10,000 USD refunds")
        without = ask(corpus, retrieval, "10000 USD refunds")
        assert without.evidence_ids == with_separator.evidence_ids
        assert without.results

    def test_a_registry_key_typed_as_a_query_reaches_the_prose_that_binds_it(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "retention.window")
        keys = {key for r in outcome.results for key in r.candidate.assumption_keys}
        assert "retention.window" in keys

    def test_every_returned_span_still_verifies_against_the_stored_text(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """The anti-hallucination primitive, re-asserted on the retrieval path.

        A retrieved result is a citation. If its offsets no longer recover the quoted text
        from the canonical version, the reader cannot check it - and every claim built on
        top of retrieval inherits that.
        """
        outcome = ask(corpus, retrieval, "retention personal data days refunds approval", top_k=20)
        assert outcome.results
        for result in outcome.results:
            evidence = result.candidate.evidence
            text = corpus.sources.get_text(evidence.source_version_id)
            assert text is not None
            assert text[evidence.char_start : evidence.char_end] == evidence.text_span

    def test_every_returned_evidence_id_resolves_in_the_store(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "personal data retained", top_k=20)
        assert outcome.results
        for result in outcome.results:
            assert corpus.sources.get_evidence(result.evidence_id) == result.candidate.evidence

    def test_two_runs_of_one_query_are_byte_identical(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        first = ask(corpus, retrieval, "personal data retained days")
        second = ask(corpus, retrieval, "personal data retained days")
        assert first.model_dump_json() == second.model_dump_json()


# ---------------------------------------------------------------------------
# Filters against real registration metadata
# ---------------------------------------------------------------------------


class TestFilters:
    def test_a_source_type_filter_returns_only_that_type(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(
            corpus,
            retrieval,
            "personal data retained days",
            source_types=(SourceType.POLICY,),
            top_k=20,
        )
        assert outcome.results
        assert {r.source_type for r in outcome.results} == {SourceType.POLICY}
        assert outcome.trace.excluded_by_source_type > 0

    def test_a_source_type_absent_from_the_corpus_returns_nothing(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "personal data", source_types=(SourceType.EMAIL_TEMPLATE,))
        assert outcome.is_empty
        assert outcome.trace.excluded_by_source_type == outcome.trace.considered

    def test_a_disjoint_region_excludes_the_eu_spans(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """``region:US`` and ``region:EU`` are siblings, so they do not intersect."""
        eu = ask(
            corpus,
            retrieval,
            "personal data retained days",
            scope=ScopeSelector.parse("region:EU"),
            top_k=20,
        )
        us = ask(
            corpus,
            retrieval,
            "personal data retained days",
            scope=ScopeSelector.parse("region:US"),
            top_k=20,
        )
        assert set(us.evidence_ids) < set(eu.evidence_ids)
        assert us.trace.excluded_by_scope == 2

    def test_a_global_declaration_survives_a_regional_filter(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """Inheritance: the global retention policy still governs an EU question."""
        outcome = ask(
            corpus,
            retrieval,
            "personal data retained days",
            scope=ScopeSelector.parse("region:EU"),
            top_k=20,
        )
        uris = {r.candidate.document.uri for r in outcome.results}
        assert "corpus://policy/data-retention-policy.md" in uris

    def test_an_as_of_before_a_version_comes_into_force_excludes_it(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """The second P1 runbook is effective 2025-09-01, so a March query cannot see it."""
        march = ask(
            corpus,
            retrieval,
            "P1 incident acknowledged minutes",
            as_of=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            top_k=20,
        )
        later = ask(corpus, retrieval, "P1 incident acknowledged minutes", as_of=AS_OF, top_k=20)
        assert set(march.evidence_ids) < set(later.evidence_ids)
        assert march.trace.excluded_by_as_of > 0
        assert later.trace.excluded_by_as_of == 0

    def test_an_as_of_before_the_whole_corpus_returns_nothing(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(
            corpus,
            retrieval,
            "refunds approval",
            as_of=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
        )
        assert outcome.is_empty
        assert outcome.trace.excluded_by_as_of == outcome.trace.considered


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


class TestNegative:
    def test_a_question_the_corpus_does_not_answer_returns_nothing(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        outcome = ask(corpus, retrieval, "bicycle helmet procurement schedule")
        assert outcome.is_empty
        assert outcome.trace.excluded_zero_score == outcome.trace.considered

    def test_another_tenant_sees_an_empty_corpus(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """Tenant is a store parameter, so a foreign tenant has nothing to consider."""
        outcome = retrieval.execute(
            RetrievalQuery(
                tenant_id=FOREIGN_TENANT, text="refunds approval", permission_scope=OPEN_SCOPE
            )
        )
        assert outcome.is_empty
        assert outcome.trace.considered == 0

    def test_the_store_will_not_enumerate_another_tenants_spans(self, corpus: CorpusStores) -> None:
        assert corpus.sources.evidence_for_tenant(FOREIGN_TENANT) == ()

    def test_retrieval_makes_no_model_call(
        self, corpus: CorpusStores, retrieval: RetrieveEvidenceUseCase
    ) -> None:
        """There is no seam for one: the use case takes stores, and nothing else.

        Asserted structurally rather than by counting calls - a retriever that could be
        handed a gateway would already have failed ``tests/contract/test_layering.py``.
        """
        outcome = ask(corpus, retrieval, "refunds approval")
        assert outcome.strategies_executed == (outcome.results[0].strategy,)
        assert "bm25" in outcome.scorer_version


# ---------------------------------------------------------------------------
# Authorization over the real corpus
# ---------------------------------------------------------------------------

#: The label put on one real corpus document by :func:`restricted`. The shipped
#: ``manifest.yaml`` deliberately stays unchanged: registration metadata is operator input,
#: and rewriting it to make a test pass would change what every other suite ingests.
PRIVACY_GRANT = "privacy-office"
RESTRICTED_URI = "corpus://policy/data-retention-policy.md"


@pytest.fixture
def restricted(corpus: CorpusStores) -> Iterator[SourceDocument]:
    """Label the retention policy, then put it back.

    Re-registering the document through the store port is exactly what an operator applying
    an ACL does, so the join, the retriever and the trace are all exercised against a real
    restricted document rather than a fixture-shaped one.
    """
    document = corpus.sources.find_document_by_uri(corpus.tenant_id, RESTRICTED_URI)
    assert document is not None, RESTRICTED_URI
    corpus.sources.put_document(document.model_copy(update={"permissions": (PRIVACY_GRANT,)}))
    try:
        yield document
    finally:
        corpus.sources.put_document(document)


class TestAuthorization:
    def test_the_shipped_corpus_carries_no_labels(self, corpus: CorpusStores) -> None:
        """Stated rather than assumed: it is why ``OPEN_SCOPE`` reads the whole corpus here.

        The benchmark works around the same gap by declaring an ACL overlay in its label
        file and applying it through the store port, which is enough to measure
        authorization-filtered recall on one document and not enough to measure it under a
        realistic label distribution (``docs/STATUS.md`` 30).
        """
        built = corpus.sources.evidence_for_tenant(corpus.tenant_id)
        documents = {
            corpus.sources.get_version(item.source_version_id).source_id  # type: ignore[union-attr]
            for item in built
        }
        assert documents
        assert all(
            corpus.sources.get_document(corpus.tenant_id, source_id).permissions == ()  # type: ignore[union-attr]
            for source_id in documents
        )

    def test_a_restricted_document_is_withheld_without_the_grant(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        outcome = ask(corpus, retrieval, "personal data retained days", top_k=20)
        uris = {r.candidate.document.uri for r in outcome.results}
        assert RESTRICTED_URI not in uris
        assert outcome.trace.excluded_by_permission > 0

    def test_the_same_query_returns_it_with_the_grant(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        granted = ask(
            corpus,
            retrieval,
            "personal data retained days",
            permission_scope=PermissionScope.of(PRIVACY_GRANT),
            top_k=20,
        )
        uris = {r.candidate.document.uri for r in granted.results}
        assert RESTRICTED_URI in uris
        assert granted.trace.excluded_by_permission == 0

    def test_the_two_scopes_are_distinguishable_by_hash(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        """A differing scope must force recomputation, never a filtered replay
        (``ARCHITECTURE.md`` 14.1), which needs the two runs to be telling apart at all."""
        without = ask(corpus, retrieval, "personal data retained days", top_k=20)
        with_grant = ask(
            corpus,
            retrieval,
            "personal data retained days",
            permission_scope=PermissionScope.of(PRIVACY_GRANT),
            top_k=20,
        )
        assert without.permission_scope_hash != with_grant.permission_scope_hash
        assert set(without.evidence_ids) < set(with_grant.evidence_ids)

    def test_nothing_about_the_restricted_document_reaches_the_caller(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        """Serialized whole and searched for the document's own identifiers."""
        outcome = ask(corpus, retrieval, "personal data retained days", top_k=20)
        serialized = outcome.model_dump_json()
        version = corpus.sources.versions_for(restricted.source_id)[-1]
        for secret in (restricted.source_id, restricted.uri, restricted.title, version.version_id):
            assert secret not in serialized, secret

    def test_the_public_candidate_accessor_filters_too(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        """The retriever is not the only door; the join must not hand out what it excludes."""
        closed = retrieval.authorized_candidates(corpus.tenant_id, OPEN_SCOPE)
        open_to_privacy = retrieval.authorized_candidates(
            corpus.tenant_id, PermissionScope.of(PRIVACY_GRANT)
        )
        assert all(c.document.uri != RESTRICTED_URI for c in closed.candidates)
        assert any(c.document.uri == RESTRICTED_URI for c in open_to_privacy.candidates)

    def test_a_grant_does_not_leak_across_tenants(
        self,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        restricted: SourceDocument,
    ) -> None:
        outcome = retrieval.execute(
            RetrievalQuery(
                tenant_id=FOREIGN_TENANT,
                text="personal data retained days",
                permission_scope=PermissionScope.of(PRIVACY_GRANT),
            )
        )
        assert outcome.is_empty
        assert outcome.trace.considered == 0
