"""The retrieval orchestration layer: selection, refusal, propagation, one shape.

This layer adds *dispatch* and nothing else, so the tests are about what it must not do:
not choose a strategy nobody asked for, not silently substitute one when the requested one
cannot run, not lose the permission scope on the way through, and not return a different
shape depending on which strategy answered.

The tests carrying the most weight:

* :func:`TestRefusal.test_a_strategy_needing_an_embedder_is_refused_not_downgraded` - a
  caller who asked for semantic retrieval and received lexical results has no way to know.
* :func:`TestPermissionPropagation.test_the_requested_scope_reaches_every_strategy` - the
  scope is the authorization decision; a layer that dropped it would widen access.
* :func:`TestOutputConsistency.test_the_three_strategies_agree_on_their_filter_accounting` -
  the observable form of "every filter guarantee is inherited, not restated".
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest
from pydantic import ValidationError

from app.application.orchestration.retrieval_service import (
    EMBEDDING_STRATEGIES,
    SELECTABLE_STRATEGIES,
    EmbedderRequired,
    FinalizedRetrieval,
    RetrievalConfigurationError,
    RetrievalRequest,
    RetrievalService,
    UnknownRetrievalStrategy,
    parse_strategy,
)
from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.domain.interfaces.retrieval import RetrievalOutcome, RetrievalQuery
from app.domain.value_objects.enums import RetrievalStrategy, SourceType
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.evaluation.corpus import CorpusStores, load_corpus
from app.infrastructure.embeddings.hashing import HashingEmbedder
from tests.conftest import make_settings

EMBEDDER = HashingEmbedder()
PRIVACY_GRANT = "privacy-office"
RESTRICTED_URI = "corpus://policy/data-retention-policy.md"


@pytest.fixture(scope="module")
def corpus() -> CorpusStores:
    stores = asyncio.run(load_corpus())
    assert not stores.extraction_failures, stores.extraction_failures
    return stores


@pytest.fixture(scope="module")
def use_case(corpus: CorpusStores) -> RetrieveEvidenceUseCase:
    return RetrieveEvidenceUseCase(sources=corpus.sources, assumptions=corpus.assumptions)


@pytest.fixture
def service(use_case: RetrieveEvidenceUseCase) -> RetrievalService:
    return RetrievalService(retrieval=use_case, embedder=EMBEDDER)


def query(corpus: CorpusStores, **over: Any) -> RetrievalQuery:
    over.setdefault("permission_scope", PermissionScope())
    over.setdefault("text", "refunds require manager approval")
    return RetrievalQuery(tenant_id=corpus.tenant_id, **over)


def ask(
    service: RetrievalService,
    corpus: CorpusStores,
    strategy: RetrievalStrategy | None = None,
    **over: Any,
) -> FinalizedRetrieval:
    request = RetrievalRequest(query=query(corpus, **over), strategy=strategy)
    return asyncio.run(service.retrieve(request))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestSelection:
    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_every_selectable_strategy_can_be_requested(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        finalized = ask(service, corpus, strategy)
        assert finalized.strategy is strategy
        assert finalized.results

    def test_the_configured_default_answers_a_request_that_names_none(
        self, use_case: RetrieveEvidenceUseCase, corpus: CorpusStores
    ) -> None:
        """A caller who does not care about the strategy should not have to name one."""
        service = RetrievalService(
            retrieval=use_case, default_strategy=RetrievalStrategy.VECTOR, embedder=EMBEDDER
        )
        assert ask(service, corpus).strategy is RetrievalStrategy.VECTOR

    def test_a_request_overrides_the_configured_default(
        self, use_case: RetrieveEvidenceUseCase, corpus: CorpusStores
    ) -> None:
        service = RetrievalService(
            retrieval=use_case, default_strategy=RetrievalStrategy.LEXICAL, embedder=EMBEDDER
        )
        assert ask(service, corpus, RetrievalStrategy.HYBRID).strategy is RetrievalStrategy.HYBRID

    def test_the_shipped_default_is_the_strongest_measured_strategy(self) -> None:
        """`lexical` is the default because it measures best and needs no embedder, not
        because it was first (``docs/EVALUATION.md``)."""
        assert make_settings().retrieval_strategy == "lexical"
        assert parse_strategy(make_settings().retrieval_strategy) is RetrievalStrategy.LEXICAL

    def test_a_configured_name_is_accepted_as_a_string(
        self, use_case: RetrieveEvidenceUseCase
    ) -> None:
        """Which is how it arrives from ``CCE_RETRIEVAL_STRATEGY``."""
        service = RetrievalService(retrieval=use_case, default_strategy="hybrid", embedder=EMBEDDER)
        assert service.default_strategy is RetrievalStrategy.HYBRID

    def test_available_strategies_reflect_what_the_service_holds(
        self, use_case: RetrieveEvidenceUseCase
    ) -> None:
        """Reported rather than discovered by failure, so a caller can present the choice
        honestly instead of offering one that raises."""
        with_embedder = RetrievalService(retrieval=use_case, embedder=EMBEDDER)
        without = RetrievalService(retrieval=use_case)

        assert with_embedder.available_strategies() == SELECTABLE_STRATEGIES
        assert without.available_strategies() == (RetrievalStrategy.LEXICAL,)
        assert with_embedder.has_embedder and not without.has_embedder

    def test_selection_covers_every_member_of_the_enum(self) -> None:
        """A strategy added to the enum without a handler must not resolve to a default.
        This is what turns that into a failing test rather than a silent fall-through."""
        assert set(SELECTABLE_STRATEGIES) == set(RetrievalStrategy)


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_an_unknown_configured_name_is_refused_and_names_the_alternatives(self) -> None:
        with pytest.raises(UnknownRetrievalStrategy, match="known strategies are"):
            parse_strategy("semantic")

    def test_an_unknown_name_is_refused_at_construction(
        self, use_case: RetrieveEvidenceUseCase
    ) -> None:
        """So a misconfiguration surfaces when the service is built, not mid-request."""
        with pytest.raises(UnknownRetrievalStrategy):
            RetrievalService(retrieval=use_case, default_strategy="graph")

    def test_an_unknown_name_on_a_request_cannot_be_constructed(self, corpus: CorpusStores) -> None:
        """The request carries the enum, so Pydantic refuses the name before a request
        object exists - a different mechanism from the configured path, and both must fail."""
        with pytest.raises(ValidationError):
            RetrievalRequest(query=query(corpus), strategy="reranked")

    @pytest.mark.parametrize("strategy", sorted(EMBEDDING_STRATEGIES))
    def test_a_strategy_needing_an_embedder_is_refused_not_downgraded(
        self,
        use_case: RetrieveEvidenceUseCase,
        corpus: CorpusStores,
        strategy: RetrievalStrategy,
    ) -> None:
        """The refusal that matters most.

        Falling back to lexical would give a caller who asked for semantic retrieval a
        result they cannot distinguish from the real thing, and would attribute a figure to
        a strategy that never ran.
        """
        service = RetrievalService(retrieval=use_case)
        with pytest.raises(EmbedderRequired, match="not a fallback"):
            ask(service, corpus, strategy)

    def test_the_refusal_names_what_is_available_instead(
        self, use_case: RetrieveEvidenceUseCase, corpus: CorpusStores
    ) -> None:
        service = RetrievalService(retrieval=use_case)
        with pytest.raises(EmbedderRequired, match="lexical"):
            ask(service, corpus, RetrievalStrategy.VECTOR)

    def test_lexical_still_works_without_an_embedder(
        self, use_case: RetrieveEvidenceUseCase, corpus: CorpusStores
    ) -> None:
        """The refusal above is narrow: it must not disable the strategy that needs nothing."""
        service = RetrievalService(retrieval=use_case)
        assert ask(service, corpus, RetrievalStrategy.LEXICAL).results

    def test_both_refusals_share_one_catchable_base(self) -> None:
        """A caller handling misconfiguration should not have to enumerate its kinds."""
        assert issubclass(UnknownRetrievalStrategy, RetrievalConfigurationError)
        assert issubclass(EmbedderRequired, RetrievalConfigurationError)
        assert issubclass(RetrievalConfigurationError, ValueError)

    def test_a_request_refuses_an_unknown_field(self, corpus: CorpusStores) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(query=query(corpus), rerank=True)  # type: ignore[call-arg]

    def test_the_envelope_refuses_inconsistent_hybrid_detail(
        self, service: RetrievalService, corpus: CorpusStores
    ) -> None:
        """Contribution detail is present exactly for a fused run. Either half missing
        would let a consumer read strategy provenance that does not exist, or miss some
        that does."""
        lexical = ask(service, corpus, RetrievalStrategy.LEXICAL)
        with pytest.raises(ValueError, match="exactly for a hybrid run"):
            FinalizedRetrieval(strategy=RetrievalStrategy.HYBRID, outcome=lexical.outcome)


# ---------------------------------------------------------------------------
# Permission propagation
# ---------------------------------------------------------------------------


class TestPermissionPropagation:
    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_the_requested_scope_reaches_every_strategy(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        """The scope *is* the authorization decision. A dispatch layer that dropped or
        substituted it would widen access without anyone deciding to."""
        scope = PermissionScope.of(PRIVACY_GRANT)
        finalized = ask(service, corpus, strategy, permission_scope=scope)

        assert finalized.permission_scope_hash == scope.scope_hash
        assert finalized.outcome.query.permission_scope == scope

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_two_scopes_produce_distinguishable_outcomes(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        """``ARCHITECTURE.md`` 14.1: a differing scope must force recomputation rather than
        a filtered replay, which needs the two to be distinguishable by hash."""
        open_scope = ask(service, corpus, strategy, permission_scope=PermissionScope())
        granted = ask(service, corpus, strategy, permission_scope=PermissionScope.of(PRIVACY_GRANT))
        assert open_scope.permission_scope_hash != granted.permission_scope_hash

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_a_restricted_span_is_withheld_under_every_strategy(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        """Authorization is enforced inside each retriever, and the orchestration layer
        must not be a way round it. The retention policy is labelled for the duration of
        this test, exactly as an operator applying an ACL would.
        """
        document = corpus.sources.find_document_by_uri(corpus.tenant_id, RESTRICTED_URI)
        assert document is not None
        corpus.sources.put_document(document.model_copy(update={"permissions": (PRIVACY_GRANT,)}))
        try:
            withheld = ask(
                service,
                corpus,
                strategy,
                text="personal data is retained for 90 days",
                permission_scope=PermissionScope(),
            )
            granted = ask(
                service,
                corpus,
                strategy,
                text="personal data is retained for 90 days",
                permission_scope=PermissionScope.of(PRIVACY_GRANT),
            )
        finally:
            corpus.sources.put_document(document)

        assert RESTRICTED_URI not in {r.candidate.document.uri for r in withheld.outcome.results}
        assert RESTRICTED_URI in {r.candidate.document.uri for r in granted.outcome.results}
        assert withheld.trace.excluded_by_permission == 2
        assert granted.trace.excluded_by_permission == 0

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_the_tenant_comes_from_the_request_and_is_not_inherited(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        foreign = RetrievalQuery(
            tenant_id="tn_" + "9" * 32,
            text="refunds approval",
            permission_scope=PermissionScope.of(PRIVACY_GRANT),
        )
        finalized = asyncio.run(
            service.retrieve(RetrievalRequest(query=foreign, strategy=strategy))
        )
        assert finalized.is_empty


# ---------------------------------------------------------------------------
# Filters still apply
# ---------------------------------------------------------------------------


class TestFiltersAreInherited:
    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_a_source_type_filter_narrows_every_strategy(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        finalized = ask(
            service,
            corpus,
            strategy,
            text="personal data",
            source_types=(SourceType.POLICY,),
        )
        assert finalized.results
        assert all(r.source_type is SourceType.POLICY for r in finalized.outcome.results)

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_a_disjoint_scope_excludes_the_same_spans_everywhere(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        finalized = ask(
            service,
            corpus,
            strategy,
            text="personal data retained",
            scope=ScopeSelector.parse("region:US"),
        )
        assert finalized.trace.excluded_by_scope == 2

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_a_temporal_filter_applies_under_every_strategy(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        import datetime as dt

        early = ask(
            service,
            corpus,
            strategy,
            text="P1 incident acknowledged",
            as_of=dt.datetime(2025, 5, 1, tzinfo=dt.UTC),
        )
        late = ask(
            service,
            corpus,
            strategy,
            text="P1 incident acknowledged",
            as_of=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        )
        assert early.trace.excluded_by_as_of > late.trace.excluded_by_as_of

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_provenance_still_verifies_under_every_strategy(
        self,
        service: RetrievalService,
        corpus: CorpusStores,
        strategy: RetrievalStrategy,
    ) -> None:
        """Retrieval is the path that hands evidence to an agent, so a result whose offsets
        no longer recover the quoted text would be a citation nobody can check."""
        finalized = ask(service, corpus, strategy, text="personal data refunds approval")
        assert finalized.results

        for result in finalized.outcome.results:
            item = result.candidate.evidence
            text = corpus.sources.get_text(item.source_version_id)
            assert text is not None
            assert text[item.char_start : item.char_end] == item.text_span


# ---------------------------------------------------------------------------
# One shape, whichever ran
# ---------------------------------------------------------------------------


class TestOutputConsistency:
    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_every_strategy_returns_the_same_outcome_type(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        """So a downstream consumer needs no strategy-specific branch."""
        finalized = ask(service, corpus, strategy)
        assert isinstance(finalized.outcome, RetrievalOutcome)

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_every_strategy_reports_the_fields_a_caller_needs(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        finalized = ask(service, corpus, strategy)

        assert finalized.strategy is strategy
        assert finalized.retrieval_config_version == "retrieval-1"
        assert finalized.permission_scope_hash
        assert finalized.trace.considered == 21
        assert finalized.scorer_version
        assert finalized.evidence_ids == finalized.outcome.evidence_ids

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_the_scorer_version_identifies_the_strategy_that_ran(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        """Three strategies, three fingerprints - so a figure cannot be attributed to the
        wrong one."""
        expected = {
            RetrievalStrategy.LEXICAL: "lexical-bm25-1",
            RetrievalStrategy.VECTOR: "vector-cosine-1",
            RetrievalStrategy.HYBRID: "hybrid-rrf-1",
        }
        assert ask(service, corpus, strategy).scorer_version.startswith(expected[strategy])

    def test_the_selected_strategy_is_not_always_what_executed(
        self, service: RetrievalService, corpus: CorpusStores
    ) -> None:
        """The reason the envelope carries ``strategy`` at all.

        A fused run reports ``(lexical, vector)`` in ``strategies_executed`` because both
        genuinely ran and corroboration counts distinct strategies (checkpoint 9.3), so the
        selected strategy has nowhere else to live.
        """
        hybrid = ask(service, corpus, RetrievalStrategy.HYBRID)
        assert hybrid.strategy is RetrievalStrategy.HYBRID
        assert hybrid.strategies_executed == (RetrievalStrategy.LEXICAL, RetrievalStrategy.VECTOR)

    @pytest.mark.parametrize("strategy", [RetrievalStrategy.LEXICAL, RetrievalStrategy.VECTOR])
    def test_a_single_strategy_run_carries_no_contribution_detail(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        assert ask(service, corpus, strategy).hybrid is None

    def test_a_fused_run_preserves_the_per_strategy_contributions(
        self, service: RetrievalService, corpus: CorpusStores
    ) -> None:
        """`lexical-only` / `vector-only` / `both` must survive the trip through a uniform
        interface, or the uniform interface has cost the thing hybrid exists to report."""
        hybrid = ask(service, corpus, RetrievalStrategy.HYBRID)
        assert hybrid.hybrid is not None
        assert hybrid.hybrid.results
        assert len(hybrid.hybrid.by_both) + len(hybrid.hybrid.lexical_only) + len(
            hybrid.hybrid.vector_only
        ) == len(hybrid.hybrid.results)

    def test_the_three_strategies_agree_on_their_filter_accounting(
        self, service: RetrievalService, corpus: CorpusStores
    ) -> None:
        """The observable form of "every filter guarantee is inherited, not restated".

        All three narrow the same candidate set to the same tenant and the same authorized
        slice, then apply the same filters in the same fixed order. Only the *scoring*
        buckets may differ, so a disagreement on the filter buckets would mean the
        orchestration layer had changed what a strategy sees.
        """
        traces = [
            ask(
                service,
                corpus,
                strategy,
                text="personal data retained",
                source_types=(SourceType.POLICY,),
                scope=ScopeSelector.parse("region:EU"),
            ).trace
            for strategy in SELECTABLE_STRATEGIES
        ]
        for bucket in (
            "considered",
            "excluded_by_tenant",
            "excluded_by_permission",
            "excluded_by_source_type",
            "excluded_by_as_of",
            "excluded_by_scope",
            "excluded_unscoped",
        ):
            values = {getattr(trace, bucket) for trace in traces}
            assert len(values) == 1, f"{bucket} differs across strategies: {values}"

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_the_trace_buckets_still_sum_to_considered(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        """What makes the trace usable as negative provenance for an ``EvidenceGap``."""
        trace = ask(service, corpus, strategy, text="personal data").trace
        assert trace.excluded + trace.matched == trace.considered

    @pytest.mark.parametrize("strategy", sorted(SELECTABLE_STRATEGIES))
    def test_two_runs_of_one_request_are_identical(
        self, service: RetrievalService, corpus: CorpusStores, strategy: RetrievalStrategy
    ) -> None:
        first = ask(service, corpus, strategy, text="personal data retained")
        second = ask(service, corpus, strategy, text="personal data retained")
        assert first.outcome.model_dump_json() == second.outcome.model_dump_json()


class TestNothingUnderneathChanged:
    def test_the_service_delegates_rather_than_retrieving(
        self, service: RetrievalService, corpus: CorpusStores, use_case: RetrieveEvidenceUseCase
    ) -> None:
        """Dispatched output must equal what the use case returns directly, or the
        orchestration layer has become a fourth strategy."""
        q = query(corpus, text="refunds manager approval")
        direct = use_case.execute(q)
        dispatched = asyncio.run(
            service.retrieve(RetrievalRequest(query=q, strategy=RetrievalStrategy.LEXICAL))
        )
        assert dispatched.outcome.model_dump_json() == direct.model_dump_json()

    def test_the_three_fingerprints_are_the_frozen_ones(self) -> None:
        """The orchestration layer must not have perturbed a measured baseline."""
        from app.retrieval.hybrid import HybridConfig
        from app.retrieval.lexical import LexicalConfig
        from app.retrieval.vectors import VectorConfig

        lexical = LexicalConfig().fingerprint()
        vector = VectorConfig(embedder="hashing-1", dimensions=256).fingerprint()

        assert lexical == "lexical-bm25-1:tokens-3:k1=1.2/b=0.75"
        assert vector == "vector-cosine-1:hashing-1:d=256"
        assert HybridConfig().fingerprint(lexical=lexical, vector=vector) == (
            "hybrid-rrf-1:k=60:"
            "[lexical-bm25-1:tokens-3:k1=1.2/b=0.75]+[vector-cosine-1:hashing-1:d=256]"
        )

    def test_the_service_imports_no_infrastructure_and_no_provider(self) -> None:
        """It takes an ``EmbeddingPort`` and cannot construct one.

        Checked by parsing the imports rather than scanning the text - the first version of
        this test grepped for the string and matched the docstring that explains the rule,
        which is a test that reads correctly and asserts nothing.
        ``tests/contract/test_layering.py`` enforces the same rule for the whole layer;
        this pins it for the module that would most plausibly break it.
        """
        import ast

        import app.application.orchestration.retrieval_service as module

        assert module.__file__ is not None
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module)

        assert not any(name.startswith("app.infrastructure") for name in imported), imported
        assert not any(name.startswith("app.agents") for name in imported), imported
        for vendor in ("httpx", "groq", "google", "mistralai", "ollama"):
            assert vendor not in imported
