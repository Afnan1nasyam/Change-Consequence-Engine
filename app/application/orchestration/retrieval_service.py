"""One entry point for retrieval. Selects a strategy, returns one shape.

Three strategies exist and each is reached differently: :meth:`~RetrieveEvidenceUseCase.execute`
is synchronous, :meth:`~RetrieveEvidenceUseCase.execute_vector` is async and needs an
embedder, and :meth:`~RetrieveEvidenceUseCase.execute_hybrid` needs both and returns a
richer object. That asymmetry is correct one layer down - it reflects what each strategy
genuinely requires - and it is wrong to push onto every caller. This module is the seam that
absorbs it.

**It orchestrates and finalizes. It does not retrieve.** Every call below delegates to the
use case, which delegates to a retriever. No filter, no scoring, no ranking and no merging
happens here, so every guarantee is inherited rather than restated: tenant isolation and
authorization are applied inside each retriever by narrowing the index before anything is
scored, source-type, scope and temporal filters run in the same fixed order in all three, and
provenance is carried by reference so a citation from any strategy is the one the admission
gate verified. A test asserts the three strategies agree on their filter accounting for the
same query, which is the observable form of that claim.

**Selection is configuration, overridable per request.** ``CCE_RETRIEVAL_STRATEGY`` names the
default; a request may name a different one. Neither is inferred from what happens to be
available - in particular, asking for ``vector`` or ``hybrid`` without an embedder is an
error and never a silent fall back to ``lexical``. A caller who asked for semantic retrieval
and got lexical results would have no way to know, and a benchmark figure produced that way
would be attributed to the wrong strategy.

**An unknown strategy fails explicitly, in both directions.** A request carries a
:class:`~app.domain.value_objects.enums.RetrievalStrategy`, so an unknown name is a
validation error before the request exists. A misconfigured ``CCE_RETRIEVAL_STRATEGY`` raises
:class:`UnknownRetrievalStrategy` naming what is available. And dispatch is a table rather
than an ``if`` chain, so a strategy member added to the enum without a handler here raises
instead of falling through to a default.

What every strategy returns
---------------------------

:class:`FinalizedRetrieval` wraps the outcome rather than replacing it. ``.outcome`` is a
plain :class:`~app.domain.interfaces.retrieval.RetrievalOutcome` for all three strategies -
same type, same validators, same fields - so a downstream consumer needs no strategy-specific
branch. The envelope adds the two things the contract cannot express:

``strategy``
    the strategy that was *selected*, which is not always what ``strategies_executed`` says:
    a hybrid run reports ``(lexical, vector)`` there, because both genuinely ran and
    corroboration counts distinct strategies (checkpoint 9.3).
``hybrid``
    the per-strategy rank contributions, present only for a fused run and ``None``
    otherwise, so ``lexical-only`` / ``vector-only`` / ``both`` survives the trip through a
    uniform interface.

Deliberately absent: reranking, query expansion, caching, retry, a QueryPlan that runs
several strategies and merges them, and any new retrieval algorithm. This layer adds
*dispatch*, and adding scoring to it would put a fourth strategy in the one place nobody
would look for one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict

from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.domain.interfaces.llm import EmbeddingPort
from app.domain.interfaces.retrieval import RetrievalOutcome, RetrievalQuery, RetrievalTrace
from app.domain.value_objects.enums import RetrievalStrategy
from app.retrieval.hybrid import HybridConfig, HybridOutcome

#: The strategies this service can dispatch. Every member of the enum must appear here or
#: :meth:`RetrievalService.retrieve` raises - which is what stops a newly added strategy
#: silently resolving to a default.
SELECTABLE_STRATEGIES: Final = (
    RetrievalStrategy.LEXICAL,
    RetrievalStrategy.VECTOR,
    RetrievalStrategy.HYBRID,
)

#: The strategies that cannot answer a query without an embedder.
EMBEDDING_STRATEGIES: Final = (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID)


class RetrievalConfigurationError(ValueError):
    """The retrieval request or configuration cannot be served as written.

    A ``ValueError`` rather than a provider error: nothing was attempted and no provider was
    involved. It is raised before any store read, so a misconfiguration surfaces at the
    boundary and never as a half-executed retrieval.
    """


class UnknownRetrievalStrategy(RetrievalConfigurationError):
    """A strategy name that does not exist, or exists and has no handler here."""


class EmbedderRequired(RetrievalConfigurationError):
    """A strategy needing an embedder was selected without one.

    Deliberately not a fallback to lexical. A caller who asked for semantic retrieval and
    quietly received lexical results has no way to detect it, and a figure produced that way
    would be attributed to a strategy that never ran.
    """


class RetrievalRequest(BaseModel):
    """One retrieval request: the query, and optionally which strategy should answer it.

    ``strategy`` is typed as the enum, so an unknown name is a validation error before a
    request object exists. ``None`` means "use the configured default", which is the case a
    caller who does not care about the strategy should be able to express without naming one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: RetrievalQuery
    strategy: RetrievalStrategy | None = None


@dataclass(frozen=True)
class FinalizedRetrieval:
    """What every strategy returns: one common outcome, plus what selected it.

    The delegating properties exist so a consumer can read the fields requirement asks for -
    strategy, configuration version, permission scope hash, trace, results - off one object
    without knowing which strategy produced it.
    """

    #: The strategy that was selected. For a fused run this is ``HYBRID``, while
    #: ``outcome.strategies_executed`` names the two that actually ran.
    strategy: RetrievalStrategy
    #: The common contract, identical in type across all three strategies.
    outcome: RetrievalOutcome
    #: Per-strategy rank contributions. Present only for ``HYBRID``.
    hybrid: HybridOutcome | None = None

    def __post_init__(self) -> None:
        if (self.strategy is RetrievalStrategy.HYBRID) != (self.hybrid is not None):
            raise ValueError(
                "the hybrid contribution detail is present exactly for a hybrid run, got "
                f"strategy={self.strategy.value} hybrid={'set' if self.hybrid else 'None'}"
            )

    # -- the common surface ----------------------------------------------

    @property
    def results(self) -> tuple[object, ...]:
        return self.outcome.results

    @property
    def trace(self) -> RetrievalTrace:
        """The strategy-specific trace: why the corpus shrank, for the strategy that ran."""
        return self.outcome.trace

    @property
    def permission_scope_hash(self) -> str:
        return self.outcome.permission_scope_hash

    @property
    def retrieval_config_version(self) -> str:
        return self.outcome.retrieval_config_version

    @property
    def scorer_version(self) -> str:
        return self.outcome.scorer_version

    @property
    def strategies_executed(self) -> tuple[RetrievalStrategy, ...]:
        return self.outcome.strategies_executed

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.outcome.evidence_ids

    @property
    def is_empty(self) -> bool:
        return self.outcome.is_empty


class RetrievalService:
    """Selects a retrieval strategy and returns one shape whichever ran.

    Holds the use case, the default strategy and - optionally - an embedder. The embedder is
    injected rather than constructed: this layer must not import
    ``app.infrastructure.embeddings``, and a service that could build its own provider would
    be a service that could reach the network from the application layer.
    """

    def __init__(
        self,
        *,
        retrieval: RetrieveEvidenceUseCase,
        default_strategy: RetrievalStrategy | str = RetrievalStrategy.LEXICAL,
        embedder: EmbeddingPort | None = None,
        hybrid_config: HybridConfig | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._default_strategy = parse_strategy(default_strategy)
        self._embedder = embedder
        self._hybrid_config = hybrid_config

    @property
    def default_strategy(self) -> RetrievalStrategy:
        return self._default_strategy

    @property
    def has_embedder(self) -> bool:
        return self._embedder is not None

    def available_strategies(self) -> tuple[RetrievalStrategy, ...]:
        """The strategies this service can actually serve, given what it holds.

        Reported rather than discovered by failure, so a caller can present the choice
        honestly instead of offering one that raises.
        """
        if self._embedder is not None:
            return SELECTABLE_STRATEGIES
        return tuple(s for s in SELECTABLE_STRATEGIES if s not in EMBEDDING_STRATEGIES)

    async def retrieve(self, request: RetrievalRequest) -> FinalizedRetrieval:
        """Answer one request with the selected strategy.

        Resolution order: the request's strategy if it named one, otherwise the configured
        default. The result is the same type whichever ran.
        """
        strategy = request.strategy or self._default_strategy
        query = request.query

        if strategy not in SELECTABLE_STRATEGIES:
            # Reachable when a member is added to the enum and not to the table above. A
            # dispatch table rather than an `if` chain is what makes that an error instead
            # of a silent fall-through to lexical.
            raise UnknownRetrievalStrategy(
                f"no handler for retrieval strategy {strategy.value!r}; "
                f"this service can dispatch {[s.value for s in SELECTABLE_STRATEGIES]}"
            )

        if strategy in EMBEDDING_STRATEGIES and self._embedder is None:
            raise EmbedderRequired(
                f"strategy {strategy.value!r} needs an embedding provider and none is "
                f"configured. Available without one: "
                f"{[s.value for s in self.available_strategies()]}. This is not a fallback "
                "to lexical: a caller who asked for semantic retrieval and received lexical "
                "results would have no way to know."
            )

        if strategy is RetrievalStrategy.LEXICAL:
            return FinalizedRetrieval(strategy=strategy, outcome=self._retrieval.execute(query))

        assert self._embedder is not None  # guarded above; narrows for the type checker

        if strategy is RetrievalStrategy.VECTOR:
            outcome = await self._retrieval.execute_vector(query, embedder=self._embedder)
            return FinalizedRetrieval(strategy=strategy, outcome=outcome)

        fused = await self._retrieval.execute_hybrid(
            query, embedder=self._embedder, config=self._hybrid_config
        )
        return FinalizedRetrieval(strategy=strategy, outcome=fused.outcome, hybrid=fused)


def parse_strategy(value: RetrievalStrategy | str) -> RetrievalStrategy:
    """Resolve a configured strategy name, failing explicitly on an unknown one.

    Separate from the request path because the two fail differently and both must fail: a
    request is validated by Pydantic before it exists, while ``CCE_RETRIEVAL_STRATEGY`` is a
    plain string that nothing checks until it is used. This is that check, and the message
    names what is available rather than only what was wrong.
    """
    if isinstance(value, RetrievalStrategy):
        return value
    try:
        return RetrievalStrategy(value)
    except ValueError as exc:
        raise UnknownRetrievalStrategy(
            f"unknown retrieval strategy {value!r}; known strategies are "
            f"{[s.value for s in SELECTABLE_STRATEGIES]}. Set CCE_RETRIEVAL_STRATEGY to one "
            "of them."
        ) from exc
