"""Vector retrieval over the stored evidence spans. Cosine similarity, no model, no network.

The second retrieval strategy, and deliberately the *same shape* as the first. Everything
structural in ``app/retrieval/lexical.py`` is repeated here rather than reinvented, because
each of those properties was a decision and none of them is about BM25:

**The retrievable unit is the** :class:`~app.domain.entities.source.EvidenceItem` - the span
the admission gate verified - not the chunk and not the document. What retrieval hands back
is exactly what a finding may cite, offsets included.

**This is a pure function of its inputs.** The retriever is constructed from candidates and
their vectors and answers queries; it holds no store, opens no connection and - the point
worth stating for a *vector* retriever - **makes no embedding call**. The store join and
the embedding both happen once, above it, in
``app/application/use_cases/retrieve_evidence.py``.

**Determinism is total, including the tie break.** Results are ordered by descending
similarity and then by ascending ``evidence_id``, and scores are rounded before comparison.
Two runs over the same corpus return the same list in the same order. This matters more here
than for BM25: cosine over floating-point vectors produces near-ties constantly, and without
a rounding step plus a total tie break the ordering would drift between platforms.

**Authorization is applied first, by narrowing the index.** Everything downstream iterates a
collection an unreadable span is not in, so no ordering mistake or later edit can leak one
(``docs/SECURITY.md`` 4). Unlike BM25 there are no corpus statistics to recompute
afterwards, because cosine is pairwise: a span's score depends on the query and on itself
and on nothing else in the corpus. So a requester's scope changes *which* spans come back
and never the score of a span that survives, and ``excluded_by_permission`` is
query-independent for a structural reason rather than an arranged one.

**Filters select; they do not re-weight.** Same fixed order as lexical - source type, then
as-of, then scope - and each candidate is charged to the first filter it fails, so the trace
buckets sum to the number considered.

Two things are genuinely different, and both are reported rather than smoothed over:

**There is no similarity floor.** A result must score above zero because
:class:`~app.domain.interfaces.retrieval.RetrievalResult` requires it, and that is the only
cutoff. A dense embedder puts almost every span at a positive cosine to almost every query,
so this strategy returns ``top_k`` results for nearly any question - including one the corpus
cannot answer. A floor would fix that and would be an untuned knob; choosing one needs a
measurement, and this slice is the measurement (``docs/EVALUATION.md``).

**It does not satisfy** :class:`~app.domain.interfaces.retrieval.RetrievalPort`, and that is
a contract observation rather than an oversight. The port is synchronous on purpose - "a
lexical scan is computation, not I/O" - but interpreting a *query* for this strategy requires
an embedding call, which is I/O. So the pure scorer below takes an already-embedded
:class:`VectorQuery`, and the ``RetrievalPort``-shaped surface belongs to the async use case
above it. The port is left alone; the mismatch is recorded in ``docs/ARCHITECTURE.md`` for
the QueryPlan slice to resolve, because widening it now would be guessing at the shape of
fusion.

What this module does not do: hybrid fusion with lexical, reranking, query expansion, and
any similarity function other than cosine. Each is a later, separately measurable change.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.domain.interfaces.retrieval import (
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalResult,
    RetrievalTrace,
)
from app.domain.services import scope_lattice
from app.domain.value_objects.enums import RetrievalStrategy
from app.domain.value_objects.permissions import PermissionScope

#: The similarity function itself. Bumped when the arithmetic changes; the embedder and its
#: dimensionality travel in the fingerprint instead, so swapping providers is visible
#: without pretending the scorer became a different one.
VECTOR_SCORER: Final = "vector-cosine-1"

#: Scores are rounded to this many decimals before ranking, for the same reason and to the
#: same precision as ``lexical.SCORE_PRECISION``. Defined separately rather than imported so
#: that the two strategies can be re-tuned independently - coupling them would mean a
#: lexical change silently altering vector rankings.
SCORE_PRECISION: Final = 6


class VectorConfig(BaseModel):
    """What identifies this index, beyond the vectors themselves.

    There is nothing to tune here, and that is deliberate: cosine has no parameters. The
    class exists to carry the **embedder's identity** into ``scorer_version``, because a
    recall figure measured under one embedder says nothing about another and the two must
    not be comparable by accident.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The ``EmbeddingPort.name`` of the embedder that produced every vector in the index.
    embedder: str = Field(min_length=1)
    #: Its ``dimensions``. Part of the identity because the same embedder at a different
    #: width is a different index.
    dimensions: int = Field(gt=0)

    def fingerprint(self) -> str:
        return f"{VECTOR_SCORER}:{self.embedder}:d={self.dimensions}"


@dataclass(frozen=True)
class EmbeddedCandidate:
    """One retrievable span paired with its vector.

    A pair rather than a field on :class:`EvidenceCandidate`, deliberately: the candidate is
    a domain contract shared with the lexical path, and a vector is a property of one
    strategy's index rather than of the evidence. Putting it on the candidate would make
    every lexical result carry an embedding it never uses.
    """

    candidate: EvidenceCandidate
    vector: tuple[float, ...]

    @property
    def evidence_id(self) -> str:
        return self.candidate.evidence_id

    @property
    def tenant_id(self) -> str:
        return self.candidate.tenant_id


@dataclass(frozen=True)
class VectorQuery:
    """A retrieval query together with its embedding.

    The embedding is an input rather than something this layer computes, which is what keeps
    the retriever pure and testable without a provider. See the module docstring on why this
    means the ``RetrievalPort`` surface lives one layer up.
    """

    query: RetrievalQuery
    vector: tuple[float, ...]


@dataclass(frozen=True)
class _TenantIndex:
    """The embedded spans of one tenant. Never shared across tenants.

    Tenant is a partition here exactly as it is for lexical retrieval. There are no pooled
    statistics for it to protect - cosine has none - but the isolation argument is the same
    and stronger for being structural: another tenant's spans are not in the list, so no
    filter-order mistake can return one (``docs/SECURITY.md`` 8).
    """

    postings: tuple[EmbeddedCandidate, ...]

    @property
    def size(self) -> int:
        return len(self.postings)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, or ``0.0`` when either side carries no signal.

    A zero vector has no direction, so its similarity to anything is undefined rather than
    high or low. Returning ``0.0`` reports "no match", which is the answer that keeps an
    empty text out of the results instead of placing it somewhere arbitrary.
    """
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


class VectorRetriever:
    """Cosine similarity over embedded evidence spans.

    Constructed from candidates rather than from a store, and rebuilt rather than mutated -
    the index is a projection of the evidence store in the same sense the coupling index is
    a projection of the assumption store, so it cannot drift from what the store holds.
    """

    def __init__(
        self,
        candidates: Iterable[EmbeddedCandidate],
        *,
        config: VectorConfig,
    ) -> None:
        self._config = config

        # The same span offered twice is one span. Deduplicating on `evidence_id` keeps the
        # trace's arithmetic honest and stops one span occupying two result ranks.
        unique: dict[str, EmbeddedCandidate] = {}
        for embedded in candidates:
            if len(embedded.vector) != config.dimensions:
                raise ValueError(
                    f"span {embedded.evidence_id} has a {len(embedded.vector)}-dimensional "
                    f"vector but the index is {config.dimensions}-dimensional. Mixing "
                    "embedders in one index makes every cosine in it meaningless."
                )
            unique.setdefault(embedded.evidence_id, embedded)

        by_tenant: dict[str, list[EmbeddedCandidate]] = {}
        for embedded in unique.values():
            by_tenant.setdefault(embedded.tenant_id, []).append(embedded)

        self._total = len(unique)
        self._indexes = {
            tenant_id: _TenantIndex(postings=tuple(postings))
            for tenant_id, postings in by_tenant.items()
        }

    # -- surface ---------------------------------------------------------

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.VECTOR

    @property
    def scorer_version(self) -> str:
        return self._config.fingerprint()

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    @property
    def size(self) -> int:
        """Indexed spans across every tenant. Useful in a report, never in a filter."""
        return self._total

    def retrieve(self, vector_query: VectorQuery) -> RetrievalOutcome:
        """Filter, score, rank and truncate - in that order, and the same order as lexical.

        Tenant selection and authorization happen first and structurally, by narrowing the
        index. Everything below them is already inside one tenant's authorized corpus.
        """
        query = vector_query.query
        if len(vector_query.vector) != self._config.dimensions:
            raise ValueError(
                f"query vector is {len(vector_query.vector)}-dimensional but the index is "
                f"{self._config.dimensions}-dimensional; it was produced by a different "
                "embedder and no cosine between them would mean anything"
            )

        tenant_index = self._indexes.get(query.tenant_id)
        counts = dict.fromkeys(
            (
                "excluded_by_tenant",
                "excluded_by_permission",
                "excluded_by_source_type",
                "excluded_by_as_of",
                "excluded_by_scope",
                "excluded_unscoped",
                "excluded_zero_score",
            ),
            0,
        )
        counts["excluded_by_tenant"] = self._total - (tenant_index.size if tenant_index else 0)

        index = _authorized_slice(tenant_index, query.permission_scope)
        if tenant_index is not None and index is not None:
            counts["excluded_by_permission"] = tenant_index.size - index.size

        scored: list[tuple[float, str, EvidenceCandidate]] = []
        if index is not None:
            for posting in index.postings:
                bucket = _first_failing_filter(posting.candidate, query)
                if bucket is not None:
                    counts[bucket] += 1
                    continue

                score = round(cosine(vector_query.vector, posting.vector), SCORE_PRECISION)
                # A non-positive cosine is not a weak result; it is not a result. The
                # contract refuses a zero score, and orthogonal or opposed vectors say the
                # span does not answer the query.
                if score <= 0.0:
                    counts["excluded_zero_score"] += 1
                    continue
                scored.append((score, posting.candidate.evidence_id, posting.candidate))

        # Descending score, then ascending evidence_id. The second key is what makes the
        # ordering total, and cosine near-ties make it load-bearing rather than defensive.
        scored.sort(key=lambda row: (-row[0], row[1]))
        selected = scored[: query.top_k]

        return RetrievalOutcome(
            query=query,
            results=tuple(
                RetrievalResult(
                    candidate=candidate,
                    strategy=self.strategy,
                    score=score,
                    rank=rank,
                )
                for rank, (score, _evidence_id, candidate) in enumerate(selected, start=1)
            ),
            strategies_executed=(self.strategy,),
            permission_scope_hash=query.permission_scope_hash,
            trace=RetrievalTrace(
                considered=self._total,
                matched=len(scored),
                returned=len(selected),
                **counts,
            ),
            scorer_version=self.scorer_version,
        )


def _authorized_slice(index: _TenantIndex | None, scope: PermissionScope) -> _TenantIndex | None:
    """Reduce a tenant index to the spans ``scope`` may read.

    Returning a *narrower index* rather than adding a per-candidate check is the security
    argument of this module, and it is the same one lexical retrieval makes: everything
    downstream - the content filters, the scoring, the ranking, the trace's remaining
    buckets - iterates a collection an unauthorized span is not in. Leaking one would
    require putting it back (``docs/SECURITY.md`` 4).
    """
    if index is None:
        return None
    readable = [posting for posting in index.postings if posting.candidate.is_readable_by(scope)]
    if len(readable) == index.size:
        return index
    return _TenantIndex(postings=tuple(readable))


def _first_failing_filter(candidate: EvidenceCandidate, query: RetrievalQuery) -> str | None:
    """The name of the first filter this candidate fails, or ``None`` if it passes.

    The order is fixed and identical to the lexical retriever's, so a filtered vector run
    and a filtered lexical run exclude the same spans for the same reasons and differ only
    in how they score what remains. Tenant and authorization are deliberately **not** among
    these checks: the index is partitioned per tenant and then narrowed to the authorized
    slice, so neither another tenant's spans nor unreadable ones are iterated here at all.
    """
    if query.source_types and candidate.source_type not in query.source_types:
        return "excluded_by_source_type"
    if query.as_of is not None and not candidate.is_in_force(query.as_of):
        return "excluded_by_as_of"
    if query.scope is not None:
        if not candidate.scopes:
            return "excluded_unscoped"
        if not any(scope_lattice.intersects(scope, query.scope) for scope in candidate.scopes):
            return "excluded_by_scope"
    return None
