"""Lexical retrieval over the stored evidence spans. Okapi BM25, no model, no network.

The retrievable unit is the :class:`~app.domain.entities.source.EvidenceItem` - the span the
admission gate already verified - and not the chunk and not the document. That choice is
what makes a result citable without any further work: the thing retrieval hands back is
exactly the thing a finding may cite, offsets included.

**This is a pure function of its inputs.** The retriever is constructed from candidate
values and answers queries; it holds no store, opens no connection and makes no model call.
The store join happens once, in ``app/application/use_cases/retrieve_evidence.py``, for the
same reason the coupling engine works that way: every rule below is testable by building
values, and store iteration order cannot change a ranking.

**Determinism is total, including the tie break.** Results are ordered by descending score
and then by ascending ``evidence_id``, and scores are rounded before comparison, so two runs
over the same corpus return the same list in the same order rather than the same set.

**Filters select; they do not re-weight.** Corpus statistics (document frequency and mean
length) are computed over the tenant's authorized slice, before any *content* filter runs. A
source-type, scope or as-of filter therefore changes *which* spans come back and never the
score of a span that survives, which is what lets a filtered query be compared with an
unfiltered one.

**Authorization is not one of those filters, and that is the point.** It is applied first,
and the corpus statistics are computed after it, so a requester's scope defines *the corpus
they have* rather than selecting from a corpus everyone shares. Two consequences, both
wanted: a score cannot depend on a document the requester may not read, and two scopes
legitimately produce different rankings - which is exactly what ``permission_scope_hash``
exists to make distinguishable and reproducible (``ARCHITECTURE.md`` 14.1). It also closes
an inference channel: filtering last would make ``excluded_by_permission`` an oracle
answering "how many restricted documents mention my search terms?".

What this module does not do: embeddings, vector similarity, reranking, fusion with another
strategy, and permission filtering. Each is a later slice.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
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
from app.retrieval.tokens import TOKENIZER_VERSION, term_frequencies, tokenize

#: The scoring function itself. Bumped when the formula changes; the tuning parameters
#: travel in the fingerprint instead, so retuning ``k1`` is visible without pretending the
#: scorer became a different one.
LEXICAL_SCORER: Final = "lexical-bm25-1"

#: Scores are rounded to this many decimals before ranking. Two candidates whose scores
#: differ only in floating-point noise must tie and be broken by ``evidence_id``, otherwise
#: the ordering could differ between platforms and a ranking would not be reproducible.
SCORE_PRECISION: Final = 6


class LexicalConfig(BaseModel):
    """BM25 tuning parameters.

    The values are the standard defaults, and they are **untuned against this corpus**.
    They stayed untuned when the retrieval benchmark landed, deliberately: fitting a
    parameter in the same change that authored the labels produces a number that measures
    the fitting. The fingerprint travels with every outcome, and the benchmark suite pins
    it, so a later tuning pass is attributable and cannot reprint stale recall figures
    (``data/benchmarks/retrieval_queries.yaml``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Term-frequency saturation. Higher means repeated terms keep adding weight for longer.
    k1: float = Field(default=1.2, gt=0.0)
    #: Length normalisation, ``0`` = off, ``1`` = full. Spans vary in length, and without
    #: this a long span outranks a precise short one purely by having more words.
    b: float = Field(default=0.75, ge=0.0, le=1.0)

    def fingerprint(self) -> str:
        return f"{LEXICAL_SCORER}:{TOKENIZER_VERSION}:k1={self.k1}/b={self.b}"


@dataclass(frozen=True)
class _Posting:
    """One indexed span: the candidate, plus what scoring needs from its text."""

    candidate: EvidenceCandidate
    term_freq: Mapping[str, int]
    length: int


@dataclass(frozen=True)
class _TenantIndex:
    """Corpus statistics for one tenant. Never shared across tenants.

    Document frequency is a property of a corpus, and one tenant's corpus is not another's.
    Pooling them would both leak the shape of a neighbouring tenant's documents into a score
    and make a ranking depend on who else is on the system (``docs/SECURITY.md`` 8).
    """

    postings: tuple[_Posting, ...]
    doc_freq: Mapping[str, int]
    mean_length: float

    @property
    def size(self) -> int:
        return len(self.postings)


def _index_tenant(postings: Sequence[_Posting]) -> _TenantIndex:
    doc_freq: dict[str, int] = {}
    total_length = 0
    for posting in postings:
        total_length += posting.length
        for term in posting.term_freq:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    mean = (total_length / len(postings)) if postings else 0.0
    return _TenantIndex(postings=tuple(postings), doc_freq=doc_freq, mean_length=mean)


class LexicalRetriever:
    """BM25 over evidence spans, satisfying :class:`RetrievalPort`.

    Constructed from candidates rather than from a store, and rebuilt rather than mutated -
    the index is a projection of the evidence store in the same sense the coupling index is
    a projection of the assumption store, so it can never drift from what the store holds.
    """

    def __init__(
        self,
        candidates: Iterable[EvidenceCandidate],
        *,
        config: LexicalConfig | None = None,
    ) -> None:
        self._config = config or LexicalConfig()

        # The same span offered twice is one span. Deduplicating on `evidence_id` keeps
        # document frequency honest: counting a span twice would inflate its own term
        # statistics and quietly demote every term it contains.
        unique: dict[str, EvidenceCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.evidence_id, candidate)

        by_tenant: dict[str, list[_Posting]] = {}
        for candidate in unique.values():
            tokens = tokenize(candidate.text)
            posting = _Posting(
                candidate=candidate,
                term_freq=term_frequencies(tokens),
                length=len(tokens),
            )
            by_tenant.setdefault(candidate.tenant_id, []).append(posting)

        self._total = len(unique)
        self._indexes = {
            tenant_id: _index_tenant(postings) for tenant_id, postings in by_tenant.items()
        }

    # -- port ------------------------------------------------------------

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.LEXICAL

    @property
    def scorer_version(self) -> str:
        return self._config.fingerprint()

    @property
    def size(self) -> int:
        """Indexed spans across every tenant. Useful in a report, never in a filter."""
        return self._total

    def retrieve(self, query: RetrievalQuery) -> RetrievalOutcome:
        """Filter, score, rank and truncate - in that order.

        Filtering precedes scoring because ``CLAUDE.md`` requires retrieval before
        reasoning and, at this level, cheap exclusion before arithmetic. It also fixes the
        exclusion accounting: a candidate is charged to the *first* filter it fails, so the
        trace buckets sum to the number of candidates considered.

        Tenant selection and authorization happen first and structurally, by narrowing
        the index. Everything below them is already inside one tenant's authorized corpus.
        """
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
        # Every span belonging to another tenant is excluded before anything else happens.
        counts["excluded_by_tenant"] = self._total - (tenant_index.size if tenant_index else 0)

        # Then authorization, before any content filter and before any arithmetic. The
        # authorized slice is re-indexed so document frequency and mean length describe the
        # corpus this requester actually has (see the module docstring).
        index = _authorized_slice(tenant_index, query.permission_scope)
        if tenant_index is not None and index is not None:
            counts["excluded_by_permission"] = tenant_index.size - index.size

        query_terms = term_frequencies(tokenize(query.text))
        scored: list[tuple[float, str, EvidenceCandidate]] = []

        if index is not None:
            for posting in index.postings:
                bucket = _first_failing_filter(posting.candidate, query)
                if bucket is not None:
                    counts[bucket] += 1
                    continue

                score = self._score(posting, query_terms, index)
                if score <= 0.0:
                    counts["excluded_zero_score"] += 1
                    continue
                scored.append((score, posting.candidate.evidence_id, posting.candidate))

        # Descending score, then ascending evidence_id. The second key is what makes the
        # ordering total: two spans quoting the same words must not swap places between
        # runs just because a dictionary was built in a different order.
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

    # -- scoring ---------------------------------------------------------

    def _score(
        self,
        posting: _Posting,
        query_terms: Mapping[str, int],
        index: _TenantIndex,
    ) -> float:
        """Okapi BM25, summed over the query terms the span actually contains.

        The IDF form is the one that stays non-negative for a term present in more than
        half the corpus. The textbook form goes negative there, which on a corpus this
        small would let a common term *subtract* from a span's score and rank a matching
        span below a non-matching one.
        """
        k1 = self._config.k1
        b = self._config.b
        mean = index.mean_length or 1.0

        total = 0.0
        for term in query_terms:
            freq = posting.term_freq.get(term)
            if not freq:
                continue
            doc_freq = index.doc_freq.get(term, 0)
            idf = math.log(1.0 + (index.size - doc_freq + 0.5) / (doc_freq + 0.5))
            saturation = freq * (k1 + 1.0) / (freq + k1 * (1.0 - b + b * posting.length / mean))
            total += idf * saturation

        return round(total, SCORE_PRECISION)


def _authorized_slice(index: _TenantIndex | None, scope: PermissionScope) -> _TenantIndex | None:
    """Reduce a tenant index to the spans ``scope`` may read, and re-derive its statistics.

    Returning a *narrower index* rather than adding a per-candidate check is the security
    argument of this module: everything downstream - the content filters, the scoring, the
    ranking, the trace's remaining buckets - iterates a collection an unauthorized span is
    not in. No ordering mistake, later edit or early ``return`` can leak one, because
    leaking one would require putting it back (``docs/SECURITY.md`` 4).
    """
    if index is None:
        return None
    readable = [posting for posting in index.postings if posting.candidate.is_readable_by(scope)]
    if len(readable) == index.size:
        return index
    return _index_tenant(readable)


def _first_failing_filter(candidate: EvidenceCandidate, query: RetrievalQuery) -> str | None:
    """The name of the first filter this candidate fails, or ``None`` if it passes.

    The order is fixed and is part of the retrieval configuration, and each candidate is
    charged to exactly one bucket. Tenant and authorization are deliberately **not** among
    these checks: the index is partitioned per tenant and then narrowed to the authorized
    slice, so neither another tenant's spans nor unreadable ones are iterated here at all. A
    filter that could be reordered or short-circuited is a weaker guarantee than a span that
    is not in the loop (``docs/SECURITY.md`` 4, 8).
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
