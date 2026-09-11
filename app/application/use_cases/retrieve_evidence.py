"""Build retrieval candidates from the stores, then answer a query against them.

This is the only module in the retrieval slice that touches a store, deliberately: the
retriever below it is a pure function of values, so a ranking cannot depend on store
iteration order and every filter rule is testable without a database. It is the same split
the coupling slice uses, and for the same reason.

The join it performs is the one the retriever cannot: a span carries offsets and a version
id, but *which document* holds that version - and therefore its tenant, source type,
authority and **permissions** - lives on
:class:`~app.domain.entities.source.SourceDocument`, where all four were put by registration
and never by document content (``docs/SECURITY.md``). Scope and assumption keys come from the
admitted bindings extracted from that exact span, so a scope filter is answered by what the
corpus committed to, not by what the text resembles.

**Authorization.** The indexing step deliberately builds candidates the requester may *not*
read, because they have to be counted before they can be excluded - a restricted span that
never entered the index would be indistinguishable from one that does not exist, and
``RetrievalTrace.excluded_by_permission`` would silently under-report. Those candidates go
two places only: into :class:`~app.retrieval.lexical.LexicalRetriever` and into
:class:`~app.retrieval.vectors.VectorRetriever`, each of which narrows the index to the
authorized slice before anything is filtered, scored or returned. The raw builder is
therefore private, and :meth:`RetrieveEvidenceUseCase.authorized_candidates` is the only way
a caller gets candidates out - already filtered, and asking for a scope in order to do so
(``docs/SECURITY.md`` 4, 11).

**Three strategies, three methods, no shared mode flag.** :meth:`execute` is lexical and
stays synchronous and provider-free; :meth:`execute_vector` is vector and is async because it
must embed; :meth:`execute_hybrid` runs both and fuses their *ranks*. Keeping them apart is
what makes the frozen baselines in ``docs/EVALUATION.md`` trustworthy: no work on fusion can
perturb either input, because fusion calls the same two public methods anyone else would and
neither retriever knows it exists.

Candidates are rebuilt per call, and under :meth:`execute_vector` so are their embeddings.
That is honest for 21 spans and badly wrong for a real corpus, where embedding the whole
tenant per query would dominate both latency and cost - so it is recorded as the largest
open issue in the slice rather than papered over. A stored vector index keyed on
(corpus version, embedder) is the fix, and it belongs with pgvector rather than here
(``ARCHITECTURE.md`` 10).

**On model calls.** The lexical path makes none and has no parameter through which one could
be supplied. The vector path makes exactly one kind - an embedding call, through
``EmbeddingPort``, passed in by the caller and never constructed here. No completion call is
reachable from this module: there is no gateway, no agent runtime and no provider SDK in its
imports, and ``tests/contract/test_layering.py`` enforces that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.domain.interfaces.llm import EmbeddingPort
from app.domain.interfaces.repositories import AssumptionStore, SourceStore
from app.domain.interfaces.retrieval import (
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalQuery,
)
from app.domain.value_objects.identifiers import AssumptionKey
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.retrieval.hybrid import HybridConfig, HybridOutcome, fuse
from app.retrieval.lexical import LexicalConfig, LexicalRetriever
from app.retrieval.vectors import (
    EmbeddedCandidate,
    VectorConfig,
    VectorQuery,
    VectorRetriever,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateSet:
    """The candidates a tenant's corpus yields, plus the spans that could not be joined.

    ``unresolvable`` is reported rather than swallowed. A span whose version or document is
    missing from the store is a corpus defect, and silently dropping it would make a recall
    number look like a retrieval property when it was a storage one.
    """

    candidates: tuple[EvidenceCandidate, ...] = ()
    unresolvable: tuple[str, ...] = ()


class RetrieveEvidenceUseCase:
    """Lexical retrieval over one tenant's stored evidence spans."""

    def __init__(
        self,
        *,
        sources: SourceStore,
        assumptions: AssumptionStore,
        config: LexicalConfig | None = None,
    ) -> None:
        self._sources = sources
        self._assumptions = assumptions
        self._config = config

    def execute(self, query: RetrievalQuery) -> RetrievalOutcome:
        """Join, index and retrieve.

        Both the tenant and the permission scope come from the query, never from ambient
        state, so a request cannot inherit an authorization it did not present.
        """
        built = self._index_candidates(query.tenant_id)
        if built.unresolvable:
            # A count and a tenant. Never an id or a title: this line is written whatever
            # the requester's scope is, so anything richer would log restricted metadata.
            logger.warning(
                "evidence spans excluded from retrieval: no joinable version or document",
                extra={"tenant_id": query.tenant_id, "count": len(built.unresolvable)},
            )
        retriever = LexicalRetriever(built.candidates, config=self._config)
        return retriever.retrieve(query)

    async def execute_vector(
        self,
        query: RetrievalQuery,
        *,
        embedder: EmbeddingPort,
    ) -> RetrievalOutcome:
        """Join, embed, index and retrieve by cosine similarity.

        A **separate method rather than a mode of** :meth:`execute`, deliberately. The
        lexical path stays synchronous and provider-free, so nothing about it changes when
        this exists and no lexical caller can accidentally acquire an embedding call. The
        frozen lexical baseline in ``docs/EVALUATION.md`` is only a baseline if it cannot
        be perturbed by work on the second strategy.

        **This method owns the async boundary that** ``RetrievalPort`` **does not model.**
        The port is synchronous because a lexical scan is computation; interpreting a query
        for the vector strategy needs an embedding call, which is I/O. So the embedding
        happens here and :class:`~app.retrieval.vectors.VectorRetriever` stays a pure
        function of values.

        Spans and the query are embedded in **one** call each, and the span texts are passed
        in the candidate order the join produced - so the vector attached to a span is the
        vector of that span's text, positionally, and a provider that reorders its output
        would be a defect the dimension check below cannot catch. That is why the batch is
        zipped strictly.
        """
        built = self._index_candidates(query.tenant_id)
        if built.unresolvable:
            logger.warning(
                "evidence spans excluded from retrieval: no joinable version or document",
                extra={"tenant_id": query.tenant_id, "count": len(built.unresolvable)},
            )

        embedded = await self._embed_candidates(built.candidates, embedder=embedder)
        config = VectorConfig(embedder=embedder.name, dimensions=embedder.dimensions)
        retriever = VectorRetriever(embedded, config=config)

        query_vectors = await embedder.embed([query.text])
        if len(query_vectors) != 1:
            raise ValueError(
                f"embedder {embedder.name!r} returned {len(query_vectors)} vectors for one text"
            )
        return retriever.retrieve(VectorQuery(query=query, vector=tuple(query_vectors[0])))

    async def execute_hybrid(
        self,
        query: RetrievalQuery,
        *,
        embedder: EmbeddingPort,
        config: HybridConfig | None = None,
    ) -> HybridOutcome:
        """Run both strategies over the same query and fuse their rankings.

        A **third method rather than a mode on either of the others**, for the reason the
        first two are separate: the lexical baseline and the vector baseline are only
        baselines if work on fusion cannot perturb them. Neither retriever is touched here,
        and both are called through the same public methods any other caller would use - so
        a fused run is exactly the two runs a reader could reproduce independently, plus
        arithmetic over their ranks.

        Every filter, every authorization decision and every provenance validator has
        already been applied twice - once per strategy - before :func:`fuse` sees anything.
        Fusion can only merge, reorder and truncate what the two strategies were each
        already allowed to return; it cannot introduce a span neither of them did.
        """
        lexical = self.execute(query)
        vector = await self.execute_vector(query, embedder=embedder)
        return fuse(lexical, vector, config=config)

    async def _embed_candidates(
        self,
        candidates: tuple[EvidenceCandidate, ...],
        *,
        embedder: EmbeddingPort,
    ) -> tuple[EmbeddedCandidate, ...]:
        """Embed every span's text, preserving the pairing positionally.

        The text embedded is ``candidate.text`` - the span itself and nothing else, the same
        text the lexical retriever scores. Section labels, titles and document metadata are
        excluded for the same reason as there: a match must be attributable to the words
        inside the citation, or a result is justified by something the reader cannot see in
        the quoted span.
        """
        if not candidates:
            return ()
        vectors = await embedder.embed([candidate.text for candidate in candidates])
        if len(vectors) != len(candidates):
            raise ValueError(
                f"embedder {embedder.name!r} returned {len(vectors)} vectors for "
                f"{len(candidates)} spans; the pairing is positional and cannot be recovered"
            )
        return tuple(
            EmbeddedCandidate(candidate=candidate, vector=tuple(vector))
            for candidate, vector in zip(candidates, vectors, strict=True)
        )

    def authorized_candidates(self, tenant_id: str, scope: PermissionScope) -> CandidateSet:
        """The spans of ``tenant_id`` that ``scope`` may read.

        The only candidate accessor on the public surface, and it takes a scope because
        there is no correct answer without one. A caller wanting "all of them" must say
        which grants it holds, and an empty scope is the least privileged answer rather than
        a wildcard.
        """
        built = self._index_candidates(tenant_id)
        return CandidateSet(
            candidates=tuple(c for c in built.candidates if c.is_readable_by(scope)),
            unresolvable=built.unresolvable,
        )

    # -- internals --------------------------------------------------------

    def _index_candidates(self, tenant_id: str) -> CandidateSet:
        """Every span for ``tenant_id``, joined to its provenance, **authorization aside**.

        Private, and the docstring says why: the result includes spans the requester may not
        read. Its only legitimate consumer is the retriever, which narrows to the authorized
        slice before anything leaves it.

        Ordered by ``evidence_id`` so the candidate list itself is reproducible, which is
        what lets the retriever's tie break be the only thing deciding equal-scoring order.
        """
        scopes, keys = self._bindings_by_evidence(tenant_id)

        candidates: list[EvidenceCandidate] = []
        unresolvable: list[str] = []

        for item in self._sources.evidence_for_tenant(tenant_id):
            version = self._sources.get_version(item.source_version_id)
            document = (
                self._sources.get_document(tenant_id, version.source_id)
                if version is not None
                else None
            )
            if version is None or document is None:
                unresolvable.append(item.evidence_id)
                continue

            candidates.append(
                EvidenceCandidate(
                    evidence=item,
                    document=document,
                    version=version,
                    scopes=tuple(scopes.get(item.evidence_id, ())),
                    assumption_keys=tuple(keys.get(item.evidence_id, ())),
                )
            )

        candidates.sort(key=lambda candidate: candidate.evidence_id)
        return CandidateSet(candidates=tuple(candidates), unresolvable=tuple(sorted(unresolvable)))

    def _bindings_by_evidence(
        self, tenant_id: str
    ) -> tuple[dict[str, tuple[ScopeSelector, ...]], dict[str, tuple[AssumptionKey, ...]]]:
        """What each span binds: its scopes and its registry keys, deduplicated and sorted.

        One span can bind more than one assumption, and the scope filter is satisfied by
        *any* of them - a span is relevant to a scope if anything it commits to reaches
        that scope.
        """
        scopes: dict[str, list[ScopeSelector]] = {}
        keys: dict[str, list[str]] = {}

        for record in self._assumptions.all_records(tenant_id):
            bucket = scopes.setdefault(record.evidence_id, [])
            if record.scope not in bucket:
                bucket.append(record.scope)
            key_bucket = keys.setdefault(record.evidence_id, [])
            if record.key not in key_bucket:
                key_bucket.append(record.key)

        return (
            {
                evidence_id: tuple(sorted(paths, key=lambda scope: scope.render()))
                for evidence_id, paths in scopes.items()
            },
            {evidence_id: tuple(sorted(names)) for evidence_id, names in keys.items()},
        )
