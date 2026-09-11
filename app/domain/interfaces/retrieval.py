"""Retrieval contracts: what a query is, what a result carries, and the port behind both.

``ARCHITECTURE.md`` 6 makes retrieval a separate service layer with several strategies
behind one interface. This module is that interface, and it is deliberately authored before
the first retriever so a later strategy is a new implementation rather than a new contract.

Three decisions are structural here rather than left to each retriever:

**A result carries its provenance by reference, never by copy.** A
:class:`EvidenceCandidate` holds the :class:`~app.domain.entities.source.EvidenceItem`, its
:class:`~app.domain.entities.source.SourceVersion` and its
:class:`~app.domain.entities.source.SourceDocument` as they are stored. Nothing is
flattened into a retrieval-local shape, so ``evidence_id``, the offsets, the version and the
owning document cannot drift from the store, and a citation taken from a result is the same
citation the admission gate verified.

**Tenant is a field on the candidate, not a filter applied afterwards.** The join that
builds a candidate reads the document, so a candidate that crossed a tenant boundary is
already a bug at construction time; the retriever re-checks it anyway (``docs/SECURITY.md``
8, 9).

**Authorization is a required input, not an option.** ``RetrievalQuery.permission_scope``
has no default: a scope defaulting to "no filtering" would make *forgetting* the parameter
the unrestricted case (``docs/SECURITY.md`` 4, 11). The scope's hash is stamped onto the
outcome and a validator refuses an outcome whose hash disagrees with the query it answers,
so a result set cannot be relabelled as belonging to a scope that did not produce it
(``ARCHITECTURE.md`` 14.1).

**Every excluded candidate is counted.** :class:`RetrievalTrace` records why the corpus
shrank, which is what makes "retrieval returned nothing" a reportable observation about the
corpus rather than a silent empty list. That is the raw material for negative provenance on
an ``EvidenceGap``, and it is why filters are applied in a fixed order with the first
failure winning.

What is deliberately **not** here: embeddings, vector similarity, reranking and fusion
weights. Each is a later slice, and a contract that anticipated them would be guessing at
their shape.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Final, Protocol, Self, runtime_checkable

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel
from app.domain.entities.source import EvidenceItem, SourceDocument, SourceVersion
from app.domain.value_objects.enums import AuthorityClass, RetrievalStrategy, SourceType
from app.domain.value_objects.identifiers import (
    AssumptionKey,
    ChunkId,
    EvidenceId,
    SourceId,
    SourceVersionId,
    TenantId,
    VersionLabel,
)
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.provenance import Sha256
from app.domain.value_objects.scope import ScopeSelector

#: The retrieval configuration, taken as a whole: which filters exist and how they compose.
#: Pinned onto every outcome and recorded by ``RunProvenance.retrieval_config_version``,
#: because a filter change can alter what an analysis saw with no document changing.
RETRIEVAL_CONFIG_VERSION: Final = "retrieval-1"

#: Bounds on ``top_k``. An unbounded ``top_k`` is a cost-control hole: retrieval feeds
#: reasoning, and reasoning is priced per token (``CLAUDE.md``, cost discipline).
MIN_TOP_K: Final = 1
MAX_TOP_K: Final = 100

QueryText = Annotated[str, StringConstraints(min_length=1, max_length=2048)]


class EvidenceCandidate(DomainModel):
    """One retrievable span, joined to everything needed to filter and cite it.

    The join is the application layer's job, because it is the only layer that may touch a
    store. Doing it once and passing values down keeps every retriever a pure function of
    its inputs - the same property that makes the coupling engine testable without a
    database.

    ``scopes`` and ``assumption_keys`` come from the admitted bindings extracted from this
    exact span. They are empty for a span that binds nothing, and that emptiness is
    meaningful: a scope filter cannot judge a span with no scope, so it excludes it and
    says so (see :class:`RetrievalTrace`).
    """

    evidence: EvidenceItem
    document: SourceDocument
    version: SourceVersion
    scopes: tuple[ScopeSelector, ...] = ()
    assumption_keys: tuple[AssumptionKey, ...] = ()

    @model_validator(mode="after")
    def _join_is_consistent(self) -> Self:
        """A mis-joined candidate is rejected at the boundary rather than retrieved.

        Retrieval is the path that hands evidence to an agent. A candidate whose evidence
        belongs to a different version than the document it claims would attribute a real
        span to the wrong artifact and the wrong authority, which is a fabricated citation
        assembled out of genuine parts.
        """
        if self.evidence.source_version_id != self.version.version_id:
            raise ValueError(
                f"candidate {self.evidence.evidence_id} cites version "
                f"{self.evidence.source_version_id} but was joined to "
                f"{self.version.version_id}"
            )
        if self.version.source_id != self.document.source_id:
            raise ValueError(
                f"candidate {self.evidence.evidence_id}: version {self.version.version_id} "
                f"belongs to {self.version.source_id}, not to {self.document.source_id}"
            )
        return self

    # -- delegation, so a filter reads like the specification -------------

    @property
    def evidence_id(self) -> EvidenceId:
        return self.evidence.evidence_id

    @property
    def chunk_id(self) -> ChunkId:
        return self.evidence.chunk_id

    @property
    def source_version_id(self) -> SourceVersionId:
        return self.version.version_id

    @property
    def source_id(self) -> SourceId:
        return self.document.source_id

    @property
    def tenant_id(self) -> TenantId:
        return self.document.tenant_id

    @property
    def source_type(self) -> SourceType:
        return self.document.source_type

    @property
    def authority_class(self) -> AuthorityClass:
        return self.document.authority_class

    @property
    def text(self) -> str:
        """The text a lexical retriever scores: the span itself, nothing else.

        Section labels and titles are deliberately excluded. A match must be attributable
        to the words inside the citation, otherwise a result is justified by metadata the
        reader cannot see in the quoted span.
        """
        return self.evidence.text_span

    def is_in_force(self, as_of: dt.datetime) -> bool:
        """Whether the holder version governs at ``as_of``.

        Version-level, deliberately. Binding-level temporal validity decides *which value
        is authoritative*, and that is ``resolve_authority``'s question, not retrieval's.
        """
        return self.version.is_in_force(as_of)

    def is_readable_by(self, scope: PermissionScope) -> bool:
        """Whether ``scope`` may read this span.

        Read from the holder **document**, where the labels were put by registration - the
        same reason ``authority_class`` lives there. A span cannot carry its own permissions
        and document content cannot alter them, so a planted file can neither widen its own
        audience nor claim to be readable (``docs/SECURITY.md`` 3).
        """
        return scope.permits(self.document.permissions)


class RetrievalQuery(DomainModel):
    """A single-strategy retrieval request.

    ``ARCHITECTURE.md`` 6 calls the multi-strategy form a QueryPlan. This is deliberately
    smaller: one strategy, one set of filters, one ``top_k``. A plan composes queries once
    there is more than one strategy to compose.
    """

    tenant_id: TenantId
    text: QueryText
    #: **Required.** The grants the requester holds. There is no default and no ``None``
    #: case: an omitted scope must be a validation error rather than an open door.
    permission_scope: PermissionScope
    #: Empty means every source type. Restricting to nothing is expressed by omitting the
    #: filter, never by an empty allow-list, so a caller cannot silence retrieval by
    #: accident.
    source_types: tuple[SourceType, ...] = ()
    #: A span matches when one of its bound scopes intersects this one. Intersection, not
    #: equality: a ``global`` declaration is relevant to a ``region:EU`` question, which is
    #: the same reading invalidation uses (checkpoint 5.5).
    scope: ScopeSelector | None = None
    #: Evaluated against the holder version's validity interval.
    as_of: dt.datetime | None = None
    top_k: int = Field(default=10, ge=MIN_TOP_K, le=MAX_TOP_K)

    @property
    def filters_source_type(self) -> bool:
        return bool(self.source_types)

    @property
    def permission_scope_hash(self) -> Sha256:
        """The effective scope this query runs under, as a digest."""
        return self.permission_scope.scope_hash


class RetrievalResult(DomainModel):
    """One retrieved span, with its score and its rank.

    ``score`` is ``> 0`` by construction: a candidate that shares no term with the query is
    not a weak result, it is not a result. Keeping that in the type stops a zero-scored span
    reaching an agent as though retrieval had endorsed it.
    """

    candidate: EvidenceCandidate
    strategy: RetrievalStrategy
    score: float = Field(gt=0.0)
    rank: int = Field(ge=1)

    @property
    def evidence_id(self) -> EvidenceId:
        return self.candidate.evidence_id

    @property
    def source_id(self) -> SourceId:
        return self.candidate.source_id

    @property
    def source_version_id(self) -> SourceVersionId:
        return self.candidate.source_version_id

    @property
    def source_type(self) -> SourceType:
        return self.candidate.source_type


class RetrievalTrace(DomainModel):
    """Why the corpus shrank, bucket by bucket.

    Each candidate is counted in exactly **one** bucket - the first filter it failed - so
    the counts sum to ``considered``. That is asserted by a test, and it is what makes the
    trace usable as negative provenance: "nothing matched the words" and "everything was
    filtered out before scoring" are different observations about the corpus, and a single
    empty result list cannot tell them apart.
    """

    considered: int = Field(ge=0)
    excluded_by_tenant: int = Field(default=0, ge=0)
    #: Candidates the requester is not authorized to read. A **count and nothing else**:
    #: no id, no title, no source type, no label, no span. The count is deliberately
    #: query-independent - authorization is applied before the content filters and before
    #: scoring, so this number reports how much of the tenant's corpus is closed to the
    #: requester and never whether any of it matched their words. Applying it later would
    #: turn the count into an oracle answering "how many restricted documents mention X?".
    excluded_by_permission: int = Field(default=0, ge=0)
    excluded_by_source_type: int = Field(default=0, ge=0)
    excluded_by_as_of: int = Field(default=0, ge=0)
    excluded_by_scope: int = Field(default=0, ge=0)
    #: Filtered by scope, but for want of a scope rather than a mismatching one. Counted
    #: apart because it is a *coverage* signal: these spans bind nothing, so no scope filter
    #: can ever reach them, and a large number here means the registry needs expanding
    #: rather than the query rewriting.
    excluded_unscoped: int = Field(default=0, ge=0)
    excluded_zero_score: int = Field(default=0, ge=0)
    #: Candidates that survived every filter and scored above zero, before ``top_k``.
    matched: int = Field(ge=0)
    returned: int = Field(ge=0)

    @property
    def excluded(self) -> int:
        return (
            self.excluded_by_tenant
            + self.excluded_by_permission
            + self.excluded_by_source_type
            + self.excluded_by_as_of
            + self.excluded_by_scope
            + self.excluded_unscoped
            + self.excluded_zero_score
        )

    @property
    def truncated(self) -> bool:
        """True when ``top_k`` cut results off. A truncated result set is not exhaustive."""
        return self.returned < self.matched


class RetrievalOutcome(DomainModel):
    """Results plus everything needed to explain why two runs of one query differ."""

    query: RetrievalQuery
    results: tuple[RetrievalResult, ...] = ()
    #: Named, and only ever strategies that actually ran. An ``EvidenceGap`` cites this as
    #: the search it performed, so a strategy listed here must have been executed.
    strategies_executed: tuple[RetrievalStrategy, ...] = ()
    trace: RetrievalTrace
    #: The effective permission scope these results were computed under. An analysis is
    #: keyed on it, so a requester with a different scope forces recomputation rather than
    #: a filtered replay of somebody else's findings (``ARCHITECTURE.md`` 14.1).
    permission_scope_hash: Sha256
    retrieval_config_version: VersionLabel = RETRIEVAL_CONFIG_VERSION
    #: The scorer that produced ``score``, versioned separately from the configuration so
    #: retuning a scorer cannot look like a change to which filters exist.
    scorer_version: VersionLabel

    @model_validator(mode="after")
    def _ranks_are_dense_and_ordered(self) -> Self:
        """Ranks are 1..n in order, and the trace agrees with the list it describes."""
        expected = tuple(range(1, len(self.results) + 1))
        actual = tuple(result.rank for result in self.results)
        if actual != expected:
            raise ValueError(f"ranks must be dense and ascending from 1, got {list(actual)}")
        if len(self.results) != self.trace.returned:
            raise ValueError(
                f"trace reports {self.trace.returned} returned but {len(self.results)} "
                "results are present"
            )
        return self

    @model_validator(mode="after")
    def _scope_hash_matches_the_query(self) -> Self:
        """The stamped scope must be the scope that was asked for.

        Without this the hash is a label a caller could set to anything, and results
        computed under a wide scope could be served to a narrow one by relabelling them -
        precisely the cross-ACL replay ``ARCHITECTURE.md`` 14.1 forbids.
        """
        if self.permission_scope_hash != self.query.permission_scope_hash:
            raise ValueError(
                "permission_scope_hash does not match the scope of the query it answers"
            )
        return self

    @property
    def evidence_ids(self) -> tuple[EvidenceId, ...]:
        return tuple(result.evidence_id for result in self.results)

    @property
    def is_empty(self) -> bool:
        return not self.results


@runtime_checkable
class RetrievalPort(Protocol):
    """One retrieval strategy.

    Synchronous on purpose: a lexical scan over stored spans is computation, not I/O, and a
    strategy that does need I/O belongs behind the store ports it calls. Deterministic on
    purpose too - the same candidates and the same query must produce the same ranking,
    including the tie break, or an analysis is not reproducible.
    """

    @property
    def strategy(self) -> RetrievalStrategy: ...

    @property
    def scorer_version(self) -> str: ...

    def retrieve(self, query: RetrievalQuery) -> RetrievalOutcome: ...
