"""The labelled retrieval suite, and the runners that score each retriever against it.

Everything the retrieval slice first claimed was a *wiring* claim: the filters compose, the
ranking is total, an unauthorized span does not come back. Not one of those is a statement
about recall, and without one, adding a second retrieval strategy would have meant tuning a
fusion weight against nothing. This module is the number, and it now scores **two**
strategies against one set of labels.

**The labels are strategy-independent, and that is what makes the comparison legitimate.**
An expectation is a relevance judgment about the *question* - not a prediction about any
retriever - so the same 62 expectations score lexical and vector alike, with the same
denominators and the same reachability accounting. Nothing in the suite was changed for the
vector run.

**The two are scored separately and never merged.** :func:`run_suite` is lexical,
:func:`run_vector_suite` is vector, and there is deliberately no third function that
combines them: hybrid fusion needs a weighting nobody has measured, and inventing one here
would destroy both baselines at once. Each strategy is compared against its own pinned
scorer, so one cannot be reprinted under the other's heading.

**No stemmer, no synonyms, no query expansion, no reranking, no fusion.** The suite measures
what is built. Two of its categories - ``known_lexical_miss`` and
``assumption_key_lookup`` - exist precisely to price the absence of those things, so a
change that adds one has a before figure to be judged against.

How a label is written, and why it is not an ``evidence_id``
------------------------------------------------------------

An expectation names a **source path and a quote**, and the runner resolves that pair to
the ``evidence_id`` of the admitted span containing it. Writing the 32-hex id directly was
the first draft and is worse in the way that matters: an id is opaque, so a reviewer cannot
tell whether the label is right, and - decisively - a stale id is indistinguishable from a
span that was never admitted. Resolving through the text tells those two apart, which is
the whole of the reachability accounting below. It is the same discipline the extraction
fixtures use (``app/evaluation/replay.py``), for the same reason.

A quote must occur **exactly once** in the document. A quote occurring twice does not
identify a span, and silently taking the first match would let a label drift onto a
different sentence after an edit.

Three ways an expectation can fail to arrive
--------------------------------------------

Kept apart, because they have different owners and only one of them is retrieval's:

1. **retrieval miss** - the span was admitted, authorized and unfiltered, and the retriever
   did not return it. The retriever's problem, and the number to improve.
2. **not admitted** - the text is in the corpus and no verified binding covers it, so no
   evidence span exists. Ingestion's boundary (``docs/STATUS.md`` 25); no retrieval
   strategy can fix it.
3. **excluded by authorization** - relevant to the question, unreadable by the requester.
   Correct behaviour, and the same expectation resolves to a hit under a scope that may
   read it. Never counted as a miss, and asserted never to be returned.

The suite ships its own ACL overlay
-----------------------------------

The sample corpus carries no permission labels (``docs/STATUS.md`` 30), so authorization
over it was previously exercised only by a test that labelled one document for the duration
of a single test. A benchmark cannot measure authorization against a corpus that has none,
so the suite **declares** the labels it wants and the runner applies them through the store
port - exactly what an operator applying an ACL does. ``manifest.yaml`` is deliberately not
touched: registration metadata is operator input, and rewriting it would change what every
other suite ingests.

Where the gold labels live
--------------------------

In ``data/benchmarks/``, and they are read by this module - which is the evaluation
harness, not the system under test. The separation is structural rather than promised:
``tests/contract/test_layering.py`` allows ``app/domain``, ``app/ingestion``,
``app/assumptions``, ``app/retrieval`` and ``app/application`` to import only each other,
so no module on the retrieval path can import ``app.evaluation`` and therefore none can
reach a label file.

There is no model call in this module, and no parameter through which one could be
supplied.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.config.settings import REPO_ROOT
from app.domain.interfaces.llm import EmbeddingPort
from app.domain.interfaces.retrieval import (
    MAX_TOP_K,
    RETRIEVAL_CONFIG_VERSION,
    EvidenceCandidate,
    RetrievalQuery,
)
from app.domain.services import scope_lattice
from app.domain.value_objects.enums import RetrievalStrategy, SourceType
from app.domain.value_objects.identifiers import TenantId
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.evaluation.corpus import CorpusStores, load_corpus
from app.evaluation.retrieval_metrics import (
    K_VALUES,
    BenchmarkMetrics,
    ExpectationOutcome,
    ExpectationResult,
    QueryScore,
    score,
)
from app.retrieval.hybrid import HybridConfig, HybridOutcome

#: The gold labels. Read by the harness and by tests; unreachable from the retrieval path.
SUITE_PATH: Final = REPO_ROOT / "data" / "benchmarks" / "retrieval_queries.yaml"

#: Every query runs at the maximum ``top_k`` and recall@k is derived by truncating the
#: ranking. That is exactly equivalent to running at each ``k`` - the ordering is total and
#: deterministic, so a prefix of the long run is the short run - and it buys the one thing
#: separate runs could not give: a miss can be reported as *shares no term with the query*
#: rather than merely *ranked below k*, because a span that scored at all has a rank here.
MEASURE_TOP_K: Final = MAX_TOP_K

#: Corpus paths are addressed by the URI scheme the manifest registers them under.
URI_PREFIX: Final = "corpus://"

Name = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")]


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


class ExpectedEvidence(BaseModel):
    """One span an analyst judged relevant to the query, addressed by source and quote.

    This is a **relevance judgment about the question**, not a prediction about the
    retriever and not a statement about who may read it. That distinction is what lets the
    same expectation be a hit under one permission scope and a correct withholding under
    another, without the label changing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Corpus-relative path, as registered in ``data/sample_corpus/manifest.yaml``.
    source: str = Field(min_length=1)
    #: Verbatim text from the document, occurring exactly once in it.
    quote: str = Field(min_length=1)
    why: str | None = None


class BenchmarkQuery(BaseModel):
    """One labelled query. The field names mirror ``RetrievalQuery`` deliberately.

    ``tenant`` and ``permission_scope`` are **required**, and they name entries declared at
    suite level rather than carrying raw values. Required because the production contract
    makes them required for the same reason - a scope defaulting to "no filtering" would
    make forgetting the parameter the unrestricted case - and indirected because a raw
    32-hex tenant id repeated on every query is a thing nobody proof-reads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Name
    #: What the query is testing. Free-form, and reported per category, because a pooled
    #: recall figure hides that exact-terminology and paraphrase recall are different
    #: numbers with different causes.
    category: Name
    text: str = Field(min_length=1)
    tenant: Name
    permission_scope: Name
    #: Only where a temporal question is being asked. Absent means no as-of filter, and the
    #: report says so rather than substituting today.
    as_of: dt.date | None = None
    #: A ``ScopeSelector`` expression, e.g. ``region:EU``. Filters on what a span binds.
    scope: str | None = None
    source_types: tuple[SourceType, ...] = ()
    #: Empty is meaningful: a negative control, whose correct answer is nothing.
    expect: tuple[ExpectedEvidence, ...] = ()
    note: str | None = None

    @model_validator(mode="after")
    def _scope_parses(self) -> Self:
        if self.scope is not None:
            ScopeSelector.parse(self.scope)
        return self

    @property
    def as_of_datetime(self) -> dt.datetime | None:
        if self.as_of is None:
            return None
        return dt.datetime.combine(self.as_of, dt.time(tzinfo=dt.UTC))


class DocumentLabels(BaseModel):
    """An ACL the suite asks the runner to apply to one corpus document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)


class RetrievalSuite(BaseModel):
    """The labelled benchmark, as loaded.

    ``retrieval_config_version`` and ``scorer_version`` are **pinned here and checked
    against the running retriever**. A measured recall figure belongs to one scorer and one
    filter set; if either moves and the numbers are reprinted unchanged, the report is
    quietly wrong. Failing loudly is the cheapest possible guard against that.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    retrieval_config_version: str = Field(min_length=1)
    #: The lexical scorer these labels were last measured against.
    scorer_version: str = Field(min_length=1)
    #: The vector scorer, when one has been measured. Optional because the suite predates
    #: the vector strategy and its labels are strategy-independent: an expectation is a
    #: relevance judgment about the question, so the same labels score both retrievers.
    vector_scorer_version: str | None = None
    #: The fused scorer. It contains both input fingerprints, so retuning either strategy
    #: or swapping the embedder invalidates this pin - which is the point.
    hybrid_scorer_version: str | None = None
    #: Named tenants. More than one so tenant isolation can be probed from inside the
    #: suite. Typed as an identifier, so a mistyped tenant is a load error rather than a
    #: query that ran against a corpus nobody owns.
    tenants: Mapping[Name, TenantId]
    #: Named permission scopes, each a grant list. An empty list is the least-privileged
    #: scope, not a wildcard.
    scopes: Mapping[Name, tuple[str, ...]]
    document_labels: tuple[DocumentLabels, ...] = ()
    queries: tuple[BenchmarkQuery, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _references_resolve(self) -> Self:
        """Every name a query uses must be declared, and no id may repeat.

        Checked at load time rather than at run time so a typo in a scope name is a load
        error instead of a query that silently ran under the wrong authorization.
        """
        seen: set[str] = set()
        for query in self.queries:
            if query.id in seen:
                raise ValueError(f"duplicate query id {query.id!r}")
            seen.add(query.id)
            if query.tenant not in self.tenants:
                raise ValueError(f"{query.id}: undeclared tenant {query.tenant!r}")
            if query.permission_scope not in self.scopes:
                raise ValueError(f"{query.id}: undeclared scope {query.permission_scope!r}")
        return self

    def pinned_scorer(self, strategy: RetrievalStrategy) -> str | None:
        """The scorer this suite's figures were measured under, for one strategy."""
        if strategy is RetrievalStrategy.VECTOR:
            return self.vector_scorer_version
        if strategy is RetrievalStrategy.HYBRID:
            return self.hybrid_scorer_version
        return self.scorer_version

    def tenant_id(self, name: str) -> str:
        return self.tenants[name]

    def scope(self, name: str) -> PermissionScope:
        return PermissionScope(grants=self.scopes[name])

    @property
    def all_grants(self) -> tuple[str, ...]:
        """Every grant the suite issues, across all its scopes and document labels.

        The runner holds these while *classifying* an expectation, because a benchmark has
        to be able to see the span it withheld in order to report that it withheld it.
        Deliberately assembled from the suite's own declarations rather than from a wildcard
        grant - there is no wildcard, and inventing one here would put a bypass in the
        codebase for the sake of a report.
        """
        issued = {grant for grants in self.scopes.values() for grant in grants}
        issued.update(label for entry in self.document_labels for label in entry.labels)
        return tuple(sorted(issued))

    def resolve_query(self, query: BenchmarkQuery) -> RetrievalQuery:
        """Turn a labelled query into the production contract, at the measurement ``top_k``."""
        return RetrievalQuery(
            tenant_id=self.tenant_id(query.tenant),
            text=query.text,
            permission_scope=self.scope(query.permission_scope),
            source_types=query.source_types,
            scope=ScopeSelector.parse(query.scope) if query.scope is not None else None,
            as_of=query.as_of_datetime,
            top_k=MEASURE_TOP_K,
        )


def load_suite(path: Path = SUITE_PATH) -> RetrievalSuite:
    """Load and validate the labelled suite. A malformed label is a load error."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return RetrievalSuite.model_validate(raw)


# ---------------------------------------------------------------------------
# Resolving a label against the corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Resolution:
    """Where a (source, quote) label landed. Exactly one field is set."""

    candidate: EvidenceCandidate | None = None
    defect: str | None = None
    #: True when the quote was found in the document but no admitted span covers it.
    not_admitted: bool = False


class _CorpusView:
    """The corpus as the benchmark needs to read it: text by path, spans by path.

    Built once per run. It holds candidates for **every** declared tenant and under every
    grant the suite issues, because classification needs to reason about spans a given
    query may not see - which is the difference between reporting a withheld span and
    losing track of it.
    """

    def __init__(
        self,
        *,
        corpus: CorpusStores,
        retrieval: RetrieveEvidenceUseCase,
        tenants: Iterable[str],
        grants: Sequence[str],
    ) -> None:
        scope = PermissionScope(grants=tuple(grants))
        self._by_path: dict[str, list[EvidenceCandidate]] = {}
        self._text: dict[str, str] = {}
        self._candidates: list[EvidenceCandidate] = []

        for tenant_id in dict.fromkeys(tenants):
            for candidate in retrieval.authorized_candidates(tenant_id, scope).candidates:
                path = candidate.document.uri.removeprefix(URI_PREFIX)
                self._by_path.setdefault(path, []).append(candidate)
                self._candidates.append(candidate)

        for path, report in corpus.reports.items():
            text = corpus.sources.get_text(report.version_id)
            if text is not None:
                self._text[path] = text

    @property
    def span_count(self) -> int:
        return len(self._candidates)

    @property
    def documents(self) -> tuple[str, ...]:
        return tuple(sorted(self._text))

    def resolve(self, expected: ExpectedEvidence) -> _Resolution:
        """Find the admitted span containing ``expected.quote``.

        Four answers, and the order matters: an unknown source and a missing quote are
        defects in the *label*, a quote covered by no span is a property of the *corpus*,
        and only the last case hands anything to the retriever.
        """
        text = self._text.get(expected.source)
        if text is None:
            return _Resolution(defect=f"unknown source {expected.source!r}")

        occurrences = text.count(expected.quote)
        if occurrences == 0:
            return _Resolution(defect=f"quote absent from {expected.source}")
        if occurrences > 1:
            return _Resolution(
                defect=f"quote occurs {occurrences} times in {expected.source}; it must be unique"
            )

        start = text.index(expected.quote)
        end = start + len(expected.quote)
        covering = [
            candidate
            for candidate in self._by_path.get(expected.source, ())
            if candidate.evidence.char_start <= start and end <= candidate.evidence.char_end
        ]
        if not covering:
            return _Resolution(not_admitted=True)
        if len(covering) > 1:
            return _Resolution(
                defect=(
                    f"{len(covering)} admitted spans cover the quote in {expected.source}; "
                    "the label does not identify one"
                )
            )
        return _Resolution(candidate=covering[0])


def _exclusion(candidate: EvidenceCandidate, query: RetrievalQuery) -> ExpectationOutcome | None:
    """Which filter, if any, removes ``candidate`` from ``query``'s corpus.

    A **diagnostic**, and it is written here rather than imported because the retriever's
    filter chain is deliberately private - it enforces authorization by never iterating an
    unreadable span, and exposing a per-candidate entry point into it would weaken exactly
    that. The order below mirrors ``lexical._first_failing_filter``, and each decision is
    taken by the same public predicate the retriever calls: ``is_readable_by``,
    ``is_in_force`` and ``scope_lattice.intersects``.

    Divergence from the retriever is caught rather than assumed: an expectation charged to a
    bucket here must be matched by a non-zero count in that bucket on the real trace, which
    ``tests/evaluation/test_retrieval_benchmark.py`` asserts for every query.
    """
    if candidate.tenant_id != query.tenant_id:
        return ExpectationOutcome.EXCLUDED_BY_TENANT
    if not candidate.is_readable_by(query.permission_scope):
        return ExpectationOutcome.EXCLUDED_BY_PERMISSION
    if query.source_types and candidate.source_type not in query.source_types:
        return ExpectationOutcome.EXCLUDED_BY_SOURCE_TYPE
    if query.as_of is not None and not candidate.is_in_force(query.as_of):
        return ExpectationOutcome.EXCLUDED_BY_AS_OF
    if query.scope is not None:
        if not candidate.scopes:
            return ExpectationOutcome.EXCLUDED_UNSCOPED
        if not any(scope_lattice.intersects(bound, query.scope) for bound in candidate.scopes):
            return ExpectationOutcome.EXCLUDED_BY_SCOPE
    return None


def classify(
    *,
    query: BenchmarkQuery,
    expected: ExpectedEvidence,
    resolution: _Resolution,
    resolved_query: RetrievalQuery,
    ranks: Mapping[str, int],
) -> ExpectationResult:
    """Decide the single outcome of one expectation under one query."""
    if resolution.defect is not None:
        return ExpectationResult(
            query_id=query.id,
            source=expected.source,
            quote=expected.quote,
            outcome=ExpectationOutcome.LABEL_DEFECT,
            why=resolution.defect,
        )
    if resolution.not_admitted or resolution.candidate is None:
        return ExpectationResult(
            query_id=query.id,
            source=expected.source,
            quote=expected.quote,
            outcome=ExpectationOutcome.NOT_ADMITTED,
            why=expected.why,
        )

    candidate = resolution.candidate
    excluded = _exclusion(candidate, resolved_query)
    if excluded is not None:
        return ExpectationResult(
            query_id=query.id,
            source=expected.source,
            quote=expected.quote,
            outcome=excluded,
            evidence_id=candidate.evidence_id,
            why=expected.why,
        )

    rank = ranks.get(candidate.evidence_id)
    return ExpectationResult(
        query_id=query.id,
        source=expected.source,
        quote=expected.quote,
        outcome=(
            ExpectationOutcome.RETRIEVED if rank is not None else ExpectationOutcome.RETRIEVAL_MISS
        ),
        evidence_id=candidate.evidence_id,
        rank=rank,
        why=expected.why,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkReport:
    """One scored run of the suite, with the provenance needed to compare two of them."""

    suite_version: str
    #: Which strategy produced it. Present because the two are scored separately and never
    #: merged - a report that did not say which retriever ran would be unreadable.
    strategy: RetrievalStrategy
    #: Observed on the retriever, not copied from the suite. The suite's pinned values are
    #: compared against these by :meth:`version_drift`.
    scorer_version: str
    retrieval_config_version: str
    #: Admitted, retrievable spans across the declared tenants.
    corpus_spans: int
    #: Documents the suite's ACL overlay restricted, with their labels.
    restricted: tuple[tuple[str, tuple[str, ...]], ...]
    queries: tuple[QueryScore, ...]
    metrics: BenchmarkMetrics
    k_values: tuple[int, ...] = K_VALUES

    @property
    def label_defects(self) -> tuple[ExpectationResult, ...]:
        return tuple(
            e
            for q in self.queries
            for e in q.expectations
            if e.outcome is ExpectationOutcome.LABEL_DEFECT
        )

    @property
    def leaks(self) -> tuple[ExpectationResult, ...]:
        return tuple(e for q in self.queries for e in q.leaked)

    def version_drift(self, suite: RetrievalSuite) -> tuple[str, ...]:
        """Disagreements between the pinned versions and the retriever that just ran.

        Each strategy is compared against its **own** pin. A vector run must not be able
        to satisfy the lexical pin or vice versa: the two scorers are unrelated, and a
        crossed comparison would let one strategy's figures be reprinted under the other's
        heading.
        """
        drift: list[str] = []
        pinned = suite.pinned_scorer(self.strategy)
        if pinned is None:
            drift.append(f"the suite pins no scorer for the {self.strategy.value} strategy")
        elif pinned != self.scorer_version:
            drift.append(
                f"scorer_version: suite pins {pinned!r}, retriever reports {self.scorer_version!r}"
            )
        if suite.retrieval_config_version != self.retrieval_config_version:
            drift.append(
                f"retrieval_config_version: suite pins {suite.retrieval_config_version!r}, "
                f"retriever reports {self.retrieval_config_version!r}"
            )
        return tuple(drift)


async def run_suite(
    suite: RetrievalSuite,
    *,
    corpus: CorpusStores | None = None,
) -> BenchmarkReport:
    """Ingest the corpus if needed, apply the ACL overlay, run every query, score it.

    Deterministic end to end: the corpus load replays recorded proposals, the retriever is
    a pure function of its candidates, and the ranking has a total tie break. Two runs
    return the same report.
    """
    stores = corpus if corpus is not None else await load_corpus()
    if stores.extraction_failures:
        raise RuntimeError(
            "the corpus load reported extraction failures, so no recall figure over it "
            f"would mean anything: {stores.extraction_failures}"
        )

    restricted = _apply_document_labels(stores, suite.document_labels)
    retrieval = RetrieveEvidenceUseCase(sources=stores.sources, assumptions=stores.assumptions)
    view = _CorpusView(
        corpus=stores,
        retrieval=retrieval,
        tenants=suite.tenants.values(),
        grants=suite.all_grants,
    )

    scores: list[QueryScore] = []
    scorer_version = ""
    for query in suite.queries:
        resolved = suite.resolve_query(query)
        outcome = retrieval.execute(resolved)
        scorer_version = outcome.scorer_version
        ranks = {result.evidence_id: result.rank for result in outcome.results}

        scores.append(
            QueryScore(
                query_id=query.id,
                category=query.category,
                returned=outcome.evidence_ids,
                trace=outcome.trace,
                expectations=tuple(
                    classify(
                        query=query,
                        expected=expected,
                        resolution=view.resolve(expected),
                        resolved_query=resolved,
                        ranks=ranks,
                    )
                    for expected in query.expect
                ),
            )
        )

    return BenchmarkReport(
        suite_version=suite.version,
        strategy=RetrievalStrategy.LEXICAL,
        scorer_version=scorer_version,
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        corpus_spans=view.span_count,
        restricted=restricted,
        queries=tuple(scores),
        metrics=score(scores),
    )


async def run_vector_suite(
    suite: RetrievalSuite,
    *,
    embedder: EmbeddingPort,
    corpus: CorpusStores | None = None,
) -> BenchmarkReport:
    """The same suite, the same labels, scored by cosine similarity instead of BM25.

    A separate function from :func:`run_suite` rather than a flag on it, for the same
    reason the use case has two methods: the lexical baseline is only a baseline if work on
    the second strategy cannot perturb it, and a shared code path with a mode switch is
    exactly how that protection gets lost.

    ``embedder`` is required and has no default. A benchmark whose embedder could be omitted
    would silently measure whichever one happened to be wired in, and a recall figure
    belongs to exactly one embedder - which is why the embedder's identity travels into
    ``scorer_version`` and is compared against the suite's pin.
    """
    stores = corpus if corpus is not None else await load_corpus()
    if stores.extraction_failures:
        raise RuntimeError(
            "the corpus load reported extraction failures, so no recall figure over it "
            f"would mean anything: {stores.extraction_failures}"
        )

    restricted = _apply_document_labels(stores, suite.document_labels)
    retrieval = RetrieveEvidenceUseCase(sources=stores.sources, assumptions=stores.assumptions)
    view = _CorpusView(
        corpus=stores,
        retrieval=retrieval,
        tenants=suite.tenants.values(),
        grants=suite.all_grants,
    )

    scores: list[QueryScore] = []
    scorer_version = ""
    for query in suite.queries:
        resolved = suite.resolve_query(query)
        outcome = await retrieval.execute_vector(resolved, embedder=embedder)
        scorer_version = outcome.scorer_version
        ranks = {result.evidence_id: result.rank for result in outcome.results}

        scores.append(
            QueryScore(
                query_id=query.id,
                category=query.category,
                returned=outcome.evidence_ids,
                trace=outcome.trace,
                expectations=tuple(
                    classify(
                        query=query,
                        expected=expected,
                        resolution=view.resolve(expected),
                        resolved_query=resolved,
                        ranks=ranks,
                    )
                    for expected in query.expect
                ),
            )
        )

    return BenchmarkReport(
        suite_version=suite.version,
        strategy=RetrievalStrategy.VECTOR,
        scorer_version=scorer_version,
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        corpus_spans=view.span_count,
        restricted=restricted,
        queries=tuple(scores),
        metrics=score(scores),
    )


async def run_hybrid_suite(
    suite: RetrievalSuite,
    *,
    embedder: EmbeddingPort,
    corpus: CorpusStores | None = None,
    config: HybridConfig | None = None,
) -> tuple[BenchmarkReport, tuple[HybridOutcome, ...]]:
    """The same suite and the same labels, scored by reciprocal rank fusion.

    Returns the report **and the fused outcomes**, because the headline recall figures cannot
    express the thing this slice exists to measure: which strategy contributed what. The
    lexical-only / vector-only / both split is read off the outcomes by
    :func:`~app.retrieval.hybrid.strategy_split`, and a recall number cannot carry it.

    A third function rather than a flag on the other two, for the reason they are separate:
    both baselines must stay unperturbable by work on fusion.
    """
    stores = corpus if corpus is not None else await load_corpus()
    if stores.extraction_failures:
        raise RuntimeError(
            "the corpus load reported extraction failures, so no recall figure over it "
            f"would mean anything: {stores.extraction_failures}"
        )

    restricted = _apply_document_labels(stores, suite.document_labels)
    retrieval = RetrieveEvidenceUseCase(sources=stores.sources, assumptions=stores.assumptions)
    view = _CorpusView(
        corpus=stores,
        retrieval=retrieval,
        tenants=suite.tenants.values(),
        grants=suite.all_grants,
    )

    scores: list[QueryScore] = []
    fused: list[HybridOutcome] = []
    scorer_version = ""
    for query in suite.queries:
        resolved = suite.resolve_query(query)
        hybrid = await retrieval.execute_hybrid(resolved, embedder=embedder, config=config)
        fused.append(hybrid)
        scorer_version = hybrid.outcome.scorer_version
        ranks = {result.evidence_id: result.rank for result in hybrid.outcome.results}

        scores.append(
            QueryScore(
                query_id=query.id,
                category=query.category,
                returned=hybrid.outcome.evidence_ids,
                trace=hybrid.outcome.trace,
                expectations=tuple(
                    classify(
                        query=query,
                        expected=expected,
                        resolution=view.resolve(expected),
                        resolved_query=resolved,
                        ranks=ranks,
                    )
                    for expected in query.expect
                ),
            )
        )

    report = BenchmarkReport(
        suite_version=suite.version,
        strategy=RetrievalStrategy.HYBRID,
        scorer_version=scorer_version,
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        corpus_spans=view.span_count,
        restricted=restricted,
        queries=tuple(scores),
        metrics=score(scores),
    )
    return report, tuple(fused)


def _apply_document_labels(
    stores: CorpusStores, labels: Sequence[DocumentLabels]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Re-register the named documents carrying the suite's ACL labels.

    Through the store port, which is what an operator applying an ACL does, so the join,
    the retriever and the trace all see a genuinely restricted document rather than a
    fixture-shaped one. The stores are the in-memory ones this run created, so nothing
    outside it is affected and ``manifest.yaml`` stays untouched.
    """
    applied: list[tuple[str, tuple[str, ...]]] = []
    for entry in labels:
        uri = f"{URI_PREFIX}{entry.source}"
        document = stores.sources.find_document_by_uri(stores.tenant_id, uri)
        if document is None:
            raise ValueError(f"the suite labels {entry.source!r}, which the corpus does not hold")
        stores.sources.put_document(document.model_copy(update={"permissions": entry.labels}))
        applied.append((entry.source, entry.labels))
    return tuple(sorted(applied))
