"""Recall accounting for the retrieval benchmark. Pure arithmetic over labelled outcomes.

Separated from the runner deliberately: every number below is computed from values, so the
metric definitions are testable without a corpus, without a store and without running a
retriever. A recall figure that can only be produced by running the whole pipeline is a
figure nobody can check.

**The whole point of this module is the denominator.** A single "recall" number over a
corpus like this one would be uninterpretable, because an expected span can fail to come
back for reasons that are not the retriever's:

* it was never **admitted** - the corpus stores it as text, but no verified binding covers
  it, so no ``EvidenceItem`` exists and nothing can retrieve it (``docs/STATUS.md`` 25);
* it was **withheld by authorization** - the requester may not read its holder document,
  which is the retriever working correctly;
* it was removed by a **content filter** the query itself asked for, which means the label
  and the query disagree and is therefore a defect in the *suite*, not in the retriever.

So three quantities are reported instead of one:

``coverage``
    of the labelled evidence, how much the retriever could reach at all.
``recall@k``
    of the evidence it could reach, how much it ranked in the top *k*. This is the number
    that measures the retriever, and the one a later strategy has to beat.
``end_to_end_recall@k``
    of everything an analyst labelled relevant, how much actually arrived. This is the
    number a *user* experiences, and it is never higher than ``recall@k``.

Reporting only the second flatters the retriever; reporting only the third blames it for
the ingestion boundary. Both are printed, side by side, always.

``None`` rather than ``0.0`` is returned wherever a denominator is empty. A ratio over
nothing is not zero, and a negative-control category scoring "0.0 recall" would read as a
failure when emptiness is the correct and intended answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from app.domain.interfaces.retrieval import RetrievalTrace

#: The cut-offs recall is reported at. Fixed rather than configurable: a benchmark whose
#: reported ``k`` moves between runs cannot be compared with its own history.
K_VALUES: Final = (1, 3, 5, 10)


class ExpectationOutcome(StrEnum):
    """What became of one labelled piece of expected evidence, under one query.

    Exactly one outcome per expectation, so the counts partition the label set. The
    ``excluded_*`` members deliberately carry the same names as the
    :class:`~app.domain.interfaces.retrieval.RetrievalTrace` buckets, because they are
    claims about the same event seen from the other side: the trace says *how many* spans a
    filter removed, and an outcome here says *which labelled one* it removed.
    """

    #: Returned by the retriever. Carries a rank.
    RETRIEVED = "retrieved"
    #: The retriever held it, was allowed to return it, and did not: it shares no term with
    #: the query. This is the miss the benchmark exists to count.
    RETRIEVAL_MISS = "retrieval_miss"
    #: No verified binding covers the labelled text, so no evidence span exists to return.
    #: A property of ingestion, and unfixable by any retrieval strategy.
    NOT_ADMITTED = "not_admitted"
    #: Relevant to the question, unreadable by the requester. Correct behaviour.
    EXCLUDED_BY_PERMISSION = "excluded_by_permission"
    EXCLUDED_BY_TENANT = "excluded_by_tenant"
    EXCLUDED_BY_SOURCE_TYPE = "excluded_by_source_type"
    EXCLUDED_BY_AS_OF = "excluded_by_as_of"
    EXCLUDED_BY_SCOPE = "excluded_by_scope"
    EXCLUDED_UNSCOPED = "excluded_unscoped"
    #: The label itself is broken: unknown source, or a quote that is absent from the
    #: document or occurs more than once in it. Never counted as a retrieval result of any
    #: kind, and asserted to be zero by the suite tests.
    LABEL_DEFECT = "label_defect"


#: The outcomes that describe a span the retriever actually had in front of it. The recall
#: denominator, and nothing else.
RETRIEVABLE: Final = frozenset({ExpectationOutcome.RETRIEVED, ExpectationOutcome.RETRIEVAL_MISS})

#: Removed by a filter the query asked for. Non-zero here means a label and a query
#: disagree - the suite is wrong, not the retriever.
FILTERED: Final = frozenset(
    {
        ExpectationOutcome.EXCLUDED_BY_TENANT,
        ExpectationOutcome.EXCLUDED_BY_SOURCE_TYPE,
        ExpectationOutcome.EXCLUDED_BY_AS_OF,
        ExpectationOutcome.EXCLUDED_BY_SCOPE,
        ExpectationOutcome.EXCLUDED_UNSCOPED,
    }
)


def ratio(numerator: int, denominator: int) -> float | None:
    """``numerator / denominator``, or ``None`` when there is nothing to divide by."""
    return (numerator / denominator) if denominator else None


@dataclass(frozen=True)
class ExpectationResult:
    """One labelled expectation, resolved and classified against one query's results."""

    query_id: str
    #: Corpus-relative path of the document the label names.
    source: str
    #: The labelled text, verbatim, as written in the suite.
    quote: str
    outcome: ExpectationOutcome
    #: The span the label resolved to, or ``None`` when it resolved to no span at all
    #: (``NOT_ADMITTED``, ``LABEL_DEFECT``).
    evidence_id: str | None = None
    #: 1-based position in the ranking. Present exactly when ``outcome`` is ``RETRIEVED``.
    rank: int | None = None
    #: Why the label was written. Free text from the suite, carried through for the report.
    why: str | None = None

    def __post_init__(self) -> None:
        retrieved = self.outcome is ExpectationOutcome.RETRIEVED
        if retrieved != (self.rank is not None):
            raise ValueError(
                f"{self.query_id}/{self.source}: a rank is present exactly for a retrieved "
                f"expectation, got outcome={self.outcome} rank={self.rank}"
            )
        if self.rank is not None and self.rank < 1:
            raise ValueError(f"{self.query_id}/{self.source}: rank must be 1-based")
        if self.outcome in RETRIEVABLE and self.evidence_id is None:
            raise ValueError(
                f"{self.query_id}/{self.source}: outcome {self.outcome} claims the retriever "
                "held this span, so it must name one"
            )

    @property
    def retrievable(self) -> bool:
        return self.outcome in RETRIEVABLE

    def found_within(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


@dataclass(frozen=True)
class QueryScore:
    """One benchmark query, what came back, and how its labels fared."""

    query_id: str
    category: str
    #: Evidence ids in rank order, as returned.
    returned: tuple[str, ...]
    trace: RetrievalTrace
    expectations: tuple[ExpectationResult, ...] = ()

    @property
    def is_negative_control(self) -> bool:
        """No expected evidence: the correct answer is approximately nothing.

        Defined by the absence of labels rather than by the declared category, so a query
        cannot be scored as a negative control while quietly carrying expectations.
        """
        return not self.expectations

    @property
    def restricts_by_permission(self) -> bool:
        """Whether authorization actually removed anything for this requester.

        Read from the trace rather than from the scope, because a scope that happens to
        read everything is not a restricted run, and averaging it into the authorization
        figure would dilute exactly the thing that figure measures.
        """
        return self.trace.excluded_by_permission > 0

    @property
    def leaked(self) -> tuple[ExpectationResult, ...]:
        """Expectations that were withheld and came back anyway. Must always be empty."""
        returned = set(self.returned)
        return tuple(
            e
            for e in self.expectations
            if e.outcome is ExpectationOutcome.EXCLUDED_BY_PERMISSION
            and e.evidence_id is not None
            and e.evidence_id in returned
        )


@dataclass(frozen=True)
class RecallMetrics:
    """Recall over one set of queries, with the denominators kept apart."""

    queries: int
    expectations: int
    label_defects: int
    #: Expectations that are not label defects. The end-to-end denominator.
    labelled: int
    #: Expectations the retriever could actually reach. The recall denominator.
    retrievable: int
    not_admitted: int
    withheld_by_permission: int
    filtered_by_query: int
    hits: Mapping[int, int] = field(default_factory=dict)
    recall: Mapping[int, float | None] = field(default_factory=dict)
    end_to_end_recall: Mapping[int, float | None] = field(default_factory=dict)
    #: Mean of per-query recall, over queries with at least one reachable expectation.
    #: Reported next to the pooled figure because the two differ whenever queries carry
    #: unequal numbers of labels, and quoting only one of them is a choice worth showing.
    macro_recall: Mapping[int, float | None] = field(default_factory=dict)
    outcomes: Mapping[ExpectationOutcome, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float | None:
        """Of the labelled evidence, the fraction retrieval can reach at all."""
        return ratio(self.retrievable, self.labelled)


@dataclass(frozen=True)
class AuthorizationMetrics:
    """Recall as a restricted requester experiences it, plus the leak check.

    Two figures that are easy to conflate and must not be. ``recall`` is computed over the
    expectations a restricted requester *may* read, and answers "does authorization damage
    the results they are entitled to?". ``withheld`` counts the expectations they may not
    read, and is not a failure of anything - it is the size of the answer they were
    correctly not given.

    ``leaks`` is the only metric in this module that is a security property rather than a
    quality one, and the only one whose acceptable value is fixed in advance: zero. A span
    the requester may not read must never appear in their results, whatever that costs a
    recall figure.
    """

    restricted_queries: int
    expectations: int
    retrievable: int
    withheld: int
    leaks: int
    recall: Mapping[int, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class NegativeControlMetrics:
    """What the system says when the honest answer is nothing.

    ``CLAUDE.md`` requires the system to be quiet on quiet changes. The retrieval-level
    form of that bar is this: a query no admitted span answers should come back empty.
    """

    queries: int
    silent: int
    returned: int

    @property
    def noise_per_query(self) -> float | None:
        return ratio(self.returned, self.queries)


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Everything the suite measured, sliced the ways that change the reading."""

    overall: RecallMetrics
    authorization: AuthorizationMetrics
    negative_controls: NegativeControlMetrics
    by_category: Mapping[str, RecallMetrics]


def score_recall(
    queries: Sequence[QueryScore], *, k_values: Sequence[int] = K_VALUES
) -> RecallMetrics:
    """Pool the expectations of ``queries`` and count them at each cut-off."""
    expectations = [e for q in queries for e in q.expectations]
    outcomes: dict[ExpectationOutcome, int] = {}
    for expectation in expectations:
        outcomes[expectation.outcome] = outcomes.get(expectation.outcome, 0) + 1

    defects = outcomes.get(ExpectationOutcome.LABEL_DEFECT, 0)
    labelled = len(expectations) - defects
    retrievable = sum(1 for e in expectations if e.retrievable)
    hits = {k: sum(1 for e in expectations if e.found_within(k)) for k in k_values}

    return RecallMetrics(
        queries=len(queries),
        expectations=len(expectations),
        label_defects=defects,
        labelled=labelled,
        retrievable=retrievable,
        not_admitted=outcomes.get(ExpectationOutcome.NOT_ADMITTED, 0),
        withheld_by_permission=outcomes.get(ExpectationOutcome.EXCLUDED_BY_PERMISSION, 0),
        filtered_by_query=sum(outcomes.get(outcome, 0) for outcome in FILTERED),
        hits=hits,
        recall={k: ratio(hits[k], retrievable) for k in k_values},
        end_to_end_recall={k: ratio(hits[k], labelled) for k in k_values},
        macro_recall=_macro_recall(queries, k_values),
        outcomes=outcomes,
    )


def _macro_recall(
    queries: Sequence[QueryScore], k_values: Sequence[int]
) -> dict[int, float | None]:
    """Mean per-query recall, ignoring queries with nothing reachable to find.

    A query whose every label is unreachable has no recall to average - counting it as a
    zero would charge the retriever for the ingestion boundary, and counting it as a one
    would reward it for a question it never answered.
    """
    scored = [q for q in queries if any(e.retrievable for e in q.expectations)]
    result: dict[int, float | None] = {}
    for k in k_values:
        per_query = [
            ratio(
                sum(1 for e in q.expectations if e.found_within(k)),
                sum(1 for e in q.expectations if e.retrievable),
            )
            for q in scored
        ]
        present = [value for value in per_query if value is not None]
        result[k] = (sum(present) / len(present)) if present else None
    return result


def score_authorization(
    queries: Sequence[QueryScore], *, k_values: Sequence[int] = K_VALUES
) -> AuthorizationMetrics:
    """Recall over the queries where authorization actually removed something."""
    restricted = [q for q in queries if q.restricts_by_permission]
    expectations = [e for q in restricted for e in q.expectations]
    retrievable = sum(1 for e in expectations if e.retrievable)
    hits = {k: sum(1 for e in expectations if e.found_within(k)) for k in k_values}

    return AuthorizationMetrics(
        restricted_queries=len(restricted),
        expectations=len(expectations),
        retrievable=retrievable,
        withheld=sum(
            1 for e in expectations if e.outcome is ExpectationOutcome.EXCLUDED_BY_PERMISSION
        ),
        # Counted over *every* query, not only the restricted ones. A leak is a leak
        # wherever it happens, and scoping the check to the queries expected to restrict
        # would be trusting the suite to name the risky cases correctly.
        leaks=sum(len(q.leaked) for q in queries),
        recall={k: ratio(hits[k], retrievable) for k in k_values},
    )


def score_negative_controls(queries: Sequence[QueryScore]) -> NegativeControlMetrics:
    controls = [q for q in queries if q.is_negative_control]
    return NegativeControlMetrics(
        queries=len(controls),
        silent=sum(1 for q in controls if not q.returned),
        returned=sum(len(q.returned) for q in controls),
    )


def score(queries: Sequence[QueryScore], *, k_values: Sequence[int] = K_VALUES) -> BenchmarkMetrics:
    """Every metric, over every query. Deterministic, and a pure function of its input."""
    categories = sorted({q.category for q in queries})
    return BenchmarkMetrics(
        overall=score_recall(queries, k_values=k_values),
        authorization=score_authorization(queries, k_values=k_values),
        negative_controls=score_negative_controls(queries),
        by_category={
            category: score_recall(
                [q for q in queries if q.category == category], k_values=k_values
            )
            for category in categories
        },
    )
