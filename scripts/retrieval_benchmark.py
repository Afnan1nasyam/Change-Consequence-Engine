"""Run the labelled retrieval benchmark and print what the lexical retriever recalled.

    python -m scripts.retrieval_benchmark
    python -m scripts.retrieval_benchmark --strategy vector
    python -m scripts.retrieval_benchmark --per-query
    python -m scripts.retrieval_benchmark --json

Scored over the real sample corpus with extraction replayed, so the run needs no credentials
and no network. The lexical run makes **zero model calls**; the vector run makes embedding
calls only, and only against the offline hashing embedder (``ARCHITECTURE.md`` 11.4).

**The two strategies are printed separately and never merged.** There is no ``--strategy
hybrid``, because fusion needs a weighting nobody has measured and adding one would destroy
both baselines at the same time.

The measured figures this prints are the ones quoted in ``docs/EVALUATION.md``; if they
disagree, the document is stale and the code is right.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from app.domain.value_objects.enums import RetrievalStrategy
from app.evaluation.retrieval_benchmark import (
    BenchmarkReport,
    RetrievalSuite,
    load_suite,
    run_hybrid_suite,
    run_suite,
    run_vector_suite,
)
from app.evaluation.retrieval_metrics import (
    ExpectationOutcome,
    QueryScore,
    RecallMetrics,
)
from app.infrastructure.embeddings.hashing import HashingEmbedder
from app.retrieval.hybrid import HybridOutcome, strategy_split

RULE = "=" * 100


def pct(value: float | None) -> str:
    """A ratio for a report. ``None`` prints as a dash, never as zero."""
    return "  -  " if value is None else f"{value:6.1%}"


def _recall_row(label: str, metrics: RecallMetrics, key: str) -> str:
    source: Mapping[int, float | None] = getattr(metrics, key)
    cells = "".join(pct(source.get(k)).rjust(10) for k in sorted(source))
    return f"  {label:<34}{cells}"


def print_report(report: BenchmarkReport, suite: RetrievalSuite, *, per_query: bool) -> None:
    overall = report.metrics.overall
    ks = sorted(overall.recall)

    print(RULE)
    print(f"RETRIEVAL BENCHMARK - {report.strategy.value} strategy, scored alone")
    print(RULE)
    print(f"suite                     {report.suite_version}")
    print(f"scorer                    {report.scorer_version}")
    print(f"retrieval config          {report.retrieval_config_version}")
    print(f"queries / expectations    {overall.queries} / {overall.expectations}")
    print(f"retrievable corpus        {report.corpus_spans} admitted spans")
    for source, labels in report.restricted:
        print(f"restricted by the suite   {source} -> {list(labels)}")

    if drift := report.version_drift(suite):
        print()
        print("VERSION DRIFT - the pinned scorer is not the one that ran:")
        for line in drift:
            print(f"  ! {line}")

    print()
    print("Reachability of the expected evidence")
    print(f"  labelled                          {overall.labelled}")
    print(f"  retrievable                       {overall.retrievable}  <- the recall denominator")
    print(f"  never admitted (no binding)       {overall.not_admitted}")
    print(f"  withheld by authorization         {overall.withheld_by_permission}")
    print(f"  removed by a query filter         {overall.filtered_by_query}")
    print(f"  label defects                     {overall.label_defects}")
    print(f"  coverage                          {pct(overall.coverage)}")

    print()
    print("Recall".ljust(36) + "".join(f"@{k}".rjust(10) for k in ks))
    print(_recall_row("of reachable evidence", overall, "recall"))
    print(_recall_row("of all labelled evidence", overall, "end_to_end_recall"))
    print(_recall_row("mean per query (reachable)", overall, "macro_recall"))

    authorization = report.metrics.authorization
    print()
    print("Authorization")
    print(f"  restricted queries                {authorization.restricted_queries}")
    print(f"  expectations under them           {authorization.expectations}")
    print(f"  of those, withheld                {authorization.withheld}")
    print(f"  of those, reachable               {authorization.retrievable}")
    cells = "".join(pct(authorization.recall.get(k)).rjust(10) for k in ks)
    print(f"  {'authorization-filtered recall':<34}{cells}")
    print(f"  withheld evidence returned anyway {authorization.leaks}  <- must be 0")

    controls = report.metrics.negative_controls
    print()
    print("Negative controls (correct answer: nothing)")
    print(f"  queries                           {controls.queries}")
    print(f"  silent                            {controls.silent}")
    print(f"  results returned                  {controls.returned}")

    print()
    print("By category".ljust(36) + "".join(f"@{k}".rjust(10) for k in ks) + "   cov   reach")
    for category, metrics in report.metrics.by_category.items():
        cells = "".join(pct(metrics.recall.get(k)).rjust(10) for k in ks)
        print(
            f"  {category:<34}{cells}"
            f"{pct(metrics.coverage):>8}{metrics.retrievable:>5}/{metrics.labelled}"
        )

    if report.label_defects:
        print()
        print("LABEL DEFECTS - the suite is wrong, not the retriever:")
        for defect in report.label_defects:
            print(f"  ! {defect.query_id}: {defect.source}: {defect.why}")

    print()
    print("Misses and unreachable evidence")
    for query in report.queries:
        for expectation in query.expectations:
            if expectation.outcome is ExpectationOutcome.RETRIEVED:
                continue
            print(
                f"  {expectation.outcome.value:<24} {query.query_id:<44} "
                f"{expectation.source} :: {expectation.quote[:52]}"
            )

    if per_query:
        print()
        print(RULE)
        print("PER QUERY")
        print(RULE)
        for query in report.queries:
            _print_query(query)

    print(RULE)


def _print_query(query: QueryScore) -> None:
    trace = query.trace
    print()
    print(f"{query.query_id}  [{query.category}]")
    print(f"  returned {len(query.returned)}; matched {trace.matched} of {trace.considered}")
    excluded = {
        "tenant": trace.excluded_by_tenant,
        "permission": trace.excluded_by_permission,
        "source_type": trace.excluded_by_source_type,
        "as_of": trace.excluded_by_as_of,
        "scope": trace.excluded_by_scope,
        "unscoped": trace.excluded_unscoped,
        "zero_score": trace.excluded_zero_score,
    }
    print(f"  excluded {', '.join(f'{k}={v}' for k, v in excluded.items() if v)}")
    for expectation in query.expectations:
        rank = f"rank {expectation.rank}" if expectation.rank is not None else "-"
        print(f"    {expectation.outcome.value:<24} {rank:<8} {expectation.source}")


def _print_split(fused: Sequence[HybridOutcome]) -> None:
    """Which strategy contributed what. The thing a recall figure cannot express."""
    split = strategy_split(fused)
    returned = split["returned"] or 1
    print()
    print("Strategy contribution (over returned results)")
    for label in ("both", "lexical_only", "vector_only"):
        share = split[label] / returned
        print(f"  {label:<34}{split[label]:>6}{share:>9.1%}")
    print(f"  {'returned in total':<34}{split['returned']:>6}")

    agreements = [h.agreement for h in fused if h.agreement is not None]
    if agreements:
        mean = sum(agreements) / len(agreements)
        print(f"  {'mean per-query agreement':<34}{mean:>15.1%}")


def as_json(report: BenchmarkReport) -> dict[str, Any]:
    """The report as data, for a comparison over time that is not a diff of printed text."""
    overall = report.metrics.overall
    authorization = report.metrics.authorization
    controls = report.metrics.negative_controls
    return {
        "suite_version": report.suite_version,
        "strategy": report.strategy.value,
        "scorer_version": report.scorer_version,
        "retrieval_config_version": report.retrieval_config_version,
        "corpus_spans": report.corpus_spans,
        "queries": overall.queries,
        "expectations": overall.expectations,
        "reachability": {
            "labelled": overall.labelled,
            "retrievable": overall.retrievable,
            "not_admitted": overall.not_admitted,
            "withheld_by_permission": overall.withheld_by_permission,
            "filtered_by_query": overall.filtered_by_query,
            "label_defects": overall.label_defects,
            "coverage": overall.coverage,
        },
        "recall": {str(k): v for k, v in overall.recall.items()},
        "end_to_end_recall": {str(k): v for k, v in overall.end_to_end_recall.items()},
        "macro_recall": {str(k): v for k, v in overall.macro_recall.items()},
        "authorization": {
            "restricted_queries": authorization.restricted_queries,
            "withheld": authorization.withheld,
            "retrievable": authorization.retrievable,
            "leaks": authorization.leaks,
            "recall": {str(k): v for k, v in authorization.recall.items()},
        },
        "negative_controls": {
            "queries": controls.queries,
            "silent": controls.silent,
            "returned": controls.returned,
        },
        "by_category": {
            category: {
                "recall": {str(k): v for k, v in metrics.recall.items()},
                "coverage": metrics.coverage,
                "retrievable": metrics.retrievable,
                "labelled": metrics.labelled,
            }
            for category, metrics in report.metrics.by_category.items()
        },
        "outcomes": {outcome.value: count for outcome, count in sorted(overall.outcomes.items())},
    }


async def run(*, strategy: RetrievalStrategy, per_query: bool, emit_json: bool) -> int:
    suite = load_suite()
    fused: tuple[HybridOutcome, ...] = ()
    if strategy is RetrievalStrategy.VECTOR:
        report = await run_vector_suite(suite, embedder=HashingEmbedder())
    elif strategy is RetrievalStrategy.HYBRID:
        report, fused = await run_hybrid_suite(suite, embedder=HashingEmbedder())
    else:
        report = await run_suite(suite)

    payload = as_json(report)
    if fused:
        payload["strategy_split"] = strategy_split(fused)

    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_report(report, suite, per_query=per_query)
        if fused:
            _print_split(fused)

    # A label defect, a leak or a scorer that is not the pinned one all mean the printed
    # numbers do not measure what they claim to, so the exit code says so.
    if report.label_defects:
        print(f"{len(report.label_defects)} label defect(s)", file=sys.stderr)
        return 1
    if report.metrics.authorization.leaks:
        print(f"{report.metrics.authorization.leaks} authorization leak(s)", file=sys.stderr)
        return 1
    if drift := report.version_drift(suite):
        print(f"pinned versions disagree with the retriever: {drift}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        default=RetrievalStrategy.LEXICAL.value,
        choices=[s.value for s in RetrievalStrategy],
        help="which retriever to score. The two are never combined.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="print each query's trace and its expectations",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    return asyncio.run(
        run(
            strategy=RetrievalStrategy(args.strategy),
            per_query=args.per_query,
            emit_json=args.json,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
