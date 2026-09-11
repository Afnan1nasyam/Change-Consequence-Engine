"""Build the blind-spot partition, retrieve it, invoke the hunter, suppress Class A duplicates.

The application path for the second Class B stage. It differs from
:mod:`app.application.use_cases.hunt_consequences` in exactly one respect that matters: it
**retrieves its own evidence**, because its whole reason to exist is to read what the main
query did not hand over.

Three extra queries, all through the same :class:`RetrievalService` with the *same*
:class:`~app.domain.value_objects.permissions.PermissionScope` and tenant as the main one:

1. the same text at a widened ``top_k``, which reaches the low-ranked region;
2. the same text restricted to the source types the main retrieval returned nothing from;
3. **the undocumented keys as the query text** - the single-holder and unwritten-rule keys
   from the existing decay report.

The third is the one that earns the stage. Slices 1 and 2 are reachable only through the
change's own words, which is exactly the hook the main analysis already pulled; measured over
the sample corpus they return nothing the main hunter had not already been shown. Querying
the *subject* of the undocumented coupling instead reaches spans the change's wording never
would - which is what "a different evidence basis" has to mean to be worth a call.

Going through the retriever rather than reading the coupling index directly is deliberate.
The index is tenant-scoped but not permission-filtered, so assembling a partition from
bindings would hand this agent spans the requester is not cleared to read - an isolation
defect dressed as a feature.

Suppression is the existing deterministic rule, reused rather than reimplemented: the same
:func:`~app.assumptions.class_a.merge_class_b` the main hunt uses, so a blind-spot finding
that duplicates a Class A consequence is absorbed by the identical
``(assumption_key, affected_source_version_id)`` comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from app.application.orchestration.retrieval_service import (
    FinalizedRetrieval,
    RetrievalRequest,
    RetrievalService,
)
from app.assumptions.class_a import merge_class_b
from app.blindspots.hunter import BlindSpotHunter, BlindSpotOutcome, SilenceRecord
from app.blindspots.partition import (
    EvidencePartition,
    PartitionSlice,
    partition_from,
    undocumented_keys,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding, ClassBMergeOutcome
from app.domain.entities.decay import KnowledgeDecayReport
from app.domain.interfaces.retrieval import MAX_TOP_K, RetrievalOutcome, RetrievalQuery
from app.domain.value_objects.enums import SourceType

logger = logging.getLogger(__name__)

#: How much deeper the partition query reaches than the main one. The low-ranked slice is
#: exactly the difference, so a multiplier of 1 would make the slice empty by construction.
DEPTH_MULTIPLIER: Final = 3

#: ``RetrievalQuery.text`` is bounded; the undocumented-key query is assembled, so it is the
#: one that could exceed the bound.
MAX_QUERY_CHARS: Final = 2048


@dataclass(frozen=True)
class BlindSpotReport:
    """The blind-spot stage, end to end, reported apart from the main Class B stage."""

    #: Hypotheses that survived Class A suppression. Class B by producer, like every
    #: agent-authored finding.
    retained: tuple[ConsequenceHypothesis, ...] = ()
    suppressed: tuple[ConsequenceHypothesis, ...] = ()
    merge: ClassBMergeOutcome | None = None
    class_a: tuple[ClassAFinding, ...] = ()
    outcome: BlindSpotOutcome | None = None
    partition: EvidencePartition | None = None
    #: One extra retrieval per partition query, kept so the stage's cost is visible.
    partition_outcomes: tuple[RetrievalOutcome, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.outcome is not None and self.outcome.succeeded

    @property
    def silences(self) -> tuple[SilenceRecord, ...]:
        return self.outcome.silences if self.outcome else ()

    @property
    def open_questions(self) -> tuple[str, ...]:
        return tuple(record.question for record in self.silences)

    @property
    def yield_by_slice(self) -> dict[str, int]:
        return self.outcome.yield_by_slice if self.outcome else {}

    @property
    def partition_size(self) -> int:
        return len(self.partition.spans) if self.partition else 0

    @property
    def independent_evidence_ids(self) -> tuple[str, ...]:
        """Every id the retained findings rest on. Disjoint from the main analysis by build."""
        found: set[str] = set()
        for hypothesis in self.retained:
            found.update(hypothesis.evidence_ids)
        return tuple(sorted(found))


class HuntBlindSpotsUseCase:
    """Run the blind-spot stage for one change. Holds the hunter and the retriever."""

    def __init__(self, *, hunter: BlindSpotHunter, retrieval: RetrievalService) -> None:
        self._hunter = hunter
        self._retrieval = retrieval

    async def execute(
        self,
        *,
        change: ChangeSpec,
        query: RetrievalQuery,
        main: FinalizedRetrieval,
        main_offered_ids: frozenset[str],
        decay: KnowledgeDecayReport,
        class_a: tuple[ClassAFinding, ...] = (),
        existing_claims: tuple[str, ...] = (),
        correlation_id: str | None = None,
    ) -> BlindSpotReport:
        """Partition, retrieve, invoke, suppress."""
        single, unwritten = undocumented_keys(decay)
        returned_types = {r.candidate.source_type for r in main.outcome.results}

        outcomes: list[RetrievalOutcome] = []
        for partition_query in partition_queries(
            query,
            returned_types=returned_types,
            undocumented_keys=(*single, *unwritten),
        ):
            finalized = await self._retrieval.retrieve(RetrievalRequest(query=partition_query))
            outcomes.append(finalized.outcome)

        partition = partition_from(
            outcomes,
            main_offered_ids=main_offered_ids,
            main_source_types=returned_types,
            single_holder_keys=single,
            unwritten_rule_keys=unwritten,
        )

        outcome = await self._hunter.hunt(
            change=change,
            partition=partition,
            existing_claims=existing_claims,
            correlation_id=correlation_id,
        )

        if not outcome.succeeded:
            return BlindSpotReport(
                class_a=class_a,
                outcome=outcome,
                partition=partition,
                partition_outcomes=tuple(outcomes),
            )

        merge, absorbed = merge_class_b(
            outcome.hypotheses,
            class_a,
            affected_artifact_of=outcome.affected_artifact_of,
        )
        suppressed_ids = set(merge.suppressed_hypothesis_ids)
        retained = tuple(h for h in outcome.hypotheses if h.hypothesis_id not in suppressed_ids)
        suppressed = tuple(h for h in outcome.hypotheses if h.hypothesis_id in suppressed_ids)

        logger.info(
            "blind_spot_stage_complete",
            extra={
                "change_id": change.change_id,
                "partition": len(partition.spans),
                "uncovered_slices": [s.value for s in partition.uncovered],
                "retained": len(retained),
                "suppressed": len(suppressed),
                "silences": len(outcome.silences),
            },
        )

        return BlindSpotReport(
            retained=retained,
            suppressed=suppressed,
            merge=merge,
            class_a=absorbed,
            outcome=outcome,
            partition=partition,
            partition_outcomes=tuple(outcomes),
        )


def partition_queries(
    query: RetrievalQuery,
    *,
    returned_types: set[SourceType],
    undocumented_keys: tuple[str, ...] = (),
) -> tuple[RetrievalQuery, ...]:
    """The queries that reach the partition. Same tenant, same permission scope.

    Derived from the main query by copy, so nothing about the authorization boundary can be
    re-decided here: a partition query that widened the scope would be a privilege escalation
    with a plausible motive. Only ``text``, ``top_k`` and ``source_types`` ever change.
    """
    queries = [query.model_copy(update={"top_k": min(MAX_TOP_K, query.top_k * DEPTH_MULTIPLIER)})]

    unqueried = tuple(sorted(set(SourceType) - returned_types, key=lambda t: t.value))
    if unqueried:
        queries.append(query.model_copy(update={"source_types": unqueried}))

    if undocumented_keys:
        # The keys become the query text. Dot-separated key names are poor lexical hooks on
        # their own, so each is also split into words - `retention.window` looks for
        # "retention" and "window" as well as itself.
        terms: list[str] = []
        for key in undocumented_keys:
            for term in (key, *key.split(".")):
                if term not in terms:
                    terms.append(term)
        queries.append(
            query.model_copy(
                update={
                    "text": " ".join(terms)[:MAX_QUERY_CHARS],
                    "top_k": min(MAX_TOP_K, query.top_k * DEPTH_MULTIPLIER),
                }
            )
        )

    return tuple(queries)


def slices_of(report: BlindSpotReport) -> tuple[PartitionSlice, ...]:
    """Which slices this run's partition actually covered. Empty when the stage did not run."""
    return report.partition.covered if report.partition else ()
