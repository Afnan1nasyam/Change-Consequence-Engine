"""The blind-spot hunter's evidence partition. Deterministic, and disjoint by construction.

A second, more creative pass over the same evidence is not an independent analysis; it is the
same analysis at a different temperature. So the hunter's distinctness is made a property of
its **input**: this module computes a span set that excludes everything the main
consequence-hunter was offered, and refuses to hold one that does not.

Four slices, in the frozen priority order attribution uses. The first two outrank the
retrieval-shaped ones because that is where undocumented coupling actually lives.

Attribution is derived here from the span, never taken from the model: a hunter that could
name its own slice could claim a yield it did not earn.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from app.domain.entities.decay import KnowledgeDecayReport
from app.domain.interfaces.retrieval import RetrievalOutcome
from app.domain.value_objects.enums import DecayCategory, SourceType

#: Bumped when the slice set or the attribution order changes: both change which evidence the
#: hunter sees, and therefore what its output means.
PARTITION_VERSION: Final = "blind-spot-partition-1"

#: Spans handed to one blind-spot call.
MAX_PARTITION_SPANS: Final = 20


class PartitionSlice(StrEnum):
    """Where a partition span came from. Assigned by :func:`partition_from`, never proposed."""

    #: Keys exactly one artifact binds: nothing corroborates the value and nothing depends on
    #: it in writing, so a change there leaves no trace the main analysis could follow.
    SINGLE_HOLDER = "single_holder"
    #: Keys declared only by code, configuration or informal notes.
    UNWRITTEN_RULE = "unwritten_rule"
    #: Source types the main retrieval returned nothing from.
    UNQUERIED_SOURCE_TYPE = "unqueried_source_type"
    #: The corpus region the main query reached but ranked below what it handed over.
    LOW_RANKED = "low_ranked"


#: Attribution order. Frozen, so the slice a span is credited to does not depend on which
#: partition query happened to return it first.
SLICE_PRIORITY: Final = (
    PartitionSlice.SINGLE_HOLDER,
    PartitionSlice.UNWRITTEN_RULE,
    PartitionSlice.UNQUERIED_SOURCE_TYPE,
    PartitionSlice.LOW_RANKED,
)


class PartitionError(ValueError):
    """The partition overlaps the main analysis's evidence, so it is not a partition."""


@dataclass(frozen=True)
class PartitionSpan:
    """One span the blind-spot hunter may read and cite, and the slice that produced it."""

    evidence_id: str
    text: str
    source_type: str
    authority_class: str
    slice_: PartitionSlice
    assumption_keys: tuple[str, ...] = ()
    source_version_id: str = ""


@dataclass(frozen=True)
class EvidencePartition:
    """The whole partition, plus what building it could not cover.

    ``uncovered`` is reported rather than dropped: a slice with no spans is a statement about
    the corpus, and a reader comparing two analyses needs it to tell a quiet partition from
    an exhausted one.
    """

    spans: tuple[PartitionSpan, ...] = ()
    #: Ids the main consequence-hunter was offered. Nothing here may appear in ``spans``.
    main_offered_ids: frozenset[str] = field(default_factory=frozenset)
    #: Spans a partition query returned that the main hunter had already been shown.
    excluded_as_already_seen: int = 0
    #: Spans dropped by the span budget, after attribution.
    excluded_by_budget: int = 0
    single_holder_keys: tuple[str, ...] = ()
    unwritten_rule_keys: tuple[str, ...] = ()
    version: str = PARTITION_VERSION

    def __post_init__(self) -> None:
        overlap = {span.evidence_id for span in self.spans} & self.main_offered_ids
        if overlap:
            raise PartitionError(
                f"{len(overlap)} span(s) in the blind-spot partition were already offered to "
                f"the consequence-hunter ({sorted(overlap)[:3]}). A partition that overlaps "
                "the main analysis produces a second opinion, not an independent one."
            )

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(span.evidence_id for span in self.spans)

    @property
    def is_empty(self) -> bool:
        return not self.spans

    def by_slice(self, slice_: PartitionSlice) -> tuple[PartitionSpan, ...]:
        return tuple(span for span in self.spans if span.slice_ is slice_)

    def slice_of(self, evidence_id: str) -> PartitionSlice | None:
        for span in self.spans:
            if span.evidence_id == evidence_id:
                return span.slice_
        return None

    @property
    def covered(self) -> tuple[PartitionSlice, ...]:
        present = {span.slice_ for span in self.spans}
        return tuple(s for s in SLICE_PRIORITY if s in present)

    @property
    def uncovered(self) -> tuple[PartitionSlice, ...]:
        present = {span.slice_ for span in self.spans}
        return tuple(s for s in SLICE_PRIORITY if s not in present)


def undocumented_keys(report: KnowledgeDecayReport) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The single-holder and unwritten-rule keys, read off the existing decay report.

    Read rather than recomputed: ``app/assumptions/decay.py`` already derives both categories,
    and a second implementation would be a second definition of "undocumented".
    """
    single = sorted({e.key for e in report.of_category(DecayCategory.SINGLE_HOLDER)})
    unwritten = sorted({e.key for e in report.of_category(DecayCategory.UNWRITTEN_RULE)})
    return tuple(single), tuple(unwritten)


def partition_from(
    outcomes: Sequence[RetrievalOutcome],
    *,
    main_offered_ids: Iterable[str],
    main_source_types: Iterable[SourceType | str],
    single_holder_keys: Iterable[str] = (),
    unwritten_rule_keys: Iterable[str] = (),
    limit: int = MAX_PARTITION_SPANS,
) -> EvidencePartition:
    """Build the partition from the executed partition queries.

    Every span comes from a :class:`RetrievalOutcome`, so tenant and permission scope were
    applied by the retriever. This function cannot widen what may be read, because it never
    touches a store.
    """
    offered = frozenset(main_offered_ids)
    main_types = {t.value if isinstance(t, SourceType) else t for t in main_source_types}
    single = frozenset(single_holder_keys)
    unwritten = frozenset(unwritten_rule_keys)

    seen: set[str] = set()
    already = 0
    attributed: list[PartitionSpan] = []

    for outcome in outcomes:
        for result in outcome.results:
            evidence_id = result.candidate.evidence_id
            if evidence_id in offered:
                already += 1
                continue
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            keys = tuple(result.candidate.assumption_keys)
            attributed.append(
                PartitionSpan(
                    evidence_id=evidence_id,
                    text=result.candidate.text,
                    source_type=result.candidate.source_type.value,
                    authority_class=result.candidate.authority_class.value,
                    slice_=_attribute(
                        keys=keys,
                        source_type=result.candidate.source_type.value,
                        single=single,
                        unwritten=unwritten,
                        main_types=main_types,
                    ),
                    assumption_keys=keys,
                    source_version_id=result.candidate.source_version_id,
                )
            )

    # Sorted by the frozen slice order, then by id, so two builds over the same outcomes
    # produce the same prompt and the budget drops the same spans.
    ordered = sorted(attributed, key=lambda s: (SLICE_PRIORITY.index(s.slice_), s.evidence_id))
    return EvidencePartition(
        spans=tuple(ordered[:limit]),
        main_offered_ids=offered,
        excluded_as_already_seen=already,
        excluded_by_budget=max(0, len(ordered) - limit),
        single_holder_keys=tuple(sorted(single)),
        unwritten_rule_keys=tuple(sorted(unwritten)),
    )


def _attribute(
    *,
    keys: tuple[str, ...],
    source_type: str,
    single: frozenset[str],
    unwritten: frozenset[str],
    main_types: set[str],
) -> PartitionSlice:
    """Which slice this span belongs to, by the frozen priority."""
    if any(key in single for key in keys):
        return PartitionSlice.SINGLE_HOLDER
    if any(key in unwritten for key in keys):
        return PartitionSlice.UNWRITTEN_RULE
    if source_type not in main_types:
        return PartitionSlice.UNQUERIED_SOURCE_TYPE
    return PartitionSlice.LOW_RANKED
