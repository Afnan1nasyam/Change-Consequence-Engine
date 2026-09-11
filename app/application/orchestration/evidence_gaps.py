"""Turn a search that returned nothing into an ``EvidenceGap``. Provenance from the trace.

An ``EvidenceGap`` is a claim **about the corpus**, not about the world (checkpoint 2, type
E). That distinction is what this module exists to protect, and it is protected by refusing
to build a gap out of anything except an executed search:

**The model cannot supply the provenance.** :func:`produce_gaps` takes model-authored text
in exactly one parameter - ``questions`` - and it reaches exactly one field of the resulting
gap, ``question``. Every other field is read off a
:class:`~app.application.orchestration.retrieval_service.FinalizedRetrieval` by
:meth:`SearchProvenance.from_retrieval`. There is no parameter through which a
model-authored value could reach a provenance field, which is stronger than validating that
none did, and a test asserts the signature rather than trusting this paragraph.

:class:`SearchProvenance` additionally validates that its trace is a **coherent account of a
search**: the exclusion buckets must sum to ``considered``, which is the invariant all three
retrievers guarantee. So a hand-assembled provenance cannot be a plausible-looking set of
numbers; it has to add up the way a real search adds up.

**"I don't know" is not a gap.** Checkpoint 2.2 defines ``negative_provenance`` as *the
executed QueryPlan that returned nothing*, so a gap needs both halves. An unanswered question
with no search behind it is a model's uncertainty, and recording it as a statement about the
corpus would be the system laundering a model's silence into evidence. :func:`produce_gaps`
therefore requires:

1. a search that **ran** - at least one strategy executed, over a corpus this requester
   could read; and
2. a search that **found nothing** - ``trace.returned == 0``.

A search that returned results produces no gap for that search, even if the hunter still had
questions: the negative provenance would be false, because the plan did not return nothing.

**Authorization is represented, not leaked.** If the requester's readable slice was empty -
every span withheld or belonging to another tenant - then *they* searched nothing they could
read, and no gap is produced: "the corpus cannot establish this" would be a claim about a
corpus this requester never saw. Where the slice was non-empty but spans were withheld, the
gap is produced and marked :attr:`SearchProvenance.authorization_limited`, so a reader knows
it describes one requester's view. What escapes about a withheld document is a **count** and
a scope hash, exactly as in the trace - no id, no title, no span.

**A gap is never an edge, a contradiction or a Class A finding.** It carries no evidence (the
entity's own validator refuses any), it attaches to the analysis, and it may *annotate* a
node through ``partition``. Inventing a relationship out of silence is the structural form of
treating absence of evidence as evidence of absence (checkpoint 2.4, E).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.application.orchestration.retrieval_service import FinalizedRetrieval
from app.consequences.hunter import HuntOutcome
from app.domain.entities.claims import EvidenceGap
from app.domain.interfaces.retrieval import RetrievalTrace
from app.domain.value_objects.enums import RetrievalStrategy
from app.domain.value_objects.identifiers import new_id

logger = logging.getLogger(__name__)

#: Bumped when the gap-production rule changes, because it changes what a gap asserts.
GAP_PRODUCER_VERSION: Final = "gap-producer-1"

#: Recorded when a query named no source-type filter. ``EvidenceGap.source_types_searched``
#: requires at least one entry, and an empty tuple would misreport an unfiltered search as a
#: search of nothing - the opposite of what happened.
ALL_SOURCE_TYPES: Final = "*"

#: What a gap says would resolve it, when the caller supplies nothing better. Deliberately
#: about the *corpus* rather than about the world: a gap is closed by a document existing,
#: not by the answer being true.
DEFAULT_RESOLUTION: Final = (
    "a source admitted into the corpus that binds or states this, retrievable under the "
    "same filters and readable by this requester"
)


class SearchProvenance(BaseModel):
    """The executed search, as the gap's negative provenance.

    Holds what :class:`~app.domain.entities.claims.EvidenceGap` cannot: the query text, the
    filters, the permission scope hash, the retrieval configuration and scorer versions, the
    run identifier and the trace itself. Those live here rather than on the entity because
    the entity's shape is frozen by the checkpoint, and an envelope is the established way to
    add what a contract cannot express without amending it.

    **Built by :meth:`from_retrieval`, from a real search.** That is the production path,
    and it is the enforcement of "the model cannot supply the provenance": nothing
    model-authored is in scope where these fields are assigned. The validator below then
    requires the trace to add up the way a search adds up, so a fabricated instance has to
    be arithmetically coherent rather than merely well-typed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The search text, verbatim. The requester's own words, so echoing it leaks nothing.
    query_text: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)

    #: Which strategy was selected, and which actually executed. Both, because a fused run
    #: selects ``hybrid`` and executes two - and an ``EvidenceGap`` claims the searches that
    #: ran, not the one that was asked for.
    strategy_selected: RetrievalStrategy
    #: Empty means **no strategy ran**, which is not a validation error here: it is the
    #: "no retrieval attempt" case, and :func:`produce_gaps` has to refuse it as a fact
    #: about the run rather than crash on it.
    strategies_executed: tuple[RetrievalStrategy, ...] = ()

    #: The filters as applied. ``source_types`` empty means every type was searched.
    source_types: tuple[str, ...] = ()
    scope: str | None = None
    as_of: dt.datetime | None = None
    top_k: int = Field(gt=0)

    permission_scope_hash: str = Field(min_length=1)
    retrieval_config_version: str = Field(min_length=1)
    scorer_version: str = Field(min_length=1)

    #: The bucket-by-bucket account of why the corpus shrank. The raw material a gap is
    #: made of, carried whole so a reader can see the search rather than a summary of it.
    trace: RetrievalTrace

    #: The run this search belongs to, supplied by the caller. A gap with no run identifier
    #: cannot be attributed to the analysis that produced it.
    run_id: str = Field(min_length=1)
    recorded_at: dt.datetime
    producer_version: str = GAP_PRODUCER_VERSION

    @model_validator(mode="after")
    def _the_trace_accounts_for_every_candidate(self) -> Self:
        """The buckets must sum to ``considered``, as every retriever guarantees.

        This is what makes the trace usable as *negative* provenance. A gap asserts that a
        search covered a corpus and returned nothing from it; if the account of that corpus
        does not add up, the assertion has no denominator and "the corpus cannot establish
        this" is a claim about an unknown number of documents.
        """
        accounted = self.trace.excluded + self.trace.matched
        if accounted != self.trace.considered:
            raise ValueError(
                f"the retrieval trace does not account for every candidate: "
                f"{self.trace.excluded} excluded + {self.trace.matched} matched = "
                f"{accounted}, but {self.trace.considered} were considered. A gap cites this "
                "trace as the search it performed, so an incomplete account is not "
                "negative provenance."
            )
        return self

    @classmethod
    def from_retrieval(
        cls,
        finalized: FinalizedRetrieval,
        *,
        run_id: str,
        recorded_at: dt.datetime | None = None,
    ) -> Self:
        """Read the provenance off a real search. The only way to construct one."""
        query = finalized.outcome.query
        return cls(
            query_text=query.text,
            tenant_id=query.tenant_id,
            strategy_selected=finalized.strategy,
            strategies_executed=finalized.strategies_executed,
            source_types=tuple(source_type.value for source_type in query.source_types),
            scope=query.scope.render() if query.scope is not None else None,
            as_of=query.as_of,
            top_k=query.top_k,
            permission_scope_hash=finalized.permission_scope_hash,
            retrieval_config_version=finalized.retrieval_config_version,
            scorer_version=finalized.scorer_version,
            trace=finalized.trace,
            run_id=run_id,
            recorded_at=recorded_at or dt.datetime.now(tz=dt.UTC),
        )

    # -- what the trace says about the search ----------------------------

    @property
    def readable_corpus_size(self) -> int:
        """Spans this requester could actually search.

        The corpus minus another tenant's spans and minus what authorization withheld. This
        is the number that decides whether a search *happened* for this requester: zero
        means they searched nothing they were allowed to read.
        """
        return max(
            0,
            self.trace.considered
            - self.trace.excluded_by_tenant
            - self.trace.excluded_by_permission,
        )

    @property
    def search_was_attempted(self) -> bool:
        """Whether a strategy ran over a corpus this requester could read.

        Both halves are required. A run with no ``strategies_executed`` performed no search
        at all. A tenant with no indexed spans, or a requester whose every span was
        withheld, ran a strategy over nothing they were allowed to look in. Neither
        establishes that the corpus lacks an answer, and recording either as a gap would
        misattribute a configuration or authorization state to the corpus.
        """
        return bool(self.strategies_executed) and self.readable_corpus_size > 0

    @property
    def found_evidence(self) -> bool:
        return self.trace.returned > 0

    @property
    def authorization_limited(self) -> bool:
        """Whether authorization removed anything from this requester's search.

        A gap carrying this describes **one requester's view of the corpus**, not the corpus.
        Surfaced as a boolean over a count that is already query-independent, so it says
        nothing about whether the withheld spans matched.
        """
        return self.trace.excluded_by_permission > 0

    @property
    def produces_gap(self) -> bool:
        """Both halves of ``negative_provenance``: a plan that ran, and returned nothing."""
        return self.search_was_attempted and not self.found_evidence

    @property
    def source_types_searched(self) -> tuple[str, ...]:
        """What the entity records. An unfiltered search searched every type."""
        return self.source_types or (ALL_SOURCE_TYPES,)


@dataclass(frozen=True)
class GapRecord:
    """One gap and the search that produced it.

    Paired rather than merged: ``EvidenceGap`` is the frozen domain claim and
    :class:`SearchProvenance` is the application-layer detail behind it. A consumer that
    only needs the claim reads ``.gap``; one auditing why the gap exists reads
    ``.provenance``.
    """

    gap: EvidenceGap
    provenance: SearchProvenance

    @property
    def gap_id(self) -> str:
        return self.gap.gap_id

    @property
    def question(self) -> str:
        return self.gap.question

    @property
    def authorization_limited(self) -> bool:
        return self.provenance.authorization_limited


@dataclass(frozen=True)
class GapProductionOutcome:
    """The gaps produced, and why the questions that produced none did not.

    Refusals are counted rather than silent: a stage that produced no gaps because nothing
    was searched and one that produced none because everything was answered are different
    facts about an analysis.
    """

    gaps: tuple[GapRecord, ...] = ()
    #: Questions refused because no readable corpus was searched.
    refused_no_search: tuple[str, ...] = ()
    #: Questions refused because the search returned evidence.
    refused_evidence_found: tuple[str, ...] = ()

    @property
    def considered(self) -> int:
        return len(self.gaps) + len(self.refused_no_search) + len(self.refused_evidence_found)


def produce_gaps(
    *,
    analysis_id: str,
    questions: tuple[str, ...],
    retrieval: FinalizedRetrieval,
    run_id: str,
    partition: str | None = None,
    would_be_resolved_by: str = DEFAULT_RESOLUTION,
    recorded_at: dt.datetime | None = None,
) -> GapProductionOutcome:
    """Convert unanswered questions into gaps, where the search justifies one.

    ``questions`` are the hunter's - the **only** model-authored input, and it contributes
    the question text and nothing else. Everything a gap asserts about the search comes from
    ``retrieval``.

    ``partition`` optionally names the node or entity the gap annotates. It never makes the
    gap a relationship: a gap annotates one node and connects none (checkpoint 2.4, E).
    """
    provenance = SearchProvenance.from_retrieval(retrieval, run_id=run_id, recorded_at=recorded_at)

    if not provenance.search_was_attempted:
        # No readable corpus, so nothing was established about the corpus. Reported at
        # warning level because it is almost always a configuration or authorization
        # problem rather than a knowledge gap.
        logger.warning(
            "no evidence gaps produced: the requester had no readable corpus to search",
            extra={
                "analysis_id": analysis_id,
                "run_id": run_id,
                "strategies": len(provenance.strategies_executed),
                "considered": provenance.trace.considered,
                "withheld": provenance.trace.excluded_by_permission,
            },
        )
        return GapProductionOutcome(refused_no_search=questions)

    if provenance.found_evidence:
        # The plan did not return nothing, so `negative_provenance` would be false. The
        # hunter may still have had questions; a gap is not how they are recorded.
        return GapProductionOutcome(refused_evidence_found=questions)

    if provenance.authorization_limited:
        logger.info(
            "evidence gaps describe one requester's view: authorization withheld spans",
            extra={
                "analysis_id": analysis_id,
                "run_id": run_id,
                "withheld": provenance.trace.excluded_by_permission,
            },
        )

    gaps = tuple(
        GapRecord(
            gap=EvidenceGap(
                gap_id=new_id("gap"),
                analysis_id=analysis_id,
                question=question,
                strategies_executed=tuple(
                    strategy.value for strategy in provenance.strategies_executed
                ),
                source_types_searched=provenance.source_types_searched,
                # The as-of the search actually ran under. Where the query named none, the
                # recording time is used: a gap has to state the moment its claim about the
                # corpus was true, and "no filter" is not a moment.
                as_of=provenance.as_of or provenance.recorded_at,
                would_be_resolved_by=would_be_resolved_by,
                partition=partition,
            ),
            provenance=provenance,
        )
        for question in questions
    )
    return GapProductionOutcome(gaps=gaps)


def gaps_from_hunt(
    hunt: HuntOutcome,
    *,
    analysis_id: str,
    retrieval: FinalizedRetrieval,
    run_id: str,
    partition: str | None = None,
    would_be_resolved_by: str = DEFAULT_RESOLUTION,
    recorded_at: dt.datetime | None = None,
) -> GapProductionOutcome:
    """The hunter's unanswered questions, as gaps where the search justifies one.

    A thin adapter, deliberately: it takes exactly one thing from the
    :class:`~app.consequences.hunter.HuntOutcome` - ``unanswered_questions`` - and everything
    else from ``retrieval``. Written as a separate seam rather than folded into
    :mod:`app.application.use_cases.hunt_consequences` so the hunter keeps no access to the
    trace, which is the reason it cannot author a gap's provenance.
    """
    if not hunt.succeeded:
        # A stage that failed asked nothing and answered nothing. Its silence is a fact
        # about the provider, and turning it into a claim about the corpus would be the
        # worst version of this component's failure mode.
        logger.warning(
            "no evidence gaps produced: the consequence-hunter stage failed",
            extra={"analysis_id": analysis_id, "run_id": run_id, "failure": hunt.failure},
        )
        return GapProductionOutcome()

    return produce_gaps(
        analysis_id=analysis_id,
        questions=hunt.unanswered_questions,
        retrieval=retrieval,
        run_id=run_id,
        partition=partition,
        would_be_resolved_by=would_be_resolved_by,
        recorded_at=recorded_at,
    )
