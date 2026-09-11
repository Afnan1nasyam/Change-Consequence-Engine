"""Persist what a reviewer saw. A snapshot, not the whole analysis, and never a second truth.

Two finished features were inert without this. `RecordDispositionUseCase` had to be *handed*
the consequences a reviewer looked at, because there was nowhere to look them up - so a caller
passing the wrong list would have had its verdict accepted. And persistent suppression was
recorded, scoped and expiry-tested with **nothing reading it**, so the next analysis
re-reported a finding a reviewer had already rejected: precisely the failure ``IDEA.md`` 21.2
says makes reviewers stop reading.

**A snapshot, deliberately, not a serialized ``AnalysisResult``.** The result holds live
working objects - the whole coupling index, every retrieval candidate, every stage's internal
outcome - and storing them would make the store a second copy of the pipeline's intermediate
state, which would then be able to disagree with it. :class:`StoredAnalysis` keeps exactly what
a review needs: the change, the provenance, the stage statuses, the arbitrated findings with
their decisions and scores, the graph, the evidence spans a citation resolves to, and the
fingerprint.

**The fingerprint is what makes the snapshot trustworthy.** It is stored verbatim, so a
reviewer's verdict can be checked against the analysis that produced it: if a re-run of the
same store and configuration yields a different fingerprint, the two are different analyses and
a verdict from one does not apply to the other. Nothing here recomputes it - recomputing would
let the stored value drift from the result it came from.

**Tenant is a parameter, not a filter applied afterwards.** Every read takes ``tenant_id`` in
its signature, matching the ingestion ports, so a caller cannot forget to scope a lookup - the
type checker will not let them. The stored record carries its own tenant and
:meth:`InMemoryAnalysisStore.get` refuses a mismatch rather than returning ``None``: a
cross-tenant read is an isolation defect, and reporting it as "not found" would hide it.

The tenant is read off the **retrieval query**, because that is the permission scope the
analysis actually ran under - and it is cross-checked against the Class A findings, which carry
their own. A snapshot whose retrieval and findings disagreed about the tenant would be an
analysis assembled across a boundary.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import ArbitrationDecision, Consequence, ConsequenceScore
from app.domain.entities.graph import ConsequenceGraph
from app.domain.entities.source import EvidenceItem, HistoricalEvent
from app.domain.value_objects.provenance import RunProvenance

if TYPE_CHECKING:  # the orchestrator imports this module to persist its own result, so the
    # dependency has to run one way only. Nothing here touches either name at runtime.
    from app.application.orchestration.analysis import AnalysisResult, StageOutcome

#: Bumped when the snapshot's shape changes, because a stored analysis read back under a
#: different shape is not the analysis the reviewer saw.
ANALYSIS_SNAPSHOT_VERSION: Final = "analysis-snapshot-2"


class AnalysisStoreError(ValueError):
    """A stored analysis cannot be written or read as asked."""


class UnknownStoredAnalysis(AnalysisStoreError):
    """No analysis with this id exists for this tenant."""


class TenantMismatch(AnalysisStoreError):
    """The analysis exists, under a different tenant.

    Raised rather than reported as "not found": a cross-tenant read is an isolation defect,
    and a ``None`` would let it look like an ordinary miss (``docs/SECURITY.md`` 8, 9).
    """


@dataclass(frozen=True)
class ReviewedFinding:
    """One finding exactly as the reviewer saw it: the claim, the disposition, the number.

    Bundled per finding rather than kept as three parallel tuples, because "what did the
    reviewer see for this consequence?" is the question a disposition has to be validated
    against, and answering it by zipping three lists is how they drift out of step.
    """

    consequence: Consequence
    decision: ArbitrationDecision | None = None
    score: ConsequenceScore | None = None
    #: The source version the finding lands on, where the pipeline knows one.
    #:
    #: Class A only: a derivation names its affected version, so a suppression against it can
    #: expire when the document is revised. A Class B hypothesis has no single source version,
    #: so a suppression on one cannot expire that way - recorded as ``None`` rather than
    #: guessed, and reported as such by the suppression read.
    artifact_version_id: str | None = None
    #: Whether the judge foregrounded it. What the reviewer was shown *first*.
    in_default_view: bool = False

    @property
    def consequence_id(self) -> str:
        return self.consequence.consequence_id

    @property
    def confidence(self) -> float | None:
        return self.score.confidence if self.score else None


@dataclass(frozen=True)
class StoredAnalysis:
    """One analysis as a reviewer can retrieve it later.

    Everything needed to reconstruct the review surface and nothing needed only to re-run the
    pipeline. A caller wanting to re-run has the store the analysis was built from; a caller
    wanting to know what a verdict was about has this.
    """

    analysis_id: str
    tenant_id: str
    run_id: str
    change: ChangeSpec
    run_provenance: RunProvenance | None
    stages: tuple[StageOutcome, ...]
    findings: tuple[ReviewedFinding, ...]
    graph: ConsequenceGraph | None = None
    #: The spans a citation on any stored finding resolves to. Without them a reviewer can
    #: read a claim and not the evidence it rests on, which is not a review.
    evidence: tuple[EvidenceItem, ...] = ()
    #: The prior cases a stored ``historical_analogue_id`` refers to. Carried so the anchor
    #: behind a non-speculative Class B finding is still readable after a restart - the id is
    #: re-derivable from spans, but a reviewer should not have to re-derive it to audit one.
    historical_events: tuple[HistoricalEvent, ...] = ()
    evidence_gap_ids: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    #: Stored verbatim from the result. Never recomputed here - see the module docstring.
    fingerprint: dict[str, Any] = field(default_factory=dict)
    stored_at: dt.datetime | None = None
    pipeline_version: str = ""
    snapshot_version: str = ANALYSIS_SNAPSHOT_VERSION

    # -- the review surface ----------------------------------------------

    @property
    def consequences(self) -> tuple[Consequence, ...]:
        return tuple(f.consequence for f in self.findings)

    @property
    def consequence_ids(self) -> frozenset[str]:
        """The set a disposition is validated against."""
        return frozenset(f.consequence_id for f in self.findings)

    @property
    def decisions(self) -> tuple[ArbitrationDecision, ...]:
        return tuple(f.decision for f in self.findings if f.decision is not None)

    @property
    def scores(self) -> tuple[ConsequenceScore, ...]:
        return tuple(f.score for f in self.findings if f.score is not None)

    @property
    def foregrounded(self) -> tuple[str, ...]:
        """The consequence ids the judge put in the default view."""
        return tuple(f.consequence_id for f in self.findings if f.in_default_view)

    @property
    def evidence_by_id(self) -> dict[str, EvidenceItem]:
        return {item.evidence_id: item for item in self.evidence}

    def finding(self, consequence_id: str) -> ReviewedFinding | None:
        for candidate in self.findings:
            if candidate.consequence_id == consequence_id:
                return candidate
        return None

    def analogue(self, event_id: str) -> HistoricalEvent | None:
        """The prior case a finding cites. ``None`` when the id is not in this snapshot."""
        for event in self.historical_events:
            if event.event_id == event_id:
                return event
        return None

    def artifact_version_of(self, consequence_id: str) -> str | None:
        """The source version a finding lands on, for suppression expiry."""
        found = self.finding(consequence_id)
        return found.artifact_version_id if found else None


@runtime_checkable
class AnalysisStore(Protocol):
    """Stored analyses, addressable by ``(tenant_id, analysis_id)``.

    Write-once per analysis id. There is deliberately no mutator for a stored finding: a
    reviewer's verdict is recorded *beside* an analysis (``FindingDisposition``), never into
    it, so an analysis a verdict refers to cannot change under the verdict. An editable
    snapshot would make "what did the reviewer see?" unanswerable, which is the one question
    the snapshot exists to answer.

    ``tenant_id`` is a parameter on every read rather than a filter applied to the result,
    matching the ingestion ports: a caller cannot forget to scope a lookup because the
    signature will not let them.
    """

    def put(self, analysis: StoredAnalysis) -> None: ...

    def get(self, tenant_id: str, analysis_id: str) -> StoredAnalysis | None: ...

    def require(self, tenant_id: str, analysis_id: str) -> StoredAnalysis:
        """Like :meth:`get`, raising :class:`UnknownStoredAnalysis` on a miss.

        Both, because the two callers differ: a UI listing analyses wants ``None``, and a path
        about to accept a reviewer's verdict wants a refusal it cannot ignore.
        """
        ...

    def for_tenant(self, tenant_id: str) -> tuple[str, ...]:
        """The analysis ids this tenant may read, in a stable order."""
        ...


def snapshot(
    result: AnalysisResult,
    *,
    stored_at: dt.datetime,
    tenant_id: str | None = None,
) -> StoredAnalysis:
    """Project a finished analysis into the record a reviewer retrieves.

    ``tenant_id`` is derived from the retrieval query when the caller does not supply one -
    that query carried the permission scope the analysis ran under, so it is the boundary the
    stored record belongs to. Supplying one that disagrees with it raises: a snapshot filed
    under the wrong tenant would be readable by the wrong reviewer.

    Decisions and scores are joined onto their consequence by ``consequence_id``, which the
    judge guarantees is shared across all three records. The affected source version comes
    from the Class A findings, matched through the relation the consequence reports.
    """
    resolved = _tenant_of(result, supplied=tenant_id)
    decisions = {d.consequence_id: d for d in result.decisions}
    scores = {s.consequence_id: s for s in result.scores}
    foregrounded = (
        {a.consequence_id for a in result.judgment.outcome.default_view}
        if result.judgment
        else set()
    )
    versions = _artifact_versions(result)

    findings = tuple(
        ReviewedFinding(
            consequence=consequence,
            decision=decisions.get(consequence.consequence_id),
            score=scores.get(consequence.consequence_id),
            artifact_version_id=_version_for(consequence, versions),
            in_default_view=consequence.consequence_id in foregrounded,
        )
        for consequence in sorted(result.consequences, key=lambda c: c.consequence_id)
    )

    return StoredAnalysis(
        analysis_id=result.analysis_id,
        tenant_id=resolved,
        run_id=result.run_id,
        change=result.change,
        run_provenance=result.run_provenance,
        stages=result.stages,
        findings=findings,
        graph=result.graph.graph if result.graph else None,
        evidence=_evidence(result),
        historical_events=result.historical_events,
        evidence_gap_ids=result.evidence_gap_ids,
        open_questions=result.open_questions,
        fingerprint=result.fingerprint(),
        stored_at=stored_at,
        pipeline_version=result.pipeline_version,
    )


def _tenant_of(result: AnalysisResult, *, supplied: str | None) -> str:
    """The tenant this analysis belongs to, and a cross-check that everything agrees.

    The retrieval query is authoritative: it is where the permission scope entered, so it is
    the boundary every downstream stage was confined to. The Class A findings carry their own
    tenant and must match - a snapshot whose findings and retrieval disagreed would be an
    analysis assembled across a boundary, and filing it under either would be wrong.
    """
    from_query = result.retrieval.outcome.query.tenant_id
    tenants = {from_query} | {f.tenant_id for f in result.class_a}
    if len(tenants) != 1:
        raise TenantMismatch(
            f"analysis {result.analysis_id} spans {len(tenants)} tenants ({sorted(tenants)}); "
            "retrieval and the Class A findings must agree before it can be stored"
        )
    if supplied is not None and supplied != from_query:
        raise TenantMismatch(
            f"analysis {result.analysis_id} ran for tenant {from_query!r} but was offered for "
            f"{supplied!r}. A snapshot filed under the wrong tenant is readable by the wrong "
            "reviewer."
        )
    return from_query


def _artifact_versions(result: AnalysisResult) -> Mapping[str, str]:
    """Relation id -> the source version its Class A finding affects.

    The join that makes suppression expiry work at all: a ``Consequence`` records
    ``affected_entity`` as a title, and a title does not change when a document is revised.
    The finding behind it records ``affected_source_version_id``, which is content-derived and
    does.
    """
    return {finding.relation_id: finding.affected_source_version_id for finding in result.class_a}


def _version_for(consequence: Consequence, versions: Mapping[str, str]) -> str | None:
    """The affected version for one consequence, or ``None`` for a Class B hypothesis.

    ``None`` rather than a guess. A hypothesis spans several artifacts by construction, so
    there is no single version whose revision should lift a suppression on it - and inventing
    one would make a suppression expire for a reason nobody chose.
    """
    for relation_id in consequence.derived_relation_ids:
        found = versions.get(relation_id)
        if found is not None:
            return found
    return None


def _evidence(result: AnalysisResult) -> tuple[EvidenceItem, ...]:
    """Every span a stored citation could resolve to.

    Taken from retrieval *results*, which the retrievers already narrowed by tenant and by
    permission - so a stored analysis cannot contain a span its reviewer was not allowed to
    read. The blind-spot stage issues its own queries through the same retriever under the
    same permission scope, and its findings cite those spans, so its outcomes are included:
    a reviewer who can read a claim and not its evidence is not reviewing.

    Sorted and deduplicated for a stable snapshot.
    """
    outcomes = [
        result.retrieval.outcome,
        *(result.blind_spots.partition_outcomes if result.blind_spots else ()),
    ]
    spans = {
        candidate.evidence_id: candidate.evidence
        for outcome in outcomes
        for candidate in (r.candidate for r in outcome.results)
    }
    return tuple(spans[key] for key in sorted(spans))
