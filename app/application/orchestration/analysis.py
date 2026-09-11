"""One analysis, end to end. Composition only - no stage's logic lives here.

Every stage in this system was built and verified against its own inputs. That made each one
trustworthy and the *pipeline* unproven: the interfaces between stages had only ever been
exercised by test fixtures. This module is the composition, and it deliberately contains no
business logic at all - it decides order, dependency and what to report, and nothing else.

Stage order, and why it is this order
-------------------------------------

===== ======================= ============ ==================================================
order  stage                   kind         depends on
===== ======================= ============ ==================================================
1      coupling                determin.    the store
2      class_a                 determin.    the coupling index + the change's deltas
3      retrieval               determin.    the store + the requester's permission scope
4      specialists             agentic      one source-type slice per domain analyst
5      class_b                 agentic      retrieval + the slices + the specialists' output
6      evidence_gaps           determin.    the Class B hunt + the retrieval **trace**
7      contradictions          agentic      the collision matrix + the bindings
8      blind_spots             agentic      a partition disjoint from the Class B hunt's slice
9      challenge               agentic      Class A + **both** Class B hypothesis sets
10     evidence_arbitration    agentic      the candidate set + the measured confidence inputs
11     judgment                agentic      everything above
12     graph                   determin.    bindings + relations + arbitrated consequences
13     report                  determin.    the finalized findings + reviewer verdicts
===== ======================= ============ ==================================================

**The specialists feed the hunter; they are not a second pipeline.** Seven domain analysts
each read their own source-type slice and return evidence-backed *observations* - no class, no
status, no severity, no number. Those observations and the slices behind them go into the
existing Class B stage, whose agent exists to combine "independent evidence from multiple
domains". So a specialist's work reaches challenge, arbitration and judgment through the
hypotheses it informed, and there is exactly one path from evidence to a consequence.

**Discovery finishes before review begins.** Both Class B producers - the consequence-hunter
and the blind-spot hunter - run before the challenger, so every hypothesis faces the same
falsification obligation. An earlier ordering ran the challenger first, which left blind-spot
findings reaching the judge unchallenged: the newest and least corroborated findings were the
only ones nobody had tried to refute.

Two consequences of that order, both wiring rather than logic:

* the challenger is handed the **union** of the two hunts' retained hypotheses;
* the evidence map it sees includes the blind-spot partition's spans. Without them a
  blind-spot candidate would be offered nothing citable and could only ever come back
  ``no_refutation_found`` for want of anything to cite - a refusal that would read like a
  survived challenge.

Class A comes **second**, before retrieval, and that ordering is the product's central claim
made operational: Class A recall does not depend on retrieval, so no retrieval failure can
damage it (``CLAUDE.md``, "any design in which recall depends on how much corpus fits in a
context window is the wrong design"). Retrieval feeds the *agentic* stages only.

Required and optional, and why a failure is never silent
--------------------------------------------------------

The **deterministic** stages are required. They have no provider, so a failure is a defect
rather than an outage, and :class:`AnalysisStageError` aborts the run naming the stage.
Continuing past one would produce an analysis whose foundation silently did not run.

The **agentic** stages are optional and already report their own failures. The orchestrator
records a :class:`StageOutcome` per stage with a *status*, so ``EMPTY`` and ``FAILED`` are
different values rather than both being "no results" - which is the whole of requirement 5.
A stage that could not run because its input came from a failed stage is ``SKIPPED``, a third
distinct thing.

One dependency is hard: **evidence gaps require the hunt**. A failed hunt asked nothing, and a
gap needs an executed search *and* an unanswered question - so on hunt failure the gap stage is
skipped, never reported as "no gaps". Every other downstream stage degrades instead of
skipping: one analyst's outage costs one domain rather than the fan-out, the hunter runs
without domain observations if the whole fan-out failed, the blind-spot hunter works a
*larger* partition if the hunt failed, the challenger reviews whichever hypothesis sets
survived - Class A alone if both hunts failed - the arbiter positions what it was given, the
judge arbitrates from the raw findings if the challenger failed, and the graph still projects
the bindings if the judge failed. A review stage must not be skipped because a discovery stage
failed.

``report`` is required and deterministic: every section is a projection of a finalized record,
so composition failing is a defect. The ``report-composer`` agent is optional and its failure
is recorded on the document rather than the stage - the report still exists, with the
deterministic surface and the deterministic executive summary.

**Reviewer feedback** enters in exactly one place, the report stage. Standing verdicts are read
through the analysis snapshot, because the artifact version a suppression expires on is a
property of the stored finding. ``wrong`` and ``not_material`` take a finding off the default
surface and leave it in ``consequence_ids``; ``already_known`` suppresses nothing and is
reported as a rate. The finished analysis is then persisted, which is what gives the *next*
run something to read.

What "quiet on a quiet change" means here
-----------------------------------------

The coupling index describes the **corpus**: standing divergences exist whether or not anybody
proposed a change, and reporting them as consequences *of* a change would make every analysis
look productive. So the change-relevant Class A set is selected, not re-derived:

* findings the change's deltas **propagate to**, via the existing ``apply_delta``; plus
* standing findings on a key the change touches - two documents already disagreeing about
  ``retention.window`` is something a reader changing it must know.

A change with no assumption deltas therefore selects nothing, and the standing corpus state
stays where it belongs: the knowledge-decay report, which ``ResolveCouplingUseCase`` already
produces.

Determinism
-----------

``analysis_id``, ``run_id`` and ``now`` are parameters. Given the same store, the same
configuration and the same provider responses, two runs agree on every status, count, score
and graph shape - :meth:`AnalysisResult.fingerprint` is that claim in a comparable form.
Minted record ids (hypothesis, gap, consequence, challenge) differ by design: they are fresh
per run so a re-run cannot collide with a stored analysis, which is why the fingerprint
excludes them rather than pretending they are stable.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Final

from app.application.orchestration.analysis_store import AnalysisStore, snapshot
from app.application.orchestration.evidence_gaps import GapProductionOutcome, gaps_from_hunt
from app.application.orchestration.retrieval_service import (
    FinalizedRetrieval,
    RetrievalRequest,
    RetrievalService,
)
from app.application.use_cases.arbitrate_evidence import (
    ArbitrateEvidenceUseCase,
    ArbitrationReport,
)
from app.application.use_cases.challenge_findings import (
    ChallengeFindingsUseCase,
    ChallengeReport,
)
from app.application.use_cases.compose_report import (
    SUPPRESSING,
    ComposeReportUseCase,
    ReportDocument,
    ReviewerSignals,
)
from app.application.use_cases.consult_specialists import (
    ConsultSpecialistsUseCase,
    SpecialistReport,
)
from app.application.use_cases.hunt_blind_spots import (
    BlindSpotReport,
    HuntBlindSpotsUseCase,
)
from app.application.use_cases.hunt_consequences import (
    ClassBOutcome,
    HuntConsequencesUseCase,
    evidence_for_hunting,
)
from app.application.use_cases.hunt_contradictions import (
    ContradictionReport,
    HuntContradictionsUseCase,
)
from app.application.use_cases.judge_consequences import (
    JudgeConsequencesUseCase,
    JudgmentReport,
)
from app.application.use_cases.record_disposition import (
    RecordDispositionUseCase,
    StandingVerdict,
)
from app.application.use_cases.resolve_coupling import CouplingAnalysis, ResolveCouplingUseCase
from app.assumptions.class_a import derive_class_a
from app.blindspots.hunter import BLIND_SPOT_AGENT
from app.blindspots.partition import PartitionSlice
from app.challenge.candidates import ChallengeTarget
from app.challenge.challenger import ChallengedCandidate
from app.consequences.prompting import AdmissibleAnalogue, SpecialistObservation
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.findings import (
    ArbitrationDecision,
    Consequence,
    ConsequenceScore,
    SpecialistFinding,
)
from app.domain.entities.source import EvidenceItem, HistoricalEvent
from app.domain.interfaces.retrieval import RetrievalOutcome, RetrievalQuery
from app.domain.value_objects.enums import Disposition, Status
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import RunProvenance
from app.domain.value_objects.rules import CONFIDENCE_FN_VERSION, COUPLING_INDEX_VERSION
from app.graph.projection import ProjectionResult, project
from app.judgment.confidence_inputs import AssembledInputs

logger = logging.getLogger(__name__)

#: Bumped when the composition changes - the order, a dependency, or which stage is
#: required - because any of those changes what an analysis contains without any stage
#: changing. Pinned onto the result.
#: ``-2`` moved ``blind_spots`` ahead of ``challenge``, so both Class B hunts are reviewed.
#: ``-3`` added the specialist fan-out ahead of ``class_b``.
ANALYSIS_PIPELINE_VERSION: Final = "analysis-pipeline-3"

#: The schema version recorded on every finding's ``RunProvenance``.
ANALYSIS_SCHEMA_VERSION: Final = "domain-1"

#: Statuses a reader is asked to act on. A finding below this is reachable, not foregrounded,
#: and the negative control is measured against this set rather than against raw counts.
MATERIAL_STATUSES: Final = frozenset({Status.SUPPORTED, Status.LIKELY, Status.DISPUTED})


class AnalysisStage(StrEnum):
    """The thirteen stages, in execution order."""

    COUPLING = "coupling"
    CLASS_A = "class_a"
    RETRIEVAL = "retrieval"
    SPECIALISTS = "specialists"
    CLASS_B = "class_b"
    EVIDENCE_GAPS = "evidence_gaps"
    CONTRADICTIONS = "contradictions"
    BLIND_SPOTS = "blind_spots"
    CHALLENGE = "challenge"
    EVIDENCE_ARBITRATION = "evidence_arbitration"
    JUDGMENT = "judgment"
    GRAPH = "graph"
    REPORT = "report"


#: Stages whose failure aborts the analysis. Exactly the deterministic ones: they have no
#: provider, so a failure is a defect and continuing would build an analysis on a foundation
#: that silently did not run.
#:
#: ``REPORT`` is here because composition is deterministic - every section is a projection of
#: a finalized record. The ``report-composer`` agent is optional and its failure is recorded
#: on the document, not on the stage: the report still exists, with the deterministic surface.
REQUIRED_STAGES: Final = frozenset(
    {
        AnalysisStage.COUPLING,
        AnalysisStage.CLASS_A,
        AnalysisStage.RETRIEVAL,
        AnalysisStage.GRAPH,
        AnalysisStage.REPORT,
    }
)


class StageStatus(StrEnum):
    """What happened to one stage.

    ``EMPTY`` and ``FAILED`` are separate values, which is the whole of requirement 5: a
    stage that ran and found nothing and a stage that could not run are different facts about
    an analysis, and a reader must be able to tell them apart without reading a log.
    """

    #: Ran and produced something.
    COMPLETED = "completed"
    #: Ran and produced nothing. A legitimate result.
    EMPTY = "empty"
    #: Ran and reported a failure. Not the same as empty.
    FAILED = "failed"
    #: Did not run, because a stage it depends on failed.
    SKIPPED = "skipped"
    #: Did not run, because no runtime for it was supplied.
    NOT_CONFIGURED = "not_configured"


class AnalysisStageError(RuntimeError):
    """A required deterministic stage failed. The analysis is abandoned, loudly.

    Carries the stage so a caller can report *which* foundation gave way rather than that
    something did.
    """

    def __init__(self, stage: AnalysisStage, cause: Exception) -> None:
        super().__init__(f"required stage {stage.value!r} failed: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


@dataclass(frozen=True)
class StageOutcome:
    """One stage's status, and how much it produced."""

    stage: AnalysisStage
    status: StageStatus
    produced: int = 0
    detail: str = ""

    @property
    def ran(self) -> bool:
        return self.status in (StageStatus.COMPLETED, StageStatus.EMPTY, StageStatus.FAILED)


@dataclass(frozen=True)
class AnalysisResult:
    """One analysis, whole. Every stage's output, and every stage's status.

    Nothing here is summarized away: a caller that wants the headline reads the properties,
    and a caller auditing a number reads the stage output it came from.
    """

    analysis_id: str
    run_id: str
    change: ChangeSpec
    stages: tuple[StageOutcome, ...]

    coupling: CouplingAnalysis
    class_a: tuple[ClassAFinding, ...]
    retrieval: FinalizedRetrieval
    specialists: SpecialistReport | None = None
    class_b: ClassBOutcome | None = None
    gaps: GapProductionOutcome | None = None
    contradictions: ContradictionReport | None = None
    challenge: ChallengeReport | None = None
    blind_spots: BlindSpotReport | None = None
    arbitration: ArbitrationReport | None = None
    judgment: JudgmentReport | None = None
    graph: ProjectionResult | None = None
    report: ReportDocument | None = None

    run_provenance: RunProvenance | None = None
    pipeline_version: str = ANALYSIS_PIPELINE_VERSION

    # -- the headline ----------------------------------------------------

    @property
    def hypotheses(self) -> tuple[ConsequenceHypothesis, ...]:
        """Class B hypotheses from the main hunt only. The blind-spot set is reported apart."""
        return self.class_b.retained if self.class_b else ()

    @property
    def specialist_findings(self) -> tuple[SpecialistFinding, ...]:
        """The domain analysts' retained observations. Never candidates themselves."""
        return self.specialists.retained if self.specialists else ()

    @property
    def historical_events(self) -> tuple[HistoricalEvent, ...]:
        """Prior cases the historical analyst evidenced, ids derived from their spans.

        The anchors among them are what let a Class B finding exceed ``speculative``; the rest
        are precedents with no recorded outcome, kept because they are worth reading and
        structurally unable to raise anything.
        """
        return self.specialists.historical_events if self.specialists else ()

    @property
    def specialist_evidence_ids(self) -> frozenset[str]:
        """Ids the specialists' observations rest on. Used to trace their reach downstream."""
        return self.specialists.evidence_ids if self.specialists else frozenset()

    @property
    def blind_spot_hypotheses(self) -> tuple[ConsequenceHypothesis, ...]:
        """Kept separate from ``hypotheses`` so blind-spot yield is evaluable on its own."""
        return self.blind_spots.retained if self.blind_spots else ()

    @property
    def all_hypotheses(self) -> tuple[ConsequenceHypothesis, ...]:
        return (*self.hypotheses, *self.blind_spot_hypotheses)

    @property
    def blind_spot_consequences(self) -> tuple[Consequence, ...]:
        """Finalized consequences the blind-spot hunter produced, by its own attribution.

        Read off ``agent_sources``, which the judge derives from the hypothesis's own
        ``ModelProvenance``. So attribution survives arbitration without any stage having to
        carry a flag.
        """
        return tuple(c for c in self.consequences if BLIND_SPOT_AGENT in c.agent_sources)

    @property
    def blind_spot_slices(self) -> tuple[PartitionSlice, ...]:
        """Which partition slices this run could fill. Empty when the stage did not run."""
        if self.blind_spots is None or self.blind_spots.partition is None:
            return ()
        return self.blind_spots.partition.covered

    @property
    def evidence_positions(self) -> dict[str, str]:
        """The arbiter's position per consequence. A second axis beside status, never merged."""
        if self.arbitration is None:
            return {}
        return {c: p.value for c, p in self.arbitration.positions.items()}

    @property
    def consequences(self) -> tuple[Consequence, ...]:
        return self.judgment.consequences if self.judgment else ()

    @property
    def decisions(self) -> tuple[ArbitrationDecision, ...]:
        return self.judgment.decisions if self.judgment else ()

    @property
    def scores(self) -> tuple[ConsequenceScore, ...]:
        return self.judgment.scores if self.judgment else ()

    @property
    def confidence_inputs(self) -> dict[str, AssembledInputs]:
        return self.judgment.confidence_coverage if self.judgment else {}

    @property
    def material_consequences(self) -> tuple[Consequence, ...]:
        """Consequences a reader is asked to act on.

        The measure the negative control is judged against. Raw counts would let a pile of
        ``speculative`` findings look like productivity, which is the failure mode
        ``CLAUDE.md`` calls out: a system that always finds consequences has not been shown
        to find any.
        """
        return tuple(c for c in self.consequences if c.status in MATERIAL_STATUSES)

    @property
    def open_questions(self) -> tuple[str, ...]:
        """Every stage's unanswered questions, including the blind-spot silences."""
        collected = list(self.judgment.open_questions if self.judgment else ())
        for question in (
            *(self.blind_spots.open_questions if self.blind_spots else ()),
            *(self.specialists.open_questions if self.specialists else ()),
        ):
            if question not in collected:
                collected.append(question)
        return tuple(collected)

    @property
    def evidence_gap_ids(self) -> tuple[str, ...]:
        if self.gaps is None:
            return ()
        return tuple(sorted(record.gap_id for record in self.gaps.gaps))

    # -- stage accounting ------------------------------------------------

    def stage(self, stage: AnalysisStage) -> StageOutcome:
        for outcome in self.stages:
            if outcome.stage is stage:
                return outcome
        raise KeyError(f"stage {stage.value!r} is not in this analysis")

    @property
    def failed_stages(self) -> tuple[AnalysisStage, ...]:
        return tuple(s.stage for s in self.stages if s.status is StageStatus.FAILED)

    @property
    def empty_stages(self) -> tuple[AnalysisStage, ...]:
        return tuple(s.stage for s in self.stages if s.status is StageStatus.EMPTY)

    @property
    def skipped_stages(self) -> tuple[AnalysisStage, ...]:
        return tuple(s.stage for s in self.stages if s.status is StageStatus.SKIPPED)

    @property
    def succeeded(self) -> bool:
        """Whether every stage that ran completed without reporting a failure.

        Deliberately not "produced results": an analysis that correctly found nothing
        succeeded, and one whose judge timed out did not.
        """
        return not self.failed_stages

    @property
    def is_quiet(self) -> bool:
        """Whether this analysis reports approximately nothing material.

        The negative-control predicate. Named as a property rather than computed in a test so
        the definition lives with the pipeline and cannot drift between the two.
        """
        return not self.material_consequences and not self.class_a

    def fingerprint(self) -> dict[str, Any]:
        """Everything about this analysis that two identical runs must agree on.

        Excludes minted record ids and timestamps, and says so: those are fresh per run by
        design, so a re-run does not collide with a stored analysis. What must not vary is
        every *decision* - which stages ran, what they concluded, and what a reader is shown.
        """
        return {
            "pipeline_version": self.pipeline_version,
            "stages": {s.stage.value: [s.status.value, s.produced] for s in self.stages},
            "class_a_rules": sorted(f.rule_id for f in self.class_a),
            "class_a_keys": sorted({f.assumption_key for f in self.class_a}),
            "retrieved": sorted(self.retrieval.evidence_ids),
            "retrieval_strategy": self.retrieval.strategy.value,
            "specialist_agents": sorted({f.agent for f in self.specialist_findings}),
            "specialist_claims": sorted(f.claim for f in self.specialist_findings),
            "historical_anchors": sorted(
                e.event_id for e in self.historical_events if e.qualifies_as_anchor
            ),
            "hypothesis_claims": sorted(h.claim for h in self.hypotheses),
            "blind_spot_claims": sorted(h.claim for h in self.blind_spot_hypotheses),
            "blind_spot_slices": sorted(s.value for s in self.blind_spot_slices),
            "evidence_positions": sorted(self.evidence_positions.values()),
            "report_sections_empty": list(self.report.empty_sections) if self.report else [],
            "gap_questions": sorted(r.question for r in self.gaps.gaps) if self.gaps else [],
            "divergence_classes": sorted(
                d.divergence_class.value for d in self.contradictions.outcome.assessed
            )
            if self.contradictions
            else [],
            "challenge_outcomes": sorted(r.outcome.value for r in self.challenge.results)
            if self.challenge
            else [],
            "consequences": sorted(
                [c.status.value, c.severity.value, c.category, c.affected_entity]
                for c in self.consequences
            ),
            "confidence": sorted(round(s.confidence, 6) for s in self.scores),
            "graph_edges": sorted(
                [e.relationship.value, e.derivation.value] for e in self.graph.graph.edges
            )
            if self.graph
            else [],
            "graph_nodes": sorted([n.entity_type.value, n.label] for n in self.graph.graph.nodes)
            if self.graph
            else [],
        }


class AnalyzeChangeUseCase:
    """Compose the pipeline for one change. Holds the stages; implements none of them.

    The four agentic stages are optional. Passing ``None`` for one makes it
    ``NOT_CONFIGURED`` rather than an error, which is what lets a deterministic-only run
    happen - and a deterministic-only run is the thing that proves Class A recall does not
    depend on any model.
    """

    def __init__(
        self,
        *,
        coupling: ResolveCouplingUseCase,
        retrieval: RetrievalService,
        specialists: ConsultSpecialistsUseCase | None = None,
        class_b: HuntConsequencesUseCase | None = None,
        contradictions: HuntContradictionsUseCase | None = None,
        challenge: ChallengeFindingsUseCase | None = None,
        blind_spots: HuntBlindSpotsUseCase | None = None,
        arbitration: ArbitrateEvidenceUseCase | None = None,
        judgment: JudgeConsequencesUseCase | None = None,
        report: ComposeReportUseCase | None = None,
        analyses: AnalysisStore | None = None,
        reviews: RecordDispositionUseCase | None = None,
    ) -> None:
        self._coupling = coupling
        self._retrieval = retrieval
        self._specialists = specialists
        self._class_b = class_b
        self._contradictions = contradictions
        self._challenge = challenge
        self._blind_spots = blind_spots
        self._arbitration = arbitration
        self._judgment = judgment
        # Deterministic composition, so a caller that supplies nothing still gets a report.
        self._report = report or ComposeReportUseCase()
        self._analyses = analyses
        self._reviews = reviews

    async def execute(
        self,
        *,
        change: ChangeSpec,
        query: RetrievalQuery,
        tenant_id: str,
        registry_version: str,
        as_of: dt.datetime,
        analysis_id: str | None = None,
        run_id: str | None = None,
        now: dt.datetime | None = None,
        correlation_id: str | None = None,
    ) -> AnalysisResult:
        """Run every stage in order and return the whole analysis.

        ``query`` carries the requester's permission scope, and it is the only place
        authorization enters: every agentic stage downstream sees the spans that query
        returned and nothing else, because that is what this method passes them.
        """
        run = _Run(
            change=change,
            analysis_id=analysis_id or new_id("analysis"),
            run_id=run_id or new_id("analysis"),
            now=now or dt.datetime.now(tz=dt.UTC),
            tenant_id=tenant_id,
            registry_version=registry_version,
            as_of=as_of,
            correlation_id=correlation_id,
        )

        coupling = self._stage_coupling(run)
        class_a = self._stage_class_a(run, coupling)
        retrieval = await self._stage_retrieval(run, query)

        specialists = await self._stage_specialists(run, query, class_a)
        class_b = await self._stage_class_b(run, retrieval, class_a, specialists)
        gaps = self._stage_gaps(run, retrieval, class_b)
        contradictions = await self._stage_contradictions(run, coupling, retrieval)
        blind_spots = await self._stage_blind_spots(
            run, query, coupling, retrieval, class_a, class_b
        )

        # Assembled here rather than after the challenge, because the challenger needs the
        # blind-spot partition's spans: a candidate offered nothing citable can only come back
        # `no_refutation_found`, which reads like a survived challenge.
        outcomes = (
            retrieval.outcome,
            *(specialists.slice_outcomes if specialists else ()),
            *(blind_spots.partition_outcomes if blind_spots else ()),
        )
        spans = _stored_spans(retrieval, extra=outcomes[1:])

        challenge = await self._stage_challenge(run, spans, class_a, class_b, blind_spots)

        # One candidate set, computed once and handed to both the arbiter and the judge. Two
        # stages that disagreed about the candidate set would produce rulings addressed to
        # consequences the judge never arbitrated.
        candidates = _Candidates.of(
            class_a=class_a, class_b=class_b, blind_spots=blind_spots, challenge=challenge
        )

        arbitration = await self._stage_arbitration(
            run, candidates, gaps, contradictions, spans, outcomes
        )
        judgment = await self._stage_judgment(
            run, retrieval, candidates, gaps, contradictions, arbitration, spans, outcomes
        )
        graph = self._stage_graph(run, coupling, class_a, judgment, gaps)

        interim = AnalysisResult(
            analysis_id=run.analysis_id,
            run_id=run.run_id,
            change=change,
            stages=tuple(run.stages),
            coupling=coupling,
            class_a=class_a,
            retrieval=retrieval,
            specialists=specialists,
            class_b=class_b,
            gaps=gaps,
            contradictions=contradictions,
            challenge=challenge,
            blind_spots=blind_spots,
            arbitration=arbitration,
            judgment=judgment,
            graph=graph,
            run_provenance=run.provenance(retrieval),
        )

        report = await self._stage_report(run, interim, spans)
        result = replace(interim, stages=tuple(run.stages), report=report)
        self._persist(run, result)

        logger.info(
            "analysis_complete",
            extra={
                "analysis_id": run.analysis_id,
                "run_id": run.run_id,
                "change_id": change.change_id,
                "class_a": len(class_a),
                "specialists": len(result.specialist_findings),
                "historical_anchors": len(
                    [e for e in result.historical_events if e.qualifies_as_anchor]
                ),
                "class_b": len(result.hypotheses),
                "blind_spots": len(result.blind_spot_hypotheses),
                "consequences": len(result.consequences),
                "material": len(result.material_consequences),
                "foregrounded": len(report.high_priority),
                "suppressed_by_reviewers": report.reviewer.suppressed_count,
                "failed_stages": [s.value for s in result.failed_stages],
                "quiet": result.is_quiet,
            },
        )
        return result

    # -- deterministic stages: a failure is a defect ----------------------

    def _stage_coupling(self, run: _Run) -> CouplingAnalysis:
        with run.required(AnalysisStage.COUPLING) as record:
            analysis = self._coupling.execute(
                tenant_id=run.tenant_id,
                as_of=run.as_of,
                registry_version=run.registry_version,
            )
            record(len(analysis.index.bindings))
            return analysis

    def _stage_class_a(self, run: _Run, coupling: CouplingAnalysis) -> tuple[ClassAFinding, ...]:
        """Select the change-relevant Class A findings. Derives nothing new.

        Two sources, both existing: the findings the change's deltas propagate to, through
        ``apply_delta``, and the standing findings on a key the change touches. Selection
        rather than derivation is what keeps a quiet change quiet - the corpus's standing
        divergences belong to the knowledge-decay report, not to every analysis that runs.
        """
        with run.required(AnalysisStage.CLASS_A) as record:
            delta_keys = {delta.key for delta in run.change.assumption_deltas}
            bindings = coupling.index.bindings

            propagated: list[ClassAFinding] = []
            for delta in sorted(run.change.assumption_deltas, key=lambda d: d.delta_id):
                report = self._coupling.apply_delta(delta, coupling, as_of=run.as_of)
                propagated.extend(
                    derive_class_a(
                        report.relations,
                        bindings=bindings,
                        tenant_id=run.tenant_id,
                        registry_version=run.registry_version,
                        strengths=coupling.strengths,
                    )
                )

            standing = [f for f in coupling.findings if f.assumption_key in delta_keys]

            # Deduplicated on the finding id, which is derived from (relation, affected
            # version) - so a finding reached both by propagation and by key selection is one
            # finding, not two.
            merged = {f.finding_id: f for f in (*propagated, *standing)}
            findings = tuple(sorted(merged.values(), key=lambda f: f.finding_id))
            record(len(findings))
            return findings

    async def _stage_retrieval(self, run: _Run, query: RetrievalQuery) -> FinalizedRetrieval:
        with run.required(AnalysisStage.RETRIEVAL) as record:
            finalized = await self._retrieval.retrieve(RetrievalRequest(query=query))
            record(len(finalized.results))
            return finalized

    def _stage_graph(
        self,
        run: _Run,
        coupling: CouplingAnalysis,
        class_a: tuple[ClassAFinding, ...],
        judgment: JudgmentReport | None,
        gaps: GapProductionOutcome | None,
    ) -> ProjectionResult:
        """Project whatever the pipeline concluded. Runs even if the judge failed.

        With no arbitrated consequences the derived edges cannot project - they are
        consequence-bearing - and the projector reports them as skipped. The bindings still
        project, so the assumption-key joins remain visible: those are facts about what the
        documents say and do not depend on any model having run.
        """
        _ = class_a  # the graph reads relations and consequences, not findings
        with run.required(AnalysisStage.GRAPH) as record:
            result = project(
                analysis_id=run.analysis_id,
                change=run.change,
                bindings=coupling.index.bindings,
                relations=coupling.index.relations,
                consequences=judgment.consequences if judgment else (),
                gaps=tuple(record_.gap for record_ in gaps.gaps) if gaps else (),
                registry_version=run.registry_version,
                coupling_index_version=COUPLING_INDEX_VERSION,
                projected_at=run.now,
            )
            record(len(result.graph.edges))
            return result

    # -- agentic stages: a failure is recorded, never silent --------------

    async def _stage_specialists(
        self,
        run: _Run,
        query: RetrievalQuery,
        class_a: tuple[ClassAFinding, ...],
    ) -> SpecialistReport | None:
        """Fan out to the domain analysts that have evidence to read.

        Selection is deterministic and evidence-based: a specialist with an empty slice is not
        invoked, which is reported with its reason rather than looking like a stage that found
        nothing. ``query`` is the caller's, so every slice is a narrowing of what the requester
        was authorized to see.
        """
        if self._specialists is None:
            run.record(AnalysisStage.SPECIALISTS, StageStatus.NOT_CONFIGURED)
            return None
        report = await self._specialists.execute(
            change=run.change,
            query=query,
            class_a=class_a,
            as_of=run.as_of,
            correlation_id=run.correlation_id,
        )
        run.record_agentic(
            AnalysisStage.SPECIALISTS,
            succeeded=report.succeeded,
            produced=report.produced,
            # One analyst's failure is named, and the stage still carries the others'
            # findings: `record_agentic` marks the stage FAILED, and `retained` is not empty.
            failure=", ".join(f"{agent} failed" for agent in report.failed),
        )
        return report

    async def _stage_class_b(
        self,
        run: _Run,
        retrieval: FinalizedRetrieval,
        class_a: tuple[ClassAFinding, ...],
        specialists: SpecialistReport | None,
    ) -> ClassBOutcome | None:
        """The one Class B hunt, now reading the domain analysts' output as well.

        The specialists change what the hunter *reads*; the gate over what it may return is
        unchanged, and so is every rule downstream.
        """
        if self._class_b is None:
            run.record(AnalysisStage.CLASS_B, StageStatus.NOT_CONFIGURED)
            return None
        outcome = await self._class_b.execute(
            change=run.change,
            evidence=retrieval.outcome,
            class_a=class_a,
            specialists=_observations(specialists),
            slice_evidence=specialists.slice_outcomes if specialists else (),
            analogues=_analogues(specialists),
            correlation_id=run.correlation_id,
        )
        failure = outcome.hunt.failure if outcome.hunt else "the hunter returned no outcome"
        run.record_agentic(
            AnalysisStage.CLASS_B,
            succeeded=outcome.succeeded,
            produced=len(outcome.retained),
            failure=failure or "",
        )
        return outcome

    def _stage_gaps(
        self,
        run: _Run,
        retrieval: FinalizedRetrieval,
        class_b: ClassBOutcome | None,
    ) -> GapProductionOutcome | None:
        """The one hard dependency in the pipeline.

        A gap needs an executed search **and** an unanswered question. A failed hunt asked
        nothing, so there is no question - and reporting "no gaps" would claim the corpus was
        interrogated and found complete. Skipped instead.
        """
        if class_b is None or class_b.hunt is None:
            run.record(AnalysisStage.EVIDENCE_GAPS, StageStatus.NOT_CONFIGURED)
            return None
        if not class_b.succeeded:
            run.record(
                AnalysisStage.EVIDENCE_GAPS,
                StageStatus.SKIPPED,
                detail="the consequence-hunter failed, so no question was asked",
            )
            return None

        outcome = gaps_from_hunt(
            class_b.hunt,
            analysis_id=run.analysis_id,
            retrieval=retrieval,
            run_id=run.run_id,
            recorded_at=run.now,
        )
        run.record(
            AnalysisStage.EVIDENCE_GAPS,
            StageStatus.COMPLETED if outcome.gaps else StageStatus.EMPTY,
            produced=len(outcome.gaps),
        )
        return outcome

    async def _stage_contradictions(
        self, run: _Run, coupling: CouplingAnalysis, retrieval: FinalizedRetrieval
    ) -> ContradictionReport | None:
        if self._contradictions is None:
            run.record(AnalysisStage.CONTRADICTIONS, StageStatus.NOT_CONFIGURED)
            return None
        # Scoped to the keys the change touches, for the same reason the Class A selection
        # is: the corpus's standing divergences are the knowledge-decay report's subject, and
        # rating all of them against every change would make an unrelated change look busy -
        # and would pay for a provider call to do it (``CLAUDE.md``, cost discipline).
        keys = {delta.key for delta in run.change.assumption_deltas}
        report = await self._contradictions.execute(
            change=run.change,
            collisions=tuple(
                coupling.index.collisions[key] for key in coupling.index.keys if key in keys
            ),
            bindings={b.assumption_id: b for b in coupling.index.bindings},
            evidence=_stored_spans(retrieval),
            correlation_id=run.correlation_id,
        )
        run.record_agentic(
            AnalysisStage.CONTRADICTIONS,
            succeeded=report.outcome.succeeded,
            produced=len(report.findings),
            failure=report.outcome.failure or "",
        )
        return report

    async def _stage_challenge(
        self,
        run: _Run,
        spans: Mapping[str, EvidenceItem],
        class_a: tuple[ClassAFinding, ...],
        class_b: ClassBOutcome | None,
        blind_spots: BlindSpotReport | None,
    ) -> ChallengeReport | None:
        """Reviews **both** Class B hunts, and Class A even if both of them failed.

        The union of the two hunts' retained hypotheses, so a blind-spot finding faces the same
        falsification obligation as a main-hunt one. A failed hunt contributes an empty tuple,
        which is how degradation stays automatic: a review stage skipped because a discovery
        stage failed would leave the deterministic findings unreviewed *and* unmarked, which is
        worse than reviewing fewer candidates.

        ``spans`` covers the main retrieval **and** the blind-spot partition, which is the
        wiring this ordering needs - see the module docstring.
        """
        if self._challenge is None:
            run.record(AnalysisStage.CHALLENGE, StageStatus.NOT_CONFIGURED)
            return None
        report = await self._challenge.execute(
            change=run.change,
            class_a=class_a,
            hypotheses=(
                *(class_b.retained if class_b else ()),
                *(blind_spots.retained if blind_spots else ()),
            ),
            evidence=spans,
            correlation_id=run.correlation_id,
        )
        run.record_agentic(
            AnalysisStage.CHALLENGE,
            succeeded=report.outcome.succeeded,
            produced=len(report.results),
            failure=report.outcome.failure or "",
        )
        return report

    async def _stage_blind_spots(
        self,
        run: _Run,
        query: RetrievalQuery,
        coupling: CouplingAnalysis,
        retrieval: FinalizedRetrieval,
        class_a: tuple[ClassAFinding, ...],
        class_b: ClassBOutcome | None,
    ) -> BlindSpotReport | None:
        """Runs whether or not the main hunt succeeded, over evidence the main hunt never saw.

        A failed main hunt makes the partition *larger*, not smaller: every retrievable span
        is then unseen. Skipping here would mean a model outage in one stage silently removed
        the stage whose job is to cover what the other one missed.

        Runs **before** the challenger, so what it retains is reviewed rather than delivered
        on its own authority.
        """
        if self._blind_spots is None:
            run.record(AnalysisStage.BLIND_SPOTS, StageStatus.NOT_CONFIGURED)
            return None
        offered = frozenset(span.evidence_id for span in evidence_for_hunting(retrieval.outcome))
        report = await self._blind_spots.execute(
            change=run.change,
            query=query,
            main=retrieval,
            main_offered_ids=offered,
            # The existing decay report, which already derives the single-holder and
            # unwritten-rule categories the partition is built from.
            decay=coupling.report,
            class_a=class_a,
            existing_claims=tuple(h.claim for h in (class_b.retained if class_b else ())),
            correlation_id=run.correlation_id,
        )
        run.record_agentic(
            AnalysisStage.BLIND_SPOTS,
            succeeded=report.succeeded,
            produced=len(report.retained) + len(report.silences),
            failure=(report.outcome.failure if report.outcome else "no outcome") or "",
        )
        return report

    async def _stage_arbitration(
        self,
        run: _Run,
        candidates: _Candidates,
        gaps: GapProductionOutcome | None,
        contradictions: ContradictionReport | None,
        spans: Mapping[str, EvidenceItem],
        outcomes: tuple[RetrievalOutcome, ...],
    ) -> ArbitrationReport | None:
        """Positions each candidate on the evidence axis. Produces no number and no status."""
        if self._arbitration is None:
            run.record(AnalysisStage.EVIDENCE_ARBITRATION, StageStatus.NOT_CONFIGURED)
            return None
        report = await self._arbitration.execute(
            change=run.change,
            targets=candidates.targets,
            challenged=candidates.challenged,
            class_a=candidates.class_a,
            hypotheses=candidates.hypotheses,
            divergences=contradictions.outcome.assessed if contradictions else (),
            evidence_gap_ids=tuple(r.gap_id for r in gaps.gaps) if gaps else (),
            hunt_questions=candidates.questions,
            evidence=spans,
            outcomes=outcomes,
            bindings={},
            now=run.now,
            correlation_id=run.correlation_id,
        )
        run.record_agentic(
            AnalysisStage.EVIDENCE_ARBITRATION,
            succeeded=report.succeeded,
            produced=len(report.rulings),
            failure=report.outcome.failure or "",
        )
        return report

    async def _stage_judgment(
        self,
        run: _Run,
        retrieval: FinalizedRetrieval,
        candidates: _Candidates,
        gaps: GapProductionOutcome | None,
        contradictions: ContradictionReport | None,
        arbitration: ArbitrationReport | None,
        spans: Mapping[str, EvidenceItem],
        outcomes: tuple[RetrievalOutcome, ...],
    ) -> JudgmentReport | None:
        """Arbitrates the same candidate set the arbiter positioned.

        The arbiter's residual disagreements arrive as ``extra_disagreements`` and travel into
        every decision. Nothing in the status rules reads them, which is why an arbiter cannot
        raise a status - only add to what a reader is told is unsettled.
        """
        if self._judgment is None:
            run.record(AnalysisStage.JUDGMENT, StageStatus.NOT_CONFIGURED)
            return None

        report = await self._judgment.execute(
            change=run.change,
            run_provenance=run.provenance(retrieval),
            targets=candidates.targets,
            challenged=candidates.challenged,
            class_a=candidates.class_a,
            hypotheses=candidates.hypotheses,
            divergences=contradictions.outcome.assessed if contradictions else (),
            evidence_gap_ids=tuple(r.gap_id for r in gaps.gaps) if gaps else (),
            hunt_questions=candidates.questions,
            evidence=spans,
            outcomes=outcomes,
            bindings={},
            extra_disagreements=arbitration.extra_disagreements if arbitration else None,
            now=run.now,
            correlation_id=run.correlation_id,
        )
        run.record_agentic(
            AnalysisStage.JUDGMENT,
            succeeded=report.outcome.succeeded,
            produced=len(report.consequences),
            failure=report.outcome.failure or "",
        )
        return report

    # -- the report, and the reviewer state it reads ----------------------

    async def _stage_report(
        self, run: _Run, interim: AnalysisResult, spans: Mapping[str, EvidenceItem]
    ) -> ReportDocument:
        """Compose the report. Deterministic, so a failure here is a defect.

        The reviewer lookup happens first and through the *snapshot*, because the artifact
        version a suppression expires on is a property of the stored finding rather than of
        the consequence - a ``Consequence`` records a title, and a title does not change when
        a document is revised.
        """
        with run.required(AnalysisStage.REPORT) as record:
            reviewer = self._reviewer_signals(run, interim)
            document = await self._report.execute(
                analysis_id=run.analysis_id,
                change=run.change,
                run_provenance=run.provenance(interim.retrieval),
                consequences=interim.consequences,
                decisions=interim.decisions,
                scores=interim.scores,
                class_a=interim.class_a,
                contradictions=(
                    interim.contradictions.outcome.assessed if interim.contradictions else ()
                ),
                challenges=interim.challenge.results if interim.challenge is not None else (),
                blind_spots=interim.blind_spots,
                rulings=interim.arbitration.by_consequence if interim.arbitration else {},
                evidence_gap_ids=interim.evidence_gap_ids,
                open_questions=interim.open_questions,
                unmeasured_confidence_factors=(
                    interim.judgment.unmeasured_confidence_factors if interim.judgment else ()
                ),
                evidence=spans,
                reviewer=reviewer,
                correlation_id=run.correlation_id,
            )
            record(len(document.high_priority))
            return document

    def _reviewer_signals(self, run: _Run, interim: AnalysisResult) -> ReviewerSignals:
        """Standing reviewer verdicts that bear on *this* analysis's findings.

        Measured over the findings a reviewer has already ruled on for the same
        ``(assumption_key, source version)`` - which is the population the already-known rate
        is about: how often the system re-delivers something a reviewer told it they knew.

        Requires both stores. With neither, ``measured`` is ``False`` rather than a rate of
        zero: "nobody has reviewed" and "nothing is wired" are different facts.
        """
        if self._reviews is None or self._analyses is None:
            return ReviewerSignals()

        stored = snapshot(interim, stored_at=run.now, tenant_id=run.tenant_id)
        verdicts = self._reviews.verdicts(tenant_id=run.tenant_id)

        suppressed: list[str] = []
        reasons: dict[str, str] = {}
        already_known: list[str] = []
        reviewed = 0

        for finding in stored.findings:
            keys = tuple(finding.consequence.assumption_keys)
            version = finding.artifact_version_id
            matching = [
                v for v in verdicts if v.covers(assumption_keys=keys, artifact_version_id=version)
            ]
            if not matching:
                continue
            reviewed += 1
            if any(v.disposition is Disposition.ALREADY_KNOWN for v in matching):
                already_known.append(finding.consequence_id)
            verdict = _suppressing(
                matching,
                reviews=self._reviews,
                tenant_id=run.tenant_id,
                keys=keys,
                version=version,
            )
            if verdict is not None:
                suppressed.append(finding.consequence_id)
                reasons[finding.consequence_id] = verdict.value

        return ReviewerSignals(
            measured=True,
            already_known=tuple(already_known),
            suppressed=tuple(suppressed),
            suppressed_by=reasons,
            active_suppressions=len(self._reviews.suppressions(tenant_id=run.tenant_id)),
            reviewed=reviewed,
        )

    def _persist(self, run: _Run, result: AnalysisResult) -> None:
        """Store the finished analysis, so the next run's reviewer lookup has something to read.

        Silently skipped when no store was supplied, which is the deterministic-only and
        unit-test configuration. A store that raises is not swallowed: an analysis a reviewer
        cannot retrieve is an analysis nobody can act on.
        """
        if self._analyses is None:
            return
        self._analyses.put(snapshot(result, stored_at=run.now, tenant_id=run.tenant_id))


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    """Per-run context and the stage ledger. Mutable, and private to one execution."""

    change: ChangeSpec
    analysis_id: str
    run_id: str
    now: dt.datetime
    tenant_id: str
    registry_version: str
    as_of: dt.datetime
    correlation_id: str | None
    stages: list[StageOutcome] = field(default_factory=list)

    def record(
        self,
        stage: AnalysisStage,
        status: StageStatus,
        *,
        produced: int = 0,
        detail: str = "",
    ) -> None:
        self.stages.append(
            StageOutcome(stage=stage, status=status, produced=produced, detail=detail)
        )

    def record_agentic(
        self, stage: AnalysisStage, *, succeeded: bool, produced: int, failure: str
    ) -> None:
        """One place decides ``COMPLETED`` / ``EMPTY`` / ``FAILED`` for an agentic stage.

        Centralized so the three cannot drift apart between stages: a failure is a failure
        whether or not it also produced nothing, and an empty success is never recorded as a
        failure.
        """
        if not succeeded:
            self.record(stage, StageStatus.FAILED, produced=produced, detail=failure)
            logger.warning(
                "analysis_stage_failed",
                extra={"analysis_id": self.analysis_id, "stage": stage.value, "detail": failure},
            )
            return
        self.record(
            stage,
            StageStatus.COMPLETED if produced else StageStatus.EMPTY,
            produced=produced,
        )

    def required(self, stage: AnalysisStage) -> _RequiredStage:
        return _RequiredStage(run=self, stage=stage)

    def provenance(self, retrieval: FinalizedRetrieval) -> RunProvenance:
        """Everything needed to explain why two runs of this analysis differ."""
        return RunProvenance(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            confidence_fn_version=CONFIDENCE_FN_VERSION,
            coupling_index_version=COUPLING_INDEX_VERSION,
            registry_version=self.registry_version,
            retrieval_config_version=retrieval.retrieval_config_version,
            permission_scope_hash=retrieval.permission_scope_hash,
        )


@dataclass
class _RequiredStage:
    """Context manager for a deterministic stage: record on success, abort on failure."""

    run: _Run
    stage: AnalysisStage
    _produced: int = 0

    def __enter__(self) -> Any:
        def record(produced: int) -> None:
            self._produced = produced

        return record

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        """Records on success, re-raises named on failure. Never swallows.

        Returning ``None`` rather than a bool is deliberate and enforced by the type
        checker: a context manager that *could* return ``True`` might suppress the exception,
        and a required stage whose failure got suppressed is exactly the silent continuation
        requirement 6 forbids.
        """
        if exc is None:
            self.run.record(
                self.stage,
                StageStatus.COMPLETED if self._produced else StageStatus.EMPTY,
                produced=self._produced,
            )
            return
        if isinstance(exc, AnalysisStageError):
            return
        # A deterministic stage has no provider, so this is a defect rather than an outage.
        # Raising named is the opposite of silently continuing past it.
        raise AnalysisStageError(self.stage, exc) from exc


@dataclass(frozen=True)
class _Candidates:
    """The candidate set, resolved once and shared by the arbiter and the judge.

    Both stages call ``dossiers_from`` with these arguments, so both resolve the same
    candidates in the same order and therefore agree on every handle and every
    ``consequence_id``. Computing it twice from different arguments is exactly how a ruling
    ends up addressed to a consequence nobody arbitrated.
    """

    targets: tuple[ChallengeTarget, ...] = ()
    challenged: tuple[ChallengedCandidate, ...] = ()
    class_a: tuple[ClassAFinding, ...] = ()
    hypotheses: tuple[ConsequenceHypothesis, ...] = ()
    questions: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        *,
        class_a: tuple[ClassAFinding, ...],
        class_b: ClassBOutcome | None,
        blind_spots: BlindSpotReport | None,
        challenge: ChallengeReport | None,
    ) -> _Candidates:
        """Challenge targets where the challenger ran, the raw findings where it did not.

        **Both hypothesis sets are treated identically**, because the challenger now sees
        both: where it ran, its targets carry the established ``consequence_id`` for every
        candidate including the blind-spot ones, and passing the raw hypotheses again would
        mint a second id for a finding the challenge layer already keyed. Where it did not
        run, the raw findings from both hunts are the candidates - a review stage that failed
        must not delete what it failed to review.
        """
        reviewed = challenge is not None and bool(challenge.outcome.challenged)
        discovered = (
            *(class_b.retained if class_b else ()),
            *(blind_spots.retained if blind_spots else ()),
        )
        questions: list[str] = []
        for question in (
            *(class_b.hunt.unanswered_questions if class_b and class_b.hunt else ()),
            *(blind_spots.open_questions if blind_spots else ()),
        ):
            if question not in questions:
                questions.append(question)

        return cls(
            targets=challenge.targets if challenge is not None and reviewed else (),
            challenged=(challenge.outcome.challenged if challenge is not None and reviewed else ()),
            class_a=() if reviewed else class_a,
            hypotheses=() if reviewed else discovered,
            questions=tuple(questions),
        )


def _observations(report: SpecialistReport | None) -> tuple[SpecialistObservation, ...]:
    """Project the retained specialist findings into what the hunter is shown.

    A flat projection, because ``app/consequences`` may not import ``app.specialists`` - and
    because the hunter needs a claim, its citations and who made it, not a domain entity.
    Suppressed findings are not projected: a duplicate the fan-out already absorbed should not
    reappear in the prompt that pays for it.
    """
    if report is None:
        return ()
    return tuple(
        SpecialistObservation(
            agent=finding.agent,
            category=finding.category,
            claim=finding.claim,
            evidence_position=finding.evidence_position.value,
            evidence_ids=finding.evidence_ids,
        )
        for finding in report.retained
    )


def _analogues(report: SpecialistReport | None) -> tuple[AdmissibleAnalogue, ...]:
    """Project the admitted **anchors** into what the hunter may cite.

    Anchors only - an event with no recorded outcome is not offered, because offering one
    would invite a citation the status gate then has to refuse. The ``event_id`` here was
    derived from evidence spans by :func:`app.ingestion.identity.event_id_for`, so the set the
    hunter is checked against is one no model authored.
    """
    if report is None:
        return ()
    return tuple(
        AdmissibleAnalogue(
            event_id=event.event_id,
            title=event.title,
            observed_outcome=event.observed_outcome or "",
        )
        for event in report.anchors
    )


def _suppressing(
    matching: Sequence[StandingVerdict],
    *,
    reviews: RecordDispositionUseCase,
    tenant_id: str,
    keys: tuple[str, ...],
    version: str | None,
) -> Disposition | None:
    """The verdict that suppresses this finding, if one does and has not expired.

    Expiry is delegated to ``is_suppressed``, which delegates to
    ``suppression_has_expired``. There is no second expiry rule here, and this function
    cannot suppress on ``already_known`` because that value is not in ``SUPPRESSING``.
    """
    for verdict in matching:
        if verdict.disposition not in SUPPRESSING:
            continue
        if any(
            reviews.is_suppressed(
                tenant_id=tenant_id, assumption_key=key, artifact_version_id=version
            )
            for key in keys
        ):
            return verdict.disposition
    return None


def _stored_spans(
    retrieval: FinalizedRetrieval, *, extra: Sequence[RetrievalOutcome] = ()
) -> Mapping[str, EvidenceItem]:
    """The spans the requester actually received, keyed by evidence id.

    Built from retrieval outcomes and nowhere else, which is how the permission scope reaches
    every downstream stage: they see this map or nothing. No stage below retrieval touches a
    store, so none can widen it - and ``extra`` holds the blind-spot partition's outcomes,
    which came from the same retriever under the same query's scope.
    """
    spans = {
        result.candidate.evidence_id: result.candidate.evidence
        for result in retrieval.outcome.results
    }
    for outcome in extra:
        for result in outcome.results:
            spans.setdefault(result.candidate.evidence_id, result.candidate.evidence)
    return spans
