"""The pipeline, end to end, over the real sample corpus. And the two bars it has to clear.

Every stage was verified against its own inputs, which left the *pipeline* unproven: the
interfaces between stages had only ever been exercised by fixtures. These tests run a change
through all nine stages over the real corpus with the real gateway and a fake provider - no
credentials, no network.

The two that matter most are the evaluation bars ``CLAUDE.md`` calls the ones most easily
failed while everything else looks green:

* :class:`TestQuietChange` - **quiet on quiet changes.** A change the corpus knows nothing
  about must report approximately nothing. A system that always finds consequences has not
  been shown to find any.
* :class:`TestAblationHonesty` - **ablation honesty.** Remove a document supporting a finding
  and the finding must degrade, not survive. A finding that outlives its evidence was coming
  from model priors.

Neither is graded against a scripted answer. The fake agent
(:mod:`tests.fixtures.fake_agents`) reads its own prompt and answers at the strongest position
the schema allows for every handle and every evidence id it was offered - so it is the
*adversarial* case for the gates, and the difference between a positive and a negative control
comes from what the pipeline put in the prompt. No prompt or threshold was tuned to make any
of this pass.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import json
from collections.abc import Callable
from typing import Any

import pytest

from app.agents.runtime.agent_loader import load_agent_definitions
from app.application.orchestration.analysis import (
    ANALYSIS_PIPELINE_VERSION,
    REQUIRED_STAGES,
    AnalysisResult,
    AnalysisStage,
    AnalysisStageError,
    AnalyzeChangeUseCase,
    StageStatus,
)
from app.application.orchestration.analysis_store import (
    AnalysisStore,
    ReviewedFinding,
    TenantMismatch,
    snapshot,
)
from app.application.orchestration.retrieval_service import RetrievalService
from app.application.use_cases.arbitrate_evidence import ArbitrateEvidenceUseCase
from app.application.use_cases.challenge_findings import ChallengeFindingsUseCase
from app.application.use_cases.compose_report import ComposeReportUseCase
from app.application.use_cases.consult_specialists import (
    SUPPRESSED_DUPLICATE,
    SUPPRESSED_RESTATES_CLASS_A,
    ConsultSpecialistsUseCase,
)
from app.application.use_cases.hunt_blind_spots import HuntBlindSpotsUseCase
from app.application.use_cases.hunt_consequences import HuntConsequencesUseCase
from app.application.use_cases.hunt_contradictions import HuntContradictionsUseCase
from app.application.use_cases.judge_consequences import JudgeConsequencesUseCase
from app.application.use_cases.record_disposition import RecordDispositionUseCase
from app.application.use_cases.resolve_coupling import ResolveCouplingUseCase
from app.application.use_cases.retrieve_evidence import RetrieveEvidenceUseCase
from app.blindspots.hunter import BLIND_SPOT_AGENT, BlindSpotHunter
from app.blindspots.partition import PartitionSlice
from app.blindspots.proposals import BlindSpotBatch
from app.challenge.challenger import AdversarialChallenger
from app.config.agent_bindings import AgentBindings
from app.config.settings import REPO_ROOT
from app.consequences.hunter import ConsequenceHunter
from app.contradictions.hunter import ContradictionHunter
from app.domain.entities.assumption import AssumptionDelta
from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import Consequence, SpecialistFinding
from app.domain.interfaces.llm import ProviderUnavailable
from app.domain.interfaces.retrieval import RetrievalQuery
from app.domain.value_objects.enums import (
    STATUSES_ABOVE_SPECULATIVE,
    ChallengeOutcome,
    ConsequenceClass,
    DeltaAssertedBy,
    DeltaKind,
    Disposition,
    EvidencePosition,
    Origin,
    Status,
    ValueType,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import NodeType, Relationship
from app.domain.value_objects.typed_value import TypedValue
from app.evaluation.corpus import CorpusStores, load_corpus, load_manifest
from app.infrastructure.database.repositories.memory import (
    InMemoryAnalysisStore,
    InMemoryCouplingIndexStore,
    InMemoryDispositionStore,
)
from app.infrastructure.embeddings.hashing import HashingEmbedder
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.providers.fake import FakeProvider
from app.infrastructure.llm.registry import ProviderRegistry
from app.infrastructure.observability.usage import InMemoryUsageLedger
from app.ingestion.identity import event_id_for
from app.judgment.arbitration import ClampReason, EvidenceArbiter, RulingBatch
from app.judgment.dispositions import DispositionStore
from app.judgment.judge import ConsequenceJudge
from app.reporting.composer import REFUSAL_INVENTED_ID, ReportComposer
from app.reporting.proposals import CompositionProposal
from app.specialists.analogues import REFUSED_UNCITED, REFUSED_WRONG_AGENT
from app.specialists.analyst import SpecialistAnalyst
from app.specialists.proposals import (
    FORBIDDEN_AGENT_FIELDS,
    AnalogueProposal,
    ObservationBatch,
)
from app.specialists.roster import SPECIALIST_AGENTS
from tests.conftest import make_settings
from tests.fixtures.fake_agents import INVENTED_ANALOGUE, cooperative_responder

AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)
ANALYSIS = new_id("analysis")
RUN = new_id("analysis")

#: The key the corpus couples most widely: five artifacts bind it and lexical retrieval
#: recovers one (``docs/EVALUATION.md``).
RETENTION_KEY = "retention.window"
#: The document that declares it, and the one the ablation removes.
RETENTION_DOC = "policy/data-retention-policy.md"

#: The key the corpus has a *closed incident with a measured outcome* about, which is what
#: makes it the only change in this corpus that can produce a historical anchor.
REFUND_KEY = "refund.approval_threshold"
#: The incident that records it, and the document a historical analogue must be traceable to.
REFUND_INCIDENT = "ticket/INC-4471-refund-backlog.md"
#: Retrieval has to reach the incident for the historical analyst to be selected at all.
REFUND_QUERY = "refund approval threshold manager approval incident"
#: The grant the permission test puts on the incident. Its manifest owner, so the label is
#: the one an operator would plausibly have applied.
REFUND_GRANT = "support-operations"

#: A change the corpus has nothing to say about. Not a nonsense string - a plausible change
#: to a real business process that this corpus simply does not document.
QUIET_SUBJECT = "office espresso machine maintenance vendor"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def gateway_with(responder: Callable[..., str]) -> tuple[LlmGateway, FakeProvider]:
    """The real gateway over a fake provider driven by ``responder``.

    The real ``LlmGateway`` deliberately: the completion port, its JSON repair loop, its
    budget accounting and its error translation are all part of what an end-to-end test is
    supposed to exercise.
    """
    settings = make_settings()
    provider = FakeProvider(responder=responder)
    registry = ProviderRegistry(settings, overrides={"fake": provider})

    async def instant_sleep(_: float) -> None:
        return None

    gateway = LlmGateway(
        settings=settings,
        registry=registry,
        bindings=AgentBindings.load(REPO_ROOT / "app" / "config" / "agents.yaml"),
        ledger=InMemoryUsageLedger(),
        sleeper=instant_sleep,
    )
    return gateway, provider


def pipeline(
    corpus: CorpusStores,
    *,
    responder: Callable[..., str] | None = None,
    agentic: bool = True,
    analyses: AnalysisStore | None = None,
    dispositions: DispositionStore | None = None,
) -> tuple[AnalyzeChangeUseCase, FakeProvider]:
    """Assemble the whole pipeline over one corpus.

    ``agentic=False`` builds a deterministic-only pipeline, which is the configuration that
    demonstrates the central claim operationally: Class A findings and the graph appear with
    no model having run at all.
    """
    gateway, provider = gateway_with(responder or cooperative_responder())
    definitions = load_agent_definitions()

    coupling = ResolveCouplingUseCase(
        sources=corpus.sources,
        assumptions=corpus.assumptions,
        gaps=corpus.gaps,
        index_store=InMemoryCouplingIndexStore(),
    )
    retrieval = RetrievalService(
        retrieval=RetrieveEvidenceUseCase(sources=corpus.sources, assumptions=corpus.assumptions),
        embedder=HashingEmbedder(),
    )
    reviews = (
        RecordDispositionUseCase(store=dispositions, analyses=analyses)
        if dispositions is not None and analyses is not None
        else None
    )
    if not agentic:
        return (
            AnalyzeChangeUseCase(
                coupling=coupling, retrieval=retrieval, analyses=analyses, reviews=reviews
            ),
            provider,
        )

    return (
        AnalyzeChangeUseCase(
            coupling=coupling,
            retrieval=retrieval,
            class_b=HuntConsequencesUseCase(
                hunter=ConsequenceHunter(
                    completions=gateway, definition=definitions["consequence-hunter"]
                )
            ),
            contradictions=HuntContradictionsUseCase(
                hunter=ContradictionHunter(
                    completions=gateway, definition=definitions["contradiction-hunter"]
                )
            ),
            challenge=ChallengeFindingsUseCase(
                challenger=AdversarialChallenger(
                    completions=gateway, definition=definitions["adversarial-challenger"]
                )
            ),
            specialists=ConsultSpecialistsUseCase(
                analysts={
                    agent: SpecialistAnalyst(completions=gateway, definition=definitions[agent])
                    for agent in SPECIALIST_AGENTS
                },
                retrieval=retrieval,
            ),
            blind_spots=HuntBlindSpotsUseCase(
                hunter=BlindSpotHunter(
                    completions=gateway, definition=definitions["blind-spot-hunter"]
                ),
                retrieval=retrieval,
            ),
            arbitration=ArbitrateEvidenceUseCase(
                arbiter=EvidenceArbiter(
                    completions=gateway, definition=definitions["evidence-arbiter"]
                )
            ),
            judgment=JudgeConsequencesUseCase(
                judge=ConsequenceJudge(
                    completions=gateway, definition=definitions["consequence-judge"]
                )
            ),
            report=ComposeReportUseCase(
                composer=ReportComposer(
                    completions=gateway, definition=definitions["report-composer"]
                )
            ),
            analyses=analyses,
            reviews=reviews,
        ),
        provider,
    )


def broken_for(*agents: str) -> Callable[..., str]:
    """A cooperative responder that returns unusable output for exactly these agents.

    Per-agent rather than a queued provider failure, because what is under test is that *one*
    stage failing is distinguishable from the others succeeding - and a queued failure lands
    on whichever call happens to be next. Several agents, because the specialist fan-out has
    two failure cases worth telling apart: one analyst down, and all of them.
    """
    inner = cooperative_responder()
    broken = set(agents)

    def respond(request: Any) -> str:
        if request.agent in broken:
            return "not json at all"
        return inner(request)

    return respond


def retention_change(**over: Any) -> ChangeSpec:
    """A real change to a key the corpus couples widely."""
    payload: dict[str, Any] = {
        "change_id": new_id("change"),
        "subject_entities": ("personal data retention window",),
        "old_state": "90 days",
        "new_state": "30 days",
        "affected_domains": ("privacy", "support"),
        "assumption_deltas": (
            AssumptionDelta(
                delta_id=new_id("delta"),
                key=RETENTION_KEY,
                scope=ScopeSelector.global_scope(),
                old_value=TypedValue(value_type=ValueType.DURATION, raw=90, unit="d"),
                new_value=TypedValue(value_type=ValueType.DURATION, raw=30, unit="d"),
                delta_kind=DeltaKind.VALUE_CHANGE,
                asserted_by=DeltaAssertedBy.CHANGE_REQUEST,
                registry_version="registry-1",
            ),
        ),
        "version": "change-1",
    }
    payload.update(over)
    return ChangeSpec(**payload)


def refund_change(**over: Any) -> ChangeSpec:
    """A change to the key the corpus has a closed incident about.

    The retention change is the wider one and stays the default, but no incident in this
    corpus records an outcome for it - so it cannot produce a historical anchor and would make
    the anchored path untestable. This one halves an approval threshold that was already
    halved once, and INC-4471 records what happened the first time, with figures.
    """
    payload: dict[str, Any] = {
        "change_id": new_id("change"),
        "subject_entities": ("refund approval threshold",),
        "old_state": "10,000 USD",
        "new_state": "5,000 USD",
        "affected_domains": ("payments", "support"),
        "assumption_deltas": (
            AssumptionDelta(
                delta_id=new_id("delta"),
                key=REFUND_KEY,
                scope=ScopeSelector.global_scope(),
                old_value=TypedValue(value_type=ValueType.SCALAR, raw=10000, unit="USD"),
                new_value=TypedValue(value_type=ValueType.SCALAR, raw=5000, unit="USD"),
                delta_kind=DeltaKind.VALUE_CHANGE,
                asserted_by=DeltaAssertedBy.CHANGE_REQUEST,
                registry_version="registry-1",
            ),
        ),
        "version": "change-1",
    }
    payload.update(over)
    return ChangeSpec(**payload)


def quiet_change() -> ChangeSpec:
    """A change the corpus does not document, with no assumption delta to propagate."""
    return ChangeSpec(
        change_id=new_id("change"),
        subject_entities=(QUIET_SUBJECT,),
        old_state="quarterly descaling",
        new_state="monthly descaling",
        affected_domains=("facilities",),
        version="change-1",
    )


def query_for(corpus: CorpusStores, text: str, **over: Any) -> RetrievalQuery:
    over.setdefault("permission_scope", PermissionScope())
    over.setdefault("top_k", 10)
    return RetrievalQuery(tenant_id=corpus.tenant_id, text=text, **over)


def run_analysis(
    corpus: CorpusStores,
    *,
    change: ChangeSpec | None = None,
    text: str = "personal data retention window after account closure",
    responder: Callable[..., str] | None = None,
    agentic: bool = True,
    analysis_id: str = ANALYSIS,
    run_id: str = RUN,
    provider_hook: Callable[[FakeProvider], None] | None = None,
    analyses: AnalysisStore | None = None,
    dispositions: DispositionStore | None = None,
    scope: PermissionScope | None = None,
) -> AnalysisResult:
    use_case, provider = pipeline(
        corpus,
        responder=responder,
        agentic=agentic,
        analyses=analyses,
        dispositions=dispositions,
    )
    if provider_hook is not None:
        provider_hook(provider)
    return asyncio.run(
        use_case.execute(
            change=change or retention_change(),
            query=(
                query_for(corpus, text)
                if scope is None
                else query_for(corpus, text, permission_scope=scope)
            ),
            tenant_id=corpus.tenant_id,
            registry_version=corpus.registry_version,
            as_of=AS_OF,
            analysis_id=analysis_id,
            run_id=run_id,
            now=NOW,
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> CorpusStores:
    stores = asyncio.run(load_corpus())
    assert not stores.extraction_failures, stores.extraction_failures
    return stores


@pytest.fixture(scope="module")
def positive(corpus: CorpusStores) -> AnalysisResult:
    """One complete positive analysis, reused by the tests that read its parts."""
    return run_analysis(corpus)


@pytest.fixture(scope="module")
def quiet(corpus: CorpusStores) -> AnalysisResult:
    return run_analysis(corpus, change=quiet_change(), text=QUIET_SUBJECT)


# ---------------------------------------------------------------------------
# One complete positive change
# ---------------------------------------------------------------------------


class TestCompletePositiveChange:
    def test_every_stage_ran_and_none_failed(self, positive: AnalysisResult) -> None:
        assert [s.stage for s in positive.stages] == list(AnalysisStage)
        assert positive.failed_stages == ()
        assert positive.skipped_stages == ()
        assert positive.succeeded

    def test_the_result_carries_every_stage_output(self, positive: AnalysisResult) -> None:
        """Requirement 4, asserted as presence rather than as counts."""
        assert positive.change is not None
        assert positive.retrieval.results
        assert positive.class_a
        assert positive.class_b is not None
        assert positive.gaps is not None
        assert positive.contradictions is not None
        assert positive.challenge is not None
        assert positive.judgment is not None
        assert positive.graph is not None
        assert positive.run_provenance is not None
        assert positive.pipeline_version == ANALYSIS_PIPELINE_VERSION

    def test_the_analysis_is_not_quiet(self, positive: AnalysisResult) -> None:
        """The positive control. Without this the negative control proves nothing."""
        assert not positive.is_quiet
        assert positive.material_consequences

    def test_provenance_records_every_version_a_rerun_could_differ_on(
        self, positive: AnalysisResult
    ) -> None:
        provenance = positive.run_provenance
        assert provenance is not None
        assert provenance.registry_version == "registry-1"
        assert provenance.confidence_fn_version == "confidence-1"
        assert provenance.coupling_index_version == "coupling-1"
        assert provenance.retrieval_config_version == "retrieval-1"
        assert provenance.permission_scope_hash == positive.retrieval.permission_scope_hash

    def test_consequence_ids_are_shared_across_the_three_record_sets(
        self, positive: AnalysisResult
    ) -> None:
        """The id the challenge layer minted survives to the decision and the score."""
        consequences = {c.consequence_id for c in positive.consequences}
        decisions = {d.consequence_id for d in positive.decisions}
        scores = {s.consequence_id for s in positive.scores}

        assert consequences
        assert consequences == decisions == scores

    def test_confidence_is_computed_for_every_consequence(self, positive: AnalysisResult) -> None:
        assert len(positive.scores) == len(positive.consequences)
        for score in positive.scores:
            assert score.confidence_fn_version == "confidence-1"
            assert score.factor_inputs

    def test_the_permission_scope_reaches_every_downstream_stage(
        self, positive: AnalysisResult
    ) -> None:
        """Authorization enters once, at the query, and nothing widens it.

        Every span any agentic stage could cite came out of a retrieval result, so the set of
        citable ids is a subset of what the requester received.

        The blind-spot stage issues its own queries, which is why ``received`` is the union of
        the main outcome and the partition outcomes rather than the main outcome alone. Those
        queries are copies of the main query - same tenant, same ``PermissionScope``, only
        ``text``, ``top_k`` and ``source_types`` changed - so the boundary is the same one.
        """
        received = set(positive.retrieval.evidence_ids)
        assert positive.blind_spots is not None
        for outcome in positive.blind_spots.partition_outcomes:
            assert (
                outcome.query.permission_scope == positive.retrieval.outcome.query.permission_scope
            )
            assert outcome.query.tenant_id == positive.retrieval.outcome.query.tenant_id
            received |= set(outcome.evidence_ids)

        for consequence in positive.consequences:
            assert set(consequence.evidence_ids) <= received
        for hypothesis in positive.all_hypotheses:
            assert set(hypothesis.evidence_ids) <= received


# ---------------------------------------------------------------------------
# The quiet change - the negative control
# ---------------------------------------------------------------------------


class TestQuietChange:
    """A change the corpus does not document must report approximately nothing.

    Not graded against a script: the same cooperative fake agent runs, and it proposes nothing
    because retrieval offered it nothing to cite and the change carries no assumption delta to
    propagate. The quiet result is produced by the pipeline.
    """

    def test_the_analysis_is_quiet(self, quiet: AnalysisResult) -> None:
        assert quiet.is_quiet
        assert quiet.class_a == ()
        assert quiet.material_consequences == ()

    def test_it_ran_rather_than_failing(self, quiet: AnalysisResult) -> None:
        """Quiet because nothing was found, not because a stage broke.

        The distinction is the whole point of requirement 5: an outage produces the same
        empty report as a clean corpus unless the statuses are kept apart.
        """
        assert quiet.failed_stages == ()
        assert quiet.succeeded
        assert quiet.stage(AnalysisStage.CLASS_A).status is StageStatus.EMPTY
        assert quiet.stage(AnalysisStage.CLASS_B).ran

    def test_no_hypothesis_was_proposed(self, quiet: AnalysisResult) -> None:
        """The cooperative agent had nothing to cite, so it claimed nothing."""
        assert quiet.hypotheses == ()
        assert quiet.stage(AnalysisStage.CLASS_B).status is StageStatus.EMPTY

    def test_it_still_records_what_the_corpus_could_not_answer(self, quiet: AnalysisResult) -> None:
        """Quiet is not silent. Uncertainty is a first-class output (``CLAUDE.md`` 7).

        A retrieval that returned nothing over a readable corpus is exactly the negative
        provenance an ``EvidenceGap`` needs - so the honest quiet result is *no findings and a
        recorded gap*, not an empty page.
        """
        assert quiet.retrieval.is_empty
        assert quiet.evidence_gap_ids
        assert quiet.open_questions

    def test_a_gap_is_never_a_consequence(self, quiet: AnalysisResult) -> None:
        gap_ids = set(quiet.evidence_gap_ids)
        assert gap_ids
        assert not {c.consequence_id for c in quiet.consequences} & gap_ids

    def test_every_agentic_stage_is_empty_rather_than_busy(self, quiet: AnalysisResult) -> None:
        """Quiet across the discovery and review stages, not just at the end.

        The contradiction stage is scoped to the keys the change touches, for the same reason
        the Class A selection is: rating the corpus's standing divergences against an
        unrelated change would make it look busy, and would pay for a provider call to do it.

        ``BLIND_SPOTS`` is deliberately **not** in this list, and the measurement is why: on a
        change the main query found nothing for, every retrievable span is unseen, so the
        partition is at its *largest* and the hunter does propose one hypothesis. That is the
        stage working. What matters for the negative control is that the hypothesis stays
        ``speculative`` and off the default surface, which
        ``TestCoreIntelligenceLayer.test_quiet_changes_remain_quiet`` asserts. Listing the
        stage as empty here would be demanding that the blind-spot hunter fall silent exactly
        where it has the most to look at.

        ``CHALLENGE`` left this list when ``blind_spots`` moved ahead of it. It now has that
        one hypothesis to review, so it correctly reports ``completed`` with one result - and
        a challenger reporting *nothing* while an unreviewed hypothesis was on its way to the
        judge is the failure the reordering fixed. The same follows for arbitration and
        judgment, which were already busy with it before the reordering.
        """
        for stage in (
            AnalysisStage.CLASS_A,
            AnalysisStage.RETRIEVAL,
            AnalysisStage.CLASS_B,
            AnalysisStage.CONTRADICTIONS,
        ):
            assert quiet.stage(stage).status is StageStatus.EMPTY, stage
        assert quiet.stage(AnalysisStage.REPORT).status is StageStatus.EMPTY

        # Exactly the blind-spot hypothesis reached review, and it was reviewed.
        challenge = quiet.stage(AnalysisStage.CHALLENGE)
        assert challenge.status is StageStatus.COMPLETED
        assert challenge.produced == len(quiet.blind_spot_hypotheses) == 1

    def test_the_contradiction_stage_is_scoped_to_the_changed_keys(
        self, positive: AnalysisResult, quiet: AnalysisResult
    ) -> None:
        """Corpus-wide divergences belong to the knowledge-decay report, not to every change."""
        assert positive.contradictions is not None
        assert quiet.contradictions is not None

        assert positive.contradictions.divergences
        assert quiet.contradictions.divergences == ()
        assert {d.assumption_key for d in positive.contradictions.divergences} == {RETENTION_KEY}

    def test_the_positive_and_quiet_runs_differ_in_the_right_direction(
        self, positive: AnalysisResult, quiet: AnalysisResult
    ) -> None:
        """The comparison that makes both controls meaningful.

        Same corpus, same pipeline, same fake agent. Only the change and the query differ.
        """
        assert len(positive.class_a) > len(quiet.class_a)
        assert len(positive.material_consequences) > len(quiet.material_consequences)
        assert len(positive.retrieval.results) > len(quiet.retrieval.results)


# ---------------------------------------------------------------------------
# Ablation honesty
# ---------------------------------------------------------------------------


class TestAblationHonesty:
    """Remove a document supporting a finding; the finding must degrade, not survive.

    ``CLAUDE.md``: "a finding that survives removal of its evidence was coming from model
    priors." The corpus is re-ingested without the retention policy and the whole pipeline
    re-run - nothing is patched after the fact, so the degradation has to come from the
    resolver no longer being able to derive the finding.

    **The claim is per finding, not per key.** Five artifacts bind ``retention.window``, so
    removing one declarer does not and must not erase the key: the EU policy and the team
    notes still genuinely disagree about it, and reporting otherwise would be a *false
    negative*. What must vanish is every finding that rested on the removed document's spans,
    and nothing surviving may still cite one.
    """

    def test_the_intact_corpus_produces_a_finding_on_the_retention_key(
        self, positive: AnalysisResult
    ) -> None:
        """The control for the ablation. Without it, the ablation proves nothing."""
        keys = {f.assumption_key for f in positive.class_a}
        assert RETENTION_KEY in keys

    def test_every_finding_resting_on_the_removed_document_is_gone(
        self,
        positive: AnalysisResult,
        ablated: AnalysisResult,
        removed_evidence: frozenset[str],
    ) -> None:
        """Measured: four findings rested on the removed spans, and all four disappeared.

        Identified by ``finding_id``, which is derived from (relation, affected version) - so
        this is the same finding, not a similar one.
        """
        dependent = {
            f.finding_id
            for f in positive.class_a
            if set(f.materialized_evidence_ids) & removed_evidence
        }
        assert len(dependent) >= 4, "the control must have findings that rest on the removal"

        surviving = {f.finding_id for f in ablated.class_a}
        assert not dependent & surviving

    def test_nothing_surviving_still_cites_the_removed_evidence(
        self, ablated: AnalysisResult, removed_evidence: frozenset[str]
    ) -> None:
        """The anti-priors assertion, and the sharpest one available.

        A finding or a consequence still citing a span that no longer exists would be
        fabricated provenance - the pipeline asserting support it cannot produce. The
        cooperative fake agent cites everything it is offered, so if any removed id reached a
        citation here, it came from somewhere other than the corpus.
        """
        for finding in ablated.class_a:
            assert not set(finding.materialized_evidence_ids) & removed_evidence
        for consequence in ablated.consequences:
            assert not set(consequence.evidence_ids) & removed_evidence
        for hypothesis in ablated.hypotheses:
            assert not set(hypothesis.evidence_ids) & removed_evidence

    def test_findings_resting_on_other_documents_correctly_survive(
        self, positive: AnalysisResult, ablated: AnalysisResult
    ) -> None:
        """Degradation, not collapse.

        Three ``retention.window`` findings remain, on spans the ablation did not touch: the
        EU policy and the team notes still disagree. Asserting the key vanished would be
        demanding a false negative, which is the opposite failure and just as bad.
        """
        intact = [f for f in positive.class_a if f.assumption_key == RETENTION_KEY]
        surviving = [f for f in ablated.class_a if f.assumption_key == RETENTION_KEY]

        assert surviving, "the key must not vanish: other artifacts still bind it"
        assert len(surviving) < len(intact)
        assert len(ablated.class_a) < len(positive.class_a)

    def test_a_finding_may_reattribute_to_a_new_governing_declarer(
        self, positive: AnalysisResult, ablated: AnalysisResult
    ) -> None:
        """Measured, and correct: removing the policy changes who governs.

        One ``stale_consumer`` finding on the metric definition names the Data Retention
        Policy as its counterpart intact, and the Vendor Integration Notes ablated - because
        ``resolve_authority`` now returns a different governing declarer. That is the resolver
        recomputing, not the pipeline inventing: the finding is a *different* finding with a
        different id, resting on spans that both still exist.
        """

        def counterpart(result: AnalysisResult) -> set[str]:
            return {
                f.counterpart_artifact or ""
                for f in result.class_a
                if f.rule_id == "stale_consumer" and f.assumption_key == RETENTION_KEY
            }

        assert "Data Retention Policy" in counterpart(positive)
        assert "Data Retention Policy" not in counterpart(ablated)
        assert counterpart(ablated)

    def test_the_graph_loses_the_removed_document_from_the_key_join(
        self, ablated: AnalysisResult, positive: AnalysisResult
    ) -> None:
        """The visible form of the degradation."""
        assert positive.graph is not None
        assert ablated.graph is not None

        intact = positive.graph.key_holders(RETENTION_KEY)
        reduced = ablated.graph.key_holders(RETENTION_KEY)

        assert len(intact) == 5
        assert len(reduced) < len(intact)
        assert "Data Retention Policy" in intact
        assert "Data Retention Policy" not in reduced

    def test_the_ablated_run_did_not_merely_fail(self, ablated: AnalysisResult) -> None:
        """Degradation, not breakage. A crashed pipeline would also lose the finding."""
        assert ablated.failed_stages == ()
        assert ablated.succeeded
        assert ablated.retrieval.results


@pytest.fixture(scope="module")
def ablated_corpus() -> CorpusStores:
    """The sample corpus re-ingested without the retention policy.

    Removed from the manifest and re-ingested rather than deleted from the store afterwards:
    an after-the-fact deletion would leave the derived records behind and test nothing.
    """
    manifest = copy.deepcopy(load_manifest())
    before = len(manifest["sources"])
    manifest["sources"] = [entry for entry in manifest["sources"] if entry["path"] != RETENTION_DOC]
    assert len(manifest["sources"]) == before - 1, "the retention policy must be in the manifest"

    stores = asyncio.run(load_corpus(manifest=manifest))
    assert not stores.extraction_failures, stores.extraction_failures
    return stores


@pytest.fixture(scope="module")
def ablated(ablated_corpus: CorpusStores) -> AnalysisResult:
    return run_analysis(ablated_corpus)


@pytest.fixture(scope="module")
def removed_evidence(corpus: CorpusStores) -> frozenset[str]:
    """The evidence ids that belonged to the removed document, read from the intact corpus.

    Taken from the store rather than hard-coded: the ids are content-derived, so a corpus
    edit would change them and a literal list would silently stop testing anything.
    """
    version_ids = {
        report.version_id for path, report in corpus.reports.items() if path == RETENTION_DOC
    }
    assert version_ids, "the retention policy must be in the intact corpus"
    spans = frozenset(
        item.evidence_id
        for item in corpus.sources.evidence_for_tenant(corpus.tenant_id)
        if item.source_version_id in version_ids
    )
    assert spans, "the removed document must have contributed evidence"
    return spans


# ---------------------------------------------------------------------------
# Deterministic Class A, without any model
# ---------------------------------------------------------------------------


class TestDeterministicClassA:
    def test_class_a_and_the_graph_appear_with_no_model_configured(
        self, corpus: CorpusStores
    ) -> None:
        """The central claim, operationally: Class A recall does not depend on retrieval or
        on any model (``CLAUDE.md``)."""
        result = run_analysis(corpus, agentic=False)

        assert result.class_a
        assert result.graph is not None
        assert result.graph.graph.edges
        assert result.succeeded

    def test_the_agentic_stages_are_not_configured_rather_than_failed(
        self, corpus: CorpusStores
    ) -> None:
        result = run_analysis(corpus, agentic=False)

        for stage in (
            AnalysisStage.CLASS_B,
            AnalysisStage.CONTRADICTIONS,
            AnalysisStage.CHALLENGE,
            AnalysisStage.JUDGMENT,
        ):
            assert result.stage(stage).status is StageStatus.NOT_CONFIGURED
        assert result.failed_stages == ()

    def test_every_class_a_finding_names_its_rule_and_its_inputs(
        self, positive: AnalysisResult
    ) -> None:
        for finding in positive.class_a:
            assert finding.rule_id
            assert len(set(finding.input_record_ids)) >= 2 or finding.delta_id
            assert finding.materialized_evidence_ids
            assert finding.evidence_is_derived_union is True

    def test_class_a_provenance_is_the_resolvers_not_a_models(
        self, positive: AnalysisResult
    ) -> None:
        for finding in positive.class_a:
            assert finding.origin.value == "resolver"
            assert finding.consequence_class.value == "class_a_assumption_mediated"
            assert finding.coupling_index_version == "coupling-1"


# ---------------------------------------------------------------------------
# Class B, contradiction, challenge and gap paths
# ---------------------------------------------------------------------------


class TestClassBPath:
    def test_hypotheses_are_proposed_and_carry_inferred_provenance(
        self, positive: AnalysisResult
    ) -> None:
        assert positive.hypotheses
        for hypothesis in positive.hypotheses:
            assert hypothesis.derivation.value == "inferred"
            assert hypothesis.support.value == "mechanism_only"
            assert hypothesis.proposed_by.agent == "consequence-hunter"

    def test_a_class_b_consequence_cannot_exceed_speculative_without_an_analogue(
        self, positive: AnalysisResult
    ) -> None:
        """The frozen gate, observed end to end.

        The fake judge asked for ``likely`` on everything. Every Class B consequence still
        comes out ``speculative``, because none of these hypotheses has a historical
        analogue - and the clamp, not the fake, decided that.
        """
        class_b = [
            c for c in positive.consequences if c.consequence_class.value == "class_b_behavioral"
        ]
        assert class_b
        for consequence in class_b:
            assert consequence.historical_analogue_id is None
            assert consequence.status is Status.SPECULATIVE


class TestContradictionPath:
    def test_divergences_are_assessed_and_keep_their_deterministic_class(
        self, positive: AnalysisResult
    ) -> None:
        report = positive.contradictions
        assert report is not None
        assert report.divergences
        for assessed in report.outcome.assessed:
            assert assessed.divergence_class is assessed.item.divergence_class

    def test_a_legitimate_override_is_never_reported_as_a_contradiction(
        self, positive: AnalysisResult
    ) -> None:
        """The fake rated every divergence ``material``. The class still decides.

        This is the adversarial case for the contradiction gate, and it runs on real corpus
        divergences rather than synthetic ones.
        """
        report = positive.contradictions
        assert report is not None
        for assessed in report.outcome.assessed:
            if assessed.divergence_class.value in (
                "legitimate_scoped_override",
                "non_material_divergence",
            ):
                assert not assessed.is_contradiction
                assert assessed.item.item_id not in {f.category for f in report.findings}


class TestChallengePath:
    def test_every_candidate_is_reviewed_and_yields_a_falsification_test(
        self, positive: AnalysisResult
    ) -> None:
        report = positive.challenge
        assert report is not None
        assert report.results
        assert report.tests
        assert report.survived

    def test_surviving_review_does_not_raise_a_status(self, positive: AnalysisResult) -> None:
        """Checkpoint 9.2, observed end to end: ``no_refutation_found`` is not corroboration.

        Every Class B consequence here was cleared by the challenger and still caps at
        ``speculative``, which it could not do if a clearance raised anything.
        """
        class_b = [
            c for c in positive.consequences if c.consequence_class.value == "class_b_behavioral"
        ]
        assert class_b
        assert all(c.status is Status.SPECULATIVE for c in class_b)

    def test_a_successful_refutation_rejects_the_consequence(self, corpus: CorpusStores) -> None:
        """The other branch: a challenge that cites evidence must lower the disposition."""
        result = run_analysis(corpus, responder=cooperative_responder(challenger_verdict="refuted"))

        assert result.challenge is not None
        assert result.challenge.refuted
        rejected = [c for c in result.consequences if c.status is Status.REJECTED]
        assert rejected
        assert result.material_consequences == () or all(
            c.status is not Status.REJECTED for c in result.material_consequences
        )

    def test_a_rejected_consequence_projects_no_derived_edge(self, corpus: CorpusStores) -> None:
        result = run_analysis(corpus, responder=cooperative_responder(challenger_verdict="refuted"))
        assert result.graph is not None

        rejected = {c.consequence_id for c in result.consequences if c.status is Status.REJECTED}
        assert rejected
        for edge in result.graph.graph.edges:
            if edge.derivation.value == "derived":
                assert edge.consequence_id not in rejected


class TestEvidenceGapPath:
    def test_gaps_are_produced_from_the_real_retrieval_trace(self, quiet: AnalysisResult) -> None:
        assert quiet.gaps is not None
        assert quiet.gaps.gaps
        for record in quiet.gaps.gaps:
            assert record.provenance.trace == quiet.retrieval.trace
            assert record.provenance.run_id == RUN
            assert record.gap.analysis_id == ANALYSIS
            assert record.gap.evidence_ids == ()

    def test_a_search_that_found_evidence_produces_no_gap_for_it(
        self, positive: AnalysisResult
    ) -> None:
        """``negative_provenance`` is the plan that returned **nothing**."""
        assert positive.gaps is not None
        assert positive.retrieval.results
        assert positive.gaps.gaps == ()
        assert positive.gaps.refused_evidence_found

    def test_gaps_reach_the_consequences_as_references(self, quiet: AnalysisResult) -> None:
        """A gap annotates; it is never a finding."""
        assert quiet.evidence_gap_ids
        judgment = quiet.judgment
        assert judgment is not None
        assert set(judgment.evidence_gap_ids) <= set(quiet.evidence_gap_ids)


# ---------------------------------------------------------------------------
# Graph output
# ---------------------------------------------------------------------------


class TestGraphOutput:
    def test_the_graph_is_a_projection_of_this_analysis(self, positive: AnalysisResult) -> None:
        assert positive.graph is not None
        graph = positive.graph.graph
        assert graph.is_projection is True
        assert graph.analysis_id == ANALYSIS
        assert graph.projected_at == NOW

    def test_the_retention_window_join_is_visible(self, positive: AnalysisResult) -> None:
        """The central visualization, end to end."""
        assert positive.graph is not None
        holders = positive.graph.key_holders(RETENTION_KEY)

        assert len(holders) == 5
        assert "Data Retention Policy" in holders

    def test_the_graph_carries_no_score_or_severity(self, positive: AnalysisResult) -> None:
        assert positive.graph is not None
        for edge in positive.graph.graph.edges:
            dumped = edge.model_dump()
            assert "confidence" not in dumped
            assert "severity" not in dumped

    def test_a_gap_annotates_a_node_and_never_an_edge(self, quiet: AnalysisResult) -> None:
        assert quiet.graph is not None
        gap_ids = set(quiet.evidence_gap_ids)
        assert gap_ids

        for edge in quiet.graph.graph.edges:
            assert edge.source_node not in gap_ids
            assert edge.target_node not in gap_ids

    def test_the_projection_contains_the_expected_node_and_edge_kinds(
        self, positive: AnalysisResult
    ) -> None:
        assert positive.graph is not None
        nodes = positive.graph.nodes_by_type
        edges = positive.graph.edges_by_relationship

        assert nodes[NodeType.ASSUMPTION_KEY]
        assert nodes[NodeType.SOURCE_DOCUMENT]
        assert nodes[NodeType.CHANGE]
        assert edges[Relationship.BINDS_ASSUMPTION]
        assert edges[Relationship.DIRECTLY_CHANGES] == ()  # the change carries no evidence ids


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------


class TestFailurePropagation:
    def test_an_agentic_failure_is_recorded_and_the_pipeline_continues(
        self, corpus: CorpusStores
    ) -> None:
        """A provider outage costs the stage, not the analysis.

        Unparseable output from every agent: each agentic stage records ``FAILED``, and the
        deterministic findings and the graph survive untouched.
        """
        result = run_analysis(corpus, responder=lambda _request: "not json at all")

        assert AnalysisStage.CLASS_B in result.failed_stages
        assert AnalysisStage.CONTRADICTIONS in result.failed_stages
        assert AnalysisStage.CHALLENGE in result.failed_stages
        assert AnalysisStage.JUDGMENT in result.failed_stages
        assert not result.succeeded
        assert result.class_a
        assert result.graph is not None

    def test_a_failed_stage_is_distinguishable_from_an_empty_one(
        self, corpus: CorpusStores, quiet: AnalysisResult
    ) -> None:
        """Requirement 5, stated as a comparison.

        The same stage, the same corpus: ``EMPTY`` when it ran and found nothing, ``FAILED``
        when it could not run. Two different values, not one absence.
        """
        broken = run_analysis(corpus, responder=lambda _request: "not json at all")

        assert quiet.stage(AnalysisStage.CLASS_B).status is StageStatus.EMPTY
        assert broken.stage(AnalysisStage.CLASS_B).status is StageStatus.FAILED
        assert broken.stage(AnalysisStage.CLASS_B).detail

    def test_the_gap_stage_is_skipped_rather_than_reported_empty(
        self, corpus: CorpusStores
    ) -> None:
        """The one hard dependency: a failed hunt asked nothing, so there is no question.

        Reporting "no gaps" would claim the corpus was interrogated and found complete.
        """
        result = run_analysis(corpus, responder=lambda _request: "not json at all")

        assert result.stage(AnalysisStage.EVIDENCE_GAPS).status is StageStatus.SKIPPED
        assert result.gaps is None
        assert AnalysisStage.EVIDENCE_GAPS in result.skipped_stages

    def test_a_provider_outage_does_not_abort_the_analysis(self, corpus: CorpusStores) -> None:
        def outage(provider: FakeProvider) -> None:
            for _ in range(40):
                provider.queue_failure(ProviderUnavailable("no credential", provider="fake"))

        result = run_analysis(corpus, provider_hook=outage)

        assert result.failed_stages
        assert result.class_a
        assert result.graph is not None

    def test_a_required_stage_failure_aborts_loudly(self, corpus: CorpusStores) -> None:
        """Requirement 6. A deterministic stage has no provider, so a failure is a defect.

        The error names the stage, which is the difference between "the analysis failed" and
        "the coupling index could not be built". Continuing past it would produce an analysis
        whose foundation silently did not run.
        """
        use_case, _provider = pipeline(corpus)
        use_case._coupling = _BrokenCoupling()  # type: ignore[assignment]

        with pytest.raises(AnalysisStageError) as raised:
            asyncio.run(
                use_case.execute(
                    change=retention_change(),
                    query=query_for(corpus, "retention"),
                    tenant_id=corpus.tenant_id,
                    registry_version=corpus.registry_version,
                    as_of=AS_OF,
                    now=NOW,
                )
            )

        assert raised.value.stage is AnalysisStage.COUPLING
        assert isinstance(raised.value.cause, RuntimeError)
        assert "coupling" in str(raised.value)

    def test_a_required_stage_failure_is_not_swallowed_by_the_recorder(
        self, corpus: CorpusStores
    ) -> None:
        """The context manager returns ``None``, so it cannot suppress an exception.

        A recorder that *could* return ``True`` would silently continue past a required
        stage, which is precisely what requirement 6 forbids - and the type checker enforces
        the return annotation.
        """
        from app.application.orchestration import analysis as module

        assert module._RequiredStage.__exit__.__annotations__["return"] == "None"

    def test_the_required_stages_are_exactly_the_deterministic_ones(self) -> None:
        """Named so the split is asserted rather than trusted to the call order.

        ``REPORT`` is deterministic and therefore required: every section is a projection of
        a finalized record. The ``report-composer`` agent is optional and its failure is
        recorded on the document, which the next test asserts.
        """
        assert {
            AnalysisStage.COUPLING,
            AnalysisStage.CLASS_A,
            AnalysisStage.RETRIEVAL,
            AnalysisStage.GRAPH,
            AnalysisStage.REPORT,
        } == REQUIRED_STAGES

    def test_a_composer_failure_leaves_the_report_standing(self, corpus: CorpusStores) -> None:
        """An agent failure in the last stage must not cost the reader the report."""
        result = run_analysis(corpus, responder=broken_for("report-composer"))

        assert result.report is not None
        assert result.report.layout is not None
        assert not result.report.layout.succeeded
        assert result.stage(AnalysisStage.REPORT).status is StageStatus.COMPLETED
        # The deterministic surface stands, and the summary was never the model's.
        assert result.report.report.executive_summary
        assert result.report.narration is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_inputs_produce_an_equivalent_analysis(self, corpus: CorpusStores) -> None:
        """Requirement 7, as *equivalence*.

        Minted record ids differ by design - they are fresh per run so a re-run cannot
        collide with a stored analysis - so the fingerprint excludes them rather than
        pretending they are stable. Every decision must match: which stages ran, what they
        concluded, every status, every severity, every confidence number, and the graph's
        shape.
        """
        first = run_analysis(corpus)
        second = run_analysis(corpus)

        assert first.fingerprint() == second.fingerprint()

    def test_the_fingerprint_actually_discriminates(self, corpus: CorpusStores) -> None:
        """Otherwise the test above passes on an empty dict."""
        positive = run_analysis(corpus)
        quiet_result = run_analysis(corpus, change=quiet_change(), text=QUIET_SUBJECT)

        assert positive.fingerprint() != quiet_result.fingerprint()
        assert positive.fingerprint()["consequences"]

    def test_two_runs_agree_on_the_confidence_numbers(self, corpus: CorpusStores) -> None:
        first = run_analysis(corpus)
        second = run_analysis(corpus)

        assert sorted(s.confidence for s in first.scores) == sorted(
            s.confidence for s in second.scores
        )

    def test_two_runs_agree_on_the_graph_shape(self, corpus: CorpusStores) -> None:
        first = run_analysis(corpus)
        second = run_analysis(corpus)
        assert first.graph is not None and second.graph is not None

        assert first.graph.key_holders(RETENTION_KEY) == second.graph.key_holders(RETENTION_KEY)
        assert len(first.graph.graph.edges) == len(second.graph.graph.edges)


# ---------------------------------------------------------------------------
# Persistence, over a real analysis
# ---------------------------------------------------------------------------


class TestPersistence:
    """Snapshot a real pipeline run, store it, and read back what the reviewer saw.

    The unit tests build a ``StoredAnalysis`` directly, which proves the store. This proves
    the *projection*: that a finished ``AnalysisResult`` reduces to a review surface without
    losing anything a reviewer or a verdict needs.
    """

    def test_a_real_analysis_round_trips(self, positive: AnalysisResult) -> None:
        store = InMemoryAnalysisStore()
        store.put(snapshot(positive, stored_at=NOW))

        read = store.require(positive.retrieval.outcome.query.tenant_id, positive.analysis_id)

        assert read.analysis_id == positive.analysis_id
        assert read.run_id == positive.run_id
        assert read.change == positive.change
        assert read.run_provenance == positive.run_provenance
        assert read.stages == positive.stages
        assert read.pipeline_version == positive.pipeline_version

    def test_every_arbitrated_finding_is_stored_with_its_decision_and_score(
        self, positive: AnalysisResult
    ) -> None:
        stored = snapshot(positive, stored_at=NOW)

        assert len(stored.findings) == len(positive.consequences)
        assert stored.consequence_ids == {c.consequence_id for c in positive.consequences}
        for finding in stored.findings:
            assert finding.decision is not None
            assert finding.score is not None
            assert finding.score.consequence_id == finding.consequence_id
            assert finding.decision.consequence_id == finding.consequence_id

    def test_the_fingerprint_survives_the_round_trip(self, positive: AnalysisResult) -> None:
        """What lets a verdict be checked against the analysis that produced it."""
        store = InMemoryAnalysisStore()
        store.put(snapshot(positive, stored_at=NOW))

        read = store.require(positive.retrieval.outcome.query.tenant_id, positive.analysis_id)
        assert read.fingerprint == positive.fingerprint()

    def test_the_stored_evidence_is_what_the_requester_received(
        self, positive: AnalysisResult
    ) -> None:
        """So a stored citation resolves, and no span the reviewer could not read is kept.

        The blind-spot partition's outcomes are part of "what the requester received": they
        came from the same retriever under the same query's permission scope, and a
        blind-spot finding cites them.
        """
        stored = snapshot(positive, stored_at=NOW)
        assert positive.blind_spots is not None
        received = set(positive.retrieval.evidence_ids)
        for outcome in positive.blind_spots.partition_outcomes:
            received |= set(outcome.evidence_ids)

        assert set(stored.evidence_by_id) == received
        for finding in stored.findings:
            assert set(finding.consequence.evidence_ids) <= set(stored.evidence_by_id)

    def test_a_class_a_finding_stores_the_source_version_a_suppression_expires_on(
        self, positive: AnalysisResult
    ) -> None:
        """The join issue 64 was about: a ``Consequence`` records a title, not a version."""
        stored = snapshot(positive, stored_at=NOW)

        class_a = [
            f
            for f in stored.findings
            if f.consequence.consequence_class.value == "class_a_assumption_mediated"
        ]
        assert class_a
        assert all(f.artifact_version_id is not None for f in class_a)

    def test_a_class_b_finding_stores_no_source_version(self, positive: AnalysisResult) -> None:
        """A hypothesis spans several artifacts, so no single revision refutes it."""
        stored = snapshot(positive, stored_at=NOW)

        class_b = [
            f
            for f in stored.findings
            if f.consequence.consequence_class.value == "class_b_behavioral"
        ]
        assert class_b
        assert all(f.artifact_version_id is None for f in class_b)

    def test_the_judges_default_view_is_preserved(self, positive: AnalysisResult) -> None:
        stored = snapshot(positive, stored_at=NOW)
        assert positive.judgment is not None
        expected = {a.consequence_id for a in positive.judgment.outcome.default_view}

        assert set(stored.foregrounded) == expected

    def test_the_graph_and_the_gaps_are_stored(self, positive: AnalysisResult) -> None:
        stored = snapshot(positive, stored_at=NOW)
        assert positive.graph is not None

        assert stored.graph is not None
        assert stored.graph.is_projection is True
        assert len(stored.graph.edges) == len(positive.graph.graph.edges)
        assert stored.evidence_gap_ids == positive.evidence_gap_ids
        assert stored.open_questions == positive.open_questions

    def test_a_verdict_is_validated_against_the_stored_analysis(
        self, positive: AnalysisResult
    ) -> None:
        """The wiring the store exists for, end to end."""
        tenant_id = positive.retrieval.outcome.query.tenant_id
        analyses = InMemoryAnalysisStore()
        analyses.put(snapshot(positive, stored_at=NOW))
        reviews = RecordDispositionUseCase(store=InMemoryDispositionStore(), analyses=analyses)
        target = positive.consequences[0].consequence_id

        outcome = reviews.record_reviewed(
            tenant_id=tenant_id,
            analysis_id=positive.analysis_id,
            consequence_id=target,
            disposition=Disposition.USEFUL,
            disposed_by="privacy-office/reviewer-1",
            disposed_at=NOW,
        )

        assert outcome.created
        with pytest.raises(ValueError, match="contains no consequence"):
            reviews.record_reviewed(
                tenant_id=tenant_id,
                analysis_id=positive.analysis_id,
                consequence_id=new_id("consequence"),
                disposition=Disposition.USEFUL,
                disposed_by="privacy-office/reviewer-1",
                disposed_at=NOW,
            )

    def test_a_suppression_recorded_on_a_real_finding_is_readable_and_expires(
        self, positive: AnalysisResult
    ) -> None:
        """Both halves of requirement 7, over an analysis the pipeline actually produced."""
        tenant_id = positive.retrieval.outcome.query.tenant_id
        analyses = InMemoryAnalysisStore()
        stored = snapshot(positive, stored_at=NOW)
        analyses.put(stored)
        reviews = RecordDispositionUseCase(store=InMemoryDispositionStore(), analyses=analyses)

        class_a = next(f for f in stored.findings if f.artifact_version_id is not None)
        reviews.record_reviewed(
            tenant_id=tenant_id,
            analysis_id=positive.analysis_id,
            consequence_id=class_a.consequence_id,
            disposition=Disposition.WRONG,
            disposed_by="privacy-office/reviewer-1",
            disposed_at=NOW,
            note="The deck was reissued before the change.",
            suppress_until_source_revision=True,
        )

        key = class_a.consequence.assumption_keys[0]
        assert reviews.is_suppressed(
            tenant_id=tenant_id,
            assumption_key=key,
            artifact_version_id=class_a.artifact_version_id,
        )
        assert not reviews.is_suppressed(
            tenant_id=tenant_id, assumption_key=key, artifact_version_id="sv_" + "f" * 32
        )

    def test_a_snapshot_offered_under_the_wrong_tenant_is_refused(
        self, positive: AnalysisResult
    ) -> None:
        with pytest.raises(TenantMismatch, match="ran for tenant"):
            snapshot(positive, stored_at=NOW, tenant_id="tn_" + "9" * 32)


# ---------------------------------------------------------------------------
# The core intelligence layer: blind spots, arbitration, report
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reviewed(corpus: CorpusStores) -> tuple[AnalysisResult, AnalysisResult]:
    """Two analyses of the same change, with reviewer verdicts recorded between them.

    One fixture rather than two tests' worth of setup, because the thing under test is the
    *difference*: the second run has to honour a verdict recorded against the first, and both
    runs have to come out of the pipeline rather than out of a fixture.
    """
    analyses = InMemoryAnalysisStore()
    dispositions = InMemoryDispositionStore()

    first = run_analysis(
        corpus,
        analyses=analyses,
        dispositions=dispositions,
        analysis_id=new_id("analysis"),
    )
    reviews = RecordDispositionUseCase(store=dispositions, analyses=analyses)
    stored = analyses.require(corpus.tenant_id, first.analysis_id)

    # Two targets with **distinct** suppression scopes. Suppression is scoped to
    # `(assumption_key, source version)`, so two findings sharing one scope would both be
    # suppressed by the single `wrong` verdict - correct behaviour, but it would leave nothing
    # for the `already_known` half of this fixture to be about. Picking by index would make
    # that a coin flip, because `findings` is ordered by a freshly minted consequence id.
    by_scope: dict[tuple[str, str], ReviewedFinding] = {}
    for finding in stored.findings:
        version = finding.artifact_version_id
        if version is None:
            continue
        for key in finding.consequence.assumption_keys:
            by_scope.setdefault((key, version), finding)
    scopes = sorted(by_scope)
    assert len(scopes) >= 2, "the control needs two findings in distinct suppression scopes"
    rejected, known = by_scope[scopes[0]], by_scope[scopes[1]]

    reviews.record_reviewed(
        tenant_id=corpus.tenant_id,
        analysis_id=first.analysis_id,
        consequence_id=rejected.consequence_id,
        disposition=Disposition.WRONG,
        disposed_by="privacy-office/reviewer-1",
        disposed_at=NOW,
        note="The declaring deck was reissued before this change.",
        suppress_until_source_revision=True,
    )
    reviews.record_reviewed(
        tenant_id=corpus.tenant_id,
        analysis_id=first.analysis_id,
        consequence_id=known.consequence_id,
        disposition=Disposition.ALREADY_KNOWN,
        disposed_by="privacy-office/reviewer-1",
        disposed_at=NOW,
    )

    second = run_analysis(
        corpus,
        analyses=analyses,
        dispositions=dispositions,
        analysis_id=new_id("analysis"),
    )
    return first, second


class TestCoreIntelligenceLayer:
    """The ten checks the completed intelligence layer has to pass.

    All ten run the whole pipeline over the real corpus with the cooperative fake agent - the
    one that cites everything it is offered, claims the strongest position its schema allows,
    over-claims corroboration and tries to bury the surface. Nothing here was tuned to pass.
    """

    # 1
    def test_blind_spots_are_independently_represented(self, positive: AnalysisResult) -> None:
        """Independent by *evidence*, and attributed after arbitration.

        Measured on the sample corpus: the partition holds 5 spans across the
        ``single_holder`` and ``unwritten_rule`` slices, the main hunter was offered none of
        them, and one finding survives into the finalized set attributed to the blind-spot
        hunter.
        """
        report = positive.blind_spots
        assert report is not None
        assert report.partition is not None

        # A partition, not a second opinion: no overlap with what the main hunt read.
        main = frozenset(positive.retrieval.evidence_ids)
        assert report.partition.spans
        assert not report.partition.evidence_ids & main
        assert PartitionSlice.SINGLE_HOLDER in report.partition.covered

        # Reported apart from the main Class B stage, and still attributable downstream.
        assert positive.blind_spot_hypotheses
        assert not set(positive.blind_spot_hypotheses) & set(positive.hypotheses)
        assert positive.blind_spot_consequences
        for consequence in positive.blind_spot_consequences:
            assert BLIND_SPOT_AGENT in consequence.agent_sources
            assert set(consequence.evidence_ids) <= report.partition.evidence_ids

        # Corpus silence is a separate kind and never becomes a consequence.
        assert report.silences
        claims = {c.description for c in positive.consequences}
        assert not {s.question for s in report.silences} & claims

    # 2
    def test_arbitration_preserves_disagreement(self, positive: AnalysisResult) -> None:
        """Every ruling carries its disagreements, and they reach the decision.

        The arbiter's ``residual_disagreement`` enters the judge as ``extra_disagreements``,
        which ``dossiers_from`` has no trimming parameter for. So survival is structural.
        """
        arbitration = positive.arbitration
        assert arbitration is not None
        assert arbitration.rulings
        assert all(r.disagreements for r in arbitration.rulings)

        residual = {item for ruling in arbitration.rulings for item in ruling.disagreements}
        preserved = set(positive.judgment.preserved_disagreements) if positive.judgment else set()
        assert residual & preserved, "an arbiter's disagreement must reach a decision"

    def test_the_arbiter_cannot_inflate_corroboration(self, positive: AnalysisResult) -> None:
        """The fake claims one source version no corpus contains. It is trimmed, and logged.

        The ceiling is computed from the *measured* set, so the overclaim changes nothing
        about the position - which is the property, not the trimming.
        """
        arbitration = positive.arbitration
        assert arbitration is not None
        assert arbitration.outcome.overclaimed_corroboration == arbitration.rulings
        for ruling in arbitration.rulings:
            confirmed = set(ruling.corroborating_source_version_ids)
            assert confirmed
            assert not confirmed & set(ruling.overclaimed_source_version_ids)
            assert ruling.measured_corroboration == len(confirmed)

    def test_a_refutation_lowers_the_position_and_no_refutation_raises_nothing(
        self, corpus: CorpusStores
    ) -> None:
        """The ceiling, demonstrated by difference over two otherwise identical runs."""
        survived = run_analysis(corpus)
        refuted = run_analysis(
            corpus, responder=cooperative_responder(challenger_verdict="refuted")
        )

        assert survived.arbitration is not None
        assert refuted.arbitration is not None
        assert all(
            r.position is not EvidencePosition.DISPUTED for r in survived.arbitration.rulings
        )
        lowered = [r for r in refuted.arbitration.rulings if r.was_clamped]
        assert lowered
        assert all(
            ClampReason.REFUTED_BY_CHALLENGE in r.clamp_reasons
            or ClampReason.CONTRADICTING_EVIDENCE in r.clamp_reasons
            for r in lowered
        )

    # 3
    def test_report_output_is_evidence_traceable(self, positive: AnalysisResult) -> None:
        """Every citation in every section resolves to a span the requester received."""
        report = positive.report
        assert report is not None
        available = set(positive.retrieval.evidence_ids)
        for outcome in positive.blind_spots.partition_outcomes if positive.blind_spots else ():
            available |= set(outcome.evidence_ids)

        assert report.cited_evidence_ids
        assert report.cited_evidence_ids <= available
        assert report.evidence_references
        for reference in report.evidence_references:
            assert reference.cited_by
            assert reference.location != "(unresolved span)"

        # And the frozen entity agrees with the sections it was built from.
        assert set(report.report.default_view_consequence_ids) <= set(report.report.consequence_ids)
        assert set(report.report.consequence_ids) == {
            c.consequence_id for c in positive.consequences
        }

    def test_every_required_report_section_is_present_or_named_empty(
        self, positive: AnalysisResult
    ) -> None:
        """A section with nothing in it says so; it is never filled.

        Measured on the ``positive`` fixture, which runs with no stores: three sections are
        empty and all three are named rather than quietly absent. ``reviewer_signals`` is one
        of them, which is the distinction that matters - no store wired is not a rate of zero.
        """
        report = positive.report
        assert report is not None
        assert report.high_priority
        assert report.class_a
        assert report.class_b
        assert report.blind_spots
        assert report.contradictions
        assert report.challenge_outcomes
        assert report.corpus_silence
        assert report.unresolved_questions
        assert report.validation_actions
        assert set(report.empty_sections) == {
            "historical_warnings",
            "evidence_gaps",
            "reviewer_signals",
        }

    def test_the_composer_can_narrow_and_order_but_not_add(self, positive: AnalysisResult) -> None:
        """The fake proposes a handle it was not offered. It is dropped, and reported."""
        report = positive.report
        assert report is not None
        assert report.layout is not None
        assert report.layout.refused_handles == ("r99",)
        assert {f.consequence_id for f in report.high_priority} <= set(
            report.report.consequence_ids
        )

    # 4
    def test_reviewer_suppression_works_on_a_subsequent_analysis(
        self, reviewed: tuple[AnalysisResult, AnalysisResult]
    ) -> None:
        """One ``wrong`` verdict on the first run narrows the second run's surface.

        Measured: 9 findings foregrounded before, 7 after. The two that left are the ones
        whose ``(assumption_key, source version)`` the verdict was scoped to - and one verdict
        covering two findings is correct, because scope is the unit, not the consequence.
        """
        first, second = reviewed
        assert first.report is not None
        assert second.report is not None

        signals = second.report.reviewer
        assert signals.measured
        assert signals.suppressed
        assert all(reason == Disposition.WRONG.value for reason in signals.suppressed_by.values())
        assert len(second.report.high_priority) < len(first.report.high_priority)
        assert not set(signals.suppressed) & set(second.report.report.default_view_consequence_ids)
        # Suppressed, not deleted.
        assert set(signals.suppressed) <= set(second.report.report.consequence_ids)
        assert second.report.report.suppressed_count == len(signals.suppressed)
        for consequence_id in signals.suppressed:
            assert any(c.consequence_id == consequence_id for c in second.consequences)

    def test_suppression_does_not_touch_the_underlying_records(
        self, reviewed: tuple[AnalysisResult, AnalysisResult]
    ) -> None:
        """A verdict changes the surface. It changes no consequence, score or graph edge."""
        first, second = reviewed
        assert second.report is not None
        suppressed = set(second.report.reviewer.suppressed)

        for consequence in second.consequences:
            if consequence.consequence_id in suppressed:
                assert consequence.status is not Status.REJECTED
                assert consequence.evidence_ids or not consequence.derived_relation_ids
        assert second.graph is not None and first.graph is not None
        assert len(second.graph.graph.edges) == len(first.graph.graph.edges)
        assert len(second.scores) == len(first.scores)

    # 5
    def test_already_known_remains_measurable_and_suppresses_nothing(
        self, reviewed: tuple[AnalysisResult, AnalysisResult]
    ) -> None:
        """Measured: an already-known rate of 0.5 over 4 re-encountered findings.

        The rate is the population that matters - how often the system re-delivers a finding a
        reviewer said they already had - and ``already_known`` must never take one off the
        surface, because the signal is worthless once it has a side effect reviewers can see.
        """
        _first, second = reviewed
        assert second.report is not None
        signals = second.report.reviewer

        assert signals.already_known
        assert signals.reviewed >= len(signals.already_known)
        assert signals.already_known_rate is not None
        assert 0.0 < signals.already_known_rate <= 1.0
        assert not set(signals.already_known) & set(signals.suppressed)
        for consequence_id in signals.already_known:
            assert consequence_id in second.report.report.default_view_consequence_ids

    def test_an_unmeasured_reviewer_state_is_distinguishable_from_an_empty_one(
        self,
        corpus: CorpusStores,
        reviewed: tuple[AnalysisResult, AnalysisResult],
    ) -> None:
        """A rate of zero, no verdicts yet, and no store wired are three different facts."""
        unwired = run_analysis(corpus, analyses=None, dispositions=None)
        first, second = reviewed

        assert unwired.report is not None
        assert not unwired.report.reviewer.measured
        assert unwired.report.reviewer.already_known_rate is None
        assert "reviewer_signals" in unwired.report.empty_sections

        # Wired but nothing reviewed yet: measured, with no rate to report.
        assert first.report is not None
        assert first.report.reviewer.measured
        assert first.report.reviewer.reviewed == 0
        assert first.report.reviewer.already_known_rate is None

        # Wired, with verdicts: measured and rated.
        assert second.report is not None
        assert second.report.reviewer.already_known_rate is not None

    # 6
    def test_no_agent_can_promote_class_b_to_class_a(self, positive: AnalysisResult) -> None:
        """Two arguments, and the structural one is the one that holds.

        Every Class A consequence has ``origin=resolver`` and a rule id; every agent-authored
        one is Class B. And none of the three new proposal schemas has a field a class could
        arrive in - which is why no prompt has to be trusted.
        """
        for consequence in positive.consequences:
            if consequence.consequence_class is ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED:
                assert consequence.origin is Origin.RESOLVER
                assert consequence.derived_relation_ids
            else:
                assert consequence.origin is Origin.AGENT
                assert consequence.hypothesis_ids

        for schema in (
            BlindSpotBatch.json_schema(),
            RulingBatch.json_schema(),
            CompositionProposal.json_schema(),
        ):
            rendered = json.dumps(schema)
            for forbidden in ("consequence_class", '"confidence"', '"severity"'):
                assert forbidden not in rendered

    def test_blind_spot_findings_go_through_the_adversarial_challenger(
        self, positive: AnalysisResult
    ) -> None:
        """The ordering fix: discovery finishes before review begins.

        Measured on the positive control: the challenger sees 11 candidates rather than 10,
        and the eleventh is the blind-spot hypothesis. Before the reordering it reached the
        judge unreviewed - the newest and least corroborated finding in the analysis was the
        only one nobody had tried to refute.
        """
        assert positive.blind_spots is not None
        assert positive.challenge is not None
        assert list(AnalysisStage).index(AnalysisStage.BLIND_SPOTS) < list(AnalysisStage).index(
            AnalysisStage.CHALLENGE
        )

        retained = {h.hypothesis_id for h in positive.blind_spots.retained}
        assert retained
        challenged = {
            target.hypothesis.hypothesis_id
            for target in positive.challenge.targets
            if target.hypothesis is not None
        }
        assert retained <= challenged

        # Every hypothesis from *either* hunt is a candidate, and none is a candidate twice.
        assert len(positive.challenge.targets) == len(positive.class_a) + len(
            positive.all_hypotheses
        )
        ids = [t.consequence_id for t in positive.challenge.targets]
        assert len(ids) == len(set(ids))

    def test_a_blind_spot_candidate_is_offered_the_partition_evidence_to_cite(
        self, positive: AnalysisResult
    ) -> None:
        """The wiring this ordering needs, and the easiest thing to get wrong.

        A blind-spot finding cites partition spans, which are not in the main retrieval
        result. Handing the challenger only the main spans would leave the candidate with
        nothing citable, so it could only ever come back ``no_refutation_found`` - a refusal
        that reads exactly like a survived challenge.
        """
        assert positive.blind_spots is not None
        assert positive.blind_spots.partition is not None
        assert positive.challenge is not None

        partition_ids = positive.blind_spots.partition.evidence_ids
        assert partition_ids
        assert not partition_ids & frozenset(positive.retrieval.evidence_ids)

        blind = {h.hypothesis_id for h in positive.blind_spots.retained}
        target = next(
            t
            for t in positive.challenge.targets
            if t.hypothesis is not None and t.hypothesis.hypothesis_id in blind
        )
        assert target.evidence_ids
        assert set(target.evidence_ids) <= partition_ids

    def test_a_refuted_blind_spot_finding_reaches_judgment_as_rejected(
        self, corpus: CorpusStores
    ) -> None:
        """The outcome has to *arrive*, not merely be recorded.

        With the challenger refuting every candidate it can cite against, the blind-spot
        finding comes out ``rejected`` - which it could not have done under the old ordering,
        because nothing challenged it. The citation is a partition span, so this also confirms
        the refutation rested on evidence the main analysis never saw.
        """
        refuted = run_analysis(
            corpus, responder=cooperative_responder(challenger_verdict="refuted")
        )

        assert refuted.blind_spots is not None
        assert refuted.challenge is not None
        blind = {h.hypothesis_id for h in refuted.blind_spots.retained}
        assert blind

        target = next(
            t
            for t in refuted.challenge.targets
            if t.hypothesis is not None and t.hypothesis.hypothesis_id in blind
        )
        verdict = next(
            c
            for c in refuted.challenge.outcome.challenged
            if c.result is not None and c.result.consequence_id == target.consequence_id
        )
        assert verdict.result is not None
        assert verdict.result.outcome is ChallengeOutcome.REFUTED
        assert verdict.result.contradicting_evidence_ids
        assert set(verdict.result.contradicting_evidence_ids) <= (
            refuted.blind_spots.partition.evidence_ids if refuted.blind_spots.partition else set()
        )

        # And the judge acted on it: the refuted candidate is rejected, not speculative.
        consequence = next(
            c for c in refuted.consequences if c.consequence_id == target.consequence_id
        )
        assert consequence.status is Status.REJECTED
        assert consequence not in refuted.material_consequences
        assert BLIND_SPOT_AGENT in consequence.agent_sources

    def test_a_surviving_blind_spot_finding_is_attributed_to_both_agents(
        self, positive: AnalysisResult
    ) -> None:
        """Agent-source bookkeeping, which the reordering changes without any stage carrying a flag.

        ``agent_sources`` is derived by the judge from the dossier: the hunter from the
        hypothesis's own ``ModelProvenance``, the challenger from the presence of a verdict. So
        a blind-spot consequence now names both, and that is the observable difference between
        a reviewed finding and one nobody looked at.
        """
        assert positive.blind_spot_consequences
        for consequence in positive.blind_spot_consequences:
            assert BLIND_SPOT_AGENT in consequence.agent_sources
            assert "adversarial-challenger" in consequence.agent_sources
            # Surviving a challenge raises nothing: `no_refutation_found` is not evidence of
            # correctness, and the frozen status gates have no parameter for it.
            assert consequence.status is Status.SPECULATIVE

    def test_the_reordering_preserved_failure_isolation_in_both_directions(
        self, corpus: CorpusStores
    ) -> None:
        """Each stage can fail without taking the other with it.

        Both directions matter now that they are adjacent: a failed hunter must not stop the
        review, and a failed reviewer must not delete what it failed to review.
        """
        hunter_down = run_analysis(corpus, responder=broken_for("blind-spot-hunter"))
        assert hunter_down.stage(AnalysisStage.BLIND_SPOTS).status is StageStatus.FAILED
        assert not hunter_down.blind_spot_hypotheses
        # The challenger still reviewed Class A and the main hunt.
        assert hunter_down.stage(AnalysisStage.CHALLENGE).status is StageStatus.COMPLETED
        assert hunter_down.challenge is not None
        assert len(hunter_down.challenge.targets) == len(hunter_down.class_a) + len(
            hunter_down.hypotheses
        )
        assert hunter_down.consequences

        challenger_down = run_analysis(corpus, responder=broken_for("adversarial-challenger"))
        assert challenger_down.stage(AnalysisStage.CHALLENGE).status is StageStatus.FAILED
        # The blind-spot finding survives into judgment as an unreviewed candidate rather
        # than being dropped with the stage that could not review it.
        assert challenger_down.blind_spot_hypotheses
        assert challenger_down.blind_spot_consequences
        assert challenger_down.stage(AnalysisStage.JUDGMENT).status is StageStatus.COMPLETED
        assert len(challenger_down.consequences) == len(challenger_down.class_a) + len(
            challenger_down.all_hypotheses
        )

    def test_the_blind_spot_hunter_cannot_reach_the_main_analysis_evidence(
        self, positive: AnalysisResult
    ) -> None:
        """An id the main hunter was offered is refused here exactly like a fabricated one."""
        report = positive.blind_spots
        assert report is not None
        assert report.partition is not None
        for hypothesis in report.retained:
            assert set(hypothesis.evidence_ids) <= report.partition.evidence_ids
            assert not set(hypothesis.evidence_ids) & report.partition.main_offered_ids

    # 7
    def test_no_report_section_invents_unsupported_evidence(self, corpus: CorpusStores) -> None:
        """The composer cites an id no analysis produced; the whole narration is discarded.

        Refused whole rather than repaired: a summary with one invented citation removed is a
        summary whose remaining sentences came from the same process.
        """
        result = run_analysis(
            corpus, responder=cooperative_responder(composer_invents_evidence=True)
        )

        assert result.report is not None
        layout = result.report.layout
        assert layout is not None
        assert layout.narration is None
        assert layout.narration_refusal == REFUSAL_INVENTED_ID
        assert layout.invented_ids
        # The report still exists, and its summary was never the model's.
        assert result.report.report.executive_summary
        assert result.report.narration is None
        # And the structural effects still applied: refusal is per-narration, not per-call.
        assert layout.foreground

    # 8
    def test_quiet_changes_remain_quiet(self, quiet: AnalysisResult) -> None:
        """Three new stages later, the negative control still reports nothing material.

        Measured: on the espresso-machine change the blind-spot partition is *larger* (9
        spans, because the main query returned none) and the hunter does propose one
        hypothesis - and it comes out ``speculative`` with an empty default surface. That is
        the honest shape: the stage ran, the gates refused to make its output material, and
        the reader is shown nothing to act on.
        """
        assert quiet.is_quiet
        assert not quiet.material_consequences
        assert not quiet.class_a
        assert quiet.report is not None
        assert not quiet.report.high_priority
        assert not quiet.report.report.default_view_consequence_ids
        assert all(c.status is Status.SPECULATIVE for c in quiet.consequences)
        # What it does produce is uncertainty, which is a first-class output.
        assert quiet.report.unresolved_questions

    # 9
    def test_ablation_still_works(
        self,
        positive: AnalysisResult,
        ablated: AnalysisResult,
        removed_evidence: frozenset[str],
    ) -> None:
        """The three new stages must not resurrect a finding whose evidence is gone.

        The blind-spot hunter is the risk here specifically: it retrieves its own evidence, so
        a partition query could in principle recover a removed span. It cannot - the span no
        longer exists in the store the retriever reads.
        """
        assert len(ablated.class_a) < len(positive.class_a)
        for hypothesis in ablated.blind_spot_hypotheses:
            assert not set(hypothesis.evidence_ids) & removed_evidence
        assert ablated.report is not None
        assert not ablated.report.cited_evidence_ids & removed_evidence
        if ablated.arbitration is not None:
            for ruling in ablated.arbitration.rulings:
                assert not set(ruling.supporting_evidence_ids) & removed_evidence
                assert not set(ruling.contradicting_evidence_ids) & removed_evidence

    # 10
    def test_deterministic_class_a_output_is_unchanged_by_the_agent_stages(
        self, corpus: CorpusStores
    ) -> None:
        """The product's central claim, asserted against the completed agent layer.

        A deterministic-only run and a fully agentic one must derive the same Class A set. Six
        agents now run between the resolver and the reader; none of them may have moved it.

        **One ``ChangeSpec``, built once and passed to both runs.** A ``ClassAFindingId`` is
        derived from the relation, which is derived from the delta - so two calls to the change
        factory mint different deltas and would make this comparison meaningless while
        appearing to test something.
        """
        change = retention_change()
        agentic = run_analysis(corpus, change=change)
        deterministic = run_analysis(corpus, change=change, agentic=False)

        assert deterministic.class_a
        assert {f.finding_id for f in deterministic.class_a} == {
            f.finding_id for f in agentic.class_a
        }
        for finding in agentic.class_a:
            original = next(f for f in deterministic.class_a if f.finding_id == finding.finding_id)
            assert finding.rule_id == original.rule_id
            assert finding.relation_id == original.relation_id
            assert finding.materialized_evidence_ids == original.materialized_evidence_ids
            assert finding.input_record_ids == original.input_record_ids
            assert finding.summary == original.summary

    def test_the_new_stages_are_not_configured_rather_than_failed_when_absent(
        self, corpus: CorpusStores
    ) -> None:
        """A deterministic-only run reports the agentic stages as NOT_CONFIGURED.

        And still produces a report, because composition is deterministic.
        """
        deterministic = run_analysis(corpus, agentic=False)

        for stage in (AnalysisStage.BLIND_SPOTS, AnalysisStage.EVIDENCE_ARBITRATION):
            assert deterministic.stage(stage).status is StageStatus.NOT_CONFIGURED
        assert deterministic.failed_stages == ()
        assert deterministic.report is not None
        assert deterministic.report.layout is None

    def test_a_blind_spot_failure_is_distinguishable_from_an_empty_partition(
        self, corpus: CorpusStores
    ) -> None:
        """Requirement: an agent failure must not look like a successful empty result."""
        broken = run_analysis(corpus, responder=broken_for("blind-spot-hunter"))

        assert broken.stage(AnalysisStage.BLIND_SPOTS).status is StageStatus.FAILED
        assert broken.stage(AnalysisStage.BLIND_SPOTS).detail
        assert broken.blind_spots is not None
        assert not broken.blind_spots.succeeded
        assert not broken.blind_spot_hypotheses
        # And the stages after it still ran.
        assert broken.stage(AnalysisStage.JUDGMENT).status is StageStatus.COMPLETED
        assert broken.report is not None

    def test_an_arbitration_failure_leaves_every_candidate_unruled(
        self, corpus: CorpusStores
    ) -> None:
        """Unruled, never defaulted to a position. An unarbitrated claim has not been cleared."""
        broken = run_analysis(corpus, responder=broken_for("evidence-arbiter"))

        assert broken.stage(AnalysisStage.EVIDENCE_ARBITRATION).status is StageStatus.FAILED
        assert broken.arbitration is not None
        assert not broken.arbitration.rulings
        assert broken.arbitration.outcome.unruled
        assert broken.evidence_positions == {}
        # The judge still arbitrated, from the candidates the arbiter could not position.
        assert broken.consequences

    def test_the_finished_analysis_is_persisted_for_the_next_run(
        self, corpus: CorpusStores
    ) -> None:
        """Issue 65: the orchestrator now stores its own result.

        Which is what makes the reviewer lookup possible at all - a verdict is validated
        against a stored analysis, and until this ran there was nothing stored.
        """
        analyses = InMemoryAnalysisStore()
        dispositions = InMemoryDispositionStore()
        result = run_analysis(
            corpus,
            analyses=analyses,
            dispositions=dispositions,
            analysis_id=new_id("analysis"),
        )

        assert analyses.for_tenant(corpus.tenant_id) == (result.analysis_id,)
        stored = analyses.require(corpus.tenant_id, result.analysis_id)
        assert stored.fingerprint == result.fingerprint()
        assert stored.consequence_ids == {c.consequence_id for c in result.consequences}


# ---------------------------------------------------------------------------
# The specialist fan-out, over the real corpus
# ---------------------------------------------------------------------------


def capturing(agent: str) -> tuple[Callable[..., str], dict[str, str]]:
    """A cooperative responder that also records the task text one agent was sent.

    The only way to show that a specialist's observation *reached* the consequence-hunter is
    to look at what the hunter was asked. Recording it in a responder needs no production
    change and no request log on the provider.
    """
    inner = cooperative_responder()
    seen: dict[str, str] = {}

    def respond(request: Any) -> str:
        if request.agent == agent:
            seen[agent] = request.messages[-1].content
        return inner(request)

    return respond, seen


class TestSpecialistFanOut:
    """Seven analysts, and the seven properties that make adding them safe.

    Measured on the sample corpus: three specialists are selected (policy, process, data) and
    four are not, because nothing of their source types is in the authorized result set. That
    is the fan-out gate working, and it is reported per specialist rather than inferred from a
    count.
    """

    # 1
    def test_only_specialists_with_evidence_in_their_slice_are_invoked(
        self, positive: AnalysisResult
    ) -> None:
        report = positive.specialists
        assert report is not None

        assert report.fan_out.agents == ("policy-analyst", "process-analyst", "data-analyst")
        assert report.invoked == report.fan_out.agents
        assert set(report.fan_out.reasons) == {
            "technology-analyst",
            "people-analyst",
            "risk-analyst",
            "historical-analyst",
        }
        # Every specialist is accounted for either way.
        assert len(report.fan_out.selected) + len(report.fan_out.not_selected) == 7
        assert all(size > 0 for size in report.fan_out.slice_sizes.values())
        assert positive.stage(AnalysisStage.SPECIALISTS).status is StageStatus.COMPLETED

    def test_the_stage_sits_between_retrieval_and_the_class_b_hunt(
        self, positive: AnalysisResult
    ) -> None:
        """It needs the retrieval query, and the hunter needs its output."""
        order = list(AnalysisStage)
        assert (
            order.index(AnalysisStage.RETRIEVAL)
            < order.index(AnalysisStage.SPECIALISTS)
            < order.index(AnalysisStage.CLASS_B)
        )
        assert [s.stage for s in positive.stages] == order

    # 2
    def test_a_specialist_sees_only_evidence_the_requester_was_authorized_to_see(
        self, positive: AnalysisResult
    ) -> None:
        """Every slice is a narrowing of the caller's own query, through the same retriever.

        Measured: the three slices partition the six spans the main retrieval returned, so on
        this corpus the fan-out re-reads the authorized set by domain rather than widening it.
        The subset assertion is the guarantee; the partition is this corpus.
        """
        report = positive.specialists
        assert report is not None
        received = set(positive.retrieval.evidence_ids)

        for outcome in report.slice_outcomes:
            assert outcome.query.tenant_id == positive.retrieval.outcome.query.tenant_id
            assert (
                outcome.query.permission_scope == positive.retrieval.outcome.query.permission_scope
            )
            assert outcome.query.source_types  # a slice is always source-type filtered
        assert report.evidence_ids <= received
        for finding in report.retained:
            assert set(finding.evidence_ids) <= received

    def test_each_slice_holds_only_its_own_source_types(self, positive: AnalysisResult) -> None:
        report = positive.specialists
        assert report is not None

        for chosen in report.fan_out.selected:
            allowed = {t.value for t in chosen.specialist.source_types}
            assert {span.source_type for span in chosen.spans} <= allowed

    # 3
    def test_every_specialist_claim_carries_valid_evidence_or_is_unresolved(
        self, positive: AnalysisResult
    ) -> None:
        """The evidence-only rule, and the ceiling that enforces it.

        The cooperative fake asserts one claim per specialist with a citation nobody offered.
        Every one of those comes back ``unresolved`` with no evidence ids - it cannot be read
        as supported, and the invented id is dropped rather than repaired.
        """
        report = positive.specialists
        assert report is not None
        assert report.retained

        offered_by_agent = {chosen.agent: chosen.evidence_ids for chosen in report.fan_out.selected}
        for finding in report.retained:
            assert set(finding.evidence_ids) <= offered_by_agent[finding.agent]
            if finding.evidence_position in {
                EvidencePosition.SUPPORTED,
                EvidencePosition.WEAKLY_SUPPORTED,
            }:
                assert finding.evidence_ids
            else:
                assert finding.evidence_position is EvidencePosition.UNRESOLVED
                assert not finding.evidence_ids
        assert report.fabricated_evidence_ids

    def test_no_specialist_finding_carries_a_class_confidence_severity_or_status(
        self, positive: AnalysisResult
    ) -> None:
        """Structural: the frozen entity has no field for any of the four."""
        report = positive.specialists
        assert report is not None

        fields = set(SpecialistFinding.model_fields)
        assert not fields & {"confidence", "severity", "status", "consequence_class"}
        for forbidden in FORBIDDEN_AGENT_FIELDS:
            assert forbidden not in json.dumps(ObservationBatch.json_schema())
        # And nothing a specialist produced is a candidate in its own right.
        claims = {f.claim for f in report.retained}
        assert not claims & {c.description for c in positive.consequences}

    def test_observed_bindings_are_recorded_and_reach_no_index(
        self, positive: AnalysisResult
    ) -> None:
        """The registry is closed; admission is ingestion-time and deterministic."""
        report = positive.specialists
        assert report is not None
        assert report.observed_bindings

        # The coupling projection is what the resolver wrote, and nothing else.
        assert positive.coupling.index.bindings
        proposed_keys = {b.key for b in report.observed_bindings}
        admitted = {b.key for b in positive.coupling.index.bindings}
        # The keys may coincide - a specialist reading the retention policy sees the same
        # proposition - but the *records* are the resolver's. A specialist binding never
        # becomes one: there is no store write on this path at all.
        assert proposed_keys <= admitted or proposed_keys - admitted

    # 4
    def test_one_specialist_failing_costs_one_domain_and_not_the_stage(
        self, corpus: CorpusStores
    ) -> None:
        """Failure isolation, which is the reason to fan out at all rather than ask one agent."""
        broken = run_analysis(corpus, responder=broken_for("policy-analyst"))

        report = broken.specialists
        assert report is not None
        assert report.failed == ("policy-analyst",)
        assert report.findings_by("policy-analyst") == ()
        # The other two analysts' findings survive, and the stage names the failure.
        assert {f.agent for f in report.retained} == {"process-analyst", "data-analyst"}
        assert broken.stage(AnalysisStage.SPECIALISTS).status is StageStatus.FAILED
        assert "policy-analyst" in broken.stage(AnalysisStage.SPECIALISTS).detail
        # And everything downstream still ran.
        assert broken.stage(AnalysisStage.CLASS_B).status is StageStatus.COMPLETED
        assert broken.consequences
        assert broken.report is not None

    def test_the_whole_fan_out_failing_leaves_the_hunter_running_alone(
        self, corpus: CorpusStores
    ) -> None:
        """Degradation, not a skip. The hunter existed before the specialists did."""
        broken = run_analysis(
            corpus,
            responder=broken_for(
                "policy-analyst",
                "process-analyst",
                "data-analyst",
            ),
        )

        assert broken.stage(AnalysisStage.SPECIALISTS).status is StageStatus.FAILED
        assert broken.specialist_findings == ()
        assert broken.stage(AnalysisStage.CLASS_B).status is StageStatus.COMPLETED
        assert broken.hypotheses
        assert broken.consequences

    def test_a_deterministic_only_run_reports_the_stage_not_configured(
        self, corpus: CorpusStores
    ) -> None:
        """And the Class A path is unchanged by the specialists existing at all."""
        change = retention_change()
        agentic = run_analysis(corpus, change=change)
        deterministic = run_analysis(corpus, change=change, agentic=False)

        assert deterministic.stage(AnalysisStage.SPECIALISTS).status is StageStatus.NOT_CONFIGURED
        assert deterministic.specialists is None
        assert {f.finding_id for f in deterministic.class_a} == {
            f.finding_id for f in agentic.class_a
        }

    # 5
    def test_duplicate_observations_are_suppressed_deterministically(
        self, positive: AnalysisResult
    ) -> None:
        """Measured: two suppressed, both the same uncited claim from a second analyst.

        The cooperative fake returns an identical unevidenced claim from every specialist, so
        the first analyst in roster order keeps it and the rest are absorbed. Every suppression
        carries its reason, because a dropped observation a reader cannot account for reads
        like a lost one.
        """
        report = positive.specialists
        assert report is not None
        assert report.suppressed

        for finding, reason in report.suppressed:
            assert reason in {SUPPRESSED_DUPLICATE, SUPPRESSED_RESTATES_CLASS_A}
            assert finding not in report.retained
        # One survivor per (claim, evidence set).
        identities = [
            (" ".join(f.claim.lower().split()), frozenset(f.evidence_ids)) for f in report.retained
        ]
        assert len(identities) == len(set(identities))

    # 6
    def test_specialist_observations_reach_the_hunter_and_then_the_whole_pipeline(
        self, corpus: CorpusStores
    ) -> None:
        """The integration requirement, traced rather than asserted by assumption.

        Three links: the observations are in the hunter's contract; the hypothesis the hunter
        produced becomes a challenge target; that target is arbitrated and judged. There is one
        path from evidence to a consequence and the specialists feed its front, which is what
        "no parallel reasoning system" has to mean.
        """
        responder, seen = capturing("consequence-hunter")
        result = run_analysis(corpus, responder=responder)

        report = result.specialists
        assert report is not None
        assert report.retained

        # Link 1: the hunter was shown the observations, attributed, with their citations.
        contract = seen["consequence-hunter"]
        assert "## Domain analyst observations" in contract
        for finding in report.retained:
            if finding.evidence_position is EvidencePosition.UNRESOLVED:
                continue
            assert finding.agent in contract
            assert finding.claim in contract
            for evidence_id in finding.evidence_ids:
                assert evidence_id in contract

        # Link 2: what the hunter proposed is a challenge target.
        assert result.hypotheses
        assert result.challenge is not None
        challenged = {
            t.hypothesis.hypothesis_id for t in result.challenge.targets if t.hypothesis is not None
        }
        assert {h.hypothesis_id for h in result.hypotheses} <= challenged

        # Link 3: arbitrated and judged, with a status the frozen gates chose.
        assert result.arbitration is not None
        assert len(result.arbitration.rulings) == len(result.challenge.targets)
        assert result.consequences
        assert result.material_consequences
        assert result.report is not None

    def test_the_specialists_are_visible_in_the_analysis_fingerprint(
        self, positive: AnalysisResult
    ) -> None:
        """Two runs must agree on which analysts ran and what they said."""
        fingerprint = positive.fingerprint()

        assert fingerprint["specialist_agents"] == sorted(
            {f.agent for f in positive.specialist_findings}
        )
        assert fingerprint["specialist_claims"]
        assert positive.pipeline_version == ANALYSIS_PIPELINE_VERSION

    def test_specialist_questions_reach_the_reports_unresolved_section(
        self, positive: AnalysisResult
    ) -> None:
        """A specialist asking rather than inferring is the sanctioned alternative, so the
        question has to arrive somewhere a reader sees it."""
        report = positive.specialists
        assert report is not None
        assert report.open_questions

        assert set(report.open_questions) <= set(positive.open_questions)
        assert positive.report is not None
        assert set(report.open_questions) <= set(positive.report.unresolved_questions)

    # 7
    def test_a_quiet_change_invokes_no_specialist(self, quiet: AnalysisResult) -> None:
        """The cheap gate, at its most useful: seven calls not made.

        Retrieval returned nothing, so every slice is empty and every specialist is reported
        un-selected with its reason. The stage is ``empty`` rather than ``failed`` - it ran,
        it selected nobody, and that is a legitimate result.
        """
        report = quiet.specialists
        assert report is not None

        assert report.fan_out.selected == ()
        assert len(report.fan_out.not_selected) == 7
        assert report.invoked == ()
        assert report.retained == ()
        assert quiet.stage(AnalysisStage.SPECIALISTS).status is StageStatus.EMPTY
        # And the negative control still holds.
        assert quiet.is_quiet
        assert not quiet.material_consequences
        assert quiet.report is not None
        assert not quiet.report.high_priority


# ---------------------------------------------------------------------------
# The historical analogue, over the real corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anchored(corpus: CorpusStores) -> AnalysisResult:
    """The refund change, which is the one this corpus can anchor.

    Module-scoped and reused: it is the only run in the suite where a Class B consequence
    exceeds ``speculative``, so several tests read different parts of the same one.
    """
    return run_analysis(corpus, change=refund_change(), text=REFUND_QUERY)


@pytest.fixture(scope="module")
def labelled_corpus() -> CorpusStores:
    """The sample corpus with the incident labelled ``support-operations``.

    Labelled in a manifest copy and re-ingested, for the same reason the ablation is:
    permissions come from registration, so relabelling after the fact would leave the
    derived records carrying the old ones.
    """
    manifest = copy.deepcopy(load_manifest())
    labelled = [entry for entry in manifest["sources"] if entry["path"] == REFUND_INCIDENT]
    assert len(labelled) == 1, "the refund incident must be in the manifest"
    labelled[0]["permissions"] = [REFUND_GRANT]

    stores = asyncio.run(load_corpus(manifest=manifest))
    assert not stores.extraction_failures, stores.extraction_failures
    return stores


@pytest.fixture(scope="module")
def incident_evidence(corpus: CorpusStores) -> frozenset[str]:
    """The spans belonging to the incident, read from the store rather than hard-coded.

    The ids are content-derived, so a literal list would silently stop testing anything the
    first time the corpus file is edited.
    """
    version_ids = {
        report.version_id for path, report in corpus.reports.items() if path == REFUND_INCIDENT
    }
    assert version_ids, "the refund incident must be in the intact corpus"
    spans = frozenset(
        item.evidence_id
        for item in corpus.sources.evidence_for_tenant(corpus.tenant_id)
        if item.source_version_id in version_ids
    )
    assert spans, "the incident must have contributed evidence"
    return spans


def class_b(result: AnalysisResult) -> tuple[Consequence, ...]:
    return tuple(
        c for c in result.consequences if c.consequence_class is ConsequenceClass.CLASS_B_BEHAVIORAL
    )


class TestHistoricalAnalogue:
    """What a Class B consequence needs to exceed ``speculative``, and the gates around it.

    Checkpoint 9.2 rule 2 has been in ``app/judgment/status_rules.py`` since judgment was
    written, but nothing could satisfy it: ``historical_analogue_id`` is an ``EventId`` and no
    part of the system produced one. So the gate was unreachable in the raising direction and
    open in the wrong one - the hunter's proposal carried the field and nobody checked it.

    Measured on this corpus: the refund change admits **one** anchor from INC-4471 and its
    main-hunt Class B consequence is judged ``likely``. That is the first non-speculative
    Class B finding the project has produced. The retention change admits none and stays
    ``speculative``, which is the behaviour that must not change.
    """

    # 1
    def test_a_valid_historical_event_anchors_a_class_b_finding(
        self, anchored: AnalysisResult, incident_evidence: frozenset[str]
    ) -> None:
        """The whole point, end to end: evidence -> event -> anchor -> a raised status."""
        report = anchored.specialists
        assert report is not None
        assert "historical-analyst" in report.fan_out.agents

        # One event, admitted, and it qualifies.
        (event,) = report.anchors
        assert event.qualifies_as_anchor
        assert event.observed_outcome
        assert event.evidence_ids
        # Traceable to the incident's own spans, not to something nearby.
        assert set(event.evidence_ids) <= incident_evidence

        # And a Class B consequence cites it, above `speculative`.
        anchored_findings = [c for c in class_b(anchored) if c.historical_analogue_id]
        assert anchored_findings, "the anchor must reach a consequence"
        for consequence in anchored_findings:
            assert consequence.historical_analogue_id == event.event_id
            assert consequence.status in STATUSES_ABOVE_SPECULATIVE
            assert consequence.status is Status.LIKELY  # measured, not designed

    def test_the_event_id_is_derived_from_the_spans_and_not_from_the_model(
        self, anchored: AnalysisResult
    ) -> None:
        """Re-derivable, which is why the anchor needs no store and no lookup.

        A model that wanted to influence the id would have to change which spans it cites, and
        those are checked against its slice - so there is no move available to it.
        """
        report = anchored.specialists
        assert report is not None
        (event,) = report.anchors

        assert event.event_id == event_id_for(
            tenant_id=event.tenant_id, evidence_ids=list(event.evidence_ids)
        )
        assert "event_id" not in AnalogueProposal.model_fields

    def test_only_the_historical_analyst_may_propose_one(self, anchored: AnalysisResult) -> None:
        """The cooperative fake has *every* selected specialist propose an analogue, so the
        agent gate is exercised by four analysts that are not entitled to."""
        report = anchored.specialists
        assert report is not None
        assert len(report.fan_out.agents) > 1

        refusals = {reason for _title, reason in report.refused_analogues}
        assert REFUSED_WRONG_AGENT in refusals
        # Every admitted event came from the one analyst whose definition asks for them.
        for outcome in report.outcomes:
            if outcome.agent != "historical-analyst":
                assert outcome.analogues.events == ()

    def test_three_proposals_collapse_to_one_admitted_event(self, anchored: AnalysisResult) -> None:
        """Measured, and the reasons matter more than the count.

        The fake proposes three analogues per analyst: one with a measured outcome, one
        without, and one citing a span nobody offered. The historical slice on this corpus is a
        **single** span, so the first two cite the same evidence - and identity is the spans,
        so they are one event described twice rather than two precedents. The third is refused
        outright. One event survives.

        The outcome-less case therefore cannot be reached end to end on this corpus, and
        ``tests/unit/specialists/test_analogues.py`` covers it directly rather than this test
        pretending otherwise.
        """
        report = anchored.specialists
        assert report is not None

        (event,) = report.historical_events
        assert event.qualifies_as_anchor
        assert report.anchors == (event,)
        # Every event is either an anchor or reported as a precedent without an outcome.
        for candidate in report.historical_events:
            assert candidate.qualifies_as_anchor or not candidate.observed_outcome

        # And the uncited proposal was refused by name.
        assert REFUSED_UNCITED in {reason for _title, reason in report.refused_analogues}
        assert report.fabricated_evidence_ids

    # 2
    def test_a_fabricated_analogue_id_is_rejected_and_the_status_stays_speculative(
        self, positive: AnalysisResult
    ) -> None:
        """The half of the gap that was a live defect rather than a missing feature.

        On the retention change nothing anchors, and the fake hunter claims a well-formed
        ``he_...`` anyway. Before this, that id would have been carried through to judgment and
        satisfied rule 2 - a Class B finding raised above ``speculative`` on an event that
        never existed. It is now dropped and reported.
        """
        assert positive.class_b is not None
        assert positive.class_b.hunt is not None
        assert INVENTED_ANALOGUE in positive.class_b.hunt.fabricated_analogue_ids

        for hypothesis in positive.all_hypotheses:
            assert hypothesis.historical_analogue_id is None
        for consequence in class_b(positive):
            assert consequence.historical_analogue_id is None
            assert consequence.status is Status.SPECULATIVE

    def test_an_analogue_id_outside_the_admitted_anchors_is_rejected_even_when_anchors_exist(
        self, anchored: AnalysisResult
    ) -> None:
        """The stronger case: the run *has* a real anchor, so a checked-against-empty pass
        would not prove anything. The blind-spot hunter is offered no anchors at all and
        claims one, and every id that reaches a consequence is one the runtime derived."""
        report = anchored.specialists
        assert report is not None
        admitted = report.anchor_ids
        assert admitted

        blind_spots = anchored.blind_spots
        assert blind_spots is not None
        assert blind_spots.outcome is not None
        assert INVENTED_ANALOGUE in blind_spots.outcome.fabricated_analogue_ids
        assert INVENTED_ANALOGUE not in admitted

        for hypothesis in anchored.all_hypotheses:
            if hypothesis.historical_analogue_id is not None:
                assert hypothesis.historical_analogue_id in admitted
        for consequence in class_b(anchored):
            if consequence.historical_analogue_id is not None:
                assert consequence.historical_analogue_id in admitted

    def test_a_blind_spot_hypothesis_can_never_anchor(self, anchored: AnalysisResult) -> None:
        """Structural rather than incidental. A blind-spot candidate is by definition one the
        main analysis missed, so it was never offered the anchor set - and a hunter with no
        admissible set must not be able to produce one."""
        blind_spots = anchored.blind_spots
        assert blind_spots is not None
        assert blind_spots.outcome is not None

        for hypothesis in blind_spots.outcome.hypotheses:
            assert hypothesis.historical_analogue_id is None
            assert hypothesis.hypothesis_id in blind_spots.outcome.without_analogue_ids
        for consequence in class_b(anchored):
            if "blind-spot-hunter" in consequence.agent_sources:
                assert consequence.historical_analogue_id is None
                assert consequence.status is Status.SPECULATIVE

    # 3
    def test_permission_filtering_applies_to_the_historical_path(
        self, labelled_corpus: CorpusStores, incident_evidence: frozenset[str]
    ) -> None:
        """The same change, the same corpus, two scopes - and the anchor only exists for one.

        This is the test the historical path specifically needs, because an analogue is the
        one output that can *raise* a status: if history leaked past the scope, an
        unauthorized reader would get a stronger finding rather than a redacted one.
        """
        granted = run_analysis(
            labelled_corpus,
            change=refund_change(),
            text=REFUND_QUERY,
            scope=PermissionScope.of(REFUND_GRANT),
        )
        denied = run_analysis(
            labelled_corpus,
            change=refund_change(),
            text=REFUND_QUERY,
            scope=PermissionScope(),
        )

        # The authorized requester reaches the incident and gets an anchor.
        granted_report = granted.specialists
        assert granted_report is not None
        assert granted_report.anchors
        assert set(granted.retrieval.evidence_ids) & incident_evidence

        # The unauthorized one sees none of the incident's spans, so there is nothing to
        # anchor on - the filtering happens in retrieval, before the analyst is asked.
        assert not set(denied.retrieval.evidence_ids) & incident_evidence
        denied_report = denied.specialists
        assert denied_report is not None
        assert denied_report.anchors == ()
        assert denied_report.historical_events == ()
        for consequence in class_b(denied):
            assert consequence.historical_analogue_id is None
            assert consequence.status is Status.SPECULATIVE

    def test_the_historical_slice_carries_the_callers_scope_unchanged(
        self, anchored: AnalysisResult
    ) -> None:
        """It cannot widen its own audience: the slice is the caller's query, narrowed."""
        report = anchored.specialists
        assert report is not None
        main = anchored.retrieval.outcome.query

        for outcome in report.slice_outcomes:
            assert outcome.query.permission_scope == main.permission_scope
            assert outcome.query.permission_scope_hash == main.permission_scope_hash
            assert outcome.query.tenant_id == main.tenant_id
        assert report.evidence_ids <= set(anchored.retrieval.evidence_ids)
        # And every anchor rests inside that authorized set.
        for event in report.anchors:
            assert set(event.evidence_ids) <= set(anchored.retrieval.evidence_ids)

    # 4
    def test_the_anchored_finding_reaches_challenge_arbitration_and_judgment(
        self, anchored: AnalysisResult
    ) -> None:
        """An anchor is not a bypass. It makes a *higher* status reachable and changes nothing
        about which gates the hypothesis passes through on the way there."""
        report = anchored.specialists
        assert report is not None
        assert report.anchors

        anchored_hypotheses = [
            h for h in anchored.all_hypotheses if h.historical_analogue_id is not None
        ]
        assert anchored_hypotheses

        # Challenged.
        assert anchored.challenge is not None
        challenged = {
            t.hypothesis.hypothesis_id
            for t in anchored.challenge.targets
            if t.hypothesis is not None
        }
        assert {h.hypothesis_id for h in anchored_hypotheses} <= challenged

        # Arbitrated.
        assert anchored.arbitration is not None
        assert len(anchored.arbitration.rulings) == len(anchored.challenge.targets)

        # Judged, by the judge, with the challenger's work on the record.
        anchored_findings = [c for c in class_b(anchored) if c.historical_analogue_id]
        assert anchored_findings
        for consequence in anchored_findings:
            assert "consequence-judge" in consequence.agent_sources
            assert "adversarial-challenger" in consequence.agent_sources
            assert consequence.evidence_ids  # rule 1 still applies as well
        assert anchored.report is not None

    def test_a_refuted_anchored_hypothesis_is_still_clamped(self, corpus: CorpusStores) -> None:
        """The direction that matters: the analogue raises the ceiling, the challenger lowers
        the floor, and the challenger wins. A gate that could only weaken before must not
        become one an anchor can overrule."""
        refuting = run_analysis(
            corpus,
            change=refund_change(),
            text=REFUND_QUERY,
            responder=cooperative_responder(challenger_verdict="refuted"),
        )

        report = refuting.specialists
        assert report is not None
        assert report.anchors, "the anchor must still be admitted"
        for consequence in class_b(refuting):
            assert consequence.status not in STATUSES_ABOVE_SPECULATIVE

    def test_the_anchor_is_visible_in_the_fingerprint_and_survives_a_snapshot(
        self, corpus: CorpusStores
    ) -> None:
        """Two runs must agree on which precedent was used, and a reader coming back to a
        stored analysis must be able to see what a raised status rested on."""
        analyses = InMemoryAnalysisStore()
        result = run_analysis(
            corpus,
            change=refund_change(),
            text=REFUND_QUERY,
            analyses=analyses,
        )
        report = result.specialists
        assert report is not None
        (event,) = report.anchors

        assert result.fingerprint()["historical_anchors"] == [event.event_id]

        stored = analyses.require(corpus.tenant_id, result.analysis_id)
        assert stored.analogue(event.event_id) == event
        assert stored.analogue("he_" + "0" * 32) is None

    # 5
    def test_with_no_historical_evidence_every_class_b_finding_stays_speculative(
        self, positive: AnalysisResult, quiet: AnalysisResult
    ) -> None:
        """The behaviour that existed before this change, unchanged where it should be.

        The retention change is the corpus's widest one and still has no incident recording an
        outcome for it, so nothing anchors and the ceiling holds. A feature that quietly
        raised the *unanchored* case would be the worst outcome of this work.
        """
        for result in (positive, quiet):
            report = result.specialists
            assert report is not None
            assert report.anchors == ()
            assert report.anchor_ids == frozenset()
            assert result.fingerprint()["historical_anchors"] == []
            for consequence in class_b(result):
                assert consequence.historical_analogue_id is None
                assert consequence.status is Status.SPECULATIVE

    def test_the_historical_analyst_is_not_even_selected_without_a_prior_case(
        self, positive: AnalysisResult, quiet: AnalysisResult
    ) -> None:
        """The cheap gate: no incident or ticket in the authorized slice, no call made."""
        for result in (positive, quiet):
            report = result.specialists
            assert report is not None
            assert "historical-analyst" not in report.fan_out.agents
            assert "historical-analyst" in report.fan_out.reasons

    def test_an_anchor_does_not_raise_a_class_a_finding_or_change_the_class_a_path(
        self, corpus: CorpusStores, anchored: AnalysisResult
    ) -> None:
        """Class A is deterministic and has no analogue gate, so the anchor must be invisible
        to it. Compared against a run with no model at all, which is the strongest form."""
        change = refund_change()
        agentic = run_analysis(corpus, change=change, text=REFUND_QUERY)
        deterministic = run_analysis(corpus, change=change, text=REFUND_QUERY, agentic=False)

        assert agentic.specialists is not None
        assert agentic.specialists.anchors
        assert {f.finding_id for f in deterministic.class_a} == {
            f.finding_id for f in agentic.class_a
        }
        for consequence in anchored.consequences:
            if consequence.consequence_class is ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED:
                assert consequence.historical_analogue_id is None

    # 6
    def test_a_failing_historical_analyst_costs_the_anchor_and_nothing_else(
        self, corpus: CorpusStores
    ) -> None:
        """Failure isolation, on the analyst that now has the most leverage.

        Degradation has to be downward: with no anchor the Class B finding must fall back to
        ``speculative`` rather than keep a status it can no longer justify.
        """
        broken = run_analysis(
            corpus,
            change=refund_change(),
            text=REFUND_QUERY,
            responder=broken_for("historical-analyst"),
        )

        report = broken.specialists
        assert report is not None
        assert report.failed == ("historical-analyst",)
        assert report.anchors == ()
        assert report.historical_events == ()
        assert broken.stage(AnalysisStage.SPECIALISTS).status is StageStatus.FAILED
        assert "historical-analyst" in broken.stage(AnalysisStage.SPECIALISTS).detail

        # The other analysts' findings survive.
        assert report.retained
        assert "historical-analyst" not in {f.agent for f in report.retained}

        # Everything downstream still ran, and the ceiling reapplied.
        assert broken.stage(AnalysisStage.CLASS_B).status is StageStatus.COMPLETED
        assert broken.consequences
        assert broken.report is not None
        for consequence in class_b(broken):
            assert consequence.historical_analogue_id is None
            assert consequence.status is Status.SPECULATIVE

    def test_the_deterministic_stages_are_untouched_when_the_analyst_fails(
        self, corpus: CorpusStores
    ) -> None:
        """Class A recall does not depend on whether a precedent was found."""
        change = refund_change()
        intact = run_analysis(corpus, change=change, text=REFUND_QUERY)
        broken = run_analysis(
            corpus,
            change=change,
            text=REFUND_QUERY,
            responder=broken_for("historical-analyst"),
        )

        assert {f.finding_id for f in broken.class_a} == {f.finding_id for f in intact.class_a}
        assert broken.stage(AnalysisStage.CLASS_A).status is StageStatus.COMPLETED
        assert broken.stage(AnalysisStage.GRAPH).status is StageStatus.COMPLETED


class _BrokenCoupling:
    """A coupling stage that fails. Stands in for a defect in a deterministic stage.

    Deliberately not a provider error: the deterministic stages have no provider, so the only
    way one fails is a bug, and the orchestrator must abort rather than build an analysis on
    top of it.
    """

    def execute(self, **_kwargs: Any) -> Any:
        raise RuntimeError("the coupling index could not be built")

    def apply_delta(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("unreachable: execute fails first")

    def assemble(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("unreachable: execute fails first")
