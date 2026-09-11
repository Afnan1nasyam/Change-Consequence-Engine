"""The evidence arbiter: independence is the factor that gets faked, so it is measured.

An arbiter that reports a corroboration count is reporting a claim. The tests below are almost
all one property: the *ceiling* on a position is computed from the deterministically measured
set of distinct ``source_version_id`` values, so no claim the model makes - and no number of
agreeing agents - can raise it.

The negative-by-absence tests carry the most weight:

* :func:`TestTheCeiling.test_the_ceiling_has_no_parameter_for_a_surviving_challenge`
* :func:`TestTheCeiling.test_the_ceiling_has_no_parameter_for_an_agent`
* :func:`TestTheGate.test_a_corroboration_claim_beyond_the_measurement_is_trimmed`
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition, load_agent_definitions
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.findings import ChallengeResult
from app.domain.interfaces.llm import ProviderUnavailable
from app.domain.services.confidence import ConfidenceInputs
from app.domain.value_objects.enums import (
    ChallengeOutcome,
    ConsequenceClass,
    EvidencePosition,
    Origin,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import ModelProvenance
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.providers.fake import FakeProvider
from app.judgment.arbitration import (
    ARBITER_AGENT,
    ASSERTIVENESS,
    CORROBORATION_FOR_SUPPORTED,
    FORBIDDEN_AGENT_FIELDS,
    ArbitrationOutcome,
    ClampReason,
    EvidenceArbiter,
    RulingBatch,
    RulingProposal,
    apply_ceiling,
    arbiter_version,
    ceiling_for,
)
from app.judgment.confidence_inputs import AssembledInputs, FactorCoverage
from app.judgment.dossier import CandidateDossier

EVIDENCE_A = "ev_" + "a" * 32
EVIDENCE_B = "ev_" + "b" * 32
FABRICATED = "ev_" + "f" * 32
VERSION_A = "sv_" + "1" * 32
VERSION_B = "sv_" + "2" * 32
INVENTED_VERSION = "sv_" + "9" * 32
NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def definition() -> AgentDefinition:
    return load_agent_definitions()[ARBITER_AGENT]


def change() -> ChangeSpec:
    return ChangeSpec(
        change_id=new_id("change"),
        subject_entities=("refund approval threshold",),
        old_state="10,000 USD",
        new_state="5,000 USD",
        version="change-1",
    )


def provenance() -> ModelProvenance:
    return ModelProvenance(
        agent="consequence-hunter",
        prompt_version="2",
        prompt_sha256="a" * 64,
        provider="fake",
        model="test-standard-model",
        tier="standard",
    )


def dossier(
    *,
    handle: str = "j1",
    evidence_ids: tuple[str, ...] = (EVIDENCE_A, EVIDENCE_B),
    challenges: tuple[ChallengeResult, ...] = (),
    disagreements: tuple[str, ...] = ("The effective date is not stated anywhere.",),
) -> CandidateDossier:
    """A Class B dossier, which is the simpler of the two and enough for every test here."""
    consequence_id = new_id("consequence")
    return CandidateDossier(
        handle=handle,
        consequence_id=consequence_id,
        consequence_class=ConsequenceClass.CLASS_B_BEHAVIORAL,
        origin=Origin.AGENT,
        claim="Downstream handling volume shifts materially after the change.",
        mechanism="threshold moves -> more items cross -> capacity unchanged",
        evidence_ids=evidence_ids,
        hypothesis=ConsequenceHypothesis(
            hypothesis_id=new_id("hypothesis"),
            claim="Downstream handling volume shifts materially after the change.",
            mechanism="threshold moves -> more items cross -> capacity unchanged",
            weakest_link="capacity unchanged",
            evidence_ids=evidence_ids,
            proposed_by=provenance(),
        ),
        challenges=challenges,
        upstream_disagreements=disagreements,
    )


def refutation(consequence_id: str) -> ChallengeResult:
    return ChallengeResult(
        challenge_id=new_id("challenge"),
        consequence_id=consequence_id,
        outcome=ChallengeOutcome.REFUTED,
        contradicting_evidence_ids=(EVIDENCE_B,),
        contradicts_what="The cited span states a different value.",
        produced_by=provenance(),
    )


def measured(*versions: str) -> AssembledInputs:
    return AssembledInputs(
        inputs=ConfidenceInputs(
            corroborating_source_version_ids=versions,
            corroborating_strategies=("lexical",) if versions else (),
        ),
        coverage=FactorCoverage(
            supporting_ids=2,
            matched_candidates=2,
            distinct_versions=len(versions),
            strategies_executed=1,
            versions_dated=0,
            bindings_supplied=0,
        ),
    )


def ruling(**over: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "handle": "j1",
        "position": "supported",
        "supporting_evidence_ids": [EVIDENCE_A, EVIDENCE_B],
        "corroborating_source_version_ids": [VERSION_A, VERSION_B],
        "independence_note": "Two distinct source versions, each retrieved.",
    }
    payload.update(over)
    return payload


async def arbitrate(
    gateway: LlmGateway,
    definition: AgentDefinition,
    dossiers: tuple[CandidateDossier, ...],
    response: str | None = None,
    *,
    versions: tuple[str, ...] = (VERSION_A, VERSION_B),
) -> ArbitrationOutcome:
    if response is not None:
        gateway._registry.get("fake").queue_response(response)  # type: ignore[attr-defined]
    arbiter = EvidenceArbiter(completions=gateway, definition=definition)
    return await arbiter.arbitrate(
        change=change(),
        dossiers=dossiers,
        measured={d.consequence_id: measured(*versions) for d in dossiers},
        now=NOW,
    )


# ---------------------------------------------------------------------------
# The deterministic ceiling
# ---------------------------------------------------------------------------


class TestTheCeiling:
    def test_no_supporting_evidence_caps_the_position_at_unsupported(self) -> None:
        ceiling = ceiling_for(
            has_supporting_evidence=False,
            has_contradicting_evidence=False,
            was_refuted=False,
            measured_corroboration=5,
        )

        assert ceiling.position is EvidencePosition.UNSUPPORTED
        assert ClampReason.NO_SUPPORTING_EVIDENCE in ceiling.reasons

    def test_contradicting_evidence_caps_the_position_at_disputed(self) -> None:
        ceiling = ceiling_for(
            has_supporting_evidence=True,
            has_contradicting_evidence=True,
            was_refuted=False,
            measured_corroboration=5,
        )

        assert ceiling.position is EvidencePosition.DISPUTED
        assert ClampReason.CONTRADICTING_EVIDENCE in ceiling.reasons

    def test_a_refutation_caps_the_position_at_disputed(self) -> None:
        ceiling = ceiling_for(
            has_supporting_evidence=True,
            has_contradicting_evidence=False,
            was_refuted=True,
            measured_corroboration=5,
        )

        assert ceiling.position is EvidencePosition.DISPUTED
        assert ClampReason.REFUTED_BY_CHALLENGE in ceiling.reasons

    def test_one_source_version_caps_the_position_below_supported(self) -> None:
        """Genuine corroboration means distinct source versions, and one is not two."""
        ceiling = ceiling_for(
            has_supporting_evidence=True,
            has_contradicting_evidence=False,
            was_refuted=False,
            measured_corroboration=CORROBORATION_FOR_SUPPORTED - 1,
        )

        assert ceiling.position is EvidencePosition.WEAKLY_SUPPORTED
        assert ClampReason.INSUFFICIENT_INDEPENDENT_CORROBORATION in ceiling.reasons

    def test_every_reason_is_reported_not_only_the_binding_one(self) -> None:
        """A claim with two problems needs both, or a reader fixes the wrong one."""
        ceiling = ceiling_for(
            has_supporting_evidence=False,
            has_contradicting_evidence=True,
            was_refuted=False,
            measured_corroboration=0,
        )

        assert ceiling.position is EvidencePosition.UNSUPPORTED
        assert len(ceiling.reasons) == 3

    def test_a_fully_supported_claim_has_no_ceiling(self) -> None:
        ceiling = ceiling_for(
            has_supporting_evidence=True,
            has_contradicting_evidence=False,
            was_refuted=False,
            measured_corroboration=CORROBORATION_FOR_SUPPORTED,
        )

        assert ceiling.position is EvidencePosition.SUPPORTED
        assert ceiling.reasons == ()

    @pytest.mark.parametrize("proposed", list(EvidencePosition))
    def test_applying_a_ceiling_never_raises_a_position(self, proposed: EvidencePosition) -> None:
        ceiling = ceiling_for(
            has_supporting_evidence=False,
            has_contradicting_evidence=False,
            was_refuted=False,
            measured_corroboration=0,
        )

        clamped = apply_ceiling(proposed, ceiling)

        assert ASSERTIVENESS[clamped.position] <= ASSERTIVENESS[proposed]

    def test_the_ceiling_has_no_parameter_for_a_surviving_challenge(self) -> None:
        """``no_refutation_found`` cannot raise a position, because there is nowhere to say it.

        Asserted on the signature rather than on an outcome: a rule enforced by absence stays
        enforced when somebody adds a branch, and a rule enforced by a branch does not.
        """
        parameters = set(inspect.signature(ceiling_for).parameters)
        assert parameters == {
            "has_supporting_evidence",
            "has_contradicting_evidence",
            "was_refuted",
            "measured_corroboration",
        }

    def test_the_ceiling_has_no_parameter_for_an_agent(self) -> None:
        """Checkpoint 9.3, structurally: corroboration counts sources, never agents."""
        parameters = " ".join(inspect.signature(ceiling_for).parameters)
        for word in ("agent", "agree", "model", "count_of_agents"):
            assert word not in parameters


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestTheGate:
    @pytest.mark.asyncio
    async def test_a_corroboration_claim_beyond_the_measurement_is_trimmed(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        """The claim is intersected with the measurement, and the ceiling reads the latter."""
        candidate = dossier()

        outcome = await arbitrate(
            gateway,
            definition,
            (candidate,),
            json.dumps(
                {
                    "rulings": [
                        ruling(
                            corroborating_source_version_ids=[
                                VERSION_A,
                                INVENTED_VERSION,
                            ]
                        )
                    ]
                }
            ),
            versions=(VERSION_A,),
        )

        [result] = outcome.rulings
        assert result.corroborating_source_version_ids == (VERSION_A,)
        assert result.overclaimed_source_version_ids == (INVENTED_VERSION,)
        assert result.measured_corroboration == 1
        # And the overclaim bought nothing: one measured version caps the position.
        assert result.position is EvidencePosition.WEAKLY_SUPPORTED
        assert ClampReason.INSUFFICIENT_INDEPENDENT_CORROBORATION in result.clamp_reasons

    @pytest.mark.asyncio
    async def test_a_fabricated_citation_is_dropped_and_reported(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidate = dossier()

        outcome = await arbitrate(
            gateway,
            definition,
            (candidate,),
            json.dumps({"rulings": [ruling(supporting_evidence_ids=[EVIDENCE_A, FABRICATED])]}),
        )

        [result] = outcome.rulings
        assert result.supporting_evidence_ids == (EVIDENCE_A,)
        assert result.fabricated_evidence_ids == (FABRICATED,)

    @pytest.mark.asyncio
    async def test_a_refuted_candidate_cannot_be_called_supported(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidate = dossier()
        refuted = dossier(handle="j1", challenges=(refutation(candidate.consequence_id),))
        # The challenge has to name the dossier it is on, so rebuild around one id.
        refuted = dossier(handle="j1", challenges=(refutation(refuted.consequence_id),))

        outcome = await arbitrate(
            gateway, definition, (refuted,), json.dumps({"rulings": [ruling()]})
        )

        [result] = outcome.rulings
        assert result.proposed_position is EvidencePosition.SUPPORTED
        assert result.position is EvidencePosition.DISPUTED
        assert ClampReason.REFUTED_BY_CHALLENGE in result.clamp_reasons

    @pytest.mark.asyncio
    async def test_a_handle_that_was_not_offered_is_refused(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        outcome = await arbitrate(
            gateway,
            definition,
            (dossier(),),
            json.dumps({"rulings": [ruling(handle="j9")]}),
        )

        assert outcome.rulings == ()
        assert outcome.refused_unknown_handles == ("j9",)
        assert len(outcome.unruled) == 1

    @pytest.mark.asyncio
    async def test_a_second_ruling_on_one_candidate_is_refused(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        """One ruling per candidate. A second is a repair artefact, not a second opinion."""
        outcome = await arbitrate(
            gateway,
            definition,
            (dossier(),),
            json.dumps({"rulings": [ruling(), ruling(position="unsupported")]}),
        )

        assert len(outcome.rulings) == 1
        assert outcome.refused_unknown_handles == ("j1",)

    @pytest.mark.asyncio
    async def test_every_upstream_disagreement_survives_into_the_ruling(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidate = dossier(disagreements=("A and B disagree about the window.",))

        outcome = await arbitrate(
            gateway,
            definition,
            (candidate,),
            json.dumps({"rulings": [ruling(residual_disagreement="No span states the date.")]}),
        )

        [result] = outcome.rulings
        assert "No span states the date." in result.disagreements
        assert "A and B disagree about the window." in result.disagreements

    @pytest.mark.asyncio
    async def test_no_candidates_means_no_call(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        outcome = await arbitrate(gateway, definition, ())

        assert outcome.calls_made == 0
        assert outcome.succeeded
        assert outcome.rulings == ()

    @pytest.mark.asyncio
    async def test_a_provider_failure_leaves_every_candidate_unruled(
        self, gateway: LlmGateway, definition: AgentDefinition, fake_provider: FakeProvider
    ) -> None:
        """Unruled, never defaulted: an unarbitrated claim has not been cleared."""
        for _ in range(10):
            fake_provider.queue_failure(ProviderUnavailable("no credential", provider="fake"))
        candidates = (dossier(handle="j1"), dossier(handle="j2"))

        outcome = await arbitrate(gateway, definition, candidates)

        assert not outcome.succeeded
        assert outcome.rulings == ()
        assert outcome.unruled == candidates
        assert outcome.positions == {}


class TestSchema:
    @pytest.mark.parametrize("field", FORBIDDEN_AGENT_FIELDS)
    def test_the_four_pipeline_assigned_fields_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            RulingProposal(**ruling(**{field: "anything"}))

    def test_the_published_schema_offers_no_confidence_and_no_agent_count(self) -> None:
        """Asserted over *field names*, not over the rendered text.

        ``residual_disagreement`` contains the substring "agreement", so a grep over the JSON
        reads correctly and asserts the wrong thing.
        """
        fields = set(RulingProposal.model_fields) | set(RulingBatch.model_fields)

        assert not fields & {"confidence", "severity", "status", "consequence_class"}
        for name in fields:
            assert "agent" not in name
            assert name != "agreement"
            assert name != "corroboration_count"

    def test_the_position_values_are_exactly_the_frozen_evidence_positions(self) -> None:
        literal = RulingProposal.model_json_schema()["properties"]["position"]["enum"]
        assert set(literal) == {p.value for p in EvidencePosition}

    def test_the_version_label_names_prompt_contract_and_schema(
        self, definition: AgentDefinition
    ) -> None:
        label = arbiter_version(definition)
        assert label.startswith("arbiter-")
        assert label.count(":") == 2

    def test_the_arbiter_refuses_a_definition_for_another_agent(self, gateway: LlmGateway) -> None:
        other = load_agent_definitions()["consequence-judge"]
        with pytest.raises(ValueError, match=ARBITER_AGENT):
            EvidenceArbiter(completions=gateway, definition=other)
