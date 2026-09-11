"""The specialist fan-out: seven agents, and the gates that stop them being seven opinions.

Adding seven agents at once is the largest single increase in model-authored content this
system has taken. What keeps that safe is not the prompts - it is that nothing a specialist
returns can assign a class, a status, a severity or a number, and that what it *can* return is
capped by two deterministic facts. Almost every test here is one of those:

* :class:`TestThePositionCeiling` - an observation with no admitted citation cannot be
  reported as supported, however confidently it was worded;
* :class:`TestTheGate` - a citation outside this specialist's slice is dropped like a
  fabricated one, and the four pipeline-assigned fields have nowhere to arrive;
* :class:`TestSuppression` - the two deterministic duplicate rules, including the one the
  sample corpus does not exercise;
* :class:`TestSliceQueries` - a slice narrows the caller's query and cannot widen it, which is
  the whole permission story.

Driven through the real ``LlmGateway`` over ``FakeProvider``: no credentials, no network.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition, load_agent_definitions
from app.application.use_cases.consult_specialists import (
    SUPPRESSED_DUPLICATE,
    SUPPRESSED_RESTATES_CLASS_A,
    ConsultSpecialistsUseCase,
    slice_query,
    suppress,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.findings import SpecialistFinding
from app.domain.interfaces.llm import ProviderUnavailable
from app.domain.interfaces.retrieval import RetrievalQuery
from app.domain.value_objects.enums import EvidencePosition, SourceType
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.provenance import ModelProvenance
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.providers.fake import FakeProvider
from app.specialists.analyst import (
    POSITION_WITHOUT_EVIDENCE,
    SpecialistAnalyst,
    SpecialistOutcome,
    position_for,
    specialist_version,
)
from app.specialists.proposals import (
    FORBIDDEN_AGENT_FIELDS,
    ObservationBatch,
    ObservationProposal,
)
from app.specialists.roster import (
    BY_AGENT,
    NOT_SELECTED_EMPTY_SLICE,
    NOT_SELECTED_NO_DEFINITION,
    ROSTER,
    SPECIALIST_AGENTS,
    SliceSpan,
    SpecialistSlice,
    select,
)

TENANT = "tn_" + "0" * 31 + "1"
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


@pytest.fixture
def definitions() -> dict[str, AgentDefinition]:
    return load_agent_definitions()


def _hex(prefix: str, tag: str) -> str:
    return f"{prefix}_{tag * 32}"


def change() -> ChangeSpec:
    return ChangeSpec(
        change_id=new_id("change"),
        subject_entities=("personal data retention window",),
        old_state="90 days",
        new_state="30 days",
        affected_domains=("privacy",),
        version="change-1",
    )


def span(tag: str, *, keys: tuple[str, ...] = ()) -> SliceSpan:
    return SliceSpan(
        evidence_id=_hex("ev", tag),
        text=f"span {tag} states a retention window of 90 days.",
        source_type="policy",
        authority_class="policy",
        source_title=f"Document {tag}",
        assumption_keys=keys,
    )


def a_slice(agent: str = "policy-analyst", *tags: str, **kw: Any) -> SpecialistSlice:
    return SpecialistSlice(specialist=BY_AGENT[agent], spans=tuple(span(t) for t in tags), **kw)


def proposal(**over: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim": "The retention policy commits to a window this change alters.",
        "basis": "direct_evidence",
        "cited_evidence_ids": [_hex("ev", "a")],
    }
    payload.update(over)
    return payload


def finding(
    *,
    agent: str = "policy-analyst",
    claim: str = "The policy commits to a window this change alters.",
    evidence_ids: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
    position: EvidencePosition = EvidencePosition.SUPPORTED,
) -> SpecialistFinding:
    return SpecialistFinding(
        finding_id=new_id("finding"),
        agent=agent,
        claim=claim,
        category=BY_AGENT[agent].category,
        evidence_ids=evidence_ids,
        evidence_position=position,
        assumption_keys=keys,
        produced_by=ModelProvenance(
            agent=agent,
            prompt_version=2,
            prompt_sha256="0" * 64,
            provider="fake",
            model="test-standard-model",
            tier="standard",
        ),
    )


def class_a_finding(*, key: str, evidence: tuple[str, ...]) -> ClassAFinding:
    return ClassAFinding(
        finding_id=_hex("ca", "a"),
        tenant_id=TENANT,
        relation_id=_hex("dr", "a"),
        relationship=Relationship.CONFLICTS_WITH,
        rule_id="same_scope_contradiction",
        rule_version="class-a-1",
        assumption_key=key,
        scope=ScopeSelector.global_scope(),
        as_of=AS_OF,
        affected_source_version_id=_hex("sv", "a"),
        affected_source_id=_hex("src", "a"),
        affected_artifact="Data Retention Policy",
        summary=f"Two artifacts disagree about {key}.",
        # Two inputs: a derived relation from one binding would be a restatement, and the
        # entity refuses it.
        input_record_ids=(_hex("as", "a"), _hex("as", "b")),
        materialized_evidence_ids=evidence,
        registry_version="registry-1",
        coupling_index_version="coupling-1",
        authority_model_version="authority-1",
    )


async def observe(
    gateway: LlmGateway,
    definitions: dict[str, AgentDefinition],
    slice_: SpecialistSlice,
    response: str,
) -> SpecialistOutcome:
    gateway._registry.get("fake").queue_response(response)  # type: ignore[attr-defined]
    analyst = SpecialistAnalyst(completions=gateway, definition=definitions[slice_.agent])
    return await analyst.observe(change=change(), slice_=slice_)


# ---------------------------------------------------------------------------
# The roster and the selection rule
# ---------------------------------------------------------------------------


class TestTheRoster:
    def test_the_roster_is_exactly_the_seven_specialists_with_definitions(
        self, definitions: dict[str, AgentDefinition]
    ) -> None:
        assert len(ROSTER) == 7
        assert set(SPECIALIST_AGENTS) <= set(definitions)
        # Every one gets a category a reader can group by, and no two share one.
        categories = [s.category for s in ROSTER]
        assert len(set(categories)) == len(categories)

    def test_every_specialist_has_a_non_empty_source_type_affinity(self) -> None:
        """The slice *is* the difference between them (``CLAUDE.md`` 17)."""
        for specialist in ROSTER:
            assert specialist.source_types
            assert all(isinstance(t, SourceType) for t in specialist.source_types)

    def test_a_specialist_with_an_empty_slice_is_not_invoked_and_the_reason_is_reported(
        self,
    ) -> None:
        """The fan-out gate: nothing to cite means nothing but speculation to offer."""
        fan_out = select(
            [a_slice("policy-analyst", "a"), a_slice("data-analyst")],
            available=SPECIALIST_AGENTS,
        )

        assert fan_out.agents == ("policy-analyst",)
        assert fan_out.reasons["data-analyst"] == NOT_SELECTED_EMPTY_SLICE
        # Every specialist is accounted for, selected or not.
        assert len(fan_out.selected) + len(fan_out.not_selected) == len(ROSTER)

    def test_a_specialist_with_no_definition_is_excluded_with_its_own_reason(self) -> None:
        """A renamed ``.md`` costs one domain, not the analysis."""
        fan_out = select(
            [a_slice("policy-analyst", "a"), a_slice("data-analyst", "b")],
            available=("policy-analyst",),
        )

        assert fan_out.agents == ("policy-analyst",)
        assert fan_out.reasons["data-analyst"] == NOT_SELECTED_NO_DEFINITION

    def test_selection_is_in_roster_order_whatever_order_the_slices_arrive_in(self) -> None:
        """So which of two duplicate observations survives is not a race."""
        forward = select(
            [a_slice("policy-analyst", "a"), a_slice("data-analyst", "b")],
            available=SPECIALIST_AGENTS,
        )
        reversed_ = select(
            [a_slice("data-analyst", "b"), a_slice("policy-analyst", "a")],
            available=SPECIALIST_AGENTS,
        )

        assert forward.agents == reversed_.agents == ("policy-analyst", "data-analyst")

    def test_overlapping_slices_are_reported_rather_than_prevented(self) -> None:
        """Risk and history share incidents. Corroboration counts versions, so it is safe."""
        shared = SliceSpan(
            evidence_id=_hex("ev", "c"),
            text="INC-4471 recorded a backlog after the last threshold change.",
            source_type="incident",
            authority_class="ticket",
            source_title="INC-4471",
        )
        fan_out = select(
            [
                SpecialistSlice(specialist=BY_AGENT["risk-analyst"], spans=(shared,)),
                SpecialistSlice(specialist=BY_AGENT["historical-analyst"], spans=(shared,)),
            ],
            available=SPECIALIST_AGENTS,
        )

        assert fan_out.shared_evidence_ids == frozenset({_hex("ev", "c")})
        assert len(fan_out.selected) == 2


class TestSliceQueries:
    def test_a_slice_query_narrows_the_callers_query_and_changes_nothing_else(self) -> None:
        """The permission story: a specialist sees a subset of what the requester could."""
        base = RetrievalQuery(
            tenant_id=TENANT,
            text="personal data retention window",
            permission_scope=PermissionScope(grants=("privacy-office",)),
            as_of=AS_OF,
            scope=ScopeSelector.global_scope(),
            top_k=10,
        )

        narrowed = slice_query(base, BY_AGENT["policy-analyst"].source_types)

        assert narrowed.tenant_id == base.tenant_id
        assert narrowed.permission_scope == base.permission_scope
        assert narrowed.permission_scope_hash == base.permission_scope_hash
        assert narrowed.as_of == base.as_of
        assert narrowed.scope == base.scope
        assert narrowed.text == base.text
        assert narrowed.source_types == (SourceType.POLICY,)

    def test_the_fan_out_signature_takes_a_query_and_no_store(self) -> None:
        """It cannot read the corpus any other way, so it cannot widen the scope.

        Asserted on the signature because a rule enforced by absence stays enforced when
        somebody adds a branch, and one enforced by a branch does not.
        """
        parameters = set(inspect.signature(ConsultSpecialistsUseCase.execute).parameters)
        assert "query" in parameters
        assert not parameters & {"sources", "assumptions", "store", "index", "coupling"}

    def test_the_fan_out_refuses_an_analyst_that_is_not_on_the_roster(self) -> None:
        with pytest.raises(ValueError, match="not on the specialist roster"):
            ConsultSpecialistsUseCase(analysts={"consequence-hunter": None}, retrieval=None)  # type: ignore[dict-item,arg-type]


# ---------------------------------------------------------------------------
# The position ceiling
# ---------------------------------------------------------------------------


class TestThePositionCeiling:
    @pytest.mark.parametrize("basis", ["direct_evidence", "inference"])
    def test_an_observation_with_no_admitted_citation_is_unresolved(self, basis: str) -> None:
        """However it was worded. ``unresolved``, not ``unsupported``: nobody produced
        evidence either way, and saying the evidence is against it is a finding nobody made."""
        assert position_for(basis, has_evidence=False) is POSITION_WITHOUT_EVIDENCE
        assert POSITION_WITHOUT_EVIDENCE is EvidencePosition.UNRESOLVED

    def test_an_inference_cannot_reach_supported(self) -> None:
        assert position_for("inference", has_evidence=True) is EvidencePosition.WEAKLY_SUPPORTED
        assert position_for("direct_evidence", has_evidence=True) is EvidencePosition.SUPPORTED

    def test_the_ceiling_has_no_parameter_for_how_sure_the_specialist_sounded(self) -> None:
        parameters = set(inspect.signature(position_for).parameters)
        assert parameters == {"basis", "has_evidence"}

    @pytest.mark.asyncio
    async def test_a_confidently_worded_uncited_claim_comes_back_unresolved(
        self, gateway: LlmGateway, definitions: dict[str, AgentDefinition]
    ) -> None:
        """End to end through the gate, which is where it matters."""
        outcome = await observe(
            gateway,
            definitions,
            a_slice("policy-analyst", "a"),
            json.dumps(
                {
                    "observations": [
                        proposal(
                            claim="This certainly breaks the downstream reconciliation.",
                            cited_evidence_ids=[_hex("ev", "f")],
                        )
                    ]
                }
            ),
        )

        [observed] = outcome.findings
        assert observed.evidence_position is EvidencePosition.UNRESOLVED
        assert observed.evidence_ids == ()
        assert outcome.uncited_finding_ids == (observed.finding_id,)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestTheGate:
    @pytest.mark.asyncio
    async def test_a_citation_outside_this_specialists_slice_is_dropped(
        self, gateway: LlmGateway, definitions: dict[str, AgentDefinition]
    ) -> None:
        """Another specialist's id is dropped exactly like a fabricated one."""
        outcome = await observe(
            gateway,
            definitions,
            a_slice("policy-analyst", "a", "b"),
            json.dumps(
                {
                    "observations": [
                        proposal(
                            cited_evidence_ids=[
                                _hex("ev", "a"),
                                _hex("ev", "9"),  # in nobody's slice
                            ]
                        )
                    ]
                }
            ),
        )

        [observed] = outcome.findings
        assert observed.evidence_ids == (_hex("ev", "a"),)
        assert outcome.fabricated_evidence_ids == (_hex("ev", "9"),)

    @pytest.mark.asyncio
    async def test_the_category_is_fixed_by_the_roster_not_proposed(
        self, gateway: LlmGateway, definitions: dict[str, AgentDefinition]
    ) -> None:
        """A label the agent could choose is a label that tells a reader nothing."""
        outcome = await observe(
            gateway,
            definitions,
            a_slice("data-analyst", "a"),
            json.dumps({"observations": [proposal()]}),
        )

        [observed] = outcome.findings
        assert observed.category == BY_AGENT["data-analyst"].category
        assert observed.agent == "data-analyst"

    @pytest.mark.asyncio
    async def test_observed_bindings_are_recorded_and_never_admitted(
        self, gateway: LlmGateway, definitions: dict[str, AgentDefinition]
    ) -> None:
        """The registry is closed and admission is an ingestion-time deterministic gate.

        The outcome carries the proposal; nothing in it is a record, a key registration or an
        index write - and the outcome has no field that could be one.
        """
        outcome = await observe(
            gateway,
            definitions,
            a_slice("policy-analyst", "a"),
            json.dumps(
                {
                    "observations": [proposal()],
                    "observed_bindings": [
                        {
                            "key": "retention.window",
                            "value": "90 d",
                            "evidence_id": _hex("ev", "a"),
                        }
                    ],
                }
            ),
        )

        assert len(outcome.observed_bindings) == 1
        assert outcome.observed_bindings[0].key == "retention.window"
        # The value stays rendered text. Typed-parsing it is the admission gate's job, and
        # doing it here would be the first half of a promotion.
        assert outcome.observed_bindings[0].value == "90 d"
        assert not hasattr(outcome.observed_bindings[0], "admitted")
        assert not any("record" in name for name in SpecialistOutcome.__dataclass_fields__)

    @pytest.mark.asyncio
    async def test_a_provider_failure_yields_no_findings_and_names_the_analyst(
        self,
        gateway: LlmGateway,
        definitions: dict[str, AgentDefinition],
        fake_provider: FakeProvider,
    ) -> None:
        for _ in range(10):
            fake_provider.queue_failure(ProviderUnavailable("no credential", provider="fake"))
        analyst = SpecialistAnalyst(completions=gateway, definition=definitions["policy-analyst"])

        outcome = await analyst.observe(change=change(), slice_=a_slice("policy-analyst", "a"))

        assert not outcome.succeeded
        assert outcome.agent == "policy-analyst"
        assert outcome.findings == ()
        assert outcome.provenance is not None

    @pytest.mark.asyncio
    async def test_an_analyst_handed_another_specialists_slice_refuses(
        self, gateway: LlmGateway, definitions: dict[str, AgentDefinition]
    ) -> None:
        """The correlated-findings failure the slice design exists to prevent."""
        analyst = SpecialistAnalyst(completions=gateway, definition=definitions["policy-analyst"])

        with pytest.raises(ValueError, match="another's evidence"):
            await analyst.observe(change=change(), slice_=a_slice("data-analyst", "a"))

    def test_the_runtime_refuses_an_agent_that_is_not_a_specialist(
        self, gateway: LlmGateway, definitions: dict[str, AgentDefinition]
    ) -> None:
        with pytest.raises(ValueError, match="not on the specialist roster"):
            SpecialistAnalyst(completions=gateway, definition=definitions["consequence-judge"])


class TestSchema:
    @pytest.mark.parametrize("field", FORBIDDEN_AGENT_FIELDS)
    def test_the_four_pipeline_assigned_fields_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ObservationProposal(**proposal(**{field: "anything"}))

    def test_the_published_schema_offers_no_class_confidence_severity_or_status(self) -> None:
        fields = set(ObservationProposal.model_fields) | set(ObservationBatch.model_fields)
        assert not fields & set(FORBIDDEN_AGENT_FIELDS)

    def test_basis_is_the_only_position_axis_and_it_is_closed(self) -> None:
        enum = ObservationProposal.model_json_schema()["properties"]["basis"]["enum"]
        assert set(enum) == {"direct_evidence", "inference"}

    def test_a_specialist_finding_has_no_field_for_a_class_or_a_number(self) -> None:
        """The stronger guarantee: the frozen entity could not carry one."""
        fields = set(SpecialistFinding.model_fields)
        assert not fields & {"confidence", "severity", "status", "consequence_class"}

    def test_the_version_label_names_agent_prompt_contract_and_schema(
        self, definitions: dict[str, AgentDefinition]
    ) -> None:
        label = specialist_version(definitions["risk-analyst"])
        assert label.startswith("specialist-risk-analyst-")
        assert label.count(":") == 2


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_the_same_claim_on_the_same_evidence_from_two_specialists_is_one_observation(
        self,
    ) -> None:
        shared = (_hex("ev", "a"),)
        first = finding(agent="risk-analyst", evidence_ids=shared)
        second = finding(agent="historical-analyst", evidence_ids=shared)

        retained, suppressed = suppress((first, second), class_a=())

        assert retained == (first,)
        assert suppressed == ((second, SUPPRESSED_DUPLICATE),)

    def test_the_same_claim_on_different_evidence_is_two_observations(self) -> None:
        """Two analysts finding the same thing in different documents is corroboration to
        measure, not a duplicate to drop - and the measurement happens downstream."""
        first = finding(agent="risk-analyst", evidence_ids=(_hex("ev", "a"),))
        second = finding(agent="historical-analyst", evidence_ids=(_hex("ev", "b"),))

        retained, suppressed = suppress((first, second), class_a=())

        assert len(retained) == 2
        assert suppressed == ()

    def test_an_observation_restating_a_class_a_finding_from_its_own_spans_is_suppressed(
        self,
    ) -> None:
        """The same conclusion at a trust level the specialist did not earn."""
        spans = (_hex("ev", "a"), _hex("ev", "b"))
        deterministic = class_a_finding(key="retention.window", evidence=spans)
        restatement = finding(
            claim="The retention policy and the team notes disagree about the window.",
            evidence_ids=(_hex("ev", "a"),),
            keys=("retention.window",),
        )

        retained, suppressed = suppress((restatement,), class_a=(deterministic,))

        assert retained == ()
        assert suppressed == ((restatement, SUPPRESSED_RESTATES_CLASS_A),)

    def test_an_observation_on_a_decided_key_citing_other_evidence_survives(self) -> None:
        """New information about a decided key is new information.

        This is the half of the rule that makes it safe to have: suppressing everything on a
        Class A key would silence the specialists exactly where the analysis is densest.
        """
        deterministic = class_a_finding(key="retention.window", evidence=(_hex("ev", "a"),))
        elsewhere = finding(
            claim="The training deck still teaches the old window.",
            evidence_ids=(_hex("ev", "c"),),
            keys=("retention.window",),
        )

        retained, suppressed = suppress((elsewhere,), class_a=(deterministic,))

        assert retained == (elsewhere,)
        assert suppressed == ()

    def test_an_uncited_observation_is_never_suppressed_as_a_class_a_restatement(self) -> None:
        """It rests on no spans, so it cannot rest on *those* spans."""
        deterministic = class_a_finding(key="retention.window", evidence=(_hex("ev", "a"),))
        uncited = finding(
            evidence_ids=(),
            keys=("retention.window",),
            position=EvidencePosition.UNRESOLVED,
        )

        retained, _suppressed = suppress((uncited,), class_a=(deterministic,))

        assert retained == (uncited,)

    def test_suppression_is_order_stable(self) -> None:
        """Input order is roster order, so the survivor does not depend on call timing."""
        shared = (_hex("ev", "a"),)
        policy = finding(agent="policy-analyst", evidence_ids=shared)
        data = finding(agent="data-analyst", evidence_ids=shared)

        assert suppress((policy, data), class_a=())[0] == (policy,)
        assert suppress((data, policy), class_a=())[0] == (data,)
