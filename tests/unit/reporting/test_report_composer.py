"""The report composer: it may shorten and reorder the surface, and nothing else.

Prose is the one output this system cannot verify against evidence, so the composer's
*admitted* effects are structural - a subset and a permutation, both checkable. The tests
below are that boundary from four directions:

* a handle it was not offered cannot be added;
* a mandatory finding it omits comes back;
* an omission from the reading order appends rather than removes;
* a narration citing an identifier the analysis never produced is discarded whole.

Plus the deterministic half: :func:`summarize` composes the report's own summary from counts,
so the field a reader trusts most contains no generated text at all.
"""

from __future__ import annotations

import inspect
import json

import pytest
from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition, load_agent_definitions
from app.application.use_cases.compose_report import (
    MANDATORY_SEVERITIES,
    MANDATORY_STATUSES,
    SUPPRESSING,
    ReviewerSignals,
    summarize,
)
from app.domain.interfaces.llm import ProviderUnavailable
from app.domain.value_objects.enums import Disposition
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.providers.fake import FakeProvider
from app.reporting.composer import (
    COMPOSER_AGENT,
    REFUSAL_INVENTED_ID,
    ComposedLayout,
    CompositionCandidate,
    ReportComposer,
    composer_version,
)
from app.reporting.proposals import FORBIDDEN_AGENT_FIELDS, CompositionProposal

CONSEQUENCE_A = "cq_" + "a" * 32
CONSEQUENCE_B = "cq_" + "b" * 32
CONSEQUENCE_C = "cq_" + "c" * 32
EVIDENCE_A = "ev_" + "a" * 32
INVENTED = "ev_" + "9" * 32


@pytest.fixture
def definition() -> AgentDefinition:
    return load_agent_definitions()[COMPOSER_AGENT]


def candidate(handle: str, consequence_id: str, *, mandatory: bool = False) -> CompositionCandidate:
    return CompositionCandidate(
        handle=handle,
        consequence_id=consequence_id,
        title=f"Finding {handle}",
        status="likely" if mandatory else "speculative",
        severity="HIGH" if mandatory else "LOW",
        consequence_class="class_a_assumption_mediated",
        confidence=0.5,
        mandatory=mandatory,
        evidence_ids=(EVIDENCE_A,),
    )


async def compose(
    gateway: LlmGateway,
    definition: AgentDefinition,
    candidates: tuple[CompositionCandidate, ...],
    response: str | None = None,
    *,
    eligible: tuple[str, ...] | None = None,
    finalized: frozenset[str] | None = None,
) -> ComposedLayout:
    if response is not None:
        gateway._registry.get("fake").queue_response(response)  # type: ignore[attr-defined]
    composer = ReportComposer(completions=gateway, definition=definition)
    return await composer.compose(
        candidates=candidates,
        eligible=eligible if eligible is not None else tuple(c.handle for c in candidates),
        finalized_ids=finalized
        if finalized is not None
        else frozenset({*(c.consequence_id for c in candidates), EVIDENCE_A}),
        counts={"findings": len(candidates)},
    )


class TestTheSurface:
    @pytest.mark.asyncio
    async def test_a_handle_it_was_not_offered_cannot_be_added(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidates = (candidate("r1", CONSEQUENCE_A),)

        layout = await compose(
            gateway,
            definition,
            candidates,
            json.dumps({"foreground_handles": ["r1", "r9"]}),
        )

        assert layout.foreground == (CONSEQUENCE_A,)
        assert layout.refused_handles == ("r9",)

    @pytest.mark.asyncio
    async def test_a_mandatory_finding_it_omits_is_restored(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        """The floor that stops "reduce alert load" from becoming "hide the alarming ones"."""
        candidates = (
            candidate("r1", CONSEQUENCE_A, mandatory=True),
            candidate("r2", CONSEQUENCE_B),
        )

        layout = await compose(
            gateway, definition, candidates, json.dumps({"foreground_handles": ["r2"]})
        )

        assert CONSEQUENCE_A in layout.foreground
        assert layout.restored_handles == ("r1",)

    @pytest.mark.asyncio
    async def test_an_omission_from_the_reading_order_appends_rather_than_removes(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidates = (
            candidate("r1", CONSEQUENCE_A),
            candidate("r2", CONSEQUENCE_B),
            candidate("r3", CONSEQUENCE_C),
        )

        layout = await compose(
            gateway,
            definition,
            candidates,
            json.dumps(
                {
                    "foreground_handles": ["r1", "r2", "r3"],
                    "reading_order": ["r3"],
                }
            ),
        )

        assert layout.foreground[0] == CONSEQUENCE_C
        assert set(layout.foreground) == {CONSEQUENCE_A, CONSEQUENCE_B, CONSEQUENCE_C}

    @pytest.mark.asyncio
    async def test_narrowing_reports_what_left_the_surface(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidates = (candidate("r1", CONSEQUENCE_A), candidate("r2", CONSEQUENCE_B))

        layout = await compose(
            gateway, definition, candidates, json.dumps({"foreground_handles": ["r1"]})
        )

        assert layout.foreground == (CONSEQUENCE_A,)
        assert layout.reachable == (CONSEQUENCE_B,)
        assert layout.narrowed_by == 1

    @pytest.mark.asyncio
    async def test_no_findings_means_no_call(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        layout = await compose(gateway, definition, ())

        assert layout.calls_made == 0
        assert layout.foreground == ()


class TestTheNarrationGate:
    @pytest.mark.asyncio
    async def test_a_narration_citing_an_unproduced_identifier_is_discarded_whole(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        """Whole, not repaired: the remaining sentences came from the same process."""
        candidates = (candidate("r1", CONSEQUENCE_A),)

        layout = await compose(
            gateway,
            definition,
            candidates,
            json.dumps(
                {
                    "foreground_handles": ["r1"],
                    "executive_summary": f"Review the retention change; see {INVENTED}.",
                    "section_notes": [
                        {"section": "contradictions", "note": "Verifiable on delivery."}
                    ],
                }
            ),
        )

        assert layout.narration is None
        assert layout.narration_refusal == REFUSAL_INVENTED_ID
        assert layout.invented_ids == (INVENTED,)
        # The structural effects still applied: refusal is per-narration, not per-call.
        assert layout.foreground == (CONSEQUENCE_A,)

    @pytest.mark.asyncio
    async def test_a_narration_citing_only_finalized_identifiers_is_kept(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidates = (candidate("r1", CONSEQUENCE_A),)

        layout = await compose(
            gateway,
            definition,
            candidates,
            json.dumps(
                {
                    "foreground_handles": ["r1"],
                    "executive_summary": f"The retention change rests on {EVIDENCE_A}.",
                    "section_notes": [
                        {"section": "contradictions", "note": "Verifiable on delivery."}
                    ],
                }
            ),
        )

        assert layout.narration is not None
        assert layout.narration_refusal is None
        assert layout.narration.section_notes

    @pytest.mark.asyncio
    async def test_a_note_is_gated_as_well_as_the_summary(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        candidates = (candidate("r1", CONSEQUENCE_A),)

        layout = await compose(
            gateway,
            definition,
            candidates,
            json.dumps(
                {
                    "foreground_handles": ["r1"],
                    "section_notes": [
                        {"section": "class_a", "note": f"Rests on {INVENTED} throughout."}
                    ],
                }
            ),
        )

        assert layout.narration is None
        assert layout.narration_refusal == REFUSAL_INVENTED_ID

    @pytest.mark.asyncio
    async def test_a_provider_failure_leaves_the_deterministic_surface_standing(
        self, gateway: LlmGateway, definition: AgentDefinition, fake_provider: FakeProvider
    ) -> None:
        for _ in range(10):
            fake_provider.queue_failure(ProviderUnavailable("no credential", provider="fake"))
        candidates = (candidate("r1", CONSEQUENCE_A),)

        layout = await compose(gateway, definition, candidates)

        assert not layout.succeeded
        assert layout.failure
        assert layout.foreground == ()
        assert layout.narration is None


class TestTheDeterministicSummary:
    def test_the_summary_states_only_figures_and_stays_within_the_field_bound(self) -> None:
        signals = ReviewerSignals(
            measured=True,
            already_known=(CONSEQUENCE_A,),
            suppressed=(CONSEQUENCE_B,),
            suppressed_by={CONSEQUENCE_B: Disposition.WRONG.value},
            reviewed=2,
        )

        summary = summarize(
            findings=(),
            foreground=(),
            contradictions=(),
            gaps=(),
            questions=(),
            signals=signals,
            unmeasured=("explicitness",),
        )

        assert "No consequence was finalized" in summary
        assert "already-known rate 0.50" in summary
        assert "explicitness" in summary
        assert len(summary) <= 2000

    def test_an_unmeasured_reviewer_state_says_so_rather_than_reporting_zero(self) -> None:
        summary = summarize(
            findings=(),
            foreground=(),
            contradictions=(),
            gaps=(),
            questions=(),
            signals=ReviewerSignals(),
            unmeasured=(),
        )

        assert "No reviewer verdicts were available" in summary

    def test_the_already_known_rate_is_none_when_nothing_was_reviewed(self) -> None:
        assert ReviewerSignals(measured=True, reviewed=0).already_known_rate is None
        assert ReviewerSignals(measured=False).already_known_rate is None
        assert (
            ReviewerSignals(measured=True, reviewed=4, already_known=("a", "b")).already_known_rate
            == 0.5
        )

    def test_already_known_is_not_a_suppressing_verdict(self) -> None:
        """The frozen contract, restated nowhere: read off the one set that defines it."""
        assert Disposition.ALREADY_KNOWN not in SUPPRESSING
        assert set(SUPPRESSING) == {Disposition.WRONG, Disposition.NOT_MATERIAL}

    def test_the_surface_rule_is_the_two_frozen_sets(self) -> None:
        assert {s.value for s in MANDATORY_STATUSES} == {"supported", "likely"}
        assert {s.value for s in MANDATORY_SEVERITIES} == {"HIGH", "CRITICAL"}


class TestSchema:
    @pytest.mark.parametrize("field", FORBIDDEN_AGENT_FIELDS)
    def test_the_pipeline_assigned_fields_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            CompositionProposal(**{"foreground_handles": ["r1"], field: "anything"})

    def test_the_composer_has_no_field_for_a_claim(self) -> None:
        """It carries findings through. There is nowhere for it to write a new one."""
        fields = set(CompositionProposal.model_fields)
        assert not fields & {"claim", "consequences", "findings", "evidence_ids"}

    def test_the_compose_signature_takes_no_evidence_and_no_retriever(self) -> None:
        parameters = set(inspect.signature(ReportComposer.compose).parameters)
        assert not parameters & {"evidence", "retrieval", "sources", "store"}

    def test_the_version_label_names_prompt_contract_and_schema(
        self, definition: AgentDefinition
    ) -> None:
        label = composer_version(definition)
        assert label.startswith("composer-")
        assert label.count(":") == 2

    def test_the_composer_refuses_a_definition_for_another_agent(self, gateway: LlmGateway) -> None:
        other = load_agent_definitions()["consequence-judge"]
        with pytest.raises(ValueError, match=COMPOSER_AGENT):
            ReportComposer(completions=gateway, definition=other)
