"""The blind-spot hunter: a partition, not a personality.

The failure this stage is prone to is not a bad finding - it is a *plausible* finding that the
main analysis already produced, re-delivered as a discovery. So almost every test below is one
of two properties:

* the partition is disjoint from what the main consequence-hunter read, and the type refuses
  to hold one that is not;
* attribution is derived from the spans a finding cited, never taken from the model.

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
from app.application.use_cases.hunt_blind_spots import partition_queries
from app.blindspots.hunter import (
    BLIND_SPOT_AGENT,
    BlindSpotHunter,
    BlindSpotOutcome,
    blind_spot_version,
)
from app.blindspots.partition import (
    SLICE_PRIORITY,
    EvidencePartition,
    PartitionError,
    PartitionSlice,
    PartitionSpan,
    partition_from,
    undocumented_keys,
)
from app.blindspots.proposals import (
    FORBIDDEN_AGENT_FIELDS,
    BlindSpotBatch,
    BlindSpotProposal,
    CorpusSilenceProposal,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.decay import DecayEntry, DecayParticipant, KnowledgeDecayReport
from app.domain.entities.source import EvidenceItem, SourceDocument, SourceVersion
from app.domain.interfaces.llm import ProviderUnavailable
from app.domain.interfaces.retrieval import (
    EvidenceCandidate,
    RetrievalOutcome,
    RetrievalQuery,
    RetrievalResult,
    RetrievalTrace,
)
from app.domain.value_objects.enums import (
    AuthorityClass,
    DecayCategory,
    RetrievalStrategy,
    SourceType,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.permissions import PermissionScope
from app.domain.value_objects.scope import ScopeSelector
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.providers.fake import FakeProvider

TENANT = "tn_" + "0" * 31 + "1"
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


@pytest.fixture
def definition() -> AgentDefinition:
    return load_agent_definitions()[BLIND_SPOT_AGENT]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _hex(prefix: str, tag: str) -> str:
    return f"{prefix}_{(tag * 32)[:32]}"


def candidate(
    tag: str,
    *,
    keys: tuple[str, ...] = (),
    source_type: SourceType = SourceType.POLICY,
) -> EvidenceCandidate:
    document = SourceDocument(
        source_id=_hex("src", tag),
        tenant_id=TENANT,
        uri=f"corpus://{tag}.md",
        title=f"Document {tag}",
        source_type=source_type,
        authority_class=AuthorityClass.POLICY,
    )
    version = SourceVersion(
        version_id=_hex("sv", tag),
        source_id=document.source_id,
        content_hash="a" * 64,
        version_label="v1",
    )
    evidence = EvidenceItem(
        evidence_id=_hex("ev", tag),
        source_version_id=version.version_id,
        chunk_id=_hex("ch", tag),
        text_span=f"span {tag} records a value nothing else repeats.",
        char_start=0,
        char_end=48,
        location=f"Document {tag} > Section",
    )
    return EvidenceCandidate(
        evidence=evidence,
        document=document,
        version=version,
        scopes=(ScopeSelector.global_scope(),),
        assumption_keys=keys,
    )


def outcome(*candidates: EvidenceCandidate, top_k: int = 10) -> RetrievalOutcome:
    query = RetrievalQuery(
        tenant_id=TENANT, text="retention window", permission_scope=PermissionScope(), top_k=top_k
    )
    results = tuple(
        RetrievalResult(candidate=item, strategy=RetrievalStrategy.LEXICAL, score=1.0, rank=i + 1)
        for i, item in enumerate(candidates)
    )
    return RetrievalOutcome(
        query=query,
        results=results,
        strategies_executed=(RetrievalStrategy.LEXICAL,),
        trace=RetrievalTrace(
            considered=len(candidates),
            matched=len(candidates),
            returned=len(candidates),
        ),
        permission_scope_hash=query.permission_scope_hash,
        scorer_version="bm25-1",
    )


def change() -> ChangeSpec:
    return ChangeSpec(
        change_id=new_id("change"),
        subject_entities=("personal data retention window",),
        old_state="90 days",
        new_state="30 days",
        version="change-1",
    )


def proposal(**over: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim": "An undocumented dependency fails silently after the change.",
        "mechanism_chain": ["one artifact records it", "nothing else repeats it"],
        "weakest_link": "nothing else repeats it",
        "cited_evidence_ids": [],
    }
    payload.update(over)
    return payload


def silence(**over: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": "Who owns the downstream reconciliation step?",
        "why_it_matters": "An unowned step has nobody to notify.",
        "recommended_source": "org ownership metadata",
    }
    payload.update(over)
    return payload


async def hunt(
    gateway: LlmGateway,
    definition: AgentDefinition,
    partition: EvidencePartition,
    response: str,
    *,
    existing_claims: tuple[str, ...] = (),
) -> BlindSpotOutcome:
    gateway._registry.get("fake").queue_response(response)  # type: ignore[attr-defined]
    hunter = BlindSpotHunter(completions=gateway, definition=definition)
    return await hunter.hunt(change=change(), partition=partition, existing_claims=existing_claims)


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


class TestPartition:
    def test_a_partition_overlapping_the_main_analysis_is_refused(self) -> None:
        """The type will not hold one. A second opinion is not an independent one."""
        span = PartitionSpan(
            evidence_id=_hex("ev", "a"),
            text="x",
            source_type="policy",
            authority_class="policy",
            slice_=PartitionSlice.LOW_RANKED,
        )
        with pytest.raises(PartitionError, match="already offered"):
            EvidencePartition(spans=(span,), main_offered_ids=frozenset({span.evidence_id}))

    def test_spans_the_main_hunter_saw_are_excluded_and_counted(self) -> None:
        seen, unseen = candidate("a"), candidate("b")

        built = partition_from(
            [outcome(seen, unseen)],
            main_offered_ids={seen.evidence_id},
            main_source_types={SourceType.POLICY},
        )

        assert built.evidence_ids == frozenset({unseen.evidence_id})
        assert built.excluded_as_already_seen == 1

    def test_attribution_follows_the_frozen_priority(self) -> None:
        """A span in two slices is credited to the higher one, whatever query returned it."""
        both = candidate("a", keys=("access.review_interval",))
        unwritten = candidate("b", keys=("api.max_page_size",))
        elsewhere = candidate("c", source_type=SourceType.TICKET)
        ranked = candidate("d")

        built = partition_from(
            [outcome(both, unwritten, elsewhere, ranked)],
            main_offered_ids=(),
            main_source_types={SourceType.POLICY},
            single_holder_keys=("access.review_interval",),
            unwritten_rule_keys=("access.review_interval", "api.max_page_size"),
        )

        assert built.slice_of(both.evidence_id) is PartitionSlice.SINGLE_HOLDER
        assert built.slice_of(unwritten.evidence_id) is PartitionSlice.UNWRITTEN_RULE
        assert built.slice_of(elsewhere.evidence_id) is PartitionSlice.UNQUERIED_SOURCE_TYPE
        assert built.slice_of(ranked.evidence_id) is PartitionSlice.LOW_RANKED
        assert built.covered == SLICE_PRIORITY

    def test_an_empty_slice_is_reported_rather_than_dropped(self) -> None:
        built = partition_from(
            [outcome(candidate("a"))],
            main_offered_ids=(),
            main_source_types={SourceType.POLICY},
        )

        assert built.covered == (PartitionSlice.LOW_RANKED,)
        assert PartitionSlice.SINGLE_HOLDER in built.uncovered

    def test_the_span_budget_truncates_deterministically_and_says_how_much(self) -> None:
        spans = [candidate(c) for c in "abcdef"]

        first = partition_from(
            [outcome(*spans)], main_offered_ids=(), main_source_types=set(), limit=2
        )
        again = partition_from(
            [outcome(*spans)], main_offered_ids=(), main_source_types=set(), limit=2
        )

        assert len(first.spans) == 2
        assert first.excluded_by_budget == 4
        assert first.evidence_ids == again.evidence_ids

    def test_the_undocumented_keys_come_from_the_existing_decay_report(self) -> None:
        """Read, not recomputed: a second implementation is a second definition."""
        report = KnowledgeDecayReport(
            tenant_id=TENANT,
            as_of=AS_OF,
            entries=(
                DecayEntry(
                    category=DecayCategory.SINGLE_HOLDER,
                    key="access.review_interval",
                    summary="Only one artifact binds it.",
                    participants=(
                        DecayParticipant(source_version_id=_hex("sv", "a"), artifact="A"),
                    ),
                ),
                DecayEntry(
                    category=DecayCategory.UNWRITTEN_RULE,
                    key="api.max_page_size",
                    summary="Declared only by configuration.",
                    participants=(
                        DecayParticipant(source_version_id=_hex("sv", "b"), artifact="B"),
                    ),
                ),
            ),
            keys_examined=2,
            bindings_examined=2,
            registry_version="registry-1",
            coupling_index_version="coupling-1",
            authority_model_version="authority-1",
        )

        single, unwritten = undocumented_keys(report)

        assert single == ("access.review_interval",)
        assert unwritten == ("api.max_page_size",)

    def test_a_partition_query_changes_only_text_top_k_and_source_types(self) -> None:
        """Anything else would let this stage re-decide the authorization boundary."""
        base = RetrievalQuery(
            tenant_id=TENANT,
            text="retention window",
            permission_scope=PermissionScope(grants=("privacy-office",)),
            as_of=AS_OF,
            top_k=10,
        )

        queries = partition_queries(
            base,
            returned_types={SourceType.POLICY},
            undocumented_keys=("access.review_interval",),
        )

        assert len(queries) == 3
        for query in queries:
            assert query.tenant_id == base.tenant_id
            assert query.permission_scope == base.permission_scope
            assert query.permission_scope_hash == base.permission_scope_hash
            assert query.as_of == base.as_of
            assert query.scope == base.scope
        assert queries[-1].text != base.text
        assert "review_interval" in queries[-1].text


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestTheGate:
    @pytest.mark.asyncio
    async def test_a_citation_outside_the_partition_is_dropped(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        inside = candidate("a")
        main_id = _hex("ev", "b")
        partition = partition_from(
            [outcome(inside)], main_offered_ids={main_id}, main_source_types=set()
        )

        result = await hunt(
            gateway,
            definition,
            partition,
            json.dumps(
                {
                    "consequences": [
                        proposal(cited_evidence_ids=[inside.evidence_id, main_id, _hex("ev", "f")])
                    ]
                }
            ),
        )

        assert result.succeeded
        assert result.hypotheses[0].evidence_ids == (inside.evidence_id,)
        assert main_id in result.fabricated_evidence_ids
        assert result.reused_main_evidence_ids == (main_id,)

    @pytest.mark.asyncio
    async def test_a_self_reported_restatement_is_dropped(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        """Dropping it is cheaper than a duplicate a reader has to recognise."""
        partition = partition_from(
            [outcome(candidate("a"))], main_offered_ids=(), main_source_types=set()
        )

        result = await hunt(
            gateway,
            definition,
            partition,
            json.dumps({"consequences": [proposal(restates_existing=True)]}),
            existing_claims=("Downstream handling volume shifts.",),
        )

        assert result.hypotheses == ()
        assert result.restated_claims

    @pytest.mark.asyncio
    async def test_the_slice_is_derived_from_the_citations_not_accepted_from_the_model(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        """A hunter that could name its own slice could claim a yield it did not earn."""
        span = candidate("a", keys=("access.review_interval",))
        partition = partition_from(
            [outcome(span)],
            main_offered_ids=(),
            main_source_types=set(),
            single_holder_keys=("access.review_interval",),
        )

        result = await hunt(
            gateway,
            definition,
            partition,
            json.dumps(
                {
                    "consequences": [
                        proposal(
                            cited_evidence_ids=[span.evidence_id],
                            partition_slice="low_ranked",
                        )
                    ]
                }
            ),
        )

        hypothesis_id = result.hypotheses[0].hypothesis_id
        assert result.slice_of[hypothesis_id] is PartitionSlice.SINGLE_HOLDER
        assert result.misattributed_ids == (hypothesis_id,)
        assert result.yield_by_slice["single_holder"] == 1

    @pytest.mark.asyncio
    async def test_corpus_silence_stays_a_separate_kind(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        """No claim, no mechanism, no citations - so no path by which one becomes a finding."""
        partition = EvidencePartition()

        result = await hunt(
            gateway, definition, partition, json.dumps({"corpus_silence": [silence()]})
        )

        assert result.hypotheses == ()
        assert len(result.silences) == 1
        assert not hasattr(result.silences[0], "evidence_ids")
        assert set(CorpusSilenceProposal.model_fields) == {
            "question",
            "why_it_matters",
            "recommended_source",
            "partition_slice",
        }

    @pytest.mark.asyncio
    async def test_an_empty_partition_still_calls_but_cannot_produce_a_cited_finding(
        self, gateway: LlmGateway, definition: AgentDefinition
    ) -> None:
        result = await hunt(
            gateway,
            definition,
            EvidencePartition(),
            json.dumps(
                {
                    "consequences": [proposal(cited_evidence_ids=[_hex("ev", "a")])],
                    "unusable_slices": ["low_ranked"],
                }
            ),
        )

        assert result.calls_made == 1
        assert result.hypotheses[0].evidence_ids == ()
        assert result.mechanism_only_ids
        assert result.unusable_slices == ("low_ranked",)

    @pytest.mark.asyncio
    async def test_a_provider_failure_yields_no_findings_and_records_the_failure(
        self, gateway: LlmGateway, definition: AgentDefinition, fake_provider: FakeProvider
    ) -> None:
        for _ in range(10):
            fake_provider.queue_failure(ProviderUnavailable("no credential", provider="fake"))
        hunter = BlindSpotHunter(completions=gateway, definition=definition)

        result = await hunter.hunt(change=change(), partition=EvidencePartition())

        assert not result.succeeded
        assert result.failure
        assert result.hypotheses == ()
        assert result.silences == ()
        assert result.provenance is not None


class TestSchema:
    @pytest.mark.parametrize("field", FORBIDDEN_AGENT_FIELDS)
    def test_the_four_pipeline_assigned_fields_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            BlindSpotProposal(**proposal(**{field: "anything"}))

    def test_the_forbidden_fields_are_absent_from_the_published_schema(self) -> None:
        rendered = json.dumps(BlindSpotBatch.json_schema())
        for field in FORBIDDEN_AGENT_FIELDS:
            assert f'"{field}"' not in rendered

    def test_a_mechanism_must_have_at_least_two_links(self) -> None:
        with pytest.raises(ValidationError):
            BlindSpotProposal(**proposal(mechanism_chain=["only one"], weakest_link="only one"))

    def test_the_weakest_link_must_be_one_of_the_links(self) -> None:
        with pytest.raises(ValidationError, match="weakest_link"):
            BlindSpotProposal(**proposal(weakest_link="something else entirely"))

    def test_the_version_label_names_prompt_contract_and_schema(
        self, definition: AgentDefinition
    ) -> None:
        label = blind_spot_version(definition)
        assert label.startswith("blindspot-")
        assert label.count(":") == 2

    def test_the_hunter_refuses_a_definition_for_another_agent(self, gateway: LlmGateway) -> None:
        other = load_agent_definitions()["consequence-hunter"]
        with pytest.raises(ValueError, match=BLIND_SPOT_AGENT):
            BlindSpotHunter(completions=gateway, definition=other)

    def test_the_hunt_signature_takes_a_partition_and_no_retriever(self) -> None:
        """The stage cannot widen its own evidence: it is handed a partition, not a store."""
        parameters = set(inspect.signature(BlindSpotHunter.hunt).parameters)
        assert "partition" in parameters
        assert not parameters & {"retrieval", "sources", "store", "query"}
