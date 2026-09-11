"""Builders for valid domain objects.

Every builder returns an object that passes validation, so a negative test can change
exactly one thing and attribute the failure to it. Without this, a test asserting that
"``refuted`` requires evidence" could be passing because some unrelated required field was
also missing.

``SAMPLES`` holds one valid instance of every Phase 1 entity. The contract tests iterate it,
which means a new entity added without a sample fails the inventory test rather than
silently escaping the cross-cutting invariants.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from app.domain.entities.assumption import (
    AssumptionDelta,
    AssumptionRecord,
    CouplingHolder,
    CouplingSet,
)
from app.domain.entities.change import ChangeRequest, ChangeSpec
from app.domain.entities.claims import (
    ClaimEnvelope,
    ConsequenceHypothesis,
    DerivedRelation,
    EvidenceGap,
    ObservedFact,
)
from app.domain.entities.findings import (
    AnalysisReport,
    ArbitrationDecision,
    ChallengeResult,
    Consequence,
    ConsequenceScore,
    FalsificationTest,
    FindingDisposition,
    SpecialistFinding,
)
from app.domain.entities.graph import ConsequenceGraph, GraphEdge, GraphNode
from app.domain.entities.registry import (
    AssumptionKeyRegistryEntry,
    KeyAlias,
    KeyRegistryGap,
)
from app.domain.entities.source import (
    Chunk,
    EvidenceItem,
    HistoricalEvent,
    SourceDocument,
    SourceVersion,
)
from app.domain.value_objects.enums import (
    AuthorityClass,
    BindingRole,
    ChallengeOutcome,
    ConsequenceClass,
    DeltaAssertedBy,
    DeltaKind,
    Derivation,
    Disposition,
    EvidencePosition,
    Explicitness,
    Origin,
    Severity,
    SourceType,
    Status,
    ValueType,
)
from app.domain.value_objects.provenance import (
    DerivationProvenance,
    ModelProvenance,
    RunProvenance,
)
from app.domain.value_objects.taxonomy import Directionality, NodeType, Relationship
from app.domain.value_objects.typed_value import TypedValue
from tests.fixtures.ids import AS_OF as AS_OF
from tests.fixtures.ids import SHA as SHA
from tests.fixtures.ids import make_id as make_id

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

# `AS_OF`, `SHA` and `make_id` live in `tests.fixtures.ids` and are re-exported above
# with the explicit `as` form, so consumers keep importing them from this module.


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


def usd(amount: float = 10000) -> TypedValue:
    return TypedValue(value_type=ValueType.SCALAR, raw=amount, unit="USD")


def days(count: int = 90) -> TypedValue:
    return TypedValue(value_type=ValueType.DURATION, raw=count, unit="d")


def model_provenance(**over: Any) -> ModelProvenance:
    return ModelProvenance(
        **{
            "agent": "policy-analyst",
            "prompt_version": 2,
            "prompt_sha256": SHA,
            "provider": "fake",
            "model": "test-standard-model",
            "tier": "standard",
            **over,
        }
    )


def derivation_provenance(**over: Any) -> DerivationProvenance:
    return DerivationProvenance(
        **{
            "rule_id": "invalidation.v1",
            "rule_version": "rule-1",
            "registry_version": "registry-1",
            "coupling_index_version": "coupling-1",
            **over,
        }
    )


def run_provenance(**over: Any) -> RunProvenance:
    return RunProvenance(**{"schema_version": "domain-1", **over})


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def source_document(**over: Any) -> SourceDocument:
    return SourceDocument(
        **{
            "source_id": make_id("source"),
            "tenant_id": make_id("tenant"),
            "source_type": SourceType.POLICY,
            "title": "Refund Approval Policy",
            "uri": "s3://corpus/policy/refund-approval.md",
            "authority_class": AuthorityClass.POLICY,
            **over,
        }
    )


def source_version(**over: Any) -> SourceVersion:
    return SourceVersion(
        **{
            "version_id": make_id("source_version"),
            "source_id": make_id("source"),
            "content_hash": SHA,
            "version_label": "v3",
            **over,
        }
    )


def chunk(**over: Any) -> Chunk:
    return Chunk(
        **{
            "chunk_id": make_id("chunk"),
            "source_version_id": make_id("source_version"),
            "ordinal": 0,
            "text": "Refunds above 10,000 USD require manager approval.",
            "char_start": 0,
            "char_end": 49,
            **over,
        }
    )


def evidence_item(**over: Any) -> EvidenceItem:
    return EvidenceItem(
        **{
            "evidence_id": make_id("evidence"),
            "source_version_id": make_id("source_version"),
            "chunk_id": make_id("chunk"),
            "text_span": "Refunds above 10,000 USD require manager approval.",
            "char_start": 0,
            "char_end": 49,
            "retrieval_scores": {"lexical": 0.8, "vector": 0.7},
            **over,
        }
    )


def historical_event(**over: Any) -> HistoricalEvent:
    return HistoricalEvent(
        **{
            "event_id": make_id("event"),
            "tenant_id": make_id("tenant"),
            "title": "2024 refund threshold change caused approval backlog",
            "occurred_at": dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
            "observed_outcome": "Approval queue depth tripled for three weeks.",
            "evidence_ids": (make_id("evidence", 9),),
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def registry_entry(**over: Any) -> AssumptionKeyRegistryEntry:
    return AssumptionKeyRegistryEntry(
        **{
            "key": "refund.approval_threshold",
            "value_type": ValueType.SCALAR,
            "unit": "USD",
            "canonical_form": "refund.approval_threshold >= X USD requires approval",
            "allowed_scopes": ("region", "tier"),
            "description": "Refund amount above which manager approval is required.",
            "domain": "payments",
            "introduced_in_version": "registry-1",
            **over,
        }
    )


def registry_gap(**over: Any) -> KeyRegistryGap:
    return KeyRegistryGap(
        **{
            "gap_id": make_id("registry_gap"),
            "proposed_key": "refund.escalation_ceiling",
            "proposed_value": usd(25000),
            "evidence_id": make_id("evidence"),
            "holder_source_version_id": make_id("source_version"),
            "proposed_by": model_provenance(agent="policy-analyst"),
            "as_of": AS_OF,
            "registry_version": "registry-1",
            **over,
        }
    )


def key_alias(**over: Any) -> KeyAlias:
    return KeyAlias(
        **{
            "alias_id": make_id("alias"),
            "source_key": "refund.manager_approval_limit",
            "target_key": "refund.approval_threshold",
            "proposal_evidence": (make_id("evidence"),),
            "proposed_by": model_provenance(),
            "proposed_at": AS_OF,
            "registry_version": "registry-1",
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------


def assumption_record(**over: Any) -> AssumptionRecord:
    return AssumptionRecord(
        **{
            "assumption_id": make_id("assumption"),
            "tenant_id": make_id("tenant"),
            "key": "refund.approval_threshold",
            "registry_version": "registry-1",
            "value": usd(),
            "role": BindingRole.DECLARES,
            "explicitness": Explicitness.EXPLICIT,
            "holder_source_version_id": make_id("source_version"),
            "evidence_id": make_id("evidence"),
            "extractor_version": "extractor-1",
            "span_verified": True,
            "value_parsed": True,
            "key_resolved": True,
            **over,
        }
    )


def coupling_holder(**over: Any) -> CouplingHolder:
    return CouplingHolder(
        **{
            "source_version_id": make_id("source_version"),
            "assumption_id": make_id("assumption"),
            "value": usd(),
            "role": BindingRole.DECLARES,
            "authority_class": AuthorityClass.POLICY,
            **over,
        }
    )


def coupling_set(**over: Any) -> CouplingSet:
    return CouplingSet(
        **{
            "key": "refund.approval_threshold",
            "as_of": AS_OF,
            "holders": (coupling_holder(),),
            "authority_model_version": "authority-1",
            "coupling_index_version": "coupling-1",
            "computed_at": AS_OF,
            **over,
        }
    )


def assumption_delta(**over: Any) -> AssumptionDelta:
    return AssumptionDelta(
        **{
            "delta_id": make_id("delta"),
            "key": "refund.approval_threshold",
            "old_value": usd(10000),
            "new_value": usd(5000),
            "delta_kind": DeltaKind.VALUE_CHANGE,
            "asserted_by": DeltaAssertedBy.CHANGE_REQUEST,
            "registry_version": "registry-1",
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Change
# ---------------------------------------------------------------------------


def change_request(**over: Any) -> ChangeRequest:
    return ChangeRequest(
        **{
            "change_id": make_id("change"),
            "tenant_id": make_id("tenant"),
            "title": "Lower refund approval threshold to 5,000 USD",
            "description": "Reduce the manager-approval threshold for refunds.",
            "effective_date": AS_OF,
            **over,
        }
    )


def change_spec(**over: Any) -> ChangeSpec:
    return ChangeSpec(
        **{
            "change_id": make_id("change"),
            "subject_entities": ("refund approval",),
            "old_state": "10,000 USD",
            "new_state": "5,000 USD",
            "assumption_deltas": (assumption_delta(),),
            "version": "spec-1",
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def observed_fact(**over: Any) -> ObservedFact:
    return ObservedFact(
        **{
            "fact_id": make_id("fact"),
            "statement": "The policy states a 10,000 USD refund approval threshold.",
            "evidence_ids": (make_id("evidence"),),
            **over,
        }
    )


def derived_relation(**over: Any) -> DerivedRelation:
    return DerivedRelation(
        **{
            "relation_id": make_id("relation"),
            "relationship": Relationship.INVALIDATES_ASSUMPTION,
            "subject_ref": make_id("delta"),
            "object_ref": make_id("assumption", 2),
            "input_record_ids": (make_id("assumption"), make_id("assumption", 2)),
            "provenance": derivation_provenance(),
            **over,
        }
    )


def unauthorized_override_relation(**over: Any) -> DerivedRelation:
    """The E4 worked example: a wiki page narrowing a policy declaration."""
    return derived_relation(
        **{
            "relationship": Relationship.UNAUTHORIZED_OVERRIDE,
            "provenance": derivation_provenance(
                rule_id="override_legitimacy.v1",
                authority_model_version="authority-1",
                detail={
                    "narrower_authority_class": "wiki",
                    "broader_authority_class": "policy",
                    "ordering": "authority-1",
                    "narrower_scope": "region:eu",
                    "broader_scope": "global",
                },
            ),
            **over,
        }
    )


def consequence_hypothesis(**over: Any) -> ConsequenceHypothesis:
    return ConsequenceHypothesis(
        **{
            "hypothesis_id": make_id("hypothesis"),
            "claim": "Manager approval volume will roughly double.",
            "mechanism": "Halving the threshold moves mid-size refunds into the approval path.",
            "weakest_link": "The refund amount distribution is assumed unchanged.",
            "proposed_by": model_provenance(agent="consequence-hunter"),
            **over,
        }
    )


def evidence_gap(**over: Any) -> EvidenceGap:
    return EvidenceGap(
        **{
            "gap_id": make_id("gap"),
            "analysis_id": make_id("analysis"),
            "question": "Who owns the refund approval SOP?",
            "strategies_executed": ("lexical", "vector"),
            "source_types_searched": ("sop", "org_metadata"),
            "as_of": AS_OF,
            "would_be_resolved_by": "An ownership record for the refund approval SOP.",
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def specialist_finding(**over: Any) -> SpecialistFinding:
    return SpecialistFinding(
        **{
            "finding_id": make_id("finding"),
            "agent": "policy-analyst",
            "claim": "The training deck teaches the superseded threshold.",
            "category": "policy",
            "evidence_ids": (make_id("evidence"),),
            "evidence_position": EvidencePosition.SUPPORTED,
            "produced_by": model_provenance(),
            **over,
        }
    )


def falsification_test(**over: Any) -> FalsificationTest:
    return FalsificationTest(
        **{
            "test_id": make_id("test"),
            "consequence_id": make_id("consequence"),
            "observation": "Approval queue depth is unchanged four weeks after rollout.",
            **over,
        }
    )


def challenge_result(**over: Any) -> ChallengeResult:
    return ChallengeResult(
        **{
            "challenge_id": make_id("challenge"),
            "consequence_id": make_id("consequence"),
            "outcome": ChallengeOutcome.NO_REFUTATION_FOUND,
            "falsification_test_id": make_id("test"),
            "produced_by": model_provenance(agent="adversarial-challenger"),
            **over,
        }
    )


def arbitration_decision(**over: Any) -> ArbitrationDecision:
    return ArbitrationDecision(
        **{
            "consequence_id": make_id("consequence"),
            "status": Status.SUPPORTED,
            "severity": Severity.HIGH,
            "rationale": "Two independent sources bind the superseded value.",
            "supporting_evidence_ids": (make_id("evidence"),),
            "produced_by": model_provenance(agent="consequence-judge"),
            **over,
        }
    )


def consequence_score(**over: Any) -> ConsequenceScore:
    return ConsequenceScore(
        **{
            "consequence_id": make_id("consequence"),
            "confidence": 0.72,
            "confidence_fn_version": "confidence-1",
            "factor_inputs": {"authority": 0.8, "corroboration": 0.5},
            "corroborating_source_version_ids": (
                make_id("source_version"),
                make_id("source_version", 2),
            ),
            "corroborating_strategies": ("lexical", "vector"),
            "computed_at": AS_OF,
            **over,
        }
    )


def class_a_consequence(**over: Any) -> Consequence:
    return Consequence(
        **{
            "consequence_id": make_id("consequence"),
            "change_id": make_id("change"),
            "title": "Training deck teaches a superseded refund threshold",
            "description": "Deck v7 consumes refund.approval_threshold = 10,000 USD.",
            "category": "training",
            "consequence_class": ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED,
            "origin": Origin.RESOLVER,
            "affected_entity": "Refund training deck v7",
            "mechanism": "The deck binds the same key the change invalidates.",
            "derived_relation_ids": (make_id("relation"),),
            "assumption_keys": ("refund.approval_threshold",),
            "status": Status.SUPPORTED,
            "severity": Severity.MEDIUM,
            "evidence_ids": (make_id("evidence"),),
            "run_provenance": run_provenance(),
            **over,
        }
    )


def class_b_consequence(**over: Any) -> Consequence:
    return Consequence(
        **{
            "consequence_id": make_id("consequence", 2),
            "change_id": make_id("change"),
            "title": "Manager approval volume rises",
            "description": "More refunds enter the approval path.",
            "category": "operations",
            "consequence_class": ConsequenceClass.CLASS_B_BEHAVIORAL,
            "origin": Origin.AGENT,
            "affected_entity": "Support operations",
            "mechanism": "Threshold reduction moves mid-size refunds into approval.",
            "hypothesis_ids": (make_id("hypothesis"),),
            "status": Status.SPECULATIVE,
            "severity": Severity.MEDIUM,
            "run_provenance": run_provenance(),
            **over,
        }
    )


def finding_disposition(**over: Any) -> FindingDisposition:
    return FindingDisposition(
        **{
            "disposition_id": make_id("disposition"),
            "consequence_id": make_id("consequence"),
            "analysis_id": make_id("analysis"),
            "disposition": Disposition.USEFUL,
            "disposed_by": "reviewer@example.com",
            "disposed_at": AS_OF,
            **over,
        }
    )


def analysis_report(**over: Any) -> AnalysisReport:
    return AnalysisReport(
        **{
            "analysis_id": make_id("analysis"),
            "change_id": make_id("change"),
            "executive_summary": "One Class A staleness finding and one Class B load hypothesis.",
            "consequence_ids": (make_id("consequence"), make_id("consequence", 2)),
            "default_view_consequence_ids": (make_id("consequence"),),
            "run_provenance": run_provenance(),
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def graph_node(**over: Any) -> GraphNode:
    return GraphNode(
        **{
            "node_id": make_id("node"),
            "entity_type": NodeType.SOURCE_DOCUMENT,
            "label": "Refund Approval Policy v3",
            **over,
        }
    )


def assumption_key_node(**over: Any) -> GraphNode:
    return graph_node(
        **{
            "node_id": make_id("node", 2),
            "entity_type": NodeType.ASSUMPTION_KEY,
            "label": "refund.approval_threshold",
            "annotations": ("single holder",),
            "evidence_gap_ids": (make_id("gap"),),
            **over,
        }
    )


def derived_edge(**over: Any) -> GraphEdge:
    return GraphEdge(
        **{
            "edge_id": make_id("edge"),
            "source_node": make_id("node"),
            "target_node": make_id("node", 2),
            "relationship": Relationship.INVALIDATES_ASSUMPTION,
            "derivation": Derivation.DERIVED,
            "directionality": Directionality.DIRECTED,
            "consequence_id": make_id("consequence"),
            "derived_relation_id": make_id("relation"),
            **over,
        }
    )


def binds_edge(**over: Any) -> GraphEdge:
    return GraphEdge(
        **{
            "edge_id": make_id("edge", 2),
            "source_node": make_id("node"),
            "target_node": make_id("node", 2),
            "relationship": Relationship.BINDS_ASSUMPTION,
            "derivation": Derivation.EXTRACTED,
            "directionality": Directionality.DIRECTED,
            "evidence_ids": (make_id("evidence"),),
            **over,
        }
    )


def hypothesized_edge(**over: Any) -> GraphEdge:
    return GraphEdge(
        **{
            "edge_id": make_id("edge", 3),
            "source_node": make_id("node"),
            "target_node": make_id("node", 2),
            "relationship": Relationship.CREATES_RISK_FOR,
            "derivation": Derivation.INFERRED,
            "directionality": Directionality.DIRECTED,
            "consequence_id": make_id("consequence", 2),
            "is_hypothesized": True,
            **over,
        }
    )


def consequence_graph(**over: Any) -> ConsequenceGraph:
    return ConsequenceGraph(
        **{
            "analysis_id": make_id("analysis"),
            "nodes": (graph_node(), assumption_key_node()),
            "edges": (derived_edge(), binds_edge(), hypothesized_edge()),
            "registry_version": "registry-1",
            "coupling_index_version": "coupling-1",
            "projected_at": AS_OF,
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

#: One valid instance per Phase 1 entity, keyed by class name. The contract tests iterate
#: this, so an entity added without a sample is caught by the inventory test.
SAMPLES: dict[str, Any] = {
    "AnalysisReport": analysis_report(),
    "ArbitrationDecision": arbitration_decision(),
    "AssumptionDelta": assumption_delta(),
    "AssumptionKeyRegistryEntry": registry_entry(),
    "AssumptionRecord": assumption_record(),
    "ChallengeResult": challenge_result(),
    "ChangeRequest": change_request(),
    "ChangeSpec": change_spec(),
    "ClaimEnvelope": ClaimEnvelope(claim=observed_fact()),
    "Chunk": chunk(),
    "Consequence": class_a_consequence(),
    "ConsequenceGraph": consequence_graph(),
    "ConsequenceHypothesis": consequence_hypothesis(),
    "ConsequenceScore": consequence_score(),
    "CouplingHolder": coupling_holder(),
    "CouplingSet": coupling_set(),
    "DerivedRelation": derived_relation(),
    "EvidenceGap": evidence_gap(),
    "EvidenceItem": evidence_item(),
    "FalsificationTest": falsification_test(),
    "FindingDisposition": finding_disposition(),
    "GraphEdge": derived_edge(),
    "GraphNode": graph_node(),
    "HistoricalEvent": historical_event(),
    "KeyAlias": key_alias(),
    "KeyRegistryGap": registry_gap(),
    "ObservedFact": observed_fact(),
    "SourceDocument": source_document(),
    "SourceVersion": source_version(),
    "SpecialistFinding": specialist_finding(),
}

# The coupling phase's entities live in a sibling module so this file stays about Phase 1.
# Imported at the bottom rather than the top because `tests.fixtures.coupling` builds on
# `AS_OF` and `make_id` from here, and a top-level import would be circular.
from tests.fixtures.coupling import COUPLING_SAMPLES  # noqa: E402

SAMPLES.update(COUPLING_SAMPLES)
