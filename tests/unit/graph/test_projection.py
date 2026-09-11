"""The graph projector: rebuilt from the store, never authored, and identical twice.

The graph is the one component that must contain no judgement at all. Everything upstream
decides something - what a span says, which rule fired, whether a divergence matters, what
status the evidence reaches - and the projector decides nothing. It renders what those stages
concluded, or it reports that it could not.

So the tests are about four properties:

* **the taxonomy is obeyed exactly** - seven edges, their admissible derivations, their
  directionality, and nothing else;
* **nothing is invented from absence** - a record that cannot legally project is reported,
  never approximated;
* **type E never becomes an edge** - a gap annotates a node or nothing;
* **two rebuilds are byte-identical** - which is what makes a projection comparable at all.

The tests carrying the most weight:

* :func:`TestDeterminism.test_two_rebuilds_are_byte_identical`
* :func:`TestEvidenceGaps.test_a_gap_never_produces_an_edge`
* :func:`TestSymmetry.test_a_symmetric_relation_yields_one_canonically_ordered_edge`
* :func:`TestTheRetentionWindowJoin` - the join the ``assumption_key`` node type exists for,
  measured over the real sample corpus.
"""

from __future__ import annotations

import ast
import asyncio
import datetime as dt
import inspect
import pathlib
from typing import Any

import pytest
from pydantic import ValidationError

from app.application.use_cases.resolve_coupling import CouplingAnalysis, ResolveCouplingUseCase
from app.domain.entities.assumption import AssumptionDelta
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import DerivedRelation, EvidenceGap
from app.domain.entities.coupling import ResolvedBinding
from app.domain.entities.findings import Consequence
from app.domain.entities.graph import ConsequenceGraph, GraphEdge, GraphNode
from app.domain.value_objects.enums import (
    AuthorityClass,
    BindingRole,
    ConsequenceClass,
    DeltaAssertedBy,
    DeltaKind,
    Derivation,
    Origin,
    Severity,
    Status,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import DerivationProvenance, RunProvenance
from app.domain.value_objects.rules import COUPLING_INDEX_VERSION, RuleId
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import (
    ADMISSIBLE_DERIVATIONS,
    CONSEQUENCE_BEARING,
    DERIVED_RELATIONSHIPS,
    DIRECTIONALITY,
    Directionality,
    NodeType,
    Relationship,
)
from app.evaluation.corpus import load_corpus
from app.graph.projection import (
    GRAPH_PROJECTION_VERSION,
    SKIP_GAP_NODE_UNKNOWN,
    SKIP_NO_CHANGE_EVIDENCE,
    SKIP_NO_CONSEQUENCE,
    SKIP_REJECTED,
    SKIP_UNKNOWN_ENDPOINT,
    ProjectionResult,
    project,
)
from app.infrastructure.database.repositories.memory import InMemoryCouplingIndexStore
from app.ingestion.identity import edge_id_for, node_id_for
from tests.fixtures.coupling import KEY, NOW, REGISTRY_VERSION, binding, days

PROJECTED_AT = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)
ANALYSIS = new_id("analysis")
TENANT = "tn_" + "0" * 31 + "1"
EVIDENCE = "ev_" + "e" * 32
ANALOGUE = "he_" + "9" * 32
DECLARES = BindingRole.DECLARES
CONSUMES = BindingRole.CONSUMES


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def change(**over: Any) -> ChangeSpec:
    payload: dict[str, Any] = {
        "change_id": new_id("change"),
        "subject_entities": ("personal data retention window",),
        "old_state": "90 days",
        "new_state": "30 days",
        "affected_domains": ("privacy",),
        "evidence_ids": (EVIDENCE,),
        "version": "change-1",
    }
    payload.update(over)
    return ChangeSpec(**payload)


def delta(**over: Any) -> AssumptionDelta:
    payload: dict[str, Any] = {
        "delta_id": new_id("delta"),
        "key": KEY,
        "scope": ScopeSelector.global_scope(),
        "old_value": days(90),
        "new_value": days(30),
        "delta_kind": DeltaKind.VALUE_CHANGE,
        "asserted_by": DeltaAssertedBy.CHANGE_REQUEST,
        "registry_version": REGISTRY_VERSION,
    }
    payload.update(over)
    return AssumptionDelta(**payload)


def relation(
    left: ResolvedBinding,
    right: ResolvedBinding,
    *,
    relationship: Relationship = Relationship.CONFLICTS_WITH,
    rule_id: str = RuleId.SAME_SCOPE_CONTRADICTION,
) -> DerivedRelation:
    """A derived relation over two bindings, with version ids as its refs.

    Version ids, matching the real collision matrix: that is what makes the projector's
    version-to-document map necessary and testable.
    """
    detail: dict[str, Any] = {"assumption_key": left.key, "scope": left.scope.render()}
    if relationship is Relationship.UNAUTHORIZED_OVERRIDE:
        detail |= {
            "narrower_authority_class": left.authority_class.value,
            "broader_authority_class": right.authority_class.value,
        }
    return DerivedRelation(
        relation_id=new_id("relation"),
        relationship=relationship,
        subject_ref=left.source_version_id,
        object_ref=right.source_version_id,
        input_record_ids=(left.assumption_id, right.assumption_id),
        provenance=DerivationProvenance(
            rule_id=rule_id,
            rule_version=COUPLING_INDEX_VERSION,
            registry_version=REGISTRY_VERSION,
            coupling_index_version=COUPLING_INDEX_VERSION,
            authority_model_version="authority-1",
            detail=detail,
        ),
    )


def class_a_consequence(relation_record: DerivedRelation, **over: Any) -> Consequence:
    payload: dict[str, Any] = {
        "consequence_id": new_id("consequence"),
        "change_id": new_id("change"),
        "title": "Two policies disagree on the retention window",
        "description": "Two policies declare different retention windows at global scope.",
        "category": "same_scope_contradiction",
        "consequence_class": ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED,
        "origin": Origin.RESOLVER,
        "affected_entity": "Retention Policy",
        "mechanism": "rule same_scope_contradiction over two bindings of retention.window",
        "derived_relation_ids": (relation_record.relation_id,),
        "assumption_keys": (KEY,),
        "status": Status.LIKELY,
        "severity": Severity.HIGH,
        "evidence_ids": (EVIDENCE,),
        "run_provenance": RunProvenance(schema_version="schema-1"),
    }
    payload.update(over)
    return Consequence(**payload)


def class_b_consequence(**over: Any) -> Consequence:
    payload: dict[str, Any] = {
        "consequence_id": new_id("consequence"),
        "change_id": new_id("change"),
        "title": "Deletion queue depth roughly doubles",
        "description": "Shortening the window moves more records into the deletion queue.",
        "category": "class_b_behavioral",
        "consequence_class": ConsequenceClass.CLASS_B_BEHAVIORAL,
        "origin": Origin.AGENT,
        "affected_entity": "privacy deletion queue",
        "mechanism": "more records cross the line -> the queue lengthens",
        "hypothesis_ids": (new_id("hypothesis"),),
        "status": Status.SPECULATIVE,
        "severity": Severity.MEDIUM,
        "evidence_ids": (EVIDENCE,),
        "run_provenance": RunProvenance(schema_version="schema-1"),
    }
    payload.update(over)
    return Consequence(**payload)


def gap(**over: Any) -> EvidenceGap:
    payload: dict[str, Any] = {
        "gap_id": new_id("gap"),
        "analysis_id": ANALYSIS,
        "question": "Does any regional policy set a shorter retention window?",
        "strategies_executed": ("lexical",),
        "source_types_searched": ("policy",),
        "as_of": NOW,
        "would_be_resolved_by": "a regional retention policy admitted into the corpus",
    }
    payload.update(over)
    return EvidenceGap(**payload)


def peers() -> tuple[ResolvedBinding, ResolvedBinding]:
    """Two policies declaring different windows at the same scope. Different sources."""
    return (
        binding(role=DECLARES, value=days(90), scope="global", tag="policy-a"),
        binding(role=DECLARES, value=days(30), scope="global", tag="policy-b"),
    )


def run(**over: Any) -> ProjectionResult:
    kwargs: dict[str, Any] = {
        "analysis_id": ANALYSIS,
        "change": change(),
        "registry_version": REGISTRY_VERSION,
        "coupling_index_version": COUPLING_INDEX_VERSION,
        "projected_at": PROJECTED_AT,
    }
    kwargs.update(over)
    return project(**kwargs)


def node_for(entity_type: NodeType, natural_key: str) -> str:
    return node_id_for(entity_type=entity_type.value, natural_key=natural_key)


# ---------------------------------------------------------------------------
# The seven edge types
# ---------------------------------------------------------------------------


class TestAllAdmissibleEdgeTypes:
    def test_the_taxonomy_has_exactly_seven_edges(self) -> None:
        """A permissive catch-all absorbs the graph and stops carrying information."""
        assert len(Relationship) == 7
        assert set(ADMISSIBLE_DERIVATIONS) == set(Relationship)
        assert set(DIRECTIONALITY) == set(Relationship)

    def test_binds_assumption_projects_the_record_itself(self) -> None:
        """Document -> assumption key, one per admitted binding, citing its span.

        The edge the whole product rests on: two documents joined to one key neither
        mentions is the central visualization (checkpoint 10.4).
        """
        left, right = peers()
        result = run(bindings=(left, right))

        edges = result.edges_by_relationship[Relationship.BINDS_ASSUMPTION]
        assert len(edges) == 2
        for edge in edges:
            assert edge.derivation is Derivation.EXTRACTED
            assert edge.directionality is Directionality.DIRECTED
            assert edge.evidence_ids
            assert edge.derived_relation_id is None
            assert not edge.is_hypothesized
            assert edge.target_node == node_for(NodeType.ASSUMPTION_KEY, KEY)

    def test_directly_changes_projects_each_delta(self) -> None:
        """Observed, because a delta is what the change request states."""
        result = run(change=change(assumption_deltas=(delta(),)), bindings=peers())

        edges = result.edges_by_relationship[Relationship.DIRECTLY_CHANGES]
        assert len(edges) == 1
        edge = edges[0]
        assert edge.derivation is Derivation.OBSERVED
        assert edge.evidence_ids == (EVIDENCE,)
        change_nodes = result.nodes_by_type[NodeType.CHANGE]
        assert len(change_nodes) == 1
        assert edge.source_node == change_nodes[0].node_id
        assert edge.target_node == node_for(NodeType.ASSUMPTION_KEY, KEY)

    @pytest.mark.parametrize("relationship", sorted(DERIVED_RELATIONSHIPS))
    def test_each_derived_relationship_projects(self, relationship: Relationship) -> None:
        left, right = peers()
        record = relation(left, right, relationship=relationship)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        edges = result.edges_by_relationship[relationship]
        assert len(edges) == 1
        edge = edges[0]
        assert edge.derivation is Derivation.DERIVED
        assert edge.derived_relation_id == record.relation_id
        assert edge.consequence_id is not None
        # A derived edge carries no evidence of its own: its evidence is the computed
        # transitive union of its inputs (checkpoint 2.4).
        assert edge.evidence_ids == ()
        assert not edge.is_hypothesized

    def test_creates_risk_for_projects_a_class_b_consequence(self) -> None:
        consequence = class_b_consequence()
        result = run(consequences=(consequence,))

        edges = result.edges_by_relationship[Relationship.CREATES_RISK_FOR]
        assert len(edges) == 1
        edge = edges[0]
        assert edge.derivation is Derivation.INFERRED
        assert edge.is_hypothesized
        assert edge.consequence_id == consequence.consequence_id
        assert edge.target_node == node_for(NodeType.ENTITY, consequence.affected_entity)

    def test_historically_analogous_to_projects_an_analogue(self) -> None:
        consequence = class_b_consequence(
            historical_analogue_id=ANALOGUE, status=Status.SPECULATIVE
        )
        result = run(consequences=(consequence,))

        edges = result.edges_by_relationship[Relationship.HISTORICALLY_ANALOGOUS_TO]
        assert len(edges) == 1
        edge = edges[0]
        assert edge.derivation is Derivation.INFERRED
        assert edge.is_hypothesized
        assert edge.target_node == node_for(NodeType.HISTORICAL_EVENT, ANALOGUE)

    def test_all_seven_edge_types_are_reachable_in_one_projection(self) -> None:
        """A relationship nothing can produce is either dead taxonomy or a missing rule."""
        left, right = peers()
        records = tuple(
            relation(left, right, relationship=r) for r in sorted(DERIVED_RELATIONSHIPS)
        )
        result = run(
            change=change(assumption_deltas=(delta(),)),
            bindings=(left, right),
            relations=records,
            consequences=(
                *(class_a_consequence(r) for r in records),
                class_b_consequence(historical_analogue_id=ANALOGUE),
            ),
        )

        produced = {
            relationship for relationship, edges in result.edges_by_relationship.items() if edges
        }
        assert produced == set(Relationship)

    def test_all_five_node_types_are_reachable(self) -> None:
        left, right = peers()
        record = relation(left, right)
        result = run(
            change=change(assumption_deltas=(delta(),)),
            bindings=(left, right),
            relations=(record,),
            consequences=(
                class_a_consequence(record),
                class_b_consequence(historical_analogue_id=ANALOGUE),
            ),
        )

        produced = {node_type for node_type, nodes in result.nodes_by_type.items() if nodes}
        assert produced == set(NodeType)


# ---------------------------------------------------------------------------
# Directionality and symmetry
# ---------------------------------------------------------------------------


class TestSymmetry:
    def test_every_edge_carries_the_taxonomys_directionality(self) -> None:
        left, right = peers()
        records = tuple(
            relation(left, right, relationship=r) for r in sorted(DERIVED_RELATIONSHIPS)
        )
        result = run(
            change=change(assumption_deltas=(delta(),)),
            bindings=(left, right),
            relations=records,
            consequences=(
                *(class_a_consequence(r) for r in records),
                class_b_consequence(historical_analogue_id=ANALOGUE),
            ),
        )

        for edge in result.graph.edges:
            assert edge.directionality is DIRECTIONALITY[edge.relationship]

    def test_a_symmetric_relation_yields_one_canonically_ordered_edge(self) -> None:
        """``conflicts_with`` between A and B is one fact, not two.

        Emitting both directions would double-count every contradiction in the corpus and
        make two projections of one store differ.
        """
        left, right = peers()
        record = relation(left, right, relationship=Relationship.CONFLICTS_WITH)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        edges = result.edges_by_relationship[Relationship.CONFLICTS_WITH]
        assert len(edges) == 1
        assert edges[0].source_node < edges[0].target_node

    def test_the_symmetric_edge_is_the_same_whichever_side_the_relation_names(self) -> None:
        """The canonical ordering is the projector's, not the resolver's."""
        left, right = peers()
        forward = relation(left, right, relationship=Relationship.CONFLICTS_WITH)
        backward = relation(right, left, relationship=Relationship.CONFLICTS_WITH)

        first = run(
            bindings=(left, right),
            relations=(forward,),
            consequences=(class_a_consequence(forward),),
        ).edges_by_relationship[Relationship.CONFLICTS_WITH][0]
        second = run(
            bindings=(left, right),
            relations=(backward,),
            consequences=(class_a_consequence(backward),),
        ).edges_by_relationship[Relationship.CONFLICTS_WITH][0]

        assert (first.source_node, first.target_node) == (
            second.source_node,
            second.target_node,
        )

    def test_a_directed_relation_keeps_the_resolvers_orientation(self) -> None:
        """``unauthorized_override`` runs narrower -> broader: the narrower one deviates."""
        narrower = binding(
            role=DECLARES, value=days(30), scope="region:eu", authority=AuthorityClass.TRAINING
        )
        broader = binding(
            role=DECLARES, value=days(90), scope="global", authority=AuthorityClass.POLICY
        )
        record = relation(
            narrower,
            broader,
            relationship=Relationship.UNAUTHORIZED_OVERRIDE,
            rule_id=RuleId.UNAUTHORIZED_OVERRIDE,
        )
        result = run(
            bindings=(narrower, broader),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        edge = result.edges_by_relationship[Relationship.UNAUTHORIZED_OVERRIDE][0]
        assert edge.source_node == node_for(NodeType.SOURCE_DOCUMENT, narrower.source_id)
        assert edge.target_node == node_for(NodeType.SOURCE_DOCUMENT, broader.source_id)


# ---------------------------------------------------------------------------
# Illegal combinations
# ---------------------------------------------------------------------------


class TestIllegalCombinations:
    @pytest.mark.parametrize("relationship", sorted(Relationship))
    def test_every_inadmissible_derivation_is_refused(self, relationship: Relationship) -> None:
        """The taxonomy is enforced by the entity, so the projector cannot bypass it."""
        admissible = ADMISSIBLE_DERIVATIONS[relationship]
        for derivation in Derivation:
            if derivation in admissible:
                continue
            with pytest.raises(ValidationError, match="admits derivation"):
                GraphEdge(
                    edge_id=new_id("edge"),
                    source_node=new_id("node"),
                    target_node=new_id("node"),
                    relationship=relationship,
                    derivation=derivation,
                    directionality=DIRECTIONALITY[relationship],
                    consequence_id=new_id("consequence"),
                )

    def test_absent_derivation_is_admissible_for_nothing(self) -> None:
        """Which is why an ``EvidenceGap`` is structurally incapable of becoming an edge."""
        for admissible in ADMISSIBLE_DERIVATIONS.values():
            assert Derivation.ABSENT not in admissible

    def test_a_wrong_directionality_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"is symmetric|is directed"):
            GraphEdge(
                edge_id=new_id("edge"),
                source_node="nd_" + "1" * 32,
                target_node="nd_" + "2" * 32,
                relationship=Relationship.CONFLICTS_WITH,
                derivation=Derivation.DERIVED,
                directionality=Directionality.DIRECTED,
                consequence_id=new_id("consequence"),
                derived_relation_id=new_id("relation"),
            )

    def test_a_derived_relation_with_no_consequence_does_not_project(self) -> None:
        """Not a choice: the three derived relationships are consequence-bearing, so the
        edge validator refuses one without a ``consequence_id``. The relation is reported."""
        left, right = peers()
        record = relation(left, right)
        result = run(bindings=(left, right), relations=(record,))

        assert result.edges_by_relationship[Relationship.CONFLICTS_WITH] == ()
        assert [s.reason for s in result.skipped] == [SKIP_NO_CONSEQUENCE]
        assert result.skipped[0].record_id == record.relation_id

    def test_the_three_derived_relationships_are_consequence_bearing(self) -> None:
        assert DERIVED_RELATIONSHIPS <= CONSEQUENCE_BEARING

    def test_a_rejected_consequence_does_not_project_a_live_edge(self) -> None:
        """The arbitration concluded the claim does not hold. Drawing it would show a reader
        a relationship the system decided was false.

        The two ``binds_assumption`` edges survive, and must: those are facts about what the
        documents say, and a rejected *consequence* does not un-bind an assumption record.
        Only the derived relationship the arbitration rejected disappears.
        """
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record, status=Status.REJECTED),),
        )

        assert result.edges_by_relationship[Relationship.CONFLICTS_WITH] == ()
        assert len(result.edges_by_relationship[Relationship.BINDS_ASSUMPTION]) == 2
        assert [s.reason for s in result.skipped] == [SKIP_REJECTED]

    def test_a_rejected_class_b_consequence_does_not_project(self) -> None:
        result = run(consequences=(class_b_consequence(status=Status.REJECTED),))

        assert result.edges_by_relationship[Relationship.CREATES_RISK_FOR] == ()
        assert [s.reason for s in result.skipped] == [SKIP_REJECTED]

    def test_a_relation_whose_endpoint_was_not_supplied_is_reported(self) -> None:
        """Rather than producing a dangling endpoint the graph validator would refuse."""
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left,), relations=(record,), consequences=(class_a_consequence(record),)
        )

        assert result.graph.edges != ()  # the binds_assumption edge still projects
        assert result.edges_by_relationship[Relationship.CONFLICTS_WITH] == ()
        assert SKIP_UNKNOWN_ENDPOINT in {s.reason for s in result.skipped}

    def test_a_change_with_no_evidence_projects_no_directly_changes_edge(self) -> None:
        """An observed edge must cite a span. Manufacturing one to satisfy the validator
        would be exactly the fabricated provenance the validator exists to prevent."""
        result = run(change=change(evidence_ids=(), assumption_deltas=(delta(),)))

        assert result.edges_by_relationship[Relationship.DIRECTLY_CHANGES] == ()
        assert [s.reason for s in result.skipped] == [SKIP_NO_CHANGE_EVIDENCE]

    def test_a_self_loop_is_impossible(self) -> None:
        with pytest.raises(ValidationError, match="cannot relate to itself"):
            node = "nd_" + "1" * 32
            GraphEdge(
                edge_id=new_id("edge"),
                source_node=node,
                target_node=node,
                relationship=Relationship.CONFLICTS_WITH,
                derivation=Derivation.DERIVED,
                directionality=Directionality.SYMMETRIC,
                consequence_id=new_id("consequence"),
                derived_relation_id=new_id("relation"),
            )

    def test_an_intra_document_relation_is_skipped_rather_than_looped(self) -> None:
        """One document disagreeing with itself is an anomaly, not a relationship."""
        held = binding(role=DECLARES, value=days(90), scope="global", tag="same")
        twin = held.model_copy(
            update={
                "record": held.record.model_copy(
                    update={"assumption_id": "as_" + "9" * 32, "value": days(30)}
                )
            }
        )
        record = relation(held, twin)
        result = run(
            bindings=(held, twin),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        assert result.edges_by_relationship[Relationship.CONFLICTS_WITH] == ()
        assert "endpoints_identical" in {s.reason for s in result.skipped}


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


class TestDuplicatePrevention:
    def test_one_relation_projected_twice_yields_one_edge(self) -> None:
        """A ``conflicts_with`` relation yields a Class A finding per affected artifact, so
        the projector visits it twice. Both visits derive the same edge id."""
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left, right),
            relations=(record, record),
            consequences=(class_a_consequence(record),),
        )

        assert len(result.edges_by_relationship[Relationship.CONFLICTS_WITH]) == 1

    def test_the_same_binding_supplied_twice_yields_one_edge(self) -> None:
        left, _ = peers()
        result = run(bindings=(left, left))

        assert len(result.edges_by_relationship[Relationship.BINDS_ASSUMPTION]) == 1
        assert len(result.nodes_by_type[NodeType.SOURCE_DOCUMENT]) == 1

    def test_two_relations_over_one_pair_are_two_edges(self) -> None:
        """A contradiction on one key and a staleness on another are two facts.

        The edge id includes the record, so distinguishing them is not a special case.
        """
        left, right = peers()
        conflict = relation(left, right, relationship=Relationship.CONFLICTS_WITH)
        stale = relation(
            left,
            right,
            relationship=Relationship.INVALIDATES_ASSUMPTION,
            rule_id=RuleId.STALE_CONSUMER,
        )
        result = run(
            bindings=(left, right),
            relations=(conflict, stale),
            consequences=(class_a_consequence(conflict), class_a_consequence(stale)),
        )

        assert len(result.graph.edges) == 4  # two binds_assumption, two derived
        assert len({e.edge_id for e in result.graph.edges}) == 4

    def test_node_and_edge_ids_are_unique_by_construction(self) -> None:
        """The graph validator refuses a repeat, so a duplicate would be a hard failure."""
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        node_ids = [n.node_id for n in result.graph.nodes]
        edge_ids = [e.edge_id for e in result.graph.edges]
        assert len(node_ids) == len(set(node_ids))
        assert len(edge_ids) == len(set(edge_ids))

    def test_an_assumption_key_and_an_entity_sharing_a_label_stay_apart(self) -> None:
        """Node identity includes the type, so two kinds of thing cannot collapse into one."""
        assert node_for(NodeType.ASSUMPTION_KEY, KEY) != node_for(NodeType.ENTITY, KEY)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_a_derived_edge_names_the_relation_it_projects(self) -> None:
        """Or its rule and inputs are unauditable."""
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        edge = result.edges_by_relationship[Relationship.CONFLICTS_WITH][0]
        assert edge.derived_relation_id == record.relation_id

    def test_consequence_ids_are_preserved_not_reminted(self) -> None:
        left, right = peers()
        record = relation(left, right)
        class_a = class_a_consequence(record)
        class_b = class_b_consequence(historical_analogue_id=ANALOGUE)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a, class_b),
        )

        by_relationship = result.edges_by_relationship
        assert (
            by_relationship[Relationship.CONFLICTS_WITH][0].consequence_id == class_a.consequence_id
        )
        assert (
            by_relationship[Relationship.CREATES_RISK_FOR][0].consequence_id
            == class_b.consequence_id
        )
        assert (
            by_relationship[Relationship.HISTORICALLY_ANALOGOUS_TO][0].consequence_id
            == class_b.consequence_id
        )

    def test_no_edge_carries_a_confidence_or_a_severity(self) -> None:
        """They live on the consequence and the score. Duplicated scores diverge."""
        fields = set(GraphEdge.model_fields)
        assert not {"confidence", "severity", "status", "produced_by"} & fields

    def test_an_edge_cannot_be_attributed_to_a_model(self) -> None:
        """There is nowhere to record which model authored an edge, because nothing may."""
        with pytest.raises(ValidationError, match=r"forbidden|extra"):
            GraphEdge.model_validate(
                {
                    "edge_id": new_id("edge"),
                    "source_node": "nd_" + "1" * 32,
                    "target_node": "nd_" + "2" * 32,
                    "relationship": Relationship.CONFLICTS_WITH,
                    "derivation": Derivation.DERIVED,
                    "directionality": Directionality.SYMMETRIC,
                    "consequence_id": new_id("consequence"),
                    "derived_relation_id": new_id("relation"),
                    "produced_by": {"agent": "consequence-hunter"},
                }
            )

    def test_the_projector_cannot_import_an_agent_runtime(self) -> None:
        """Checkpoint 10.2 as a property of the import graph.

        Parsed rather than grepped: a substring search matches the docstring explaining the
        rule, which is a test that reads correctly and asserts nothing.
        """
        source = pathlib.Path("app/graph/projection.py").read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        forbidden = (
            "app.agents",
            "app.consequences",
            "app.contradictions",
            "app.challenge",
            "app.judgment",
            "app.retrieval",
            "app.infrastructure",
        )
        for prefix in forbidden:
            assert not any(m.startswith(prefix) for m in imported), prefix

    def test_the_projector_signature_takes_only_records_and_a_timestamp(self) -> None:
        """No store, no clock, no model. A pure function of admitted records."""
        parameters = set(inspect.signature(project).parameters)

        assert parameters == {
            "analysis_id",
            "change",
            "bindings",
            "relations",
            "consequences",
            "gaps",
            "registry_version",
            "coupling_index_version",
            "projected_at",
        }

    def test_the_graph_declares_itself_a_projection(self) -> None:
        result = run(bindings=peers())

        assert result.graph.is_projection is True
        assert result.graph.registry_version == REGISTRY_VERSION
        assert result.graph.coupling_index_version == COUPLING_INDEX_VERSION
        assert result.projector_version == GRAPH_PROJECTION_VERSION

    def test_a_projection_cannot_be_relabelled_as_authored(self) -> None:
        """``is_projection`` is a ``Literal``, so there is no writeable-graph payload."""
        with pytest.raises(ValidationError):
            ConsequenceGraph.model_validate(
                {
                    "is_projection": False,
                    "analysis_id": ANALYSIS,
                    "registry_version": REGISTRY_VERSION,
                    "coupling_index_version": COUPLING_INDEX_VERSION,
                    "projected_at": PROJECTED_AT,
                }
            )


# ---------------------------------------------------------------------------
# Class B linkage
# ---------------------------------------------------------------------------


class TestClassBLinkage:
    def test_a_class_b_edge_is_structurally_distinct_from_a_derivation(self) -> None:
        """So a client cannot render a hypothesis identically to a derivation (10.6)."""
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record), class_b_consequence()),
        )

        hypothesized = result.graph.hypothesized_edges
        derived = result.graph.derived_edges
        assert {e.relationship for e in hypothesized} == {Relationship.CREATES_RISK_FOR}
        assert {e.relationship for e in derived} == {Relationship.CONFLICTS_WITH}
        assert not set(hypothesized) & set(derived)

    def test_a_class_b_edge_may_cite_its_spans(self) -> None:
        """Legitimate for an inferred edge and forbidden for a derived one: a hypothesis
        genuinely rests on the spans it cites, while a derivation's evidence is the computed
        union of its inputs."""
        result = run(consequences=(class_b_consequence(),))

        edge = result.edges_by_relationship[Relationship.CREATES_RISK_FOR][0]
        assert edge.evidence_ids == (EVIDENCE,)

    def test_a_class_b_consequence_with_no_analogue_projects_one_edge(self) -> None:
        result = run(consequences=(class_b_consequence(historical_analogue_id=None),))

        assert len(result.edges_by_relationship[Relationship.CREATES_RISK_FOR]) == 1
        assert result.edges_by_relationship[Relationship.HISTORICALLY_ANALOGOUS_TO] == ()

    def test_a_class_a_consequence_produces_no_inferred_edge(self) -> None:
        """Class A is derived. It has no hypothesis to draw."""
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        assert result.graph.hypothesized_edges == ()


# ---------------------------------------------------------------------------
# EvidenceGap
# ---------------------------------------------------------------------------


class TestEvidenceGaps:
    def test_a_gap_never_produces_an_edge(self) -> None:
        """Type E is a claim about the corpus. Inventing a relationship out of silence is
        the structural form of treating absence of evidence as evidence of absence."""
        left, right = peers()
        gaps = tuple(gap(partition=KEY) for _ in range(3))

        without = run(bindings=(left, right))
        with_gaps = run(bindings=(left, right), gaps=gaps)

        assert len(with_gaps.graph.edges) == len(without.graph.edges)
        gap_ids = {g.gap_id for g in gaps}
        for edge in with_gaps.graph.edges:
            assert edge.source_node not in gap_ids
            assert edge.target_node not in gap_ids
            assert not set(edge.evidence_ids) & gap_ids

    def test_a_gap_annotates_the_node_its_partition_names(self) -> None:
        """The only place a type E claim may touch the graph."""
        left, right = peers()
        record = gap(partition=KEY)
        result = run(bindings=(left, right), gaps=(record,))

        key_node = next(n for n in result.graph.nodes if n.entity_type is NodeType.ASSUMPTION_KEY)
        assert key_node.evidence_gap_ids == (record.gap_id,)

    def test_a_gap_naming_no_node_is_reported_rather_than_dropped(self) -> None:
        """It still attaches to the analysis; it simply annotates no vertex."""
        left, right = peers()
        unmatched = gap(partition="a key nobody bound")
        no_partition = gap(partition=None)
        result = run(bindings=(left, right), gaps=(unmatched, no_partition))

        reasons = [s.reason for s in result.skipped]
        assert reasons == [SKIP_GAP_NODE_UNKNOWN, SKIP_GAP_NODE_UNKNOWN]
        assert {s.record_id for s in result.skipped} == {
            unmatched.gap_id,
            no_partition.gap_id,
        }

    def test_the_gap_annotation_field_exists_only_on_nodes(self) -> None:
        assert "evidence_gap_ids" in set(GraphNode.model_fields)
        assert "evidence_gap_ids" not in set(GraphEdge.model_fields)

    def test_an_evidence_gap_carries_the_absent_derivation_nothing_admits(self) -> None:
        assert gap().derivation is Derivation.ABSENT
        for admissible in ADMISSIBLE_DERIVATIONS.values():
            assert gap().derivation not in admissible


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _inputs(self) -> dict[str, Any]:
        left, right = peers()
        record = relation(left, right)
        return {
            "change": change(assumption_deltas=(delta(),)),
            "bindings": (left, right),
            "relations": (record,),
            "consequences": (
                class_a_consequence(record),
                class_b_consequence(historical_analogue_id=ANALOGUE),
            ),
            "gaps": (gap(partition=KEY),),
        }

    def test_two_rebuilds_are_byte_identical(self) -> None:
        """Requirement 12, and the reason the graph is comparable at all.

        Content-derived ids, sorted collections, and a caller-supplied ``projected_at`` -
        which also pins ``created_at``, since a default clock read would make every rebuild
        differ in a field nobody set.
        """
        inputs = self._inputs()

        first = run(**inputs).graph.model_dump_json()
        second = run(**inputs).graph.model_dump_json()

        assert first == second

    def test_input_ordering_does_not_change_the_output(self) -> None:
        inputs = self._inputs()
        reversed_inputs = {
            **inputs,
            "bindings": tuple(reversed(inputs["bindings"])),
            "consequences": tuple(reversed(inputs["consequences"])),
        }

        assert (
            run(**inputs).graph.model_dump_json() == run(**reversed_inputs).graph.model_dump_json()
        )

    def test_node_and_edge_ids_are_content_derived(self) -> None:
        """A minted id would make every rebuild produce a different graph, and every stored
        edge dangle after one."""
        left, right = peers()
        record = relation(left, right)
        result = run(
            bindings=(left, right),
            relations=(record,),
            consequences=(class_a_consequence(record),),
        )

        edge = result.edges_by_relationship[Relationship.CONFLICTS_WITH][0]
        assert edge.edge_id == edge_id_for(
            relationship=Relationship.CONFLICTS_WITH.value,
            source_node=edge.source_node,
            target_node=edge.target_node,
            record_id=record.relation_id,
        )
        assert node_for(NodeType.ASSUMPTION_KEY, KEY) in {n.node_id for n in result.graph.nodes}

    def test_the_projector_reads_no_clock(self) -> None:
        """``projected_at`` is the caller's, so nothing about the output depends on when."""
        source = pathlib.Path("app/graph/projection.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "now" not in called
        assert "utcnow" not in called

    def test_the_empty_projection_is_well_formed(self) -> None:
        """One change node, no edges, and nothing invented to fill the space."""
        result = run()

        assert len(result.graph.nodes) == 1
        assert result.graph.nodes[0].entity_type is NodeType.CHANGE
        assert result.graph.edges == ()
        assert result.skipped == ()


# ---------------------------------------------------------------------------
# The retention.window join, over the real corpus
# ---------------------------------------------------------------------------


class TestTheRetentionWindowJoin:
    """The join the ``assumption_key`` node type exists for.

    Checkpoint 10.4: ``AssumptionKey`` is a node type *because that is what makes hidden
    coupling visible* - two documents joined through a key neither mentions is the central
    visualization.

    ``retention.window`` is the case ``docs/EVALUATION.md`` measured: five artifacts bind it
    and lexical retrieval recovered one of them, so a reader relying on text search would see
    a single document while the graph shows the whole coupling set. If the projection does not
    make that visible, the projection is wrong rather than the corpus.
    """

    def test_the_key_node_joins_all_five_holders(self, corpus_projection: ProjectionResult) -> None:
        """Measured: five artifacts bind ``retention.window`` and the graph shows all five.

        Lexical retrieval recovered **one** of them (``docs/EVALUATION.md``, coupling recall),
        so this is the gap between what text search shows a reader and what co-binding
        establishes - which is the entire argument for the mechanism.
        """
        holders = corpus_projection.key_holders(KEY)

        assert len(holders) == 5, holders
        assert node_for(NodeType.ASSUMPTION_KEY, KEY) in {
            n.node_id for n in corpus_projection.graph.nodes
        }

    def test_the_holders_span_unrelated_kinds_of_artifact(
        self, corpus_projection: ProjectionResult
    ) -> None:
        """Distinct documents of *different kinds*, not one policy quoted five times.

        A key bound five times inside one document is not hidden coupling. The measured set
        spans a retention policy, a regional policy, a metric definition, vendor notes and
        team notes - and nobody reading the retention policy would think to check the metric
        definition, which is exactly the coupling a reader cannot find by reading.
        """
        holders = corpus_projection.key_holders(KEY)

        assert len(set(holders)) == len(holders) == 5
        assert "Data Retention Policy" in holders
        # At least one holder that is not a policy at all.
        assert any("Policy" not in holder for holder in holders)

    def test_the_holder_documents_are_five_distinct_nodes(
        self, corpus_projection: ProjectionResult
    ) -> None:
        """Five source-document nodes, not one node reached five times."""
        key_node = node_for(NodeType.ASSUMPTION_KEY, KEY)
        holders = {
            edge.source_node
            for edge in corpus_projection.graph.edges
            if edge.relationship is Relationship.BINDS_ASSUMPTION and edge.target_node == key_node
        }
        assert len(holders) == 5

    def test_the_join_is_visible_without_any_document_citing_another(
        self, corpus_projection: ProjectionResult
    ) -> None:
        """The whole claim: the coupling is derived from co-binding, not cross-reference.

        There is no edge type for "document A mentions document B", so every path between two
        holders in this graph runs *through* the assumption-key node.
        """
        key_node = node_for(NodeType.ASSUMPTION_KEY, KEY)
        binds = [
            e
            for e in corpus_projection.graph.edges
            if e.relationship is Relationship.BINDS_ASSUMPTION and e.target_node == key_node
        ]
        holders = {e.source_node for e in binds}

        document_to_document = [
            e
            for e in corpus_projection.graph.edges
            if e.source_node in holders
            and e.target_node in holders
            and e.relationship is not Relationship.BINDS_ASSUMPTION
        ]
        # Any direct edge between two holders is a *derived* consequence of the join, never a
        # citation - the taxonomy has no "mentions" edge to express one.
        for edge in document_to_document:
            assert edge.derivation is Derivation.DERIVED

    def test_every_binds_assumption_edge_cites_its_span(
        self, corpus_projection: ProjectionResult
    ) -> None:
        for edge in corpus_projection.graph.edges:
            if edge.relationship is Relationship.BINDS_ASSUMPTION:
                assert edge.evidence_ids
                assert edge.derivation is Derivation.EXTRACTED

    def test_the_corpus_projection_rebuilds_identically(
        self, corpus_analysis: tuple[CouplingAnalysis, tuple[ResolvedBinding, ...], str]
    ) -> None:
        """Requirement 12 over the real store rather than synthetic records."""
        analysis, bindings, tenant_id = corpus_analysis
        kwargs: dict[str, Any] = {
            "analysis_id": ANALYSIS,
            "change": change(),
            "bindings": bindings,
            "relations": analysis.index.relations,
            "registry_version": REGISTRY_VERSION,
            "coupling_index_version": COUPLING_INDEX_VERSION,
            "projected_at": PROJECTED_AT,
        }
        _ = tenant_id

        assert (
            project(**kwargs).graph.model_dump_json() == project(**kwargs).graph.model_dump_json()
        )

    def test_unarbitrated_relations_are_reported_not_drawn(
        self, corpus_projection: ProjectionResult
    ) -> None:
        """The corpus fixture supplies no arbitrated consequences, so every derived relation
        the resolver found is skipped with a reason rather than silently absent."""
        reasons = {s.reason for s in corpus_projection.skipped}
        assert reasons <= {SKIP_NO_CONSEQUENCE, SKIP_GAP_NODE_UNKNOWN}
        assert any(s.kind == "derived_relation" for s in corpus_projection.skipped)


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_analysis() -> tuple[CouplingAnalysis, tuple[ResolvedBinding, ...], str]:
    """The real coupling analysis over the sample corpus, with its resolved bindings."""
    stores = asyncio.run(load_corpus())
    assert not stores.extraction_failures, stores.extraction_failures

    use_case = ResolveCouplingUseCase(
        sources=stores.sources,
        assumptions=stores.assumptions,
        gaps=stores.gaps,
        index_store=InMemoryCouplingIndexStore(),
    )
    analysis = use_case.execute(
        tenant_id=stores.tenant_id,
        as_of=NOW,
        registry_version=stores.registry_version,
    )
    bindings, _unresolvable = use_case.assemble(stores.tenant_id)
    return analysis, bindings, stores.tenant_id


@pytest.fixture(scope="module")
def corpus_projection(
    corpus_analysis: tuple[CouplingAnalysis, tuple[ResolvedBinding, ...], str],
) -> ProjectionResult:
    analysis, bindings, _tenant = corpus_analysis
    return project(
        analysis_id=ANALYSIS,
        change=change(),
        bindings=bindings,
        relations=analysis.index.relations,
        registry_version=REGISTRY_VERSION,
        coupling_index_version=COUPLING_INDEX_VERSION,
        projected_at=PROJECTED_AT,
    )
