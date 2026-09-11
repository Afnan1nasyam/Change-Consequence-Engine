"""The graph projection.

The graph is the easiest place in the system to quietly acquire a second source of truth,
so most of this module is negative. The load-bearing assertions:

* ``test_no_relationship_admits_an_absent_derivation`` - an EvidenceGap can never become an
  edge, and that is a property of the closed taxonomy rather than a rule someone remembers.
* ``test_an_edge_has_no_model_attribution`` - there is nowhere to record which model
  authored an edge, because nothing may.
* ``test_a_derived_edge_must_not_carry_direct_evidence`` - the type C evidence rule survives
  into the projection.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.entities.graph import ConsequenceGraph, GraphEdge, GraphNode
from app.domain.value_objects.enums import Derivation
from app.domain.value_objects.taxonomy import (
    ADMISSIBLE_DERIVATIONS,
    CONSEQUENCE_BEARING,
    DIRECTIONALITY,
    PROJECTION_ONLY,
    Directionality,
    NodeType,
    Relationship,
)
from tests.fixtures.domain import (
    assumption_key_node,
    binds_edge,
    consequence_graph,
    derived_edge,
    graph_node,
    hypothesized_edge,
    make_id,
)

# ---------------------------------------------------------------------------
# Taxonomy closure
# ---------------------------------------------------------------------------


def test_the_taxonomy_is_seven_edges() -> None:
    assert {r.value for r in Relationship} == {
        "directly_changes",
        "binds_assumption",
        "invalidates_assumption",
        "conflicts_with",
        "unauthorized_override",
        "creates_risk_for",
        "historically_analogous_to",
    }


def test_there_is_no_catch_all_edge() -> None:
    """A permissive catch-all absorbs most of the graph and stops carrying information."""
    assert "potentially_impacts" not in {r.value for r in Relationship}


def test_every_relationship_declares_its_admissible_derivations_and_direction() -> None:
    """A relationship missing from either table would validate by accident."""
    for relationship in Relationship:
        assert ADMISSIBLE_DERIVATIONS[relationship]
        assert relationship in DIRECTIONALITY


def test_no_relationship_admits_an_absent_derivation() -> None:
    """Type E never projects to an edge.

    Inventing a relationship out of silence is the structural form of treating absence of
    evidence as evidence of absence. Nothing enforces this at the edge level because
    nothing needs to: no edge type can carry `absent`.
    """
    for relationship, admissible in ADMISSIBLE_DERIVATIONS.items():
        assert Derivation.ABSENT not in admissible, relationship


def test_conflicts_with_is_the_only_symmetric_edge() -> None:
    symmetric = {r for r, d in DIRECTIONALITY.items() if d is Directionality.SYMMETRIC}
    assert symmetric == {Relationship.CONFLICTS_WITH}


def test_unauthorized_override_is_directed_narrower_to_broader() -> None:
    """The narrower artifact is the one deviating, so the edge is not symmetric."""
    assert DIRECTIONALITY[Relationship.UNAUTHORIZED_OVERRIDE] is Directionality.DIRECTED


def test_historically_analogous_to_is_inferred_not_observed() -> None:
    """The incident is observed; the analogy is a judgment."""
    assert ADMISSIBLE_DERIVATIONS[Relationship.HISTORICALLY_ANALOGOUS_TO] == frozenset(
        {Derivation.INFERRED}
    )


def test_the_node_types_are_the_five_mvp_types() -> None:
    assert {n.value for n in NodeType} == {
        "source_document",
        "assumption_key",
        "entity",
        "change",
        "historical_event",
    }


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------


def test_an_assumption_key_is_a_first_class_node() -> None:
    """This is what makes hidden coupling visible in the central visualization."""
    node = assumption_key_node()
    assert node.entity_type is NodeType.ASSUMPTION_KEY
    assert node.label == "refund.approval_threshold"


def test_a_gap_may_annotate_a_node_without_becoming_an_edge() -> None:
    node = assumption_key_node()
    assert node.evidence_gap_ids
    assert node.annotations == ("single holder",)


def test_node_gap_references_must_be_well_shaped() -> None:
    with pytest.raises(ValidationError):
        graph_node(evidence_gap_ids=("gap about ownership",))


def test_a_node_carries_no_score() -> None:
    with pytest.raises(ValidationError):
        GraphNode(**{**graph_node().model_dump(), "confidence": 0.5})
    with pytest.raises(ValidationError):
        GraphNode(**{**graph_node().model_dump(), "severity": "HIGH"})


# ---------------------------------------------------------------------------
# GraphEdge - scores and authorship
# ---------------------------------------------------------------------------


def test_an_edge_carries_no_confidence_and_no_severity() -> None:
    """They live on the consequence and its score. Duplicated scores diverge."""
    assert "confidence" not in GraphEdge.model_fields
    assert "severity" not in GraphEdge.model_fields
    with pytest.raises(ValidationError):
        GraphEdge(**{**derived_edge().model_dump(), "confidence": 0.9})
    with pytest.raises(ValidationError):
        GraphEdge(**{**derived_edge().model_dump(), "severity": "CRITICAL"})


def test_an_edge_has_no_model_attribution() -> None:
    """No agent writes an edge, so there is nowhere to record that one did."""
    for field in ("produced_by", "proposed_by", "agent", "rationale", "prompt_sha256"):
        assert field not in GraphEdge.model_fields
        with pytest.raises(ValidationError):
            GraphEdge(**{**derived_edge().model_dump(), field: "consequence-hunter"})


# ---------------------------------------------------------------------------
# GraphEdge - derivation admissibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relationship", "derivation"),
    [
        (Relationship.INVALIDATES_ASSUMPTION, Derivation.INFERRED),
        (Relationship.CONFLICTS_WITH, Derivation.OBSERVED),
        (Relationship.UNAUTHORIZED_OVERRIDE, Derivation.EXTRACTED),
        (Relationship.CREATES_RISK_FOR, Derivation.DERIVED),
        (Relationship.HISTORICALLY_ANALOGOUS_TO, Derivation.OBSERVED),
        (Relationship.DIRECTLY_CHANGES, Derivation.DERIVED),
        (Relationship.BINDS_ASSUMPTION, Derivation.DERIVED),
    ],
)
def test_an_inadmissible_derivation_is_rejected(
    relationship: Relationship, derivation: Derivation
) -> None:
    """A model-inferred edge must not be able to present itself as a deterministic one."""
    with pytest.raises(ValidationError, match="admits"):
        derived_edge(relationship=relationship, derivation=derivation)


def test_each_relationship_validates_with_its_own_derivation() -> None:
    for relationship in Relationship:
        derivation = next(iter(ADMISSIBLE_DERIVATIONS[relationship]))
        edge = GraphEdge(
            edge_id=make_id("edge"),
            source_node=make_id("node"),
            target_node=make_id("node", 2),
            relationship=relationship,
            derivation=derivation,
            directionality=DIRECTIONALITY[relationship],
            consequence_id=(
                make_id("consequence") if relationship in CONSEQUENCE_BEARING else None
            ),
            derived_relation_id=(make_id("relation") if derivation is Derivation.DERIVED else None),
            evidence_ids=(
                (make_id("evidence"),)
                if derivation in (Derivation.OBSERVED, Derivation.EXTRACTED)
                else ()
            ),
            is_hypothesized=derivation is Derivation.INFERRED,
        )
        assert edge.relationship is relationship


# ---------------------------------------------------------------------------
# GraphEdge - direction
# ---------------------------------------------------------------------------


def test_direction_is_a_property_of_the_relationship() -> None:
    with pytest.raises(ValidationError, match="is directed"):
        derived_edge(
            relationship=Relationship.INVALIDATES_ASSUMPTION,
            directionality=Directionality.SYMMETRIC,
        )
    with pytest.raises(ValidationError, match="is symmetric"):
        derived_edge(
            relationship=Relationship.CONFLICTS_WITH, directionality=Directionality.DIRECTED
        )


def test_a_symmetric_edge_is_stored_once_in_canonical_order() -> None:
    """Two copies of one contradiction would double-count it and break idempotence."""
    with pytest.raises(ValidationError, match="canonical"):
        derived_edge(
            relationship=Relationship.CONFLICTS_WITH,
            directionality=Directionality.SYMMETRIC,
            source_node=make_id("node", 2),
            target_node=make_id("node"),
        )
    ok = derived_edge(
        relationship=Relationship.CONFLICTS_WITH,
        directionality=Directionality.SYMMETRIC,
        source_node=make_id("node"),
        target_node=make_id("node", 2),
    )
    assert ok.directionality is Directionality.SYMMETRIC


def test_an_edge_cannot_be_a_self_loop() -> None:
    with pytest.raises(ValidationError, match="cannot relate to itself"):
        derived_edge(target_node=make_id("node"))


# ---------------------------------------------------------------------------
# GraphEdge - back-references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relationship", sorted(CONSEQUENCE_BEARING))
def test_a_consequence_bearing_edge_must_reference_its_consequence(
    relationship: Relationship,
) -> None:
    derivation = next(iter(ADMISSIBLE_DERIVATIONS[relationship]))
    with pytest.raises(ValidationError, match="requires consequence_id"):
        GraphEdge(
            edge_id=make_id("edge"),
            source_node=make_id("node"),
            target_node=make_id("node", 2),
            relationship=relationship,
            derivation=derivation,
            directionality=DIRECTIONALITY[relationship],
            derived_relation_id=(make_id("relation") if derivation is Derivation.DERIVED else None),
            is_hypothesized=derivation is Derivation.INFERRED,
        )


def test_a_projection_only_edge_needs_no_consequence() -> None:
    """`binds_assumption` states a fact about a record, not a consequence claim."""
    assert Relationship.BINDS_ASSUMPTION in PROJECTION_ONLY
    assert binds_edge().consequence_id is None


def test_a_derived_edge_must_name_the_relation_it_projects() -> None:
    with pytest.raises(ValidationError, match="unauditable"):
        derived_edge(derived_relation_id=None)


def test_a_non_derived_edge_must_not_claim_a_derived_relation() -> None:
    with pytest.raises(ValidationError, match="not produced by the resolver"):
        hypothesized_edge(derived_relation_id=make_id("relation"))
    with pytest.raises(ValidationError, match="not produced by the resolver"):
        binds_edge(derived_relation_id=make_id("relation"))


# ---------------------------------------------------------------------------
# GraphEdge - evidence
# ---------------------------------------------------------------------------


def test_a_derived_edge_must_not_carry_direct_evidence() -> None:
    """Its evidence is the transitive union of its inputs, materialized on read."""
    with pytest.raises(ValidationError, match="fabricated provenance"):
        derived_edge(evidence_ids=(make_id("evidence"),))


def test_an_extracted_or_observed_edge_must_cite_its_span() -> None:
    with pytest.raises(ValidationError, match="requires at"):
        binds_edge(evidence_ids=())


def test_a_hypothesized_edge_may_cite_nothing() -> None:
    assert hypothesized_edge().evidence_ids == ()


# ---------------------------------------------------------------------------
# GraphEdge - hypothesis marking
# ---------------------------------------------------------------------------


def test_a_hypothesized_edge_is_structurally_distinct() -> None:
    """A client must not be able to render a hypothesis like a derivation."""
    assert hypothesized_edge().is_hypothesized
    assert not derived_edge().is_hypothesized
    assert not binds_edge().is_hypothesized


def test_the_hypothesis_flag_cannot_disagree_with_the_derivation() -> None:
    with pytest.raises(ValidationError, match="contradicts"):
        hypothesized_edge(is_hypothesized=False)
    with pytest.raises(ValidationError, match="contradicts"):
        derived_edge(is_hypothesized=True)


# ---------------------------------------------------------------------------
# ConsequenceGraph
# ---------------------------------------------------------------------------


def test_a_projection_records_what_it_was_rebuilt_from() -> None:
    graph = consequence_graph()
    assert graph.is_projection is True
    assert graph.registry_version == "registry-1"
    assert graph.coupling_index_version == "coupling-1"


def test_a_projection_cannot_declare_itself_authored() -> None:
    with pytest.raises(ValidationError):
        consequence_graph(is_projection=False)


def test_a_projection_requires_the_versions_that_make_it_reproducible() -> None:
    for field in ("registry_version", "coupling_index_version", "projected_at"):
        payload = consequence_graph().model_dump()
        del payload[field]
        with pytest.raises(ValidationError):
            ConsequenceGraph(**payload)


def test_an_edge_endpoint_must_be_a_projected_node() -> None:
    """A dangling endpoint means the graph was assembled rather than derived."""
    with pytest.raises(ValidationError, match="referencing absent nodes"):
        consequence_graph(nodes=(graph_node(),))


def test_a_projection_refuses_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="repeats node ids"):
        consequence_graph(nodes=(graph_node(), assumption_key_node(), graph_node()))
    with pytest.raises(ValidationError, match="repeats edge ids"):
        consequence_graph(edges=(derived_edge(), derived_edge()))


def test_a_projection_separates_hypothesized_from_derived_edges() -> None:
    graph = consequence_graph()
    assert graph.hypothesized_edges == (hypothesized_edge(),)
    assert graph.derived_edges == (derived_edge(),)


def test_an_empty_projection_is_valid() -> None:
    """A quiet change must be able to project approximately nothing."""
    graph = consequence_graph(nodes=(), edges=())
    assert graph.nodes == ()
    assert graph.edges == ()
