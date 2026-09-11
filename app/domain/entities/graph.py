"""The consequence graph: nodes, edges, and the projection that holds them.

The graph is a **pure projection** (checkpoint 10.1). It is rebuilt deterministically from
the assumption store, the derived relations, and the arbitrated consequences. Three design
decisions in this module are what make that claim structural rather than aspirational:

* **No agent writes an edge.** :class:`GraphEdge` has no ``ModelProvenance`` field at all.
  There is nowhere to record which model authored an edge, because nothing may. An edge
  points back at the record it projects - ``derived_relation_id``, ``consequence_id``,
  ``evidence_ids`` - and carries no authorship of its own.
* **Edges carry no scores.** ``confidence`` lives on ``ConsequenceScore`` and ``severity``
  on the consequence. With ``extra="forbid"``, an edge payload containing either is a
  validation error. Duplicated scores diverge.
* **Type E never projects.** No relationship in the taxonomy admits ``Derivation.ABSENT``,
  so an ``EvidenceGap`` is structurally incapable of becoming an edge. A gap may
  *annotate* a node (:attr:`GraphNode.evidence_gap_ids`); inventing a relationship out of
  silence is the structural form of treating absence of evidence as evidence of absence.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.value_objects.enums import Derivation
from app.domain.value_objects.identifiers import (
    ConsequenceId,
    EdgeId,
    EvidenceId,
    GapId,
    NodeId,
    RelationId,
    VersionLabel,
)
from app.domain.value_objects.taxonomy import (
    ADMISSIBLE_DERIVATIONS,
    CONSEQUENCE_BEARING,
    DIRECTIONALITY,
    Directionality,
    NodeType,
    Relationship,
)

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]

#: Derivations whose edges must cite the span they project from.
_EVIDENCE_REQUIRED: frozenset[Derivation] = frozenset({Derivation.OBSERVED, Derivation.EXTRACTED})


class GraphNode(DomainModel):
    """A vertex in the projection.

    ``assumption_key`` is a node type *because that is what makes hidden coupling visible*:
    two documents joined through a key neither mentions is the central visualization
    (checkpoint 10.4).

    ``evidence_gap_ids`` is the only place a type E claim may touch the graph. It annotates
    a node - "this key has a single holder" - and never relates two of them.
    """

    node_id: NodeId
    entity_type: NodeType
    label: ShortText
    metadata: dict[str, str] = Field(default_factory=dict)
    annotations: tuple[ShortText, ...] = ()
    evidence_gap_ids: tuple[GapId, ...] = ()


class GraphEdge(DomainModel):
    """An edge in the projection. Deterministically derived, never authored.

    Every field is either an identity, a taxonomy fact, or a back-reference to the record
    being projected. There is deliberately no rationale text and no model attribution: an
    edge that could explain itself in prose would be a second source of truth.
    """

    edge_id: EdgeId
    source_node: NodeId
    target_node: NodeId

    relationship: Relationship
    derivation: Derivation
    directionality: Directionality

    #: Required for consequence-bearing relationships; the score lives there, not here.
    consequence_id: ConsequenceId | None = None
    #: Required when ``derivation`` is ``derived`` - the type C record this projects.
    derived_relation_id: RelationId | None = None
    evidence_ids: tuple[EvidenceId, ...] = ()

    #: Structurally distinct so a client cannot render a hypothesis like a derivation
    #: (checkpoint 10.6).
    is_hypothesized: bool = False

    @model_validator(mode="after")
    def _derivation_is_admissible(self) -> Self:
        """A model-inferred edge may not masquerade as a deterministic one."""
        admissible = ADMISSIBLE_DERIVATIONS[self.relationship]
        if self.derivation not in admissible:
            raise ValueError(
                f"edge {self.edge_id}: relationship={self.relationship.value!r} admits "
                f"derivation {sorted(d.value for d in admissible)}, got "
                f"{self.derivation.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _directionality_matches_taxonomy(self) -> Self:
        """Direction is a property of the relationship, not a per-edge choice."""
        expected = DIRECTIONALITY[self.relationship]
        if self.directionality is not expected:
            raise ValueError(
                f"edge {self.edge_id}: relationship={self.relationship.value!r} is "
                f"{expected.value}, got {self.directionality.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _no_self_loop(self) -> Self:
        if self.source_node == self.target_node:
            raise ValueError(
                f"edge {self.edge_id}: source_node and target_node are identical "
                f"({self.source_node}); an artifact cannot relate to itself"
            )
        return self

    @model_validator(mode="after")
    def _symmetric_edges_are_canonically_ordered(self) -> Self:
        """One symmetric edge, not two.

        ``conflicts_with`` between A and B is a single fact. Emitting both A->B and B->A
        would double-count every contradiction and make two projections of the same store
        differ, which defeats the point of a rebuildable graph.
        """
        if self.directionality is Directionality.SYMMETRIC and self.source_node > self.target_node:
            raise ValueError(
                f"edge {self.edge_id}: a symmetric relationship must be stored in canonical "
                "order (source_node < target_node) so the projection is idempotent"
            )
        return self

    @model_validator(mode="after")
    def _consequence_bearing_edges_reference_one(self) -> Self:
        if self.relationship in CONSEQUENCE_BEARING and self.consequence_id is None:
            raise ValueError(
                f"edge {self.edge_id}: relationship={self.relationship.value!r} is a "
                "consequence claim and requires consequence_id. Status and severity are "
                "read from the consequence; the edge never carries them."
            )
        return self

    @model_validator(mode="after")
    def _derived_edges_name_their_relation(self) -> Self:
        if self.derivation is Derivation.DERIVED:
            if self.derived_relation_id is None:
                raise ValueError(
                    f"edge {self.edge_id}: a derived edge must reference the "
                    "derived_relation_id it projects, or its rule and inputs are unauditable"
                )
        elif self.derived_relation_id is not None:
            raise ValueError(
                f"edge {self.edge_id}: derived_relation_id is set on a "
                f"{self.derivation.value!r} edge, which was not produced by the resolver"
            )
        return self

    @model_validator(mode="after")
    def _evidence_matches_derivation(self) -> Self:
        """The type-C evidence rule, applied to the projection (checkpoint 2.4).

        A derived edge carries no evidence of its own: its evidence set is the computed
        transitive union of its inputs, materialized on read. Nominating an arbitrary input
        span here and re-presenting it as direct support would manufacture exactly the
        false provenance this project exists to prevent.
        """
        if self.derivation in _EVIDENCE_REQUIRED and not self.evidence_ids:
            raise ValueError(
                f"edge {self.edge_id}: derivation={self.derivation.value!r} requires at "
                "least one evidence id - it projects a span"
            )
        if self.derivation is Derivation.DERIVED and self.evidence_ids:
            raise ValueError(
                f"edge {self.edge_id}: a derived edge must not carry evidence_ids. Its "
                "evidence is the transitive union of its inputs, materialized on read; a "
                "direct citation here would be fabricated provenance."
            )
        return self

    @model_validator(mode="after")
    def _hypothesized_flag_tracks_derivation(self) -> Self:
        """The flag is not a hint. It must agree with the epistemology exactly."""
        should_be = self.derivation is Derivation.INFERRED
        if self.is_hypothesized is not should_be:
            raise ValueError(
                f"edge {self.edge_id}: is_hypothesized={self.is_hypothesized} contradicts "
                f"derivation={self.derivation.value!r}. Hypothesized edges are exactly the "
                "inferred ones, and the distinction must survive into the payload."
            )
        return self


class ConsequenceGraph(TimestampedModel):
    """A materialized projection of the store at one point in time.

    ``is_projection`` is a literal rather than a comment. Rebuilding is the only way this
    object comes into existence; a writeable graph is a divergence bug (checkpoint 10.1),
    so the version labels that make the rebuild reproducible are mandatory.
    """

    is_projection: Literal[True] = True

    analysis_id: ShortText
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()

    registry_version: VersionLabel
    coupling_index_version: VersionLabel
    projected_at: dt.datetime

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        for label, seen in (
            ("node", [n.node_id for n in self.nodes]),
            ("edge", [e.edge_id for e in self.edges]),
        ):
            if len(seen) != len(set(seen)):
                duplicates = sorted({i for i in seen if seen.count(i) > 1})
                raise ValueError(f"projection repeats {label} ids: {duplicates}")
        return self

    @model_validator(mode="after")
    def _edges_reference_present_nodes(self) -> Self:
        """A dangling endpoint means the projection was assembled, not derived."""
        known = {n.node_id for n in self.nodes}
        dangling = sorted(
            {
                endpoint
                for edge in self.edges
                for endpoint in (edge.source_node, edge.target_node)
                if endpoint not in known
            }
        )
        if dangling:
            raise ValueError(
                f"projection has edges referencing absent nodes: {dangling}. The graph is "
                "rebuilt from the store, so every endpoint must have been projected too."
            )
        return self

    @property
    def hypothesized_edges(self) -> tuple[GraphEdge, ...]:
        return tuple(e for e in self.edges if e.is_hypothesized)

    @property
    def derived_edges(self) -> tuple[GraphEdge, ...]:
        return tuple(e for e in self.edges if e.derivation is Derivation.DERIVED)
