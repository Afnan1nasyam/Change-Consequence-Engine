"""Rebuild the consequence graph from the store. A pure function, and never a source of truth.

Checkpoint 10.1: the graph is a **pure projection**, rebuilt deterministically from the
assumption store, the derived relations and the arbitrated consequences. A writeable graph is
a divergence bug. This module is the rebuild, and four properties make that claim structural
rather than a convention somebody maintains:

**No agent can author a node or an edge, because this layer cannot reach one.** ``app/graph``
is a guarded layer that may not import ``app.agents`` or any of the four agentic packages. It
takes domain records and returns a projection; there is no parameter through which a model's
output could arrive except as a record the deterministic gates already admitted. Checkpoint
10.2 is therefore a property of the import graph.

**Nothing is invented from absence.** A record that cannot legally project - a derived
relation with no arbitrated consequence, an endpoint outside the supplied bindings, a change
with no evidence - is **reported** as skipped with its reason, never approximated into an
edge. :class:`ProjectionResult` carries those refusals, because a projection that silently
dropped a contradiction and one that found none look identical otherwise.

**Type E never becomes an edge.** No relationship in the taxonomy admits
``Derivation.ABSENT``, so :class:`~app.domain.entities.graph.GraphEdge` is structurally
incapable of carrying a gap. This module's only use of an ``EvidenceGap`` is to annotate the
node its ``partition`` names, matched by exact label. Inventing a relationship out of silence
is the structural form of treating absence of evidence as evidence of absence.

**Two rebuilds of one store are byte-identical.** Node and edge ids are content-derived
(:mod:`app.ingestion.identity`), collections are sorted on a total key, and ``projected_at``
is supplied by the caller rather than read from the clock. So the only thing that can change
between two projections is the store.

The seven edges, and where each comes from
------------------------------------------

============================== =========== ========== ====================================
relationship                   derivation  direction  projected from
============================== =========== ========== ====================================
``binds_assumption``           extracted   directed   one admitted ``AssumptionRecord``
``directly_changes``           observed    directed   one ``AssumptionDelta`` on the change
``invalidates_assumption``     derived     directed   a ``DerivedRelation`` + its consequence
``conflicts_with``             derived     symmetric  a ``DerivedRelation`` + its consequence
``unauthorized_override``      derived     directed   a ``DerivedRelation`` + its consequence
``creates_risk_for``           inferred    directed   an arbitrated **Class B** consequence
``historically_analogous_to``  inferred    directed   a Class B consequence's analogue
============================== =========== ========== ====================================

``binds_assumption`` is the one that matters most, and it is the reason ``assumption_key`` is
a node type at all: two documents joined through a key neither mentions is the central
visualization (checkpoint 10.4). Every other edge is a consequence *of* that join being
visible.

Two projection rules are decisions rather than plumbing:

**The three derived edges require an arbitrated consequence, so an unarbitrated relation does
not project.** That is not a choice this module makes - those relationships are
``CONSEQUENCE_BEARING`` and the edge validator refuses one without a ``consequence_id``. The
consequence of the rule is worth stating: the graph shows what the pipeline *concluded*, not
everything the resolver derived, and the difference is reported as skipped.

**A rejected consequence does not project an edge.** The arbitration concluded the claim does
not hold, so drawing it as a live relationship would show a reader something the system
decided was false. It is skipped with a reason, which keeps it reachable without
foregrounding it - the same treatment the judge gives it in the default view.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import DerivedRelation, EvidenceGap
from app.domain.entities.coupling import ResolvedBinding
from app.domain.entities.findings import Consequence
from app.domain.entities.graph import ConsequenceGraph, GraphEdge, GraphNode
from app.domain.value_objects.enums import ConsequenceClass, Derivation, Status
from app.domain.value_objects.taxonomy import (
    DERIVED_RELATIONSHIPS,
    DIRECTIONALITY,
    Directionality,
    NodeType,
    Relationship,
)
from app.ingestion.identity import edge_id_for, node_id_for

#: Bumped when a projection rule changes, because it changes which edges exist without any
#: record changing. Pinned onto the result so two projections are diffable and the diff is
#: attributable to the projector rather than to the corpus.
GRAPH_PROJECTION_VERSION: Final = "graph-projection-1"

#: Statuses whose consequence does not project a live edge. Exactly one: the arbitration
#: concluded the claim does not hold, and drawing it would show a reader a relationship the
#: system decided was false.
NON_PROJECTING_STATUSES: Final = frozenset({Status.REJECTED})

#: Why a record did not project. Reported rather than dropped: a projection that silently
#: lost a contradiction and one that found none look identical otherwise.
SKIP_NO_CONSEQUENCE: Final = "no_arbitrated_consequence"
SKIP_REJECTED: Final = "consequence_rejected"
SKIP_UNKNOWN_ENDPOINT: Final = "endpoint_not_in_supplied_bindings"
SKIP_SELF_LOOP: Final = "endpoints_identical"
SKIP_NO_CHANGE_EVIDENCE: Final = "change_carries_no_evidence"
SKIP_UNPROJECTABLE_RELATIONSHIP: Final = "relationship_is_not_projectable"
SKIP_GAP_NODE_UNKNOWN: Final = "gap_partition_matches_no_node"

#: Truncation for a node label. ``GraphNode.label`` is ``ShortText``.
LABEL_CHARS: Final = 512


@dataclass(frozen=True)
class SkippedRecord:
    """A record that could not legally project, and why.

    ``record_id`` is the domain identifier, so a reader can go and look at the thing the
    graph does not show. That is the difference between a projection that is incomplete and
    one that is wrong.
    """

    record_id: str
    kind: str
    reason: str


@dataclass(frozen=True)
class ProjectionResult:
    """The graph, and everything that did not reach it."""

    graph: ConsequenceGraph
    skipped: tuple[SkippedRecord, ...] = ()
    projector_version: str = GRAPH_PROJECTION_VERSION

    @property
    def edges_by_relationship(self) -> dict[Relationship, tuple[GraphEdge, ...]]:
        """The seven buckets, for a reader who needs them apart."""
        return {
            relationship: tuple(e for e in self.graph.edges if e.relationship is relationship)
            for relationship in Relationship
        }

    @property
    def nodes_by_type(self) -> dict[NodeType, tuple[GraphNode, ...]]:
        return {
            node_type: tuple(n for n in self.graph.nodes if n.entity_type is node_type)
            for node_type in NodeType
        }

    def key_holders(self, assumption_key: str) -> tuple[str, ...]:
        """Documents joined to one assumption key by a ``binds_assumption`` edge.

        The query the ``assumption_key`` node type exists to answer. Two documents appearing
        here that do not cite each other *are* the hidden coupling this system is for.
        """
        key_node = node_id_for(
            entity_type=NodeType.ASSUMPTION_KEY.value, natural_key=assumption_key
        )
        labels = {n.node_id: n.label for n in self.graph.nodes}
        return tuple(
            sorted(
                labels[edge.source_node]
                for edge in self.graph.edges
                if edge.relationship is Relationship.BINDS_ASSUMPTION
                and edge.target_node == key_node
            )
        )


def project(
    *,
    analysis_id: str,
    change: ChangeSpec,
    bindings: Sequence[ResolvedBinding] = (),
    relations: Sequence[DerivedRelation] = (),
    consequences: Sequence[Consequence] = (),
    gaps: Sequence[EvidenceGap] = (),
    registry_version: str,
    coupling_index_version: str,
    projected_at: dt.datetime,
) -> ProjectionResult:
    """Rebuild the graph for one analysis.

    Every argument is a domain record the deterministic gates already admitted, and
    ``projected_at`` is the caller's so two rebuilds of an unchanged store are byte-identical.
    Nothing here reads a clock, a store, or a model.
    """
    builder = _Builder(
        analysis_id=analysis_id,
        change=change,
        registry_version=registry_version,
        coupling_index_version=coupling_index_version,
        projected_at=projected_at,
    )
    builder.add_change()
    builder.add_bindings(bindings)
    builder.add_deltas()
    builder.add_derived(relations, consequences)
    builder.add_class_b(consequences)
    builder.annotate_gaps(gaps)
    return builder.finish()


class _Builder:
    """Accumulates nodes and edges, deduplicating by derived id.

    Deduplication is by **id**, not by a separate seen-set: the id already encodes the
    relationship, the endpoints and the record, so two derivations of the same fact collide
    on it. A separate set would be a second definition of "the same edge" and the two would
    drift.
    """

    def __init__(
        self,
        *,
        analysis_id: str,
        change: ChangeSpec,
        registry_version: str,
        coupling_index_version: str,
        projected_at: dt.datetime,
    ) -> None:
        self._analysis_id = analysis_id
        self._change = change
        self._registry_version = registry_version
        self._coupling_index_version = coupling_index_version
        self._projected_at = projected_at

        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._skipped: list[SkippedRecord] = []
        #: Source version -> the document node it belongs to. Built from the bindings, which
        #: is how derived relations - whose refs are version ids - resolve to documents
        #: without this module touching a store.
        self._version_to_node: dict[str, str] = {}
        self._version_to_label: dict[str, str] = {}
        self._change_node = ""

    # -- nodes -----------------------------------------------------------

    def _node(
        self,
        *,
        entity_type: NodeType,
        natural_key: str,
        label: str,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        node_id = node_id_for(entity_type=entity_type.value, natural_key=natural_key)
        if node_id not in self._nodes:
            self._nodes[node_id] = GraphNode(
                node_id=node_id,
                entity_type=entity_type,
                label=label[:LABEL_CHARS] or natural_key[:LABEL_CHARS],
                metadata=dict(metadata or {}),
            )
        return node_id

    def add_change(self) -> None:
        self._change_node = self._node(
            entity_type=NodeType.CHANGE,
            natural_key=self._change.change_id,
            label=" / ".join(self._change.subject_entities) or self._change.change_id,
            metadata={
                "old_state": self._change.old_state or "",
                "new_state": self._change.new_state or "",
                "scope": self._change.scope.render(),
            },
        )

    # -- binds_assumption: the edge the whole product rests on -----------

    def add_bindings(self, bindings: Iterable[ResolvedBinding]) -> None:
        """One ``binds_assumption`` edge per admitted binding: document -> assumption key.

        This is the projection of the record itself (checkpoint 10.5), not a derivation, so
        the derivation is ``extracted`` and the edge cites the span it came from.

        Sorted by ``assumption_id`` so the node and edge sets are built in a fixed order.
        Order does not affect the output - collections are sorted again at the end - but it
        keeps a debugging session reproducible too.
        """
        for binding in sorted(bindings, key=lambda b: b.assumption_id):
            document = self._node(
                entity_type=NodeType.SOURCE_DOCUMENT,
                natural_key=binding.source_id,
                label=binding.source_title,
                metadata={"authority_class": binding.authority_class.value},
            )
            self._version_to_node[binding.source_version_id] = document
            self._version_to_label[binding.source_version_id] = binding.source_title

            key_node = self._node(
                entity_type=NodeType.ASSUMPTION_KEY,
                natural_key=binding.key,
                label=binding.key,
            )
            self._edge(
                relationship=Relationship.BINDS_ASSUMPTION,
                derivation=Derivation.EXTRACTED,
                source_node=document,
                target_node=key_node,
                record_id=binding.assumption_id,
                kind="assumption_record",
                evidence_ids=(binding.evidence_id,),
            )

    # -- directly_changes ------------------------------------------------

    def add_deltas(self) -> None:
        """``directly_changes``: the change -> each assumption key it moves.

        ``observed`` because a delta is what the change request *states*, not something the
        system inferred - and the taxonomy admits only ``observed`` here. An observed edge
        must cite a span, so a ChangeSpec carrying no evidence projects nothing and is
        reported: manufacturing an evidence id to satisfy the validator would be exactly the
        fabricated provenance the validator exists to prevent.
        """
        if not self._change.assumption_deltas:
            return
        if not self._change.evidence_ids:
            self._skipped.append(
                SkippedRecord(
                    record_id=self._change.change_id,
                    kind="change_spec",
                    reason=SKIP_NO_CHANGE_EVIDENCE,
                )
            )
            return
        for delta in sorted(self._change.assumption_deltas, key=lambda d: d.delta_id):
            key_node = self._node(
                entity_type=NodeType.ASSUMPTION_KEY,
                natural_key=delta.key,
                label=delta.key,
            )
            self._edge(
                relationship=Relationship.DIRECTLY_CHANGES,
                derivation=Derivation.OBSERVED,
                source_node=self._change_node,
                target_node=key_node,
                record_id=delta.delta_id,
                kind="assumption_delta",
                evidence_ids=tuple(self._change.evidence_ids),
            )

    # -- the three derived edges -----------------------------------------

    def add_derived(
        self, relations: Sequence[DerivedRelation], consequences: Sequence[Consequence]
    ) -> None:
        """Project each ``DerivedRelation`` that an arbitrated consequence stands behind.

        The consequence is required rather than optional: all three derived relationships are
        ``CONSEQUENCE_BEARING``, so the edge validator refuses one without a
        ``consequence_id``. A relation the pipeline derived but never arbitrated is therefore
        reported as skipped - the graph shows what was concluded, not everything computed.
        """
        by_relation = _consequences_by_relation(consequences)

        for relation in sorted(relations, key=lambda r: r.relation_id):
            if relation.relationship not in DERIVED_RELATIONSHIPS:
                self._skipped.append(
                    SkippedRecord(
                        record_id=relation.relation_id,
                        kind="derived_relation",
                        reason=SKIP_UNPROJECTABLE_RELATIONSHIP,
                    )
                )
                continue

            consequence = by_relation.get(relation.relation_id)
            if consequence is None:
                self._skipped.append(
                    SkippedRecord(
                        record_id=relation.relation_id,
                        kind="derived_relation",
                        reason=SKIP_NO_CONSEQUENCE,
                    )
                )
                continue
            if consequence.status in NON_PROJECTING_STATUSES:
                self._skipped.append(
                    SkippedRecord(
                        record_id=relation.relation_id,
                        kind="derived_relation",
                        reason=SKIP_REJECTED,
                    )
                )
                continue

            source = self._version_to_node.get(relation.subject_ref)
            target = self._version_to_node.get(relation.object_ref)
            if source is None or target is None:
                # The relation's refs are source *version* ids, resolved through the
                # bindings. A missing one means the caller supplied a relation without the
                # bindings behind it, which would otherwise produce a dangling endpoint.
                self._skipped.append(
                    SkippedRecord(
                        record_id=relation.relation_id,
                        kind="derived_relation",
                        reason=SKIP_UNKNOWN_ENDPOINT,
                    )
                )
                continue
            if source == target:
                # One document disagreeing with itself is an intra-document anomaly, not a
                # relationship between two artifacts (the collision matrix already refuses
                # to derive one; this is the projection-side guard).
                self._skipped.append(
                    SkippedRecord(
                        record_id=relation.relation_id,
                        kind="derived_relation",
                        reason=SKIP_SELF_LOOP,
                    )
                )
                continue

            self._edge(
                relationship=relation.relationship,
                derivation=Derivation.DERIVED,
                source_node=source,
                target_node=target,
                record_id=relation.relation_id,
                kind="derived_relation",
                consequence_id=consequence.consequence_id,
                derived_relation_id=relation.relation_id,
            )

    # -- the two inferred edges ------------------------------------------

    def add_class_b(self, consequences: Sequence[Consequence]) -> None:
        """``creates_risk_for`` and ``historically_analogous_to``, from Class B consequences.

        Both are ``inferred``, so both come out ``is_hypothesized=True`` - which the payload
        keeps structurally distinct so a client cannot render a hypothesis identically to a
        derivation (checkpoint 10.6).

        ``creates_risk_for`` carries the consequence's evidence ids. That is legitimate for an
        inferred edge and forbidden for a derived one: a derivation's evidence is the computed
        union of its inputs, so citing one span directly there would be fabricated provenance,
        while a hypothesis genuinely rests on the spans it cites.
        """
        for consequence in sorted(consequences, key=lambda c: c.consequence_id):
            if consequence.consequence_class is not ConsequenceClass.CLASS_B_BEHAVIORAL:
                continue
            if consequence.status in NON_PROJECTING_STATUSES:
                self._skipped.append(
                    SkippedRecord(
                        record_id=consequence.consequence_id,
                        kind="consequence",
                        reason=SKIP_REJECTED,
                    )
                )
                continue

            entity = self._node(
                entity_type=NodeType.ENTITY,
                natural_key=consequence.affected_entity,
                label=consequence.affected_entity,
            )
            self._edge(
                relationship=Relationship.CREATES_RISK_FOR,
                derivation=Derivation.INFERRED,
                source_node=self._change_node,
                target_node=entity,
                record_id=consequence.consequence_id,
                kind="consequence",
                consequence_id=consequence.consequence_id,
                evidence_ids=tuple(consequence.evidence_ids),
                is_hypothesized=True,
            )

            if consequence.historical_analogue_id is not None:
                event = self._node(
                    entity_type=NodeType.HISTORICAL_EVENT,
                    natural_key=consequence.historical_analogue_id,
                    label=consequence.historical_analogue_id,
                )
                self._edge(
                    relationship=Relationship.HISTORICALLY_ANALOGOUS_TO,
                    derivation=Derivation.INFERRED,
                    source_node=self._change_node,
                    target_node=event,
                    record_id=consequence.consequence_id,
                    kind="consequence",
                    # Carried although this relationship is not consequence-bearing: the
                    # analogy belongs to a specific hypothesis, and dropping the link would
                    # leave a reader unable to ask which one.
                    consequence_id=consequence.consequence_id,
                    is_hypothesized=True,
                )

    # -- type E: annotation only -----------------------------------------

    def annotate_gaps(self, gaps: Iterable[EvidenceGap]) -> None:
        """A gap annotates the node its ``partition`` names. It never becomes an edge.

        Matched by **exact label**, deliberately: interpreting a gap's prose to decide which
        node it concerns would put a judgement in the projector, and the projector is the one
        component that must contain none. A gap whose partition matches nothing is reported,
        because it still attaches to the analysis - it simply annotates no vertex.

        There is no branch in this method that constructs an edge, and none could succeed:
        no relationship in the taxonomy admits ``Derivation.ABSENT``.
        """
        by_label: dict[str, list[str]] = {}
        for gap in sorted(gaps, key=lambda g: g.gap_id):
            if gap.partition is None:
                self._skipped.append(
                    SkippedRecord(
                        record_id=gap.gap_id, kind="evidence_gap", reason=SKIP_GAP_NODE_UNKNOWN
                    )
                )
                continue
            by_label.setdefault(gap.partition, []).append(gap.gap_id)

        matched: set[str] = set()
        for node_id, node in list(self._nodes.items()):
            gap_ids = by_label.get(node.label)
            if not gap_ids:
                continue
            matched.add(node.label)
            self._nodes[node_id] = node.model_copy(
                update={"evidence_gap_ids": tuple(sorted(set(gap_ids)))}
            )

        for label, gap_ids in by_label.items():
            if label not in matched:
                self._skipped.extend(
                    SkippedRecord(
                        record_id=gap_id, kind="evidence_gap", reason=SKIP_GAP_NODE_UNKNOWN
                    )
                    for gap_id in gap_ids
                )

    # -- edges and assembly ----------------------------------------------

    def _edge(
        self,
        *,
        relationship: Relationship,
        derivation: Derivation,
        source_node: str,
        target_node: str,
        record_id: str,
        kind: str,
        consequence_id: str | None = None,
        derived_relation_id: str | None = None,
        evidence_ids: tuple[str, ...] = (),
        is_hypothesized: bool = False,
    ) -> None:
        """Add one edge, canonically ordered and deduplicated by derived id.

        Symmetric relationships are ordered here rather than by the caller. ``conflicts_with``
        between A and B is one fact; emitting both directions would double-count every
        contradiction and make two projections of one store differ. The entity validator
        refuses an out-of-order symmetric edge, so this is the place that guarantees the
        validator never fires.
        """
        directionality = DIRECTIONALITY[relationship]
        if directionality is Directionality.SYMMETRIC and source_node > target_node:
            source_node, target_node = target_node, source_node

        edge_id = edge_id_for(
            relationship=relationship.value,
            source_node=source_node,
            target_node=target_node,
            record_id=record_id,
        )
        if edge_id in self._edges:
            # The same fact reached twice - a symmetric relation visited from both sides, or
            # one relation projected once per affected artifact. Dropped rather than counted.
            return
        self._edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            relationship=relationship,
            derivation=derivation,
            directionality=directionality,
            consequence_id=consequence_id,
            derived_relation_id=derived_relation_id,
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            is_hypothesized=is_hypothesized,
        )
        _ = kind  # kind is carried only on skips; edges name the record they project

    def finish(self) -> ProjectionResult:
        """Assemble the projection with every collection on a total order.

        Sorting is what makes requirement 12 hold: with content-derived ids and a total key,
        two rebuilds of an unchanged store serialize to the same bytes. ``created_at`` is
        pinned to ``projected_at`` for the same reason - a default clock read would make every
        rebuild differ in a field nobody set.
        """
        graph = ConsequenceGraph(
            analysis_id=self._analysis_id,
            nodes=tuple(
                sorted(self._nodes.values(), key=lambda n: (n.entity_type.value, n.node_id))
            ),
            edges=tuple(
                sorted(
                    self._edges.values(),
                    key=lambda e: (
                        e.relationship.value,
                        e.source_node,
                        e.target_node,
                        e.edge_id,
                    ),
                )
            ),
            registry_version=self._registry_version,
            coupling_index_version=self._coupling_index_version,
            projected_at=self._projected_at,
            created_at=self._projected_at,
        )
        return ProjectionResult(
            graph=graph,
            skipped=tuple(sorted(self._skipped, key=lambda s: (s.kind, s.reason, s.record_id))),
        )


def _consequences_by_relation(
    consequences: Sequence[Consequence],
) -> dict[str, Consequence]:
    """Index arbitrated consequences by the derived relation each reports.

    A relation may be reported by more than one consequence only if the pipeline duplicated
    it, which the identity recipes already prevent; the lowest ``consequence_id`` wins so the
    projection stays deterministic if it ever happens.
    """
    indexed: dict[str, Consequence] = {}
    for consequence in sorted(consequences, key=lambda c: c.consequence_id):
        for relation_id in consequence.derived_relation_ids:
            indexed.setdefault(relation_id, consequence)
    return indexed
