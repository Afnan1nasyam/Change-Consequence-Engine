"""Relationship taxonomy: seven edges with fixed semantics.

Deliberately small. A large taxonomy applied by many agents is applied inconsistently, and
a permissive catch-all edge absorbs most of the graph and stops carrying information -
which is why ``potentially_impacts`` is excluded.

The taxonomy grew from six to seven by the documented extension path: written semantics
(checkpoint 5.4.3) plus an admissible derivation, decided *before* any agent could emit a
finding that projects to the new edge.
"""

from __future__ import annotations

from enum import StrEnum

from app.domain.value_objects.enums import Derivation


class Relationship(StrEnum):
    DIRECTLY_CHANGES = "directly_changes"
    BINDS_ASSUMPTION = "binds_assumption"
    INVALIDATES_ASSUMPTION = "invalidates_assumption"
    CONFLICTS_WITH = "conflicts_with"
    UNAUTHORIZED_OVERRIDE = "unauthorized_override"
    CREATES_RISK_FOR = "creates_risk_for"
    HISTORICALLY_ANALOGOUS_TO = "historically_analogous_to"


class Directionality(StrEnum):
    DIRECTED = "directed"
    SYMMETRIC = "symmetric"


#: Which derivations each edge may carry. Enforced by a model validator on GraphEdge, so a
#: model-inferred edge cannot masquerade as a deterministic one.
ADMISSIBLE_DERIVATIONS: dict[Relationship, frozenset[Derivation]] = {
    Relationship.DIRECTLY_CHANGES: frozenset({Derivation.OBSERVED}),
    Relationship.BINDS_ASSUMPTION: frozenset({Derivation.EXTRACTED}),
    Relationship.INVALIDATES_ASSUMPTION: frozenset({Derivation.DERIVED}),
    Relationship.CONFLICTS_WITH: frozenset({Derivation.DERIVED}),
    Relationship.UNAUTHORIZED_OVERRIDE: frozenset({Derivation.DERIVED}),
    Relationship.CREATES_RISK_FOR: frozenset({Derivation.INFERRED}),
    # The incident is observed; the analogy is a judgment. An earlier draft admitted
    # `observed` here, which would have let a judgment claim the status of an observation.
    Relationship.HISTORICALLY_ANALOGOUS_TO: frozenset({Derivation.INFERRED}),
}

DIRECTIONALITY: dict[Relationship, Directionality] = {
    Relationship.DIRECTLY_CHANGES: Directionality.DIRECTED,
    Relationship.BINDS_ASSUMPTION: Directionality.DIRECTED,
    Relationship.INVALIDATES_ASSUMPTION: Directionality.DIRECTED,
    # Two peers disagreeing at the same scope: neither is the subject.
    Relationship.CONFLICTS_WITH: Directionality.SYMMETRIC,
    # Directed narrower -> broader: the narrower artifact is the one deviating.
    Relationship.UNAUTHORIZED_OVERRIDE: Directionality.DIRECTED,
    Relationship.CREATES_RISK_FOR: Directionality.DIRECTED,
    Relationship.HISTORICALLY_ANALOGOUS_TO: Directionality.DIRECTED,
}

#: Edges a DerivedRelation (ontology type C) may express.
DERIVED_RELATIONSHIPS: frozenset[Relationship] = frozenset(
    {
        Relationship.INVALIDATES_ASSUMPTION,
        Relationship.CONFLICTS_WITH,
        Relationship.UNAUTHORIZED_OVERRIDE,
    }
)

#: Edges that must reference a consequence, because they *are* a consequence claim.
CONSEQUENCE_BEARING: frozenset[Relationship] = frozenset(
    {
        Relationship.INVALIDATES_ASSUMPTION,
        Relationship.CONFLICTS_WITH,
        Relationship.UNAUTHORIZED_OVERRIDE,
        Relationship.CREATES_RISK_FOR,
    }
)

#: Edges produced only by the projector from an underlying record, never authored.
PROJECTION_ONLY: frozenset[Relationship] = frozenset({Relationship.BINDS_ASSUMPTION})


class NodeType(StrEnum):
    SOURCE_DOCUMENT = "source_document"
    #: A node type *because* that is what makes hidden coupling visible: two documents
    #: joined through a key neither mentions is the central visualization.
    ASSUMPTION_KEY = "assumption_key"
    ENTITY = "entity"
    CHANGE = "change"
    HISTORICAL_EVENT = "historical_event"
