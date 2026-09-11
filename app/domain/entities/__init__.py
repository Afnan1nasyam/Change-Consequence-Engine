"""Domain entities - the frozen Phase 1 contracts.

Import from this package rather than from the individual modules. The module split is an
implementation detail; the contract surface is this ``__all__``, and it is what
``docs/DOMAIN_CHECKPOINT.md`` freezes.
"""

from __future__ import annotations

from app.domain.entities.assumption import (
    AssumptionDelta,
    AssumptionRecord,
    CouplingHolder,
    CouplingSet,
)
from app.domain.entities.base import (
    DOMAIN_SCHEMA_VERSION,
    DomainModel,
    TimestampedModel,
    utc_now,
)
from app.domain.entities.change import ChangeRequest, ChangeSpec
from app.domain.entities.claims import (
    Claim,
    ClaimEnvelope,
    ConsequenceHypothesis,
    DerivedRelation,
    EvidenceGap,
    ObservedFact,
)
from app.domain.entities.class_a import ClassAFinding, ClassBMergeOutcome
from app.domain.entities.coupling import (
    AuthorityCandidate,
    AuthorityResolution,
    CouplingAnomaly,
    CouplingStrength,
    OverrideRecord,
    ResolvedBinding,
)
from app.domain.entities.decay import (
    DecayEntry,
    DecayParticipant,
    KnowledgeDecayReport,
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

__all__ = [
    "DOMAIN_SCHEMA_VERSION",
    "AnalysisReport",
    "ArbitrationDecision",
    "AssumptionDelta",
    "AssumptionKeyRegistryEntry",
    "AssumptionRecord",
    "AuthorityCandidate",
    "AuthorityResolution",
    "ChallengeResult",
    "ChangeRequest",
    "ChangeSpec",
    "Chunk",
    "Claim",
    "ClaimEnvelope",
    "ClassAFinding",
    "ClassBMergeOutcome",
    "Consequence",
    "ConsequenceGraph",
    "ConsequenceHypothesis",
    "ConsequenceScore",
    "CouplingAnomaly",
    "CouplingHolder",
    "CouplingSet",
    "CouplingStrength",
    "DecayEntry",
    "DecayParticipant",
    "DerivedRelation",
    "DomainModel",
    "EvidenceGap",
    "EvidenceItem",
    "FalsificationTest",
    "FindingDisposition",
    "GraphEdge",
    "GraphNode",
    "HistoricalEvent",
    "KeyAlias",
    "KeyRegistryGap",
    "KnowledgeDecayReport",
    "ObservedFact",
    "OverrideRecord",
    "ResolvedBinding",
    "SourceDocument",
    "SourceVersion",
    "SpecialistFinding",
    "TimestampedModel",
    "utc_now",
]
