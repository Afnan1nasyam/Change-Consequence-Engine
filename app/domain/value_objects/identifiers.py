"""Prefixed, shape-validated identifiers.

Every id carries a type prefix and a 32-hex body. This is not cosmetic: it makes a
**fabricated citation fail validation at the schema boundary**. An agent that invents
``"EV-12"`` or ``"evidence_3"`` for an ``evidence_id`` is rejected by Pydantic before any
service sees it, which converts the single most damaging failure mode in the system from a
detection problem into a parse error.

Existence of the referenced row is a separate, deferred service check (span verification).
Shape validation cannot prove an id is real - only that it was not casually made up.
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from pydantic import StringConstraints

# ---------------------------------------------------------------------------
# Prefixes
# ---------------------------------------------------------------------------

PREFIXES: dict[str, str] = {
    "tenant": "tn",
    "source": "src",
    "source_version": "sv",
    "chunk": "ch",
    "evidence": "ev",
    "assumption": "as",
    "registry_gap": "rg",
    "alias": "al",
    "change": "chg",
    "delta": "dl",
    "relation": "dr",
    "fact": "of",
    "hypothesis": "hy",
    "gap": "eg",
    "finding": "sf",
    "challenge": "cl",
    "test": "ft",
    "consequence": "cq",
    "disposition": "dp",
    "analysis": "an",
    "node": "nd",
    "edge": "ed",
    "event": "he",
    "class_a_finding": "ca",
}

_BODY = "[0-9a-f]{32}"


def _pattern(prefix: str) -> str:
    return rf"^{prefix}_{_BODY}$"


def new_id(kind: str) -> str:
    """Mint an id for ``kind``. Raises for an unknown kind rather than guessing a prefix."""
    try:
        prefix = PREFIXES[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown identifier kind {kind!r}; known kinds are {sorted(PREFIXES)}"
        ) from exc
    return f"{prefix}_{uuid.uuid4().hex}"


def id_kind(value: str) -> str | None:
    """Reverse lookup: which kind does this id belong to? ``None`` if it matches none."""
    for kind, prefix in PREFIXES.items():
        if re.match(_pattern(prefix), value):
            return kind
    return None


def _typed(kind: str) -> object:
    return StringConstraints(pattern=_pattern(PREFIXES[kind]))


# ---------------------------------------------------------------------------
# Annotated aliases
# ---------------------------------------------------------------------------

TenantId = Annotated[str, _typed("tenant")]
SourceId = Annotated[str, _typed("source")]
SourceVersionId = Annotated[str, _typed("source_version")]
ChunkId = Annotated[str, _typed("chunk")]
EvidenceId = Annotated[str, _typed("evidence")]
AssumptionId = Annotated[str, _typed("assumption")]
RegistryGapId = Annotated[str, _typed("registry_gap")]
AliasId = Annotated[str, _typed("alias")]
ChangeId = Annotated[str, _typed("change")]
DeltaId = Annotated[str, _typed("delta")]
RelationId = Annotated[str, _typed("relation")]
FactId = Annotated[str, _typed("fact")]
HypothesisId = Annotated[str, _typed("hypothesis")]
GapId = Annotated[str, _typed("gap")]
FindingId = Annotated[str, _typed("finding")]
ChallengeId = Annotated[str, _typed("challenge")]
TestId = Annotated[str, _typed("test")]
ConsequenceId = Annotated[str, _typed("consequence")]
DispositionId = Annotated[str, _typed("disposition")]
AnalysisId = Annotated[str, _typed("analysis")]
NodeId = Annotated[str, _typed("node")]
EdgeId = Annotated[str, _typed("edge")]
EventId = Annotated[str, _typed("event")]
ClassAFindingId = Annotated[str, _typed("class_a_finding")]

# ---------------------------------------------------------------------------
# Assumption keys
# ---------------------------------------------------------------------------

#: Canonical shape only. *Existence* in the closed registry is a deferred service check;
#: a well-shaped but unregistered key is quarantined as a KeyRegistryGap, never indexed.
ASSUMPTION_KEY_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*(?:\.[a-z][a-z0-9]*(?:_[a-z0-9]+)*)+$"

AssumptionKey = Annotated[str, StringConstraints(pattern=ASSUMPTION_KEY_PATTERN)]

#: Free-form version labels for the registry, confidence function, coupling index, etc.
#: A provenance label: a registry version, a rule version, a scorer fingerprint.
#:
#: Widened from 64 to 128 when the first real-embedding recording was attempted. A composed
#: fingerprint has more components than it did when 64 was chosen - a replayed remote
#: embedder reports
#: ``vector-cosine-1:embed-replay-1:openai-compatible/<model>@<date>:d=<width>``, which
#: exceeds 64 for most real model names. The alternative was truncating provenance, and a
#: label that drops the model identifier to fit a cap is worse than a longer label.
VersionLabel = Annotated[str, StringConstraints(min_length=1, max_length=128)]
