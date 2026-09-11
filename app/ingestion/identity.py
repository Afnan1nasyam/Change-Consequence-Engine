"""Deterministic identity and content hashing for ingestion.

Every identifier produced here is a **pure function of its inputs**. That is what makes
two of the Phase 2 acceptance criteria true rather than aspirational:

* *evidence IDs remain stable* - an evidence id is derived from the version, the chunk and
  the span offsets, so re-ingesting unchanged bytes yields the same id, and a citation
  recorded in an old analysis still resolves.
* *re-ingesting an unchanged source performs zero extraction calls* - the version id is
  derived from the content hash, so an unchanged document maps onto the version that is
  already in the store and the use case can skip extraction on identity alone rather than
  by comparing text.

Ids keep the Phase 1 shape (``prefix_`` + 32 lowercase hex) so nothing downstream can tell
a derived id from a minted one; :func:`app.domain.value_objects.identifiers.new_id` remains
correct for anything genuinely non-deterministic.

Text is canonicalized before hashing (:func:`canonical_text`). Every character offset in
the system indexes into that canonical text, so a CRLF checkout and an LF checkout of the
same document produce identical spans.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence

from app.domain.value_objects.identifiers import PREFIXES

#: Bumped when canonicalization, offset semantics, or an id recipe changes. Persisted on
#: every SourceVersion, because a change here changes every downstream offset.
PARSER_VERSION = "parser-1"

_HEX_BODY = 32


def canonical_text(raw: str) -> str:
    """The single textual representation offsets are defined against.

    Three normalizations, each chosen because *not* doing it would move offsets between
    environments rather than merely look untidy:

    * NFC composition - so an accented character is one code point on every platform;
    * newline folding - CRLF and CR become LF;
    * BOM removal.

    Deliberately **not** done: whitespace collapsing, case folding, punctuation
    normalization. Those would break the span-verification invariant, which requires the
    stored text to contain the cited value *verbatim*.
    """
    text = raw.lstrip("﻿")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def content_hash(text: str) -> str:
    """SHA-256 of the canonical text. The identity of a source *version*."""
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def bytes_hash(payload: bytes) -> str:
    """SHA-256 of raw bytes, for artifacts stored verbatim in object storage."""
    return hashlib.sha256(payload).hexdigest()


def _derive(kind: str, *parts: object) -> str:
    """A stable id of the right shape for ``kind``, derived from ``parts``.

    Parts are joined with a delimiter that cannot occur in a hash or an integer, so
    ``("a", "bc")`` and ``("ab", "c")`` cannot collide.
    """
    try:
        prefix = PREFIXES[kind]
    except KeyError as exc:  # pragma: no cover - programming error
        raise ValueError(f"unknown identifier kind {kind!r}") from exc
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(f"{kind}\x1e{joined}".encode()).hexdigest()
    return f"{prefix}_{digest[:_HEX_BODY]}"


def source_id_for(*, tenant_id: str, uri: str) -> str:
    """A document's identity is its URI within its tenant.

    Tenant is part of the recipe, not a filter applied later: two tenants ingesting the
    same URI must not collide onto one document (``docs/SECURITY.md`` 8).
    """
    return _derive("source", tenant_id, uri)


def version_id_for(*, source_id: str, content_hash_hex: str) -> str:
    """A version's identity is (document, exact content). Re-ingestion is idempotent."""
    return _derive("source_version", source_id, content_hash_hex)


def chunk_id_for(*, version_id: str, ordinal: int, char_start: int, char_end: int) -> str:
    return _derive("chunk", version_id, ordinal, char_start, char_end)


def evidence_id_for(*, version_id: str, char_start: int, char_end: int) -> str:
    """Evidence identity is the span itself, not the chunk that happened to contain it.

    Two chunkings of the same version that both cover a span therefore agree on its
    evidence id, which is what keeps a citation from an earlier analysis resolvable after a
    chunker change.
    """
    return _derive("evidence", version_id, char_start, char_end)


def event_id_for(*, tenant_id: str, evidence_ids: Sequence[str]) -> str:
    """A historical event's identity is the spans that record it, within its tenant.

    **Not the title.** A model wording "the June 2024 threshold change" two ways must not
    produce two events, and a model wording two different incidents the same way must not
    collapse them - so identity is the evidence, which is the only part of an analogue the
    model does not author.

    Sorted, because a citation order is not part of what happened. Tenant is in the recipe for
    the reason it is in every other recipe here: two tenants whose corpora contain the same
    span must not collide onto one event (``docs/SECURITY.md`` 8).

    The consequence worth naming: this id is **reproducible without a store**. Two analyses
    citing the same spans agree on it, so ``historical_analogue_id`` resolves by re-derivation
    rather than by lookup - which is why closing this gap needed no new table.
    """
    return _derive("event", tenant_id, *sorted(set(evidence_ids)))


def assumption_id_for(
    *,
    version_id: str,
    key: str,
    scope: str,
    role: str,
    evidence_id: str,
    extractor_version: str,
) -> str:
    """A binding's identity is what makes it that binding.

    ``extractor_version`` is included so that re-extracting under a new prompt or a new
    parser produces a *new* record rather than silently overwriting the one a past analysis
    cited.
    """
    return _derive("assumption", version_id, key, scope, role, evidence_id, extractor_version)


def registry_gap_id_for(*, version_id: str, proposed_key: str, evidence_id: str) -> str:
    """Repeated proposals of the same unknown key from the same span are one gap.

    Gap *volume* is a product signal, so the same span proposing the same key twice must
    not inflate it (``DOMAIN_CHECKPOINT.md`` 4.1.7).
    """
    return _derive("registry_gap", version_id, proposed_key, evidence_id)


def alias_id_for(*, source_key: str, target_key: str) -> str:
    return _derive("alias", source_key, target_key)


def relation_id_for(*, rule_id: str, key: str, scope: str, input_ids: Sequence[str]) -> str:
    """A derived relation's identity is its rule plus the records it consumed.

    ``input_ids`` is sorted, so the same finding derived from the same two records gets one
    id regardless of which side the resolver happened to visit first. Without that, a
    symmetric ``conflicts_with`` would be emitted twice - once per direction - and every
    contradiction in the corpus would be double-counted.

    ``rule_id`` is part of the recipe because two different rules over the same inputs are
    two different claims, and both must be independently citable.
    """
    return _derive("relation", rule_id, key, scope, "\x1d".join(sorted(input_ids)))


def node_id_for(*, entity_type: str, natural_key: str) -> str:
    """A graph node's identity is its type plus what it stands for.

    Derived rather than minted because the graph is a **pure projection**: rebuilding it
    from an unchanged store must produce the same node ids, or two projections of one corpus
    could not be compared and every stored edge would dangle after a rebuild.

    ``entity_type`` is part of the recipe so an assumption key and an entity that happen to
    share a label cannot collapse onto one node - which would silently merge two different
    kinds of thing in the one visualization the product exists for.
    """
    return _derive("node", entity_type, natural_key)


def edge_id_for(*, relationship: str, source_node: str, target_node: str, record_id: str) -> str:
    """An edge's identity is the relationship, its endpoints, and the record it projects.

    ``record_id`` is part of the recipe because two different derived relations may relate
    the same pair of documents - a contradiction on one key and a staleness on another - and
    both must be independently citable.

    It is also what makes the projection idempotent: one ``conflicts_with`` relation yields a
    Class A *finding per affected artifact*, so projecting per finding visits the same
    relation twice. Both visits derive the same edge id and the second is dropped as a
    duplicate rather than double-counting the contradiction.
    """
    return _derive("edge", relationship, source_node, target_node, record_id)


def class_a_finding_id_for(*, relation_id: str, affected_version_id: str) -> str:
    """A Class A finding is one relation as it lands on one affected artifact.

    A single ``conflicts_with`` relation concerns two artifacts and therefore yields a
    finding for each: both authors have something to fix, and a reviewer filtering by
    document must see it from either side.
    """
    return _derive("class_a_finding", relation_id, affected_version_id)
