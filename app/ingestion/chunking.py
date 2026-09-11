"""Deterministic, section-aware chunking with document-absolute offsets.

Three decisions worth stating, because each rules out a plausible alternative:

**Offsets are document-absolute.** A chunk's ``char_start``/``char_end`` index into the
version's canonical text, not into the chunk. So does every evidence span. Span
verification is then a single slice comparison against one string, and a chunker change
cannot silently invalidate a citation recorded by an earlier analysis.

**No overlap between chunks.** Overlapping windows are a retrieval convenience, but they
duplicate spans, and a duplicated span is exactly what corroboration counting must not
mistake for a second source (``DOMAIN_CHECKPOINT.md`` 9.3). Retrieval can widen a window
at query time; the store must not double-count.

**Splits prefer paragraph, then sentence, then hard cut.** A binding is usually one
sentence ("refunds above 10,000 USD require manager approval"), and a chunk boundary
through that sentence loses the value or the qualifier. The budget is therefore a *target*
that yields to structure until the hard ceiling.
"""

from __future__ import annotations

import re
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field

from app.domain.entities.source import Chunk
from app.ingestion.identity import chunk_id_for, content_hash
from app.ingestion.parsers.base import ParsedDocument

#: Bumped when boundary selection changes. Persisted alongside the version, because a
#: different chunking is a different (though equally valid) set of chunk ids.
CHUNKER_VERSION = "chunker-1"

_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


class ChunkingConfig(BaseModel):
    """Chunking is configuration, but its values are provenance.

    ``target_chars`` is a soft budget; ``max_chars`` is the ceiling a hard cut enforces.
    ``min_chars`` stops a heading line becoming a chunk of its own, which would produce
    evidence spans with no content.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_chars: int = Field(default=1200, gt=0)
    max_chars: int = Field(default=2000, gt=0)
    min_chars: int = Field(default=80, ge=0)

    def fingerprint(self) -> str:
        return f"{CHUNKER_VERSION}:{self.target_chars}/{self.max_chars}/{self.min_chars}"


class ChunkWithContext(BaseModel):
    """A chunk plus the section trail it came from.

    The section label is not stored on :class:`~app.domain.entities.source.Chunk` (a frozen
    Phase 1 contract), so it travels alongside. It matters for two things: telling the
    extractor where it is looking, and rendering a citation a human can find.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    section_path: tuple[str, ...] = ()
    text_sha256: str

    @property
    def section_label(self) -> str:
        return " > ".join(self.section_path) if self.section_path else "(document)"


def chunk_document(
    parsed: ParsedDocument,
    *,
    version_id: str,
    config: ChunkingConfig | None = None,
) -> tuple[ChunkWithContext, ...]:
    """Split a parsed document into chunks. Same inputs, same chunk ids, always."""
    cfg = config or ChunkingConfig()
    text = parsed.text
    chunks: list[ChunkWithContext] = []
    ordinal = 0

    for span_start, span_end, section_path in _section_spans(parsed):
        for raw_start, raw_end in _split_span(text, span_start, span_end, cfg):
            # Trim to the content itself. `Chunk.text` is whitespace-stripped by the frozen
            # domain config, so a span that included surrounding blank lines would report
            # offsets that no longer recover the stored text - and an evidence span framed
            # by blank lines is not what a reviewer would cite anyway.
            start, end = _trim(text, raw_start, raw_end)
            if end <= start:
                continue
            body = text[start:end]
            chunks.append(
                ChunkWithContext(
                    chunk=Chunk(
                        chunk_id=chunk_id_for(
                            version_id=version_id,
                            ordinal=ordinal,
                            char_start=start,
                            char_end=end,
                        ),
                        source_version_id=version_id,
                        ordinal=ordinal,
                        text=body,
                        char_start=start,
                        char_end=end,
                    ),
                    section_path=section_path,
                    text_sha256=content_hash(body),
                )
            )
            ordinal += 1

    return tuple(chunks)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Narrow a span to its non-whitespace content."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _section_spans(parsed: ParsedDocument) -> list[tuple[int, int, tuple[str, ...]]]:
    """Leaf regions of the section map, in document order, with no gaps and no overlaps.

    A nested heading's parent would otherwise cover its children and every span would be
    chunked twice. Only the innermost section owning each offset contributes.
    """
    text_length = len(parsed.text)
    if not parsed.sections:
        return [(0, text_length, ())]

    boundaries = sorted({0, text_length} | {s.char_start for s in parsed.sections})
    spans: list[tuple[int, int, tuple[str, ...]]] = []
    for start, end in pairwise(boundaries):
        if end <= start:
            continue
        owner = parsed.section_for(start)
        spans.append((start, end, owner.path if owner else ()))
    return spans


def _split_span(text: str, start: int, end: int, cfg: ChunkingConfig) -> list[tuple[int, int]]:
    """Break ``[start, end)`` into pieces under the budget, preferring natural boundaries."""
    if end - start <= cfg.max_chars:
        return [(start, end)]

    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        remaining = end - cursor
        if remaining <= cfg.max_chars:
            pieces.append((cursor, end))
            break

        window_end = cursor + cfg.max_chars
        cut = _best_cut(text, cursor, window_end, cfg)
        pieces.append((cursor, cut))
        cursor = cut

    return _merge_runts(pieces, cfg)


def _best_cut(text: str, start: int, ceiling: int, cfg: ChunkingConfig) -> int:
    """The latest natural boundary at or before ``ceiling``, else a hard cut."""
    target = min(start + cfg.target_chars, ceiling)
    window = text[start:ceiling]

    paragraph_cuts = [m.end() for m in _PARAGRAPH.finditer(window)]
    preferred = [start + c for c in paragraph_cuts if start + c >= target]
    if paragraph_cuts and not preferred:
        preferred = [start + paragraph_cuts[-1]]
    if preferred:
        return preferred[0]

    sentence_cuts = [m.end() for m in _SENTENCE_END.finditer(window)]
    sentence_preferred = [start + c for c in sentence_cuts if start + c >= target]
    if sentence_cuts and not sentence_preferred:
        sentence_preferred = [start + sentence_cuts[-1]]
    if sentence_preferred:
        return sentence_preferred[0]

    return ceiling


def _merge_runts(pieces: list[tuple[int, int]], cfg: ChunkingConfig) -> list[tuple[int, int]]:
    """Fold a trailing fragment into its predecessor when it is below ``min_chars``."""
    if len(pieces) < 2:
        return pieces
    merged = list(pieces)
    last_start, last_end = merged[-1]
    if last_end - last_start < cfg.min_chars:
        prev_start, _ = merged[-2]
        merged[-2:] = [(prev_start, last_end)]
    return merged
