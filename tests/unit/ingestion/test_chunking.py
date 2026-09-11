"""Chunking: deterministic, gapless, non-overlapping, offset-faithful.

The properties asserted here are what later phases silently depend on:

* offsets are document-absolute, so span verification is one slice comparison;
* chunks do not overlap, so one span cannot be counted as two sources;
* chunks do not leave gaps, so no part of a document is unreachable as evidence;
* the same document chunks identically every time, so chunk and evidence ids are stable.
"""

from __future__ import annotations

from itertools import pairwise

from app.ingestion.chunking import CHUNKER_VERSION, ChunkingConfig, chunk_document
from app.ingestion.parsers import MarkdownParser, ParsedDocument, PlainTextParser

VERSION_ID = "sv_" + "a" * 32

DOC = b"""# Retention

Personal data is retained for 90 days.

## EU

For the EU the period is 30 days.

## Dormancy

An account idle for 400 days is dormant.
"""


def _parse(payload: bytes = DOC) -> ParsedDocument:
    return MarkdownParser().parse(payload)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_offsets_are_document_absolute_and_recover_the_chunk_text() -> None:
    parsed = _parse()
    for item in chunk_document(parsed, version_id=VERSION_ID):
        chunk = item.chunk
        assert parsed.text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunks_do_not_overlap() -> None:
    """An overlapping span would let one sentence look like two corroborating sources."""
    chunks = [c.chunk for c in chunk_document(_parse(), version_id=VERSION_ID)]
    for earlier, later in pairwise(chunks):
        assert earlier.char_end <= later.char_start


def test_chunks_leave_no_gap_in_the_document() -> None:
    """A gap would make part of a document permanently uncitable."""
    parsed = _parse()
    covered = sorted(
        (c.chunk.char_start, c.chunk.char_end)
        for c in chunk_document(parsed, version_id=VERSION_ID)
    )
    cursor = 0
    for start, end in covered:
        assert parsed.text[cursor:start].strip() == "", "dropped non-whitespace text"
        cursor = max(cursor, end)
    assert parsed.text[cursor:].strip() == ""


def test_ordinals_are_dense_and_ascending() -> None:
    chunks = chunk_document(_parse(), version_id=VERSION_ID)
    assert [c.chunk.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunking_is_deterministic() -> None:
    first = chunk_document(_parse(), version_id=VERSION_ID)
    second = chunk_document(_parse(), version_id=VERSION_ID)
    assert [c.chunk.chunk_id for c in first] == [c.chunk.chunk_id for c in second]
    assert first == second


def test_chunk_ids_depend_on_the_version() -> None:
    a = chunk_document(_parse(), version_id=VERSION_ID)
    b = chunk_document(_parse(), version_id="sv_" + "b" * 32)
    assert {c.chunk.chunk_id for c in a}.isdisjoint({c.chunk.chunk_id for c in b})


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_each_chunk_carries_its_section_trail() -> None:
    """Without it, a citation cannot be rendered where a reviewer would look for it."""
    chunks = chunk_document(_parse(), version_id=VERSION_ID)
    eu = next(c for c in chunks if "30 days" in c.chunk.text)
    assert eu.section_path[-1] == "EU"
    assert eu.section_label.endswith("EU")


def test_each_chunk_carries_a_content_hash() -> None:
    for item in chunk_document(_parse(), version_id=VERSION_ID):
        assert len(item.text_sha256) == 64


def test_a_chunk_with_no_section_reports_the_document_label() -> None:
    parsed = PlainTextParser().parse(b"Retention is 90 days.")
    chunks = chunk_document(parsed, version_id=VERSION_ID)
    assert chunks[0].section_label == "(document)"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_a_long_document_is_split_under_the_ceiling() -> None:
    body = ("Refunds above 10,000 USD require approval. " * 200).encode()
    chunks = chunk_document(
        PlainTextParser().parse(body),
        version_id=VERSION_ID,
        config=ChunkingConfig(target_chars=300, max_chars=400, min_chars=50),
    )
    assert len(chunks) > 1
    assert all(len(c.chunk.text) <= 400 for c in chunks)


def test_splitting_prefers_a_paragraph_boundary() -> None:
    """A boundary through a sentence loses either the value or its qualifier."""
    body = ("A" * 300 + "\n\n" + "B" * 300 + "\n\n" + "C" * 300).encode()
    chunks = chunk_document(
        PlainTextParser().parse(body),
        version_id=VERSION_ID,
        config=ChunkingConfig(target_chars=280, max_chars=400, min_chars=10),
    )
    assert any(c.chunk.text.strip() == "A" * 300 for c in chunks)


def test_splitting_falls_back_to_a_sentence_boundary() -> None:
    body = ("The threshold is 10,000 USD. " * 40).encode()
    chunks = chunk_document(
        PlainTextParser().parse(body),
        version_id=VERSION_ID,
        config=ChunkingConfig(target_chars=200, max_chars=260, min_chars=20),
    )
    assert len(chunks) > 1
    for item in chunks[:-1]:
        assert item.chunk.text.rstrip().endswith(".")


def test_a_trailing_fragment_is_merged_rather_than_left_as_a_runt() -> None:
    """A chunk of a few characters produces evidence spans with no content."""
    body = ("X" * 405).encode()
    chunks = chunk_document(
        PlainTextParser().parse(body),
        version_id=VERSION_ID,
        config=ChunkingConfig(target_chars=200, max_chars=200, min_chars=50),
    )
    assert all(len(c.chunk.text) >= 50 for c in chunks)


def test_the_config_fingerprint_is_provenance() -> None:
    """Two chunkings of one version are both valid but produce different ids."""
    default = ChunkingConfig()
    other = ChunkingConfig(target_chars=500)
    assert default.fingerprint() != other.fingerprint()
    assert CHUNKER_VERSION in default.fingerprint()


def test_a_document_with_no_sections_still_chunks() -> None:
    parsed = PlainTextParser().parse(b"one line only")
    assert len(chunk_document(parsed, version_id=VERSION_ID)) == 1
