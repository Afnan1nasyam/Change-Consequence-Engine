"""Parsers: deterministic, offset-faithful, syntax-preserving.

The invariant under test throughout: **every offset a parser reports indexes into the text
it returned, and that text contains the document verbatim.** A parser that tidied its input
would make span verification unable to distinguish a real citation from an invented one.
"""

from __future__ import annotations

import pytest

from app.domain.value_objects.enums import SourceType
from app.ingestion.parsers import (
    JsonParser,
    MarkdownParser,
    ParseError,
    PlainTextParser,
    UnsupportedFormatError,
    parse_payload,
    select_parser,
)

MARKDOWN = b"""Preamble line.

# Refund Policy

Intro text.

## Approval limits

Refunds of 10,000 USD or more require manager approval.

### Delegation

A deputy may be named for 14 days.

## Exceptions

Billing defects are exempt.
"""


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [(".md", "markdown"), (".markdown", "markdown"), (".json", "json"), (".txt", "plaintext")],
)
def test_parser_selection(suffix: str, expected: str) -> None:
    assert select_parser(source_type=SourceType.POLICY, suffix=suffix).name == expected


def test_selection_is_case_insensitive() -> None:
    assert select_parser(source_type=SourceType.POLICY, suffix=".MD").name == "markdown"


def test_an_unsupported_format_is_a_coverage_gap_not_a_corrupt_document() -> None:
    with pytest.raises(UnsupportedFormatError, match="no parser for suffix"):
        select_parser(source_type=SourceType.POLICY, suffix=".pdf")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_preserves_text_exactly() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)
    assert parsed.text == MARKDOWN.decode("utf-8")
    assert "# Refund Policy" in parsed.text, "heading syntax must survive"


def test_markdown_offsets_index_into_the_returned_text() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)
    for section in parsed.sections:
        assert 0 <= section.char_start < section.char_end <= len(parsed.text)


def test_markdown_builds_a_nested_heading_trail() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)
    paths = [s.path for s in parsed.sections]
    assert ("Refund Policy",) in paths
    assert ("Refund Policy", "Approval limits") in paths
    assert ("Refund Policy", "Approval limits", "Delegation") in paths
    assert ("Refund Policy", "Exceptions") in paths


def test_a_section_ends_at_the_next_heading_of_equal_or_shallower_level() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)
    limits = next(s for s in parsed.sections if s.path[-1:] == ("Approval limits",))
    exceptions = next(s for s in parsed.sections if s.path[-1:] == ("Exceptions",))
    assert limits.char_end == exceptions.char_start


def test_text_before_the_first_heading_is_kept_as_a_preamble_section() -> None:
    """Dropping it would make a value stated in a document's lead-in uncitable."""
    parsed = MarkdownParser().parse(MARKDOWN)
    preamble = next(s for s in parsed.sections if s.path == ())
    assert parsed.text[preamble.char_start : preamble.char_end].startswith("Preamble")


def test_section_lookup_returns_the_innermost_owner() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)
    offset = parsed.text.index("A deputy may be named")
    owner = parsed.section_for(offset)
    assert owner is not None
    assert owner.path[-1] == "Delegation"


def test_an_empty_document_fails_loudly() -> None:
    """A silently-empty version looks ingested but holds no evidence."""
    with pytest.raises(ParseError, match="empty"):
        MarkdownParser().parse(b"   \n\n  ")


def test_markdown_is_deterministic() -> None:
    first = MarkdownParser().parse(MARKDOWN)
    second = MarkdownParser().parse(MARKDOWN)
    assert first == second


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

CONFIG = b'{"auth": {"reset_token": {"ttl_hours": 24, "single_use": true}}, "api": {"max_page_size": 100}}'


def test_json_renders_a_citable_flat_form() -> None:
    parsed = JsonParser().parse(CONFIG)
    assert "auth.reset_token.ttl_hours: 24" in parsed.text
    assert "api.max_page_size: 100" in parsed.text


def test_json_rendering_is_key_sorted_so_reformatting_is_idempotent() -> None:
    """A reordered or reindented config must not create a new source version."""
    reordered = b'{"api": {"max_page_size": 100}, "auth": {"reset_token": {"single_use": true, "ttl_hours": 24}}}'
    assert JsonParser().parse(CONFIG).text == JsonParser().parse(reordered).text


def test_json_offsets_locate_each_leaf() -> None:
    parsed = JsonParser().parse(CONFIG)
    for section in parsed.sections:
        rendered = parsed.text[section.char_start : section.char_end]
        assert rendered.endswith("\n")
        assert ".".join(section.path) in rendered


def test_json_records_the_raw_hash_so_the_stored_artifact_stays_verifiable() -> None:
    parsed = JsonParser().parse(CONFIG)
    assert len(parsed.metadata["raw_sha256"]) == 64


def test_json_flattens_arrays_by_index() -> None:
    parsed = JsonParser().parse(b'{"tiers": ["gold", "silver"]}')
    assert 'tiers.0: "gold"' in parsed.text
    assert 'tiers.1: "silver"' in parsed.text


def test_json_leaf_rendering_cannot_break_the_one_leaf_per_line_invariant() -> None:
    """An embedded newline in a value would otherwise split one leaf across two lines."""
    parsed = JsonParser().parse(b'{"note": "line one\\nline two"}')
    assert len([ln for ln in parsed.text.splitlines() if ln]) == 1


def test_invalid_json_fails_loudly() -> None:
    with pytest.raises(ParseError, match="invalid JSON"):
        JsonParser().parse(b"{not json")


def test_json_with_no_leaves_fails() -> None:
    with pytest.raises(ParseError, match="no leaf values"):
        JsonParser().parse(b"{}")


def test_deeply_nested_json_is_bounded() -> None:
    """A hostile document must not exhaust the stack during ingestion."""
    nested = b'{"a":' * 40 + b"1" + b"}" * 40
    with pytest.raises(ParseError, match="nesting exceeds"):
        JsonParser().parse(nested)


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_plaintext_is_the_identity_parser() -> None:
    parsed = PlainTextParser().parse(b"Retention is 90 days.")
    assert parsed.text == "Retention is 90 days."
    assert parsed.sections[0].char_start == 0
    assert parsed.sections[0].char_end == len(parsed.text)


def test_undecodable_bytes_degrade_rather_than_drop_the_document() -> None:
    """Unreadable evidence is visible to a reviewer; missing evidence is not."""
    parsed = PlainTextParser().parse(b"caf\xe9 threshold 90")
    assert "threshold 90" in parsed.text


def test_parse_payload_dispatches() -> None:
    parsed = parse_payload(MARKDOWN, source_type=SourceType.POLICY, suffix=".md")
    assert parsed.parser_name == "markdown"


def test_metadata_never_carries_authority() -> None:
    """Authority comes from registration; a document cannot promote itself."""
    for parsed in (
        MarkdownParser().parse(MARKDOWN),
        JsonParser().parse(CONFIG),
        PlainTextParser().parse(b"text"),
    ):
        assert "authority_class" not in parsed.metadata
        assert "tenant_id" not in parsed.metadata
        assert "permissions" not in parsed.metadata
