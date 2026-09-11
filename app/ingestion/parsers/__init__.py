"""Parser selection.

The registry is an explicit ordered tuple rather than a plugin scan: which parser handled
a document is part of its provenance, so the mapping must be readable and stable rather
than discovered at import time.

Adding PDF, HTML, a Git connector or a ticket connector means appending a module here.
Nothing downstream changes, because everything downstream consumes
:class:`~app.ingestion.parsers.base.ParsedDocument`.
"""

from __future__ import annotations

from app.domain.value_objects.enums import SourceType
from app.ingestion.parsers.base import (
    ParsedDocument,
    ParseError,
    Section,
    SourceParser,
    decode,
)
from app.ingestion.parsers.json_doc import JsonParser
from app.ingestion.parsers.text import MarkdownParser, PlainTextParser

#: Ordered by specificity. First match wins.
PARSERS: tuple[SourceParser, ...] = (MarkdownParser(), JsonParser(), PlainTextParser())


class UnsupportedFormatError(ParseError):
    """No registered parser claims this format.

    Distinct from :class:`ParseError` so that "we cannot read this yet" is reported as a
    corpus-coverage gap rather than a corrupt document.
    """


def select_parser(*, source_type: SourceType, suffix: str) -> SourceParser:
    normalized = suffix.lower()
    for parser in PARSERS:
        if parser.supports(source_type=source_type, suffix=normalized):
            return parser
    raise UnsupportedFormatError(
        f"no parser for suffix {suffix!r} (source_type={source_type.value!r}); "
        f"registered parsers: {[p.name for p in PARSERS]}"
    )


def parse_payload(payload: bytes, *, source_type: SourceType, suffix: str) -> ParsedDocument:
    return select_parser(source_type=source_type, suffix=suffix).parse(payload)


__all__ = [
    "PARSERS",
    "JsonParser",
    "MarkdownParser",
    "ParseError",
    "ParsedDocument",
    "PlainTextParser",
    "Section",
    "SourceParser",
    "UnsupportedFormatError",
    "decode",
    "parse_payload",
    "select_parser",
]
