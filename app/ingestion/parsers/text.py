"""Plain-text and Markdown parsers.

Both return the canonical text unmodified. The Markdown parser adds a heading map; it does
not strip syntax, because an evidence span quoted from a policy is quoted from the file a
reviewer will open.
"""

from __future__ import annotations

import re

from app.domain.value_objects.enums import SourceType
from app.ingestion.identity import PARSER_VERSION, canonical_text
from app.ingestion.parsers.base import ParsedDocument, ParseError, Section, decode

_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)


class PlainTextParser:
    """The identity parser. One section, the whole document."""

    name = "plaintext"

    def supports(self, *, source_type: SourceType, suffix: str) -> bool:
        return suffix in {".txt", ".log", ""}

    def parse(self, payload: bytes) -> ParsedDocument:
        text = canonical_text(decode(payload))
        if not text.strip():
            raise ParseError("plain-text document is empty")
        return ParsedDocument(
            text=text,
            sections=(Section(path=(), char_start=0, char_end=len(text), level=0),),
            parser_name=self.name,
            parser_version=PARSER_VERSION,
        )


class MarkdownParser:
    """Heading-aware, syntax-preserving.

    Sections nest by ATX heading level, and each runs to the start of the next heading of
    the same or shallower level. Offsets index into the unmodified canonical text.
    """

    name = "markdown"

    def supports(self, *, source_type: SourceType, suffix: str) -> bool:
        return suffix in {".md", ".markdown"}

    def parse(self, payload: bytes) -> ParsedDocument:
        text = canonical_text(decode(payload))
        if not text.strip():
            raise ParseError("markdown document is empty")

        headings = [
            (m.start(), len(m.group(1)), m.group(2).strip()) for m in _ATX_HEADING.finditer(text)
        ]
        sections: list[Section] = []

        if not headings or headings[0][0] > 0:
            preamble_end = headings[0][0] if headings else len(text)
            sections.append(Section(path=(), char_start=0, char_end=preamble_end, level=0))

        trail: list[tuple[int, str]] = []
        for index, (start, level, title) in enumerate(headings):
            while trail and trail[-1][0] >= level:
                trail.pop()
            trail.append((level, title))

            end = len(text)
            for later_start, later_level, _ in headings[index + 1 :]:
                if later_level <= level:
                    end = later_start
                    break

            sections.append(
                Section(
                    path=tuple(t for _, t in trail),
                    char_start=start,
                    char_end=end,
                    level=level,
                )
            )

        title = headings[0][2] if headings else ""
        metadata = {"title": title} if title else {}
        return ParsedDocument(
            text=text,
            sections=tuple(sections),
            metadata=metadata,
            parser_name=self.name,
            parser_version=PARSER_VERSION,
        )
