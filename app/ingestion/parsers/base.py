"""Parser contract.

A parser turns raw source bytes into one canonical text plus a section map. It does
**not** rewrite, summarize, or clean the text, because every character offset in the
system indexes into what it returns and the span-verification invariant requires the
stored text to contain a cited value verbatim.

Adding a format means adding a module here and registering it. Nothing downstream knows
which parser ran, only ``parser_version``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.domain.value_objects.enums import SourceType


class ParseError(ValueError):
    """The payload is not the format it claimed to be.

    Fail loudly: a document that silently parses to nothing produces an empty version and
    a corpus that looks ingested but holds no evidence.
    """


class Section(BaseModel):
    """A titled region of the canonical text.

    Sections carry offsets rather than copies of their text so that a chunk, an evidence
    span and a section all refer to one string. ``path`` is the heading trail
    (``["Refunds", "Approval limits"]``) or the JSON pointer trail, which is what makes an
    evidence citation legible to a human reviewer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: tuple[str, ...] = ()
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    level: int = Field(default=0, ge=0)

    @property
    def label(self) -> str:
        return " > ".join(self.path) if self.path else "(document)"


class ParsedDocument(BaseModel):
    """Parser output: one canonical text, a section map, and format-derived metadata.

    ``metadata`` is descriptive only. It never carries authority, tenancy, or permissions:
    those come from *registration*, so that a planted file cannot claim policy authority
    (``docs/SECURITY.md``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    sections: tuple[Section, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)
    parser_name: str
    parser_version: str

    def section_for(self, char_start: int) -> Section | None:
        """The innermost section containing an offset, or ``None`` for preamble text."""
        best: Section | None = None
        for section in self.sections:
            if section.char_start <= char_start < section.char_end and (
                best is None or section.level >= best.level
            ):
                best = section
        return best


@runtime_checkable
class SourceParser(Protocol):
    @property
    def name(self) -> str: ...

    def supports(self, *, source_type: SourceType, suffix: str) -> bool: ...

    def parse(self, payload: bytes) -> ParsedDocument: ...


def decode(payload: bytes) -> str:
    """UTF-8 with a strict fallback chain.

    Latin-1 never fails, so a mis-declared encoding degrades to mojibake rather than
    dropping the document. That is the right trade here: unreadable evidence is visible to
    a reviewer, missing evidence is not.
    """
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("latin-1", errors="replace")
