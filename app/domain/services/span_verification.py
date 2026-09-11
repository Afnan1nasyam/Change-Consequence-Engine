"""Span verification: the anti-hallucination primitive.

Everything the system claims to have found rests on one mechanical question - *does the
cited span exist in the source, and does it contain what the claim says it contains?* This
module answers exactly that and nothing else.

Deliberately **exact**. There is no fuzzy matching, no similarity threshold, no
normalization beyond what the parser already applied. An approximate citation check is
worse than none: it tells a reviewer the span was verified while permitting a span that
does not say what the binding claims. A quote that fails here is quarantined, and a
quarantined proposal costs recall - which is the correct price.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class SpanFailure(StrEnum):
    """Why verification failed. Recorded on the quarantine record, never suppressed."""

    OFFSETS_OUT_OF_RANGE = "offsets_out_of_range"
    OFFSETS_NOT_ORDERED = "offsets_not_ordered"
    TEXT_MISMATCH = "text_mismatch"
    QUOTE_NOT_FOUND = "quote_not_found"
    VALUE_NOT_IN_SPAN = "value_not_in_span"


class SpanVerification(NamedTuple):
    ok: bool
    failure: SpanFailure | None = None
    detail: str = ""


OK = SpanVerification(ok=True)


class LocatedSpan(NamedTuple):
    """A quote resolved to document-absolute offsets."""

    char_start: int
    char_end: int
    text: str


def verify_offsets(
    document_text: str, *, char_start: int, char_end: int, claimed: str
) -> SpanVerification:
    """The span at those offsets must be exactly ``claimed``.

    This is the check that runs on stored evidence. It is a slice comparison because the
    parser guaranteed one canonical text and the chunker guaranteed document-absolute
    offsets into it.
    """
    if char_end <= char_start:
        return SpanVerification(
            False,
            SpanFailure.OFFSETS_NOT_ORDERED,
            f"char_end ({char_end}) must exceed char_start ({char_start})",
        )
    if char_start < 0 or char_end > len(document_text):
        return SpanVerification(
            False,
            SpanFailure.OFFSETS_OUT_OF_RANGE,
            f"span [{char_start}, {char_end}) outside document of length {len(document_text)}",
        )

    actual = document_text[char_start:char_end]
    if actual != claimed:
        return SpanVerification(
            False,
            SpanFailure.TEXT_MISMATCH,
            f"document holds {actual[:80]!r} at those offsets, claim cites {claimed[:80]!r}",
        )
    return OK


def locate_quote(
    document_text: str, quote: str, *, search_start: int = 0, search_end: int | None = None
) -> LocatedSpan | None:
    """Find ``quote`` verbatim within a window, returning absolute offsets.

    The extractor supplies a quote, never offsets: a model that could supply offsets could
    supply *plausible* offsets, and there would be nothing left to verify. Locating the
    quote ourselves is what converts "the model says this is in the document" into "this is
    in the document at exactly here".

    First occurrence wins, which is deterministic. A quote appearing twice in one chunk is
    the same sentence twice; either occurrence verifies the same claim.
    """
    if not quote:
        return None
    end = len(document_text) if search_end is None else min(search_end, len(document_text))
    index = document_text.find(quote, max(0, search_start), end)
    if index == -1:
        return None
    return LocatedSpan(index, index + len(quote), quote)


def verify_value_in_span(span_text: str, *, rendered_value: str) -> SpanVerification:
    """The cited span must contain the claimed value verbatim.

    Without this, a proposal could cite a real sentence and attach an unrelated number to
    it - a verified span carrying an invented value, which is the most dangerous shape a
    false binding can take, because every surface downstream reports it as span-verified.
    """
    if not rendered_value:
        return SpanVerification(False, SpanFailure.VALUE_NOT_IN_SPAN, "empty value")
    if rendered_value not in span_text:
        return SpanVerification(
            False,
            SpanFailure.VALUE_NOT_IN_SPAN,
            f"value {rendered_value!r} does not appear in the cited span {span_text[:80]!r}",
        )
    return OK
