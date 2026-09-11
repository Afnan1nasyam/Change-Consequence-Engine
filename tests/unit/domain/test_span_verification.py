"""Span verification: exact, and deliberately unforgiving.

The one property that matters most in the whole system is asserted here:
:func:`test_verification_is_exact_not_fuzzy`. An approximate citation check is worse than
none, because it reports a span as verified while permitting a span that does not say what
the binding claims.
"""

from __future__ import annotations

import pytest

from app.domain.services.span_verification import (
    SpanFailure,
    locate_quote,
    verify_offsets,
    verify_value_in_span,
)

DOC = "Refunds of 10,000 USD or more require manager approval before release."


# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------


def test_a_correct_span_verifies() -> None:
    assert verify_offsets(DOC, char_start=0, char_end=7, claimed="Refunds").ok


@pytest.mark.parametrize(
    ("claimed", "failure"),
    [
        ("Refunds of 9,000 USD", SpanFailure.TEXT_MISMATCH),
        ("refunds", SpanFailure.TEXT_MISMATCH),
    ],
)
def test_verification_is_exact_not_fuzzy(claimed: str, failure: str) -> None:
    """No similarity threshold. A near-miss is a miss."""
    result = verify_offsets(DOC, char_start=0, char_end=len(claimed), claimed=claimed)
    assert not result.ok
    assert result.failure is failure


def test_offsets_beyond_the_document_are_rejected() -> None:
    result = verify_offsets(DOC, char_start=0, char_end=len(DOC) + 5, claimed=DOC)
    assert result.failure is SpanFailure.OFFSETS_OUT_OF_RANGE


def test_a_negative_start_is_rejected() -> None:
    assert verify_offsets(DOC, char_start=-1, char_end=5, claimed="Refun").failure is (
        SpanFailure.OFFSETS_OUT_OF_RANGE
    )


@pytest.mark.parametrize(("start", "end"), [(5, 5), (9, 4)])
def test_a_degenerate_span_is_rejected(start: int, end: int) -> None:
    assert verify_offsets(DOC, char_start=start, char_end=end, claimed="x").failure is (
        SpanFailure.OFFSETS_NOT_ORDERED
    )


def test_the_failure_detail_shows_both_sides() -> None:
    """A reviewer must be able to see what the document says without opening it."""
    result = verify_offsets(DOC, char_start=0, char_end=7, claimed="Credits")
    assert "Refunds" in result.detail
    assert "Credits" in result.detail


# ---------------------------------------------------------------------------
# Locating a quote
# ---------------------------------------------------------------------------


def test_a_quote_is_located_and_verifies_at_its_offsets() -> None:
    located = locate_quote(DOC, "10,000 USD")
    assert located is not None
    assert DOC[located.char_start : located.char_end] == "10,000 USD"
    assert verify_offsets(
        DOC, char_start=located.char_start, char_end=located.char_end, claimed=located.text
    ).ok


def test_an_absent_quote_is_not_located() -> None:
    """This is the whole point: the extractor cannot supply offsets, only a quote."""
    assert locate_quote(DOC, "Refunds of 1 USD require approval") is None


def test_an_empty_quote_is_not_located() -> None:
    assert locate_quote(DOC, "") is None


def test_the_search_window_bounds_the_match() -> None:
    """A proposal must be verified against the chunk it was actually shown."""
    assert locate_quote(DOC, "Refunds", search_start=10) is None
    assert locate_quote(DOC, "Refunds", search_start=0, search_end=5) is None


def test_first_occurrence_wins_deterministically() -> None:
    text = "90 days. Later, 90 days again."
    first = locate_quote(text, "90 days")
    again = locate_quote(text, "90 days")
    assert first == again
    assert first is not None and first.char_start == 0


# ---------------------------------------------------------------------------
# Value inside the span
# ---------------------------------------------------------------------------


def test_a_value_present_in_the_span_passes() -> None:
    assert verify_value_in_span(DOC, rendered_value="10,000").ok


def test_a_value_absent_from_the_span_is_rejected() -> None:
    """The dangerous case: a real sentence carrying an invented number."""
    result = verify_value_in_span(DOC, rendered_value="5,000")
    assert not result.ok
    assert result.failure is SpanFailure.VALUE_NOT_IN_SPAN


def test_an_empty_value_is_rejected() -> None:
    assert verify_value_in_span(DOC, rendered_value="").failure is (SpanFailure.VALUE_NOT_IN_SPAN)


def test_the_value_check_is_not_normalized() -> None:
    """`10000` does not appear in a document that writes `10,000`."""
    assert not verify_value_in_span(DOC, rendered_value="10000").ok
