"""Unit-aware comparison, and the mismatch gate that stops false contradictions.

The negative cases carry the weight here. ``INCOMPARABLE`` must never behave like a weak
``DIFFERENT``: a unit mismatch blocks coupling outright (checkpoint 5.6), and every caller
in the resolver branches on that distinction.
"""

from __future__ import annotations

import pytest

from app.domain.services.value_comparison import (
    are_divergent,
    are_equal,
    compare,
    is_blocked,
)
from app.domain.value_objects.enums import ValueComparison, ValueType
from app.domain.value_objects.typed_value import TypedValue
from tests.fixtures.coupling import days, score, usd


def test_identical_scalars_are_equal() -> None:
    assert compare(usd(10000), usd(10000)) is ValueComparison.EQUAL


def test_different_scalars_are_different() -> None:
    assert compare(usd(10000), usd(7500)) is ValueComparison.DIFFERENT


def test_an_int_and_the_equivalent_float_are_equal() -> None:
    """Typed parse over document text can yield either; 90 and 90.0 are one value."""
    left = TypedValue(value_type=ValueType.DURATION, raw=90, unit="d")
    right = TypedValue(value_type=ValueType.DURATION, raw=90.0, unit="d")
    assert are_equal(left, right)


def test_a_float_that_differs_in_the_last_bit_is_still_equal() -> None:
    """A JSON round trip must not manufacture a contradiction."""
    left = TypedValue(value_type=ValueType.SCALAR, raw=0.1 + 0.2, unit="USD")
    right = TypedValue(value_type=ValueType.SCALAR, raw=0.3, unit="USD")
    assert are_equal(left, right)


def test_a_genuinely_close_but_distinct_value_is_different() -> None:
    """The tolerance is for representation, not for rounding two real values together."""
    assert are_divergent(days(90), days(90.0001))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_a_unit_mismatch_is_incomparable_not_different() -> None:
    """The canonical counterexample: a USD threshold and a 0-1 fraud score."""
    result = compare(usd(10000), score(0.85))
    assert result is ValueComparison.INCOMPARABLE
    # And explicitly not the weaker verdict: callers branch on the difference.
    assert result.value != ValueComparison.DIFFERENT.value


def test_a_unit_mismatch_blocks_rather_than_diverges() -> None:
    """`are_divergent` is the contradiction trigger and must refuse a blocked pair."""
    assert is_blocked(usd(10000), score(0.85))
    assert not are_divergent(usd(10000), score(0.85))
    assert not are_equal(usd(10000), score(0.85))


def test_a_type_mismatch_is_incomparable() -> None:
    boolean = TypedValue(value_type=ValueType.BOOLEAN, raw=True)
    assert compare(boolean, usd(1)) is ValueComparison.INCOMPARABLE


def test_the_same_number_in_different_units_is_incomparable() -> None:
    """There is no unit conversion. 24h and 24d are not the same, and 1440min is not 24h."""
    hours = TypedValue(value_type=ValueType.DURATION, raw=24, unit="h")
    days_value = TypedValue(value_type=ValueType.DURATION, raw=24, unit="d")
    assert compare(hours, days_value) is ValueComparison.INCOMPARABLE

    minutes = TypedValue(value_type=ValueType.DURATION, raw=1440, unit="min")
    assert compare(minutes, hours) is ValueComparison.INCOMPARABLE


# ---------------------------------------------------------------------------
# Non-numeric payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value_type,left,right,expected",
    [
        (ValueType.ENUM, "manual", "manual", ValueComparison.EQUAL),
        (ValueType.ENUM, "manual", "automatic", ValueComparison.DIFFERENT),
        (ValueType.BOOLEAN, True, True, ValueComparison.EQUAL),
        (ValueType.BOOLEAN, True, False, ValueComparison.DIFFERENT),
        (ValueType.REFERENCE, "SOP-14", "SOP-14", ValueComparison.EQUAL),
        (ValueType.REFERENCE, "SOP-14", "SOP-15", ValueComparison.DIFFERENT),
    ],
)
def test_exact_equality_for_categorical_payloads(
    value_type: ValueType, left: object, right: object, expected: ValueComparison
) -> None:
    a = TypedValue(value_type=value_type, raw=left)
    b = TypedValue(value_type=value_type, raw=right)
    assert compare(a, b) is expected


def test_intervals_compare_endpoint_by_endpoint() -> None:
    a = TypedValue(value_type=ValueType.INTERVAL, raw=[10, 20])
    b = TypedValue(value_type=ValueType.INTERVAL, raw=[10, 20])
    c = TypedValue(value_type=ValueType.INTERVAL, raw=[10, 30])
    assert compare(a, b) is ValueComparison.EQUAL
    assert compare(a, c) is ValueComparison.DIFFERENT


def test_an_interval_of_strings_falls_back_to_exact_equality() -> None:
    """The schema permits any payload, so a non-numeric interval must not read as blocked."""
    a = TypedValue(value_type=ValueType.INTERVAL, raw=["q1", "q4"])
    b = TypedValue(value_type=ValueType.INTERVAL, raw=["q1", "q4"])
    c = TypedValue(value_type=ValueType.INTERVAL, raw=["q1", "q3"])
    assert compare(a, b) is ValueComparison.EQUAL
    assert compare(a, c) is ValueComparison.DIFFERENT


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        (usd(10000), usd(7500)),
        (usd(10000), usd(10000)),
        (usd(10000), score(0.85)),
        (days(90), usd(90)),
    ],
)
def test_comparison_is_symmetric(left: TypedValue, right: TypedValue) -> None:
    """An asymmetric comparison would make a finding depend on visit order."""
    assert compare(left, right) is compare(right, left)


@pytest.mark.parametrize("value", [usd(10000), days(90), score(0.5)])
def test_comparison_is_reflexive(value: TypedValue) -> None:
    assert compare(value, value) is ValueComparison.EQUAL


def test_the_three_outcomes_are_mutually_exclusive() -> None:
    """Exactly one of equal / divergent / blocked holds for any pair."""
    pairs = [
        (usd(10000), usd(10000)),
        (usd(10000), usd(7500)),
        (usd(10000), score(0.85)),
    ]
    for left, right in pairs:
        flags = [are_equal(left, right), are_divergent(left, right), is_blocked(left, right)]
        assert sum(flags) == 1, (left, right, flags)
