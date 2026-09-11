"""TypedValue structural validation.

A unitless scalar cannot be compared safely, and an unsafe comparison is precisely how a
false contradiction gets manufactured - so the schema refuses to represent one. The C1
counterexample (a USD threshold versus a 0-1 fraud score) is the case this is protecting.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.value_objects.enums import UNIT_REQUIRED_TYPES, ValueType
from app.domain.value_objects.typed_value import TypedValue


@pytest.mark.parametrize("value_type", sorted(UNIT_REQUIRED_TYPES))
def test_unit_is_required_for_scalars_and_durations(value_type: ValueType) -> None:
    with pytest.raises(ValidationError, match="unit is required"):
        TypedValue(value_type=value_type, raw=10)


@pytest.mark.parametrize(
    ("value_type", "raw"),
    [
        (ValueType.BOOLEAN, True),
        (ValueType.ENUM, "manager"),
        (ValueType.REFERENCE, "policy://refund"),
        (ValueType.INTERVAL, [1, 2]),
    ],
)
def test_unit_is_forbidden_where_it_is_meaningless(value_type: ValueType, raw: object) -> None:
    with pytest.raises(ValidationError, match="unit must be omitted"):
        TypedValue(value_type=value_type, raw=raw, unit="USD")


def test_scalar_with_unit_is_accepted() -> None:
    value = TypedValue(value_type=ValueType.SCALAR, raw=10000, unit="USD")
    assert value.render() == "10000USD"


def test_a_bool_is_not_a_scalar() -> None:
    """`True` is an `int` in Python. Letting it through would make 1 == True == 1 USD."""
    with pytest.raises(ValidationError, match="received a bool"):
        TypedValue(value_type=ValueType.SCALAR, raw=True, unit="USD")
    with pytest.raises(ValidationError, match="received a bool"):
        TypedValue(value_type=ValueType.DURATION, raw=False, unit="d")


@pytest.mark.parametrize(
    ("value_type", "raw", "unit"),
    [
        (ValueType.SCALAR, "10000", "USD"),
        (ValueType.DURATION, "90", "d"),
        (ValueType.BOOLEAN, "yes", None),
        (ValueType.ENUM, 3, None),
        (ValueType.REFERENCE, 3, None),
        (ValueType.INTERVAL, 3, None),
    ],
)
def test_raw_must_match_the_declared_type(
    value_type: ValueType, raw: object, unit: str | None
) -> None:
    with pytest.raises(ValidationError, match="expects raw of"):
        TypedValue(value_type=value_type, raw=raw, unit=unit)


@pytest.mark.parametrize("raw", [[1], [1, 2, 3], []])
def test_an_interval_has_exactly_two_endpoints(raw: list[int]) -> None:
    with pytest.raises(ValidationError, match="exactly two endpoints"):
        TypedValue(value_type=ValueType.INTERVAL, raw=raw)


def test_interval_accepts_two_endpoints() -> None:
    assert TypedValue(value_type=ValueType.INTERVAL, raw=(0, 10)).raw == (0, 10)


def test_unit_mismatch_is_visible_at_the_schema_layer() -> None:
    """C1: adjacent vocabulary, distinct variables, incompatible units.

    The deferred resolver blocks coupling outright on this; the value object answers the
    cheap part of the question - are these even nominally the same quantity?
    """
    threshold = TypedValue(value_type=ValueType.SCALAR, raw=10000, unit="USD")
    fraud_score = TypedValue(value_type=ValueType.SCALAR, raw=0.7, unit="score")
    assert not threshold.is_unit_compatible_with(fraud_score)


def test_same_type_and_unit_is_compatible() -> None:
    a = TypedValue(value_type=ValueType.SCALAR, raw=10000, unit="USD")
    b = TypedValue(value_type=ValueType.SCALAR, raw=5000, unit="USD")
    assert a.is_unit_compatible_with(b)


def test_a_duration_is_not_a_scalar_even_with_the_same_unit_string() -> None:
    scalar = TypedValue(value_type=ValueType.SCALAR, raw=90, unit="d")
    duration = TypedValue(value_type=ValueType.DURATION, raw=90, unit="d")
    assert not scalar.is_unit_compatible_with(duration)


def test_typed_value_is_frozen_and_forbids_extra_fields() -> None:
    value = TypedValue(value_type=ValueType.ENUM, raw="manager")
    with pytest.raises(ValidationError):
        value.raw = "director"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TypedValue(value_type=ValueType.ENUM, raw="manager", confidence=0.9)  # type: ignore[call-arg]
