"""Typed parsing and registry agreement.

The decision under test is that **unit mismatch blocks rather than weakens**. A permissive
converter here is how a USD threshold eventually becomes comparable to a 0-1 fraud score.
"""

from __future__ import annotations

import pytest

from app.domain.services.typed_parse import (
    ParseFailure,
    parse_value,
    validate_against_entry,
)
from app.domain.value_objects.enums import ValueType
from app.domain.value_objects.typed_value import TypedValue
from tests.fixtures.domain import registry_entry

# ---------------------------------------------------------------------------
# Parsing a verbatim value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("10000", 10000), ("10,000", 10000), ("1.5", 1.5), ("-3", -3), (" 90 ", 90)],
)
def test_numbers_as_a_document_writes_them(raw: str, expected: float) -> None:
    """The extractor copies `10,000`; the parser is what turns it into a number."""
    result = parse_value(raw, value_type=ValueType.SCALAR, unit="USD")
    assert result.value is not None
    assert result.value.raw == expected


def test_a_non_number_is_refused() -> None:
    result = parse_value("twenty-four", value_type=ValueType.DURATION, unit="h")
    assert result.failure is ParseFailure.NOT_NUMERIC


def test_a_malformed_digit_group_is_refused() -> None:
    assert parse_value("10,00", value_type=ValueType.SCALAR, unit="USD").failure is (
        ParseFailure.NOT_NUMERIC
    )


def test_an_empty_value_is_refused() -> None:
    assert parse_value("  ", value_type=ValueType.SCALAR, unit="USD").failure is (
        ParseFailure.EMPTY_VALUE
    )


def test_a_scalar_without_a_unit_is_refused() -> None:
    assert parse_value("10000", value_type=ValueType.SCALAR, unit=None).failure is (
        ParseFailure.UNIT_MISSING
    )


def test_a_unit_on_a_type_that_has_none_is_refused() -> None:
    assert parse_value("true", value_type=ValueType.BOOLEAN, unit="USD").failure is (
        ParseFailure.UNIT_NOT_ALLOWED
    )


@pytest.mark.parametrize("raw", ["true", "yes", "enabled", "required", "TRUE"])
def test_boolean_truth_spellings(raw: str) -> None:
    result = parse_value(raw, value_type=ValueType.BOOLEAN, unit=None)
    assert result.value is not None and result.value.raw is True


@pytest.mark.parametrize("raw", ["false", "no", "disabled", "not required"])
def test_boolean_false_spellings(raw: str) -> None:
    result = parse_value(raw, value_type=ValueType.BOOLEAN, unit=None)
    assert result.value is not None and result.value.raw is False


def test_an_unrecognized_boolean_is_refused() -> None:
    assert parse_value("maybe", value_type=ValueType.BOOLEAN, unit=None).failure is (
        ParseFailure.NOT_BOOLEAN
    )


@pytest.mark.parametrize("raw", ["0..1", "0 - 10", "5-10", "1 to 9"])
def test_interval_spellings(raw: str) -> None:
    result = parse_value(raw, value_type=ValueType.INTERVAL, unit=None)
    assert result.value is not None
    assert len(result.value.raw) == 2


def test_a_malformed_interval_is_refused() -> None:
    assert parse_value("about ten", value_type=ValueType.INTERVAL, unit=None).failure is (
        ParseFailure.INTERVAL_MALFORMED
    )


@pytest.mark.parametrize("value_type", [ValueType.ENUM, ValueType.REFERENCE])
def test_enum_and_reference_stay_opaque_strings(value_type: ValueType) -> None:
    """The registry declares their meaning; the parser does not invent one."""
    result = parse_value("manager", value_type=value_type, unit=None)
    assert result.value is not None and result.value.raw == "manager"


# ---------------------------------------------------------------------------
# Agreement with the registry entry
# ---------------------------------------------------------------------------


def test_a_matching_value_agrees_with_its_entry() -> None:
    entry = registry_entry()  # scalar / USD
    value = TypedValue(value_type=ValueType.SCALAR, raw=10000, unit="USD")
    assert validate_against_entry(value, entry).value is value


def test_a_unit_mismatch_blocks() -> None:
    """Not converted, not weakened. Blocked."""
    entry = registry_entry()  # USD
    value = TypedValue(value_type=ValueType.SCALAR, raw=10000, unit="EUR")
    result = validate_against_entry(value, entry)
    assert result.failure is ParseFailure.UNIT_MISMATCH
    assert "blocks coupling" in result.detail


def test_hours_are_not_silently_converted_to_the_registered_unit() -> None:
    entry = registry_entry(key="auth.reset_token.ttl", value_type=ValueType.DURATION, unit="h")
    value = TypedValue(value_type=ValueType.DURATION, raw=1440, unit="min")
    assert validate_against_entry(value, entry).failure is ParseFailure.UNIT_MISMATCH


def test_a_value_type_mismatch_is_refused() -> None:
    """A different variable wearing the same key would corrupt the coupling set."""
    entry = registry_entry()  # scalar
    value = TypedValue(value_type=ValueType.ENUM, raw="10000")
    assert validate_against_entry(value, entry).failure is (ParseFailure.VALUE_TYPE_MISMATCH)


def test_a_deprecated_key_is_refused() -> None:
    entry = registry_entry(deprecated_in_version="registry-4")
    value = TypedValue(value_type=ValueType.SCALAR, raw=10000, unit="USD")
    result = validate_against_entry(value, entry)
    assert result.failure is ParseFailure.KEY_DEPRECATED
    assert "registry-4" in result.detail
