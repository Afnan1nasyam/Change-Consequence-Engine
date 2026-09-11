"""Typed parsing and unit validation for a proposed binding value.

The extractor quotes a value **as it appears in the document** ("10,000", "24", "true").
This service turns that verbatim string into a
:class:`~app.domain.value_objects.typed_value.TypedValue`, or refuses.

Why the model quotes rather than computes: if it supplied a parsed number, the number could
be plausible and unrelated to the text, and the value-in-span check would have nothing to
compare. Quoting keeps the model on the one task it is reliable at - copying - and leaves
conversion to code.

**Unit conversion is deliberately absent.** A declared unit must match the registry entry's
unit exactly; anything else is a mismatch and blocks the binding rather than weakening it
(``DOMAIN_CHECKPOINT.md`` 5.6). Converting hours to minutes here would be a small
convenience that quietly makes ``refund.approval_threshold`` in USD comparable to a
0-1 fraud score in some future refactor. Conversion belongs in one audited place, with the
registry, when coupling needs it.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import NamedTuple

from app.domain.entities.registry import AssumptionKeyRegistryEntry
from app.domain.value_objects.enums import UNIT_REQUIRED_TYPES, ValueType
from app.domain.value_objects.typed_value import TypedValue

#: Thousands separators a real document uses: comma, plain space, no-break space,
#: narrow no-break space. Written as escapes so the source is unambiguous in any editor.
_GROUP_SEPARATORS = (",", " ", chr(0x00A0), chr(0x202F))  # comma, space, NBSP, NNBSP
_SEPARATOR_CLASS = "".join(_GROUP_SEPARATORS)

#: 10000 | 10,000 | 10 000 | 1.5 | -3
_NUMERIC = re.compile(
    rf"^[+-]?\d{{1,3}}(?:[{_SEPARATOR_CLASS}]\d{{3}})+(?:\.\d+)?$|^[+-]?\d+(?:\.\d+)?$"
)

_TRUE = frozenset({"true", "yes", "enabled", "on", "required"})
_FALSE = frozenset({"false", "no", "disabled", "off", "not required"})


class ParseFailure(StrEnum):
    NOT_NUMERIC = "not_numeric"
    NOT_BOOLEAN = "not_boolean"
    EMPTY_VALUE = "empty_value"
    UNIT_MISSING = "unit_missing"
    UNIT_NOT_ALLOWED = "unit_not_allowed"
    UNIT_MISMATCH = "unit_mismatch"
    VALUE_TYPE_MISMATCH = "value_type_mismatch"
    INTERVAL_MALFORMED = "interval_malformed"
    KEY_DEPRECATED = "key_deprecated"


class ParseResult(NamedTuple):
    value: TypedValue | None
    failure: ParseFailure | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


def parse_value(raw: str, *, value_type: ValueType, unit: str | None) -> ParseResult:
    """Parse a verbatim value string under a declared type.

    Does not consult the registry - :func:`validate_against_entry` does that. Keeping the
    two apart is what lets an *unregistered* key still produce a well-typed
    ``KeyRegistryGap``: the gap records what was proposed, and it can only do that if the
    value parsed.
    """
    text = raw.strip()
    if not text:
        return ParseResult(None, ParseFailure.EMPTY_VALUE, "value is empty")

    if value_type in UNIT_REQUIRED_TYPES and not unit:
        return ParseResult(
            None,
            ParseFailure.UNIT_MISSING,
            f"value_type={value_type.value!r} requires a unit",
        )
    if value_type not in UNIT_REQUIRED_TYPES and unit:
        return ParseResult(
            None,
            ParseFailure.UNIT_NOT_ALLOWED,
            f"value_type={value_type.value!r} must not carry a unit, got {unit!r}",
        )

    if value_type in (ValueType.SCALAR, ValueType.DURATION):
        number = _parse_number(text)
        if number is None:
            return ParseResult(None, ParseFailure.NOT_NUMERIC, f"{text!r} is not a number")
        return ParseResult(TypedValue(value_type=value_type, raw=number, unit=unit))

    if value_type is ValueType.BOOLEAN:
        lowered = text.lower()
        if lowered in _TRUE:
            return ParseResult(TypedValue(value_type=value_type, raw=True))
        if lowered in _FALSE:
            return ParseResult(TypedValue(value_type=value_type, raw=False))
        return ParseResult(None, ParseFailure.NOT_BOOLEAN, f"{text!r} is not a recognized boolean")

    if value_type is ValueType.INTERVAL:
        endpoints = _parse_interval(text)
        if endpoints is None:
            return ParseResult(
                None,
                ParseFailure.INTERVAL_MALFORMED,
                f"{text!r} is not two numbers separated by '-' or '..'",
            )
        return ParseResult(TypedValue(value_type=value_type, raw=list(endpoints)))

    # enum and reference are opaque strings; the registry declares their meaning.
    return ParseResult(TypedValue(value_type=value_type, raw=text))


def validate_against_entry(value: TypedValue, entry: AssumptionKeyRegistryEntry) -> ParseResult:
    """The parsed value must be what the registry says this key holds.

    The registry is the declaration of record. A proposal that parses fine but disagrees
    with the entry is not a weaker binding, it is a different variable wearing the same
    key, and admitting it would corrupt every coupling set for that key.
    """
    if not entry.is_active:
        return ParseResult(
            None,
            ParseFailure.KEY_DEPRECATED,
            f"key {entry.key!r} was deprecated in {entry.deprecated_in_version!r}",
        )
    if value.value_type is not entry.value_type:
        return ParseResult(
            None,
            ParseFailure.VALUE_TYPE_MISMATCH,
            f"registry declares {entry.key!r} as {entry.value_type.value!r}, "
            f"proposal declares {value.value_type.value!r}",
        )
    if value.unit != entry.unit:
        return ParseResult(
            None,
            ParseFailure.UNIT_MISMATCH,
            f"registry declares unit {entry.unit!r} for {entry.key!r}, "
            f"proposal declares {value.unit!r}. Unit mismatch blocks coupling; it is not "
            "converted here.",
        )
    return ParseResult(value)


def _parse_number(text: str) -> float | int | None:
    if not _NUMERIC.match(text):
        return None
    stripped = text
    for separator in _GROUP_SEPARATORS:
        stripped = stripped.replace(separator, "")
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:  # pragma: no cover - the regex already guarantees this
        return None


def _parse_interval(text: str) -> tuple[float | int, float | int] | None:
    for separator in ("..", " - ", "-", " to "):
        if separator in text:
            left, _, right = text.partition(separator)
            low = _parse_number(left.strip())
            high = _parse_number(right.strip())
            if low is not None and high is not None:
                return (low, high)
    return None
