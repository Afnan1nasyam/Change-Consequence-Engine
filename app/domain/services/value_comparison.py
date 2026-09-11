"""Unit-aware typed comparison. The gate that stops false contradictions.

Checkpoint 5.6 and 13.5: **a unit mismatch blocks coupling rather than weakening it.**
That is the difference between a system that reports "the refund approval threshold
(10,000 USD) contradicts the fraud review threshold (0.85)" and one that does not.

There is deliberately **no unit conversion** here. A declared unit must match exactly.
Conversion is a real gap (a corpus writing ``1440 min`` where the registry says ``h``
quarantines rather than binds) but it belongs in one audited place alongside the registry,
not scattered through the comparison path where a wrong factor would silently rewrite
findings.
"""

from __future__ import annotations

import math

from app.domain.value_objects.enums import ValueComparison, ValueType
from app.domain.value_objects.typed_value import TypedValue

#: Relative tolerance for float comparison. Values arrive from typed_parse over document
#: text, so the same number can differ in the last bit after a round trip through JSON.
#: Tight enough that 90d and 90.0001d stay distinct.
_REL_TOL = 1e-9


def compare(left: TypedValue, right: TypedValue) -> ValueComparison:
    """Three-way compare. ``INCOMPARABLE`` blocks coupling; it is not a weak difference."""
    if left.value_type is not right.value_type:
        return ValueComparison.INCOMPARABLE
    if left.unit != right.unit:
        return ValueComparison.INCOMPARABLE

    match left.value_type:
        case ValueType.SCALAR | ValueType.DURATION:
            return _compare_numeric(left.raw, right.raw)
        case ValueType.INTERVAL:
            return _compare_interval(left.raw, right.raw)
        case _:
            # enum, boolean, reference: exact equality on the declared payload.
            return ValueComparison.EQUAL if left.raw == right.raw else ValueComparison.DIFFERENT


def are_equal(left: TypedValue, right: TypedValue) -> bool:
    """True only for a genuine match. Incomparable is **not** equal, and not different."""
    return compare(left, right) is ValueComparison.EQUAL


def are_divergent(left: TypedValue, right: TypedValue) -> bool:
    """True only for a comparable, unequal pair - the precondition for a contradiction.

    Incomparable values are excluded, which is what makes this safe to use as the trigger
    for a Class A finding.
    """
    return compare(left, right) is ValueComparison.DIFFERENT


def is_blocked(left: TypedValue, right: TypedValue) -> bool:
    """True when the pair may not be coupled at all (checkpoint 5.6)."""
    return compare(left, right) is ValueComparison.INCOMPARABLE


def _compare_numeric(left: object, right: object) -> ValueComparison:
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        # Unreachable through TypedValue, whose validator pins raw to the declared type.
        return ValueComparison.INCOMPARABLE  # pragma: no cover
    if math.isclose(float(left), float(right), rel_tol=_REL_TOL, abs_tol=0.0):
        return ValueComparison.EQUAL
    return ValueComparison.DIFFERENT


def _compare_interval(left: object, right: object) -> ValueComparison:
    if not isinstance(left, list | tuple) or not isinstance(right, list | tuple):
        return ValueComparison.INCOMPARABLE  # pragma: no cover
    if len(left) != len(right):
        return ValueComparison.INCOMPARABLE  # pragma: no cover
    for a, b in zip(left, right, strict=True):
        # Endpoints are usually numeric, but the schema permits any payload, so fall back
        # to exact equality rather than declaring a string interval incomparable.
        if isinstance(a, int | float) and isinstance(b, int | float):
            if _compare_numeric(a, b) is not ValueComparison.EQUAL:
                return ValueComparison.DIFFERENT
        elif a != b:
            return ValueComparison.DIFFERENT
    return ValueComparison.EQUAL
