"""Half-open interval arithmetic, boundary by boundary.

Every one of the temporal edge cases the phase requires is pinned here: adjacency,
exact ``effective_from``, exact ``effective_to``, open-ended sides, stale records, future
records, and competing overlaps. The single property that makes the rest work is that
**adjacent intervals do not overlap** - so a clean document succession produces neither a
contradiction nor an ambiguity.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.domain.services.temporal_intervals import (
    contains_instant,
    intersection,
    overlap_fraction,
    overlaps,
)

JAN = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
APR = dt.datetime(2025, 4, 1, tzinfo=dt.UTC)
JUL = dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
OCT = dt.datetime(2025, 10, 1, tzinfo=dt.UTC)
NEXT = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

HORIZON = dt.timedelta(days=365 * 5)


# ---------------------------------------------------------------------------
# contains_instant - the "is this binding in force" question
# ---------------------------------------------------------------------------


def test_an_instant_inside_a_bounded_interval_is_contained() -> None:
    assert contains_instant((JAN, JUL), APR)


def test_the_exact_effective_from_is_inside() -> None:
    """Half-open is closed on the left: the day a policy takes effect, it applies."""
    assert contains_instant((JAN, JUL), JAN)


def test_the_exact_effective_to_is_outside() -> None:
    """Half-open is open on the right: the day it expires, it no longer applies."""
    assert not contains_instant((JAN, JUL), JUL)


def test_a_stale_record_is_not_in_force() -> None:
    assert not contains_instant((JAN, APR), JUL)


def test_a_future_record_is_not_in_force() -> None:
    assert not contains_instant((OCT, NEXT), APR)


def test_an_open_ended_interval_still_applies() -> None:
    assert contains_instant((JAN, None), NEXT)


def test_an_interval_with_no_start_has_always_applied() -> None:
    assert contains_instant((None, JUL), JAN)


def test_a_wholly_unbounded_interval_contains_everything() -> None:
    assert contains_instant((None, None), JAN)
    assert contains_instant((None, None), NEXT)


# ---------------------------------------------------------------------------
# overlaps - the contradiction precondition
# ---------------------------------------------------------------------------


def test_adjacent_intervals_do_not_overlap() -> None:
    """The load-bearing case: a clean succession is superseded, not conflicting."""
    assert not overlaps((JAN, JUL), (JUL, NEXT))
    assert not overlaps((JUL, NEXT), (JAN, JUL))


def test_intervals_sharing_more_than_a_boundary_overlap() -> None:
    assert overlaps((JAN, OCT), (JUL, NEXT))


def test_a_contained_interval_overlaps_its_container() -> None:
    assert overlaps((JAN, NEXT), (APR, JUL))


def test_separated_intervals_do_not_overlap() -> None:
    assert not overlaps((JAN, APR), (OCT, NEXT))


def test_two_open_ended_intervals_always_overlap() -> None:
    """Most real documents are open-ended, so this is the common case, not the corner."""
    assert overlaps((JAN, None), (OCT, None))


def test_an_open_ended_interval_overlaps_a_later_bounded_one() -> None:
    assert overlaps((JAN, None), (OCT, NEXT))


def test_an_open_ended_interval_does_not_overlap_an_earlier_closed_one() -> None:
    assert not overlaps((OCT, None), (JAN, APR))


@pytest.mark.parametrize(
    "left,right",
    [
        ((JAN, JUL), (JUL, NEXT)),
        ((JAN, OCT), (JUL, NEXT)),
        ((JAN, None), (OCT, None)),
        ((None, None), (JAN, APR)),
    ],
)
def test_overlap_is_symmetric(
    left: tuple[dt.datetime | None, dt.datetime | None],
    right: tuple[dt.datetime | None, dt.datetime | None],
) -> None:
    assert overlaps(left, right) == overlaps(right, left)


# ---------------------------------------------------------------------------
# intersection
# ---------------------------------------------------------------------------


def test_intersection_is_the_shared_span() -> None:
    assert intersection((JAN, OCT), (APR, NEXT)) == (APR, OCT)


def test_intersection_of_non_overlapping_intervals_is_none() -> None:
    assert intersection((JAN, APR), (OCT, NEXT)) is None


def test_intersection_of_adjacent_intervals_is_none() -> None:
    assert intersection((JAN, JUL), (JUL, NEXT)) is None


def test_intersection_treats_none_as_unbounded_on_the_correct_side() -> None:
    """`None` as a lower bound is earlier than anything; as an upper bound, later."""
    assert intersection((None, OCT), (APR, None)) == (APR, OCT)


# ---------------------------------------------------------------------------
# overlap_fraction - a coupling-strength factor
# ---------------------------------------------------------------------------


def test_identical_intervals_overlap_completely() -> None:
    assert overlap_fraction((JAN, JUL), (JAN, JUL), horizon=HORIZON) == 1.0


def test_non_overlapping_intervals_score_zero() -> None:
    assert overlap_fraction((JAN, APR), (OCT, NEXT), horizon=HORIZON) == 0.0


def test_a_contained_interval_overlaps_its_container_completely() -> None:
    """The shorter side is the denominator, so containment is total overlap for it."""
    assert overlap_fraction((JAN, NEXT), (APR, JUL), horizon=HORIZON) == 1.0


def test_partial_overlap_scores_between_zero_and_one() -> None:
    fraction = overlap_fraction((JAN, JUL), (APR, OCT), horizon=HORIZON)
    assert 0.0 < fraction < 1.0


def test_the_fraction_stays_bounded() -> None:
    """An unbounded side is clamped, so the factor cannot escape [0, 1]."""
    for pair in (
        ((None, None), (None, None)),
        ((JAN, None), (APR, None)),
        ((None, JUL), (JAN, NEXT)),
    ):
        assert 0.0 <= overlap_fraction(*pair, horizon=HORIZON) <= 1.0
