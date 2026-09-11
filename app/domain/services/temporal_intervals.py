"""Half-open interval arithmetic: ``[effective_from, effective_to)``.

Half-open is not a stylistic choice. It makes succession exact: a policy ending
``2025-07-01`` and its replacement starting ``2025-07-01`` have **no overlap and no gap**,
so a corpus that revises a document cleanly produces no spurious ambiguity. Under closed
intervals the boundary instant would belong to both versions and every clean succession
would report two competing authorities.

``None`` means unbounded on that side: ``effective_from = None`` is "has always applied",
``effective_to = None`` is "still applies".

Every function here is pure and total. There is no notion of "now" - a caller must pass
``as_of`` - because a resolver that could read the clock would not be reproducible.
"""

from __future__ import annotations

import datetime as dt

#: An interval as carried by an AssumptionRecord or a SourceVersion.
Interval = tuple[dt.datetime | None, dt.datetime | None]


def contains_instant(interval: Interval, as_of: dt.datetime) -> bool:
    """Whether ``as_of`` falls inside ``[from, to)``.

    ``as_of == effective_from`` is **inside**; ``as_of == effective_to`` is **outside**.
    """
    start, end = interval
    if start is not None and as_of < start:
        return False
    return not (end is not None and as_of >= end)


def overlaps(left: Interval, right: Interval) -> bool:
    """Whether two half-open intervals share at least one instant.

    Adjacent intervals - one ending exactly where the other begins - do **not** overlap.
    This is the property that keeps a clean document succession out of the contradiction
    path (checkpoint 5.4.4: non-overlapping means *superseded*, not *conflicting*).
    """
    left_start, left_end = left
    right_start, right_end = right

    if left_end is not None and right_start is not None and left_end <= right_start:
        return False
    return not (right_end is not None and left_start is not None and right_end <= left_start)


def intersection(left: Interval, right: Interval) -> Interval | None:
    """The shared span, or ``None`` when the intervals do not overlap."""
    if not overlaps(left, right):
        return None
    left_start, left_end = left
    right_start, right_end = right

    start = _later(left_start, right_start)
    end = _earlier(left_end, right_end)
    return (start, end)


def overlap_fraction(left: Interval, right: Interval, *, horizon: dt.timedelta) -> float:
    """Share of the shorter interval that the two have in common, in ``[0, 1]``.

    Feeds coupling strength, where a longer shared validity is a stronger coupling. An
    unbounded side is clamped to ``horizon`` so the result stays finite and comparable;
    without a clamp every open-ended pair would score 1.0 and the factor would carry no
    information on a corpus where most documents are open-ended.
    """
    shared = intersection(left, right)
    if shared is None:
        return 0.0

    shared_len = _length(shared, horizon=horizon)
    shortest = min(_length(left, horizon=horizon), _length(right, horizon=horizon))
    if shortest <= 0.0:
        return 0.0
    return min(1.0, shared_len / shortest)


def _length(interval: Interval, *, horizon: dt.timedelta) -> float:
    """Duration in seconds, with unbounded sides clamped to ``horizon``."""
    start, end = interval
    if start is None and end is None:
        return horizon.total_seconds()
    if start is None or end is None:
        return horizon.total_seconds()
    return max(0.0, (end - start).total_seconds())


def _later(a: dt.datetime | None, b: dt.datetime | None) -> dt.datetime | None:
    """The later of two lower bounds; ``None`` is unbounded and therefore earlier."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _earlier(a: dt.datetime | None, b: dt.datetime | None) -> dt.datetime | None:
    """The earlier of two upper bounds; ``None`` is unbounded and therefore later."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)
