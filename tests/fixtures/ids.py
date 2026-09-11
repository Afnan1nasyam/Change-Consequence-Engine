"""Primitives shared by every fixture module.

A leaf: it imports nothing from ``tests.fixtures``. That is the point. ``domain.py`` and
``coupling.py`` both need ``make_id`` and ``AS_OF``, and ``domain.py`` also merges
``coupling.py``'s sample inventory at the bottom - so if either of those two owned the
primitives, importing them in the wrong order would be a circular import.
"""

from __future__ import annotations

import datetime as dt

from app.domain.value_objects.identifiers import PREFIXES

#: The evaluation date every fixture defaults to. Fixed, never ``now()``: a fixture that
#: read the clock would make a temporal test pass or fail depending on when it ran.
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

SHA = "a" * 64


def make_id(kind: str, n: int = 1) -> str:
    """A deterministic, well-shaped id. Stable across runs so failures are reproducible."""
    return f"{PREFIXES[kind]}_{n:032x}"
