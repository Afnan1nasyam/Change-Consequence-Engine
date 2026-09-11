"""Scope lattice operations.

Pure functions over :class:`ScopeSelector`, no store access. Kept out of the value object
so that model validators cannot accidentally depend on lattice semantics, and so the
authority-aware decisions built on top of these (checkpoint 5.4) stay in one place.

Containment is prefix containment: ``region:EU / tier:enterprise`` is *inside*
``region:EU``, and ``global`` contains everything.
"""

from __future__ import annotations

from app.domain.value_objects.scope import ScopeSelector


def equals(a: ScopeSelector, b: ScopeSelector) -> bool:
    return a.path == b.path


def contains(broader: ScopeSelector, narrower: ScopeSelector) -> bool:
    """True when ``broader`` contains ``narrower``, inclusively.

    ``global`` contains everything, and every scope contains itself.
    """
    if len(broader.path) > len(narrower.path):
        return False
    return narrower.path[: len(broader.path)] == broader.path


def strictly_contains(broader: ScopeSelector, narrower: ScopeSelector) -> bool:
    """Containment excluding equality - the shape that makes an *override* possible."""
    return contains(broader, narrower) and not equals(broader, narrower)


def comparable(a: ScopeSelector, b: ScopeSelector) -> bool:
    """True when the two scopes lie on the same branch, in either direction.

    Required for a *contradiction*: two declarations at unrelated scopes are not in
    conflict, they are simply about different things.
    """
    return contains(a, b) or contains(b, a)


def intersects(a: ScopeSelector, b: ScopeSelector) -> bool:
    """True when the two scopes share at least one point.

    On a strict tree lattice this coincides with :func:`comparable`; it is named
    separately because *invalidation* needs intersection while *contradiction* needs
    comparability, and conflating the two rules was a defect in an earlier draft
    (checkpoint 5.4/5.5).
    """
    return comparable(a, b)


def is_disjoint(a: ScopeSelector, b: ScopeSelector) -> bool:
    return not intersects(a, b)


def distance(a: ScopeSelector, b: ScopeSelector) -> int | None:
    """Depth difference along the branch, or ``None`` when the scopes are disjoint.

    Feeds the coupling-strength function: a nearer scope is a stronger coupling.
    """
    if not comparable(a, b):
        return None
    return abs(a.depth - b.depth)


def narrower_of(a: ScopeSelector, b: ScopeSelector) -> ScopeSelector | None:
    """The strictly narrower of two comparable scopes, or ``None`` if neither is."""
    if strictly_contains(a, b):
        return b
    if strictly_contains(b, a):
        return a
    return None


def broader_of(a: ScopeSelector, b: ScopeSelector) -> ScopeSelector | None:
    """The strictly broader of two comparable scopes, or ``None`` if neither is."""
    if strictly_contains(a, b):
        return a
    if strictly_contains(b, a):
        return b
    return None
