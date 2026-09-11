"""ScopeSelector parsing and the scope lattice.

Scope is the field whose absence would have been the most damaging defect in the model.
The load-bearing assertion in this module is
:func:`test_comparable_but_unequal_scopes_are_not_a_contradiction`: containment is not
equality, and only equality makes a divergence a contradiction. Conflating them would flood
the highest-trust output with false positives on any realistically scoped corpus.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.services import scope_lattice as lattice
from app.domain.value_objects.scope import ScopeDimension, ScopeSelector

GLOBAL = ScopeSelector.global_scope()
EU = ScopeSelector.parse("region:EU")
EU_ENTERPRISE = ScopeSelector.parse("region:EU / tier:enterprise")
US = ScopeSelector.parse("region:US")
RETAIL = ScopeSelector.parse("tier:retail")


# ---------------------------------------------------------------------------
# Parsing and rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "  ", "global"])
def test_global_spellings(text: str) -> None:
    assert ScopeSelector.parse(text).is_global


def test_parse_normalizes_case() -> None:
    """`region:EU` and `region:eu` are one scope, or coupling silently misses."""
    assert ScopeSelector.parse("region:EU") == ScopeSelector.parse("region:eu")
    assert EU.render() == "region:eu"


def test_parse_round_trips_through_render() -> None:
    assert ScopeSelector.parse(EU_ENTERPRISE.render()) == EU_ENTERPRISE
    assert GLOBAL.render() == "global"


def test_parse_rejects_a_step_without_a_dimension() -> None:
    with pytest.raises(ValueError, match="not 'dimension:value'"):
        ScopeSelector.parse("EU")


def test_a_dimension_may_constrain_a_scope_only_once() -> None:
    """`region:EU / region:US` is not a scope; it is two mutually exclusive claims."""
    with pytest.raises(ValidationError, match="repeats a dimension"):
        ScopeSelector(
            path=(
                ScopeDimension(dimension="region", value="eu"),
                ScopeDimension(dimension="region", value="us"),
            )
        )


def test_scope_tokens_are_shape_validated() -> None:
    with pytest.raises(ValidationError):
        ScopeDimension(dimension="Region", value="eu")
    with pytest.raises(ValidationError):
        ScopeDimension(dimension="region", value="")


def test_scope_is_frozen() -> None:
    with pytest.raises(ValidationError):
        EU.path = ()  # type: ignore[misc]


def test_shape_accessors() -> None:
    assert GLOBAL.depth == 0
    assert EU_ENTERPRISE.depth == 2
    assert EU_ENTERPRISE.dimensions == frozenset({"region", "tier"})
    assert str(EU) == "region:eu"


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------


def test_global_contains_everything_and_itself() -> None:
    assert lattice.contains(GLOBAL, EU)
    assert lattice.contains(GLOBAL, EU_ENTERPRISE)
    assert lattice.contains(GLOBAL, GLOBAL)
    assert not lattice.strictly_contains(GLOBAL, GLOBAL)


def test_containment_is_prefix_containment() -> None:
    assert lattice.strictly_contains(EU, EU_ENTERPRISE)
    assert not lattice.contains(EU_ENTERPRISE, EU)


def test_siblings_are_disjoint() -> None:
    """`region:EU` and `region:US` share no point, so they are never coupled for conflict."""
    assert lattice.is_disjoint(EU, US)
    assert not lattice.comparable(EU, US)
    assert lattice.distance(EU, US) is None


def test_different_dimensions_at_equal_depth_are_disjoint() -> None:
    assert lattice.is_disjoint(EU, RETAIL)


def test_comparable_but_unequal_scopes_are_not_a_contradiction() -> None:
    """The E3 case: retention.window global=90d, region:EU=30d.

    The scopes are comparable, which is what makes an *override* possible, but they are not
    equal, which is what a contradiction requires. A rule that keyed on the assumption key
    alone - or that treated containment as equality - would report this legitimate override
    as a conflict.
    """
    assert lattice.comparable(GLOBAL, EU)
    assert not lattice.equals(GLOBAL, EU)
    assert lattice.strictly_contains(GLOBAL, EU)


def test_invalidation_uses_intersection_while_contradiction_uses_equality() -> None:
    """A global change reaches an inheriting narrower scope; it does not conflict with it."""
    assert lattice.intersects(GLOBAL, EU_ENTERPRISE)
    assert not lattice.equals(GLOBAL, EU_ENTERPRISE)


def test_distance_feeds_coupling_strength() -> None:
    assert lattice.distance(EU, EU) == 0
    assert lattice.distance(GLOBAL, EU) == 1
    assert lattice.distance(GLOBAL, EU_ENTERPRISE) == 2


def test_narrower_and_broader_of() -> None:
    assert lattice.narrower_of(GLOBAL, EU) == EU
    assert lattice.broader_of(GLOBAL, EU) == GLOBAL
    assert lattice.narrower_of(EU, EU) is None, "equal scopes have no narrower member"
    assert lattice.narrower_of(EU, US) is None, "disjoint scopes are not ordered"
