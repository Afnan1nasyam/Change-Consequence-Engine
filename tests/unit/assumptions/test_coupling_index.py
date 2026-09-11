"""Coupling set membership, index keying, outcome summarization, reproducibility.

The keying test is the one that protects the product. Keying on ``key`` alone - the removed
``has_collision := distinct_values > 1`` rule - reports every legitimately scoped override
as a contradiction. On a corpus with any regional variation at all, that floods the
highest-trust output with false positives, and
:func:`test_keying_on_key_alone_would_report_a_false_contradiction` states the difference
numerically rather than trusting the comment.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from app.assumptions.collisions import collide_key
from app.assumptions.coupling import (
    CouplingIndex,
    build_coupling_index,
    build_coupling_set,
    members_of,
)
from app.domain.entities.coupling import ResolvedBinding
from app.domain.value_objects.enums import AuthorityClass, BindingRole, CouplingOutcome
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from tests.fixtures.coupling import REGISTRY_VERSION, binding, days, score, usd

GLOBAL = ScopeSelector.global_scope()
EU = ScopeSelector.parse("region:eu")
US = ScopeSelector.parse("region:us")
EU_ENT = ScopeSelector.parse("region:eu / tier:enterprise")

JAN = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
APR = dt.datetime(2025, 4, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

KEY = "retention.window"


def build(bindings: Sequence[ResolvedBinding], *, as_of: dt.datetime = NOW) -> CouplingIndex:
    return build_coupling_index(bindings, as_of=as_of, registry_version=REGISTRY_VERSION)


# ---------------------------------------------------------------------------
# Membership: (key, scope, as_of)
# ---------------------------------------------------------------------------


def test_membership_includes_the_same_scope() -> None:
    left = binding(scope="global", value=days(90), tag="a")
    right = binding(scope="global", value=days(45), tag="b")
    found = members_of([left, right], key=KEY, scope=GLOBAL, as_of=NOW)
    assert {b.assumption_id for b in found} == {left.assumption_id, right.assumption_id}


def test_membership_includes_a_broader_scope_that_governs() -> None:
    """Inheritance: the global declaration belongs to the EU set."""
    global_binding = binding(scope="global", value=days(90))
    eu_binding = binding(scope="region:eu", value=days(30))
    found = members_of([global_binding, eu_binding], key=KEY, scope=EU, as_of=NOW)
    assert len(found) == 2


def test_membership_includes_a_narrower_scope() -> None:
    """Comparable in either direction: the global cell can see the EU exception."""
    global_binding = binding(scope="global", value=days(90))
    eu_binding = binding(scope="region:eu", value=days(30))
    found = members_of([global_binding, eu_binding], key=KEY, scope=GLOBAL, as_of=NOW)
    assert len(found) == 2


def test_membership_excludes_a_sibling_scope() -> None:
    eu_binding = binding(scope="region:eu", value=days(30))
    us_binding = binding(scope="region:us", value=days(90))
    found = members_of([eu_binding, us_binding], key=KEY, scope=EU, as_of=NOW)
    assert [b.assumption_id for b in found] == [eu_binding.assumption_id]


def test_membership_excludes_a_binding_out_of_force() -> None:
    stale = binding(value=days(30), effective_from=JAN, effective_to=APR, tag="old")
    live = binding(value=days(90), effective_from=APR, tag="new")
    found = members_of([stale, live], key=KEY, scope=GLOBAL, as_of=NOW)
    assert [b.assumption_id for b in found] == [live.assumption_id]


def test_membership_excludes_another_key() -> None:
    right = binding(key=KEY, value=days(90))
    wrong = binding(key="refund.approval_threshold", value=usd(10000))
    found = members_of([right, wrong], key=KEY, scope=GLOBAL, as_of=NOW)
    assert [b.assumption_id for b in found] == [right.assumption_id]


def test_membership_is_returned_in_a_stable_order() -> None:
    pool = [binding(value=days(v), tag=str(v)) for v in (90, 45, 30, 3650)]
    forward = members_of(pool, key=KEY, scope=GLOBAL, as_of=NOW)
    backward = members_of(list(reversed(pool)), key=KEY, scope=GLOBAL, as_of=NOW)
    assert [b.assumption_id for b in forward] == [b.assumption_id for b in backward]


# ---------------------------------------------------------------------------
# The keying decision
# ---------------------------------------------------------------------------


def test_keying_on_key_alone_would_report_a_false_contradiction() -> None:
    """The defect the triple key exists to prevent, stated numerically.

    A legitimate regional override: four divergent values across two scopes, and zero
    contradictions. A key-only rule counting distinct values would report a collision.
    """
    pool = [
        binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
        binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY),
        binding(scope="region:us", value=days(180), authority=AuthorityClass.REGULATORY),
        binding(
            scope="region:eu / tier:enterprise", value=days(15), authority=AuthorityClass.REGULATORY
        ),
    ]
    index = build(pool)

    distinct_values = len({b.value.render() for b in pool})
    assert distinct_values == 4
    assert index.relations == ()
    assert len(index.overrides) >= 1


def test_the_index_has_one_cell_per_observed_scope() -> None:
    pool = [
        binding(scope="global", value=days(90)),
        binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY),
    ]
    index = build(pool)
    scopes = {cell.scope.render() for cell in index.sets if cell.key == KEY}
    assert scopes == {"global", "region:eu"}


def test_no_cell_is_created_for_a_scope_nothing_binds() -> None:
    """The index projects what the corpus says, not what it could have said."""
    index = build([binding(scope="global", value=days(90))])
    assert {cell.scope.render() for cell in index.sets} == {"global"}


def test_a_cell_can_be_looked_up_by_key_and_scope() -> None:
    index = build([binding(scope="region:eu", value=days(30))])
    assert index.set_for(KEY, EU) is not None
    assert index.set_for(KEY, US) is None


# ---------------------------------------------------------------------------
# Outcome summarization
# ---------------------------------------------------------------------------


def test_a_single_holder_cell_is_consistent() -> None:
    index = build([binding(value=days(90))])
    cell = index.set_for(KEY, GLOBAL)
    assert cell is not None
    assert cell.outcome is CouplingOutcome.CONSISTENT
    assert cell.is_single_holder


def test_a_same_scope_divergence_summarizes_as_conflicts_with() -> None:
    index = build([binding(value=days(90), tag="a"), binding(value=days(45), tag="b")])
    cell = index.set_for(KEY, GLOBAL)
    assert cell is not None
    assert cell.outcome is CouplingOutcome.CONFLICTS_WITH


def test_a_legitimate_override_summarizes_as_legitimate_override() -> None:
    """Visible in the cell, but emitting no finding."""
    index = build(
        [
            binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
            binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY),
        ]
    )
    cell = index.set_for(KEY, EU)
    assert cell is not None
    assert cell.outcome is CouplingOutcome.LEGITIMATE_OVERRIDE
    assert index.relations == ()


def test_an_unauthorized_override_summarizes_as_unauthorized_override() -> None:
    index = build(
        [
            binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
            binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI),
        ]
    )
    cell = index.set_for(KEY, EU)
    assert cell is not None
    assert cell.outcome is CouplingOutcome.UNAUTHORIZED_OVERRIDE


def test_ambiguous_authority_summarizes_as_ambiguous_temporal() -> None:
    """Two peers disagreeing at one scope also conflict, and conflict outranks ambiguity."""
    index = build(
        [
            binding(value=days(90), authority=AuthorityClass.POLICY, tag="a"),
            binding(value=days(45), authority=AuthorityClass.POLICY, tag="b"),
        ]
    )
    cell = index.set_for(KEY, GLOBAL)
    assert cell is not None
    assert cell.outcome is CouplingOutcome.CONFLICTS_WITH
    assert len(index.ambiguous_resolutions) == 1


def test_a_conflict_outside_a_cell_does_not_taint_it() -> None:
    """One defect must appear in one place, not in every cell that can see it."""
    pool = [
        binding(key="retention.window", scope="region:us", value=days(90), tag="a"),
        binding(key="retention.window", scope="region:us", value=days(45), tag="b"),
        binding(
            key="retention.window",
            scope="region:eu",
            value=days(30),
            authority=AuthorityClass.REGULATORY,
        ),
    ]
    index = build(pool)
    eu_cell = index.set_for(KEY, EU)
    assert eu_cell is not None
    assert eu_cell.outcome is CouplingOutcome.CONSISTENT


def test_a_cell_records_the_versions_that_produced_it() -> None:
    index = build([binding(value=days(90))])
    cell = index.set_for(KEY, GLOBAL)
    assert cell is not None
    assert cell.coupling_index_version == "coupling-1"
    assert cell.authority_model_version == "authority-1"
    assert cell.computed_at == NOW


def test_holder_roles_are_preserved_on_the_cell() -> None:
    pool = [
        binding(value=days(90), role=BindingRole.DECLARES),
        binding(value=days(90), role=BindingRole.CONSUMES),
        binding(value=days(90), role=BindingRole.DESCRIBES),
    ]
    cell = build(pool).set_for(KEY, GLOBAL)
    assert cell is not None
    assert cell.holder_count == 3
    assert cell.declarer_count == 1
    assert len(cell.consumers) == 1


def test_build_coupling_set_can_be_called_directly() -> None:
    pool = [binding(value=days(90), tag="a"), binding(value=days(45), tag="b")]
    collisions = collide_key(pool, key=KEY, as_of=NOW, registry_version=REGISTRY_VERSION)
    cell = build_coupling_set(pool, key=KEY, scope=GLOBAL, as_of=NOW, collisions=collisions)
    assert cell.outcome is CouplingOutcome.CONFLICTS_WITH


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_relations_are_deduplicated_across_overlapping_cells() -> None:
    """The global and EU cells both contain the same pair; the finding is emitted once."""
    index = build(
        [
            binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
            binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI),
        ]
    )
    overrides = index.relations_of(Relationship.UNAUTHORIZED_OVERRIDE)
    assert len(overrides) == 1
    assert len({r.relation_id for r in index.relations}) == len(index.relations)


def test_the_index_reports_every_key_it_examined() -> None:
    index = build(
        [
            binding(key="retention.window", value=days(90)),
            binding(key="refund.approval_threshold", value=usd(10000)),
        ]
    )
    assert index.keys == ("refund.approval_threshold", "retention.window")


def test_anomalies_are_aggregated() -> None:
    index = build(
        [
            binding(value=usd(10000), tag="a"),
            binding(value=score(0.85), tag="b"),
        ]
    )
    assert len(index.anomalies) == 1


def test_authority_is_queryable_per_cell() -> None:
    index = build(
        [
            binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
            binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY),
        ]
    )
    assert index.authority_for(KEY, GLOBAL) is not None
    assert index.authority_for(KEY, EU) is not None
    assert index.authority_for("refund.approval_threshold", GLOBAL) is None


def test_an_empty_store_produces_an_empty_index() -> None:
    """Quiet on a quiet corpus."""
    index = build([])
    assert index.sets == ()
    assert index.relations == ()
    assert index.keys == ()
    assert index.binding_count == 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _scenario() -> list[ResolvedBinding]:
    return [
        binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
        binding(scope="global", value=days(3650), authority=AuthorityClass.WIKI),
        binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY),
        binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI),
        binding(value=days(365), role=BindingRole.CONSUMES, authority=AuthorityClass.CONFIG),
        binding(key="refund.approval_threshold", value=usd(10000), authority=AuthorityClass.POLICY),
    ]


def test_two_builds_over_the_same_inputs_are_identical() -> None:
    """Given the same store state and versions, the result must be identical."""
    first = build(_scenario())
    second = build(_scenario())

    assert [c.model_dump() for c in first.sets] == [c.model_dump() for c in second.sets]
    assert [r.relation_id for r in first.relations] == [r.relation_id for r in second.relations]


def test_the_build_is_independent_of_input_order() -> None:
    forward = build(_scenario())
    backward = build(list(reversed(_scenario())))

    assert [c.model_dump() for c in forward.sets] == [c.model_dump() for c in backward.sets]
    assert [r.relation_id for r in forward.relations] == [r.relation_id for r in backward.relations]


def test_relation_ids_are_derived_not_random() -> None:
    """A rebuilt index must cite the same relation a past analysis cited."""
    first = build(_scenario())
    second = build(_scenario())
    assert {r.relation_id for r in first.relations} == {r.relation_id for r in second.relations}
    assert first.relations


def test_a_different_as_of_can_legitimately_change_the_result() -> None:
    """Reproducibility is per (inputs, versions, date) - not date-independence."""
    stale = binding(value=days(30), effective_from=JAN, effective_to=APR, tag="old")
    live = binding(value=days(90), effective_from=APR, tag="new")
    pool = [stale, live]

    early = build(pool, as_of=dt.datetime(2025, 2, 1, tzinfo=dt.UTC))
    late = build(pool, as_of=NOW)

    early_cell = early.set_for(KEY, GLOBAL)
    late_cell = late.set_for(KEY, GLOBAL)
    assert early_cell is not None and late_cell is not None
    assert early_cell.holders[0].value == days(30)
    assert late_cell.holders[0].value == days(90)


def test_the_index_pins_every_version_label() -> None:
    index = build(_scenario())
    assert index.registry_version == REGISTRY_VERSION
    assert index.coupling_index_version == "coupling-1"
    assert index.authority_model_version == "authority-1"
