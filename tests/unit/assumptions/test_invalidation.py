"""Delta propagation: the five preconditions, and the three cases easy to get wrong.

Checkpoint 5.5. The distinguishing tests here are the ones that refuse to merge:

* a **legitimate** narrower override reaches ``scope_inheritance_affected``, never
  ``invalidates_assumption`` - it may still be intended;
* an **unauthorized** narrower override produces two independent findings, and the
  pre-existing governance defect is not attributed to the proposed change;
* the declaration the change *edits* is not reported as a consequence of the change.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from app.assumptions.coupling import CouplingIndex, build_coupling_index
from app.assumptions.invalidation import InvalidationReport, propagate
from app.domain.entities.coupling import ResolvedBinding
from app.domain.value_objects.enums import (
    AuthorityClass,
    BindingRole,
    DeltaKind,
    InvalidationOutcome,
)
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from tests.fixtures.coupling import REGISTRY_VERSION, binding, days, delta, score

JAN = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
APR = dt.datetime(2025, 4, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

CONSUMES = BindingRole.CONSUMES
DESCRIBES = BindingRole.DESCRIBES


def index_of(bindings: Sequence[ResolvedBinding], *, as_of: dt.datetime = NOW) -> CouplingIndex:
    return build_coupling_index(bindings, as_of=as_of, registry_version=REGISTRY_VERSION)


def outcome_for(report: InvalidationReport, target: ResolvedBinding) -> InvalidationOutcome:
    impact = next(i for i in report.impacts if i.binding.assumption_id == target.assumption_id)
    return impact.outcome


# ---------------------------------------------------------------------------
# The five preconditions
# ---------------------------------------------------------------------------


def test_a_diverging_consumer_is_invalidated() -> None:
    """Checkpoint E1, the canonical Class A case."""
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    deck = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.TRAINING)
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of([policy, deck]))

    assert outcome_for(report, deck) is InvalidationOutcome.INVALIDATES_ASSUMPTION
    assert len(report.invalidated) >= 1


def test_a_binding_that_already_holds_the_new_value_is_not_affected() -> None:
    already = binding(value=days(180), role=CONSUMES)
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of([already]))
    assert outcome_for(report, already) is InvalidationOutcome.NOT_AFFECTED


def test_a_describes_binding_is_never_invalidated() -> None:
    """Historical narration does not become wrong when the value moves."""
    narration = binding(value=days(90), role=DESCRIBES, authority=AuthorityClass.TICKET)
    report = propagate(delta(new_value=days(180)), index_of([narration]))
    assert report.impacts == ()
    assert report.relations == ()


def test_a_binding_out_of_force_is_not_invalidated() -> None:
    expired = binding(value=days(90), role=CONSUMES, effective_from=JAN, effective_to=APR)
    report = propagate(delta(new_value=days(180)), index_of([expired], as_of=NOW), as_of=NOW)
    assert report.impacts == ()


def test_a_disjoint_scope_is_not_reached() -> None:
    us_binding = binding(scope="region:us", value=days(90), role=CONSUMES)
    change = delta(scope=ScopeSelector.parse("region:eu"), new_value=days(180))
    report = propagate(change, index_of([us_binding]))
    assert report.impacts == ()


def test_a_global_change_reaches_a_narrower_inheriting_scope() -> None:
    """Invalidation needs intersection, and inheritance counts."""
    eu_consumer = binding(scope="region:eu", value=days(90), role=CONSUMES)
    report = propagate(delta(new_value=days(180)), index_of([eu_consumer]))
    assert outcome_for(report, eu_consumer) is InvalidationOutcome.INVALIDATES_ASSUMPTION


def test_a_narrower_change_does_not_reach_a_broader_binding_it_cannot_govern() -> None:
    """Intersection is symmetric, so a narrow change does touch the broad binding.

    Recorded rather than asserted away: the checkpoint says *intersects*, and
    ``global`` intersects ``region:eu``. The broad binding is genuinely in question when a
    region changes, because the reader has to decide whether the global rule still holds.
    """
    global_consumer = binding(scope="global", value=days(90), role=CONSUMES)
    change = delta(
        scope=ScopeSelector.parse("region:eu"),
        old_value=None,
        new_value=days(180),
        delta_kind=DeltaKind.KEY_ADDED,
    )
    report = propagate(change, index_of([global_consumer]))
    assert outcome_for(report, global_consumer) is InvalidationOutcome.INVALIDATES_ASSUMPTION


def test_a_unit_mismatch_blocks_invalidation() -> None:
    """Blocked, not weakened - the same gate as coupling."""
    holder = binding(value=score(0.85), role=CONSUMES)
    report = propagate(
        delta(old_value=None, new_value=days(180), delta_kind=DeltaKind.KEY_ADDED),
        index_of([holder]),
    )
    assert outcome_for(report, holder) is InvalidationOutcome.NOT_AFFECTED
    assert report.relations == ()


def test_a_different_key_is_untouched() -> None:
    other = binding(key="refund.approval_threshold", value=days(90), role=CONSUMES)
    report = propagate(delta(new_value=days(180)), index_of([other]))
    assert report.impacts == ()


# ---------------------------------------------------------------------------
# The declaration the change edits
# ---------------------------------------------------------------------------


def test_the_superseded_declaration_is_not_a_consequence_of_its_own_edit() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of([policy]))

    assert outcome_for(report, policy) is InvalidationOutcome.NOT_AFFECTED
    impact = next(i for i in report.impacts if i.binding.assumption_id == policy.assumption_id)
    assert "the change edits" in impact.reason


def test_a_declaration_at_another_scope_is_not_treated_as_the_edit_target() -> None:
    """Matched on exact scope: an EU policy is not the global declaration being changed."""
    eu_policy = binding(scope="region:eu", value=days(90), authority=AuthorityClass.POLICY)
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of([eu_policy]))
    assert outcome_for(report, eu_policy) is InvalidationOutcome.INVALIDATES_ASSUMPTION


def test_a_declaration_with_a_different_old_value_is_not_the_edit_target() -> None:
    other = binding(value=days(45), authority=AuthorityClass.POLICY)
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of([other]))
    assert outcome_for(report, other) is InvalidationOutcome.INVALIDATES_ASSUMPTION


def test_without_an_old_value_nothing_is_excluded() -> None:
    """A `key_added` delta cannot identify an edit target, so it does not guess one."""
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    report = propagate(
        delta(old_value=None, new_value=days(180), delta_kind=DeltaKind.KEY_ADDED),
        index_of([policy]),
    )
    assert outcome_for(report, policy) is InvalidationOutcome.INVALIDATES_ASSUMPTION


# ---------------------------------------------------------------------------
# Scope inheritance: confirm, do not correct
# ---------------------------------------------------------------------------


def test_a_legitimate_narrower_override_is_reported_for_confirmation() -> None:
    """Checkpoint E3. The override may still be intended."""
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    regulation = binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY)
    report = propagate(
        delta(old_value=days(90), new_value=days(180)), index_of([policy, regulation])
    )

    assert outcome_for(report, regulation) is InvalidationOutcome.SCOPE_INHERITANCE_AFFECTED
    assert len(report.needs_confirmation) == 1


def test_the_legitimate_override_is_never_labelled_contradictory() -> None:
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    regulation = binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY)
    report = propagate(
        delta(old_value=days(90), new_value=days(180)), index_of([policy, regulation])
    )

    # scope_inheritance_affected emits no relation: there is no edge for "please confirm".
    targets = {r.object_ref for r in report.relations}
    assert regulation.source_version_id not in targets


def test_an_unauthorized_narrower_override_is_invalidated_not_confirmed() -> None:
    """It has no legitimate standing to differ, so it is not a "may still be intended"."""
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    wiki = binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI)
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of([policy, wiki]))

    assert outcome_for(report, wiki) is InvalidationOutcome.INVALIDATES_ASSUMPTION


def test_the_pre_existing_override_is_reported_separately_and_not_merged() -> None:
    """Checkpoint 5.5: the change did not cause the governance defect."""
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    wiki = binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI)
    index = index_of([policy, wiki])
    report = propagate(delta(old_value=days(90), new_value=days(180)), index)

    assert len(report.pre_existing_unauthorized_overrides) == 1
    pre_existing = report.pre_existing_unauthorized_overrides[0]
    assert pre_existing.relationship is Relationship.UNAUTHORIZED_OVERRIDE
    # And it is a different relation from anything the delta produced.
    assert pre_existing.relation_id not in {r.relation_id for r in report.relations}


def test_a_change_at_the_overrides_own_scope_is_an_ordinary_invalidation() -> None:
    """Editing the override itself is not an inheritance question."""
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    regulation = binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY)
    change = delta(scope=ScopeSelector.parse("region:eu"), old_value=days(30), new_value=days(60))
    report = propagate(change, index_of([policy, regulation]))

    assert outcome_for(report, regulation) is InvalidationOutcome.NOT_AFFECTED


# ---------------------------------------------------------------------------
# key_removed
# ---------------------------------------------------------------------------


def test_removing_a_key_invalidates_every_dependent_binding() -> None:
    consumer = binding(value=days(90), role=CONSUMES)
    change = delta(old_value=days(90), new_value=None, delta_kind=DeltaKind.KEY_REMOVED)
    report = propagate(change, index_of([consumer]))

    assert outcome_for(report, consumer) is InvalidationOutcome.INVALIDATES_ASSUMPTION
    impact = report.invalidated[0]
    assert "removed" in impact.reason


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_an_invalidation_relation_records_the_delta_and_both_values() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    deck = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.TRAINING)
    change = delta(old_value=days(90), new_value=days(180))
    report = propagate(change, index_of([policy, deck]))

    relation = next(r for r in report.relations if r.object_ref == deck.source_version_id)
    detail = relation.provenance.detail
    assert detail["delta_id"] == change.delta_id
    assert detail["old_value"] == "90d"
    assert detail["new_value"] == "180d"
    assert detail["bound_value"] == "90d"
    assert detail["affected_role"] == "consumes"
    assert relation.provenance.rule_id == "delta_invalidation"


def test_the_invalidation_relation_carries_no_direct_evidence() -> None:
    deck = binding(value=days(90), role=CONSUMES)
    report = propagate(delta(new_value=days(180)), index_of([deck]))
    for relation in report.relations:
        assert relation.evidence_ids == ()
        assert relation.input_record_ids


def test_the_subject_of_an_invalidation_is_the_change() -> None:
    deck = binding(value=days(90), role=CONSUMES)
    change = delta(new_value=days(180))
    report = propagate(change, index_of([deck]))
    assert report.relations[0].subject_ref == change.delta_id


def test_the_superseded_declaration_is_named_as_an_input_when_it_exists() -> None:
    """The derivation is over (change, affected binding, superseded declaration)."""
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    deck = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.TRAINING)
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of([policy, deck]))

    relation = next(r for r in report.relations if r.object_ref == deck.source_version_id)
    assert set(relation.input_record_ids) == {deck.assumption_id, policy.assumption_id}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _scenario() -> list[ResolvedBinding]:
    return [
        binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
        binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY),
        binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI),
        binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG),
        binding(value=days(90), role=DESCRIBES, authority=AuthorityClass.TICKET),
    ]


def test_propagation_is_independent_of_input_order() -> None:
    change = delta(old_value=days(90), new_value=days(180))
    forward = propagate(change, index_of(_scenario()))
    backward = propagate(change, index_of(list(reversed(_scenario()))))

    assert [(i.binding.assumption_id, i.outcome) for i in forward.impacts] == [
        (i.binding.assumption_id, i.outcome) for i in backward.impacts
    ]
    assert [r.relation_id for r in forward.relations] == [r.relation_id for r in backward.relations]


def test_every_impact_lands_in_exactly_one_bucket() -> None:
    report = propagate(delta(old_value=days(90), new_value=days(180)), index_of(_scenario()))
    total = sum(
        len(report.of_outcome(outcome))
        for outcome in (
            InvalidationOutcome.INVALIDATES_ASSUMPTION,
            InvalidationOutcome.SCOPE_INHERITANCE_AFFECTED,
            InvalidationOutcome.NOT_AFFECTED,
        )
    )
    assert total == len(report.impacts)
