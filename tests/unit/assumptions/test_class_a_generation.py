"""Class A generation by producer, provenance materialization, and the Class B gate.

Two things are asserted here that no amount of prompt wording could hold:

* a Class A finding **cannot be constructed** as anything but resolver-produced, because
  ``consequence_class`` and ``origin`` are ``Literal`` fields;
* a derived finding's evidence is the **computed union** of its inputs, explicitly flagged
  as such, while the underlying ``DerivedRelation`` still carries none of its own. A direct
  citation on a derivation would be fabricated provenance.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from app.assumptions.class_a import derive_class_a, materialize_evidence, merge_class_b
from app.assumptions.coupling import CouplingIndex, build_coupling_index
from app.assumptions.invalidation import propagate
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.coupling import ResolvedBinding
from app.domain.value_objects.enums import (
    AuthorityClass,
    BindingRole,
    ConsequenceClass,
    Origin,
)
from app.domain.value_objects.rules import RuleId
from app.domain.value_objects.taxonomy import Relationship
from tests.fixtures.coupling import (
    REGISTRY_VERSION,
    TENANT,
    binding,
    class_a_finding,
    days,
    delta,
)
from tests.fixtures.domain import consequence_hypothesis, make_id

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
CONSUMES = BindingRole.CONSUMES


def resolve(
    bindings: Sequence[ResolvedBinding],
) -> tuple[CouplingIndex, tuple[ClassAFinding, ...]]:
    index = build_coupling_index(bindings, as_of=NOW, registry_version=REGISTRY_VERSION)
    findings = derive_class_a(
        index.relations,
        bindings=bindings,
        tenant_id=TENANT,
        registry_version=REGISTRY_VERSION,
    )
    return index, findings


# ---------------------------------------------------------------------------
# Classification is by producer
# ---------------------------------------------------------------------------


def test_every_derived_finding_is_class_a_from_the_resolver() -> None:
    _index, findings = resolve([binding(value=days(90), tag="a"), binding(value=days(45), tag="b")])
    assert findings
    for finding in findings:
        assert finding.consequence_class is ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED
        assert finding.origin is Origin.RESOLVER


def test_a_class_a_finding_cannot_be_relabelled_class_b() -> None:
    sample = class_a_finding()
    with pytest.raises(ValidationError):
        ClassAFinding(**{**sample.model_dump(), "consequence_class": "class_b_behavioral"})


def test_a_class_a_finding_cannot_claim_an_agent_origin() -> None:
    sample = class_a_finding()
    with pytest.raises(ValidationError):
        ClassAFinding(**{**sample.model_dump(), "origin": "agent"})


def test_a_class_a_finding_rejects_a_confidence_or_severity_key() -> None:
    """Neither is the resolver's to supply, and `extra="forbid"` makes that structural."""
    sample = class_a_finding()
    for field, value in (("confidence", 0.9), ("severity", "HIGH")):
        with pytest.raises(ValidationError):
            ClassAFinding(**{**sample.model_dump(), field: value})


def test_a_class_a_finding_rejects_a_non_derivable_relationship() -> None:
    sample = class_a_finding()
    with pytest.raises(ValidationError):
        ClassAFinding(**{**sample.model_dump(), "relationship": "creates_risk_for"})


def test_a_single_input_derivation_is_rejected_without_a_delta() -> None:
    """A restatement of one binding is not a derivation."""
    sample = class_a_finding()
    with pytest.raises(ValidationError):
        ClassAFinding(
            **{
                **sample.model_dump(),
                "input_record_ids": (make_id("assumption", 1),),
                "delta_id": None,
            }
        )


def test_a_single_input_derivation_is_allowed_with_a_delta() -> None:
    """The change is the second input; it just is not an AssumptionRecord."""
    sample = class_a_finding()
    finding = ClassAFinding(
        **{
            **sample.model_dump(),
            "relationship": "invalidates_assumption",
            "rule_id": RuleId.DELTA_INVALIDATION,
            "input_record_ids": (make_id("assumption", 1),),
            "delta_id": make_id("delta"),
        }
    )
    assert finding.delta_id is not None


# ---------------------------------------------------------------------------
# Shaping per relationship
# ---------------------------------------------------------------------------


def test_a_symmetric_conflict_yields_one_finding_per_declarer() -> None:
    """Both authors have something to reconcile, so both must be able to see it."""
    left = binding(value=days(90), authority=AuthorityClass.POLICY)
    right = binding(value=days(45), authority=AuthorityClass.STANDARD)
    _index, findings = resolve([left, right])

    conflicts = [f for f in findings if f.relationship is Relationship.CONFLICTS_WITH]
    assert len(conflicts) == 2
    assert {f.affected_source_version_id for f in conflicts} == {
        left.source_version_id,
        right.source_version_id,
    }
    # One relation, two findings.
    assert len({f.relation_id for f in conflicts}) == 1


def test_an_unauthorized_override_names_the_narrower_artifact_as_affected() -> None:
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    wiki = binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI)
    _index, findings = resolve([policy, wiki])

    override = next(f for f in findings if f.relationship is Relationship.UNAUTHORIZED_OVERRIDE)
    assert override.affected_source_version_id == wiki.source_version_id
    assert override.counterpart_source_version_id == policy.source_version_id
    assert override.scope.render() == "region:eu"


def test_a_stale_consumer_finding_names_the_consumer_as_affected() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    dashboard = binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG)
    _index, findings = resolve([policy, dashboard])

    stale = next(f for f in findings if f.rule_id == RuleId.STALE_CONSUMER)
    assert stale.affected_source_version_id == dashboard.source_version_id
    assert stale.observed_value == days(365)
    assert stale.authoritative_value == days(90)


def test_the_affected_side_is_resolved_from_the_relations_own_inputs() -> None:
    """One document version holds several bindings, so a version-keyed lookup is wrong.

    Here one source holds both an EU retention override and a global consumer. Attaching
    the wrong one would report the finding at the wrong scope with the wrong value.
    """
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    wiki_eu = binding(
        scope="region:eu", value=days(45), authority=AuthorityClass.WIKI, title="EMEA notes"
    )
    # A second binding from a different key, same authority/title shape.
    other_key = binding(
        key="refund.approval_threshold",
        value=days(30),
        authority=AuthorityClass.WIKI,
        title="EMEA notes",
        tag="other",
    )
    _index, findings = resolve([policy, wiki_eu, other_key])

    override = next(f for f in findings if f.relationship is Relationship.UNAUTHORIZED_OVERRIDE)
    assert override.assumption_key == "retention.window"
    assert override.scope.render() == "region:eu"
    assert override.observed_value == days(45)


def test_a_finding_summary_names_both_artifacts_and_both_values() -> None:
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    wiki = binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI)
    _index, findings = resolve([policy, wiki])

    override = next(f for f in findings if f.relationship is Relationship.UNAUTHORIZED_OVERRIDE)
    assert "45d" in override.summary
    assert "90d" in override.summary
    assert "wiki" in override.summary
    assert "policy" in override.summary


# ---------------------------------------------------------------------------
# Evidence materialization
# ---------------------------------------------------------------------------


def test_evidence_is_the_computed_union_of_the_inputs() -> None:
    left = binding(value=days(90), tag="a")
    right = binding(value=days(45), tag="b")
    _index, findings = resolve([left, right])

    expected = {left.evidence_id, right.evidence_id}
    for finding in findings:
        assert set(finding.materialized_evidence_ids) == expected


def test_the_materialized_evidence_is_flagged_as_derived() -> None:
    """It must never be readable as direct support for the derivation."""
    _index, findings = resolve([binding(value=days(90), tag="a"), binding(value=days(45), tag="b")])
    for finding in findings:
        assert finding.evidence_is_derived_union is True


def test_the_underlying_relation_still_carries_no_evidence() -> None:
    index, _findings = resolve([binding(value=days(90), tag="a"), binding(value=days(45), tag="b")])
    for relation in index.relations:
        assert relation.evidence_ids == ()


def test_materialize_evidence_is_stable_and_deduplicated() -> None:
    left = binding(value=days(90), tag="a")
    right = binding(value=days(45), tag="b")
    index, _findings = resolve([left, right])
    by_assumption = {b.assumption_id: b for b in (left, right)}

    first = materialize_evidence(index.relations[0], by_assumption)
    second = materialize_evidence(index.relations[0], by_assumption)
    assert first == second == tuple(sorted({left.evidence_id, right.evidence_id}))


def test_a_finding_is_suppressed_when_no_input_evidence_can_be_resolved() -> None:
    """A derivation nobody can check must not be asserted."""
    left = binding(value=days(90), tag="a")
    right = binding(value=days(45), tag="b")
    index = build_coupling_index([left, right], as_of=NOW, registry_version=REGISTRY_VERSION)

    findings = derive_class_a(
        index.relations, bindings=[], tenant_id=TENANT, registry_version=REGISTRY_VERSION
    )
    assert findings == ()


def test_every_finding_pins_its_rule_and_version_labels() -> None:
    _index, findings = resolve([binding(value=days(90), tag="a"), binding(value=days(45), tag="b")])
    for finding in findings:
        assert finding.rule_id
        assert finding.rule_version == "coupling-1"
        assert finding.registry_version == REGISTRY_VERSION
        assert finding.coupling_index_version == "coupling-1"
        assert finding.authority_model_version == "authority-1"
        assert finding.input_record_ids


def test_a_delta_invalidation_finding_records_the_delta() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    deck = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.TRAINING)
    index = build_coupling_index([policy, deck], as_of=NOW, registry_version=REGISTRY_VERSION)
    change = delta(old_value=days(90), new_value=days(180))
    report = propagate(change, index)

    findings = derive_class_a(
        report.relations,
        bindings=[policy, deck],
        tenant_id=TENANT,
        registry_version=REGISTRY_VERSION,
    )
    assert findings
    assert all(f.delta_id == change.delta_id for f in findings)
    assert all(f.rule_id == RuleId.DELTA_INVALIDATION for f in findings)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_finding_ids_are_derived_not_random() -> None:
    pool = [binding(value=days(90), tag="a"), binding(value=days(45), tag="b")]
    _i1, first = resolve(pool)
    _i2, second = resolve(pool)
    assert [f.finding_id for f in first] == [f.finding_id for f in second]


def test_generation_is_independent_of_input_order() -> None:
    pool = [
        binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
        binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI),
        binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG),
    ]
    _i1, forward = resolve(pool)
    _i2, backward = resolve(list(reversed(pool)))
    assert [f.finding_id for f in forward] == [f.finding_id for f in backward]


# ---------------------------------------------------------------------------
# Class B merge and the rediscovery metric
# ---------------------------------------------------------------------------


def test_a_hypothesis_on_a_resolved_pair_is_suppressed_and_merged() -> None:
    finding = class_a_finding()
    hypothesis = consequence_hypothesis(
        assumption_keys=(finding.assumption_key,),
    )
    outcome, updated = merge_class_b(
        [hypothesis],
        [finding],
        affected_artifact_of={hypothesis.hypothesis_id: finding.affected_source_version_id},
    )

    assert outcome.suppressed_hypothesis_ids == (hypothesis.hypothesis_id,)
    assert outcome.retained_hypothesis_ids == ()
    assert outcome.merged_into[hypothesis.hypothesis_id] == finding.finding_id
    assert updated[0].merged_hypothesis_ids == (hypothesis.hypothesis_id,)


def test_a_merged_hypothesis_is_kept_not_dropped() -> None:
    """The articulated mechanism is worth keeping even when the discovery was not the model's."""
    finding = class_a_finding()
    hypothesis = consequence_hypothesis(assumption_keys=(finding.assumption_key,))
    _outcome, updated = merge_class_b(
        [hypothesis],
        [finding],
        affected_artifact_of={hypothesis.hypothesis_id: finding.affected_source_version_id},
    )
    assert hypothesis.hypothesis_id in updated[0].merged_hypothesis_ids


def test_a_hypothesis_on_a_different_artifact_is_retained() -> None:
    finding = class_a_finding()
    hypothesis = consequence_hypothesis(assumption_keys=(finding.assumption_key,))
    outcome, updated = merge_class_b(
        [hypothesis],
        [finding],
        affected_artifact_of={hypothesis.hypothesis_id: make_id("source_version", 99)},
    )
    assert outcome.retained_hypothesis_ids == (hypothesis.hypothesis_id,)
    assert updated[0].merged_hypothesis_ids == ()


def test_a_hypothesis_on_a_different_key_is_retained() -> None:
    finding = class_a_finding()
    hypothesis = consequence_hypothesis(assumption_keys=("refund.approval_threshold",))
    outcome, _updated = merge_class_b(
        [hypothesis],
        [finding],
        affected_artifact_of={hypothesis.hypothesis_id: finding.affected_source_version_id},
    )
    assert outcome.retained_hypothesis_ids == (hypothesis.hypothesis_id,)


def test_a_hypothesis_with_no_artifact_mapping_is_retained() -> None:
    """Ambiguity demotes to Class B; it does not silently suppress."""
    finding = class_a_finding()
    hypothesis = consequence_hypothesis(assumption_keys=(finding.assumption_key,))
    outcome, _updated = merge_class_b([hypothesis], [finding])
    assert outcome.retained_hypothesis_ids == (hypothesis.hypothesis_id,)


def test_the_rediscovery_rate_is_the_suppressed_share() -> None:
    finding = class_a_finding()
    duplicate = consequence_hypothesis(assumption_keys=(finding.assumption_key,))
    novel = consequence_hypothesis(assumption_keys=("refund.approval_threshold",))

    outcome, _updated = merge_class_b(
        [duplicate, novel],
        [finding],
        affected_artifact_of={
            duplicate.hypothesis_id: finding.affected_source_version_id,
            novel.hypothesis_id: finding.affected_source_version_id,
        },
    )
    assert outcome.considered == 2
    assert outcome.class_b_rediscovery_rate == pytest.approx(0.5)


def test_no_hypotheses_gives_a_zero_rate_and_a_zero_count() -> None:
    outcome, updated = merge_class_b([], [class_a_finding()])
    assert outcome.considered == 0
    assert outcome.class_b_rediscovery_rate == 0.0
    assert updated[0].merged_hypothesis_ids == ()


def test_the_merge_accounting_must_balance() -> None:
    """A hypothesis that is neither suppressed nor retained has been lost."""
    from app.domain.entities.class_a import ClassBMergeOutcome

    with pytest.raises(ValidationError):
        ClassBMergeOutcome(
            suppressed_hypothesis_ids=(make_id("hypothesis", 1),),
            retained_hypothesis_ids=(),
            considered=5,
        )


def test_a_class_b_hypothesis_cannot_become_a_class_a_finding() -> None:
    """The merge re-attributes to the resolver; it never promotes the hypothesis."""
    finding = class_a_finding()
    hypothesis = consequence_hypothesis(assumption_keys=(finding.assumption_key,))
    _outcome, updated = merge_class_b(
        [hypothesis],
        [finding],
        affected_artifact_of={hypothesis.hypothesis_id: finding.affected_source_version_id},
    )
    merged = updated[0]
    assert merged.origin is Origin.RESOLVER
    assert merged.relation_id == finding.relation_id
    assert "consequence_class" not in type(hypothesis).model_fields
