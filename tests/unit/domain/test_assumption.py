"""AssumptionRecord, CouplingSet, AssumptionDelta - the core mechanism unit.

Two families of assertion here matter more than the rest:

* **A binding cannot exist without a citation.** ``evidence_id`` is required and shape
  validated, so an unsourced binding is unrepresentable rather than merely discouraged.
* **The retired vocabulary is unrepresentable.** ``binding_kind`` and
  ``extraction_confidence`` are not simply absent - ``extra="forbid"`` makes an extractor
  that still emits them fail loudly instead of having the field silently dropped.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.domain.entities.assumption import AssumptionDelta, AssumptionRecord
from app.domain.value_objects.enums import (
    EXPLICITNESS_RANK,
    AuthorityClass,
    BindingRole,
    ClaimType,
    CouplingOutcome,
    DeltaKind,
    Derivation,
    Explicitness,
    Support,
)
from app.domain.value_objects.scope import ScopeSelector
from tests.fixtures.domain import (
    AS_OF,
    assumption_delta,
    assumption_record,
    coupling_holder,
    coupling_set,
    days,
    make_id,
    usd,
)

# ---------------------------------------------------------------------------
# The 7-tuple
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "assumption_id",
        "tenant_id",
        "key",
        "registry_version",
        "value",
        "role",
        "explicitness",
        "holder_source_version_id",
        "evidence_id",
        "extractor_version",
    ],
)
def test_every_element_of_the_binding_tuple_is_required(field: str) -> None:
    payload = assumption_record().model_dump()
    del payload[field]
    with pytest.raises(ValidationError) as err:
        AssumptionRecord(**payload)
    assert field in str(err.value)


def test_a_binding_without_an_evidence_id_is_invalid() -> None:
    """The single most important negative case in the module."""
    with pytest.raises(ValidationError):
        assumption_record(evidence_id=None)
    with pytest.raises(ValidationError):
        assumption_record(evidence_id="")
    with pytest.raises(ValidationError):
        assumption_record(evidence_id="ev-1")


def test_scope_defaults_to_global_rather_than_being_absent() -> None:
    """An absent scope would have to be guessed later; global is an explicit answer."""
    assert assumption_record().scope.is_global


def test_scope_is_carried_and_rendered() -> None:
    record = assumption_record(scope=ScopeSelector.parse("region:EU"))
    assert record.scope.render() == "region:eu"


def test_role_and_explicitness_are_separate_axes() -> None:
    record = assumption_record(role=BindingRole.CONSUMES, explicitness=Explicitness.IMPLIED)
    assert record.role is BindingRole.CONSUMES
    assert record.explicitness is Explicitness.IMPLIED


def test_the_retired_binding_kind_field_is_unrepresentable() -> None:
    with pytest.raises(ValidationError):
        AssumptionRecord(**{**assumption_record().model_dump(), "binding_kind": "asserts"})


def test_extraction_confidence_is_unrepresentable() -> None:
    """It was an LLM-supplied float wearing a carve-out. It is gone, not deprecated."""
    with pytest.raises(ValidationError):
        AssumptionRecord(**{**assumption_record().model_dump(), "extraction_confidence": 0.8})


def test_a_binding_cannot_declare_itself_a_different_claim_type() -> None:
    """The epistemic axes are facts about which variant this is, not writable fields."""
    with pytest.raises(ValidationError):
        assumption_record(claim_type=ClaimType.HYPOTHESIS)
    with pytest.raises(ValidationError):
        assumption_record(derivation=Derivation.DERIVED)
    with pytest.raises(ValidationError):
        assumption_record(support=Support.RULE_OVER_RECORDS)


def test_the_axes_default_to_the_type_b_values() -> None:
    record = assumption_record()
    assert record.claim_type is ClaimType.ASSUMPTION_BINDING
    assert record.derivation is Derivation.EXTRACTED
    assert record.support is Support.VERIFIED_SPAN


def test_explicitness_ordering_is_pinned() -> None:
    """The ordering feeds the confidence function, so a silent reordering is a bug."""
    assert (
        EXPLICITNESS_RANK[Explicitness.EXPLICIT]
        > EXPLICITNESS_RANK[Explicitness.DERIVED_FROM_TEXT]
        > EXPLICITNESS_RANK[Explicitness.IMPLIED]
    )


# ---------------------------------------------------------------------------
# Temporal validity
# ---------------------------------------------------------------------------


def test_effective_from_must_precede_effective_to() -> None:
    with pytest.raises(ValidationError, match="must precede"):
        assumption_record(
            effective_from=dt.datetime(2026, 6, 1, tzinfo=dt.UTC),
            effective_to=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        )


def test_a_zero_length_interval_is_invalid() -> None:
    with pytest.raises(ValidationError, match="must precede"):
        assumption_record(effective_from=AS_OF, effective_to=AS_OF)


def test_an_open_ended_interval_is_valid() -> None:
    assert assumption_record(effective_from=AS_OF, effective_to=None).is_in_force(AS_OF)


def test_validity_interval_is_half_open() -> None:
    """`[from, to)`: the end date is the first day the binding no longer governs."""
    record = assumption_record(
        effective_from=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        effective_to=dt.datetime(2026, 6, 1, tzinfo=dt.UTC),
    )
    assert record.is_in_force(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    assert record.is_in_force(dt.datetime(2026, 5, 31, tzinfo=dt.UTC))
    assert not record.is_in_force(dt.datetime(2026, 6, 1, tzinfo=dt.UTC))
    assert not record.is_in_force(dt.datetime(2025, 12, 31, tzinfo=dt.UTC))


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_admission_flags_default_to_unverified() -> None:
    """Extraction proposes; verification admits. A fresh record has passed nothing."""
    fresh = AssumptionRecord(
        assumption_id=make_id("assumption"),
        tenant_id=make_id("tenant"),
        key="refund.approval_threshold",
        registry_version="registry-1",
        value=usd(),
        role=BindingRole.DECLARES,
        explicitness=Explicitness.EXPLICIT,
        holder_source_version_id=make_id("source_version"),
        evidence_id=make_id("evidence"),
        extractor_version="extractor-1",
    )
    assert not fresh.span_verified
    assert not fresh.value_parsed
    assert not fresh.key_resolved
    assert not fresh.is_admitted
    assert fresh.extraction_quality == 0.0


def test_a_record_failing_any_check_is_not_admitted() -> None:
    for failing in ("span_verified", "value_parsed", "key_resolved"):
        record = assumption_record(**{failing: False})
        assert not record.is_admitted, f"{failing}=False must block admission"


def test_extraction_quality_is_computed_from_the_flags() -> None:
    assert assumption_record().extraction_quality == 1.0
    assert assumption_record(key_resolved=False).extraction_quality == pytest.approx(2 / 3)


def test_extraction_quality_cannot_be_supplied() -> None:
    """It is a computed property, not a field a model may fill in."""
    with pytest.raises(ValidationError):
        AssumptionRecord(**{**assumption_record().model_dump(), "extraction_quality": 1.0})


# ---------------------------------------------------------------------------
# CouplingSet
# ---------------------------------------------------------------------------


def test_coupling_set_identity_is_the_triple() -> None:
    """Keying on `key` alone was the defect behind the removed collision rule."""
    for field in ("key", "as_of"):
        payload = coupling_set().model_dump()
        del payload[field]
        with pytest.raises(ValidationError):
            type(coupling_set())(**payload)
    assert coupling_set().scope.is_global


def test_coupling_set_records_the_versions_that_produced_it() -> None:
    cs = coupling_set()
    assert cs.authority_model_version == "authority-1"
    assert cs.coupling_index_version == "coupling-1"


def test_role_partitions_are_exposed_for_collision_semantics() -> None:
    cs = coupling_set(
        holders=(
            coupling_holder(role=BindingRole.DECLARES, value=usd(10000)),
            coupling_holder(role=BindingRole.CONSUMES, value=usd(10000)),
            coupling_holder(role=BindingRole.DESCRIBES, value=usd(1)),
        )
    )
    assert cs.holder_count == 3
    assert cs.declarer_count == 1
    assert len(cs.consumers) == 1


def test_distinct_declared_values_counts_only_declarers() -> None:
    """A stale consumer is not a second authority, so it must not look like divergence."""
    cs = coupling_set(
        holders=(
            coupling_holder(role=BindingRole.DECLARES, value=usd(10000)),
            coupling_holder(role=BindingRole.CONSUMES, value=usd(5000)),
        )
    )
    assert cs.distinct_declared_values == 1


def test_two_divergent_declarers_are_visible_as_two_values() -> None:
    cs = coupling_set(
        holders=(
            coupling_holder(role=BindingRole.DECLARES, value=usd(10000)),
            coupling_holder(role=BindingRole.DECLARES, value=usd(5000)),
        )
    )
    assert cs.distinct_declared_values == 2


def test_single_holder_is_the_knowledge_decay_signal() -> None:
    assert coupling_set().is_single_holder
    assert not coupling_set(holders=(coupling_holder(), coupling_holder())).is_single_holder


def test_unwritten_rule_detects_a_key_no_governing_document_binds() -> None:
    config_only = coupling_set(holders=(coupling_holder(authority_class=AuthorityClass.CONFIG),))
    assert config_only.is_unwritten_rule

    governed = coupling_set(
        holders=(
            coupling_holder(authority_class=AuthorityClass.CONFIG),
            coupling_holder(authority_class=AuthorityClass.POLICY),
        )
    )
    assert not governed.is_unwritten_rule


def test_an_empty_coupling_set_is_not_an_unwritten_rule() -> None:
    assert not coupling_set(holders=()).is_unwritten_rule


def test_coupling_outcome_has_exactly_the_five_deterministic_values() -> None:
    assert {o.value for o in CouplingOutcome} == {
        "consistent",
        "conflicts_with",
        "legitimate_override",
        "unauthorized_override",
        "ambiguous_temporal",
    }


def test_coupling_set_defaults_to_consistent() -> None:
    """The resolver must positively establish divergence; silence is not a finding."""
    assert coupling_set().outcome is CouplingOutcome.CONSISTENT


def test_holders_carry_the_authority_the_override_test_needs() -> None:
    holder = coupling_holder(authority_class=AuthorityClass.WIKI)
    assert holder.authority_class is AuthorityClass.WIKI


def test_coupling_set_has_no_has_collision_field() -> None:
    """The scope- and authority-blind boolean was removed for the three-way outcome."""
    with pytest.raises(ValidationError):
        type(coupling_set())(**{**coupling_set().model_dump(), "has_collision": True})


# ---------------------------------------------------------------------------
# AssumptionDelta
# ---------------------------------------------------------------------------


def test_value_change_requires_both_endpoints() -> None:
    with pytest.raises(ValidationError, match="requires both"):
        assumption_delta(old_value=None)
    with pytest.raises(ValidationError, match="requires both"):
        assumption_delta(new_value=None)


def test_a_value_change_to_the_same_value_is_not_a_change() -> None:
    """A no-op delta would propagate invalidations for a change that did not happen."""
    with pytest.raises(ValidationError, match="identical values"):
        assumption_delta(old_value=usd(10000), new_value=usd(10000))


def test_key_added_carries_only_a_new_value() -> None:
    added = assumption_delta(delta_kind=DeltaKind.KEY_ADDED, old_value=None, new_value=usd(5000))
    assert added.old_value is None
    with pytest.raises(ValidationError, match="must not carry old_value"):
        assumption_delta(delta_kind=DeltaKind.KEY_ADDED, new_value=usd(5000))


def test_key_added_requires_a_new_value() -> None:
    with pytest.raises(ValidationError, match="requires new_value"):
        assumption_delta(delta_kind=DeltaKind.KEY_ADDED, old_value=None, new_value=None)


def test_key_removed_carries_only_an_old_value() -> None:
    removed = assumption_delta(delta_kind=DeltaKind.KEY_REMOVED, old_value=days(90), new_value=None)
    assert removed.new_value is None
    with pytest.raises(ValidationError, match="must not carry new_value"):
        assumption_delta(delta_kind=DeltaKind.KEY_REMOVED, new_value=usd(1))


def test_key_removed_requires_an_old_value() -> None:
    with pytest.raises(ValidationError, match="requires old_value"):
        assumption_delta(delta_kind=DeltaKind.KEY_REMOVED, old_value=None, new_value=None)


def test_a_delta_carries_no_confidence() -> None:
    """It is produced by the change-decomposer, an LLM."""
    with pytest.raises(ValidationError):
        AssumptionDelta(**{**assumption_delta().model_dump(), "confidence": 0.9})


def test_a_delta_pins_the_registry_version() -> None:
    payload = assumption_delta().model_dump()
    del payload["registry_version"]
    with pytest.raises(ValidationError):
        AssumptionDelta(**payload)


def test_a_delta_may_cite_spans_of_the_change_request() -> None:
    delta = assumption_delta(evidence_ids=(make_id("evidence"),))
    assert len(delta.evidence_ids) == 1


def test_a_delta_evidence_id_must_still_be_well_shaped() -> None:
    with pytest.raises(ValidationError):
        assumption_delta(evidence_ids=("the change request",))
