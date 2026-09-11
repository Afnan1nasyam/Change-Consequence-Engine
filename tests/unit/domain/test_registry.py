"""The closed registry and the two promotion gates.

These tests are the schema-level half of the governing invariant: *an LLM may propose a
semantic relationship, but it can never promote that relationship into the trusted
deterministic layer by itself.* Neither gate has a bypass, and there is no confidence
threshold that would open one - so no such field exists to set.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.domain.entities.registry import AssumptionKeyRegistryEntry, KeyAlias
from app.domain.value_objects.enums import AliasStatus, RegistryGapStatus, ValueType
from tests.fixtures.domain import AS_OF, key_alias, make_id, registry_entry, registry_gap

# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------


def test_registry_entry_requires_a_unit_for_a_scalar_key() -> None:
    with pytest.raises(ValidationError, match="unit is required"):
        registry_entry(unit=None)


def test_registry_entry_forbids_a_unit_where_it_is_meaningless() -> None:
    with pytest.raises(ValidationError, match="unit must be omitted"):
        registry_entry(value_type=ValueType.BOOLEAN, unit="USD")


def test_registry_entry_declares_a_canonical_form() -> None:
    """Polarity handling is a registry obligation, not inference (checkpoint 3.7)."""
    with pytest.raises(ValidationError):
        registry_entry(canonical_form="")


def test_allowed_scopes_are_declared_not_invented() -> None:
    entry = registry_entry(allowed_scopes=("region", "tier"))
    assert entry.permits_dimension("region")
    assert not entry.permits_dimension("business_unit")


def test_an_entry_with_no_allowed_scopes_permits_no_dimension() -> None:
    assert not registry_entry(allowed_scopes=()).permits_dimension("region")


def test_deprecation_is_versioned() -> None:
    assert registry_entry().is_active
    assert not registry_entry(deprecated_in_version="registry-4").is_active


def test_registry_entry_key_must_be_well_shaped() -> None:
    with pytest.raises(ValidationError):
        registry_entry(key="Refund Approval Threshold")


def test_registry_entry_forbids_a_confidence_field() -> None:
    with pytest.raises(ValidationError):
        AssumptionKeyRegistryEntry(**{**registry_entry().model_dump(), "confidence": 0.9})


# ---------------------------------------------------------------------------
# KeyRegistryGap - quarantine
# ---------------------------------------------------------------------------


def test_an_open_gap_blocks_class_a() -> None:
    """Unknown keys never silently enter the coupling index. There is no fallback path."""
    assert registry_gap(status=RegistryGapStatus.OPEN).blocks_class_a
    assert registry_gap(status=RegistryGapStatus.REJECTED).blocks_class_a
    assert not registry_gap(status=RegistryGapStatus.REGISTERED).blocks_class_a


def test_a_gap_records_what_was_proposed_and_when() -> None:
    """Gap volume is a product signal, which requires the key, the evidence and `as_of`."""
    gap = registry_gap()
    assert gap.proposed_key
    assert gap.evidence_id
    assert gap.as_of == AS_OF
    assert gap.occurrence_count == 1

    for field in ("proposed_key", "evidence_id", "as_of", "proposed_by", "registry_version"):
        payload = gap.model_dump()
        del payload[field]
        with pytest.raises(ValidationError):
            type(gap)(**payload)


def test_a_gap_may_hold_an_unregistered_key_shape() -> None:
    """The proposed key is deliberately free text: it is what a model actually emitted."""
    assert registry_gap(proposed_key="Refund Ceiling (EMEA)").proposed_key


def test_occurrence_count_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        registry_gap(occurrence_count=0)


# ---------------------------------------------------------------------------
# KeyAlias - the equivalence gate
# ---------------------------------------------------------------------------


def test_only_an_approved_alias_participates_in_class_a() -> None:
    assert not key_alias(status=AliasStatus.PROPOSED).participates_in_class_a
    assert not key_alias(
        status=AliasStatus.REJECTED, rejection_reason="different variables"
    ).participates_in_class_a
    assert not key_alias(status=AliasStatus.SUPERSEDED).participates_in_class_a
    assert key_alias(
        status=AliasStatus.APPROVED,
        approved_by="owner@example.com",
        approved_at=AS_OF,
    ).participates_in_class_a


def test_approval_must_be_auditable() -> None:
    """An approved alias with no named approver is an automatic approval in disguise."""
    with pytest.raises(ValidationError, match="approved_at"):
        key_alias(status=AliasStatus.APPROVED, approved_by="owner@example.com")
    with pytest.raises(ValidationError, match="approved_by"):
        key_alias(status=AliasStatus.APPROVED, approved_at=AS_OF)
    with pytest.raises(ValidationError, match="no automatic approval path"):
        key_alias(status=AliasStatus.APPROVED)


def test_approval_fields_cannot_be_set_while_unapproved() -> None:
    """Otherwise a proposal could carry a pre-filled approval waiting to be flipped."""
    with pytest.raises(ValidationError, match="approval fields set while status"):
        key_alias(status=AliasStatus.PROPOSED, approved_by="owner@example.com")


def test_a_rejected_alias_must_give_a_reason() -> None:
    with pytest.raises(ValidationError, match="requires a reason"):
        key_alias(status=AliasStatus.REJECTED)


def test_an_alias_cannot_point_at_itself() -> None:
    with pytest.raises(ValidationError, match="identical"):
        key_alias(source_key="refund.approval_threshold")


def test_a_proposal_may_inform_class_b_but_a_rejection_may_not() -> None:
    assert key_alias(status=AliasStatus.PROPOSED).usable_in_class_b
    assert not key_alias(
        status=AliasStatus.REJECTED, rejection_reason="unit mismatch"
    ).usable_in_class_b
    assert not key_alias(status=AliasStatus.SUPERSEDED).usable_in_class_b


def test_an_alias_pins_the_registry_version_it_was_approved_against() -> None:
    """This is what makes the promotion point auditable and two runs diffable."""
    payload = key_alias().model_dump()
    del payload["registry_version"]
    with pytest.raises(ValidationError):
        KeyAlias(**payload)


def test_an_alias_carries_no_confidence_score() -> None:
    """Aliases are gated, not scored. A model's certainty is not evidence of equivalence."""
    with pytest.raises(ValidationError):
        KeyAlias(**{**key_alias().model_dump(), "confidence": 0.99})
    with pytest.raises(ValidationError):
        KeyAlias(**{**key_alias().model_dump(), "similarity": 0.99})


def test_alias_status_has_exactly_four_states_and_no_auto_variant() -> None:
    assert {s.value for s in AliasStatus} == {"proposed", "approved", "rejected", "superseded"}


def test_target_key_must_be_a_registered_key_shape() -> None:
    with pytest.raises(ValidationError):
        key_alias(target_key="Some Free Text")


def test_alias_proposal_records_the_proposing_model() -> None:
    alias = key_alias()
    assert alias.proposed_by is not None
    assert alias.proposed_by.prompt_sha256
    assert alias.proposed_at == AS_OF


def test_a_human_proposal_needs_no_model_provenance() -> None:
    alias = key_alias(proposed_by=None)
    assert alias.proposed_by is None


def test_alias_ids_are_shape_validated() -> None:
    with pytest.raises(ValidationError):
        key_alias(alias_id="alias-1")
    assert key_alias(alias_id=make_id("alias", 7)).alias_id == make_id("alias", 7)


def test_approved_at_accepts_an_explicit_timestamp() -> None:
    approved = key_alias(
        status=AliasStatus.APPROVED,
        approved_by="owner@example.com",
        approved_at=dt.datetime(2026, 3, 1, tzinfo=dt.UTC),
    )
    assert approved.approved_at == dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
