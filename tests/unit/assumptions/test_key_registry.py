"""The closed registry: loading, resolution, and the alias gate.

The shipped registry is asserted against, not a stub, because the seed vocabulary is a
product decision: it decides what can be Class A at all.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from app.assumptions.registry import (
    AssumptionKeyRegistry,
    RegistryError,
    ResolutionFailure,
    scope_is_permitted,
)
from app.config.settings import REPO_ROOT
from app.domain.entities.registry import KeyAlias
from app.domain.value_objects.enums import AliasStatus, ValueType
from app.domain.value_objects.scope import ScopeSelector
from tests.fixtures.domain import key_alias, registry_entry

SHIPPED = REPO_ROOT / "app" / "config" / "assumption_keys.yaml"
AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def shipped() -> AssumptionKeyRegistry:
    return AssumptionKeyRegistry.load(SHIPPED)


# ---------------------------------------------------------------------------
# The shipped registry
# ---------------------------------------------------------------------------


def test_the_shipped_registry_loads_and_is_versioned(shipped: AssumptionKeyRegistry) -> None:
    assert shipped.version == "registry-1"
    assert len(shipped) > 0


def test_the_seed_vocabulary_stays_small(shipped: AssumptionKeyRegistry) -> None:
    """An enormous speculative ontology makes Class A look broad while being unvalidated."""
    assert len(shipped) <= 30, "the MVP registry is deliberately a seed, not an ontology"


def test_the_seed_covers_the_benchmark_domains(shipped: AssumptionKeyRegistry) -> None:
    expected = {
        "payments",
        "data_retention",
        "access_management",
        "account_lifecycle",
        "sla_process",
        "support_operations",
        "api_contracts",
        "reporting",
    }
    assert expected <= set(shipped.domains)


def test_every_shipped_entry_declares_a_canonical_form(shipped: AssumptionKeyRegistry) -> None:
    """Polarity is a registry obligation, not inference (checkpoint 3.7)."""
    for key in shipped.keys:
        entry = shipped.entry(key)
        assert entry is not None and entry.canonical_form.strip()


def test_every_shipped_entry_with_a_unit_type_has_a_unit(
    shipped: AssumptionKeyRegistry,
) -> None:
    for key in shipped.keys:
        entry = shipped.entry(key)
        assert entry is not None
        if entry.value_type in (ValueType.SCALAR, ValueType.DURATION):
            assert entry.unit, key


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_a_registered_key_resolves(shipped: AssumptionKeyRegistry) -> None:
    resolution = shipped.resolve("refund.approval_threshold")
    assert resolution.resolved
    assert not resolution.alias_mediated


def test_an_unregistered_key_does_not_resolve(shipped: AssumptionKeyRegistry) -> None:
    """The closed-registry decision: there is no fallback path."""
    resolution = shipped.resolve("refund.escalation_ceiling")
    assert not resolution.resolved
    assert resolution.failure is ResolutionFailure.UNREGISTERED


def test_resolution_does_not_guess_at_a_near_miss(shipped: AssumptionKeyRegistry) -> None:
    """No clustering, no similarity, no learned normalization."""
    for near in (
        "refund.approval_thresholds",
        "refund.approval_limit",
        "refunds.approval_threshold",
        "REFUND.APPROVAL_THRESHOLD",
        "refund_approval_threshold",
    ):
        assert not shipped.resolve(near).resolved, near


def test_surrounding_whitespace_is_the_only_tolerance(shipped: AssumptionKeyRegistry) -> None:
    assert shipped.resolve("  refund.approval_threshold  ").resolved


def test_a_deprecated_key_is_refused_distinctly() -> None:
    registry = AssumptionKeyRegistry(
        version="registry-2",
        entries=[registry_entry(deprecated_in_version="registry-2")],
    )
    resolution = registry.resolve("refund.approval_threshold")
    assert resolution.failure is ResolutionFailure.DEPRECATED


# ---------------------------------------------------------------------------
# The alias gate
# ---------------------------------------------------------------------------


def _approved_alias() -> KeyAlias:
    return key_alias(
        status=AliasStatus.APPROVED,
        approved_by="owner@example.com",
        approved_at=AS_OF,
    )


def test_a_proposed_alias_does_not_resolve() -> None:
    """The governing invariant: a model cannot promote an equivalence by itself."""
    registry = AssumptionKeyRegistry(
        version="registry-1",
        entries=[registry_entry()],
        aliases=[key_alias(status=AliasStatus.PROPOSED)],
    )
    resolution = registry.resolve("refund.manager_approval_limit")
    assert not resolution.resolved
    assert resolution.failure is ResolutionFailure.ALIAS_NOT_APPROVED
    assert "No confidence threshold approves it" in resolution.detail


def test_an_approved_alias_resolves_and_is_marked_alias_mediated() -> None:
    registry = AssumptionKeyRegistry(
        version="registry-1", entries=[registry_entry()], aliases=[_approved_alias()]
    )
    resolution = registry.resolve("refund.manager_approval_limit")
    assert resolution.resolved
    assert resolution.alias_mediated, "alias mediation lowers confidence downstream"
    assert resolution.entry is not None
    assert resolution.entry.key == "refund.approval_threshold"


@pytest.mark.parametrize("status", [AliasStatus.REJECTED, AliasStatus.SUPERSEDED])
def test_a_rejected_or_superseded_alias_does_not_resolve(status: AliasStatus) -> None:
    extra: dict[str, str] = (
        {"rejection_reason": "different variable"} if status is AliasStatus.REJECTED else {}
    )
    registry = AssumptionKeyRegistry(
        version="registry-1",
        entries=[registry_entry()],
        aliases=[key_alias(status=status, **extra)],
    )
    assert not registry.resolve("refund.manager_approval_limit").resolved


def test_an_approved_alias_cannot_target_an_unregistered_key() -> None:
    """Otherwise approval could smuggle in a key the registry never declared."""
    with pytest.raises(RegistryError, match="target unregistered keys"):
        AssumptionKeyRegistry(
            version="registry-1",
            entries=[
                registry_entry(key="retention.window", value_type=ValueType.DURATION, unit="d")
            ],
            aliases=[_approved_alias()],
        )


def test_an_approved_alias_to_a_deprecated_target_is_refused() -> None:
    registry = AssumptionKeyRegistry(
        version="registry-1",
        entries=[registry_entry(deprecated_in_version="registry-2")],
        aliases=[_approved_alias()],
    )
    assert registry.resolve("refund.manager_approval_limit").failure is (
        ResolutionFailure.DEPRECATED
    )


# ---------------------------------------------------------------------------
# Scope permission
# ---------------------------------------------------------------------------


def test_global_scope_is_always_permitted() -> None:
    entry = registry_entry(allowed_scopes=())
    assert scope_is_permitted(entry, ScopeSelector.global_scope())


def test_a_declared_dimension_is_permitted() -> None:
    entry = registry_entry(allowed_scopes=("region", "tier"))
    assert scope_is_permitted(entry, ScopeSelector.parse("region:EU"))
    assert scope_is_permitted(entry, ScopeSelector.parse("region:EU / tier:enterprise"))


def test_an_undeclared_dimension_is_refused() -> None:
    """A scope is validated, never invented: an undesigned axis fragments the key."""
    entry = registry_entry(allowed_scopes=("region",))
    assert not scope_is_permitted(entry, ScopeSelector.parse("business_unit:emea"))
    assert not scope_is_permitted(entry, ScopeSelector.parse("region:EU / tier:enterprise"))


# ---------------------------------------------------------------------------
# Loading failures
# ---------------------------------------------------------------------------


def test_a_missing_registry_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        AssumptionKeyRegistry.load(tmp_path / "absent.yaml")


def test_a_registry_without_a_version_is_refused() -> None:
    with pytest.raises(RegistryError, match="registry_version"):
        AssumptionKeyRegistry.from_mapping({"keys": [{"key": "a.b"}]})


def test_a_registry_with_no_keys_is_refused() -> None:
    with pytest.raises(RegistryError, match="non-empty list"):
        AssumptionKeyRegistry.from_mapping({"registry_version": "r1", "keys": []})


def test_an_invalid_entry_names_itself() -> None:
    """A registry that silently lost an entry would silently demote its consequences."""
    with pytest.raises(RegistryError, match=r"retention\.window"):
        AssumptionKeyRegistry.from_mapping(
            {
                "registry_version": "r1",
                "keys": [
                    {
                        "key": "retention.window",
                        "value_type": "duration",
                        # unit missing: mandatory for a duration
                        "canonical_form": "x",
                        "description": "y",
                        "domain": "d",
                        "introduced_in_version": "r1",
                    }
                ],
            }
        )


def test_a_duplicate_key_is_refused() -> None:
    with pytest.raises(RegistryError, match="duplicate registry key"):
        AssumptionKeyRegistry(version="r1", entries=[registry_entry(), registry_entry()])


def test_a_non_mapping_registry_is_refused() -> None:
    with pytest.raises(RegistryError, match="must be a mapping"):
        AssumptionKeyRegistry.from_mapping(["not", "a", "mapping"])
