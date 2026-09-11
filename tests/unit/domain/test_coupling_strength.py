"""The coupling-strength function: the two gates, the four directions, and what it cannot see.

:func:`test_the_function_cannot_reach_agent_identity_or_model_output` is the load-bearing
test. Checkpoint 5.7 and 9.3 forbid agent identity, agent count, model agreement and
LLM confidence from entering the score, and the enforcement is meant to be *structural* -
the function's parameters do not include them. This asserts that by inspecting the
signature and the reachable field set, so adding such an input fails a test rather than
passing review.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pytest
from pydantic import BaseModel, ValidationError

from app.domain.entities.coupling import CouplingStrength, ResolvedBinding
from app.domain.services import coupling_strength
from app.domain.value_objects.enums import AuthorityClass, BindingRole, Explicitness
from tests.fixtures.coupling import binding, score, usd

JAN = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
APR = dt.datetime(2025, 4, 1, tzinfo=dt.UTC)
JUL = dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

DECLARES = BindingRole.DECLARES
CONSUMES = BindingRole.CONSUMES
DESCRIBES = BindingRole.DESCRIBES


def compute(
    left: ResolvedBinding, right: ResolvedBinding, *, alias_mediated: bool = False
) -> CouplingStrength:
    return coupling_strength.compute(left, right, as_of=NOW, alias_mediated=alias_mediated)


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def test_a_unit_mismatch_blocks_with_strength_zero() -> None:
    """Checkpoint 5.6: a gate, not a weight."""
    result = compute(binding(value=usd(10000)), binding(value=score(0.85)))
    assert result.blocked is True
    assert result.strength == 0.0


def test_an_unverified_binding_cannot_be_constructed_at_all() -> None:
    """The span gate in the scorer is unreachable through normal construction.

    ``ResolvedBinding`` refuses a record that has not passed all three admission checks, so
    an unverified binding cannot enter the coupling index and therefore cannot reach the
    scorer. That is the stronger guarantee, and it is stated here rather than papered over.
    """
    with pytest.raises(ValidationError):
        binding(span_verified=False)


def test_the_span_gate_still_holds_for_a_caller_that_bypasses_validation() -> None:
    """Defence in depth, exercised the only way it can be reached.

    ``model_construct`` skips validators, which is what a future non-Pydantic caller - a
    SQL row hydrated without re-validation, say - would effectively do. The gate is cheap
    and the primitive it protects is the one everything else rests on, so it stays.
    """
    good = binding(tag="a")
    unverified = good.model_copy(
        update={
            "record": good.record.model_construct(
                **{**good.record.model_dump(), "span_verified": False}
            )
        }
    )
    result = compute(good, unverified)
    assert result.blocked is True
    assert result.strength == 0.0
    assert "gate_span_verification" in result.factor_inputs
    assert result.factor_inputs["right_span_verified"] == 0.0


def test_a_blocked_result_records_which_gate_failed() -> None:
    unit = compute(binding(value=usd(10000)), binding(value=score(0.85)))
    assert "gate_unit_compatibility" in unit.factor_inputs
    assert unit.factor_inputs["blocked_by_gate"] == 1.0


def test_a_blocked_score_cannot_be_non_zero() -> None:
    """The entity refuses the inconsistent state outright."""
    with pytest.raises(ValidationError):
        CouplingStrength(
            strength=0.5,
            blocked=True,
            factor_inputs={"role_composition": 1.0},
            strength_fn_version="strength-1",
        )


def test_no_combination_of_strong_factors_carries_a_blocked_pair() -> None:
    """The gate must not be outvoted by everything else being perfect."""
    result = compute(
        binding(
            value=usd(10000),
            role=DECLARES,
            explicitness=Explicitness.EXPLICIT,
            authority=AuthorityClass.REGULATORY,
        ),
        binding(
            value=score(0.85),
            role=CONSUMES,
            explicitness=Explicitness.EXPLICIT,
            authority=AuthorityClass.REGULATORY,
        ),
    )
    assert result.strength == 0.0


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------


def test_declares_to_consumes_is_the_strongest_role_composition() -> None:
    """It is the dependency the mechanism exists to find."""
    dependency = compute(binding(role=DECLARES, tag="a"), binding(role=CONSUMES, tag="b"))
    peers = compute(binding(role=DECLARES, tag="a"), binding(role=DECLARES, tag="b"))
    consumers = compute(binding(role=CONSUMES, tag="a"), binding(role=CONSUMES, tag="b"))

    assert dependency.strength > peers.strength > consumers.strength


def test_a_describes_binding_scores_lowest() -> None:
    describes = compute(binding(role=DECLARES, tag="a"), binding(role=DESCRIBES, tag="b"))
    consumers = compute(binding(role=CONSUMES, tag="a"), binding(role=CONSUMES, tag="b"))
    assert describes.strength < consumers.strength


def test_explicitness_raises_strength() -> None:
    explicit = compute(
        binding(explicitness=Explicitness.EXPLICIT, tag="a"),
        binding(explicitness=Explicitness.EXPLICIT, tag="b"),
    )
    implied = compute(
        binding(explicitness=Explicitness.IMPLIED, tag="a"),
        binding(explicitness=Explicitness.IMPLIED, tag="b"),
    )
    assert explicit.strength > implied.strength


def test_scope_distance_lowers_strength() -> None:
    same = compute(binding(scope="region:eu", tag="a"), binding(scope="region:eu", tag="b"))
    apart = compute(binding(scope="global", tag="a"), binding(scope="region:eu", tag="b"))
    assert same.strength > apart.strength
    assert same.factor_inputs["scope_distance"] == 0.0
    assert apart.factor_inputs["scope_distance"] == 1.0


def test_disjoint_scopes_score_zero_proximity_without_raising() -> None:
    """The function stays total even when the caller should not have asked."""
    result = compute(binding(scope="region:eu", tag="a"), binding(scope="region:us", tag="b"))
    assert result.factor_inputs["scope_proximity"] == 0.0
    assert result.factor_inputs["scope_distance"] == -1.0


def test_temporal_overlap_raises_strength() -> None:
    full = compute(
        binding(effective_from=JAN, effective_to=JUL, tag="a"),
        binding(effective_from=JAN, effective_to=JUL, tag="b"),
    )
    partial = compute(
        binding(effective_from=JAN, effective_to=JUL, tag="a"),
        binding(effective_from=APR, effective_to=dt.datetime(2025, 10, 1, tzinfo=dt.UTC), tag="b"),
    )
    assert full.strength > partial.strength


def test_higher_authority_raises_strength() -> None:
    high = compute(
        binding(authority=AuthorityClass.REGULATORY, tag="a"),
        binding(authority=AuthorityClass.REGULATORY, tag="b"),
    )
    low = compute(
        binding(authority=AuthorityClass.CHAT, tag="a"),
        binding(authority=AuthorityClass.CHAT, tag="b"),
    )
    assert high.strength > low.strength


def test_alias_mediation_lowers_strength() -> None:
    """An approved alias is trusted enough for Class A; a direct match is still stronger."""
    direct = compute(binding(tag="a"), binding(tag="b"))
    aliased = compute(binding(tag="a"), binding(tag="b"), alias_mediated=True)
    assert aliased.strength < direct.strength
    assert aliased.factor_inputs["alias_mediated"] == 1.0


# ---------------------------------------------------------------------------
# Shape and reproducibility
# ---------------------------------------------------------------------------


def test_strength_stays_within_bounds() -> None:
    combinations = [
        (
            binding(role=DECLARES, authority=AuthorityClass.REGULATORY, tag="a"),
            binding(role=CONSUMES, authority=AuthorityClass.REGULATORY, tag="b"),
        ),
        (
            binding(
                role=DESCRIBES,
                authority=AuthorityClass.CHAT,
                explicitness=Explicitness.IMPLIED,
                tag="a",
            ),
            binding(
                role=DESCRIBES,
                authority=AuthorityClass.CHAT,
                explicitness=Explicitness.IMPLIED,
                tag="b",
            ),
        ),
    ]
    for left, right in combinations:
        assert 0.0 <= compute(left, right).strength <= 1.0


def test_the_inputs_are_persisted_so_the_score_is_recomputable() -> None:
    result = compute(binding(tag="a"), binding(tag="b"))
    assert result.factor_inputs
    for factor in ("role_composition", "explicitness", "scope_proximity", "temporal_overlap"):
        assert factor in result.factor_inputs


def test_a_score_with_no_recorded_inputs_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CouplingStrength(strength=0.5, factor_inputs={}, strength_fn_version="strength-1")


def test_the_function_is_versioned() -> None:
    assert compute(binding(tag="a"), binding(tag="b")).strength_fn_version == "strength-1"


def test_the_function_is_symmetric() -> None:
    left = binding(role=DECLARES, authority=AuthorityClass.POLICY, tag="a")
    right = binding(role=CONSUMES, authority=AuthorityClass.WIKI, tag="b")
    assert compute(left, right).strength == compute(right, left).strength


def test_the_function_is_deterministic() -> None:
    left, right = binding(tag="a"), binding(tag="b")
    assert compute(left, right).model_dump() == compute(left, right).model_dump()


# ---------------------------------------------------------------------------
# What the function cannot see
# ---------------------------------------------------------------------------


def _reachable_fields(model: type[BaseModel]) -> set[str]:
    found: set[str] = set()
    for name, field in model.model_fields.items():
        found.add(name)
        for candidate in (field.annotation, *getattr(field.annotation, "__args__", ())):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                found |= _reachable_fields(candidate)
    return found


def test_the_function_cannot_reach_agent_identity_or_model_output() -> None:
    """Structural enforcement of checkpoint 5.7 and 9.3.

    Corroboration counts sources, never agents. The score cannot count agents because no
    parameter and no reachable field carries one.
    """
    signature = inspect.signature(coupling_strength.compute)
    forbidden = {
        "agent",
        "agents",
        "agent_count",
        "votes",
        "model",
        "provider",
        "confidence",
        "prompt",
        "prompt_sha256",
        "model_agreement",
    }
    assert not (set(signature.parameters) & forbidden), signature

    reachable = _reachable_fields(ResolvedBinding)
    offenders = {
        field
        for field in reachable
        if any(token in field for token in ("agent", "prompt", "provider", "confidence"))
    }
    assert not offenders, offenders


def test_a_strength_score_carries_no_agent_attribution_field() -> None:
    assert not (
        {"agent", "produced_by", "proposed_by", "provider", "model"}
        & set(CouplingStrength.model_fields)
    )


def test_a_strength_score_rejects_an_injected_agent_key() -> None:
    """Built from an untyped mapping, so the runtime gate is what is under test.

    Passing ``agent=`` directly is also a static type error - the field does not exist -
    which is the stronger guarantee. This asserts the second layer: even a caller that got
    past the type checker is refused by ``extra="forbid"``.
    """
    payload = {
        "strength": 0.9,
        "factor_inputs": {"role_composition": 1.0},
        "strength_fn_version": "strength-1",
        "agent": "consequence-hunter",
    }
    with pytest.raises(ValidationError):
        CouplingStrength(**payload)
