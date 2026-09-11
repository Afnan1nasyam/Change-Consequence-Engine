"""The deterministic coupling-strength function.

Checkpoint 5.7 fixes the inputs and their directions. This module fixes the arithmetic and
pins it behind ``COUPLING_STRENGTH_VERSION``.

**What the signature makes impossible.** The function takes two
:class:`ResolvedBinding` values and nothing else. There is no parameter for agent identity,
agent count, model agreement, or a model-supplied number, so it cannot read them - the
enforcement is structural rather than a rule somebody has to remember (checkpoint 9.3).
Corroboration counts sources; several agents reading one span are one source.

**The two gates are gates, not weights.** A span that failed verification and a unit
mismatch both produce strength exactly ``0.0`` with ``blocked=True``. A blocked pair must
not be coupled at all. Expressing either as a low score would let a strong enough set of
other factors carry an incomparable pair over a threshold, which is precisely how a USD
threshold ends up "contradicting" a 0-1 fraud score.

The weights are **not tuned**. They encode the orderings the checkpoint states and nothing
more; there is no calibration data yet. That is why the version label exists, and why
strength is currently reported rather than used as a filter.
"""

from __future__ import annotations

import datetime as dt

from app.domain.entities.coupling import CouplingStrength, ResolvedBinding
from app.domain.services import scope_lattice
from app.domain.services.temporal_intervals import overlap_fraction
from app.domain.services.value_comparison import is_blocked
from app.domain.value_objects.enums import (
    AUTHORITY_RANK,
    EXPLICITNESS_RANK,
    BindingRole,
)
from app.domain.value_objects.rules import COUPLING_STRENGTH_VERSION

#: Clamp for unbounded validity intervals, so an open-ended pair does not score a
#: meaningless 1.0 on a corpus where most documents are open-ended.
DEFAULT_HORIZON = dt.timedelta(days=365 * 5)

#: Role composition, checkpoint 5.7: ``declares``<->``consumes`` strongest because that is
#: the dependency the mechanism exists to find; ``describes`` weakest because it never
#: creates an obligation.
_ROLE_COMPOSITION: dict[frozenset[BindingRole], float] = {
    frozenset({BindingRole.DECLARES, BindingRole.CONSUMES}): 1.0,
    frozenset({BindingRole.DECLARES}): 0.9,
    frozenset({BindingRole.CONSUMES}): 0.5,
    frozenset({BindingRole.DECLARES, BindingRole.DESCRIBES}): 0.1,
    frozenset({BindingRole.CONSUMES, BindingRole.DESCRIBES}): 0.1,
    frozenset({BindingRole.DESCRIBES}): 0.1,
}

#: Weights over the four scored factors. Sum to 1.0 so the result needs no rescaling.
_WEIGHTS: dict[str, float] = {
    "role_composition": 0.40,
    "explicitness": 0.20,
    "scope_proximity": 0.20,
    "temporal_overlap": 0.20,
}

#: Authority acts as a mild multiplier rather than an additive factor: a coupling between
#: two low-authority documents is real, just less consequential.
_AUTHORITY_FLOOR = 0.6

#: Alias mediation penalty (checkpoint 5.7, 9.1). An approved alias is trusted enough to
#: produce Class A, but a direct key match is still the stronger signal.
_ALIAS_FACTOR = 0.8

_MAX_AUTHORITY = max(AUTHORITY_RANK.values())
_MAX_EXPLICITNESS = max(EXPLICITNESS_RANK.values())


def compute(
    left: ResolvedBinding,
    right: ResolvedBinding,
    *,
    as_of: dt.datetime,
    alias_mediated: bool = False,
    horizon: dt.timedelta = DEFAULT_HORIZON,
) -> CouplingStrength:
    """Score the coupling between two bindings of the same key.

    ``as_of`` is accepted for symmetry with the rest of the resolver and to keep the
    signature honest about being evaluated at a date; the interval overlap it feeds is a
    property of the two intervals, not of the instant.
    """
    if not left.record.span_verified or not right.record.span_verified:
        return _blocked("span_verification", left, right)

    if is_blocked(left.value, right.value):
        return _blocked("unit_compatibility", left, right)

    roles = frozenset({left.role, right.role})
    role_score = _ROLE_COMPOSITION[roles]

    explicitness = (
        EXPLICITNESS_RANK[left.explicitness] + EXPLICITNESS_RANK[right.explicitness]
    ) / (2 * _MAX_EXPLICITNESS)

    distance = scope_lattice.distance(left.scope, right.scope)
    # Disjoint scopes are not coupled at all; the caller should not have asked, but
    # scoring 0 rather than raising keeps the function total.
    scope_proximity = 0.0 if distance is None else 1.0 / (1.0 + distance)

    temporal = overlap_fraction(left.interval, right.interval, horizon=horizon)

    authority = (left.authority_rank + right.authority_rank) / (2 * _MAX_AUTHORITY)
    authority_multiplier = _AUTHORITY_FLOOR + (1.0 - _AUTHORITY_FLOOR) * authority

    factors = {
        "role_composition": role_score,
        "explicitness": explicitness,
        "scope_proximity": scope_proximity,
        "temporal_overlap": temporal,
    }
    base = sum(_WEIGHTS[name] * value for name, value in factors.items())
    strength = base * authority_multiplier
    if alias_mediated:
        strength *= _ALIAS_FACTOR

    return CouplingStrength(
        strength=round(min(1.0, max(0.0, strength)), 6),
        blocked=False,
        factor_inputs={
            **factors,
            "authority_multiplier": round(authority_multiplier, 6),
            "alias_mediated": 1.0 if alias_mediated else 0.0,
            "scope_distance": -1.0 if distance is None else float(distance),
        },
        strength_fn_version=COUPLING_STRENGTH_VERSION,
    )


def _blocked(gate: str, left: ResolvedBinding, right: ResolvedBinding) -> CouplingStrength:
    """A failed gate. Strength is exactly zero and the pair must not be coupled."""
    return CouplingStrength(
        strength=0.0,
        blocked=True,
        factor_inputs={
            "blocked_by_gate": 1.0,
            f"gate_{gate}": 0.0,
            "left_span_verified": 1.0 if left.record.span_verified else 0.0,
            "right_span_verified": 1.0 if right.record.span_verified else 0.0,
        },
        strength_fn_version=COUPLING_STRENGTH_VERSION,
    )
