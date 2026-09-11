"""Invalidation propagation: an ``AssumptionDelta`` against the coupling index.

Checkpoint 5.5. A binding is coupled for invalidation to a delta iff **all** hold:

1. same registry key (or an approved ``KeyAlias``);
2. scope **intersects** the delta's scope - inheritance counts, so a ``global`` change
   reaches narrower inheriting scopes;
3. the binding is valid at the delta's effective date;
4. ``role in {declares, consumes}`` - ``describes`` never invalidates;
5. the binding's value is inconsistent with the delta's ``new_value``.

**Intersection, not comparability.** Contradiction needs comparable scope; invalidation
needs only intersection. Conflating the two was a defect in an earlier draft, and the two
relations must not be merged.

**The three cases that are easy to get wrong**, all called out by the checkpoint:

*A legitimate narrower override is not broken by a broader change.* It reports
``scope_inheritance_affected``: the override may still be intended, and it needs
confirmation rather than correction. The override itself is **never** labelled
contradictory by an invalidation pass.

*An unauthorized narrower override yields two independent findings that must not be
merged.* The pre-existing ``unauthorized_override`` was not caused by the proposed change,
and folding it into the change's impact would blame the change for a governance defect that
predates it. This module therefore reports the invalidation and leaves the override finding
where the standing resolver put it.

*The declaration the change edits is not a consequence of the change.* A delta that moves
``retention.window`` from 90d to 180d does not make the retention policy stale - it *is* the
edit to the retention policy. Reporting it would put the document the author is already
holding at the top of their own impact report.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from app.assumptions.coupling import CouplingIndex
from app.domain.entities.assumption import AssumptionDelta
from app.domain.entities.claims import DerivedRelation
from app.domain.entities.coupling import OverrideRecord, ResolvedBinding
from app.domain.services import scope_lattice
from app.domain.services.value_comparison import are_divergent, is_blocked
from app.domain.value_objects.enums import (
    AUTHORITY_MODEL_VERSION,
    BindingRole,
    InvalidationOutcome,
)
from app.domain.value_objects.provenance import DerivationProvenance
from app.domain.value_objects.rules import COUPLING_INDEX_VERSION, RuleId
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from app.ingestion.identity import relation_id_for

#: Roles that can be invalidated. ``describes`` is absent by design: a retrospective
#: narrating what the threshold used to be does not become wrong when the threshold moves.
_INVALIDATABLE: frozenset[BindingRole] = frozenset({BindingRole.DECLARES, BindingRole.CONSUMES})


@dataclass(frozen=True)
class BindingImpact:
    """What the delta does to one binding, and why."""

    binding: ResolvedBinding
    outcome: InvalidationOutcome
    reason: str
    relation: DerivedRelation | None = None


@dataclass(frozen=True)
class InvalidationReport:
    """Everything one delta deterministically implies, with nothing merged."""

    delta: AssumptionDelta
    as_of: dt.datetime
    impacts: tuple[BindingImpact, ...] = ()
    #: Overrides that predate the change and were not caused by it. Reported separately so
    #: a pre-existing governance defect is never attributed to the proposed change.
    pre_existing_unauthorized_overrides: tuple[DerivedRelation, ...] = ()

    @property
    def relations(self) -> tuple[DerivedRelation, ...]:
        found = [i.relation for i in self.impacts if i.relation is not None]
        return tuple(sorted(found, key=lambda r: r.relation_id))

    def of_outcome(self, outcome: InvalidationOutcome) -> tuple[BindingImpact, ...]:
        return tuple(i for i in self.impacts if i.outcome is outcome)

    @property
    def invalidated(self) -> tuple[BindingImpact, ...]:
        return self.of_outcome(InvalidationOutcome.INVALIDATES_ASSUMPTION)

    @property
    def needs_confirmation(self) -> tuple[BindingImpact, ...]:
        """Legitimate overrides a broader change reached. Confirm, do not correct."""
        return self.of_outcome(InvalidationOutcome.SCOPE_INHERITANCE_AFFECTED)


def propagate(
    delta: AssumptionDelta,
    index: CouplingIndex,
    *,
    as_of: dt.datetime | None = None,
) -> InvalidationReport:
    """Apply one delta to the index. Deterministic, and free of any model call."""
    effective = as_of if as_of is not None else index.as_of
    new_value = delta.new_value

    candidates = [
        b
        for b in index.bindings
        if b.key == delta.key
        and b.role in _INVALIDATABLE
        and scope_lattice.intersects(b.scope, delta.scope)
        and b.is_in_force(effective)
    ]

    superseded = _superseded_declaration(candidates, delta=delta)
    impacts: list[BindingImpact] = []

    for binding in sorted(candidates, key=lambda b: b.assumption_id):
        if superseded is not None and binding.assumption_id == superseded.assumption_id:
            impacts.append(
                BindingImpact(
                    binding=binding,
                    outcome=InvalidationOutcome.NOT_AFFECTED,
                    reason=("this is the declaration the change edits, not a consequence of it"),
                )
            )
            continue

        if new_value is None:
            # A key_removed delta invalidates every dependent binding: the value they rely
            # on ceases to exist, so there is nothing to compare against.
            impacts.append(
                _invalidate(
                    binding,
                    delta=delta,
                    as_of=effective,
                    superseded=superseded,
                    registry_version=index.registry_version,
                    reason="the key it binds is being removed",
                )
            )
            continue

        if is_blocked(binding.value, new_value):
            impacts.append(
                BindingImpact(
                    binding=binding,
                    outcome=InvalidationOutcome.NOT_AFFECTED,
                    reason=(
                        f"{binding.value.render()} and {new_value.render()} are not "
                        "comparable; coupling is blocked, not weakened"
                    ),
                )
            )
            continue

        if not are_divergent(binding.value, new_value):
            impacts.append(
                BindingImpact(
                    binding=binding,
                    outcome=InvalidationOutcome.NOT_AFFECTED,
                    reason="it already binds the value the change introduces",
                )
            )
            continue

        legitimate = _legitimate_override_of(binding, index, delta_scope=delta.scope)
        if legitimate is not None:
            impacts.append(
                BindingImpact(
                    binding=binding,
                    outcome=InvalidationOutcome.SCOPE_INHERITANCE_AFFECTED,
                    reason=(
                        f"{binding.scope.render()} legitimately governs its own scope on "
                        f"{binding.authority_class.value} authority; the override may still "
                        "be intended and requires confirmation rather than correction"
                    ),
                )
            )
            continue

        impacts.append(
            _invalidate(
                binding,
                delta=delta,
                as_of=effective,
                superseded=superseded,
                registry_version=index.registry_version,
                reason=(
                    f"it binds {binding.value.render()} where the change sets {new_value.render()}"
                ),
            )
        )

    affected_ids = {b.assumption_id for b in candidates}
    pre_existing = tuple(
        relation
        for relation in index.relations_of(Relationship.UNAUTHORIZED_OVERRIDE)
        if set(relation.input_record_ids) & affected_ids
    )

    return InvalidationReport(
        delta=delta,
        as_of=effective,
        impacts=tuple(impacts),
        pre_existing_unauthorized_overrides=pre_existing,
    )


def _invalidate(
    binding: ResolvedBinding,
    *,
    delta: AssumptionDelta,
    as_of: dt.datetime,
    superseded: ResolvedBinding | None,
    registry_version: str,
    reason: str,
) -> BindingImpact:
    """Build the ``invalidates_assumption`` relation for one affected binding.

    ``input_record_ids`` names the affected binding and, where one exists, the declaration
    the change supersedes. The delta is the other half of the derivation and is carried in
    ``detail`` because it is not an ``AssumptionRecord``.
    """
    inputs = [binding.assumption_id]
    if superseded is not None:
        inputs.append(superseded.assumption_id)
    ordered = tuple(sorted(set(inputs)))

    relation = DerivedRelation(
        relation_id=relation_id_for(
            rule_id=RuleId.DELTA_INVALIDATION,
            key=delta.key,
            scope=binding.scope.render(),
            input_ids=(*ordered, delta.delta_id),
        ),
        relationship=Relationship.INVALIDATES_ASSUMPTION,
        subject_ref=delta.delta_id,
        object_ref=binding.source_version_id,
        input_record_ids=ordered,
        provenance=DerivationProvenance(
            rule_id=RuleId.DELTA_INVALIDATION,
            rule_version=COUPLING_INDEX_VERSION,
            registry_version=registry_version,
            coupling_index_version=COUPLING_INDEX_VERSION,
            authority_model_version=AUTHORITY_MODEL_VERSION,
            detail={
                "assumption_key": delta.key,
                "delta_id": delta.delta_id,
                "delta_kind": delta.delta_kind.value,
                "delta_scope": delta.scope.render(),
                "binding_scope": binding.scope.render(),
                "as_of": as_of.isoformat(),
                "old_value": delta.old_value.render() if delta.old_value else None,
                "new_value": delta.new_value.render() if delta.new_value else None,
                "bound_value": binding.value.render(),
                "affected_artifact": binding.source_title,
                "affected_role": binding.role.value,
                "affected_authority_class": binding.authority_class.value,
                "reason": reason,
            },
        ),
    )
    return BindingImpact(
        binding=binding,
        outcome=InvalidationOutcome.INVALIDATES_ASSUMPTION,
        reason=reason,
        relation=relation,
    )


def _superseded_declaration(
    candidates: Sequence[ResolvedBinding], *, delta: AssumptionDelta
) -> ResolvedBinding | None:
    """The declaration this delta edits, if the corpus contains it.

    Matched on exact scope, ``declares`` role, and a value equal to the delta's
    ``old_value``. Deliberately strict: guessing which document a change edits would be a
    judgment, and a wrong guess suppresses a real finding.
    """
    if delta.old_value is None:
        return None
    for binding in sorted(candidates, key=lambda b: b.assumption_id):
        if (
            binding.role is BindingRole.DECLARES
            and scope_lattice.equals(binding.scope, delta.scope)
            and not is_blocked(binding.value, delta.old_value)
            and not are_divergent(binding.value, delta.old_value)
        ):
            return binding
    return None


def _legitimate_override_of(
    binding: ResolvedBinding,
    index: CouplingIndex,
    *,
    delta_scope: ScopeSelector,
) -> OverrideRecord | None:
    """Whether ``binding`` is a **legitimate** narrower override the delta reaches.

    Two conditions, and the second is the one an earlier draft got wrong.

    The delta must be reaching the binding from a **strictly broader** scope. A change at
    the override's own scope is editing the override itself, which is an ordinary
    invalidation - reporting "please confirm" for a document the author just changed would
    be noise.

    And the binding must actually **govern its own scope**, which is decided by
    ``resolve_authority``, not by the existence of *some* legitimate override record naming
    it. Those are not the same test: in the sample corpus the EMEA wiki page legitimately
    outranks another wiki page at ``global`` while remaining an unauthorized narrowing of
    the retention *policy*. Matching on the override record alone therefore told the reader
    to "confirm" a governance defect the resolver had already flagged, which is precisely
    the merge checkpoint 5.5 forbids.
    """
    if not scope_lattice.strictly_contains(delta_scope, binding.scope):
        return None

    resolution = index.authority_for(binding.key, binding.scope)
    if resolution is None or not resolution.is_resolved:
        return None
    governing = resolution.authoritative
    assert governing is not None  # guaranteed by is_resolved
    if governing.assumption_id != binding.assumption_id:
        return None

    for override in index.overrides:
        if override.narrower.assumption_id == binding.assumption_id:
            return override
    return None
