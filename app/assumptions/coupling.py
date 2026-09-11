"""The coupling index: a materialized, deterministic projection of the assumption store.

``ARCHITECTURE.md`` 7.1 states the two load-bearing properties, and both are structural
here rather than aspirational:

**The index is keyed on the triple ``(key, scope, as_of)``.** Keying on ``key`` alone is
the defect behind the removed ``has_collision := distinct_values > 1`` rule: it is both
scope-blind and authority-blind, so it reports every legitimately scoped override as a
contradiction and floods the highest-trust output with false positives. The sample corpus
alone would produce four such false positives on ``retention.window``.

**Nothing here is written by an agent.** ``build_coupling_index`` takes bindings and a date
and returns a value. There is no mutation path, no agent parameter, and no model call - not
even an optional one. The index is rebuilt from the store, which is what makes it a
projection instead of a second source of truth.

Membership of the set at ``(key, scope, as_of)`` is every admitted binding of that key
whose scope is **comparable** with ``scope`` (equal, containing, or contained) and whose
validity interval contains ``as_of``. Comparability rather than equality is what makes
inheritance work: the set at ``region:EU`` legitimately contains the ``global`` declaration
that governs it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.assumptions.collisions import KeyCollisions, collide_key
from app.domain.entities.assumption import CouplingHolder, CouplingSet
from app.domain.entities.claims import DerivedRelation
from app.domain.entities.coupling import (
    AuthorityResolution,
    CouplingAnomaly,
    OverrideRecord,
    ResolvedBinding,
)
from app.domain.services import scope_lattice
from app.domain.value_objects.enums import (
    AUTHORITY_MODEL_VERSION,
    AuthorityOutcome,
    CouplingOutcome,
)
from app.domain.value_objects.rules import COUPLING_INDEX_VERSION
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship

#: Outcome precedence for one index cell, strongest signal first.
#:
#: A same-scope contradiction outranks everything because two peers disagreeing is the
#: least ambiguous defect in the corpus. An unauthorized override outranks ambiguity
#: because it names a specific artifact to fix, where ambiguity names only a decision to
#: make. ``LEGITIMATE_OVERRIDE`` sits above ``CONSISTENT`` so the cell still records that a
#: scoped specialization exists - it emits no finding, but a reader should see it.
_OUTCOME_PRECEDENCE: tuple[CouplingOutcome, ...] = (
    CouplingOutcome.CONFLICTS_WITH,
    CouplingOutcome.UNAUTHORIZED_OVERRIDE,
    CouplingOutcome.AMBIGUOUS_TEMPORAL,
    CouplingOutcome.LEGITIMATE_OVERRIDE,
    CouplingOutcome.CONSISTENT,
)


@dataclass(frozen=True)
class CouplingIndex:
    """The whole projection for one tenant at one ``as_of``.

    Reproducible from the same store state and the same version labels. Two builds over
    identical inputs are equal, which is what lets a diff between runs be attributed to a
    registry expansion or an authority-ordering change rather than to iteration order.
    """

    as_of: dt.datetime
    registry_version: str
    coupling_index_version: str = COUPLING_INDEX_VERSION
    authority_model_version: str = AUTHORITY_MODEL_VERSION

    bindings: tuple[ResolvedBinding, ...] = ()
    sets: tuple[CouplingSet, ...] = ()
    collisions: dict[str, KeyCollisions] = field(default_factory=dict)

    # -- lookups ----------------------------------------------------------

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.collisions))

    def set_for(self, key: str, scope: ScopeSelector) -> CouplingSet | None:
        for candidate in self.sets:
            if candidate.key == key and candidate.scope.path == scope.path:
                return candidate
        return None

    def authority_for(self, key: str, scope: ScopeSelector) -> AuthorityResolution | None:
        collisions = self.collisions.get(key)
        if collisions is None:
            return None
        return collisions.authority.get(scope.render())

    @property
    def relations(self) -> tuple[DerivedRelation, ...]:
        """Every derived relation, in a stable order."""
        found = [r for c in self.collisions.values() for r in c.relations]
        return tuple(sorted(found, key=lambda r: r.relation_id))

    def relations_of(self, relationship: Relationship) -> tuple[DerivedRelation, ...]:
        return tuple(r for r in self.relations if r.relationship is relationship)

    @property
    def overrides(self) -> tuple[OverrideRecord, ...]:
        """Legitimate overrides. Recorded, never reported as findings."""
        found = [o for c in self.collisions.values() for o in c.overrides]
        return tuple(sorted(found, key=lambda o: (o.key, o.narrower.assumption_id)))

    @property
    def anomalies(self) -> tuple[CouplingAnomaly, ...]:
        found = [a for c in self.collisions.values() for a in c.anomalies]
        return tuple(sorted(found, key=lambda a: (a.key, a.participants)))

    @property
    def ambiguous_resolutions(self) -> tuple[AuthorityResolution, ...]:
        found = [
            resolution
            for collisions in self.collisions.values()
            for resolution in collisions.authority.values()
            if resolution.outcome is AuthorityOutcome.AMBIGUOUS
        ]
        return tuple(sorted(found, key=lambda r: (r.key, r.scope.render())))

    @property
    def binding_count(self) -> int:
        return len(self.bindings)


def members_of(
    bindings: Iterable[ResolvedBinding],
    *,
    key: str,
    scope: ScopeSelector,
    as_of: dt.datetime,
) -> tuple[ResolvedBinding, ...]:
    """Membership of the coupling set at ``(key, scope, as_of)`` (checkpoint 5.1).

    Comparable scope, not equal scope: a ``global`` declaration governs ``region:EU``, so it
    belongs to the EU set. Siblings are excluded - ``region:US`` says nothing about EU.
    """
    return tuple(
        sorted(
            (
                b
                for b in bindings
                if b.key == key
                and scope_lattice.comparable(b.scope, scope)
                and b.is_in_force(as_of)
            ),
            key=lambda b: b.assumption_id,
        )
    )


def build_coupling_set(
    bindings: Iterable[ResolvedBinding],
    *,
    key: str,
    scope: ScopeSelector,
    as_of: dt.datetime,
    collisions: KeyCollisions,
) -> CouplingSet:
    """One index cell, with its holders and its summarized outcome."""
    members = members_of(bindings, key=key, scope=scope, as_of=as_of)
    return CouplingSet(
        key=key,
        scope=scope,
        as_of=as_of,
        holders=tuple(
            CouplingHolder(
                source_version_id=b.source_version_id,
                assumption_id=b.assumption_id,
                value=b.value,
                role=b.role,
                authority_class=b.authority_class,
                scope=b.scope,
            )
            for b in members
        ),
        outcome=_summarize(members, scope=scope, collisions=collisions),
        authority_model_version=AUTHORITY_MODEL_VERSION,
        coupling_index_version=COUPLING_INDEX_VERSION,
        computed_at=as_of,
    )


def build_coupling_index(
    bindings: Iterable[ResolvedBinding],
    *,
    as_of: dt.datetime,
    registry_version: str,
) -> CouplingIndex:
    """Rebuild the whole index. Deterministic, and free of any model call.

    One cell is produced per ``(key, observed scope)`` pair. Scopes are taken from the
    bindings themselves rather than enumerated from the registry's ``allowed_scopes``: a
    cell for a scope nothing binds would be an empty row, and the index is a projection of
    what the corpus says, not of what it could have said.
    """
    pool = tuple(sorted(bindings, key=lambda b: b.assumption_id))

    collisions: dict[str, KeyCollisions] = {}
    sets: list[CouplingSet] = []

    for key in sorted({b.key for b in pool}):
        key_collisions = collide_key(pool, key=key, as_of=as_of, registry_version=registry_version)
        collisions[key] = key_collisions

        for scope in _scopes_for(pool, key=key, as_of=as_of):
            sets.append(
                build_coupling_set(
                    pool,
                    key=key,
                    scope=scope,
                    as_of=as_of,
                    collisions=key_collisions,
                )
            )

    return CouplingIndex(
        as_of=as_of,
        registry_version=registry_version,
        bindings=pool,
        sets=tuple(sets),
        collisions=collisions,
    )


def _scopes_for(
    pool: Sequence[ResolvedBinding], *, key: str, as_of: dt.datetime
) -> tuple[ScopeSelector, ...]:
    seen: dict[str, ScopeSelector] = {}
    for binding in pool:
        if binding.key == key and binding.is_in_force(as_of):
            seen.setdefault(binding.scope.render(), binding.scope)
    return tuple(seen[name] for name in sorted(seen))


def _summarize(
    members: Sequence[ResolvedBinding],
    *,
    scope: ScopeSelector,
    collisions: KeyCollisions,
) -> CouplingOutcome:
    """The strongest outcome affecting this cell's holders.

    Only relations whose *every* input record is a member of this cell are counted. A
    contradiction between two bindings that both sit outside this scope is not this cell's
    problem, and attributing it here would make one defect appear in several places.
    """
    member_ids = {b.assumption_id for b in members}
    observed: set[CouplingOutcome] = {CouplingOutcome.CONSISTENT}

    for relation in collisions.relations:
        if not set(relation.input_record_ids) <= member_ids:
            continue
        if relation.relationship is Relationship.CONFLICTS_WITH:
            observed.add(CouplingOutcome.CONFLICTS_WITH)
        elif relation.relationship is Relationship.UNAUTHORIZED_OVERRIDE:
            observed.add(CouplingOutcome.UNAUTHORIZED_OVERRIDE)

    for override in collisions.overrides:
        if {override.narrower.assumption_id, override.broader.assumption_id} <= member_ids:
            observed.add(CouplingOutcome.LEGITIMATE_OVERRIDE)

    resolution = collisions.authority.get(scope.render())
    if resolution is not None and resolution.outcome is AuthorityOutcome.AMBIGUOUS:
        observed.add(CouplingOutcome.AMBIGUOUS_TEMPORAL)

    for outcome in _OUTCOME_PRECEDENCE:
        if outcome in observed:
            return outcome
    raise AssertionError("CONSISTENT is always present")  # pragma: no cover
