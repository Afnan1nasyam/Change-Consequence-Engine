"""Temporal authority resolution: ``resolve_authority(key, scope, as_of)``.

The one question this answers - *which artifact governs this variable, here, on this date* -
recurs throughout the system: version selection, staleness detection, contradictory
effective dates, retention reasoning. Checkpoint 5.3 and ``ARCHITECTURE.md`` 7.2 require it
to be a **single** domain service rather than logic re-derived per agent, and require
``Ambiguous`` to be a real return value.

**Newest-wins is forbidden, and the code contains no clock and no recency comparison.**
That is deliberate and worth stating plainly: nowhere in this module is
``effective_from`` compared between two candidates to break a tie. The only orderings used
are scope depth (specificity) and ``authority_class`` (governance). If those cannot
separate two in-force declarers with different values, the answer is ``AMBIGUOUS``. Picking
the most recent row would invent an authority the corpus does not establish, and would do
it invisibly - the failure mode is not that the answer is wrong, it is that nobody can tell
it was guessed.

The three outcomes, and why each is not an error:

``RESOLVED``
    One binding governs. Its scope is the narrowest that both contains the query scope and
    carries sufficient authority to narrow whatever sits above it.

``AMBIGUOUS``
    Two or more equally authoritative declarers are in force with divergent values. A
    finding *about the corpus*, reported for human decision.

``NO_DECLARER``
    Nothing declares this key at this scope on this date. Missing knowledge, not
    inconsistency - and never reported as a contradiction.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

from app.domain.entities.coupling import (
    AuthorityCandidate,
    AuthorityResolution,
    ResolvedBinding,
)
from app.domain.services import scope_lattice
from app.domain.services.value_comparison import compare
from app.domain.value_objects.enums import (
    AUTHORITY_MODEL_VERSION,
    AmbiguityReason,
    AuthorityOutcome,
    BindingRole,
    ValueComparison,
)
from app.domain.value_objects.rules import COUPLING_INDEX_VERSION
from app.domain.value_objects.scope import ScopeSelector


def governing_declarers(
    bindings: Iterable[ResolvedBinding],
    *,
    key: str,
    scope: ScopeSelector,
    as_of: dt.datetime,
) -> tuple[ResolvedBinding, ...]:
    """Declarers whose scope **governs** ``scope`` and which are in force at ``as_of``.

    "Governs" means contains-or-equals, not merely intersects: a binding at
    ``region:EU / tier:enterprise`` says nothing about ``region:EU`` as a whole, and a
    sibling scope says nothing at all. Using intersection here would let a narrow exception
    masquerade as the rule.
    """
    return tuple(
        b
        for b in bindings
        if b.key == key
        and b.role is BindingRole.DECLARES
        and scope_lattice.contains(b.scope, scope)
        and b.is_in_force(as_of)
    )


def resolve_authority(
    bindings: Iterable[ResolvedBinding],
    *,
    key: str,
    scope: ScopeSelector,
    as_of: dt.datetime,
) -> AuthorityResolution:
    """Which source version governs ``key`` at ``scope`` on ``as_of``.

    Resolution walks the scope lattice from the **narrowest** governing scope outward. At
    each level a candidate may only win if it matches or outranks every strictly broader
    governing candidate - that is the override legitimacy test (checkpoint 5.4.2), applied
    per candidate rather than per level, because one narrow scope can hold both a
    legitimate regional regulation and an unauthorized team wiki page.

    A candidate that narrows a higher authority is **flagged and skipped**, never applied
    and never discarded: resolution continues to the broader, higher-authority value, which
    is exactly what checkpoint 5.4.3 requires.
    """
    candidates = governing_declarers(bindings, key=key, scope=scope, as_of=as_of)
    if not candidates:
        return AuthorityResolution(
            key=key,
            scope=scope,
            as_of=as_of,
            outcome=AuthorityOutcome.NO_DECLARER,
            authority_model_version=AUTHORITY_MODEL_VERSION,
            coupling_index_version=COUPLING_INDEX_VERSION,
        )

    # One level per distinct governing scope. On a prefix lattice two scopes that both
    # contain the query scope and share a depth are necessarily the same scope, so depth
    # orders the levels unambiguously.
    depths = sorted({c.scope.depth for c in candidates}, reverse=True)
    narrowings: list[ResolvedBinding] = []

    for depth in depths:
        level = _at_depth(candidates, depth)
        broader = [c for c in candidates if c.scope.depth < depth]
        threshold = max((c.authority_rank for c in broader), default=None)

        if threshold is None:
            eligible, rejected = list(level), []
        else:
            eligible = [c for c in level if c.authority_rank >= threshold]
            rejected = [c for c in level if c.authority_rank < threshold]

        narrowings.extend(rejected)

        if not eligible:
            # Every candidate at this level narrows a higher authority. Fall through to the
            # broader level; the narrowing is reported separately as a governance defect.
            continue

        return _resolve_within_level(
            eligible,
            key=key,
            scope=scope,
            as_of=as_of,
            considered=candidates,
            narrowings=tuple(narrowings),
        )

    # Unreachable: the broadest level has no strictly broader candidates, so its threshold
    # is None and every candidate there is eligible.
    raise AssertionError(  # pragma: no cover
        f"authority resolution for {key}@{scope.render()} exhausted every scope level"
    )


def _at_depth(candidates: Sequence[ResolvedBinding], depth: int) -> list[ResolvedBinding]:
    # Sorted by assumption id so that a tie broken below is broken the same way on every
    # run, on every platform, regardless of store iteration order.
    return sorted(
        (c for c in candidates if c.scope.depth == depth),
        key=lambda c: c.assumption_id,
    )


def _resolve_within_level(
    level: Sequence[ResolvedBinding],
    *,
    key: str,
    scope: ScopeSelector,
    as_of: dt.datetime,
    considered: Sequence[ResolvedBinding],
    narrowings: tuple[ResolvedBinding, ...],
) -> AuthorityResolution:
    """Pick the governing binding among candidates at one scope, or declare ambiguity."""
    top_rank = max(c.authority_rank for c in level)
    top = [c for c in level if c.authority_rank == top_rank]

    winner: ResolvedBinding | None = None
    reason: AmbiguityReason | None = None

    if len(top) == 1:
        winner = top[0]
    else:
        # Equal authority, both in force, so their validity intervals overlap at `as_of` by
        # construction. Only agreement can save them - and no recency comparison happens.
        agreement = _agreement(top)
        if agreement is ValueComparison.EQUAL:
            # They say the same thing. Which record is returned cannot change the value, so
            # the lowest id is chosen purely for reproducibility.
            winner = top[0]
        elif agreement is ValueComparison.INCOMPARABLE:
            reason = AmbiguityReason.INCOMPARABLE_VALUES
        else:
            reason = AmbiguityReason.EQUAL_AUTHORITY_DIVERGENCE

    outcome = AuthorityOutcome.RESOLVED if winner is not None else AuthorityOutcome.AMBIGUOUS
    narrowing_ids = {n.assumption_id for n in narrowings}

    return AuthorityResolution(
        key=key,
        scope=scope,
        as_of=as_of,
        outcome=outcome,
        authoritative=winner,
        candidates=tuple(
            AuthorityCandidate(
                binding=c,
                selected=winner is not None and c.assumption_id == winner.assumption_id,
                unauthorized_narrowing_of=(
                    _broader_target(c, considered) if c.assumption_id in narrowing_ids else None
                ),
            )
            for c in sorted(considered, key=lambda b: b.assumption_id)
        ),
        ambiguity_reason=reason,
        unauthorized_narrowings=tuple(sorted(narrowings, key=lambda b: b.assumption_id)),
        authority_model_version=AUTHORITY_MODEL_VERSION,
        coupling_index_version=COUPLING_INDEX_VERSION,
    )


def _agreement(top: Sequence[ResolvedBinding]) -> ValueComparison:
    """Whether every top-authority candidate says the same thing.

    ``INCOMPARABLE`` dominates ``DIFFERENT``: if any pair cannot be compared at all, the
    ambiguity is about the values being incommensurable, which is a different human action
    from two comparable values disagreeing.
    """
    verdict = ValueComparison.EQUAL
    for index, left in enumerate(top):
        for right in top[index + 1 :]:
            result = compare(left.value, right.value)
            if result is ValueComparison.INCOMPARABLE:
                return ValueComparison.INCOMPARABLE
            if result is ValueComparison.DIFFERENT:
                verdict = ValueComparison.DIFFERENT
    return verdict


def _broader_target(narrower: ResolvedBinding, considered: Sequence[ResolvedBinding]) -> str | None:
    """The highest-authority strictly broader candidate this binding tried to narrow."""
    broader = [c for c in considered if c.scope.depth < narrower.scope.depth]
    if not broader:
        return None
    best = max(broader, key=lambda c: (c.authority_rank, c.assumption_id))
    return best.assumption_id
