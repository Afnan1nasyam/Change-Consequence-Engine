"""`resolve_authority`, case by case.

The single most important test in this module is
:func:`test_newest_does_not_win_between_equal_authorities`. Every other correctness
property here could be satisfied by an implementation that quietly falls back to the most
recent row, and that implementation would look right on a corpus where recency happens to
correlate with authority - then invent an authority the corpus does not establish on one
where it does not.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pytest

from app.assumptions.temporal import governing_declarers, resolve_authority
from app.domain.entities.coupling import AuthorityResolution, ResolvedBinding
from app.domain.value_objects.enums import (
    AmbiguityReason,
    AuthorityClass,
    AuthorityOutcome,
    BindingRole,
)
from app.domain.value_objects.scope import ScopeSelector
from tests.fixtures.coupling import binding, days, score, usd

GLOBAL = ScopeSelector.global_scope()
EU = ScopeSelector.parse("region:eu")
US = ScopeSelector.parse("region:us")
EU_ENTERPRISE = ScopeSelector.parse("region:eu / tier:enterprise")

JAN = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
APR = dt.datetime(2025, 4, 1, tzinfo=dt.UTC)
JUL = dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
OCT = dt.datetime(2025, 10, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

KEY = "retention.window"


def resolve(
    bindings: Sequence[ResolvedBinding],
    *,
    scope: ScopeSelector = GLOBAL,
    as_of: dt.datetime = NOW,
) -> AuthorityResolution:
    return resolve_authority(bindings, key=KEY, scope=scope, as_of=as_of)


# ---------------------------------------------------------------------------
# One valid authority
# ---------------------------------------------------------------------------


def test_a_single_declarer_governs() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    result = resolve([policy])

    assert result.outcome is AuthorityOutcome.RESOLVED
    assert result.authoritative is not None
    assert result.authoritative.assumption_id == policy.assumption_id
    assert result.authoritative_value == days(90)
    assert result.ambiguity_reason is None


def test_the_resolution_records_the_versions_that_produced_it() -> None:
    """A finding must be explainable without re-running the resolver."""
    result = resolve([binding()])
    assert result.authority_model_version == "authority-1"
    assert result.coupling_index_version == "coupling-1"


def test_a_higher_authority_beats_a_lower_one_at_the_same_scope() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    wiki = binding(value=days(3650), authority=AuthorityClass.WIKI)
    result = resolve([policy, wiki])

    assert result.outcome is AuthorityOutcome.RESOLVED
    assert result.authoritative is not None
    assert result.authoritative.authority_class is AuthorityClass.POLICY


def test_only_declarers_are_candidates() -> None:
    """A consumer depends on the value; it does not get to state it."""
    consumer = binding(role=BindingRole.CONSUMES, value=days(365))
    describer = binding(role=BindingRole.DESCRIBES, value=days(30))
    assert resolve([consumer, describer]).outcome is AuthorityOutcome.NO_DECLARER

    declarer = binding(value=days(90))
    result = resolve([consumer, describer, declarer])
    assert result.authoritative is not None
    assert result.authoritative.role is BindingRole.DECLARES


# ---------------------------------------------------------------------------
# No declarer, and stale authority
# ---------------------------------------------------------------------------


def test_an_empty_pool_has_no_declarer() -> None:
    result = resolve([])
    assert result.outcome is AuthorityOutcome.NO_DECLARER
    assert result.authoritative is None
    assert result.candidates == ()


def test_a_stale_authority_is_excluded() -> None:
    """Expired means expired: the value is not resurrected for want of a replacement."""
    expired = binding(value=days(90), effective_from=JAN, effective_to=APR)
    result = resolve([expired], as_of=NOW)
    assert result.outcome is AuthorityOutcome.NO_DECLARER


def test_a_future_authority_is_excluded() -> None:
    future = binding(value=days(90), effective_from=NOW + dt.timedelta(days=1))
    assert resolve([future], as_of=NOW).outcome is AuthorityOutcome.NO_DECLARER


def test_a_stale_declarer_does_not_displace_the_live_one() -> None:
    expired = binding(value=days(30), effective_from=JAN, effective_to=APR, tag="old")
    live = binding(value=days(90), effective_from=APR, tag="new")
    result = resolve([expired, live], as_of=NOW)

    assert result.outcome is AuthorityOutcome.RESOLVED
    assert result.authoritative is not None
    assert result.authoritative.value == days(90)


def test_a_succession_at_the_boundary_produces_no_ambiguity() -> None:
    """Adjacent intervals: exactly one is in force at every instant."""
    first = binding(value=days(30), effective_from=JAN, effective_to=JUL, tag="v1")
    second = binding(value=days(90), effective_from=JUL, tag="v2")
    pool = [first, second]

    before = resolve(pool, as_of=APR)
    assert before.authoritative is not None
    assert before.authoritative.value == days(30)

    on_the_boundary = resolve(pool, as_of=JUL)
    assert on_the_boundary.outcome is AuthorityOutcome.RESOLVED
    assert on_the_boundary.authoritative is not None
    assert on_the_boundary.authoritative.value == days(90)


# ---------------------------------------------------------------------------
# Ambiguity - and the forbidden fallback
# ---------------------------------------------------------------------------


def test_equal_authority_with_divergent_values_is_ambiguous() -> None:
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(45), authority=AuthorityClass.POLICY, tag="b")
    result = resolve([left, right])

    assert result.outcome is AuthorityOutcome.AMBIGUOUS
    assert result.authoritative is None
    assert result.ambiguity_reason is AmbiguityReason.EQUAL_AUTHORITY_DIVERGENCE
    assert len(result.tied_candidates) == 2


def test_newest_does_not_win_between_equal_authorities() -> None:
    """Newest-wins is forbidden (checkpoint 5.3), and the fallback must not exist.

    Both orderings of the input are checked, so an implementation that happened to return
    the last row it visited fails too.
    """
    older = binding(value=days(90), authority=AuthorityClass.POLICY, effective_from=JAN, tag="a")
    newer = binding(value=days(45), authority=AuthorityClass.POLICY, effective_from=OCT, tag="b")

    for pool in ([older, newer], [newer, older]):
        result = resolve(pool)
        assert result.outcome is AuthorityOutcome.AMBIGUOUS, pool
        assert result.authoritative is None


def test_equal_authority_agreeing_is_not_ambiguous() -> None:
    """Two peers saying the same thing is corroboration, not a conflict."""
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(90), authority=AuthorityClass.POLICY, tag="b")
    result = resolve([left, right])

    assert result.outcome is AuthorityOutcome.RESOLVED
    assert result.authoritative_value == days(90)


def test_incomparable_values_produce_a_distinct_ambiguity_reason() -> None:
    """A unit mismatch and a genuine disagreement need different human actions."""
    left = binding(value=usd(10000), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=score(0.85), authority=AuthorityClass.POLICY, tag="b")
    result = resolve([left, right])

    assert result.outcome is AuthorityOutcome.AMBIGUOUS
    assert result.ambiguity_reason is AmbiguityReason.INCOMPARABLE_VALUES


def test_a_third_lower_authority_does_not_break_a_tie() -> None:
    """Ambiguity at the top is not resolved by anything below it."""
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(45), authority=AuthorityClass.POLICY, tag="b")
    wiki = binding(value=days(90), authority=AuthorityClass.WIKI, tag="c")

    assert resolve([left, right, wiki]).outcome is AuthorityOutcome.AMBIGUOUS


def test_an_ambiguous_resolution_still_lists_every_candidate() -> None:
    """The report hands the decision to a human, so it must show the options."""
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(45), authority=AuthorityClass.POLICY, tag="b")
    result = resolve([left, right])

    ids = {c.binding.assumption_id for c in result.candidates}
    assert ids == {left.assumption_id, right.assumption_id}


# ---------------------------------------------------------------------------
# Scope: inheritance, override, and the authority gate on it
# ---------------------------------------------------------------------------


def test_a_global_declaration_governs_a_narrower_scope() -> None:
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    result = resolve([policy], scope=EU)

    assert result.outcome is AuthorityOutcome.RESOLVED
    assert result.authoritative_value == days(90)


def test_a_narrower_declaration_does_not_govern_a_broader_scope() -> None:
    """An EU exception says nothing about the world."""
    eu_only = binding(value=days(30), scope="region:eu")
    assert resolve([eu_only], scope=GLOBAL).outcome is AuthorityOutcome.NO_DECLARER


def test_a_sibling_scope_is_not_a_candidate() -> None:
    us_only = binding(value=days(30), scope="region:us")
    assert resolve([us_only], scope=EU).outcome is AuthorityOutcome.NO_DECLARER


def test_a_sufficient_narrower_authority_governs_its_own_scope() -> None:
    """Checkpoint 5.4.2: regulatory >= policy, so the EU value wins inside the EU."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    regulation = binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY)

    inside = resolve([policy, regulation], scope=EU)
    assert inside.authoritative is not None
    assert inside.authoritative_value == days(30)
    assert inside.unauthorized_narrowings == ()

    outside = resolve([policy, regulation], scope=GLOBAL)
    assert outside.authoritative_value == days(90)


def test_a_peer_authority_narrower_override_is_legitimate() -> None:
    """The test is `>=`, so equal authority may specialize."""
    broad = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    narrow = binding(value=days(30), scope="region:eu", authority=AuthorityClass.POLICY)

    result = resolve([broad, narrow], scope=EU)
    assert result.authoritative_value == days(30)
    assert result.unauthorized_narrowings == ()


def test_an_insufficient_narrower_authority_is_flagged_and_skipped() -> None:
    """Checkpoint 5.4.3: the broader higher authority still governs, narrowing is flagged."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)

    result = resolve([policy, wiki], scope=EU)
    assert result.outcome is AuthorityOutcome.RESOLVED
    assert result.authoritative is not None
    assert result.authoritative.assumption_id == policy.assumption_id
    assert [b.assumption_id for b in result.unauthorized_narrowings] == [wiki.assumption_id]


def test_neither_binding_is_discarded_when_an_override_is_unauthorized() -> None:
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)

    result = resolve([policy, wiki], scope=EU)
    considered = {c.binding.assumption_id for c in result.candidates}
    assert considered == {policy.assumption_id, wiki.assumption_id}


def test_the_narrowing_flag_names_what_was_narrowed() -> None:
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)

    result = resolve([policy, wiki], scope=EU)
    flagged = next(c for c in result.candidates if c.binding.assumption_id == wiki.assumption_id)
    assert flagged.unauthorized_narrowing_of == policy.assumption_id


def test_the_authority_test_is_applied_per_candidate_not_per_level() -> None:
    """One scope can hold both a legitimate regulation and an unauthorized wiki page.

    Rejecting the whole level would lose the regulation; accepting it would bless the wiki.
    """
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    regulation = binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)

    result = resolve([policy, regulation, wiki], scope=EU)
    assert result.authoritative is not None
    assert result.authoritative.assumption_id == regulation.assumption_id
    assert [b.assumption_id for b in result.unauthorized_narrowings] == [wiki.assumption_id]


def test_resolution_walks_outward_through_several_scope_levels() -> None:
    """The deepest sufficiently authoritative scope governs."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    eu = binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY)
    tier = binding(
        value=days(15), scope="region:eu / tier:enterprise", authority=AuthorityClass.REGULATORY
    )

    pool = [policy, eu, tier]
    assert resolve(pool, scope=EU_ENTERPRISE).authoritative_value == days(15)
    assert resolve(pool, scope=EU).authoritative_value == days(30)
    assert resolve(pool, scope=GLOBAL).authoritative_value == days(90)


def test_an_unauthorized_deepest_level_falls_through_to_the_next() -> None:
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    eu = binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY)
    tier = binding(
        value=days(15), scope="region:eu / tier:enterprise", authority=AuthorityClass.WIKI
    )

    result = resolve([policy, eu, tier], scope=EU_ENTERPRISE)
    assert result.authoritative is not None
    assert result.authoritative.assumption_id == eu.assumption_id
    assert [b.assumption_id for b in result.unauthorized_narrowings] == [tier.assumption_id]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", [GLOBAL, EU, EU_ENTERPRISE])
def test_the_result_is_independent_of_input_order(scope: ScopeSelector) -> None:
    """Store iteration order must not be able to change a finding."""
    pool = [
        binding(value=days(90), scope="global", authority=AuthorityClass.POLICY),
        binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY),
        binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI),
        binding(value=days(3650), scope="global", authority=AuthorityClass.WIKI),
    ]
    forward = resolve(pool, scope=scope)
    backward = resolve(list(reversed(pool)), scope=scope)

    assert forward.outcome is backward.outcome
    assert forward.model_dump() == backward.model_dump()


def test_governing_declarers_excludes_a_wrong_key() -> None:
    right = binding(key="retention.window")
    wrong = binding(key="refund.approval_threshold", value=usd(10000))
    found = governing_declarers([right, wrong], key=KEY, scope=GLOBAL, as_of=NOW)
    assert [b.assumption_id for b in found] == [right.assumption_id]
