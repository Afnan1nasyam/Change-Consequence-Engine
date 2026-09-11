"""The checkpoint 3.3 role matrix and the 5.4 three-way rule, row by row.

The tests that matter most are the ones asserting **nothing** happens:

* a legitimate scoped override emits no finding, because reporting correct behaviour as a
  defect is the fastest way to lose a reader who can see it is correct;
* a ``describes`` binding never produces anything, at any authority, at any divergence;
* two ``consumes`` bindings disagreeing is not a contradiction;
* a unit mismatch blocks coupling rather than weakening it;
* a consumer that agrees with the *governing* declarer is not stale merely because some
  lower-authority document disagrees.

That last one is the false positive a naive pairwise reading of "declares vs consumes,
divergent" produces on the sample corpus, so it is pinned explicitly.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pytest

from app.assumptions.collisions import KeyCollisions, collide_key
from app.domain.entities.coupling import ResolvedBinding
from app.domain.value_objects.enums import AnomalyKind, AuthorityClass, BindingRole
from app.domain.value_objects.rules import RuleId
from app.domain.value_objects.taxonomy import DIRECTIONALITY, Directionality, Relationship
from tests.fixtures.coupling import REGISTRY_VERSION, binding, days, score, usd

JAN = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
APR = dt.datetime(2025, 4, 1, tzinfo=dt.UTC)
JUL = dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

KEY = "retention.window"

DECLARES = BindingRole.DECLARES
CONSUMES = BindingRole.CONSUMES
DESCRIBES = BindingRole.DESCRIBES


def collide(bindings: Sequence[ResolvedBinding], *, as_of: dt.datetime = NOW) -> KeyCollisions:
    return collide_key(bindings, key=KEY, as_of=as_of, registry_version=REGISTRY_VERSION)


# ---------------------------------------------------------------------------
# declares vs declares, same scope => conflicts_with
# ---------------------------------------------------------------------------


def test_two_declarers_at_the_same_scope_with_divergent_values_conflict() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    standard = binding(value=days(45), authority=AuthorityClass.STANDARD)
    result = collide([policy, standard])

    conflicts = result.relations_of(Relationship.CONFLICTS_WITH)
    assert len(conflicts) == 1
    assert set(conflicts[0].input_record_ids) == {policy.assumption_id, standard.assumption_id}


def test_a_conflict_is_symmetric_and_stored_once() -> None:
    """A symmetric edge emitted twice would double-count every contradiction."""
    left = binding(value=days(90), tag="a")
    right = binding(value=days(45), tag="b")

    forward = collide([left, right]).relations_of(Relationship.CONFLICTS_WITH)
    backward = collide([right, left]).relations_of(Relationship.CONFLICTS_WITH)

    assert len(forward) == len(backward) == 1
    assert forward[0].relation_id == backward[0].relation_id
    assert DIRECTIONALITY[Relationship.CONFLICTS_WITH] is Directionality.SYMMETRIC


def test_three_divergent_declarers_produce_three_pairwise_conflicts() -> None:
    pool = [
        binding(value=days(90), authority=AuthorityClass.POLICY),
        binding(value=days(45), authority=AuthorityClass.STANDARD),
        binding(value=days(3650), authority=AuthorityClass.WIKI),
    ]
    assert len(collide(pool).relations_of(Relationship.CONFLICTS_WITH)) == 3


def test_agreeing_declarers_produce_nothing() -> None:
    """Corroboration, not conflict."""
    left = binding(value=days(90), tag="a")
    right = binding(value=days(90), tag="b")
    assert collide([left, right]).relations == ()


def test_a_conflict_records_both_values_and_both_authorities() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    wiki = binding(value=days(3650), authority=AuthorityClass.WIKI)
    relation = collide([policy, wiki]).relations_of(Relationship.CONFLICTS_WITH)[0]

    detail = relation.provenance.detail
    assert {detail["subject_value"], detail["object_value"]} == {"90d", "3650d"}
    assert {detail["subject_authority_class"], detail["object_authority_class"]} == {
        "policy",
        "wiki",
    }
    assert detail["scope"] == "global"


# ---------------------------------------------------------------------------
# declares vs declares, narrower scope => override, gated by authority
# ---------------------------------------------------------------------------


def test_a_sufficient_narrower_authority_emits_no_finding() -> None:
    """Checkpoint 5.4.2. The override is recorded, never reported."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    regulation = binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY)
    result = collide([policy, regulation])

    assert result.relations == ()
    assert len(result.overrides) == 1
    assert result.overrides[0].narrower.assumption_id == regulation.assumption_id
    assert result.overrides[0].broader.assumption_id == policy.assumption_id


def test_a_peer_authority_narrower_override_is_legitimate() -> None:
    """The rule is `>=`, so equal authority may specialize."""
    broad = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    narrow = binding(value=days(30), scope="region:eu", authority=AuthorityClass.POLICY)
    result = collide([broad, narrow])

    assert result.relations == ()
    assert len(result.overrides) == 1


def test_an_insufficient_narrower_authority_emits_unauthorized_override() -> None:
    """Checkpoint 5.4.3."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)
    result = collide([policy, wiki])

    overrides = result.relations_of(Relationship.UNAUTHORIZED_OVERRIDE)
    assert len(overrides) == 1
    assert result.overrides == ()


def test_the_unauthorized_override_edge_points_narrower_to_broader() -> None:
    """The narrower artifact is the one deviating, so it is the subject."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)
    relation = collide([policy, wiki]).relations_of(Relationship.UNAUTHORIZED_OVERRIDE)[0]

    assert relation.subject_ref == wiki.source_version_id
    assert relation.object_ref == policy.source_version_id
    assert DIRECTIONALITY[Relationship.UNAUTHORIZED_OVERRIDE] is Directionality.DIRECTED


def test_the_unauthorized_override_records_the_authority_comparison() -> None:
    """Recorded, not merely applied: a reviewer must audit without re-running."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)
    relation = collide([policy, wiki]).relations_of(Relationship.UNAUTHORIZED_OVERRIDE)[0]

    detail = relation.provenance.detail
    assert detail["narrower_authority_class"] == "wiki"
    assert detail["broader_authority_class"] == "policy"
    assert detail["narrower_authority_rank"] < detail["broader_authority_rank"]
    assert detail["narrower_scope"] == "region:eu"
    assert detail["broader_scope"] == "global"
    assert detail["temporal_overlap"] is True
    assert relation.provenance.authority_model_version == "authority-1"
    assert relation.provenance.registry_version == REGISTRY_VERSION


def test_both_bindings_survive_an_unauthorized_override() -> None:
    """Neither is discarded, suppressed, or rewritten (checkpoint 5.4.3)."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI)
    relation = collide([policy, wiki]).relations_of(Relationship.UNAUTHORIZED_OVERRIDE)[0]

    assert set(relation.input_record_ids) == {policy.assumption_id, wiki.assumption_id}


def test_a_narrower_override_that_agrees_produces_nothing() -> None:
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    wiki = binding(value=days(90), scope="region:eu", authority=AuthorityClass.WIKI)
    result = collide([policy, wiki])
    assert result.relations == ()
    assert result.overrides == ()


# ---------------------------------------------------------------------------
# Disjoint scope and non-overlapping time
# ---------------------------------------------------------------------------


def test_disjoint_scopes_are_not_coupled_for_contradiction() -> None:
    """Two regions may differ freely; that is what regions are for."""
    eu = binding(value=days(30), scope="region:eu")
    us = binding(value=days(90), scope="region:us")
    result = collide([eu, us])
    assert result.relations == ()
    assert result.overrides == ()


def test_non_overlapping_validity_means_superseded_not_conflicting() -> None:
    first = binding(value=days(30), effective_from=JAN, effective_to=APR, tag="v1")
    second = binding(value=days(90), effective_from=APR, tag="v2")
    # Only the second is in force at NOW, so there is no pair to compare at all.
    assert collide([first, second], as_of=NOW).relations == ()


def test_adjacent_intervals_in_force_together_cannot_happen() -> None:
    """Half-open intervals mean at most one of a clean succession is ever live."""
    first = binding(value=days(30), effective_from=JAN, effective_to=JUL, tag="v1")
    second = binding(value=days(90), effective_from=JUL, tag="v2")

    at_boundary = collide([first, second], as_of=JUL)
    assert at_boundary.relations == ()
    live = [h for h in (first, second) if h.is_in_force(JUL)]
    assert len(live) == 1


# ---------------------------------------------------------------------------
# The unit gate
# ---------------------------------------------------------------------------


def test_a_unit_mismatch_blocks_coupling_and_emits_no_finding() -> None:
    """Checkpoint 5.6. Blocked, not weakened."""
    left = binding(value=usd(10000), tag="a")
    right = binding(value=score(0.85), tag="b")
    result = collide([left, right])

    assert result.relations == ()
    kinds = {a.kind for a in result.anomalies}
    assert AnomalyKind.BLOCKED_BY_UNIT_MISMATCH in kinds


def test_the_blocked_pair_is_reported_rather_than_silently_dropped() -> None:
    left = binding(value=usd(10000), tag="a")
    right = binding(value=score(0.85), tag="b")
    anomaly = next(
        a
        for a in collide([left, right]).anomalies
        if a.kind is AnomalyKind.BLOCKED_BY_UNIT_MISMATCH
    )
    assert set(anomaly.participants) == {left.assumption_id, right.assumption_id}
    assert "not" in anomaly.detail


# ---------------------------------------------------------------------------
# One document disagreeing with itself
# ---------------------------------------------------------------------------


def _same_document(
    left: ResolvedBinding, right: ResolvedBinding
) -> tuple[ResolvedBinding, ResolvedBinding]:
    """Re-point ``right`` at ``left``'s source version, keeping its own value and id."""
    return left, right.model_copy(
        update={
            "record": right.record.model_copy(
                update={"holder_source_version_id": left.source_version_id}
            ),
            "source_id": left.source_id,
            "source_title": left.source_title,
        }
    )


def test_two_divergent_declarations_in_one_source_are_not_a_contradiction() -> None:
    """A single source must not be able to manufacture a cross-source conflict.

    Security property, not tidiness: an injected document that states a value twice would
    otherwise appear in the highest-trust tier as two independent authorities disagreeing
    (``docs/SECURITY.md``).
    """
    left, right = _same_document(binding(value=days(90), tag="a"), binding(value=days(45), tag="b"))
    result = collide([left, right])

    assert result.relations_of(Relationship.CONFLICTS_WITH) == ()
    assert {a.kind for a in result.anomalies} == {AnomalyKind.INTRA_DOCUMENT_DIVERGENCE}


def test_the_intra_document_anomaly_is_reported_not_suppressed() -> None:
    left, right = _same_document(binding(value=days(90), tag="a"), binding(value=days(45), tag="b"))
    anomaly = collide([left, right]).anomalies[0]

    assert set(anomaly.participants) == {left.assumption_id, right.assumption_id}
    assert "90d" in anomaly.detail
    assert "45d" in anomaly.detail
    assert "not two" in anomaly.detail


def test_the_same_values_from_two_documents_still_conflict() -> None:
    """The rule keys on source version, not on similarity: two real sources still clash."""
    left = binding(value=days(90), tag="a")
    right = binding(value=days(45), tag="b")
    assert left.source_version_id != right.source_version_id
    assert len(collide([left, right]).relations_of(Relationship.CONFLICTS_WITH)) == 1


def test_an_intra_document_divergence_emits_no_derived_relation() -> None:
    """No edge in the taxonomy fits it, and inventing one would be the catch-all."""
    left, right = _same_document(binding(value=days(90), tag="a"), binding(value=days(45), tag="b"))
    assert collide([left, right]).relations == ()


# ---------------------------------------------------------------------------
# declares vs consumes => staleness, against resolved authority
# ---------------------------------------------------------------------------


def test_a_consumer_diverging_from_the_governing_declarer_is_stale() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    dashboard = binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG)
    result = collide([policy, dashboard])

    stale = result.relations_of(Relationship.INVALIDATES_ASSUMPTION)
    assert len(stale) == 1
    assert stale[0].provenance.rule_id == RuleId.STALE_CONSUMER
    assert stale[0].object_ref == dashboard.source_version_id
    assert stale[0].subject_ref == policy.source_version_id


def test_a_consumer_agreeing_with_the_governing_declarer_is_not_stale() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    deck = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.TRAINING)
    assert collide([policy, deck]).relations_of(Relationship.INVALIDATES_ASSUMPTION) == ()


def test_a_consumer_is_not_stale_against_a_non_governing_declarer() -> None:
    """The false positive a naive pairwise reading produces.

    The deck follows the policy that actually governs. A lower-authority standard
    disagreeing does not make the deck wrong - it makes the standard wrong, which is
    reported separately as a same-scope contradiction.
    """
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    standard = binding(value=days(45), authority=AuthorityClass.STANDARD)
    deck = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.TRAINING)

    result = collide([policy, standard, deck])
    assert result.relations_of(Relationship.INVALIDATES_ASSUMPTION) == ()
    assert len(result.relations_of(Relationship.CONFLICTS_WITH)) == 1


def test_ambiguous_authority_produces_no_staleness_finding() -> None:
    """If the corpus cannot say what the value is, no consumer can be shown wrong."""
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(45), authority=AuthorityClass.POLICY, tag="b")
    consumer = binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG)

    result = collide([left, right, consumer])
    assert result.relations_of(Relationship.INVALIDATES_ASSUMPTION) == ()
    assert not result.authority["global"].is_resolved


def test_a_consumer_with_no_declarer_at_all_is_not_stale() -> None:
    """Absence is not inconsistency."""
    consumer = binding(value=days(365), role=CONSUMES)
    assert collide([consumer]).relations == ()


def test_staleness_is_evaluated_at_the_consumers_own_scope() -> None:
    """An EU consumer is judged against the EU authority, not the global one."""
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    regulation = binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY)
    eu_consumer = binding(
        value=days(30), scope="region:eu", role=CONSUMES, authority=AuthorityClass.CONFIG
    )

    result = collide([policy, regulation, eu_consumer])
    assert result.relations_of(Relationship.INVALIDATES_ASSUMPTION) == ()


def test_an_eu_consumer_following_the_global_value_is_stale_under_a_valid_override() -> None:
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    regulation = binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY)
    eu_consumer = binding(
        value=days(90), scope="region:eu", role=CONSUMES, authority=AuthorityClass.CONFIG
    )

    result = collide([policy, regulation, eu_consumer])
    stale = result.relations_of(Relationship.INVALIDATES_ASSUMPTION)
    assert len(stale) == 1
    assert stale[0].object_ref == eu_consumer.source_version_id


def test_the_staleness_finding_records_both_values_and_the_authority() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    dashboard = binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG)
    relation = collide([policy, dashboard]).relations_of(Relationship.INVALIDATES_ASSUMPTION)[0]

    detail = relation.provenance.detail
    assert detail["authoritative_value"] == "90d"
    assert detail["consumed_value"] == "365d"
    assert detail["authoritative_authority_class"] == "policy"
    assert detail["stale_role"] == "consumes"


# ---------------------------------------------------------------------------
# consumes vs consumes => an anomaly, never a contradiction
# ---------------------------------------------------------------------------


def test_two_diverging_consumers_are_not_a_contradiction() -> None:
    """Checkpoint 3.3: neither declares, so neither can be said to be wrong."""
    left = binding(value=days(365), role=CONSUMES, tag="a")
    right = binding(value=days(180), role=CONSUMES, tag="b")
    result = collide([left, right])

    assert result.relations == ()
    assert {a.kind for a in result.anomalies} == {AnomalyKind.CONSUMER_DIVERGENCE}


def test_the_consumer_anomaly_names_both_participants() -> None:
    left = binding(value=days(365), role=CONSUMES, tag="a")
    right = binding(value=days(180), role=CONSUMES, tag="b")
    anomaly = collide([left, right]).anomalies[0]
    assert set(anomaly.participants) == {left.assumption_id, right.assumption_id}
    assert "scope review" in anomaly.detail


def test_agreeing_consumers_produce_no_anomaly() -> None:
    left = binding(value=days(365), role=CONSUMES, tag="a")
    right = binding(value=days(365), role=CONSUMES, tag="b")
    assert collide([left, right]).anomalies == ()


def test_no_edge_type_exists_for_consumer_divergence() -> None:
    """Inventing one would recreate the catch-all the taxonomy excludes."""
    left = binding(value=days(365), role=CONSUMES, tag="a")
    right = binding(value=days(180), role=CONSUMES, tag="b")
    assert collide([left, right]).relations == ()


# ---------------------------------------------------------------------------
# describes => nothing, ever
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("other_role", [DECLARES, CONSUMES, DESCRIBES])
def test_a_describes_binding_never_produces_a_finding(other_role: BindingRole) -> None:
    """Historical narration does not become wrong when the value moves."""
    describer = binding(value=days(3650), role=DESCRIBES, authority=AuthorityClass.TICKET)
    other = binding(value=days(90), role=other_role, tag="other")
    result = collide([describer, other])

    assert result.relations == ()
    assert result.overrides == ()
    assert result.anomalies == ()


def test_a_describes_binding_at_a_narrower_scope_is_not_an_override() -> None:
    policy = binding(value=days(90), scope="global", authority=AuthorityClass.POLICY)
    narration = binding(
        value=days(3650), scope="region:eu", role=DESCRIBES, authority=AuthorityClass.TICKET
    )
    result = collide([policy, narration])
    assert result.relations == ()
    assert result.overrides == ()


def test_a_describes_binding_is_not_an_authority_candidate() -> None:
    describer = binding(value=days(3650), role=DESCRIBES)
    assert not collide([describer]).authority["global"].is_resolved


# ---------------------------------------------------------------------------
# Shape and determinism
# ---------------------------------------------------------------------------


def test_no_derived_relation_carries_direct_evidence() -> None:
    """Type C support is its rule and its inputs; a citation here would be fabricated."""
    pool = [
        binding(value=days(90), authority=AuthorityClass.POLICY),
        binding(value=days(45), authority=AuthorityClass.STANDARD),
        binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG),
        binding(value=days(30), scope="region:eu", authority=AuthorityClass.WIKI),
    ]
    relations = collide(pool).relations
    assert relations
    for relation in relations:
        assert relation.evidence_ids == ()
        assert relation.provenance.rule_id
        assert len(relation.input_record_ids) >= 1


def test_every_relation_pins_the_versions_that_produced_it() -> None:
    pool = [binding(value=days(90), tag="a"), binding(value=days(45), tag="b")]
    for relation in collide(pool).relations:
        assert relation.provenance.registry_version == REGISTRY_VERSION
        assert relation.provenance.coupling_index_version == "coupling-1"
        assert relation.provenance.rule_version == "coupling-1"


def test_collision_output_is_independent_of_input_order() -> None:
    pool = [
        binding(value=days(90), scope="global", authority=AuthorityClass.POLICY),
        binding(value=days(30), scope="region:eu", authority=AuthorityClass.REGULATORY),
        binding(value=days(45), scope="region:eu", authority=AuthorityClass.WIKI),
        binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG),
        binding(value=days(3650), scope="global", authority=AuthorityClass.WIKI),
    ]
    forward = collide(pool)
    backward = collide(list(reversed(pool)))

    assert [r.relation_id for r in forward.relations] == [r.relation_id for r in backward.relations]


def test_an_empty_pool_produces_an_empty_result() -> None:
    result = collide([])
    assert result.relations == ()
    assert result.overrides == ()
    assert result.anomalies == ()
    assert result.authority == {}


def test_a_different_key_is_ignored() -> None:
    right = binding(key=KEY, value=days(90))
    wrong = binding(key="refund.approval_threshold", value=usd(10000))
    assert collide([right, wrong]).relations == ()
