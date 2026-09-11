"""The knowledge-decay report: sections, and the three-way separation it must preserve.

The report's whole value rests on never confusing these:

* **known inconsistency** - an artifact is wrong, and a rule says which;
* **known ambiguity** - the corpus cannot say which artifact governs;
* **missing knowledge** - nobody wrote it down.

So the schema refuses an inconsistency row with no ``rule_id``, refuses a missing-knowledge
row *with* one, and refuses an ambiguity row that does not say what would resolve it.
Absence is never reported as contradiction.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from app.assumptions.class_a import derive_class_a
from app.assumptions.coupling import build_coupling_index
from app.assumptions.decay import build_report
from app.domain.entities.coupling import ResolvedBinding
from app.domain.entities.decay import DecayEntry, KnowledgeDecayReport
from app.domain.entities.registry import KeyRegistryGap
from app.domain.value_objects.enums import (
    AuthorityClass,
    BindingRole,
    DecayCategory,
    RegistryGapStatus,
)
from app.domain.value_objects.rules import RuleId
from tests.fixtures.coupling import (
    REGISTRY_VERSION,
    TENANT,
    binding,
    days,
    decay_entry,
    usd,
)
from tests.fixtures.domain import model_provenance

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
CONSUMES = BindingRole.CONSUMES
DESCRIBES = BindingRole.DESCRIBES


def report_for(
    bindings: Sequence[ResolvedBinding],
    *,
    gaps: Sequence[KeyRegistryGap] | None = None,
) -> KnowledgeDecayReport:
    index = build_coupling_index(bindings, as_of=NOW, registry_version=REGISTRY_VERSION)
    findings = derive_class_a(
        index.relations,
        bindings=bindings,
        tenant_id=TENANT,
        registry_version=REGISTRY_VERSION,
    )
    return build_report(index, tenant_id=TENANT, findings=findings, gaps=gaps or [])


def gap(key: str = "refund.escalation_ceiling") -> KeyRegistryGap:
    return KeyRegistryGap(
        gap_id="rg_" + "0" * 32,
        proposed_key=key,
        proposed_value=usd(50000),
        evidence_id="ev_" + "1" * 32,
        holder_source_version_id="sv_" + "1" * 32,
        proposed_by=model_provenance(),
        as_of=NOW,
        registry_version=REGISTRY_VERSION,
        status=RegistryGapStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# Section A - stale consumers
# ---------------------------------------------------------------------------


def test_a_stale_consumer_is_reported_with_both_values() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    dashboard = binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG)
    report = report_for([policy, dashboard])

    rows = report.of_category(DecayCategory.STALE_CONSUMER)
    assert len(rows) == 1
    assert rows[0].rule_id == RuleId.STALE_CONSUMER
    values = {p.rendered_value for p in rows[0].participants}
    assert values == {"90d", "365d"}


def test_a_stale_consumer_row_names_the_effective_dates() -> None:
    policy = binding(
        value=days(90),
        authority=AuthorityClass.POLICY,
        effective_from=dt.datetime(2024, 7, 1, tzinfo=dt.UTC),
    )
    dashboard = binding(
        value=days(365),
        role=CONSUMES,
        authority=AuthorityClass.CONFIG,
        effective_from=dt.datetime(2025, 2, 15, tzinfo=dt.UTC),
    )
    row = report_for([policy, dashboard]).of_category(DecayCategory.STALE_CONSUMER)[0]
    assert all(p.effective_from is not None for p in row.participants)


def test_a_stale_consumer_is_a_known_inconsistency_not_an_ambiguity() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    dashboard = binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG)
    report = report_for([policy, dashboard])

    assert len(report.known_inconsistencies) == 1
    assert report.known_ambiguities == ()
    assert not report.of_category(DecayCategory.STALE_CONSUMER)[0].requires_human_decision


# ---------------------------------------------------------------------------
# Section B - unauthorized overrides
# ---------------------------------------------------------------------------


def test_an_unauthorized_override_is_reported_with_the_authority_comparison() -> None:
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    wiki = binding(scope="region:eu", value=days(30), authority=AuthorityClass.WIKI)
    rows = report_for([policy, wiki]).of_category(DecayCategory.UNAUTHORIZED_OVERRIDE)

    assert len(rows) == 1
    assert rows[0].rule_id == RuleId.UNAUTHORIZED_OVERRIDE
    assert "wiki" in rows[0].summary
    assert "policy" in rows[0].summary
    authorities = {p.authority_class for p in rows[0].participants}
    assert authorities == {AuthorityClass.WIKI, AuthorityClass.POLICY}


def test_a_legitimate_override_is_absent_from_the_report() -> None:
    """Reporting correct behaviour as a defect is how a report loses its reader."""
    policy = binding(scope="global", value=days(90), authority=AuthorityClass.POLICY)
    regulation = binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY)
    report = report_for([policy, regulation])

    assert report.of_category(DecayCategory.UNAUTHORIZED_OVERRIDE) == ()
    assert report.known_inconsistencies == ()


# ---------------------------------------------------------------------------
# Section C - contradictory declarations
# ---------------------------------------------------------------------------


def test_two_conflicting_declarers_produce_one_row_not_two() -> None:
    """One inconsistency in the corpus, even though both authors are told about it."""
    left = binding(value=days(90), authority=AuthorityClass.POLICY)
    right = binding(value=days(45), authority=AuthorityClass.STANDARD)
    report = report_for([left, right])

    rows = report.of_category(DecayCategory.CONTRADICTORY_DECLARATIONS)
    assert len(rows) == 1
    assert len(rows[0].participants) == 2


def test_a_contradiction_row_names_both_scopes_and_the_overlap() -> None:
    left = binding(value=days(90), authority=AuthorityClass.POLICY)
    right = binding(value=days(45), authority=AuthorityClass.STANDARD)
    row = report_for([left, right]).of_category(DecayCategory.CONTRADICTORY_DECLARATIONS)[0]
    assert row.scope.render() == "global"
    assert "overlapping validity" in row.summary


# ---------------------------------------------------------------------------
# Section D - ambiguous authority
# ---------------------------------------------------------------------------


def test_ambiguous_authority_names_no_winner() -> None:
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(45), authority=AuthorityClass.POLICY, tag="b")
    rows = report_for([left, right]).of_category(DecayCategory.AMBIGUOUS_AUTHORITY)

    assert len(rows) == 1
    assert rows[0].finding_id is None
    assert rows[0].rule_id is None
    assert "none has been chosen" in rows[0].summary


def test_ambiguous_authority_lists_the_candidates_and_the_resolution() -> None:
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(45), authority=AuthorityClass.POLICY, tag="b")
    row = report_for([left, right]).of_category(DecayCategory.AMBIGUOUS_AUTHORITY)[0]

    assert {p.rendered_value for p in row.participants} == {"90d", "45d"}
    assert row.requires_human_decision is True
    assert row.what_would_resolve is not None
    assert "will not choose" in row.what_would_resolve


def test_ambiguity_is_classified_as_ambiguity_not_inconsistency() -> None:
    left = binding(value=days(90), authority=AuthorityClass.POLICY, tag="a")
    right = binding(value=days(45), authority=AuthorityClass.POLICY, tag="b")
    report = report_for([left, right])
    assert len(report.known_ambiguities) == 1


def test_two_diverging_consumers_are_reported_as_ambiguity() -> None:
    """Neither declares, so neither can be said to be wrong."""
    left = binding(value=days(365), role=CONSUMES, tag="a")
    right = binding(value=days(180), role=CONSUMES, tag="b")
    rows = report_for([left, right]).of_category(DecayCategory.CONSUMER_DIVERGENCE)

    assert len(rows) == 1
    assert rows[0].rule_id is None
    assert rows[0].requires_human_decision is True
    assert rows[0] in report_for([left, right]).known_ambiguities


# ---------------------------------------------------------------------------
# Section E and the rest of missing knowledge
# ---------------------------------------------------------------------------


def test_a_registry_gap_is_reported_as_missing_knowledge() -> None:
    report = report_for([binding(value=days(90))], gaps=[gap()])
    rows = report.of_category(DecayCategory.REGISTRY_GAP)

    assert len(rows) == 1
    assert rows[0].rule_id is None
    assert rows[0].finding_id is None
    assert "quarantined" in rows[0].summary
    assert rows[0] in report.missing_knowledge


def test_a_registry_gap_states_the_growth_path() -> None:
    row = report_for([binding(value=days(90))], gaps=[gap()]).of_category(
        DecayCategory.REGISTRY_GAP
    )[0]
    assert "Class A" in row.summary


def test_a_single_holder_key_is_reported_without_a_rule() -> None:
    """Undocumented coupling. Not a defect and not a contradiction."""
    report = report_for([binding(value=days(90), authority=AuthorityClass.POLICY)])
    rows = report.of_category(DecayCategory.SINGLE_HOLDER)

    assert len(rows) == 1
    assert rows[0].rule_id is None
    assert "exactly one artifact" in rows[0].summary


def test_a_key_with_two_holders_is_not_a_single_holder() -> None:
    pool = [
        binding(value=days(90), authority=AuthorityClass.POLICY),
        binding(value=days(90), role=CONSUMES, authority=AuthorityClass.CONFIG),
    ]
    assert report_for(pool).of_category(DecayCategory.SINGLE_HOLDER) == ()


def test_a_key_declared_only_outside_governance_is_an_unwritten_rule() -> None:
    config = binding(value=days(90), authority=AuthorityClass.CONFIG)
    report = report_for([config])
    rows = report.of_category(DecayCategory.UNWRITTEN_RULE)

    assert len(rows) == 1
    assert "no policy, standard or runbook" in rows[0].summary
    assert rows[0].rule_id is None


def test_a_key_a_policy_declares_is_not_an_unwritten_rule() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    assert report_for([policy]).of_category(DecayCategory.UNWRITTEN_RULE) == ()


def test_a_key_only_consumed_is_not_an_unwritten_rule() -> None:
    """There is no rule to be unwritten - only a dependency with no source."""
    consumer = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.CONFIG)
    report = report_for([consumer])
    assert report.of_category(DecayCategory.UNWRITTEN_RULE) == ()
    assert len(report.of_category(DecayCategory.SINGLE_HOLDER)) == 1


# ---------------------------------------------------------------------------
# The three-way separation, enforced by the schema
# ---------------------------------------------------------------------------


def test_an_inconsistency_row_must_cite_its_rule() -> None:
    """Otherwise the report becomes a second source of truth."""
    with pytest.raises(ValidationError):
        decay_entry(category=DecayCategory.STALE_CONSUMER, rule_id=None)


def test_a_missing_knowledge_row_must_not_cite_a_rule() -> None:
    """Absence is recorded, never derived."""
    with pytest.raises(ValidationError):
        decay_entry(category=DecayCategory.SINGLE_HOLDER, rule_id=RuleId.STALE_CONSUMER)


def test_a_missing_knowledge_row_must_not_reference_a_finding() -> None:
    with pytest.raises(ValidationError):
        decay_entry(
            category=DecayCategory.REGISTRY_GAP,
            rule_id=None,
            finding_id="ca_" + "0" * 32,
        )


def test_an_ambiguity_row_must_state_what_would_resolve_it() -> None:
    with pytest.raises(ValidationError):
        decay_entry(
            category=DecayCategory.AMBIGUOUS_AUTHORITY,
            rule_id=None,
            requires_human_decision=True,
            what_would_resolve=None,
        )


def test_an_ambiguity_row_requires_a_human_decision_by_definition() -> None:
    with pytest.raises(ValidationError):
        decay_entry(
            category=DecayCategory.AMBIGUOUS_AUTHORITY,
            rule_id=None,
            requires_human_decision=False,
            what_would_resolve="pick one",
        )


def test_the_three_buckets_partition_every_row() -> None:
    pool = [
        binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
        binding(scope="global", value=days(3650), authority=AuthorityClass.WIKI),
        binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI),
        binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG),
        binding(key="api.max_page_size", value=usd(100), authority=AuthorityClass.CONFIG),
    ]
    report = report_for(pool, gaps=[gap()])

    total = (
        len(report.known_inconsistencies)
        + len(report.known_ambiguities)
        + len(report.missing_knowledge)
    )
    assert total == len(report.entries)
    assert report.entries


def test_every_category_falls_into_exactly_one_bucket() -> None:
    from app.domain.value_objects.enums import (
        KNOWN_AMBIGUITY,
        KNOWN_INCONSISTENCY,
        MISSING_KNOWLEDGE,
    )

    for category in DecayCategory:
        memberships = sum(
            category in bucket
            for bucket in (KNOWN_INCONSISTENCY, KNOWN_AMBIGUITY, MISSING_KNOWLEDGE)
        )
        assert memberships == 1, category


# ---------------------------------------------------------------------------
# Quiet on a quiet corpus
# ---------------------------------------------------------------------------


def test_an_empty_corpus_produces_a_quiet_report() -> None:
    report = report_for([])
    assert report.is_quiet
    assert report.entries == ()
    assert report.keys_examined == 0


def test_a_healthy_corpus_reports_only_missing_knowledge() -> None:
    """A key a policy declares and a consumer agrees with is not decay."""
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    consumer = binding(value=days(90), role=CONSUMES, authority=AuthorityClass.CONFIG)
    report = report_for([policy, consumer])

    assert report.known_inconsistencies == ()
    assert report.known_ambiguities == ()


def test_a_describes_binding_contributes_no_inconsistency() -> None:
    policy = binding(value=days(90), authority=AuthorityClass.POLICY)
    narration = binding(value=days(3650), role=DESCRIBES, authority=AuthorityClass.TICKET)
    report = report_for([policy, narration])
    assert report.known_inconsistencies == ()


# ---------------------------------------------------------------------------
# Shape and reproducibility
# ---------------------------------------------------------------------------


def _scenario() -> list[ResolvedBinding]:
    return [
        binding(scope="global", value=days(90), authority=AuthorityClass.POLICY),
        binding(scope="global", value=days(3650), authority=AuthorityClass.WIKI),
        binding(scope="region:eu", value=days(30), authority=AuthorityClass.REGULATORY),
        binding(scope="region:eu", value=days(45), authority=AuthorityClass.WIKI),
        binding(value=days(365), role=CONSUMES, authority=AuthorityClass.CONFIG),
    ]


def test_the_report_is_reproducible_from_the_same_inputs() -> None:
    first = report_for(_scenario(), gaps=[gap()])
    second = report_for(_scenario(), gaps=[gap()])
    assert first.model_dump() == second.model_dump()


def test_the_report_is_independent_of_input_order() -> None:
    forward = report_for(_scenario(), gaps=[gap()])
    backward = report_for(list(reversed(_scenario())), gaps=[gap()])
    assert forward.counts_by_category == backward.counts_by_category
    assert [e.summary for e in forward.entries] == [e.summary for e in backward.entries]


def test_the_report_pins_every_version_label() -> None:
    report = report_for(_scenario())
    assert report.registry_version == REGISTRY_VERSION
    assert report.coupling_index_version == "coupling-1"
    assert report.authority_model_version == "authority-1"
    assert report.as_of == NOW


def test_the_report_counts_what_it_examined() -> None:
    pool = _scenario()
    report = report_for(pool)
    assert report.bindings_examined == len(pool)
    assert report.keys_examined == 1


def test_counts_by_category_matches_the_entries() -> None:
    report = report_for(_scenario(), gaps=[gap()])
    assert sum(report.counts_by_category.values()) == len(report.entries)


def test_a_delta_invalidation_is_not_standing_decay() -> None:
    """The report answers "what is already wrong", not "what would this change break"."""
    entry = DecayEntry(
        category=DecayCategory.STALE_CONSUMER,
        key="retention.window",
        summary="x",
        rule_id=RuleId.STALE_CONSUMER,
    )
    assert entry.category in {DecayCategory.STALE_CONSUMER}
    assert RuleId.DELTA_INVALIDATION not in {
        e.rule_id for e in report_for(_scenario()).entries if e.rule_id
    }
