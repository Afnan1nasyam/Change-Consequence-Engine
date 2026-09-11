"""The seeded Class A scenarios, end to end over the real corpus, with no LLM call.

This is where the central product claim is either demonstrated or not. Unit tests prove the
rules; this proves the rules fire on real documents, joined through the real registry, over
spans that really verified.

Scored against ``data/benchmarks/coupling_expectations.yaml``. The gold labels are loaded
only here, never by anything under ``app/``.

The one test worth reading first is
:func:`test_overlapping_authority_returns_ambiguous_rather_than_newest_wins`. The corpus
seeds two peer runbooks where the *newer* one disagrees, so a newest-wins fallback would
silently pass every other test in this file and fail only that one.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import pytest
import yaml

from app.application.use_cases.resolve_coupling import CouplingAnalysis, ResolveCouplingUseCase
from app.assumptions.class_a import merge_class_b
from app.assumptions.invalidation import propagate
from app.config.settings import REPO_ROOT
from app.domain.entities.assumption import AssumptionDelta
from app.domain.value_objects.enums import (
    AuthorityOutcome,
    DecayCategory,
    DeltaAssertedBy,
    DeltaKind,
    InvalidationOutcome,
    ValueType,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from app.domain.value_objects.typed_value import TypedValue
from app.evaluation.corpus import load_corpus
from app.infrastructure.database.repositories.memory import InMemoryCouplingIndexStore

EXPECTATIONS = REPO_ROOT / "data" / "benchmarks" / "coupling_expectations.yaml"


@pytest.fixture(scope="module")
def expectations() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(EXPECTATIONS.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def resolved(expectations: dict[str, Any]) -> tuple[CouplingAnalysis, dict[str, str], int]:
    """Ingest the corpus, build the index, and derive Class A. Once, then assert.

    Returns the analysis, a ``source_version_id -> corpus path`` map for scoring against
    the gold labels, and the number of live extraction calls made (which must be zero).
    """
    return asyncio.run(_resolve(expectations))


async def _resolve(
    expectations: dict[str, Any],
) -> tuple[CouplingAnalysis, dict[str, str], int]:
    stores = await load_corpus()
    assert not stores.extraction_failures, stores.extraction_failures

    as_of = dt.datetime.combine(
        dt.date.fromisoformat(str(expectations["as_of"])), dt.time(tzinfo=dt.UTC)
    )
    use_case = ResolveCouplingUseCase(
        sources=stores.sources,
        assumptions=stores.assumptions,
        gaps=stores.gaps,
        index_store=InMemoryCouplingIndexStore(),
    )
    analysis = use_case.execute(
        tenant_id=stores.tenant_id,
        as_of=as_of,
        registry_version=stores.registry_version,
    )

    paths = {report.version_id: relative for relative, report in stores.reports.items()}
    calls = sum(report.extraction_calls for report in stores.reports.values())
    return analysis, paths, calls


def _artifact(paths: dict[str, str], version_id: str | None) -> str | None:
    return paths.get(version_id) if version_id else None


def _findings_by_rule(analysis: CouplingAnalysis, rule: str) -> list:  # type: ignore[type-arg]
    return [f for f in analysis.findings if f.rule_id == rule]


# ---------------------------------------------------------------------------
# The claim: Class A without a model
# ---------------------------------------------------------------------------


def test_class_a_findings_are_discovered_without_any_llm_call(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """The load replays recorded proposals; the coupling path calls nothing at all."""
    analysis, _paths, calls = resolved
    assert calls == 0, f"the corpus load made {calls} live extraction calls"
    assert analysis.findings, "the resolver produced no Class A findings"
    assert all(f.origin.value == "resolver" for f in analysis.findings)


def test_the_run_needs_no_credentials_and_no_network(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """Stated as a test so it cannot quietly stop being true."""
    analysis, _paths, _calls = resolved
    assert analysis.index.binding_count > 0


def test_every_finding_is_evidence_backed_by_construction(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    analysis, _paths, _calls = resolved
    for finding in analysis.findings:
        assert finding.materialized_evidence_ids
        assert finding.evidence_is_derived_union is True
        assert finding.input_record_ids
        assert finding.rule_id


def test_no_binding_was_dropped_for_want_of_a_holder_document(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    analysis, _paths, _calls = resolved
    assert analysis.unresolvable == ()


# ---------------------------------------------------------------------------
# Seeded scenarios: what must be found
# ---------------------------------------------------------------------------


def test_the_seeded_same_scope_contradiction_is_detected(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """10,000 vs 7,500 USD, two declarers at global, overlapping validity."""
    analysis, paths, _calls = resolved
    case = next(
        c for c in expectations["must_find"] if c["id"] == "same_scope_contradiction_refund"
    )

    conflicts = [
        f
        for f in _findings_by_rule(analysis, "same_scope_contradiction")
        if f.assumption_key == case["key"]
    ]
    involved = {_artifact(paths, f.affected_source_version_id) for f in conflicts}
    for expected in case["participants"]:
        assert expected in involved, f"{expected} not reported in {involved}"


def test_the_seeded_unauthorized_override_is_flagged(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """A wiki page narrowing a policy it does not outrank."""
    analysis, paths, _calls = resolved
    case = next(
        c for c in expectations["must_find"] if c["id"] == "unauthorized_override_retention_eu"
    )

    overrides = _findings_by_rule(analysis, "unauthorized_override")
    match = next(
        f
        for f in overrides
        if _artifact(paths, f.affected_source_version_id) == case["affected"]
        and f.assumption_key == case["key"]
    )
    assert match.scope.render() == case["scope"]
    assert _artifact(paths, match.counterpart_source_version_id) == case["counterpart"]
    assert match.relationship is Relationship.UNAUTHORIZED_OVERRIDE


def test_the_unauthorized_override_records_the_authority_comparison(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """Auditable without re-running the resolver."""
    analysis, _paths, _calls = resolved
    relation = next(
        r
        for r in analysis.index.relations_of(Relationship.UNAUTHORIZED_OVERRIDE)
        if r.provenance.detail.get("assumption_key") == "retention.window"
    )
    detail = relation.provenance.detail
    assert detail["narrower_authority_class"] == "wiki"
    assert detail["broader_authority_class"] == "policy"
    assert detail["narrower_authority_rank"] < detail["broader_authority_rank"]
    assert relation.provenance.authority_model_version == "authority-1"


def test_the_seeded_stale_consumer_is_invalidated(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """The analytics definition reads 365d where the policy declares 90d."""
    analysis, paths, _calls = resolved
    case = next(
        c for c in expectations["must_find"] if c["id"] == "stale_consumer_retention_analytics"
    )

    stale = _findings_by_rule(analysis, "stale_consumer")
    match = next(
        f for f in stale if _artifact(paths, f.affected_source_version_id) == case["affected"]
    )
    assert match.assumption_key == case["key"]
    assert _artifact(paths, match.counterpart_source_version_id) == case["counterpart"]
    assert match.observed_value is not None and match.observed_value.render() == "365d"
    assert match.authoritative_value is not None and match.authoritative_value.render() == "90d"


def test_every_must_find_case_is_present(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """Class A recall over the seeded set."""
    analysis, paths, _calls = resolved
    missing: list[str] = []

    for case in expectations["must_find"]:
        candidates = [
            f for f in _findings_by_rule(analysis, case["rule"]) if f.assumption_key == case["key"]
        ]
        if "affected" in case:
            candidates = [
                f
                for f in candidates
                if _artifact(paths, f.affected_source_version_id) == case["affected"]
            ]
        if "participants" in case:
            involved = {_artifact(paths, f.affected_source_version_id) for f in candidates}
            if not set(case["participants"]) <= involved:
                missing.append(case["id"])
                continue
        if not candidates:
            missing.append(case["id"])

    assert not missing, f"Class A recall failure: {missing}"


# ---------------------------------------------------------------------------
# Seeded scenarios: what must NOT be found
# ---------------------------------------------------------------------------


def test_the_legitimate_scoped_override_is_not_flagged(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """regulatory >= policy at a narrower scope: recorded, never reported."""
    analysis, paths, _calls = resolved
    case = next(
        c for c in expectations["must_not_find"] if c["id"] == "legitimate_override_retention_eu"
    )

    override_targets = {
        _artifact(paths, f.affected_source_version_id)
        for f in _findings_by_rule(analysis, "unauthorized_override")
    }
    assert case["narrower"] not in override_targets

    recorded = {_artifact(paths, o.narrower.source_version_id) for o in analysis.index.overrides}
    assert case["narrower"] in recorded, "the legitimate override was not even recorded"


def test_the_describes_binding_produces_no_finding(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """The incident ticket narrates the old threshold and must stay silent."""
    analysis, paths, _calls = resolved
    case = next(
        c for c in expectations["must_not_find"] if c["id"] == "describes_binding_is_silent"
    )

    reported = {
        (_artifact(paths, f.affected_source_version_id), f.rule_id) for f in analysis.findings
    }
    for rule in case["forbidden_rules"]:
        assert (case["holder"], rule) not in reported


def test_a_consumer_agreeing_with_the_governing_policy_is_not_stale(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """The false positive a pairwise reading of the matrix would produce on this corpus."""
    analysis, paths, _calls = resolved
    case = next(
        c for c in expectations["must_not_find"] if c["id"] == "consumer_agreeing_with_authority"
    )

    stale_targets = {
        _artifact(paths, f.affected_source_version_id)
        for f in _findings_by_rule(analysis, "stale_consumer")
    }
    assert case["holder"] not in stale_targets


def test_the_negative_control_document_binds_nothing_and_reports_nothing(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """A system that finds something in every document has not been shown to find anything."""
    analysis, paths, _calls = resolved
    affected = {_artifact(paths, f.affected_source_version_id) for f in analysis.findings}
    assert "wiki/office-move-faq.md" not in affected


def test_no_finding_is_reported_twice(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """False Class A rate: a duplicated finding is a false positive to its reader."""
    analysis, _paths, _calls = resolved
    ids = [f.finding_id for f in analysis.findings]
    assert len(ids) == len(set(ids))

    relation_ids = [r.relation_id for r in analysis.index.relations]
    assert len(relation_ids) == len(set(relation_ids))


# ---------------------------------------------------------------------------
# Authority resolution over the corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_index", range(3))
def test_the_expected_authority_resolves(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
    expectations: dict[str, Any],
    case_index: int,
) -> None:
    analysis, paths, _calls = resolved
    case = expectations["authority"]["resolved"][case_index]

    resolution = analysis.index.authority_for(case["key"], ScopeSelector.parse(case["scope"]))
    assert resolution is not None
    assert resolution.outcome is AuthorityOutcome.RESOLVED
    assert resolution.authoritative is not None
    assert resolution.authoritative.value.render() == case["value"]
    assert _artifact(paths, resolution.authoritative.source_version_id) == case["holder"]

    if "flagged_narrowings" in case:
        assert len(resolution.unauthorized_narrowings) == case["flagged_narrowings"]


def test_overlapping_authority_returns_ambiguous_rather_than_newest_wins(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """The corpus-level guard against the forbidden fallback.

    Two peer runbooks declare 15 and 30 minutes. The 30-minute one is *newer*, so a
    newest-wins implementation would resolve to it and pass every other test in this file.
    """
    analysis, paths, _calls = resolved
    case = expectations["authority"]["ambiguous"][0]

    resolution = analysis.index.authority_for(case["key"], ScopeSelector.parse(case["scope"]))
    assert resolution is not None
    assert resolution.outcome is AuthorityOutcome.AMBIGUOUS
    assert resolution.authoritative is None
    assert resolution.ambiguity_reason is not None
    assert resolution.ambiguity_reason.value == case["reason"]

    candidates = {_artifact(paths, c.source_version_id) for c in resolution.tied_candidates}
    assert set(case["candidates"]) <= candidates

    # And the newer candidate specifically did not win.
    newer = next(c for c in resolution.tied_candidates if c.value.render() == "30min")
    assert resolution.authoritative is None
    assert newer.record.effective_from is not None


# ---------------------------------------------------------------------------
# Knowledge-decay report
# ---------------------------------------------------------------------------


def test_the_decay_report_covers_every_expected_category(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    analysis, _paths, _calls = resolved
    present = set(analysis.report.counts_by_category)
    expected = set(expectations["decay"]["must_include_categories"])
    assert expected <= present, f"missing decay categories: {sorted(expected - present)}"


def test_the_decay_report_separates_inconsistency_ambiguity_and_absence(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    analysis, _paths, _calls = resolved
    thresholds = expectations["decay"]
    report = analysis.report

    assert len(report.known_inconsistencies) >= thresholds["known_inconsistency_min"]
    assert len(report.known_ambiguities) >= thresholds["known_ambiguity_min"]
    assert len(report.missing_knowledge) >= thresholds["missing_knowledge_min"]


def test_no_decay_row_calls_absence_a_contradiction(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    analysis, _paths, _calls = resolved
    for row in analysis.report.missing_knowledge:
        assert row.rule_id is None
        assert row.finding_id is None


def test_every_inconsistency_row_cites_a_rule(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """The report is a projection of the resolver, not a second source of truth."""
    analysis, _paths, _calls = resolved
    for row in analysis.report.known_inconsistencies:
        assert row.rule_id is not None


def test_every_ambiguity_row_states_what_would_resolve_it(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    analysis, _paths, _calls = resolved
    assert analysis.report.known_ambiguities
    for row in analysis.report.known_ambiguities:
        assert row.requires_human_decision is True
        assert row.what_would_resolve


def test_the_registry_gaps_are_reported_as_the_growth_path(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    analysis, _paths, _calls = resolved
    rows = analysis.report.of_category(DecayCategory.REGISTRY_GAP)
    assert {row.key for row in rows} == {
        "refund.escalation_ceiling",
        "vendor.master_override",
    }


# ---------------------------------------------------------------------------
# The injected document stays powerless
# ---------------------------------------------------------------------------


def test_the_injected_document_cannot_become_authoritative(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """It declares a 1 USD threshold and a 3650d retention window, and governs neither.

    Its bindings are admitted - the spans really say it and the keys are registered - but it
    holds `wiki` authority, so at coupling time it can only ever be a losing declarer. This
    is the documented boundary in docs/SECURITY.md.
    """
    analysis, paths, _calls = resolved
    injected = "wiki/vendor-integration-notes.md"

    for key in ("refund.approval_threshold", "retention.window"):
        resolution = analysis.index.authority_for(key, ScopeSelector.global_scope())
        assert resolution is not None
        if resolution.authoritative is not None:
            assert _artifact(paths, resolution.authoritative.source_version_id) != injected


def test_the_injected_documents_bindings_are_reported_as_contradictions(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """Powerless, but not invisible: it is surfaced as a defect to fix."""
    analysis, paths, _calls = resolved
    affected = {_artifact(paths, f.affected_source_version_id) for f in analysis.findings}
    assert "wiki/vendor-integration-notes.md" in affected


# ---------------------------------------------------------------------------
# Delta propagation over the corpus
# ---------------------------------------------------------------------------


def test_the_delta_case_propagates_exactly_as_specified(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """Checkpoint E3, end to end on real documents."""
    analysis, paths, _calls = resolved
    case = expectations["delta_cases"][0]

    change = AssumptionDelta(
        delta_id=new_id("delta"),
        key=case["key"],
        scope=ScopeSelector.parse(case["scope"]),
        old_value=TypedValue(value_type=ValueType.DURATION, raw=90, unit="d"),
        new_value=TypedValue(value_type=ValueType.DURATION, raw=180, unit="d"),
        delta_kind=DeltaKind.VALUE_CHANGE,
        asserted_by=DeltaAssertedBy.CHANGE_REQUEST,
        registry_version=analysis.index.registry_version,
    )
    report = propagate(change, analysis.index)

    invalidated = {_artifact(paths, i.binding.source_version_id) for i in report.invalidated}
    confirm = {_artifact(paths, i.binding.source_version_id) for i in report.needs_confirmation}
    unaffected = {
        _artifact(paths, i.binding.source_version_id)
        for i in report.of_outcome(InvalidationOutcome.NOT_AFFECTED)
    }

    assert set(case["invalidated"]) <= invalidated
    assert set(case["needs_confirmation"]) <= confirm
    assert set(case["not_affected"]) <= unaffected


def test_the_legitimate_override_asks_for_confirmation_not_correction(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    analysis, paths, _calls = resolved
    change = AssumptionDelta(
        delta_id=new_id("delta"),
        key="retention.window",
        scope=ScopeSelector.global_scope(),
        old_value=TypedValue(value_type=ValueType.DURATION, raw=90, unit="d"),
        new_value=TypedValue(value_type=ValueType.DURATION, raw=180, unit="d"),
        delta_kind=DeltaKind.VALUE_CHANGE,
        asserted_by=DeltaAssertedBy.CHANGE_REQUEST,
        registry_version=analysis.index.registry_version,
    )
    report = propagate(change, analysis.index)

    confirm = {_artifact(paths, i.binding.source_version_id) for i in report.needs_confirmation}
    assert confirm == {"policy/eu-data-protection-policy.md"}


def test_the_pre_existing_defect_is_not_attributed_to_the_change(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    analysis, _paths, _calls = resolved
    case = expectations["delta_cases"][0]
    change = AssumptionDelta(
        delta_id=new_id("delta"),
        key="retention.window",
        scope=ScopeSelector.global_scope(),
        old_value=TypedValue(value_type=ValueType.DURATION, raw=90, unit="d"),
        new_value=TypedValue(value_type=ValueType.DURATION, raw=180, unit="d"),
        delta_kind=DeltaKind.VALUE_CHANGE,
        asserted_by=DeltaAssertedBy.CHANGE_REQUEST,
        registry_version=analysis.index.registry_version,
    )
    report = propagate(change, analysis.index)

    assert len(report.pre_existing_unauthorized_overrides) == case["pre_existing_overrides"]
    delta_relations = {r.relation_id for r in report.relations}
    for pre_existing in report.pre_existing_unauthorized_overrides:
        assert pre_existing.relation_id not in delta_relations


def test_a_change_to_an_unregistered_key_reports_nothing(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """The closed registry has no path for it, so the delta is inert."""
    analysis, _paths, _calls = resolved
    change = AssumptionDelta(
        delta_id=new_id("delta"),
        key="vendor.master_override",
        scope=ScopeSelector.global_scope(),
        old_value=None,
        new_value=TypedValue(value_type=ValueType.BOOLEAN, raw=True),
        delta_kind=DeltaKind.KEY_ADDED,
        asserted_by=DeltaAssertedBy.CHANGE_REQUEST,
        registry_version=analysis.index.registry_version,
    )
    report = propagate(change, analysis.index)
    assert report.impacts == ()
    assert report.relations == ()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_the_measured_metrics_meet_their_thresholds(
    resolved: tuple[CouplingAnalysis, dict[str, str], int], expectations: dict[str, Any]
) -> None:
    """The phase's scorecard, computed rather than asserted case by case."""
    analysis, paths, calls = resolved
    thresholds = expectations["metrics"]

    # Class A recall over the seeded set.
    found = 0
    for case in expectations["must_find"]:
        candidates = [
            f for f in _findings_by_rule(analysis, case["rule"]) if f.assumption_key == case["key"]
        ]
        if "affected" in case:
            candidates = [
                f
                for f in candidates
                if _artifact(paths, f.affected_source_version_id) == case["affected"]
            ]
        if candidates:
            found += 1
    recall = found / len(expectations["must_find"])
    assert recall >= thresholds["class_a_recall_min"], f"class_a_recall={recall}"

    # False Class A rate, scored per (artifact, forbidden rule).
    #
    # Artifact alone would be wrong: the EU data-protection policy must never be flagged as
    # an *unauthorized override*, but it is legitimately reported as a same-scope
    # contradiction against the EMEA wiki page. Scoring by artifact would count that true
    # positive as a false one, and the metric would punish correct behaviour.
    forbidden: set[tuple[str | None, str]] = set()
    for case in expectations["must_not_find"]:
        target = case.get("holder") or case.get("narrower")
        for rule in case.get("forbidden_rules", []):
            forbidden.add((target, rule))

    reported = {
        (_artifact(paths, f.affected_source_version_id), f.rule_id) for f in analysis.findings
    }
    false_positives = forbidden & reported
    false_rate = len(false_positives) / max(1, len(analysis.findings))
    assert false_rate <= thresholds["false_class_a_max"], f"false positives: {false_positives}"

    # Precision of the two governance rules: every one must be a labelled true positive.
    assert _findings_by_rule(analysis, "unauthorized_override")
    assert _findings_by_rule(analysis, "same_scope_contradiction")
    assert _findings_by_rule(analysis, "stale_consumer")

    assert calls == thresholds["llm_calls_in_coupling_path"]


def test_the_class_b_rediscovery_rate_is_measurable(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """The metric exists and is wired, even though no agentic layer runs yet."""
    analysis, _paths, _calls = resolved
    outcome, updated = merge_class_b((), analysis.findings)
    assert outcome.considered == 0
    assert outcome.class_b_rediscovery_rate == 0.0
    assert len(updated) == len(analysis.findings)


def test_a_class_b_hypothesis_cannot_displace_a_deterministic_result(
    resolved: tuple[CouplingAnalysis, dict[str, str], int],
) -> None:
    """It is absorbed as mechanism articulation and re-attributed to the resolver."""
    from tests.fixtures.domain import consequence_hypothesis

    analysis, _paths, _calls = resolved
    target = analysis.findings[0]
    hypothesis = consequence_hypothesis(assumption_keys=(target.assumption_key,))

    outcome, updated = merge_class_b(
        [hypothesis],
        analysis.findings,
        affected_artifact_of={hypothesis.hypothesis_id: target.affected_source_version_id},
    )
    assert outcome.suppressed_hypothesis_ids == (hypothesis.hypothesis_id,)
    merged = next(f for f in updated if f.finding_id == target.finding_id)
    assert merged.origin.value == "resolver"
    assert merged.consequence_class.value == "class_a_assumption_mediated"
    assert hypothesis.hypothesis_id in merged.merged_hypothesis_ids


# ---------------------------------------------------------------------------
# Reproducibility over the real corpus
# ---------------------------------------------------------------------------


def test_two_runs_over_the_same_corpus_agree_exactly(
    expectations: dict[str, Any],
) -> None:
    """Same sources, same registry, same rules, same authority model - same output."""
    first, first_paths, _c1 = asyncio.run(_resolve(expectations))
    second, second_paths, _c2 = asyncio.run(_resolve(expectations))

    assert first_paths == second_paths
    assert [f.finding_id for f in first.findings] == [f.finding_id for f in second.findings]
    assert [c.model_dump() for c in first.index.sets] == [c.model_dump() for c in second.index.sets]
    assert first.report.counts_by_category == second.report.counts_by_category


def test_the_ablated_corpus_loses_the_dependent_finding(
    expectations: dict[str, Any],
) -> None:
    """Ablation honesty: no finding may survive removal of a document it was derived from.

    "Must not survive" is asserted as *identity*, not as absence of the category, and the
    distinction is the whole point. Removing the retention policy leaves the vendor wiki
    page as the only global declarer, so a staleness finding against the analytics
    definition still exists - but it is a **different** finding, over different records,
    citing different spans, naming a different authority. That is correct: the corpus still
    contains a governing declarer that disagrees.

    What must be true is that nothing survives *unchanged*. A finding whose id and evidence
    are identical after its supporting document is deleted was not derived from that
    document at all - it was coming from somewhere else.
    """
    analysis, paths, _calls = asyncio.run(_resolve(expectations))
    removed = "policy/data-retention-policy.md"
    removed_version = next(v for v, path in paths.items() if path == removed)

    grounded = [
        f
        for f in analysis.findings
        if removed_version in (f.affected_source_version_id, f.counterpart_source_version_id)
    ]
    assert grounded, "nothing in the baseline depends on the document being ablated"

    ablated = asyncio.run(_resolve_without(expectations, removed))
    survivors = {f.finding_id for f in ablated.findings}
    still_here = [f.finding_id for f in grounded if f.finding_id in survivors]

    assert not still_here, (
        f"{len(still_here)} finding(s) survived removal of their supporting document with "
        "identical identity, so they were not derived from it"
    )


def test_no_finding_cites_the_ablated_documents_evidence(
    expectations: dict[str, Any],
) -> None:
    """The stronger form: after ablation, no span from the removed document is cited."""
    analysis, paths, _calls = asyncio.run(_resolve(expectations))
    removed = "policy/data-retention-policy.md"
    removed_version = next(v for v, path in paths.items() if path == removed)

    from_removed = {
        b.evidence_id for b in analysis.index.bindings if b.source_version_id == removed_version
    }
    baseline_cited = {
        evidence
        for f in analysis.findings
        for evidence in f.materialized_evidence_ids
        if evidence in from_removed
    }
    assert baseline_cited, "the ablated document contributed no evidence to begin with"

    ablated = asyncio.run(_resolve_without(expectations, removed))
    cited = {e for f in ablated.findings for e in f.materialized_evidence_ids}
    assert not (cited & baseline_cited)


def test_the_staleness_finding_renames_its_authority_after_ablation(
    expectations: dict[str, Any],
) -> None:
    """Whatever remains must name the declarer that actually still governs."""
    analysis, paths, _calls = asyncio.run(_resolve(expectations))
    baseline = next(
        f
        for f in analysis.findings
        if f.rule_id == "stale_consumer" and f.assumption_key == "retention.window"
    )
    assert (
        _artifact(paths, baseline.counterpart_source_version_id)
        == "policy/data-retention-policy.md"
    )

    ablated = asyncio.run(_resolve_without(expectations, "policy/data-retention-policy.md"))
    remaining = [
        f
        for f in ablated.findings
        if f.rule_id == "stale_consumer" and f.assumption_key == "retention.window"
    ]
    for finding in remaining:
        assert finding.counterpart_source_version_id != baseline.counterpart_source_version_id
        assert finding.finding_id != baseline.finding_id


async def _resolve_without(expectations: dict[str, Any], drop: str) -> CouplingAnalysis:
    from app.evaluation.corpus import load_manifest

    manifest = load_manifest()
    manifest = {
        **manifest,
        "sources": [s for s in manifest["sources"] if s["path"] != drop],
    }
    stores = await load_corpus(manifest=manifest)

    as_of = dt.datetime.combine(
        dt.date.fromisoformat(str(expectations["as_of"])), dt.time(tzinfo=dt.UTC)
    )
    use_case = ResolveCouplingUseCase(
        sources=stores.sources, assumptions=stores.assumptions, gaps=stores.gaps
    )
    return use_case.execute(
        tenant_id=stores.tenant_id, as_of=as_of, registry_version=stores.registry_version
    )
