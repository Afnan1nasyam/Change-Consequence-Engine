"""Findings, challenge, arbitration, scoring, consequences, dispositions.

The two structural claims under test:

* **An uncited objection is unrepresentable.** ``refuted`` without a citation is a
  validation error, and ``no_refutation_found`` costs a falsification test. A model
  instructed merely to disagree produces plausible objections at whatever rate it is asked
  for them; making the citation a schema requirement is what turns adversarial review into
  a proof obligation.
* **Confidence exists on exactly one entity and is computed.** Severity is the judge's by
  design, and that asymmetry is deliberate.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.entities.findings import (
    AnalysisReport,
    ArbitrationDecision,
    ChallengeResult,
    Consequence,
    ConsequenceScore,
    SpecialistFinding,
)
from app.domain.value_objects.enums import (
    ChallengeOutcome,
    ClassAGround,
    ConsequenceClass,
    Disposition,
    EvidencePosition,
    Origin,
    Severity,
    Status,
)
from tests.fixtures.domain import (
    analysis_report,
    arbitration_decision,
    challenge_result,
    class_a_consequence,
    class_b_consequence,
    consequence_score,
    falsification_test,
    finding_disposition,
    make_id,
    specialist_finding,
)

# ---------------------------------------------------------------------------
# SpecialistFinding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position", [EvidencePosition.SUPPORTED, EvidencePosition.WEAKLY_SUPPORTED]
)
def test_a_supported_position_requires_evidence(position: EvidencePosition) -> None:
    with pytest.raises(ValidationError, match="requires at least one evidence id"):
        specialist_finding(evidence_position=position, evidence_ids=())


@pytest.mark.parametrize(
    "position",
    [EvidencePosition.UNSUPPORTED, EvidencePosition.UNRESOLVED, EvidencePosition.DISPUTED],
)
def test_a_non_asserting_position_needs_no_evidence(position: EvidencePosition) -> None:
    """`unresolved` with nothing to cite is the honest answer, not a defect."""
    finding = specialist_finding(evidence_position=position, evidence_ids=())
    assert finding.evidence_position is position


def test_a_specialist_emits_a_position_not_a_number() -> None:
    with pytest.raises(ValidationError):
        SpecialistFinding(**{**specialist_finding().model_dump(), "confidence": 0.7})
    assert "confidence" not in SpecialistFinding.model_fields


def test_the_free_text_unknowns_field_is_gone() -> None:
    """It could not distinguish searched-and-absent from never-searched."""
    with pytest.raises(ValidationError):
        SpecialistFinding(**{**specialist_finding().model_dump(), "unknowns": ["who owns it"]})
    assert "evidence_gap_ids" in SpecialistFinding.model_fields


def test_a_finding_names_the_agent_and_the_prompt_that_produced_it() -> None:
    finding = specialist_finding()
    assert finding.produced_by.agent
    assert finding.produced_by.prompt_version >= 1


# ---------------------------------------------------------------------------
# ChallengeResult
# ---------------------------------------------------------------------------


def test_there_are_exactly_three_outcomes() -> None:
    assert {o.value for o in ChallengeOutcome} == {
        "refuted",
        "weakened",
        "no_refutation_found",
    }


def test_refuted_requires_a_contradicting_citation() -> None:
    with pytest.raises(ValidationError, match="uncited objection is not representable"):
        challenge_result(
            outcome=ChallengeOutcome.REFUTED,
            contradicts_what="the threshold value",
            falsification_test_id=None,
        )


def test_refuted_requires_saying_what_is_contradicted() -> None:
    with pytest.raises(ValidationError, match="contradicts_what"):
        challenge_result(
            outcome=ChallengeOutcome.REFUTED,
            contradicting_evidence_ids=(make_id("evidence"),),
            falsification_test_id=None,
        )


def test_a_valid_refutation() -> None:
    result = challenge_result(
        outcome=ChallengeOutcome.REFUTED,
        contradicting_evidence_ids=(make_id("evidence"),),
        contradicts_what="Policy v4 already sets the threshold to 5,000 USD.",
        falsification_test_id=None,
    )
    assert result.outcome is ChallengeOutcome.REFUTED


def test_weakened_requires_evidence_and_a_narrowed_scope() -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        challenge_result(
            outcome=ChallengeOutcome.WEAKENED,
            scope_narrowing="applies only outside the EU",
            falsification_test_id=None,
        )
    with pytest.raises(ValidationError, match="scope_narrowing"):
        challenge_result(
            outcome=ChallengeOutcome.WEAKENED,
            contradicting_evidence_ids=(make_id("evidence"),),
            falsification_test_id=None,
        )


def test_no_refutation_found_costs_a_falsification_test() -> None:
    """It stops being a lazy default without being punished when it is the right answer."""
    with pytest.raises(ValidationError, match="it costs a test"):
        challenge_result(falsification_test_id=None)


def test_no_refutation_found_must_carry_no_evidence() -> None:
    with pytest.raises(ValidationError, match="must carry no evidence ids"):
        challenge_result(contradicting_evidence_ids=(make_id("evidence"),))


def test_no_refutation_found_must_not_assert_a_contradiction() -> None:
    with pytest.raises(ValidationError, match="must not"):
        challenge_result(contradicts_what="something")
    with pytest.raises(ValidationError, match="must not"):
        challenge_result(scope_narrowing="something")


def test_the_admissible_class_a_grounds_are_exactly_four() -> None:
    """A derived relation is only ever rule-wrong or input-wrong."""
    assert {g.value for g in ClassAGround} == {
        "disputed_span",
        "disputed_key_or_scope",
        "disputed_temporal_validity",
        "superseding_version",
    }


def test_a_class_a_challenge_records_its_ground() -> None:
    result = challenge_result(
        outcome=ChallengeOutcome.REFUTED,
        contradicting_evidence_ids=(make_id("evidence"),),
        contradicts_what="the cited span does not mention a threshold",
        class_a_ground=ClassAGround.DISPUTED_SPAN,
        falsification_test_id=None,
    )
    assert result.class_a_ground is ClassAGround.DISPUTED_SPAN


def test_a_challenge_carries_no_confidence() -> None:
    with pytest.raises(ValidationError):
        ChallengeResult(**{**challenge_result().model_dump(), "confidence": 0.4})


def test_a_falsification_test_names_an_observation() -> None:
    test = falsification_test()
    assert test.observation
    assert test.feeds_validation_checklist


# ---------------------------------------------------------------------------
# ArbitrationDecision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [Status.SUPPORTED, Status.LIKELY])
def test_an_assertive_status_requires_supporting_evidence(status: Status) -> None:
    """Absence of contrary evidence is not support."""
    with pytest.raises(ValidationError, match="not support"):
        arbitration_decision(status=status, supporting_evidence_ids=())


@pytest.mark.parametrize("status", [Status.SPECULATIVE, Status.UNRESOLVED, Status.REJECTED])
def test_a_non_assertive_status_needs_no_evidence(status: Status) -> None:
    decision = arbitration_decision(
        status=status,
        severity=Severity.LOW,
        supporting_evidence_ids=(),
    )
    assert decision.status is status


def test_an_unresolved_high_risk_finding_must_name_what_would_settle_it() -> None:
    for severity in (Severity.HIGH, Severity.CRITICAL):
        with pytest.raises(ValidationError, match="validation action"):
            arbitration_decision(
                status=Status.UNRESOLVED,
                severity=severity,
                supporting_evidence_ids=(),
                validation_actions=(),
            )
    ok = arbitration_decision(
        status=Status.UNRESOLVED,
        severity=Severity.CRITICAL,
        supporting_evidence_ids=(),
        validation_actions=("Confirm the EU retention override with the DPO.",),
    )
    assert ok.validation_actions


def test_severity_is_categorical_not_numeric() -> None:
    assert {s.value for s in Severity} == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    with pytest.raises(ValidationError):
        arbitration_decision(severity=0.8)
    with pytest.raises(ValidationError):
        arbitration_decision(severity="high")


def test_severity_is_judge_assigned_but_confidence_is_not_available_to_the_judge() -> None:
    """The deliberate asymmetry (checkpoint 9.4), asserted rather than assumed."""
    assert "severity" in ArbitrationDecision.model_fields
    assert "confidence" not in ArbitrationDecision.model_fields
    with pytest.raises(ValidationError):
        ArbitrationDecision(**{**arbitration_decision().model_dump(), "confidence": 0.9})


def test_a_decision_preserves_disagreement_rather_than_resolving_it_away() -> None:
    decision = arbitration_decision(
        preserved_disagreements=("risk-analyst rated this HIGH; process-analyst rated LOW",)
    )
    assert decision.preserved_disagreements


# ---------------------------------------------------------------------------
# ConsequenceScore
# ---------------------------------------------------------------------------


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        consequence_score(confidence=1.4)
    with pytest.raises(ValidationError):
        consequence_score(confidence=-0.1)


def test_a_score_must_persist_its_inputs() -> None:
    """A score with no recorded inputs cannot be explained or recomputed."""
    with pytest.raises(ValidationError, match="recomputed and explained"):
        consequence_score(factor_inputs={})


def test_a_score_pins_the_function_version() -> None:
    payload = consequence_score().model_dump()
    del payload["confidence_fn_version"]
    with pytest.raises(ValidationError):
        ConsequenceScore(**payload)


def test_corroboration_counts_distinct_source_versions() -> None:
    score = consequence_score(
        corroborating_source_version_ids=(
            make_id("source_version"),
            make_id("source_version"),
            make_id("source_version", 2),
        )
    )
    assert score.corroboration_count == 2, "the same source version twice is one source"


def test_the_scorer_cannot_see_agents_so_it_cannot_count_them() -> None:
    """Structural enforcement of checkpoint 9.3, not a policy statement in a prompt."""
    assert not [f for f in ConsequenceScore.model_fields if "agent" in f]
    with pytest.raises(ValidationError):
        ConsequenceScore(**{**consequence_score().model_dump(), "agreeing_agents": ["a", "b", "c"]})


def test_a_score_carries_no_severity() -> None:
    with pytest.raises(ValidationError):
        ConsequenceScore(**{**consequence_score().model_dump(), "severity": "HIGH"})


# ---------------------------------------------------------------------------
# Consequence
# ---------------------------------------------------------------------------


def test_class_a_means_the_resolver_produced_it() -> None:
    assert class_a_consequence().origin is Origin.RESOLVER
    assert class_b_consequence().origin is Origin.AGENT


def test_an_agent_cannot_self_declare_class_a() -> None:
    """Classification is by producer, not by subject matter."""
    with pytest.raises(ValidationError, match="Classification is by producer"):
        class_b_consequence(consequence_class=ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED)


def test_the_resolver_output_cannot_be_demoted_to_class_b_either() -> None:
    with pytest.raises(ValidationError, match="Classification is by producer"):
        class_a_consequence(consequence_class=ConsequenceClass.CLASS_B_BEHAVIORAL)


def test_class_a_requires_a_derivation_and_an_assumption_key() -> None:
    with pytest.raises(ValidationError, match="defined by having been derived"):
        class_a_consequence(derived_relation_ids=())
    with pytest.raises(ValidationError, match="assumption-mediated"):
        class_a_consequence(assumption_keys=())


def test_class_b_requires_a_hypothesis_and_forbids_a_derivation() -> None:
    with pytest.raises(ValidationError, match="hypothesis_id"):
        class_b_consequence(hypothesis_ids=())
    with pytest.raises(ValidationError, match="it would be Class A"):
        class_b_consequence(derived_relation_ids=(make_id("relation"),))


@pytest.mark.parametrize("status", [Status.SUPPORTED, Status.LIKELY])
def test_an_assertive_consequence_requires_evidence(status: Status) -> None:
    with pytest.raises(ValidationError, match="speculative or unresolved"):
        class_a_consequence(status=status, evidence_ids=())


def test_class_b_above_speculative_requires_a_historical_analogue() -> None:
    """An analogue without a recorded observed outcome does not qualify (checkpoint 9.2)."""
    for status in (Status.SUPPORTED, Status.LIKELY, Status.DISPUTED):
        with pytest.raises(ValidationError, match="historical analogue"):
            class_b_consequence(
                status=status,
                evidence_ids=(make_id("evidence"),),
                historical_analogue_id=None,
            )


def test_class_b_with_an_analogue_may_exceed_speculative() -> None:
    consequence = class_b_consequence(
        status=Status.LIKELY,
        evidence_ids=(make_id("evidence"),),
        historical_analogue_id=make_id("event"),
    )
    assert consequence.status is Status.LIKELY


def test_class_a_needs_no_analogue() -> None:
    """It was derived; a behavioural precedent is not what supports it."""
    assert class_a_consequence(status=Status.SUPPORTED).historical_analogue_id is None


def test_a_rediscovered_hypothesis_merges_into_class_a_never_the_reverse() -> None:
    merged = class_a_consequence(merged_hypothesis_ids=(make_id("hypothesis"),))
    assert merged.merged_hypothesis_ids

    with pytest.raises(ValidationError, match="absorbed a rediscovered"):
        class_b_consequence(merged_hypothesis_ids=(make_id("hypothesis"),))


def test_a_consequence_carries_no_confidence_and_no_likelihood() -> None:
    with pytest.raises(ValidationError):
        Consequence(**{**class_a_consequence().model_dump(), "confidence": 0.8})
    with pytest.raises(ValidationError):
        Consequence(**{**class_a_consequence().model_dump(), "likelihood": 0.8})


def test_a_consequence_records_the_run_that_produced_it() -> None:
    assert class_a_consequence().run_provenance.schema_version == "domain-1"


# ---------------------------------------------------------------------------
# FindingDisposition
# ---------------------------------------------------------------------------


def test_suppression_must_name_what_it_suppresses() -> None:
    with pytest.raises(ValidationError, match="requires an assumption_key"):
        finding_disposition(
            disposition=Disposition.WRONG,
            suppresses_until_source_revision=True,
        )


def test_scoped_suppression_is_accepted() -> None:
    scoped = finding_disposition(
        disposition=Disposition.NOT_MATERIAL,
        assumption_key="refund.approval_threshold",
        suppresses_until_source_revision=True,
    )
    assert scoped.suppresses_until_source_revision


@pytest.mark.parametrize(
    "disposition",
    [Disposition.USEFUL, Disposition.ALREADY_KNOWN, Disposition.UNVERIFIABLE],
)
def test_a_positive_disposition_may_not_suppress(disposition: Disposition) -> None:
    """`already_known` is a product signal, not a reason to stop reporting the finding."""
    with pytest.raises(ValidationError, match="may suppress"):
        finding_disposition(
            disposition=disposition,
            artifact_ref="training deck v7",
            suppresses_until_source_revision=True,
        )


def test_dispositions_are_the_five_review_verdicts() -> None:
    assert {d.value for d in Disposition} == {
        "useful",
        "already_known",
        "not_material",
        "wrong",
        "unverifiable",
    }


# ---------------------------------------------------------------------------
# AnalysisReport
# ---------------------------------------------------------------------------


def test_the_default_view_must_be_a_subset_of_the_report() -> None:
    with pytest.raises(ValidationError, match="default view references consequences absent"):
        analysis_report(default_view_consequence_ids=(make_id("consequence", 9),))


def test_suppressed_count_is_reported_rather_than_hidden() -> None:
    """A reader who cannot see what was withheld cannot calibrate trust."""
    assert analysis_report().suppressed_count == 0
    assert analysis_report(suppressed_count=4).suppressed_count == 4
    with pytest.raises(ValidationError):
        analysis_report(suppressed_count=-1)


def test_a_report_carries_gaps_and_unresolved_questions() -> None:
    report = analysis_report(
        evidence_gap_ids=(make_id("gap"),),
        unresolved_questions=("Who owns the refund SOP?",),
    )
    assert report.evidence_gap_ids
    assert report.unresolved_questions


def test_a_report_introduces_no_score_of_its_own() -> None:
    with pytest.raises(ValidationError):
        AnalysisReport(**{**analysis_report().model_dump(), "confidence": 0.9})
