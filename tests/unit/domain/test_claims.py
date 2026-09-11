"""The five-part epistemic ontology.

The discriminated union is what stops the four kinds of claim being collapsed into one
generic "assumption". The evidence rules per type are the substance:

* A and B **must** cite. An uncited fact is not a fact.
* C **must not** cite. Its evidence is the computed transitive union of its inputs;
  nominating one input span as direct support would manufacture false provenance.
* D **may** cite nothing. A hypothesis with no evidence is not a defect; a hypothesis
  labelled as established is.
* E **must not** cite, and must carry the negative provenance that distinguishes
  searched-and-absent from nobody-looked.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.domain.entities.claims import (
    Claim,
    ClaimEnvelope,
    ConsequenceHypothesis,
    DerivedRelation,
    EvidenceGap,
    ObservedFact,
)
from app.domain.value_objects.enums import ClaimType, Derivation, Support
from app.domain.value_objects.taxonomy import DERIVED_RELATIONSHIPS, Relationship
from tests.fixtures.domain import (
    consequence_hypothesis,
    derivation_provenance,
    derived_relation,
    evidence_gap,
    make_id,
    observed_fact,
    unauthorized_override_relation,
)

# ---------------------------------------------------------------------------
# Type A - ObservedFact
# ---------------------------------------------------------------------------


def test_an_observed_fact_must_cite_at_least_one_span() -> None:
    with pytest.raises(ValidationError):
        observed_fact(evidence_ids=())


def test_multi_span_support_requires_two_spans() -> None:
    """Otherwise `multi_span` claims corroboration the citation list does not show."""
    with pytest.raises(ValidationError, match="at least two evidence ids"):
        observed_fact(support=Support.MULTI_SPAN, evidence_ids=(make_id("evidence"),))
    ok = observed_fact(
        support=Support.MULTI_SPAN,
        evidence_ids=(make_id("evidence"), make_id("evidence", 2)),
    )
    assert len(ok.evidence_ids) == 2


@pytest.mark.parametrize(
    "support",
    [Support.RULE_OVER_RECORDS, Support.MECHANISM_ONLY, Support.NEGATIVE_PROVENANCE],
)
def test_an_observed_fact_cannot_claim_a_non_span_support(support: Support) -> None:
    with pytest.raises(ValidationError, match="verified_span or multi_span"):
        observed_fact(support=support)


def test_an_observed_fact_axes_are_fixed() -> None:
    with pytest.raises(ValidationError):
        observed_fact(derivation=Derivation.INFERRED)
    with pytest.raises(ValidationError):
        observed_fact(claim_type=ClaimType.HYPOTHESIS)


# ---------------------------------------------------------------------------
# Type C - DerivedRelation
# ---------------------------------------------------------------------------


def test_a_derived_relation_must_not_carry_evidence_of_its_own() -> None:
    """Demanding a citation here would force the resolver to fabricate provenance."""
    with pytest.raises(ValidationError, match="fabricated provenance"):
        derived_relation(evidence_ids=(make_id("evidence"),))


def test_a_derived_relation_requires_its_rule_and_inputs() -> None:
    with pytest.raises(ValidationError):
        derived_relation(input_record_ids=())

    payload = derived_relation().model_dump()
    del payload["provenance"]
    with pytest.raises(ValidationError):
        DerivedRelation(**payload)


def test_derivation_provenance_carries_the_versions_that_decided_the_finding() -> None:
    relation = derived_relation()
    assert relation.provenance.rule_id
    assert relation.provenance.registry_version
    assert relation.provenance.coupling_index_version


def test_input_record_ids_must_be_assumption_ids() -> None:
    with pytest.raises(ValidationError):
        derived_relation(input_record_ids=("record-1",))


@pytest.mark.parametrize("relationship", sorted(DERIVED_RELATIONSHIPS))
def test_the_three_derivable_relationships_are_accepted(relationship: Relationship) -> None:
    extra = {}
    if relationship is Relationship.UNAUTHORIZED_OVERRIDE:
        extra["provenance"] = unauthorized_override_relation().provenance
    assert derived_relation(relationship=relationship, **extra).relationship is relationship


@pytest.mark.parametrize(
    "relationship",
    [
        Relationship.CREATES_RISK_FOR,
        Relationship.HISTORICALLY_ANALOGOUS_TO,
        Relationship.DIRECTLY_CHANGES,
        Relationship.BINDS_ASSUMPTION,
    ],
)
def test_a_derived_relation_cannot_express_a_non_derivable_edge(
    relationship: Relationship,
) -> None:
    """A model-inferred relationship may not enter the deterministic tier by relabelling."""
    with pytest.raises(ValidationError, match="cannot express"):
        derived_relation(relationship=relationship)


def test_an_unauthorized_override_must_record_the_authority_comparison() -> None:
    """The checkpoint requires the comparison recorded, not merely applied."""
    with pytest.raises(ValidationError, match="authority_model_version"):
        derived_relation(
            relationship=Relationship.UNAUTHORIZED_OVERRIDE,
            provenance=derivation_provenance(
                detail={
                    "narrower_authority_class": "wiki",
                    "broader_authority_class": "policy",
                }
            ),
        )


def test_an_unauthorized_override_must_name_both_authority_classes() -> None:
    with pytest.raises(ValidationError, match="narrower_authority_class"):
        derived_relation(
            relationship=Relationship.UNAUTHORIZED_OVERRIDE,
            provenance=derivation_provenance(
                authority_model_version="authority-1",
                detail={"broader_authority_class": "policy"},
            ),
        )
    with pytest.raises(ValidationError, match="broader_authority_class"):
        derived_relation(
            relationship=Relationship.UNAUTHORIZED_OVERRIDE,
            provenance=derivation_provenance(
                authority_model_version="authority-1",
                detail={"narrower_authority_class": "wiki"},
            ),
        )


def test_the_e4_worked_example_validates() -> None:
    """A wiki page narrowing a policy: both classes, the ordering, and its version."""
    relation = unauthorized_override_relation()
    assert relation.relationship is Relationship.UNAUTHORIZED_OVERRIDE
    assert relation.derivation is Derivation.DERIVED
    assert relation.support is Support.RULE_OVER_RECORDS
    assert relation.provenance.detail["narrower_authority_class"] == "wiki"
    assert relation.provenance.detail["broader_authority_class"] == "policy"
    assert relation.provenance.authority_model_version == "authority-1"
    assert relation.evidence_ids == ()


def test_a_derived_relation_has_no_model_provenance() -> None:
    """It had no model. That absence is the machine-readable form of the claim."""
    assert "proposed_by" not in DerivedRelation.model_fields
    assert "produced_by" not in DerivedRelation.model_fields
    with pytest.raises(ValidationError):
        DerivedRelation(**{**derived_relation().model_dump(), "produced_by": {"agent": "x"}})


def test_a_derived_relation_axes_are_fixed() -> None:
    with pytest.raises(ValidationError):
        derived_relation(derivation=Derivation.INFERRED)
    with pytest.raises(ValidationError):
        derived_relation(support=Support.MECHANISM_ONLY)


# ---------------------------------------------------------------------------
# Type D - Hypothesis
# ---------------------------------------------------------------------------


def test_a_hypothesis_may_legitimately_cite_nothing() -> None:
    assert consequence_hypothesis().evidence_ids == ()


def test_a_hypothesis_must_articulate_a_mechanism_and_its_weakest_link() -> None:
    """A mechanism with no acknowledged weak point is usually an unexamined one."""
    for field in ("mechanism", "weakest_link", "claim"):
        payload = consequence_hypothesis().model_dump()
        del payload[field]
        with pytest.raises(ValidationError):
            ConsequenceHypothesis(**payload)


def test_a_hypothesis_is_attributed_to_a_prompt_and_a_model() -> None:
    hypothesis = consequence_hypothesis()
    assert hypothesis.proposed_by.prompt_sha256
    assert hypothesis.proposed_by.provider == "fake"


@pytest.mark.parametrize("forbidden", ["confidence", "consequence_class", "severity", "likelihood"])
def test_a_hypothesis_cannot_rate_itself(forbidden: str) -> None:
    """Confidence is computed, class is pipeline-assigned, severity is the judge's."""
    with pytest.raises(ValidationError):
        ConsequenceHypothesis(**{**consequence_hypothesis().model_dump(), forbidden: 0.9})


def test_a_hypothesis_may_carry_a_historical_analogue() -> None:
    hypothesis = consequence_hypothesis(historical_analogue_id=make_id("event"))
    assert hypothesis.historical_analogue_id == make_id("event")


def test_a_hypothesis_analogue_must_be_a_real_event_id_shape() -> None:
    with pytest.raises(ValidationError):
        consequence_hypothesis(historical_analogue_id="the 2024 incident")


def test_a_hypothesis_axes_are_fixed_to_inferred() -> None:
    with pytest.raises(ValidationError):
        consequence_hypothesis(derivation=Derivation.DERIVED)
    with pytest.raises(ValidationError):
        consequence_hypothesis(support=Support.VERIFIED_SPAN)


# ---------------------------------------------------------------------------
# Type E - EvidenceGap
# ---------------------------------------------------------------------------


def test_a_gap_must_not_carry_evidence() -> None:
    with pytest.raises(ValidationError, match="recorded absence of evidence"):
        evidence_gap(evidence_ids=(make_id("evidence"),))


def test_a_gap_requires_negative_provenance() -> None:
    """Without it, a gap is indistinguishable from nobody having looked."""
    with pytest.raises(ValidationError):
        evidence_gap(strategies_executed=())
    with pytest.raises(ValidationError):
        evidence_gap(source_types_searched=())

    for field in ("as_of", "would_be_resolved_by", "question"):
        payload = evidence_gap().model_dump()
        del payload[field]
        with pytest.raises(ValidationError):
            EvidenceGap(**payload)


def test_a_gap_is_a_claim_about_the_corpus() -> None:
    gap = evidence_gap()
    assert gap.derivation is Derivation.ABSENT
    assert gap.support is Support.NEGATIVE_PROVENANCE


def test_a_gap_attaches_to_an_analysis() -> None:
    """It is scoped to the run that failed to find something, not to the world."""
    assert evidence_gap().analysis_id == make_id("analysis")
    with pytest.raises(ValidationError):
        evidence_gap(analysis_id="run-1")


# ---------------------------------------------------------------------------
# The discriminated union
# ---------------------------------------------------------------------------

_ADAPTER: TypeAdapter[Claim] = TypeAdapter(Claim)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (observed_fact, ObservedFact),
        (derived_relation, DerivedRelation),
        (consequence_hypothesis, ConsequenceHypothesis),
        (evidence_gap, EvidenceGap),
    ],
)
def test_each_variant_round_trips_through_the_discriminator(
    factory: object, expected: type
) -> None:
    claim = factory()  # type: ignore[operator]
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(claim))
    assert isinstance(restored, expected)
    assert restored == claim


def test_the_envelope_preserves_the_variant() -> None:
    envelope = ClaimEnvelope(claim=evidence_gap())
    restored = ClaimEnvelope.model_validate_json(envelope.model_dump_json())
    assert isinstance(restored.claim, EvidenceGap)


def test_an_unknown_claim_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"claim_type": "guess", "statement": "something"})


def test_a_payload_cannot_claim_one_type_while_carrying_another_shape() -> None:
    """A hypothesis body labelled `derived_relation` must not validate."""
    payload = consequence_hypothesis().model_dump(mode="json")
    payload["claim_type"] = ClaimType.DERIVED_RELATION.value
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(payload)


def test_the_ontology_has_exactly_five_types() -> None:
    assert {c.value for c in ClaimType} == {
        "observed_fact",
        "assumption_binding",
        "derived_relation",
        "hypothesis",
        "evidence_gap",
    }


def test_derived_and_inferred_are_distinct_derivations() -> None:
    """Collapsing them would make one value mean both computed and made up."""
    assert len({Derivation.DERIVED, Derivation.INFERRED}) == 2
    assert {d.value for d in Derivation} == {
        "observed",
        "extracted",
        "derived",
        "inferred",
        "absent",
    }
