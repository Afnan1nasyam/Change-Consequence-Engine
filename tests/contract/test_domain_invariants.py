"""Cross-cutting invariants asserted over *every* Phase 1 entity.

The per-entity tests check the rules someone remembered to write. These check the rules
that must hold everywhere, by enumeration - so a new entity that quietly carries a
``confidence`` field fails here even if nobody thought to test it.

``tests.fixtures.domain.SAMPLES`` is the enumeration source, and
:func:`test_every_exported_entity_has_a_sample` keeps it honest against ``__all__``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import app.domain.entities as entities_pkg
from app.domain.entities.base import DOMAIN_SCHEMA_VERSION
from app.domain.entities.findings import ConsequenceScore
from app.domain.entities.graph import GraphEdge
from app.domain.value_objects.provenance import ModelProvenance
from tests.fixtures.domain import SAMPLES

#: Entities produced by, or transcribed from, an LLM call. None of them may carry a
#: computed score or a pipeline-assigned classification.
AGENT_AUTHORED = {
    "AssumptionDelta",
    "AssumptionRecord",
    "ChallengeResult",
    "ChangeSpec",
    "ConsequenceHypothesis",
    "KeyAlias",
    "KeyRegistryGap",
    "ObservedFact",
    "EvidenceGap",
    "SpecialistFinding",
    "ArbitrationDecision",
}

#: Vocabulary retired by the checkpoint. Present in no schema, at any layer.
RETIRED_FIELDS = (
    "binding_kind",
    "extraction_confidence",
    "likelihood",
    "has_collision",
    "unknowns",
    "potentially_impacts",
)


def _entity_classes() -> dict[str, type[BaseModel]]:
    return {name: type(sample) for name, sample in SAMPLES.items()}


def _nested_models(model: type[BaseModel]) -> set[type[BaseModel]]:
    """The model plus every Pydantic model reachable through its annotations."""
    found: set[type[BaseModel]] = {model}
    for field in model.model_fields.values():
        for candidate in (field.annotation, *getattr(field.annotation, "__args__", ())):
            if (
                isinstance(candidate, type)
                and issubclass(candidate, BaseModel)
                and candidate not in found
            ):
                found |= _nested_models(candidate)
    return found


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def test_every_exported_entity_has_a_sample() -> None:
    """An entity with no sample escapes every invariant in this module."""
    exported = {
        name
        for name in entities_pkg.__all__
        if isinstance(getattr(entities_pkg, name), type)
        and issubclass(getattr(entities_pkg, name), BaseModel)
    }
    missing = sorted(exported - set(SAMPLES) - {"DomainModel", "TimestampedModel"})
    assert not missing, f"add a sample to tests/fixtures/domain.py for: {missing}"


def test_the_public_surface_is_importable_from_the_package_root() -> None:
    for name in entities_pkg.__all__:
        assert hasattr(entities_pkg, name), name


def test_the_schema_version_is_recorded() -> None:
    assert DOMAIN_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Configuration invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_entity_forbids_extra_fields(name: str) -> None:
    """This is what makes "the field does not exist" an error rather than a silent drop.

    An agent emitting ``{"confidence": 0.8}`` into a finding must fail validation. Absence
    of the field plus a permissive config would enforce nothing.
    """
    model = _entity_classes()[name]
    assert model.model_config.get("extra") == "forbid", name


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_entity_is_frozen(name: str) -> None:
    """A value that can drift after validation makes a recomputed score unexplainable."""
    model = _entity_classes()[name]
    assert model.model_config.get("frozen") is True, name


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_entity_rejects_an_unknown_field(name: str) -> None:
    sample = SAMPLES[name]
    with pytest.raises(ValidationError):
        type(sample)(**{**sample.model_dump(), "surprise_field": "x"})


# ---------------------------------------------------------------------------
# Confidence placement
# ---------------------------------------------------------------------------


def test_confidence_exists_on_exactly_one_entity() -> None:
    """Checkpoint 9.1, asserted by enumeration rather than by review."""
    carriers = sorted(
        name for name, model in _entity_classes().items() if "confidence" in model.model_fields
    )
    assert carriers == ["ConsequenceScore"]


@pytest.mark.parametrize("name", sorted(AGENT_AUTHORED))
def test_no_agent_authored_entity_accepts_a_confidence_number(name: str) -> None:
    sample = SAMPLES[name]
    with pytest.raises(ValidationError):
        type(sample)(**{**sample.model_dump(), "confidence": 0.85})


@pytest.mark.parametrize("name", sorted(AGENT_AUTHORED))
def test_no_agent_authored_entity_accepts_a_consequence_class(name: str) -> None:
    """`consequence_class` is pipeline-assigned; no agent can self-declare Class A."""
    sample = SAMPLES[name]
    assert "consequence_class" not in type(sample).model_fields
    with pytest.raises(ValidationError):
        type(sample)(**{**sample.model_dump(), "consequence_class": "class_a_assumption_mediated"})


#: Entities the *pipeline* assigns a class to. Both are resolver- or judge-produced and
#: neither is reachable from an agent payload; ``ClassAFinding`` pins the value with a
#: ``Literal``, so it cannot even be constructed as Class B.
PIPELINE_CLASSIFIED = ["ClassAFinding", "Consequence"]


def test_consequence_class_appears_only_on_pipeline_assigned_entities() -> None:
    """Checkpoint 7.2: the class is assigned by the pipeline, never self-declared.

    The list is asserted exactly rather than as a subset. A new entity growing a
    ``consequence_class`` field is a decision that has to be made deliberately, and this
    test is where it gets noticed.
    """
    carriers = sorted(
        name
        for name, model in _entity_classes().items()
        if "consequence_class" in model.model_fields
    )
    assert carriers == PIPELINE_CLASSIFIED


def test_a_class_a_finding_cannot_be_constructed_as_class_b() -> None:
    """The resolver's output is Class A by construction, not by assignment."""
    sample = SAMPLES["ClassAFinding"]
    with pytest.raises(ValidationError):
        type(sample)(**{**sample.model_dump(), "consequence_class": "class_b_behavioral"})
    with pytest.raises(ValidationError):
        type(sample)(**{**sample.model_dump(), "origin": "agent"})


def test_a_class_a_finding_carries_no_score_and_no_judgment() -> None:
    """Severity is the judge's, confidence is the scorer's. The resolver supplies neither."""
    fields = SAMPLES["ClassAFinding"].__class__.model_fields
    assert "severity" not in fields
    assert "confidence" not in fields


@pytest.mark.parametrize("field", RETIRED_FIELDS)
def test_no_entity_carries_retired_vocabulary(field: str) -> None:
    offenders = sorted(
        name for name, model in _entity_classes().items() if field in model.model_fields
    )
    assert not offenders, f"{field} is retired but present on {offenders}"


def test_severity_is_confined_to_the_judged_entities() -> None:
    """Severity is judgment and belongs to the judge - but only there."""
    carriers = sorted(
        name for name, model in _entity_classes().items() if "severity" in model.model_fields
    )
    assert carriers == ["ArbitrationDecision", "Consequence"]


def test_the_confidence_scorer_cannot_reach_agent_identity() -> None:
    """Corroboration counts sources; it cannot count agents because it cannot see them."""
    reachable = _nested_models(ConsequenceScore)
    for model in reachable:
        assert not any(
            "agent" in field or "prompt" in field or "model" in field
            for field in model.model_fields
        ), model.__name__
    assert ModelProvenance not in reachable


def test_no_graph_edge_can_be_attributed_to_a_model() -> None:
    """No agent writes an edge, enforced by the absence of any provenance path."""
    assert ModelProvenance not in _nested_models(GraphEdge)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_entity_round_trips_through_json(name: str) -> None:
    """Persistence and transport must not lose or reinterpret a field."""
    sample = SAMPLES[name]
    restored = type(sample).model_validate_json(sample.model_dump_json())
    assert restored == sample


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_entity_round_trips_through_python(name: str) -> None:
    sample = SAMPLES[name]
    assert type(sample).model_validate(sample.model_dump()) == sample


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_entity_dumps_to_json_serializable_data(name: str) -> None:
    sample = SAMPLES[name]
    payload: dict[str, Any] = sample.model_dump(mode="json")
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_entity_publishes_a_json_schema(name: str) -> None:
    """The schema is what a structured-output call is validated against."""
    schema = _entity_classes()[name].model_json_schema()
    assert schema["title"]
    assert schema.get("additionalProperties") is False, name


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_no_entity_allows_assignment_after_validation(name: str) -> None:
    sample = SAMPLES[name]
    field = next(iter(type(sample).model_fields))
    with pytest.raises(ValidationError):
        setattr(sample, field, None)
