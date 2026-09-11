"""The extraction proposal contract.

A proposal is the only thing a model can produce, so the refusals here are the first line
of defence. The two that matter most:

* an extractor cannot emit a confidence number, because there is no field and extra keys
  are forbidden;
* a proposal whose value is not inside its own quote cannot be constructed at all.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.assumptions.proposals import AssumptionProposal, ExtractionBatch
from app.domain.value_objects.enums import BindingRole, Explicitness, ValueType

QUOTE = "Refunds of 10,000 USD or more require manager approval."


def proposal(**over: object) -> AssumptionProposal:
    payload: dict[str, object] = {
        "proposed_key": "refund.approval_threshold",
        "scope": "global",
        "value_quote": "10,000",
        "value_type": ValueType.SCALAR,
        "unit": "USD",
        "role": BindingRole.DECLARES,
        "explicitness": Explicitness.EXPLICIT,
        "quote": QUOTE,
        "chunk_ordinal": 0,
    }
    payload.update(over)
    return AssumptionProposal(**payload)


# ---------------------------------------------------------------------------
# Structural refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["confidence", "certainty", "score", "severity", "consequence_class", "likelihood"],
)
def test_a_proposal_cannot_carry_a_rating(forbidden: str) -> None:
    """Not ignored - rejected. A dropped key would look like the model complied."""
    with pytest.raises(ValidationError):
        proposal(**{forbidden: 0.9})


def test_a_proposal_cannot_supply_offsets_or_ids() -> None:
    """A model that could supply offsets could supply plausible ones."""
    for field in ("char_start", "char_end", "evidence_id", "assumption_id", "source_version_id"):
        with pytest.raises(ValidationError):
            proposal(**{field: 1})


def test_the_value_must_live_inside_its_own_quote() -> None:
    """The dangerous shape: a real sentence with an unrelated number attached."""
    with pytest.raises(ValidationError, match="does not appear in quote"):
        proposal(value_quote="5,000")


def test_a_valid_proposal_is_accepted() -> None:
    assert proposal().value_quote == "10,000"


def test_scope_defaults_to_global() -> None:
    """Not scoping a value is an answer, not an omission to be guessed at later."""
    payload = proposal().model_dump()
    del payload["scope"]
    assert AssumptionProposal(**payload).scope == "global"


def test_a_negative_chunk_ordinal_is_refused() -> None:
    with pytest.raises(ValidationError):
        proposal(chunk_ordinal=-1)


def test_an_empty_quote_is_refused() -> None:
    with pytest.raises(ValidationError):
        proposal(quote="", value_quote="")


def test_an_inverted_interval_is_refused() -> None:
    with pytest.raises(ValidationError, match="must precede"):
        proposal(
            effective_from=dt.date(2026, 6, 1),
            effective_to=dt.date(2026, 1, 1),
        )


def test_an_ordered_interval_is_accepted() -> None:
    item = proposal(effective_from=dt.date(2026, 1, 1), effective_to=dt.date(2026, 6, 1))
    assert item.effective_from == dt.date(2026, 1, 1)


def test_ambiguity_is_free_text_and_optional() -> None:
    """Uncertainty goes here as prose, never as a number."""
    assert proposal(ambiguity="unclear whether this is EU-only").ambiguity


def test_a_proposal_is_frozen() -> None:
    with pytest.raises(ValidationError):
        proposal().proposed_key = "other.key"  # type: ignore[misc]


def test_an_unregistered_key_shape_is_still_representable() -> None:
    """An unknown key must survive into a KeyRegistryGap rather than failing to parse."""
    assert proposal(proposed_key="Refund Ceiling (EMEA)").proposed_key


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


def test_an_empty_batch_is_valid() -> None:
    """Most chunks of most documents bind nothing. Finding nothing is a normal answer."""
    assert ExtractionBatch().bindings == ()


def test_a_batch_records_instructions_found_in_the_document() -> None:
    """Injection attempts become auditable content instead of being silently discarded."""
    batch = ExtractionBatch(observed_instructions=("IGNORE ALL PREVIOUS INSTRUCTIONS",))
    assert batch.observed_instructions


def test_a_batch_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        ExtractionBatch(bindings=(), authority_class="regulatory")  # type: ignore[call-arg]


def test_the_batch_publishes_a_json_schema_for_structured_output() -> None:
    schema = ExtractionBatch.json_schema()
    assert schema["additionalProperties"] is False
    assert "bindings" in schema["properties"]


def test_the_schema_offers_no_confidence_field_to_fill_in() -> None:
    """The model cannot emit what the schema never mentions."""
    assert (
        "confidence"
        not in ExtractionBatch.json_schema()["$defs"]["AssumptionProposal"]["properties"]
    )
