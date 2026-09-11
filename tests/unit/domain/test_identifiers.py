"""Identifier and assumption-key shape validation.

The point of prefixed, shape-validated ids is that **a fabricated citation fails at the
schema boundary**. These tests exist to prove that an invented evidence id is a parse error
rather than something a later service has to detect.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from app.domain.value_objects.identifiers import (
    PREFIXES,
    AssumptionKey,
    EvidenceId,
    VersionLabel,
    id_kind,
    new_id,
)
from tests.fixtures.domain import evidence_item, make_id


class _Cites(BaseModel):
    evidence_id: EvidenceId


class _Keyed(BaseModel):
    key: AssumptionKey


class _Versioned(BaseModel):
    version: VersionLabel


def test_every_kind_round_trips() -> None:
    for kind in PREFIXES:
        assert id_kind(new_id(kind)) == kind


def test_new_id_refuses_an_unknown_kind() -> None:
    """Better a loud failure than a guessed prefix that silently mistypes a reference."""
    with pytest.raises(ValueError, match="unknown identifier kind"):
        new_id("nonexistent_kind")


def test_id_kind_returns_none_for_a_non_id() -> None:
    assert id_kind("EV-12") is None
    assert id_kind("") is None


@pytest.mark.parametrize(
    "fabricated",
    [
        "EV-12",  # the shape a model invents when it has no real id
        "evidence_3",
        "ev_3",  # right prefix, wrong body length
        "ev_" + "a" * 31,
        "ev_" + "a" * 33,
        "ev_" + "A" * 32,  # uppercase hex
        "ev_" + "g" * 32,  # not hex
        "sv_" + "a" * 32,  # a real id, of the wrong kind
        "",
    ],
)
def test_fabricated_evidence_ids_are_rejected(fabricated: str) -> None:
    with pytest.raises(ValidationError):
        _Cites(evidence_id=fabricated)


def test_a_well_shaped_evidence_id_is_accepted() -> None:
    assert _Cites(evidence_id=make_id("evidence")).evidence_id == make_id("evidence")


@pytest.mark.parametrize(
    "key",
    [
        "refund.approval_threshold",
        "auth.reset_token.ttl",
        "retention.window",
        "a.b",
        "sla.p1_response_time",
    ],
)
def test_valid_assumption_keys(key: str) -> None:
    assert _Keyed(key=key).key == key


@pytest.mark.parametrize(
    "key",
    [
        "Refund.Approval_Threshold",  # uppercase
        "refund",  # no dotted path: not a namespaced variable
        "refund.",
        ".threshold",
        "refund..threshold",
        "refund.approval-threshold",  # hyphen
        "refund.approval_threshold_",  # trailing underscore
        "refund._threshold",
        "1refund.threshold",  # leading digit
        "refund threshold",
        "",
    ],
)
def test_malformed_assumption_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValidationError):
        _Keyed(key=key)


def test_version_label_must_not_be_empty() -> None:
    """A version label is what makes a derivation reproducible; blank is not a version."""
    with pytest.raises(ValidationError):
        _Versioned(version="")


def test_evidence_item_requires_ordered_offsets() -> None:
    """Span verification cannot run without offsets, so a degenerate span is invalid."""
    with pytest.raises(ValidationError, match="char_end"):
        evidence_item(char_start=10, char_end=10)
    with pytest.raises(ValidationError, match="char_end"):
        evidence_item(char_start=10, char_end=4)


def test_retrieval_strategies_are_exposed_for_corroboration_counting() -> None:
    item = evidence_item(retrieval_scores={"lexical": 0.4, "vector": 0.9})
    assert item.retrieval_strategies == frozenset({"lexical", "vector"})
