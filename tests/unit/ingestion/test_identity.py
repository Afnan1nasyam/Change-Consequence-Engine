"""Deterministic identity and canonicalization.

Two acceptance criteria rest entirely on this module, so the tests are mostly about
*stability*: the same inputs must give the same ids across processes, platforms and line
endings, and different inputs must not collide.
"""

from __future__ import annotations

import pytest

from app.domain.value_objects.identifiers import id_kind
from app.ingestion.identity import (
    PARSER_VERSION,
    assumption_id_for,
    canonical_text,
    chunk_id_for,
    content_hash,
    evidence_id_for,
    registry_gap_id_for,
    source_id_for,
    version_id_for,
)

TENANT = "tn_" + "0" * 31 + "1"
OTHER_TENANT = "tn_" + "0" * 31 + "2"


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("﻿header", "header"),
    ],
)
def test_canonicalization_folds_line_endings_and_bom(raw: str, expected: str) -> None:
    assert canonical_text(raw) == expected


def test_a_crlf_checkout_hashes_the_same_as_an_lf_checkout() -> None:
    """Otherwise every Windows clone produces a different version of every document."""
    assert content_hash("policy\r\nline two") == content_hash("policy\nline two")


def test_canonicalization_composes_unicode() -> None:
    decomposed = "é"  # e + combining acute
    composed = "é"
    assert canonical_text(decomposed) == composed
    assert content_hash(decomposed) == content_hash(composed)


def test_canonicalization_preserves_everything_span_verification_needs() -> None:
    """No whitespace collapsing, no case folding: a cited span must match verbatim."""
    raw = "Refunds  of  10,000 USD   or MORE\n\n  require approval."
    assert canonical_text(raw) == raw


def test_content_hash_is_sensitive_to_a_single_character() -> None:
    assert content_hash("threshold 10000") != content_hash("threshold 10001")


# ---------------------------------------------------------------------------
# Id derivation
# ---------------------------------------------------------------------------


def test_ids_keep_the_phase_1_shape() -> None:
    """Nothing downstream may be able to tell a derived id from a minted one."""
    source_id = source_id_for(tenant_id=TENANT, uri="corpus://a.md")
    version_id = version_id_for(source_id=source_id, content_hash_hex="ab" * 32)
    chunk_id = chunk_id_for(version_id=version_id, ordinal=0, char_start=0, char_end=10)
    evidence_id = evidence_id_for(version_id=version_id, char_start=0, char_end=10)

    assert id_kind(source_id) == "source"
    assert id_kind(version_id) == "source_version"
    assert id_kind(chunk_id) == "chunk"
    assert id_kind(evidence_id) == "evidence"


def test_source_identity_is_uri_within_tenant() -> None:
    a = source_id_for(tenant_id=TENANT, uri="corpus://policy.md")
    again = source_id_for(tenant_id=TENANT, uri="corpus://policy.md")
    assert a == again


def test_two_tenants_ingesting_the_same_uri_do_not_collide() -> None:
    """A shared source id would merge two tenants' documents into one history."""
    assert source_id_for(tenant_id=TENANT, uri="corpus://policy.md") != source_id_for(
        tenant_id=OTHER_TENANT, uri="corpus://policy.md"
    )


def test_version_identity_is_document_plus_content() -> None:
    source_id = source_id_for(tenant_id=TENANT, uri="corpus://policy.md")
    unchanged = version_id_for(source_id=source_id, content_hash_hex=content_hash("v1"))
    same = version_id_for(source_id=source_id, content_hash_hex=content_hash("v1"))
    edited = version_id_for(source_id=source_id, content_hash_hex=content_hash("v2"))

    assert unchanged == same, "re-ingesting unchanged bytes must resolve to one version"
    assert unchanged != edited, "an edit must produce a new version"


def test_evidence_identity_is_the_span_not_the_chunk() -> None:
    """A chunker change must not invalidate a citation an earlier analysis recorded."""
    version_id = "sv_" + "a" * 32
    assert evidence_id_for(version_id=version_id, char_start=10, char_end=40) == evidence_id_for(
        version_id=version_id, char_start=10, char_end=40
    )
    assert evidence_id_for(version_id=version_id, char_start=10, char_end=40) != evidence_id_for(
        version_id=version_id, char_start=11, char_end=40
    )


def test_chunk_ids_distinguish_ordinal_and_offsets() -> None:
    version_id = "sv_" + "a" * 32
    base = chunk_id_for(version_id=version_id, ordinal=0, char_start=0, char_end=10)
    assert base != chunk_id_for(version_id=version_id, ordinal=1, char_start=0, char_end=10)
    assert base != chunk_id_for(version_id=version_id, ordinal=0, char_start=0, char_end=11)


def test_id_parts_cannot_be_confused_by_concatenation() -> None:
    """('a','bc') and ('ab','c') must not hash alike, or ids collide across fields."""
    version_id = "sv_" + "a" * 32
    left = assumption_id_for(
        version_id=version_id,
        key="a.b",
        scope="global",
        role="declares",
        evidence_id="ev_" + "0" * 32,
        extractor_version="x",
    )
    right = assumption_id_for(
        version_id=version_id,
        key="a",
        scope="b",
        role="declares",
        evidence_id="ev_" + "0" * 32,
        extractor_version="x",
    )
    assert left != right


def test_assumption_identity_includes_the_extractor_version() -> None:
    """Re-extraction under a new prompt must create a record, not silently overwrite one."""
    common = {
        "version_id": "sv_" + "a" * 32,
        "key": "retention.window",
        "scope": "global",
        "role": "declares",
        "evidence_id": "ev_" + "0" * 32,
    }
    assert assumption_id_for(**common, extractor_version="extractor-1") != assumption_id_for(
        **common, extractor_version="extractor-2"
    )


def test_a_gap_from_one_span_is_one_gap() -> None:
    """Gap volume is a product signal, so a repeated proposal must not inflate it."""
    args = {
        "version_id": "sv_" + "a" * 32,
        "proposed_key": "refund.escalation_ceiling",
        "evidence_id": "ev_" + "0" * 32,
    }
    assert registry_gap_id_for(**args) == registry_gap_id_for(**args)


def test_parser_version_is_recorded() -> None:
    assert PARSER_VERSION
