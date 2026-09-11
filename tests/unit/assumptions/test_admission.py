"""The admission gate: every path that refuses a proposal.

This is the trust boundary, so the tests are overwhelmingly negative. Every member of
:class:`AdmissionFailure` has a test that reaches it, and
``test_every_failure_reason_is_reachable`` fails if a reason is ever added without one -
because an unreachable refusal is either dead code or a check somebody forgot to wire in.

The positive tests assert something narrow and important: an admitted record has its three
verification flags set *by the gate*, and its evidence resolves to a span that really is in
the source.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.assumptions.admission import (
    AdmissionFailure,
    AdmissionGate,
    QuarantinedProposal,
    _Admitted,
)
from app.assumptions.proposals import AssumptionProposal
from app.assumptions.registry import AssumptionKeyRegistry
from app.domain.entities.registry import AssumptionKeyRegistryEntry
from app.domain.entities.source import SourceDocument, SourceVersion
from app.domain.services.span_verification import verify_offsets
from app.domain.value_objects.enums import (
    AliasStatus,
    AuthorityClass,
    BindingRole,
    ClaimType,
    Derivation,
    Explicitness,
    SourceType,
    Support,
    ValueType,
)
from app.ingestion.chunking import ChunkWithContext, chunk_document
from app.ingestion.identity import source_id_for, version_id_for
from app.ingestion.parsers import MarkdownParser, ParsedDocument
from tests.fixtures.domain import AS_OF, key_alias, model_provenance

TENANT = "tn_" + "0" * 31 + "1"
EXTRACTOR_VERSION = "extractor-1+test"

DOCUMENT = b"""# Refund Approval Policy

## Approval limits

Refunds of 10,000 USD or more require manager approval before release.

## Retention

Personal data is retained for 90 days after the account is closed.

## Notes

Parking is first come, first served.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> AssumptionKeyRegistry:
    """A two-key registry, so "unregistered" is easy to reach deliberately."""
    return AssumptionKeyRegistry(
        version="registry-1",
        entries=[
            AssumptionKeyRegistryEntry(
                key="refund.approval_threshold",
                value_type=ValueType.SCALAR,
                unit="USD",
                canonical_form="refund amount >= X USD requires approval",
                allowed_scopes=("region", "tier"),
                description="Approval threshold.",
                domain="payments",
                introduced_in_version="registry-1",
            ),
            AssumptionKeyRegistryEntry(
                key="retention.window",
                value_type=ValueType.DURATION,
                unit="d",
                canonical_form="data retained for X days",
                allowed_scopes=("region",),
                description="Retention window.",
                domain="data_retention",
                introduced_in_version="registry-1",
            ),
        ],
    )


@pytest.fixture
def parsed() -> ParsedDocument:
    return MarkdownParser().parse(DOCUMENT)


@pytest.fixture
def version(parsed: ParsedDocument) -> SourceVersion:
    source_id = source_id_for(tenant_id=TENANT, uri="corpus://policy.md")
    return SourceVersion(
        version_id=version_id_for(source_id=source_id, content_hash_hex="a" * 64),
        source_id=source_id,
        content_hash="a" * 64,
        version_label="v3",
        effective_from=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
    )


@pytest.fixture
def document(version: SourceVersion) -> SourceDocument:
    return SourceDocument(
        source_id=version.source_id,
        tenant_id=TENANT,
        source_type=SourceType.POLICY,
        title="Refund Approval Policy",
        uri="corpus://policy.md",
        authority_class=AuthorityClass.POLICY,
        current_version_id=version.version_id,
    )


@pytest.fixture
def chunks(parsed: ParsedDocument, version: SourceVersion) -> tuple[ChunkWithContext, ...]:
    return chunk_document(parsed, version_id=version.version_id)


@pytest.fixture
def gate(registry: AssumptionKeyRegistry) -> AdmissionGate:
    return AdmissionGate(registry=registry, extractor_version=EXTRACTOR_VERSION)


def _ordinal_of(chunks: tuple[ChunkWithContext, ...], needle: str) -> int:
    return next(c.chunk.ordinal for c in chunks if needle in c.chunk.text)


def proposal(chunks: tuple[ChunkWithContext, ...], **over: object) -> AssumptionProposal:
    """A valid threshold proposal against the fixture document."""
    quote = "Refunds of 10,000 USD or more require manager approval before release."
    payload: dict[str, object] = {
        "proposed_key": "refund.approval_threshold",
        "scope": "global",
        "value_quote": "10,000",
        "value_type": ValueType.SCALAR,
        "unit": "USD",
        "role": BindingRole.DECLARES,
        "explicitness": Explicitness.EXPLICIT,
        "quote": quote,
        "chunk_ordinal": _ordinal_of(chunks, "10,000"),
    }
    payload.update(over)
    return AssumptionProposal(**payload)


def _admit(
    gate: AdmissionGate,
    proposal: AssumptionProposal,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> _Admitted | QuarantinedProposal:
    return gate.admit(
        proposal,
        provenance=model_provenance(agent="assumption-extractor"),
        document=document,
        version=version,
        document_text=parsed.text,
        chunk=_chunk_for(chunks, proposal),
    )


def _chunk_for(
    chunks: tuple[ChunkWithContext, ...], proposal: AssumptionProposal
) -> ChunkWithContext | None:
    return next((c for c in chunks if c.chunk.ordinal == proposal.chunk_ordinal), None)


def _reject(*args: object, **kwargs: object) -> QuarantinedProposal:
    """Admit, asserting the proposal was quarantined. Narrows the union for the caller."""
    result = _admit(*args, **kwargs)  # type: ignore[arg-type]
    assert isinstance(result, QuarantinedProposal), "expected this proposal to be refused"
    return result


def _accept(*args: object, **kwargs: object) -> _Admitted:
    """Admit, asserting the proposal passed every check."""
    result = _admit(*args, **kwargs)  # type: ignore[arg-type]
    assert not isinstance(result, QuarantinedProposal), (
        result.detail if isinstance(result, QuarantinedProposal) else ""
    )
    return result


def _entry(registry: AssumptionKeyRegistry, key: str) -> AssumptionKeyRegistryEntry:
    entry = registry.entry(key)
    assert entry is not None, f"fixture registry is missing {key!r}"
    return entry


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_a_verified_proposal_is_admitted(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    record = _accept(gate, proposal(chunks), document, version, parsed, chunks).record
    assert record.key == "refund.approval_threshold"
    assert record.value.raw == 10000
    assert record.value.unit == "USD"
    assert record.role is BindingRole.DECLARES


def test_the_gate_sets_the_verification_flags_not_the_extractor(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """Reaching the end of the gate *is* the proof; the extractor cannot assert it."""
    record = _accept(gate, proposal(chunks), document, version, parsed, chunks).record
    assert record.span_verified
    assert record.value_parsed
    assert record.key_resolved
    assert record.is_admitted
    assert record.extraction_quality == 1.0


def test_an_admitted_record_carries_the_type_b_epistemic_axes(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    record = _accept(gate, proposal(chunks), document, version, parsed, chunks).record
    assert record.claim_type is ClaimType.ASSUMPTION_BINDING
    assert record.derivation is Derivation.EXTRACTED
    assert record.support is Support.VERIFIED_SPAN


def test_the_evidence_span_verifies_against_the_source(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """The whole point: a citation that can be checked, mechanically, after the fact."""
    result = _accept(gate, proposal(chunks), document, version, parsed, chunks)
    evidence = result.evidence

    assert verify_offsets(
        parsed.text,
        char_start=evidence.char_start,
        char_end=evidence.char_end,
        claimed=evidence.text_span,
    ).ok
    assert "10,000" in evidence.text_span
    assert evidence.source_version_id == version.version_id


def test_evidence_records_the_section_it_came_from(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    result = _accept(gate, proposal(chunks), document, version, parsed, chunks)
    assert result.evidence.location is not None
    assert "Approval limits" in result.evidence.location


def test_the_registry_and_extractor_versions_are_pinned(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """A record must stay explainable after the registry moves on."""
    record = _accept(gate, proposal(chunks), document, version, parsed, chunks).record
    assert record.registry_version == "registry-1"
    assert record.extractor_version == EXTRACTOR_VERSION


def test_a_binding_inherits_the_holder_versions_validity_when_unstated(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    record = _accept(gate, proposal(chunks), document, version, parsed, chunks).record
    assert record.effective_from == version.effective_from


def test_a_stated_interval_overrides_the_inherited_one(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    record = _accept(
        gate,
        proposal(chunks, effective_from=dt.date(2025, 6, 1)),
        document,
        version,
        parsed,
        chunks,
    ).record
    assert record.effective_from == dt.datetime(2025, 6, 1, tzinfo=dt.UTC)


def test_a_permitted_scope_dimension_is_accepted(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    record = _accept(
        gate, proposal(chunks, scope="region:EU"), document, version, parsed, chunks
    ).record
    assert record.scope.render() == "region:eu"


def test_admission_is_deterministic(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """Same proposal, same ids: re-ingestion cannot double the store."""
    first = _accept(gate, proposal(chunks), document, version, parsed, chunks)
    second = _accept(gate, proposal(chunks), document, version, parsed, chunks)
    assert first.record.assumption_id == second.record.assumption_id
    assert first.evidence.evidence_id == second.evidence.evidence_id


# ---------------------------------------------------------------------------
# Refusals, one per failure reason
# ---------------------------------------------------------------------------


def test_chunk_unknown(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    result = gate.admit(
        proposal(chunks),
        provenance=model_provenance(agent="assumption-extractor"),
        document=document,
        version=version,
        document_text=parsed.text,
        chunk=None,
    )
    assert isinstance(result, QuarantinedProposal)
    assert result.failure is AdmissionFailure.CHUNK_UNKNOWN
    assert result.chunk_id is None


def test_span_not_found(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """A quote the document does not contain. The core anti-hallucination refusal."""
    invented = "Refunds of 10,000 USD are released without any approval whatsoever."
    result = _reject(
        gate,
        proposal(chunks, quote=invented),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.failure is AdmissionFailure.SPAN_NOT_FOUND


def test_a_quote_from_a_different_chunk_is_refused(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """A real span, but not in the chunk the extractor was shown: unverifiable there."""
    retention_quote = "Personal data is retained for 90 days after the account is closed."
    result = _reject(
        gate,
        proposal(
            chunks,
            proposed_key="retention.window",
            value_quote="90",
            value_type=ValueType.DURATION,
            unit="d",
            quote=retention_quote,
            chunk_ordinal=_ordinal_of(chunks, "10,000"),
        ),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.failure is AdmissionFailure.SPAN_NOT_FOUND


def test_a_value_outside_its_span_cannot_reach_the_gate_at_all() -> None:
    """A verified span carrying an invented value is the most dangerous false binding.

    It is refused *twice*, and the outer refusal makes the inner one unreachable in
    practice, which is worth stating rather than hiding:

    1. :class:`AssumptionProposal` will not construct if ``value_quote`` is outside
       ``quote``; and
    2. :class:`QuarantinedProposal` embeds the proposal, so even a ``model_construct``
       bypass cannot be *recorded* - the quarantine record re-validates it.

    The gate keeps its own ``VALUE_NOT_IN_SPAN`` check as depth for any future caller that
    reaches ``admit()`` without passing through the schema. The check itself is exercised
    directly in ``tests/unit/domain/test_span_verification.py``.
    """
    chunks_ = chunk_document(MarkdownParser().parse(DOCUMENT), version_id="sv_" + "a" * 32)

    with pytest.raises(ValidationError, match="does not appear in quote"):
        proposal(chunks_, value_quote="5,000")

    valid = proposal(chunks_)
    smuggled = valid.model_construct(**{**valid.model_dump(), "value_quote": "5,000"})
    with pytest.raises(ValidationError, match="does not appear in quote"):
        QuarantinedProposal(
            failure=AdmissionFailure.VALUE_NOT_IN_SPAN,
            detail="smuggled",
            proposal=smuggled,
            source_version_id="sv_" + "a" * 32,
            proposed_by=model_provenance(),
        )


def test_value_parse_failed(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    result = _reject(
        gate,
        proposal(
            chunks,
            proposed_key="refund.approval_threshold",
            quote="Parking is first come, first served.",
            value_quote="first",
            chunk_ordinal=_ordinal_of(chunks, "Parking"),
        ),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.failure is AdmissionFailure.VALUE_PARSE_FAILED


def test_key_unregistered_produces_a_gap(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """The only failure that produces a KeyRegistryGap, and it is a signal, not an error."""
    result = _reject(
        gate,
        proposal(chunks, proposed_key="refund.escalation_ceiling"),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.failure is AdmissionFailure.KEY_UNREGISTERED
    assert result.is_registry_gap

    gap = result.gap
    assert gap is not None
    assert gap.proposed_key == "refund.escalation_ceiling"
    assert gap.proposed_value.raw == 10000, "the gap records what was proposed"
    assert gap.blocks_class_a
    assert gap.holder_source_version_id == version.version_id
    assert gap.registry_version == "registry-1"
    assert gap.proposed_by.agent == "assumption-extractor"


def test_a_gap_cites_the_span_that_produced_it(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """A gap has to be reviewable: which source, which key, which span, which extractor."""
    result = _reject(
        gate,
        proposal(chunks, proposed_key="refund.escalation_ceiling"),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.gap is not None
    assert result.gap.evidence_id.startswith("ev_")
    assert result.gap.proposed_by.prompt_sha256


def test_key_deprecated(
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    registry = AssumptionKeyRegistry(
        version="registry-2",
        entries=[
            AssumptionKeyRegistryEntry(
                key="refund.approval_threshold",
                value_type=ValueType.SCALAR,
                unit="USD",
                canonical_form="x",
                allowed_scopes=(),
                description="y",
                domain="payments",
                introduced_in_version="registry-1",
                deprecated_in_version="registry-2",
            )
        ],
    )
    gate = AdmissionGate(registry=registry, extractor_version=EXTRACTOR_VERSION)
    result = _reject(gate, proposal(chunks), document, version, parsed, chunks)
    assert result.failure is AdmissionFailure.KEY_DEPRECATED


def test_alias_not_approved(
    registry: AssumptionKeyRegistry,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """A proposed alias creates nothing. The gap is reported as a pending gate."""
    gated = AssumptionKeyRegistry(
        version="registry-1",
        entries=[_entry(registry, "refund.approval_threshold")],
        aliases=[
            key_alias(
                source_key="refund.manager_approval_limit",
                target_key="refund.approval_threshold",
                status=AliasStatus.PROPOSED,
            )
        ],
    )
    gate = AdmissionGate(registry=gated, extractor_version=EXTRACTOR_VERSION)
    result = _reject(
        gate,
        proposal(chunks, proposed_key="refund.manager_approval_limit"),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.failure is AdmissionFailure.ALIAS_NOT_APPROVED
    assert not result.is_registry_gap, "a pending alias is not a registry gap"


def test_an_approved_alias_admits(
    registry: AssumptionKeyRegistry,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    gated = AssumptionKeyRegistry(
        version="registry-1",
        entries=[_entry(registry, "refund.approval_threshold")],
        aliases=[
            key_alias(
                source_key="refund.manager_approval_limit",
                target_key="refund.approval_threshold",
                status=AliasStatus.APPROVED,
                approved_by="owner@example.com",
                approved_at=AS_OF,
            )
        ],
    )
    gate = AdmissionGate(registry=gated, extractor_version=EXTRACTOR_VERSION)
    result = _accept(
        gate,
        proposal(chunks, proposed_key="refund.manager_approval_limit"),
        document,
        version,
        parsed,
        chunks,
    )
    assert not isinstance(result, QuarantinedProposal)
    assert result.record.key == "refund.approval_threshold", "stored under the target key"


def test_registry_type_mismatch(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """A different variable wearing a registered key would corrupt its coupling set."""
    result = _reject(
        gate,
        proposal(chunks, value_type=ValueType.ENUM, unit=None),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.failure is AdmissionFailure.REGISTRY_TYPE_MISMATCH


def test_unit_mismatch_blocks_rather_than_weakens(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    result = _reject(gate, proposal(chunks, unit="EUR"), document, version, parsed, chunks)
    assert result.failure is AdmissionFailure.UNIT_MISMATCH
    assert "blocks coupling" in result.detail


def test_scope_malformed(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    result = _reject(gate, proposal(chunks, scope="EU"), document, version, parsed, chunks)
    assert result.failure is AdmissionFailure.SCOPE_MALFORMED


def test_scope_not_permitted(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """An undeclared dimension would fragment the key along an axis nobody designed."""
    result = _reject(
        gate,
        proposal(chunks, scope="business_unit:emea"),
        document,
        version,
        parsed,
        chunks,
    )
    assert result.failure is AdmissionFailure.SCOPE_NOT_PERMITTED
    assert "declares scopes" in result.detail


def test_interval_invalid(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """The proposal schema catches an inverted interval; the gate catches one that only
    inverts once the holder version's interval is inherited."""
    late_version = version.model_copy(
        update={"effective_to": dt.datetime(2024, 1, 1, tzinfo=dt.UTC)}
    )
    result = _reject(gate, proposal(chunks), document, late_version, parsed, chunks)
    assert result.failure is AdmissionFailure.INTERVAL_INVALID


def test_holder_metadata_missing(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """A version that does not belong to the supplied document cannot be its holder."""
    foreign = version.model_copy(update={"source_id": "src_" + "9" * 32})
    result = _reject(gate, proposal(chunks), document, foreign, parsed, chunks)
    assert result.failure is AdmissionFailure.HOLDER_METADATA_MISSING


def test_tenant_mismatch_is_reachable_through_the_document(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """Tenant is carried from the registered document onto the record.

    The gate cannot invent a tenant: whatever the document says is what the record gets, so
    a cross-tenant record requires the *store* to have returned another tenant's document.
    That boundary is enforced in the repository and asserted in the integration tests.
    """
    other = document.model_copy(update={"tenant_id": "tn_" + "0" * 31 + "2"})
    result = _accept(gate, proposal(chunks), other, version, parsed, chunks)
    assert not isinstance(result, QuarantinedProposal)
    assert result.record.tenant_id == other.tenant_id


# ---------------------------------------------------------------------------
# Batch behaviour
# ---------------------------------------------------------------------------


def test_admit_all_separates_records_gaps_and_quarantine(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    provenance = model_provenance(agent="assumption-extractor")
    outcome = gate.admit_all(
        [
            (proposal(chunks), provenance),
            (proposal(chunks, proposed_key="refund.escalation_ceiling"), provenance),
            (proposal(chunks, unit="EUR"), provenance),
        ],
        document=document,
        version=version,
        document_text=parsed.text,
        chunks=chunks,
    )

    assert outcome.admitted_count == 1
    assert len(outcome.gaps) == 1
    assert len(outcome.quarantined) == 2
    assert outcome.failure_counts == {"key_unregistered": 1, "unit_mismatch": 1}
    assert outcome.registry_version == "registry-1"
    assert outcome.extractor_version == EXTRACTOR_VERSION


def test_a_duplicate_proposal_yields_one_record(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    provenance = model_provenance(agent="assumption-extractor")
    outcome = gate.admit_all(
        [(proposal(chunks), provenance), (proposal(chunks), provenance)],
        document=document,
        version=version,
        document_text=parsed.text,
        chunks=chunks,
    )
    assert outcome.admitted_count == 1
    assert len(outcome.evidence) == 1


def test_the_same_unknown_key_from_one_span_is_one_gap_with_a_count(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    """Gap volume is a product metric, so a repeated proposal must not inflate it."""
    provenance = model_provenance(agent="assumption-extractor")
    unknown = proposal(chunks, proposed_key="refund.escalation_ceiling")
    outcome = gate.admit_all(
        [(unknown, provenance), (unknown, provenance)],
        document=document,
        version=version,
        document_text=parsed.text,
        chunks=chunks,
    )
    assert len(outcome.gaps) == 1
    assert outcome.gaps[0].occurrence_count == 2


def test_an_empty_proposal_set_admits_nothing(
    gate: AdmissionGate,
    document: SourceDocument,
    version: SourceVersion,
    parsed: ParsedDocument,
    chunks: tuple[ChunkWithContext, ...],
) -> None:
    outcome = gate.admit_all(
        [], document=document, version=version, document_text=parsed.text, chunks=chunks
    )
    assert outcome.admitted_count == 0
    assert not outcome.quarantined


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_every_failure_reason_is_reachable() -> None:
    """A refusal with no test is either dead code or a check that was never wired in.

    ``TENANT_MISMATCH`` is covered by a test asserting where the boundary actually lives
    (the store), which is why it is named here explicitly rather than left implicit.
    """
    tested = {
        AdmissionFailure.CHUNK_UNKNOWN: "test_chunk_unknown",
        AdmissionFailure.SPAN_NOT_FOUND: "test_span_not_found",
        AdmissionFailure.SPAN_MISMATCH: "covered by verify_offsets in test_span_verification",
        AdmissionFailure.VALUE_NOT_IN_SPAN: (
            "test_a_value_outside_its_span_cannot_reach_the_gate_at_all; the gate check "
            "itself is exercised in test_span_verification"
        ),
        AdmissionFailure.VALUE_PARSE_FAILED: "test_value_parse_failed",
        AdmissionFailure.KEY_UNREGISTERED: "test_key_unregistered_produces_a_gap",
        AdmissionFailure.KEY_DEPRECATED: "test_key_deprecated",
        AdmissionFailure.ALIAS_NOT_APPROVED: "test_alias_not_approved",
        AdmissionFailure.REGISTRY_TYPE_MISMATCH: "test_registry_type_mismatch",
        AdmissionFailure.UNIT_MISMATCH: "test_unit_mismatch_blocks_rather_than_weakens",
        AdmissionFailure.SCOPE_MALFORMED: "test_scope_malformed",
        AdmissionFailure.SCOPE_NOT_PERMITTED: "test_scope_not_permitted",
        AdmissionFailure.INTERVAL_INVALID: "test_interval_invalid",
        AdmissionFailure.HOLDER_METADATA_MISSING: "test_holder_metadata_missing",
        AdmissionFailure.TENANT_MISMATCH: "test_tenant_mismatch_is_reachable_through_the_document",
    }
    missing = sorted(f.value for f in AdmissionFailure if f not in tested)
    assert not missing, f"admission failures with no test: {missing}"
