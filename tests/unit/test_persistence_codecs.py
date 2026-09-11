"""The two JSON codecs, and the two properties their docstrings claim.

Round-tripping real data is covered end to end by ``tests/integration/test_persistence_restart``
(a full analysis, written by one process and read by another). What is left to assert here is
what that test cannot show by passing:

* a payload from a **different envelope version** is refused, not interpreted;
* a decode goes through the **domain constructor**, so a payload the pipeline could not have
  produced raises on read rather than becoming a half-valid object.

The second is the one that matters. A codec that reconstructed dataclasses field by field
without re-validating would happily serve a ``Consequence`` with a status no rule can produce,
and nothing downstream would notice.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from app.application.orchestration.analysis import AnalysisStage, StageOutcome, StageStatus
from app.application.orchestration.analysis_store import StoredAnalysis
from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import FindingDisposition
from app.domain.value_objects.enums import Disposition
from app.infrastructure.database.codecs import (
    ENVELOPE_VERSION,
    PayloadVersionError,
    decode_analysis,
    decode_disposition,
    encode_analysis,
    encode_disposition,
)
from app.judgment.dispositions import DispositionChange, DispositionRecord

NOW = dt.datetime(2026, 9, 11, 12, 0, tzinfo=dt.UTC)
ANALYSIS = "an_" + "a" * 32
CONSEQUENCE = "cq_" + "b" * 32


def a_disposition() -> DispositionRecord:
    current = FindingDisposition(
        disposition_id="dp_" + "c" * 32,
        consequence_id=CONSEQUENCE,
        analysis_id=ANALYSIS,
        assumption_key="retention.window",
        artifact_ref="sv_" + "d" * 32,
        disposition=Disposition.WRONG,
        note="the deck was reissued",
        disposed_by="privacy-office/reviewer-1",
        disposed_at=NOW,
        suppresses_until_source_revision=True,
    )
    return DispositionRecord(
        current=current,
        history=(
            DispositionChange(
                from_disposition=Disposition.USEFUL,
                to_disposition=Disposition.WRONG,
                changed_at=NOW,
                changed_by="privacy-office/reviewer-1",
                reason="re-read the source",
            ),
        ),
        # The withdrawn verdict, as it was recorded: `useful` never suppresses, so the
        # suppression fields have to come off with it or the entity refuses it.
        superseded=(
            current.model_copy(
                update={
                    "disposition": Disposition.USEFUL,
                    "note": None,
                    "suppresses_until_source_revision": False,
                }
            ),
        ),
    )


def an_analysis() -> StoredAnalysis:
    return StoredAnalysis(
        analysis_id=ANALYSIS,
        tenant_id="tn_" + "0" * 31 + "1",
        run_id="an_" + "e" * 32,
        change=ChangeSpec(
            change_id="chg_" + "f" * 32,
            subject_entities=("personal data retention window",),
            old_state="90 days",
            new_state="30 days",
            version="change-1",
        ),
        run_provenance=None,
        stages=(
            StageOutcome(stage=AnalysisStage.COUPLING, status=StageStatus.COMPLETED, produced=3),
            StageOutcome(stage=AnalysisStage.CLASS_B, status=StageStatus.FAILED, detail="boom"),
        ),
        findings=(),
        fingerprint={"pipeline_version": "analysis-pipeline-1", "counts": [1, 2, 3]},
        stored_at=NOW,
        pipeline_version="analysis-pipeline-1",
    )


class TestRoundTrip:
    def test_a_disposition_survives_encode_and_decode(self) -> None:
        original = a_disposition()

        assert decode_disposition(encode_disposition(original)) == original

    def test_an_analysis_survives_encode_and_decode(self) -> None:
        original = an_analysis()

        assert decode_analysis(encode_analysis(original)) == original

    def test_the_fingerprint_is_carried_verbatim(self) -> None:
        """Never recomputed: a regenerated fingerprint would make a re-run look identical."""
        original = an_analysis()

        payload = encode_analysis(original)
        assert payload["fingerprint"] == original.fingerprint
        assert decode_analysis(payload).fingerprint == original.fingerprint

    def test_the_payload_is_json_serialisable(self) -> None:
        """`mode="json"` throughout, so what is stored does not depend on the driver."""
        import json

        json.dumps(encode_analysis(an_analysis()))
        json.dumps(encode_disposition(a_disposition()))

    def test_a_failed_stage_stays_failed_rather_than_empty(self) -> None:
        decoded = decode_analysis(encode_analysis(an_analysis()))

        statuses = {outcome.stage: outcome.status for outcome in decoded.stages}
        assert statuses[AnalysisStage.CLASS_B] is StageStatus.FAILED
        assert statuses[AnalysisStage.COUPLING] is StageStatus.COMPLETED


class TestEnvelopeVersion:
    @pytest.mark.parametrize("stale", ["persistence-envelope-0", None, ""])
    def test_a_payload_from_another_envelope_version_is_refused(self, stale: Any) -> None:
        payload = encode_analysis(an_analysis())
        payload["envelope_version"] = stale

        with pytest.raises(PayloadVersionError, match=ENVELOPE_VERSION):
            decode_analysis(payload)

    def test_the_same_refusal_applies_to_a_disposition(self) -> None:
        payload = encode_disposition(a_disposition())
        del payload["envelope_version"]

        with pytest.raises(PayloadVersionError, match="disposition"):
            decode_disposition(payload)


class TestDecodeValidates:
    def test_a_payload_the_domain_would_refuse_raises_on_read(self) -> None:
        """The property the whole mapping rests on.

        A malformed identifier is the cheapest thing to corrupt; the point is that the
        *entity's* validator rejects it, so every other rule the entity encodes is enforced on
        read too - without this module knowing any of them.
        """
        payload = encode_disposition(a_disposition())
        payload["current"]["consequence_id"] = "not-an-id"

        with pytest.raises(ValidationError, match="consequence_id"):
            decode_disposition(payload)

    def test_an_unknown_stage_name_raises_rather_than_decoding(self) -> None:
        payload = encode_analysis(an_analysis())
        payload["stages"][0]["stage"] = "a_stage_that_does_not_exist"

        with pytest.raises(ValueError, match="a_stage_that_does_not_exist"):
            decode_analysis(payload)

    def test_an_unknown_disposition_value_raises(self) -> None:
        payload = encode_disposition(a_disposition())
        payload["history"][0]["to_disposition"] = "vaguely_unhelpful"

        with pytest.raises(ValueError, match="vaguely_unhelpful"):
            decode_disposition(payload)
