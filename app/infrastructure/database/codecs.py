"""JSON codecs for the two stored aggregates that are not Pydantic models.

Most of what this system persists is a frozen Pydantic entity, and for those the codec is
``model_dump(mode="json")`` / ``model_validate`` - no code needed, and every read re-runs every
validator. Two aggregates are frozen *dataclasses* wrapping those entities, and dataclasses
have no JSON round trip:

* :class:`~app.judgment.dispositions.DispositionRecord` - a verdict plus its audit trail;
* :class:`~app.application.orchestration.analysis_store.StoredAnalysis` - the review snapshot
  of a finished run.

Both are envelopes by design (the frozen contracts have no field for a history or for stage
accounting), so the envelope is what needs a codec and the contents do not.

**Every decode goes through the domain constructor.** ``StageOutcome(stage=AnalysisStage(...))``,
``FindingDisposition.model_validate(...)``, and so on - so a payload that could not have been
produced by the pipeline raises on read rather than becoming a half-valid object. That is the
property the restart tests actually check: what comes back is not merely shaped like the
original, it is a value the domain would have accepted.

``ENVELOPE_VERSION`` is stamped on every payload and verified on read. A stored analysis
written by an older envelope is refused with a message naming both versions, because silently
reading a payload whose meaning has changed is how a review surface starts disagreeing with the
analysis it claims to show.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Final

from app.application.orchestration.analysis import AnalysisStage, StageOutcome, StageStatus
from app.application.orchestration.analysis_store import ReviewedFinding, StoredAnalysis
from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import (
    ArbitrationDecision,
    Consequence,
    ConsequenceScore,
    FindingDisposition,
)
from app.domain.entities.graph import ConsequenceGraph
from app.domain.entities.source import EvidenceItem, HistoricalEvent
from app.domain.value_objects.enums import Disposition
from app.domain.value_objects.provenance import RunProvenance
from app.judgment.dispositions import DispositionChange, DispositionRecord

#: Bumped when the *envelope* shape changes - not when a contained entity does, because those
#: carry their own schema version and their own validators.
ENVELOPE_VERSION: Final = "persistence-envelope-2"


class PayloadVersionError(ValueError):
    """A stored payload was written by a different envelope version."""


def _check(payload: dict[str, Any], *, what: str) -> None:
    found = payload.get("envelope_version")
    if found != ENVELOPE_VERSION:
        raise PayloadVersionError(
            f"stored {what} was written by envelope {found!r}, and this build reads "
            f"{ENVELOPE_VERSION!r}. Refusing to interpret it: a payload whose meaning may have "
            "changed is worse than a missing one, because nothing downstream would notice."
        )


def _iso(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _at(value: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(value) if value is not None else None


# ---------------------------------------------------------------------------
# DispositionRecord
# ---------------------------------------------------------------------------


def encode_disposition(record: DispositionRecord) -> dict[str, Any]:
    """Serialize a verdict with its history. Nothing is summarized away."""
    return {
        "envelope_version": ENVELOPE_VERSION,
        "current": record.current.model_dump(mode="json"),
        "superseded": [item.model_dump(mode="json") for item in record.superseded],
        "history": [
            {
                "from_disposition": change.from_disposition.value,
                "to_disposition": change.to_disposition.value,
                "changed_at": change.changed_at.isoformat(),
                "changed_by": change.changed_by,
                "reason": change.reason,
            }
            for change in record.history
        ],
    }


def decode_disposition(payload: dict[str, Any]) -> DispositionRecord:
    _check(payload, what="disposition")
    return DispositionRecord(
        current=FindingDisposition.model_validate(payload["current"]),
        history=tuple(
            DispositionChange(
                from_disposition=Disposition(change["from_disposition"]),
                to_disposition=Disposition(change["to_disposition"]),
                changed_at=dt.datetime.fromisoformat(change["changed_at"]),
                changed_by=change["changed_by"],
                reason=change["reason"],
            )
            for change in payload["history"]
        ),
        superseded=tuple(FindingDisposition.model_validate(item) for item in payload["superseded"]),
    )


# ---------------------------------------------------------------------------
# StoredAnalysis
# ---------------------------------------------------------------------------


def encode_analysis(analysis: StoredAnalysis) -> dict[str, Any]:
    """Serialize the review snapshot.

    ``fingerprint`` is written **verbatim** and never recomputed, here or on read. It is what
    lets a reviewer's verdict be checked against the analysis that produced it, so a codec that
    regenerated it would make a re-run look like the same analysis.
    """
    return {
        "envelope_version": ENVELOPE_VERSION,
        "analysis_id": analysis.analysis_id,
        "tenant_id": analysis.tenant_id,
        "run_id": analysis.run_id,
        "change": analysis.change.model_dump(mode="json"),
        "run_provenance": (
            analysis.run_provenance.model_dump(mode="json")
            if analysis.run_provenance is not None
            else None
        ),
        "stages": [
            {
                "stage": outcome.stage.value,
                "status": outcome.status.value,
                "produced": outcome.produced,
                "detail": outcome.detail,
            }
            for outcome in analysis.stages
        ],
        "findings": [
            {
                "consequence": finding.consequence.model_dump(mode="json"),
                "decision": (
                    finding.decision.model_dump(mode="json")
                    if finding.decision is not None
                    else None
                ),
                "score": (
                    finding.score.model_dump(mode="json") if finding.score is not None else None
                ),
                "artifact_version_id": finding.artifact_version_id,
                "in_default_view": finding.in_default_view,
            }
            for finding in analysis.findings
        ],
        "graph": analysis.graph.model_dump(mode="json") if analysis.graph is not None else None,
        "evidence": [item.model_dump(mode="json") for item in analysis.evidence],
        "historical_events": [
            event.model_dump(mode="json") for event in analysis.historical_events
        ],
        "evidence_gap_ids": list(analysis.evidence_gap_ids),
        "open_questions": list(analysis.open_questions),
        "fingerprint": analysis.fingerprint,
        "stored_at": _iso(analysis.stored_at),
        "pipeline_version": analysis.pipeline_version,
        "snapshot_version": analysis.snapshot_version,
    }


def decode_analysis(payload: dict[str, Any]) -> StoredAnalysis:
    _check(payload, what="analysis")
    return StoredAnalysis(
        analysis_id=payload["analysis_id"],
        tenant_id=payload["tenant_id"],
        run_id=payload["run_id"],
        change=ChangeSpec.model_validate(payload["change"]),
        run_provenance=(
            RunProvenance.model_validate(payload["run_provenance"])
            if payload["run_provenance"] is not None
            else None
        ),
        stages=tuple(
            StageOutcome(
                stage=AnalysisStage(outcome["stage"]),
                status=StageStatus(outcome["status"]),
                produced=outcome["produced"],
                detail=outcome["detail"],
            )
            for outcome in payload["stages"]
        ),
        findings=tuple(
            ReviewedFinding(
                consequence=Consequence.model_validate(finding["consequence"]),
                decision=(
                    ArbitrationDecision.model_validate(finding["decision"])
                    if finding["decision"] is not None
                    else None
                ),
                score=(
                    ConsequenceScore.model_validate(finding["score"])
                    if finding["score"] is not None
                    else None
                ),
                artifact_version_id=finding["artifact_version_id"],
                in_default_view=finding["in_default_view"],
            )
            for finding in payload["findings"]
        ),
        graph=(
            ConsequenceGraph.model_validate(payload["graph"])
            if payload["graph"] is not None
            else None
        ),
        evidence=tuple(EvidenceItem.model_validate(item) for item in payload["evidence"]),
        historical_events=tuple(
            HistoricalEvent.model_validate(event) for event in payload["historical_events"]
        ),
        evidence_gap_ids=tuple(payload["evidence_gap_ids"]),
        open_questions=tuple(payload["open_questions"]),
        fingerprint=payload["fingerprint"],
        stored_at=_at(payload["stored_at"]),
        pipeline_version=payload["pipeline_version"],
        snapshot_version=payload["snapshot_version"],
    )
