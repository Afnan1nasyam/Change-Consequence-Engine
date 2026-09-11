"""Compose the final report. Every section is a projection of a finalized finding.

The last stage, and the one with the strictest rule: it introduces no new factual claim. That
is enforced structurally rather than asked for.

**The sections are projections.** Each one holds references to objects the pipeline already
produced - a :class:`~app.domain.entities.findings.Consequence`, an
:class:`~app.domain.entities.findings.ArbitrationDecision`, a
:class:`~app.domain.entities.findings.ConsequenceScore`, a rated divergence, a challenge
result, a gap record. Nothing is re-derived, re-worded or re-scored, and there is no code path
here that constructs a claim.

**The executive summary is deterministic.** Composed from counts by :func:`summarize`. A
generated summary is the one part of a report whose truth cannot be checked against evidence,
so the report does not contain one. The ``report-composer`` agent's prose is carried *beside*
the report as advisory narration, gated on identifiers, and a reader can tell which is which.

**The agent's admitted effect is structural.** It narrows and orders the default surface,
which is real work - alert load is a correctness constraint - and both effects are checkable.
The mandatory floor (``supported`` or ``likely`` at HIGH or CRITICAL) is restored whatever it
proposes.

**Reviewer verdicts suppress the surface, never the record.** A suppressed finding leaves the
default view, stays in ``consequence_ids``, and is counted in ``suppressed_count`` with its
reason. ``already_known`` never suppresses and is reported as a measured rate, because it is
the signal the product thesis rests on.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.application.use_cases.hunt_blind_spots import BlindSpotReport
from app.blindspots.hunter import BLIND_SPOT_AGENT, SilenceRecord
from app.contradictions.hunter import AssessedDivergence
from app.domain.entities.change import ChangeSpec
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.findings import (
    AnalysisReport,
    ArbitrationDecision,
    ChallengeResult,
    Consequence,
    ConsequenceScore,
)
from app.domain.entities.source import EvidenceItem
from app.domain.value_objects.enums import (
    ConsequenceClass,
    Disposition,
    EvidencePosition,
    Severity,
    Status,
)
from app.domain.value_objects.provenance import RunProvenance
from app.judgment.arbitration import EvidenceRuling
from app.reporting.composer import (
    ComposedLayout,
    CompositionCandidate,
    ReportComposer,
)

logger = logging.getLogger(__name__)

#: Bumped when a section is added, removed, or changes what it holds.
REPORT_VERSION: Final = "analysis-report-1"

#: The surface rule, frozen. A finding at or above this cannot be taken off the default view
#: by the composer - only by a reviewer verdict, which is a human decision with an audit row.
MANDATORY_STATUSES: Final = frozenset({Status.SUPPORTED, Status.LIKELY})
MANDATORY_SEVERITIES: Final = frozenset({Severity.HIGH, Severity.CRITICAL})

#: Verdicts that take a finding off the default surface without deleting it. Read from the
#: frozen contract rather than restated: ``already_known`` is deliberately not here.
SUPPRESSING: Final = frozenset({Disposition.WRONG, Disposition.NOT_MATERIAL})

#: Truncation for a question or an action carried into ``AnalysisReport``, whose fields are
#: ``ShortText``. Truncated rather than dropped: a reader losing the tail of a question is
#: better served than one who never learns it was asked.
SHORT_TEXT_CHARS: Final = 512

#: Sections the report names. A section with nothing in it says so; it is never filled.
SECTIONS: Final = (
    "proposed_change",
    "high_priority",
    "class_a",
    "class_b",
    "blind_spots",
    "hidden",
    "contradictions",
    "challenge_outcomes",
    "historical_warnings",
    "corpus_silence",
    "evidence_gaps",
    "unresolved_questions",
    "confidence_and_severity",
    "evidence_references",
    "validation_actions",
    "reviewer_signals",
)


@dataclass(frozen=True)
class ReviewerSignals:
    """What reviewers have said, as the report carries it.

    ``measured`` distinguishes "no reviewer has looked" from "no store was wired". A rate of
    zero means something different in each case, and a reader calibrating trust in the
    product needs to know which one they are looking at.
    """

    measured: bool = False
    already_known: tuple[str, ...] = ()
    suppressed: tuple[str, ...] = ()
    #: Why each suppressed finding left the surface, by consequence id.
    suppressed_by: dict[str, str] = field(default_factory=dict)
    #: Standing suppressions this tenant's reviewers have recorded, across all analyses.
    active_suppressions: int = 0
    reviewed: int = 0

    @property
    def already_known_rate(self) -> float | None:
        """Share of reviewed findings a reviewer already knew. ``None`` when none were."""
        if not self.measured or self.reviewed == 0:
            return None
        return len(self.already_known) / self.reviewed

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed)


@dataclass(frozen=True)
class ReportedFinding:
    """One finding as the report shows it: the record, its decision, its score, its position.

    Held by reference. Confidence and severity are read off the score and the decision and
    are never recomputed, blended or rounded here.
    """

    consequence: Consequence
    decision: ArbitrationDecision | None = None
    score: ConsequenceScore | None = None
    #: The evidence arbiter's position. A second axis beside status, never merged with it.
    evidence_position: EvidencePosition | None = None
    is_blind_spot: bool = False
    suppressed_by: Disposition | None = None
    already_known: bool = False

    @property
    def consequence_id(self) -> str:
        return self.consequence.consequence_id

    @property
    def status(self) -> Status:
        return self.consequence.status

    @property
    def severity(self) -> Severity:
        return self.consequence.severity

    @property
    def confidence(self) -> float | None:
        return self.score.confidence if self.score else None

    @property
    def is_class_a(self) -> bool:
        return self.consequence.consequence_class is ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED

    @property
    def has_historical_analogue(self) -> bool:
        return self.consequence.historical_analogue_id is not None

    @property
    def is_mandatory(self) -> bool:
        """Whether the surface rule pins this finding to the default view."""
        return (
            self.status in MANDATORY_STATUSES
            and self.severity in MANDATORY_SEVERITIES
            and self.suppressed_by is None
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.consequence.evidence_ids

    @property
    def validation_actions(self) -> tuple[str, ...]:
        actions = list(self.consequence.recommended_validation)
        if self.decision is not None:
            actions.extend(self.decision.validation_actions)
        return tuple(dict.fromkeys(actions))


@dataclass(frozen=True)
class EvidenceReference:
    """One span a section cites, and which findings cite it."""

    evidence_id: str
    location: str
    authority_class: str
    cited_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportDocument:
    """The whole report: the frozen entity, plus the sections a reader reads.

    ``AnalysisReport`` is the persisted, validated record; the sections are the reader's view
    of it. Both are built from the same finalized objects, so there is no second version of
    any fact.
    """

    report: AnalysisReport
    change: ChangeSpec

    high_priority: tuple[ReportedFinding, ...] = ()
    class_a: tuple[ReportedFinding, ...] = ()
    class_b: tuple[ReportedFinding, ...] = ()
    blind_spots: tuple[ReportedFinding, ...] = ()
    hidden: tuple[ReportedFinding, ...] = ()
    historical_warnings: tuple[ReportedFinding, ...] = ()

    contradictions: tuple[AssessedDivergence, ...] = ()
    challenge_outcomes: tuple[ChallengeResult, ...] = ()
    corpus_silence: tuple[SilenceRecord, ...] = ()
    evidence_gap_ids: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    validation_actions: tuple[str, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = ()
    #: Factors nothing in this run could measure. The honest companion to every number above.
    unmeasured_confidence_factors: tuple[str, ...] = ()

    reviewer: ReviewerSignals = field(default_factory=ReviewerSignals)
    layout: ComposedLayout | None = None
    version: str = REPORT_VERSION

    @property
    def findings(self) -> tuple[ReportedFinding, ...]:
        """Every finding the report holds, Class A first, then Class B."""
        return (*self.class_a, *self.class_b)

    @property
    def empty_sections(self) -> tuple[str, ...]:
        """Sections with nothing in them. Named, because a filled section would be a lie."""
        present = {
            "proposed_change": True,
            "high_priority": bool(self.high_priority),
            "class_a": bool(self.class_a),
            "class_b": bool(self.class_b),
            "blind_spots": bool(self.blind_spots),
            "hidden": bool(self.hidden),
            "contradictions": bool(self.contradictions),
            "challenge_outcomes": bool(self.challenge_outcomes),
            "historical_warnings": bool(self.historical_warnings),
            "corpus_silence": bool(self.corpus_silence),
            "evidence_gaps": bool(self.evidence_gap_ids),
            "unresolved_questions": bool(self.unresolved_questions),
            "confidence_and_severity": any(f.score for f in self.findings),
            "evidence_references": bool(self.evidence_references),
            "validation_actions": bool(self.validation_actions),
            "reviewer_signals": self.reviewer.measured,
        }
        return tuple(name for name in SECTIONS if not present[name])

    @property
    def cited_evidence_ids(self) -> frozenset[str]:
        """Every evidence id any section cites. Nothing here may be outside the analysis."""
        found: set[str] = set()
        for finding in self.findings:
            found.update(finding.consequence.evidence_ids)
            found.update(finding.consequence.contradicting_evidence_ids)
        for reference in self.evidence_references:
            found.add(reference.evidence_id)
        for challenge in self.challenge_outcomes:
            found.update(challenge.contradicting_evidence_ids)
        return frozenset(found)

    @property
    def narration(self) -> str | None:
        """The composer's advisory summary, if one passed the gate. Never the report's own."""
        if self.layout is None or self.layout.narration is None:
            return None
        return self.layout.narration.executive_summary


class ComposeReportUseCase:
    """Compose one report. Deterministic throughout; the composer agent is optional.

    Passing no composer produces a complete report with the deterministic surface, which is
    the configuration that proves the agent adds ordering rather than content.
    """

    def __init__(self, *, composer: ReportComposer | None = None) -> None:
        self._composer = composer

    async def execute(
        self,
        *,
        analysis_id: str,
        change: ChangeSpec,
        run_provenance: RunProvenance,
        consequences: Sequence[Consequence] = (),
        decisions: Sequence[ArbitrationDecision] = (),
        scores: Sequence[ConsequenceScore] = (),
        class_a: Sequence[ClassAFinding] = (),
        contradictions: Sequence[AssessedDivergence] = (),
        challenges: Sequence[ChallengeResult] = (),
        blind_spots: BlindSpotReport | None = None,
        rulings: Mapping[str, EvidenceRuling] | None = None,
        evidence_gap_ids: Sequence[str] = (),
        open_questions: Sequence[str] = (),
        unmeasured_confidence_factors: Sequence[str] = (),
        evidence: Mapping[str, EvidenceItem] | None = None,
        reviewer: ReviewerSignals | None = None,
        correlation_id: str | None = None,
    ) -> ReportDocument:
        """Project, apply reviewer suppression, ask the composer to order, then assemble."""
        signals = reviewer or ReviewerSignals()
        findings = _project(
            consequences=consequences,
            decisions=decisions,
            scores=scores,
            rulings=rulings or {},
            reviewer=signals,
        )
        eligible = tuple(
            f
            for f in findings
            if f.suppressed_by is None
            and (f.decision is None or f.decision.in_default_view or f.is_mandatory)
        )

        layout = await self._layout(
            findings=findings,
            eligible=eligible,
            finalized_ids=_finalized_ids(
                findings, class_a=class_a, evidence_gap_ids=evidence_gap_ids
            ),
            counts=_counts(findings, contradictions=contradictions, signals=signals),
            correlation_id=correlation_id,
        )

        order = layout.foreground if layout is not None and layout.succeeded else ()
        foreground = _order(eligible, order)

        silences = blind_spots.silences if blind_spots else ()
        questions = tuple(dict.fromkeys((*open_questions, *(s.question for s in silences))))
        actions = tuple(dict.fromkeys(a for f in findings for a in f.validation_actions))

        report = AnalysisReport(
            analysis_id=analysis_id,
            change_id=change.change_id,
            executive_summary=summarize(
                findings=findings,
                foreground=foreground,
                contradictions=contradictions,
                gaps=evidence_gap_ids,
                questions=questions,
                signals=signals,
                unmeasured=unmeasured_confidence_factors,
            ),
            consequence_ids=tuple(f.consequence_id for f in findings),
            default_view_consequence_ids=tuple(f.consequence_id for f in foreground),
            contradiction_relation_ids=tuple(
                sorted(
                    {
                        d.item.relation_id
                        for d in contradictions
                        if d.is_contradiction and d.item.relation_id is not None
                    }
                )
            ),
            evidence_gap_ids=tuple(sorted(set(evidence_gap_ids))),
            unresolved_questions=tuple(q[:SHORT_TEXT_CHARS] for q in questions),
            validation_checklist=tuple(a[:SHORT_TEXT_CHARS] for a in actions),
            suppressed_count=signals.suppressed_count,
            run_provenance=run_provenance,
            metadata={
                "report_version": REPORT_VERSION,
                "blind_spot_yield": blind_spots.yield_by_slice if blind_spots else {},
                "narration_refused": (layout.narration_refusal if layout is not None else None),
            },
        )

        document = ReportDocument(
            report=report,
            change=change,
            high_priority=foreground,
            class_a=tuple(f for f in findings if f.is_class_a),
            class_b=tuple(f for f in findings if not f.is_class_a and not f.is_blind_spot),
            blind_spots=tuple(f for f in findings if f.is_blind_spot),
            hidden=tuple(f for f in findings if f not in foreground),
            historical_warnings=tuple(f for f in findings if f.has_historical_analogue),
            contradictions=tuple(contradictions),
            challenge_outcomes=tuple(challenges),
            corpus_silence=silences,
            evidence_gap_ids=tuple(sorted(set(evidence_gap_ids))),
            unresolved_questions=questions,
            validation_actions=actions,
            evidence_references=_references(findings, evidence=evidence or {}),
            unmeasured_confidence_factors=tuple(unmeasured_confidence_factors),
            reviewer=signals,
            layout=layout,
        )
        logger.info(
            "report_composed",
            extra={
                "analysis_id": analysis_id,
                "findings": len(findings),
                "foregrounded": len(foreground),
                "suppressed": signals.suppressed_count,
                "empty_sections": list(document.empty_sections),
            },
        )
        return document

    async def _layout(
        self,
        *,
        findings: tuple[ReportedFinding, ...],
        eligible: tuple[ReportedFinding, ...],
        finalized_ids: frozenset[str],
        counts: dict[str, int],
        correlation_id: str | None,
    ) -> ComposedLayout | None:
        if self._composer is None or not findings:
            return None
        handles = {f.consequence_id: f"r{index}" for index, f in enumerate(findings, start=1)}
        candidates = tuple(
            CompositionCandidate(
                handle=handles[f.consequence_id],
                consequence_id=f.consequence_id,
                title=f.consequence.title,
                status=f.status.value,
                severity=f.severity.value,
                consequence_class=f.consequence.consequence_class.value,
                confidence=f.confidence,
                owner=None,
                has_historical_analogue=f.has_historical_analogue,
                is_blind_spot=f.is_blind_spot,
                mandatory=f.is_mandatory,
                evidence_ids=f.evidence_ids,
            )
            for f in findings
        )
        return await self._composer.compose(
            candidates=candidates,
            eligible=tuple(handles[f.consequence_id] for f in eligible),
            finalized_ids=finalized_ids,
            counts=counts,
            correlation_id=correlation_id,
        )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _project(
    *,
    consequences: Sequence[Consequence],
    decisions: Sequence[ArbitrationDecision],
    scores: Sequence[ConsequenceScore],
    rulings: Mapping[str, EvidenceRuling],
    reviewer: ReviewerSignals,
) -> tuple[ReportedFinding, ...]:
    """One :class:`ReportedFinding` per consequence, joined on ``consequence_id``."""
    by_decision = {d.consequence_id: d for d in decisions}
    by_score = {s.consequence_id: s for s in scores}
    suppressed = set(reviewer.suppressed)
    known = set(reviewer.already_known)

    return tuple(
        ReportedFinding(
            consequence=consequence,
            decision=by_decision.get(consequence.consequence_id),
            score=by_score.get(consequence.consequence_id),
            evidence_position=(
                rulings[consequence.consequence_id].position
                if consequence.consequence_id in rulings
                else None
            ),
            is_blind_spot=BLIND_SPOT_AGENT in consequence.agent_sources,
            suppressed_by=(
                _suppressing_verdict(reviewer, consequence.consequence_id)
                if consequence.consequence_id in suppressed
                else None
            ),
            already_known=consequence.consequence_id in known,
        )
        for consequence in consequences
    )


def _suppressing_verdict(reviewer: ReviewerSignals, consequence_id: str) -> Disposition:
    """The verdict that took a finding off the surface, or ``WRONG`` if it was not recorded."""
    raw = reviewer.suppressed_by.get(consequence_id)
    if raw is None:
        return Disposition.WRONG
    verdict = Disposition(raw)
    if verdict not in SUPPRESSING:
        raise ValueError(
            f"{consequence_id}: {verdict.value!r} does not suppress a finding under the "
            f"frozen contract; only {sorted(d.value for d in SUPPRESSING)} do"
        )
    return verdict


def _order(
    eligible: tuple[ReportedFinding, ...], foreground: tuple[str, ...]
) -> tuple[ReportedFinding, ...]:
    """Apply the composer's order where it named one, deterministically otherwise.

    The deterministic order is severity then status then confidence then id - which is what a
    report shows when no composer ran, and the tiebreak when one did.
    """
    if not foreground:
        return tuple(sorted(eligible, key=_surface_key))
    position = {consequence_id: index for index, consequence_id in enumerate(foreground)}
    kept = [f for f in eligible if f.consequence_id in position]
    kept.sort(key=lambda f: position[f.consequence_id])
    # Anything the composer left out is appended rather than removed: an omission must not
    # be able to take a finding off the surface.
    kept.extend(sorted((f for f in eligible if f.consequence_id not in position), key=_surface_key))
    return tuple(kept)


_SEVERITY_ORDER: Final = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}
_STATUS_ORDER: Final = {
    Status.SUPPORTED: 0,
    Status.LIKELY: 1,
    Status.DISPUTED: 2,
    Status.UNRESOLVED: 3,
    Status.SPECULATIVE: 4,
    Status.REJECTED: 5,
}


def _surface_key(finding: ReportedFinding) -> tuple[int, int, float, str]:
    return (
        _SEVERITY_ORDER[finding.severity],
        _STATUS_ORDER[finding.status],
        -(finding.confidence or 0.0),
        finding.consequence_id,
    )


def _references(
    findings: tuple[ReportedFinding, ...], *, evidence: Mapping[str, EvidenceItem]
) -> tuple[EvidenceReference, ...]:
    """Every cited span, with the findings that cite it. Spans only; no new text."""
    cited: dict[str, list[str]] = {}
    for finding in findings:
        for evidence_id in (
            *finding.consequence.evidence_ids,
            *finding.consequence.contradicting_evidence_ids,
        ):
            cited.setdefault(evidence_id, []).append(finding.consequence_id)

    references: list[EvidenceReference] = []
    for evidence_id in sorted(cited):
        item = evidence.get(evidence_id)
        references.append(
            EvidenceReference(
                evidence_id=evidence_id,
                location=(item.location if item and item.location else "(unresolved span)"),
                authority_class=(
                    str(item.metadata.get("authority_class", "unknown")) if item else "unknown"
                ),
                cited_by=tuple(dict.fromkeys(cited[evidence_id])),
            )
        )
    return tuple(references)


def _finalized_ids(
    findings: tuple[ReportedFinding, ...],
    *,
    class_a: Sequence[ClassAFinding],
    evidence_gap_ids: Sequence[str],
) -> frozenset[str]:
    """Every identifier this analysis produced. What the narration gate checks against."""
    found: set[str] = set(evidence_gap_ids)
    for finding in findings:
        found.add(finding.consequence_id)
        found.update(finding.consequence.evidence_ids)
        found.update(finding.consequence.contradicting_evidence_ids)
        found.update(finding.consequence.derived_relation_ids)
        found.update(finding.consequence.hypothesis_ids)
        found.update(finding.consequence.evidence_gap_ids)
    for item in class_a:
        found.add(item.finding_id)
        found.update(item.input_record_ids)
    return frozenset(found)


def _counts(
    findings: tuple[ReportedFinding, ...],
    *,
    contradictions: Sequence[AssessedDivergence],
    signals: ReviewerSignals,
) -> dict[str, int]:
    """The figures the composer is shown. Counts only - it has no evidence to reason over."""
    return {
        "findings": len(findings),
        "class_a": sum(1 for f in findings if f.is_class_a),
        "class_b": sum(1 for f in findings if not f.is_class_a),
        "blind_spots": sum(1 for f in findings if f.is_blind_spot),
        "contradictions": sum(1 for d in contradictions if d.is_contradiction),
        "with_historical_analogue": sum(1 for f in findings if f.has_historical_analogue),
        "suppressed_by_reviewers": signals.suppressed_count,
    }


def summarize(
    *,
    findings: Sequence[ReportedFinding],
    foreground: Sequence[ReportedFinding],
    contradictions: Sequence[AssessedDivergence],
    gaps: Sequence[str],
    questions: Sequence[str],
    signals: ReviewerSignals,
    unmeasured: Sequence[str],
) -> str:
    """The report's executive summary. Counts and named facts only, assembled by code.

    Deliberately flat prose. The value of a summary a reader can trust is that every clause
    in it is a figure they can go and check, and there is no sentence here that is not one.
    """
    if not findings:
        head = "No consequence was finalized for this change."
    else:
        head = (
            f"{len(findings)} consequence(s) finalized, "
            f"{sum(1 for f in findings if f.is_class_a)} deterministic (Class A) and "
            f"{sum(1 for f in findings if not f.is_class_a)} hypothesised (Class B), of which "
            f"{sum(1 for f in findings if f.is_blind_spot)} came from the blind-spot partition. "
            f"{len(foreground)} are on the default surface."
        )

    parts = [head]
    material = sum(1 for f in findings if f.status in MANDATORY_STATUSES)
    parts.append(
        f"{material} finding(s) reached `supported` or `likely`; "
        f"{sum(1 for d in contradictions if d.is_contradiction)} contradiction(s) were rated "
        f"material; {len(set(gaps))} evidence gap(s) and {len(questions)} unresolved "
        "question(s) remain."
    )
    if unmeasured:
        parts.append(
            "Confidence factors nothing in this run could measure: "
            f"{', '.join(dict.fromkeys(unmeasured))}."
        )
    if signals.measured:
        rate = signals.already_known_rate
        parts.append(
            f"{signals.reviewed} finding(s) reviewed; {signals.suppressed_count} suppressed "
            f"from the default surface by reviewer verdict; already-known rate "
            f"{'not yet measurable' if rate is None else format(rate, '.2f')}."
        )
    else:
        parts.append("No reviewer verdicts were available for this analysis.")

    return " ".join(parts)[:2000]
