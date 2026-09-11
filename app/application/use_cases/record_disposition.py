"""Record a reviewer's verdict on one finding. One per (analysis, consequence), reversible.

The application path for the primary feedback signal (``IDEA.md`` 21.2). It does three things:
records a verdict, changes one, and reports what a reader should now be shown.

**Identity is the pair, not the finding.** One verdict per ``(analysis_id, consequence_id)``.
The same consequence reviewed in two analyses is two verdicts, because a reviewer's judgement
is made against what they were shown - the evidence, the status and the confidence of *that*
run - and carrying it forward silently would attribute an opinion to a report the reviewer
never saw.

**Three rejections, kept apart** (requirement 7). Each means something different about the
caller, and collapsing them would hide the one that matters:

============================== ====================================================
:class:`UnknownAnalysis`        the verdict names an analysis other than the one under review
:class:`UnknownConsequence`     this analysis has no such finding - a typo
:class:`CrossAnalysisConsequence` the finding exists, in a **different** analysis - a wiring defect
============================== ====================================================

The third is the dangerous one. Without the store's secondary index it would look identical to
a typo, and a reviewer's "this is wrong" would silently land on the right finding in the wrong
run.

**Nothing here touches a consequence.** No branch constructs, copies or re-provenances a
``Consequence``, a ``ConsequenceScore``, an ``ArbitrationDecision`` or a graph record.
Suppression is a *view* decision computed on read, so a suppressed finding is still in the
analysis - still evidenced, still scored, still projected into the graph - and a reviewer who
was wrong can reverse it without anything having been lost. A test dumps every consequence
before and after a suppressing verdict and compares.

**Idempotency is not "ignore the second one".** Re-submitting the *same* verdict is a no-op
that returns the existing record: a double-click must not fabricate a transition in the audit
trail. Submitting a *different* verdict is a change, and it is recorded as one - with both
values, the reviewer, the timestamp and the reason. There is deliberately no way to record a
verdict that leaves no trace of what it replaced.

**Two entry points, and the store-backed one is the one an application should use.**
:meth:`RecordDispositionUseCase.record` validates a verdict against a list of consequences the
caller supplies, which is honest but weak - a caller passing the wrong list would have its
verdict accepted. :meth:`RecordDispositionUseCase.record_reviewed` resolves the analysis from
an :class:`~app.application.orchestration.analysis_store.AnalysisStore` instead, so the
membership check is against **what was actually stored for that tenant**. The list-based form
remains for a caller that has no store, and both share one validation path.

**Suppression is readable here, and the rules are not redefined here.**
:meth:`RecordDispositionUseCase.is_suppressed` answers "has a reviewer already rejected this
finding for this artifact?" by reading ``DispositionRecord.suppression_scope`` and
``suppression_has_expired`` - both already implemented and tested in
:mod:`app.judgment.dispositions`. This module adds the *lookup across analyses* that a next
run needs and no new rule at all.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, replace
from typing import Final

from app.application.orchestration.analysis_store import (
    AnalysisStore,
    StoredAnalysis,
)
from app.domain.entities.findings import Consequence, FindingDisposition
from app.domain.value_objects.enums import Disposition
from app.domain.value_objects.identifiers import new_id
from app.judgment.dispositions import (
    CrossAnalysisConsequence,
    DispositionChange,
    DispositionRecord,
    DispositionSignals,
    DispositionStore,
    ForegroundView,
    UnknownAnalysis,
    UnknownConsequence,
    foreground,
    requires_reason,
    signals,
    suppression_has_expired,
)

logger = logging.getLogger(__name__)

#: Bumped when the recording path changes, because it changes what an audit trail contains.
DISPOSITION_PATH_VERSION: Final = "disposition-path-1"


@dataclass(frozen=True)
class DispositionOutcome:
    """One recorded verdict, and whether recording it changed anything.

    ``created`` / ``changed`` / ``unchanged`` are three distinct answers because a caller
    needs to tell a first verdict from a correction from a duplicate submission - and a UI
    that could not would either lose a correction or announce one that did not happen.
    """

    record: DispositionRecord
    created: bool = False
    changed: bool = False

    @property
    def unchanged(self) -> bool:
        """A re-submission of the verdict already on file. A no-op, by design."""
        return not self.created and not self.changed


@dataclass(frozen=True)
class ActiveSuppression:
    """A reviewer's standing rejection, and the artifact it was recorded against.

    Read by a later analysis to avoid re-delivering a finding somebody already rejected. It
    carries the version the verdict was about, so the expiry rule can be applied without this
    type having to know what the rule is.
    """

    assumption_key: str
    artifact_version_id: str
    disposition: Disposition
    analysis_id: str
    consequence_id: str
    disposed_by: str
    disposed_at: dt.datetime

    @property
    def scope(self) -> tuple[str, str]:
        return (self.assumption_key, self.artifact_version_id)


@dataclass(frozen=True)
class StandingVerdict:
    """Any recorded verdict, and the keys and version the reviewer was looking at.

    Broader than :class:`ActiveSuppression` on purpose: a suppression is a verdict that also
    asked to suppress, and ``already_known`` never does. Measuring the already-known rate
    across analyses needs the verdicts that suppress nothing.
    """

    assumption_keys: tuple[str, ...]
    artifact_version_id: str | None
    disposition: Disposition
    analysis_id: str
    consequence_id: str
    disposed_by: str
    disposed_at: dt.datetime

    def covers(self, *, assumption_keys: tuple[str, ...], artifact_version_id: str | None) -> bool:
        """Whether this verdict was about the same key on the same artifact version.

        Both halves are required. A verdict about an earlier revision is a verdict about
        different text, and a verdict about a different key is a different finding.
        """
        if artifact_version_id is None or self.artifact_version_id != artifact_version_id:
            return False
        return bool(set(self.assumption_keys) & set(assumption_keys))


class RecordDispositionUseCase:
    """Record, change and apply reviewer verdicts, and read standing suppressions.

    ``analyses`` is optional so a caller with no store keeps the list-based path, but supplying
    it is what makes the membership check meaningful: a verdict is then validated against the
    analysis as it was **stored for that tenant** rather than against whatever list the caller
    happened to pass.
    """

    def __init__(self, *, store: DispositionStore, analyses: AnalysisStore | None = None) -> None:
        self._store = store
        self._analyses = analyses

    def record(
        self,
        *,
        analysis_id: str,
        consequences: tuple[Consequence, ...],
        consequence_id: str,
        disposition: Disposition,
        disposed_by: str,
        disposed_at: dt.datetime,
        note: str | None = None,
        suppress_until_source_revision: bool = False,
        artifact_ref: str | None = None,
    ) -> DispositionOutcome:
        """Record a verdict, or change the one already on file.

        ``consequences`` are the findings the reviewer was shown, which is what makes the
        membership check meaningful: a verdict is valid against the analysis it was given, not
        against every analysis that ever ran.

        ``suppress_until_source_revision`` asks for the *future-analysis* suppression from
        ``IDEA.md`` 21.2. It is scoped by the finding's assumption key and ``artifact_ref``
        (in practice the affected source version id, which is content-derived - so a revised
        document has a different one and the suppression expires by itself). The frozen entity
        refuses it for any verdict other than ``wrong`` or ``not_material``.
        """
        consequence = self._locate(
            analysis_id=analysis_id,
            consequences=consequences,
            consequence_id=consequence_id,
        )
        self._check_reason(disposition, note)

        existing = self._store.get(analysis_id, consequence_id)
        proposed = FindingDisposition(
            # A new id per *verdict*, not per finding: a corrected verdict is a different
            # record, and reusing the id would make the two indistinguishable in an audit.
            disposition_id=new_id("disposition"),
            consequence_id=consequence_id,
            analysis_id=analysis_id,
            assumption_key=consequence.assumption_keys[0] if consequence.assumption_keys else None,
            artifact_ref=artifact_ref or consequence.affected_entity,
            disposition=disposition,
            note=note,
            disposed_by=disposed_by,
            disposed_at=disposed_at,
            suppresses_until_source_revision=suppress_until_source_revision,
        )

        if existing is None:
            record = DispositionRecord(current=proposed)
            self._store.put(record)
            logger.info(
                "disposition_recorded",
                extra={
                    "analysis_id": analysis_id,
                    "consequence_id": consequence_id,
                    "disposition": disposition.value,
                    "suppresses": record.suppresses_foreground,
                },
            )
            return DispositionOutcome(record=record, created=True)

        if _is_equivalent(existing.current, proposed):
            # A double-click, a retry, an idempotent replay. Returning the existing record
            # rather than appending a transition keeps the audit trail a history of *changes*
            # rather than a log of submissions.
            return DispositionOutcome(record=existing)

        return self._change(
            existing,
            proposed=proposed,
            changed_by=disposed_by,
            changed_at=disposed_at,
            reason=note,
        )

    def change(
        self,
        *,
        analysis_id: str,
        consequences: tuple[Consequence, ...],
        consequence_id: str,
        disposition: Disposition,
        changed_by: str,
        changed_at: dt.datetime,
        reason: str,
    ) -> DispositionOutcome:
        """Reverse or amend a verdict already on file. Requires a reason, always.

        Distinct from :meth:`record` even though ``record`` would also change one: an explicit
        reversal is the operation a reviewer performs when they got it wrong, and demanding a
        reason for *every* change - not only for the two suppressing values - is what makes
        the trail worth reading. A change with no reason is the one thing an audit cannot use.
        """
        if not reason.strip():
            raise UnknownConsequence(
                f"changing the verdict on {consequence_id} requires a reason: a transition "
                "with no explanation is the one thing an audit trail cannot use"
            )
        existing = self._store.get(analysis_id, consequence_id)
        if existing is None:
            raise UnknownConsequence(
                f"no verdict on record for {consequence_id} in analysis {analysis_id}; "
                "record one before changing it"
            )
        return self.record(
            analysis_id=analysis_id,
            consequences=consequences,
            consequence_id=consequence_id,
            disposition=disposition,
            disposed_by=changed_by,
            disposed_at=changed_at,
            note=reason,
            suppress_until_source_revision=(
                existing.current.suppresses_until_source_revision
                and disposition in {Disposition.WRONG, Disposition.NOT_MATERIAL}
            ),
            artifact_ref=existing.current.artifact_ref,
        )

    def record_reviewed(
        self,
        *,
        tenant_id: str,
        analysis_id: str,
        consequence_id: str,
        disposition: Disposition,
        disposed_by: str,
        disposed_at: dt.datetime,
        note: str | None = None,
        suppress_until_source_revision: bool = False,
    ) -> DispositionOutcome:
        """Record a verdict against a **stored** analysis. The path an application uses.

        The membership check is against what was stored for that tenant, so a caller cannot
        widen it by passing a different list - and the ``artifact_ref`` comes from the stored
        finding's affected source version rather than from the caller, which is what makes a
        persistent suppression expire when the document is revised.

        A Class B finding has no single source version, so its ``artifact_ref`` falls back to
        the affected entity. A suppression on one therefore cannot expire on source revision,
        which is honest: a hypothesis spans several artifacts and no single revision refutes
        it.
        """
        stored = self._require_stored(tenant_id=tenant_id, analysis_id=analysis_id)
        return self.record(
            analysis_id=analysis_id,
            consequences=stored.consequences,
            consequence_id=consequence_id,
            disposition=disposition,
            disposed_by=disposed_by,
            disposed_at=disposed_at,
            note=note,
            suppress_until_source_revision=suppress_until_source_revision,
            artifact_ref=stored.artifact_version_of(consequence_id),
        )

    def foreground_reviewed(self, *, tenant_id: str, analysis_id: str) -> ForegroundView:
        """Reconstruct exactly what a reviewer should now see for a stored analysis.

        The judge's default view, narrowed by the verdicts recorded since. Both halves come
        from the store, so the answer does not depend on the caller remembering what was
        foregrounded.
        """
        stored = self._require_stored(tenant_id=tenant_id, analysis_id=analysis_id)
        return foreground(
            stored.consequences,
            records=self._store.for_analysis(analysis_id),
            foregrounded=stored.foregrounded,
        )

    def signals_reviewed(self, *, tenant_id: str, analysis_id: str) -> DispositionSignals:
        stored = self._require_stored(tenant_id=tenant_id, analysis_id=analysis_id)
        return signals(stored.consequences, records=self._store.for_analysis(analysis_id))

    # -- standing suppressions, for the next analysis --------------------

    def suppressions(self, *, tenant_id: str) -> tuple[ActiveSuppression, ...]:
        """Every standing suppression this tenant's reviewers have recorded.

        Requires the analysis store: a suppression is scoped to an artifact, and the artifact
        a verdict was about is a property of the analysis it was recorded against. Without it
        there is no way to answer "which artifact?" and the read would be a guess.

        Reads ``DispositionRecord.suppression_scope`` and adds nothing to it. The rule for
        *whether* a verdict may suppress lives on the frozen entity; the rule for *when* it
        expires lives in :mod:`app.judgment.dispositions`. This is the lookup, not a third
        rule.
        """
        if self._analyses is None:
            raise UnknownAnalysis(
                "reading standing suppressions needs an analysis store: a suppression is "
                "scoped to an artifact, and which artifact is a property of the analysis the "
                "verdict was recorded against"
            )

        found: list[ActiveSuppression] = []
        for analysis_id in self._analyses.for_tenant(tenant_id):
            stored = self._analyses.require(tenant_id, analysis_id)
            for record in self._store.for_analysis(analysis_id):
                scope = record.suppression_scope
                if scope is None:
                    continue
                key, artifact = scope
                if not artifact:
                    continue
                found.append(
                    ActiveSuppression(
                        assumption_key=key,
                        artifact_version_id=artifact,
                        disposition=record.disposition,
                        analysis_id=stored.analysis_id,
                        consequence_id=record.current.consequence_id,
                        disposed_by=record.current.disposed_by,
                        disposed_at=record.current.disposed_at,
                    )
                )
        return tuple(sorted(found, key=lambda s: (s.assumption_key, s.artifact_version_id)))

    def verdicts(self, *, tenant_id: str) -> tuple[StandingVerdict, ...]:
        """Every verdict this tenant's reviewers have recorded, with what it was about.

        The general read :meth:`suppressions` is the suppressing subset of. It exists because
        ``already_known`` never suppresses and is therefore invisible to that lookup - and it
        is the one verdict the product thesis rests on, so a later analysis has to be able to
        see that it is re-reporting something a reviewer already knew.

        Joined through the stored analysis, so the key and the artifact version come from what
        the reviewer was shown rather than from a caller's recollection.
        """
        if self._analyses is None:
            raise UnknownAnalysis(
                "reading standing verdicts needs an analysis store: what a verdict was about "
                "is a property of the analysis it was recorded against"
            )

        found: list[StandingVerdict] = []
        for analysis_id in self._analyses.for_tenant(tenant_id):
            stored = self._analyses.require(tenant_id, analysis_id)
            for record in self._store.for_analysis(analysis_id):
                consequence_id = record.current.consequence_id
                reviewed = stored.finding(consequence_id)
                keys = reviewed.consequence.assumption_keys if reviewed else ()
                found.append(
                    StandingVerdict(
                        assumption_keys=tuple(keys),
                        artifact_version_id=stored.artifact_version_of(consequence_id),
                        disposition=record.disposition,
                        analysis_id=stored.analysis_id,
                        consequence_id=consequence_id,
                        disposed_by=record.current.disposed_by,
                        disposed_at=record.current.disposed_at,
                    )
                )
        return tuple(sorted(found, key=lambda v: (v.analysis_id, v.consequence_id)))

    def is_suppressed(
        self, *, tenant_id: str, assumption_key: str, artifact_version_id: str | None
    ) -> bool:
        """Whether a reviewer has already rejected this finding for **this** artifact version.

        The question the next analysis asks. A suppression recorded against an earlier version
        does not answer it: the reviewer said the finding was wrong about *that* text, and a
        revised document is different text. Expiry is delegated to
        ``suppression_has_expired`` so the rule has exactly one implementation.

        ``None`` for the version - a Class B finding, which has no single source version -
        matches nothing, so a hypothesis is never silently suppressed by a verdict about a
        document.
        """
        if artifact_version_id is None:
            return False
        for suppression in self.suppressions(tenant_id=tenant_id):
            if suppression.assumption_key != assumption_key:
                continue
            record = self._store.get(suppression.analysis_id, suppression.consequence_id)
            if record is None:  # pragma: no cover - listed from the same store
                continue
            if not suppression_has_expired(record, artifact_ref=artifact_version_id):
                return True
        return False

    def _require_stored(self, *, tenant_id: str, analysis_id: str) -> StoredAnalysis:
        """The stored analysis, or a refusal a caller cannot ignore."""
        if self._analyses is None:
            raise UnknownAnalysis(
                "this use case was built without an analysis store; use record(...) with the "
                "consequences the reviewer was shown, or supply a store"
            )
        return self._analyses.require(tenant_id, analysis_id)

    # -- reads -----------------------------------------------------------

    def foreground(
        self,
        *,
        analysis_id: str,
        consequences: tuple[Consequence, ...],
        foregrounded: tuple[str, ...] | None = None,
    ) -> ForegroundView:
        """What a reader should be shown now, with the reviewers' verdicts applied."""
        return foreground(
            consequences,
            records=self._store.for_analysis(analysis_id),
            foregrounded=foregrounded,
        )

    def signals(
        self, *, analysis_id: str, consequences: tuple[Consequence, ...]
    ) -> DispositionSignals:
        """The review rates for one analysis, including ``already_known_rate``."""
        return signals(consequences, records=self._store.for_analysis(analysis_id))

    def records(self, *, analysis_id: str) -> tuple[DispositionRecord, ...]:
        return self._store.for_analysis(analysis_id)

    # -- internals -------------------------------------------------------

    def _locate(
        self,
        *,
        analysis_id: str,
        consequences: tuple[Consequence, ...],
        consequence_id: str,
    ) -> Consequence:
        """Find the finding this verdict is about, or refuse in the most specific way.

        Order matters. The analysis is checked first, because a verdict aimed at the wrong
        analysis says nothing about whether the consequence exists - and reporting "unknown
        consequence" there would send the caller looking in the wrong place.
        """
        if not consequences:
            raise UnknownAnalysis(
                f"analysis {analysis_id} was supplied with no consequences; there is nothing "
                "to review"
            )
        analyses = {c.consequence_id: c for c in consequences}
        change_ids = {c.change_id for c in consequences}
        _ = change_ids  # kept for the log context below; not a validation input

        found = analyses.get(consequence_id)
        if found is not None:
            return found

        elsewhere = self._store.find_consequence(consequence_id)
        if elsewhere:
            raise CrossAnalysisConsequence(
                f"consequence {consequence_id} belongs to analysis "
                f"{sorted({r.current.analysis_id for r in elsewhere})}, not to {analysis_id}. "
                "A verdict recorded across analyses would attribute a reviewer's judgement to "
                "a report they never saw."
            )
        raise UnknownConsequence(f"analysis {analysis_id} contains no consequence {consequence_id}")

    def _check_reason(self, disposition: Disposition, note: str | None) -> None:
        """The two verdicts that remove something from a reader's view cost an explanation.

        Symmetric with the judge's ``immaterial``: the direction that *subtracts* is the one
        that must be justified. The other three add information and need no defence.
        """
        if requires_reason(disposition) and not (note or "").strip():
            raise UnknownConsequence(
                f"disposition {disposition.value!r} suppresses the finding from the default "
                "view and therefore requires a note saying why"
            )

    def _change(
        self,
        existing: DispositionRecord,
        *,
        proposed: FindingDisposition,
        changed_by: str,
        changed_at: dt.datetime,
        reason: str | None,
    ) -> DispositionOutcome:
        """Supersede the current verdict, keeping it and recording the transition."""
        transition = DispositionChange(
            from_disposition=existing.current.disposition,
            to_disposition=proposed.disposition,
            changed_at=changed_at,
            changed_by=changed_by,
            reason=reason,
        )
        record = replace(
            existing,
            current=proposed,
            history=(*existing.history, transition),
            superseded=(*existing.superseded, existing.current),
        )
        self._store.put(record)
        logger.info(
            "disposition_changed",
            extra={
                "analysis_id": proposed.analysis_id,
                "consequence_id": proposed.consequence_id,
                "from": transition.from_disposition.value,
                "to": transition.to_disposition.value,
                "restored": transition.restored,
            },
        )
        return DispositionOutcome(record=record, changed=True)


def _is_equivalent(existing: FindingDisposition, proposed: FindingDisposition) -> bool:
    """Whether a submission says the same thing as the verdict already on file.

    Compares what a reviewer actually decided - the value, the note and the suppression
    request - and deliberately not the identity or the timestamp: a retry seconds later
    carries a new id and a new clock reading, and treating those as a change would fill the
    audit trail with transitions nobody made.
    """
    return (
        existing.disposition is proposed.disposition
        and (existing.note or "") == (proposed.note or "")
        and existing.suppresses_until_source_revision == proposed.suppresses_until_source_revision
    )
