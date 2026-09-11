"""The disposition rules: what a reviewer's verdict does, and what it must never do.

``IDEA.md`` 21.2 makes human disposition at review time the **primary** feedback signal, in
preference to outcome matching - because attribution is ambiguous, misses are unobservable,
and a prevented consequence is counterfactually invisible, so a system that works looks like
one producing false positives. A reviewer's verdict has none of those problems.

Five values, frozen. This module adds no sixth and interprets none of them loosely:

============================ ====================================================== ==========
value                        meaning                                                foreground
============================ ====================================================== ==========
``useful``                   actionable and not already known                        shown
``already_known``            correct, but the reviewer already had it                **shown**
``not_material``             correct, but not worth acting on                        suppressed
``wrong``                    the mechanism does not hold                             suppressed
``unverifiable``             cannot be determined from available information         shown
============================ ====================================================== ==========

**``already_known`` does not suppress, and that is the point.** It is the key signal for the
product thesis: a high rate means the system is mostly finding what a diligent reviewer would
find anyway. Suppressing it would destroy the measurement *and* hide a correct finding - the
reviewer already knew it, which is a statement about the reviewer, not about the finding.
``SUPPRESSING_DISPOSITIONS`` names exactly two values and this module reads that set rather
than restating it.

**Two suppressions, and they are different.** ``wrong`` and ``not_material`` remove a finding
from *this* analysis's foreground immediately (:func:`suppresses_foreground`). Separately, and
only when the reviewer asks for it, the same verdict suppresses the finding in *future*
analyses for the same ``(assumption_key, artifact_ref)`` pair until the source is revised -
which is what stops the same false positive being re-delivered every run until reviewers stop
reading. The frozen entity gates the second on ``suppresses_until_source_revision`` and
refuses it for the other three values.

**Nothing here deletes or edits anything.** A disposition is a record *beside* a consequence.
There is no code path in this module that constructs, copies or re-provenances a
``Consequence``, a ``ConsequenceScore``, an ``ArbitrationDecision`` or a graph record - a
suppressed finding is still in the analysis, still evidenced, still scored, and still
reachable. Suppression is a *view* decision, and a reviewer who was wrong can reverse it.

**Reversibility is what makes the signal safe to act on.** A verdict recorded with no way back
is a verdict reviewers hesitate to give, and hesitant feedback is worse than none.
:class:`DispositionRecord` keeps every superseded verdict and a :class:`DispositionChange` per
transition, so "this was marked wrong in error last Tuesday" is answerable from the record.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from app.domain.entities.findings import Consequence, FindingDisposition
from app.domain.value_objects.enums import SUPPRESSING_DISPOSITIONS, Disposition

#: Bumped when a rule here changes, because it changes what a reader is shown without any
#: finding or verdict changing.
DISPOSITION_RULES_VERSION: Final = "disposition-rules-1"

#: Verdicts that remove a finding from the foreground. Read from the frozen set rather than
#: restated, so the two cannot drift apart.
FOREGROUND_SUPPRESSING: Final = SUPPRESSING_DISPOSITIONS

#: Verdicts that require the reviewer to say why.
#:
#: ``wrong`` and ``not_material`` are the two that *remove* something a reader would otherwise
#: see, so they cost a reason - the same asymmetry the judge's ``immaterial`` carries. The
#: other three add information and need no defence. ``unverifiable`` is deliberately not in
#: here: "I could not tell" is a complete answer and demanding prose for it would push
#: reviewers toward a value they can justify rather than the one that is true.
REASON_REQUIRED: Final = SUPPRESSING_DISPOSITIONS


class DispositionError(ValueError):
    """A verdict cannot be recorded as written.

    A ``ValueError`` because nothing was attempted and no model was involved. Every case it
    covers would otherwise attach a reviewer's judgement to the wrong finding, which is worse
    than losing the verdict: it would be attributed, auditable, and about something else.
    """


class UnknownAnalysis(DispositionError):
    """The verdict names an analysis that is not the one being reviewed."""


class UnknownConsequence(DispositionError):
    """The verdict names a consequence this analysis does not contain."""


class CrossAnalysisConsequence(DispositionError):
    """The consequence exists, but in a different analysis.

    Kept apart from :class:`UnknownConsequence` because the two mean different things about
    the caller: one is a typo, the other is a wiring defect that would silently move a
    reviewer's verdict between analyses.
    """


@dataclass(frozen=True)
class DispositionChange:
    """One transition in a finding's review history.

    Both values, not just the new one: "changed to useful" is not auditable without knowing
    what it was, and the whole reason to keep a history is to answer that question later.
    """

    from_disposition: Disposition
    to_disposition: Disposition
    changed_at: dt.datetime
    changed_by: str
    reason: str | None = None

    @property
    def restored(self) -> bool:
        """Whether this transition brought a suppressed finding back into the foreground."""
        return (
            self.from_disposition in FOREGROUND_SUPPRESSING
            and self.to_disposition not in FOREGROUND_SUPPRESSING
        )

    @property
    def suppressed(self) -> bool:
        return (
            self.from_disposition not in FOREGROUND_SUPPRESSING
            and self.to_disposition in FOREGROUND_SUPPRESSING
        )


@dataclass(frozen=True)
class DispositionRecord:
    """The current verdict on one finding, and everything it used to be.

    An envelope, for the reason the other envelopes in this system exist: the frozen
    :class:`FindingDisposition` has no field for a history, and adding one would amend a
    contract to hold something the contract does not need. The audit trail lives beside it.

    ``superseded`` keeps the earlier verdicts as they were recorded, so the reviewer, the
    timestamp and the note attached to a withdrawn judgement all survive it.
    """

    current: FindingDisposition
    history: tuple[DispositionChange, ...] = ()
    superseded: tuple[FindingDisposition, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        """The identity a disposition is one-per: ``(analysis_id, consequence_id)``."""
        return (self.current.analysis_id, self.current.consequence_id)

    @property
    def disposition(self) -> Disposition:
        return self.current.disposition

    @property
    def suppresses_foreground(self) -> bool:
        return suppresses_foreground(self.current.disposition)

    @property
    def was_changed(self) -> bool:
        return bool(self.history)

    @property
    def original(self) -> Disposition:
        """The first verdict recorded. The current one when nothing has changed."""
        return self.history[0].from_disposition if self.history else self.current.disposition

    @property
    def suppression_scope(self) -> tuple[str, str] | None:
        """The ``(assumption_key, artifact_ref)`` pair a future-analysis suppression covers.

        ``None`` unless the reviewer asked for persistent suppression *and* the verdict is
        one of the two that may - the entity enforces both, and reading it back here keeps
        this module from inventing a third rule.
        """
        current = self.current
        if not current.suppresses_until_source_revision:
            return None
        return (current.assumption_key or "", current.artifact_ref or "")


def suppresses_foreground(disposition: Disposition) -> bool:
    """Whether this verdict removes a finding from the default view.

    A pure function of the value, so the answer cannot depend on who asked or when. Note what
    is *not* here: no confidence threshold, no severity, no age. A reviewer said the mechanism
    does not hold, or that it does not matter; nothing else is relevant to whether to show it.
    """
    return disposition in FOREGROUND_SUPPRESSING


def requires_reason(disposition: Disposition) -> bool:
    """Whether recording this verdict costs an explanation. The two that remove something."""
    return disposition in REASON_REQUIRED


def suppression_has_expired(record: DispositionRecord, *, artifact_ref: str | None) -> bool:
    """Whether a persistent suppression no longer applies, because the source was revised.

    ``artifact_ref`` is the affected artifact's *current* reference - in practice its source
    version id, which is content-derived, so a revised document has a different one. A
    suppression recorded against the old version therefore expires automatically: the reviewer
    said "this finding is wrong about *that* text", not "never tell me about this document".

    Returns ``False`` for a record that never asked to suppress persistently. There is nothing
    to expire, and answering ``True`` would read as "the suppression lifted".
    """
    scope = record.suppression_scope
    if scope is None:
        return False
    _key, suppressed_ref = scope
    return bool(suppressed_ref) and suppressed_ref != (artifact_ref or "")


@dataclass(frozen=True)
class ForegroundView:
    """What a reader sees, and what was withheld from them.

    ``suppressed`` is reported rather than dropped, matching
    :class:`~app.domain.entities.findings.AnalysisReport`: alert load is a correctness
    constraint, and a reader who cannot see *that* something was withheld cannot calibrate
    how much to trust what remains.
    """

    shown: tuple[Consequence, ...] = ()
    suppressed: tuple[Consequence, ...] = ()
    #: Why each suppressed finding is hidden, by consequence id.
    suppressed_by: dict[str, Disposition] = field(default_factory=dict)

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed)

    @property
    def reviewed_count(self) -> int:
        return len(self.suppressed_by)


def foreground(
    consequences: Sequence[Consequence],
    *,
    records: Iterable[DispositionRecord],
    foregrounded: Iterable[str] | None = None,
) -> ForegroundView:
    """Apply the reviewers' verdicts to a candidate view. Removes; never edits.

    ``foregrounded`` is the set of consequence ids the judge already put in the default view;
    anything outside it was never foregrounded and is not "suppressed" by a reviewer. Passing
    ``None`` treats every supplied consequence as a candidate, which is what a caller wanting
    "the whole analysis, minus what reviewers rejected" means.

    The consequences come back **as they were passed in**. Nothing in this function
    constructs, copies or mutates one: a suppressed finding is still evidenced, still scored,
    and still in the analysis - it is simply not shown first.
    """
    by_consequence = {r.current.consequence_id: r for r in records}
    candidates = set(foregrounded) if foregrounded is not None else None

    shown: list[Consequence] = []
    hidden: list[Consequence] = []
    reasons: dict[str, Disposition] = {}

    for consequence in consequences:
        if candidates is not None and consequence.consequence_id not in candidates:
            continue
        record = by_consequence.get(consequence.consequence_id)
        if record is not None and record.suppresses_foreground:
            hidden.append(consequence)
            reasons[consequence.consequence_id] = record.disposition
            continue
        shown.append(consequence)

    return ForegroundView(shown=tuple(shown), suppressed=tuple(hidden), suppressed_by=reasons)


@dataclass(frozen=True)
class DispositionSignals:
    """The review signal for one analysis, as rates rather than as a verdict.

    ``already_known_rate`` is the one the product thesis rests on: a high value means the
    system is mostly finding what a diligent reviewer would find anyway, which is the failure
    no internal metric can detect. It is reported next to the others rather than buried,
    because a system with excellent precision and a 90% already-known rate is not working.
    """

    reviewed: int = 0
    unreviewed: int = 0
    #: One entry per value, always - so a caller can read a zero rather than miss a key.
    counts: dict[Disposition, int] = field(default_factory=lambda: dict.fromkeys(Disposition, 0))

    def rate(self, disposition: Disposition) -> float | None:
        """Share of *reviewed* findings with this verdict. ``None`` when nothing was reviewed.

        Over the reviewed set rather than every finding, so an analysis nobody has looked at
        does not report a 0% already-known rate and read as a success.
        """
        if not self.reviewed:
            return None
        return self.counts.get(disposition, 0) / self.reviewed

    @property
    def already_known_rate(self) -> float | None:
        return self.rate(Disposition.ALREADY_KNOWN)

    @property
    def useful_rate(self) -> float | None:
        return self.rate(Disposition.USEFUL)

    @property
    def wrong_rate(self) -> float | None:
        return self.rate(Disposition.WRONG)

    @property
    def coverage(self) -> float | None:
        """Share of the analysis a reviewer has looked at."""
        total = self.reviewed + self.unreviewed
        return self.reviewed / total if total else None


def signals(
    consequences: Sequence[Consequence], *, records: Iterable[DispositionRecord]
) -> DispositionSignals:
    """Tally the verdicts on one analysis.

    Counts the **current** verdict per finding, not every verdict ever recorded: a reviewer
    who corrected themselves should move the rate, not add to both buckets.
    """
    by_consequence = {r.current.consequence_id: r for r in records}
    counts = dict.fromkeys(Disposition, 0)
    reviewed = 0

    for consequence in consequences:
        record = by_consequence.get(consequence.consequence_id)
        if record is None:
            continue
        counts[record.disposition] += 1
        reviewed += 1

    return DispositionSignals(
        reviewed=reviewed, unreviewed=len(consequences) - reviewed, counts=counts
    )


@runtime_checkable
class DispositionStore(Protocol):
    """Reviewer verdicts, one per ``(analysis_id, consequence_id)``.

    The port lives here rather than in ``app/domain/interfaces``, and that is deliberate: it
    speaks in :class:`DispositionRecord`, which is a judgment-layer envelope over the frozen
    entity. Putting it in the domain would make the domain import ``app.judgment`` - the
    layering contract forbids that, and correctly: the domain must not know which envelope a
    later layer wraps its entities in.

    Append-and-supersede rather than update-in-place. ``put`` replaces the *current* verdict
    and the record it stores keeps what it replaced, because a withdrawn disposition is still
    a fact about the review: "marked wrong on Tuesday, corrected on Thursday" has to be
    answerable, and an in-place update cannot answer it.

    There is deliberately **no delete**. A verdict is reversed by recording a different one,
    which leaves a trail; removing one would erase the reviewer's judgement and the audit of
    it together. And there is no mutator for the consequence side at all - a disposition sits
    beside a finding and never touches it.
    """

    def put(self, record: DispositionRecord) -> None: ...

    def get(self, analysis_id: str, consequence_id: str) -> DispositionRecord | None: ...

    def for_analysis(self, analysis_id: str) -> tuple[DispositionRecord, ...]: ...

    def find_consequence(self, consequence_id: str) -> tuple[DispositionRecord, ...]:
        """Every analysis holding a verdict on this consequence.

        The lookup that makes a cross-analysis submission detectable: a consequence id known
        under a *different* analysis is a wiring defect rather than a typo, and reporting the
        two identically would hide the one that matters.
        """
        ...
