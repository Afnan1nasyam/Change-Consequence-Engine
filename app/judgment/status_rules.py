"""The frozen status gates, applied in code after the judge proposes. Not judgment.

Checkpoint 9.2 fixes three rules, and none of them is a matter of opinion:

1. **Empty evidence caps at ``speculative`` or ``unresolved``.** Never ``supported``, never
   ``likely``. Absence of contrary evidence is not support.
2. **Class B above ``speculative`` requires a historical analogue** with a recorded observed
   outcome. An analogue without an outcome does not qualify.
3. **``no_refutation_found`` is not corroboration.** It means nobody found contrary evidence,
   which is strictly weaker than evidence of correctness, and it must not raise status.

Each is arithmetic over what is present, so each is applied here rather than asked of a
model. Asking would be strictly worse: the failure mode of a model applying rule 1 is a
`supported` claim with nothing behind it, which is the single most damaging output this
system can produce, and it would be produced *fluently*.

**Every rule here can only lower a status.** :func:`clamp` never returns something stronger
than it was given - asserted by a test over the full cross-product of statuses and candidate
shapes, not just the cases the implementation happens to cover. That one-directional property
is what makes the split safe: the judge's disposition is the ceiling, and code decides how far
below it the evidence actually reaches.

**Rule 3 is enforced by absence.** There is no parameter on :func:`clamp` carrying a
``no_refutation_found`` verdict, so no branch can read one, so none can raise anything on the
strength of it. That is stronger than a rule saying not to - and a test asserts two otherwise
identical candidates get the same status whether or not a challenger cleared them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.domain.value_objects.enums import (
    ASSERTIVE_STATUSES,
    STATUSES_ABOVE_SPECULATIVE,
    ConsequenceClass,
    Status,
)

#: Bumped when a gate changes, because it changes what a status means.
STATUS_RULES_VERSION: Final = "status-rules-1"

#: Statuses ordered from weakest assertion to strongest, for the "never raise" property.
#: ``rejected`` and ``unresolved`` are *terminal* rather than weak - they are not points on
#: this scale - so they sit outside it and are handled explicitly.
_ASSERTION_ORDER: Final[tuple[Status, ...]] = (
    Status.SPECULATIVE,
    Status.DISPUTED,
    Status.LIKELY,
    Status.SUPPORTED,
)


class ClampReason:
    """Why a proposed status was lowered. Recorded, never silent.

    A clamp that a reader cannot see looks identical to a judge that was cautious, and the
    difference matters: one is the corpus being thin, the other is the judge's opinion.
    """

    NO_SUPPORTING_EVIDENCE: Final = "no_admitted_supporting_evidence"
    CLASS_B_WITHOUT_ANALOGUE: Final = "class_b_without_historical_analogue"
    REFUTED_BY_CHALLENGER: Final = "refuted_by_challenger"
    WEAKENED_BY_CHALLENGER: Final = "weakened_by_challenger"


@dataclass(frozen=True)
class ClampedStatus:
    """The status that survived the gates, and what lowered it."""

    status: Status
    proposed: Status
    reasons: tuple[str, ...] = ()

    @property
    def was_clamped(self) -> bool:
        return self.status is not self.proposed


def clamp(
    proposed: Status,
    *,
    consequence_class: ConsequenceClass,
    has_supporting_evidence: bool,
    has_historical_analogue: bool,
    was_refuted: bool = False,
    was_weakened: bool = False,
) -> ClampedStatus:
    """Apply the frozen gates to a proposed status.

    Note the parameters that are **not** here. There is no ``survived_challenge``, no
    ``no_refutation_found``, no confidence, no severity and no agent count - so none of them
    can raise anything, and rule 3 of checkpoint 9.2 holds by construction rather than by
    care.

    ``was_refuted`` and ``was_weakened`` appear only in the lowering direction: a refuted
    candidate is ``rejected``, and a weakened one cannot exceed ``disputed``.
    """
    status = proposed
    reasons: list[str] = []

    if was_refuted:
        # An admitted refutation carries a citation obligation the challenger already paid,
        # so this is evidence rather than opinion.
        status = Status.REJECTED
        reasons.append(ClampReason.REFUTED_BY_CHALLENGER)
        return ClampedStatus(status=status, proposed=proposed, reasons=tuple(reasons))

    if was_weakened and status in ASSERTIVE_STATUSES:
        status = Status.DISPUTED
        reasons.append(ClampReason.WEAKENED_BY_CHALLENGER)

    if status in ASSERTIVE_STATUSES and not has_supporting_evidence:
        # Checkpoint 9.2 rule 1. Class B with nothing behind it is speculation, which is a
        # legitimate thing to be; Class A with nothing behind it is a materialization
        # problem in the resolver, and `unresolved` is the honest label for that.
        status = (
            Status.SPECULATIVE
            if consequence_class is ConsequenceClass.CLASS_B_BEHAVIORAL
            else Status.UNRESOLVED
        )
        reasons.append(ClampReason.NO_SUPPORTING_EVIDENCE)

    if (
        consequence_class is ConsequenceClass.CLASS_B_BEHAVIORAL
        and status in STATUSES_ABOVE_SPECULATIVE
        and not has_historical_analogue
    ):
        # Checkpoint 9.2 rule 2. Note this fires on `disputed` as well as the assertive
        # statuses: `disputed` claims the effect is real and contested, which is still a
        # claim about behaviour that needs an observed precedent.
        status = Status.SPECULATIVE
        reasons.append(ClampReason.CLASS_B_WITHOUT_ANALOGUE)

    return ClampedStatus(status=status, proposed=proposed, reasons=tuple(reasons))


def is_no_stronger(result: Status, than: Status) -> bool:
    """Whether ``result`` asserts no more than ``than``. The property :func:`clamp` keeps.

    ``rejected`` and ``unresolved`` are treated as no stronger than anything, because they
    assert the claim does not hold or cannot be settled - neither is a stronger version of a
    claim that it does.
    """
    if result in (Status.REJECTED, Status.UNRESOLVED):
        return True
    if than in (Status.REJECTED, Status.UNRESOLVED):
        # ``result`` is on the scale and ``than`` asserts nothing, so this *is* a raise.
        # Reachable only if a future rule moved a status out of a terminal one, which is
        # exactly the change this function exists to catch.
        return False
    return _ASSERTION_ORDER.index(result) <= _ASSERTION_ORDER.index(than)


def in_default_view(
    *,
    proposed: bool,
    consequence_class: ConsequenceClass,
    severity_is_high: bool,
    has_historical_analogue: bool,
    status: Status,
) -> bool:
    """The report-surface rule, narrowed from what the judge proposed.

    Reviewers who receive many findings of which few matter stop reading, and the system then
    has no value regardless of its precision. So this can only ever turn the judge's
    ``True`` into ``False``:

    - a ``rejected`` finding is never foregrounded - it is reachable, and it is not news;
    - Class B reaches the default view only at HIGH or CRITICAL severity **and** with a
      historical analogue, because a speculative behavioural claim at the top of a report is
      what teaches a reviewer to stop reading the report.

    Class A at any severity may be foregrounded if the judge says so: it was derived, and the
    reader can check it.
    """
    if not proposed:
        return False
    if status is Status.REJECTED:
        return False
    if consequence_class is ConsequenceClass.CLASS_B_BEHAVIORAL:
        return severity_is_high and has_historical_analogue
    return True
