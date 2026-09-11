"""Finalized candidates, projected into challengeable targets. Read-only, by reference.

The challenger is shown two kinds of candidate and they are not equally challengeable:

**Class B hypotheses** were proposed by a model. Anything is fair - the mechanism, the
analogue, the numbers, the whole claim.

**Class A findings** were *derived deterministically*. A derivation is only ever rule-wrong
or input-wrong, so the admissible grounds are exactly the four
:class:`~app.domain.value_objects.enums.ClassAGround` members and nothing else (checkpoint
8.1). Without that guard a model can talk down a computed result, and the deterministic layer
becomes only as strong as the most agreeable model in the pipeline.

That asymmetry is carried on the target itself, in
:attr:`ChallengeTarget.admissible_grounds`, so the gate reads it off the candidate's *origin*
rather than inferring it from anything the model said.

**Nothing here writes to a candidate.** A target holds the finding or hypothesis by
reference - both are frozen Pydantic models - and exposes projections of it. There is no
code path in ``app/challenge`` that constructs, copies-with-update, or re-provenances a
``ClassAFinding``, which is what makes "the challenger never modifies Class A provenance" a
property of the layer rather than a rule. A test dumps every candidate before and after a
full run, including on the refuted path, and compares.

**The handle the model cites is local.** Targets are ``c1``, ``c2``, ... and the prompt never
renders a ``relation_id``, a ``finding_id`` or a ``rule_id``. A model's output therefore
cannot name a Class A record at all - not to re-attribute one, not to rewrite one.

**One identity, minted once.** ``ChallengeResult`` and ``FalsificationTest`` are keyed on
``consequence_id``, because a challenge attaches to the arbitrated finding. The judge is not
built, so the id is minted here, once per candidate, and recorded on the target. The judge
must *adopt* it rather than mint a second one - a challenge pointing at an id no consequence
carries would be an orphaned proof obligation. Recorded as an open item in ``docs/STATUS.md``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding
from app.domain.value_objects.enums import ClassAGround, ConsequenceClass, Origin
from app.domain.value_objects.identifiers import new_id

#: Bumped when this projection changes, because it changes what the challenger was shown and
#: therefore what its verdict refers to.
CANDIDATE_PROJECTION_VERSION: Final = "challenge-candidate-1"

#: A handle is short on purpose: the model cites it once per candidate, and a 36-character
#: identifier would be both expensive and a name for a record it must not be able to address.
HANDLE_PREFIX: Final = "c"

#: The four admissible grounds for challenging a deterministic derivation, from the frozen
#: enum (checkpoint 8.1). Named as a set here so the gate and the prompt read the same source.
#:
#: **"Incorrect deterministic rule application" is not a fifth ground, and its absence is
#: deliberate.** Every rule mis-application that corpus evidence can support reduces to
#: disputing an *input*: a precondition that did not hold is a temporal-validity dispute, two
#: keys that are not one variable is a key-or-scope dispute, a value the span does not carry
#: is a span dispute. What remains - "the rule ran correctly on correct inputs and I disagree
#: with the output" - is a claim about the code, not about the corpus. It cannot be
#: evidence-backed, and routing it through the challenge channel would let a model's opinion
#: mark a derivation wrong without touching anything checkable. A resolver defect is a bug
#: report, not a ``ChallengeResult``.
CLASS_A_GROUNDS: Final = frozenset(ClassAGround)


@dataclass(frozen=True)
class ChallengeTarget:
    """One finalized candidate, as the challenger sees it.

    Frozen, and every field is either a projection of the candidate or an identity minted by
    the runtime. The candidate itself is reachable through ``finding`` / ``hypothesis`` for
    the application layer, and is never rendered into a prompt in a form the model could
    echo back as an identifier.
    """

    #: The local handle the model cites. Never a domain identifier.
    target_id: str
    #: The identity a ``ChallengeResult`` attaches to. Minted once; the judge adopts it.
    consequence_id: str
    consequence_class: ConsequenceClass
    origin: Origin

    #: What the challenger is trying to falsify, in prose.
    claim: str
    #: The closed set of evidence ids citable against this candidate.
    evidence_ids: tuple[str, ...]

    #: Set for a Class A candidate; ``None`` for a Class B one.
    finding: ClassAFinding | None = None
    #: Set for a Class B candidate; ``None`` for a Class A one.
    hypothesis: ConsequenceHypothesis | None = None

    #: Context the challenger needs to dispute an *input* rather than a conclusion.
    assumption_key: str | None = None
    scope: str | None = None
    as_of: dt.datetime | None = None
    #: The mechanism, for a Class B candidate only. Deliberately absent for Class A: it is
    #: precisely what may not be challenged there, so it is not put in front of the model.
    mechanism: str | None = None

    def __post_init__(self) -> None:
        if (self.finding is None) == (self.hypothesis is None):
            raise ChallengeProjectionError(
                f"{self.target_id}: a target is exactly one candidate, got "
                f"finding={'set' if self.finding else 'None'} and "
                f"hypothesis={'set' if self.hypothesis else 'None'}"
            )
        expected_origin = Origin.RESOLVER if self.finding is not None else Origin.AGENT
        if self.origin is not expected_origin:
            # Classification is by producer (``CLAUDE.md`` 22). A hypothesis wearing
            # `resolver` origin would inherit Class A's challenge protections without
            # having been derived.
            raise ChallengeProjectionError(
                f"{self.target_id}: a {'finding' if self.finding else 'hypothesis'} implies "
                f"origin={expected_origin.value!r}, got {self.origin.value!r}. "
                "Classification is by producer, not by subject matter."
            )
        expected_class = (
            ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED
            if self.origin is Origin.RESOLVER
            else ConsequenceClass.CLASS_B_BEHAVIORAL
        )
        if self.consequence_class is not expected_class:
            raise ChallengeProjectionError(
                f"{self.target_id}: origin={self.origin.value!r} implies class "
                f"{expected_class.value!r}, got {self.consequence_class.value!r}"
            )
        if self.finding is not None and self.mechanism is not None:
            raise ChallengeProjectionError(
                f"{self.target_id}: a Class A target must not carry a mechanism. The "
                "mechanism is what may not be challenged on a derivation, so showing one "
                "would invite the single inadmissible objection."
            )

    @property
    def is_class_a(self) -> bool:
        return self.origin is Origin.RESOLVER

    @property
    def requires_class_a_ground(self) -> bool:
        """Whether a ``refuted`` or ``weakened`` verdict here must name an admissible ground.

        True for exactly the deterministic candidates. Read off ``origin``, so a model's
        output cannot influence whether its own objection needs a ground.
        """
        return self.is_class_a

    @property
    def admissible_grounds(self) -> frozenset[ClassAGround]:
        """The grounds a challenge to this candidate may use.

        Empty for Class B - not because anything goes, but because a model-proposed claim
        needs no *ground*: it still needs a citation, which the outcome rules enforce
        regardless of class.
        """
        return CLASS_A_GROUNDS if self.is_class_a else frozenset()


class ChallengeProjectionError(ValueError):
    """A candidate cannot be projected as written.

    A ``ValueError`` because nothing was attempted and no model was involved. Raised rather
    than skipped: a candidate silently dropped here is a finding nobody challenged, reported
    as though it survived review.
    """


def targets_from(
    *,
    class_a: Iterable[ClassAFinding] = (),
    hypotheses: Iterable[ConsequenceHypothesis] = (),
    extra_evidence: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[ChallengeTarget, ...]:
    """Project finalized candidates into challengeable targets.

    Class A first, then Class B, each in a canonical order, so two runs over one analysis
    give the same candidate the same handle. A verdict recorded against ``c2`` in one run
    must not describe something else in the next.

    ``extra_evidence`` widens the citable set for a candidate, keyed by its source id
    (``finding_id`` or ``hypothesis_id``). This is the contract-required route for the
    challenger's actual job - a *newer source version*, an exception clause, a compensating
    control - none of which is in the candidate's own evidence set by definition. The caller
    performed any retrieval; the challenger cannot.
    """
    widen = extra_evidence or {}
    targets: list[ChallengeTarget] = []

    for finding in sorted(class_a, key=lambda f: f.finding_id):
        targets.append(
            ChallengeTarget(
                target_id=f"{HANDLE_PREFIX}{len(targets) + 1}",
                consequence_id=new_id("consequence"),
                consequence_class=ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED,
                origin=Origin.RESOLVER,
                claim=finding.summary,
                evidence_ids=_citable(
                    finding.materialized_evidence_ids, widen.get(finding.finding_id, ())
                ),
                finding=finding,
                assumption_key=finding.assumption_key,
                scope=finding.scope.render(),
                as_of=finding.as_of,
            )
        )

    for hypothesis in sorted(hypotheses, key=lambda h: h.hypothesis_id):
        targets.append(
            ChallengeTarget(
                target_id=f"{HANDLE_PREFIX}{len(targets) + 1}",
                consequence_id=new_id("consequence"),
                consequence_class=ConsequenceClass.CLASS_B_BEHAVIORAL,
                origin=Origin.AGENT,
                claim=hypothesis.claim,
                evidence_ids=_citable(
                    hypothesis.evidence_ids, widen.get(hypothesis.hypothesis_id, ())
                ),
                hypothesis=hypothesis,
                assumption_key=hypothesis.assumption_keys[0]
                if hypothesis.assumption_keys
                else None,
                mechanism=hypothesis.mechanism,
            )
        )

    return tuple(targets)


def by_handle(targets: Iterable[ChallengeTarget]) -> dict[str, ChallengeTarget]:
    """Index targets by the handle the model cites. Refuses a duplicate handle."""
    indexed: dict[str, ChallengeTarget] = {}
    for target in targets:
        if target.target_id in indexed:
            raise ChallengeProjectionError(
                f"duplicate target handle {target.target_id!r}; a handle must address one "
                "candidate or a verdict cannot be attributed"
            )
        indexed[target.target_id] = target
    return indexed


def _citable(own: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    """The candidate's own evidence plus whatever the caller supplied for it.

    Deduplicated and sorted. Both sources are the caller's: one is the finding's
    materialized union, the other is a span the caller retrieved and attached. Nothing here
    comes from the model.
    """
    return tuple(sorted(set(own) | set(extra)))
