"""What the adversarial challenger is allowed to say. Three outcomes, each with a price.

A model instructed to disagree produces plausible objections at roughly the rate it is asked
for them, and that is worthless. So this schema makes the citation a **structural obligation**
rather than a prompt request: the validators below refuse an uncited ``refuted``, an uncited
``weakened``, and a ``no_refutation_found`` with no falsification test - mirroring
:class:`~app.domain.entities.findings.ChallengeResult` at the boundary so a malformed
objection dies before it reaches the domain rather than after.

**There is no fourth outcome.** "I have doubts" has no representation. That is the point: a
challenger's value is measured by how many false positives it removes, not by how many
objections it raises, and an outcome for un-actionable unease would be the cheapest thing for
a model to produce and the least useful thing for a reader to receive.

**What is absent, and why absence is the safeguard.** ``extra="forbid"`` makes any of these a
validation error rather than a silently dropped key:

``confidence``
    computed by versioned code over evidence features (``CLAUDE.md`` 15, checkpoint 9.1,
    which names ``ChallengeResult`` explicitly among the models it was removed from). The
    outcome enum is what a challenger may say instead.
``severity``
    the judge's, and a judgement about business impact the corpus does not contain.
``status``
    also the judge's - and note that ``no_refutation_found`` is **not corroboration**
    (checkpoint 9.2): it means nobody found contrary evidence, which is strictly weaker than
    evidence of correctness and must never raise a status.
``consequence_class`` / ``origin``
    pipeline-assigned. Classification is by producer, and a challenger that could restate the
    class could move a candidate into or out of Class A's challenge protections.
``produced_by``
    provenance is resolved by the runtime from the agent binding. A model-supplied
    attribution is a model signing its own work.

**The falsification test is prose, not an identifier.** The model supplies the *observation*;
the runtime mints the ``TestId`` and constructs the
:class:`~app.domain.entities.findings.FalsificationTest`. A model that could mint ids could
point a challenge at a test that does not exist.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.domain.value_objects.enums import ChallengeOutcome, ClassAGround

#: Bumped when this shape changes, because it changes what the model was asked for and
#: therefore what its answer means. Pinned onto every admitted challenge.
CHALLENGE_SCHEMA_VERSION: Final = "challenge-proposal-1"

#: The fields an agent may never supply. Enforced structurally by ``extra="forbid"``; named
#: here so a test asserts the list rather than trusting the class body to stay right.
FORBIDDEN_AGENT_FIELDS: Final = (
    "confidence",
    "severity",
    "status",
    "consequence_class",
    "origin",
    "produced_by",
    "challenge_id",
    "falsification_test_id",
)

#: Bounded so one call cannot flood the report. The candidate set is finite and supplied, so
#: a batch larger than this is a model repeating itself rather than reviewing more.
MAX_CHALLENGES: Final = 32

#: The outcomes that assert something about the corpus and therefore owe a citation.
CITING_OUTCOMES: Final = frozenset({ChallengeOutcome.REFUTED, ChallengeOutcome.WEAKENED})

Handle = Annotated[str, StringConstraints(min_length=2, max_length=8, strip_whitespace=True)]
Statement = Annotated[str, StringConstraints(min_length=4, max_length=600, strip_whitespace=True)]
Reference = Annotated[str, StringConstraints(min_length=1, max_length=120, strip_whitespace=True)]


class ChallengeProposal(BaseModel):
    """One candidate, challenged. Not a :class:`ChallengeResult` until the gate admits it.

    ``target_id`` is the local handle from
    :class:`~app.challenge.candidates.ChallengeTarget`. A handle the challenger was not given
    is dropped by the runtime and reported: a verdict on a candidate that was never proposed
    has no subject, and admitting it would let a model introduce findings through the review
    channel.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: Handle
    outcome: ChallengeOutcome

    #: Required for ``refuted`` / ``weakened`` against a **Class A** candidate, and refused
    #: on a Class B one. Which of the two applies is decided by the candidate's origin, not
    #: by this field - see :mod:`app.challenge.challenger`.
    class_a_ground: ClassAGround | None = None

    #: Ids whose content the model says contradicts, narrows, or supersedes the claim.
    #: **Claimed, not verified**: the runtime keeps only ids it offered for this candidate.
    contradicting_evidence_ids: tuple[Reference, ...] = ()

    #: What specifically is contradicted. Required for ``refuted``.
    contradicts_what: Statement | None = None
    #: How the claim's scope narrows. Required for ``weakened``.
    scope_narrowing: Statement | None = None

    #: What observation would falsify the claim. Required for ``no_refutation_found``, and
    #: the reason that outcome is neither lazy nor punished: it yields a real deliverable
    #: for the validation checklist.
    falsification_observation: Statement | None = None
    #: Where that observation would have to come from, if the model can name it.
    data_source_needed: Reference | None = None

    #: Concise, reader-facing reasoning. Never chain-of-thought (``CLAUDE.md`` 14).
    rationale: Statement | None = None

    @model_validator(mode="after")
    def _outcome_requirements(self) -> Self:
        """The domain's per-outcome rules, enforced at the boundary.

        Duplicated from :class:`ChallengeResult` on purpose. Failing here rejects the batch
        and lets the runtime record *that the model produced a malformed objection*; failing
        at domain construction instead would surface as an exception from the wrong layer,
        after the gate had already decided to admit it.
        """
        if self.outcome is ChallengeOutcome.REFUTED:
            if not self.contradicting_evidence_ids:
                raise ValueError(
                    f"{self.target_id}: outcome='refuted' requires at least one contradicting "
                    "evidence id. An uncited objection is not representable."
                )
            if not self.contradicts_what:
                raise ValueError(
                    f"{self.target_id}: outcome='refuted' must state what is contradicted"
                )
        elif self.outcome is ChallengeOutcome.WEAKENED:
            if not self.contradicting_evidence_ids:
                raise ValueError(
                    f"{self.target_id}: outcome='weakened' requires at least one evidence id "
                    "for the control, exception, or newer version"
                )
            if not self.scope_narrowing:
                raise ValueError(
                    f"{self.target_id}: outcome='weakened' must state how the scope narrows"
                )
        else:
            if self.contradicting_evidence_ids:
                raise ValueError(
                    f"{self.target_id}: outcome='no_refutation_found' must carry no evidence "
                    "ids. Citing evidence while refuting nothing is a contradiction in terms."
                )
            if not self.falsification_observation:
                raise ValueError(
                    f"{self.target_id}: outcome='no_refutation_found' requires a falsification "
                    "observation. It is a legitimate outcome, but it costs a test."
                )
            if self.contradicts_what or self.scope_narrowing:
                raise ValueError(
                    f"{self.target_id}: outcome='no_refutation_found' must not assert what it "
                    "contradicts or narrows"
                )
        return self

    @property
    def owes_a_citation(self) -> bool:
        return self.outcome in CITING_OUTCOMES


class ChallengeBatch(BaseModel):
    """Everything one adversarial-challenger call returns.

    There is no field for a *new* candidate finding. The challenger's job is falsification,
    and a review stage that could also introduce findings would be able to launder an
    unreviewed claim through the one stage nothing reviews.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    challenges: tuple[ChallengeProposal, ...] = Field(default=(), max_length=MAX_CHALLENGES)

    #: Instruction-shaped text the model noticed in the evidence it was shown. Mirrors the
    #: extractor's field: corpus text is untrusted, and a span that tried to instruct the
    #: model is a security observation rather than a curiosity to drop.
    observed_instructions: tuple[Statement, ...] = ()

    #: What the supplied material could not settle. Raw material for an ``EvidenceGap``
    #: through :mod:`app.application.orchestration.evidence_gaps`; this agent does not
    #: construct one, because a gap needs the retrieval trace it does not hold.
    unanswered_questions: tuple[Statement, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The schema handed to the provider, so the ask matches what is enforced."""
        return cls.model_json_schema()
