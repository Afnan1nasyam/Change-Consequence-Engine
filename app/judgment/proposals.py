"""What the consequence judge is allowed to say: a status, a severity, and its reasons.

The judge is the only agent that assigns anything a reader acts on directly, which makes the
shape of what it *cannot* say the most important part of this module.

**Severity is the judge's. Confidence is not.** The asymmetry is deliberate and is recorded
in four places now so it is not "fixed" in the wrong direction (checkpoint 9.4,
``.claude/agents/consequence-judge.md``, :mod:`app.domain.services.confidence`, here):

- **confidence** is a function of evidence properties the system can measure, so it is
  computed by a versioned deterministic function. There is no field for it, and
  ``extra="forbid"`` makes an attempt a rejected batch.
- **severity** is a function of business impact the corpus does not contain, so it is a
  judgment. It is the judge's, constrained to four values, attributed to its prompt and
  model, and reviewer-overridable.

They are never blended. HIGH severity with LOW confidence is a legitimate and important
output, and nothing in this module or the runtime moves one because of the other.

**What else is absent.** ``consequence_class`` and ``origin`` are pipeline-assigned from what
*produced* the finding (``CLAUDE.md`` 22): a consequence does not become Class A by being easy
to explain, so the judge has nowhere to declare one. ``consequence_id``, ``challenge_id`` and
every other identifier are absent too - the judge addresses a candidate by a local handle, so
its output cannot name a domain record, which is the first step to reattributing one.

**Status is proposed, then clamped.** The judge proposes a status and
:mod:`app.judgment.status_rules` applies the frozen gates afterwards. That split exists
because the gates are not judgment: "no evidence caps at speculative" and "Class B without an
analogue caps at speculative" are arithmetic over what is present, and asking a model to
apply them correctly every time is strictly worse than applying them in code.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.domain.value_objects.enums import Severity, Status

#: Bumped when this shape changes, because it changes what the model was asked for and
#: therefore what its answer means. Pinned onto every admitted disposition.
DISPOSITION_SCHEMA_VERSION: Final = "judgment-disposition-1"

#: The fields a judge may never supply. Enforced structurally by ``extra="forbid"``; named
#: here so a test asserts the list rather than trusting the class body to stay right.
FORBIDDEN_AGENT_FIELDS: Final = (
    "confidence",
    "confidence_fn_version",
    "consequence_class",
    "origin",
    "consequence_id",
    "challenge_id",
    "derived_relation_ids",
    "hypothesis_ids",
    "produced_by",
    "likelihood",
)

#: Bounded so one call cannot flood the report. The candidate set is finite and supplied.
MAX_DISPOSITIONS: Final = 32

#: A high-severity finding left unresolved owes a concrete next step. Mirrors the domain
#: validator on :class:`~app.domain.entities.findings.ArbitrationDecision`, checked here so
#: the batch is rejected at the boundary rather than at construction.
SEVERITIES_NEEDING_ACTION: Final = frozenset({Severity.HIGH, Severity.CRITICAL})

Handle = Annotated[str, StringConstraints(min_length=2, max_length=8, strip_whitespace=True)]
Statement = Annotated[str, StringConstraints(min_length=4, max_length=600, strip_whitespace=True)]
Reference = Annotated[str, StringConstraints(min_length=1, max_length=120, strip_whitespace=True)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512, strip_whitespace=True)]


class DispositionProposal(BaseModel):
    """One candidate, dispositioned. Not an ``ArbitrationDecision`` until the gate admits it.

    ``handle`` is the local reference from
    :class:`~app.judgment.dossier.CandidateDossier`. A handle the judge was not given is
    dropped by the runtime and reported: a disposition of a candidate that was never
    assembled has no subject, and admitting it would let the last stage introduce findings
    that nothing reviewed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    handle: Handle
    #: **Proposed.** The frozen gates are applied afterwards, in code.
    status: Status
    #: The judge's, and final. Four values, never a number, never moved by confidence.
    severity: Severity

    #: Why. Required, because an unexplained disposition is unauditable.
    rationale: Statement

    #: Evidence the judge says supports the claim. **Claimed, not verified**: the runtime
    #: keeps only ids offered for this candidate, and an assertive status with nothing
    #: admitted is clamped rather than trusted.
    supporting_evidence_ids: tuple[Reference, ...] = ()
    #: Evidence against. Kept apart from support so the two cannot be confused, which is
    #: what lets "disputed" mean something specific.
    contradicting_evidence_ids: tuple[Reference, ...] = ()

    #: Material disagreement, in the judge's words. The runtime **adds** to this from what
    #: upstream disputed; it never removes. Resolving a disagreement by averaging is the
    #: failure this field exists to prevent.
    preserved_disagreements: tuple[ShortText, ...] = ()
    #: What source or check would settle an unresolved finding.
    validation_actions: tuple[ShortText, ...] = ()

    #: Whether this belongs in the default view. Proposed, then narrowed by the runtime's
    #: report-surface rule: reviewers who receive many findings of which few matter stop
    #: reading, and the system then has no value regardless of its precision.
    in_default_view: bool = False

    @model_validator(mode="after")
    def _unresolved_high_risk_names_a_next_step(self) -> Self:
        """An unresolved high-severity finding owes a concrete action.

        Duplicated from the domain validator deliberately: failing here rejects the batch and
        lets the runtime record *that the model produced an incomplete disposition*, whereas
        failing at construction would surface from the wrong layer after the gate had already
        admitted it.
        """
        if (
            self.status is Status.UNRESOLVED
            and self.severity in SEVERITIES_NEEDING_ACTION
            and not self.validation_actions
        ):
            raise ValueError(
                f"{self.handle}: an unresolved {self.severity.value} finding requires at "
                "least one validation action naming what would settle it"
            )
        return self

    @model_validator(mode="after")
    def _support_and_contradiction_are_disjoint(self) -> Self:
        """One span cannot both support and contradict the same claim.

        Not pedantry: a reader counting sources on either side would count it twice, and a
        disposition that cites the same evidence both ways has not actually taken a position.
        """
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError(
                f"{self.handle}: {sorted(overlap)} cited as both supporting and "
                "contradicting. One span cannot do both for the same claim."
            )
        return self


class DispositionBatch(BaseModel):
    """Everything one consequence-judge call returns.

    There is no field for a *new* consequence. The judge arbitrates what the pipeline
    produced; a stage that could also introduce findings would be able to add a claim that no
    hunter proposed, no challenger reviewed, and no resolver derived.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dispositions: tuple[DispositionProposal, ...] = Field(default=(), max_length=MAX_DISPOSITIONS)

    #: Instruction-shaped text the model noticed in the evidence it was shown. Corpus text is
    #: untrusted, and a span that tried to instruct the model is a security observation.
    observed_instructions: tuple[Statement, ...] = ()

    #: What the assembled material could not settle. Carried into the report rather than
    #: discarded: uncertainty is a first-class output (``CLAUDE.md`` 7).
    unanswered_questions: tuple[Statement, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The schema handed to the provider, so the ask matches what is enforced."""
        return cls.model_json_schema()
