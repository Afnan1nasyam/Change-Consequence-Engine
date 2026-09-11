"""What the contradiction-hunter is allowed to say. Materiality and timing, nothing else.

The counterpart of :mod:`app.consequences.proposals`, and narrower on purpose. The
consequence-hunter proposes *new* claims. This agent proposes nothing: the divergence set
arrives already detected and already classified by the checkpoint 3.3 / 5.4 matrix, and the
agent's contribution is the two things the resolver has no way to compute.

**What it may say.** Whether the divergence matters and to whom (:class:`Materiality`), and
where the artifacts stand in time relative to one another (:class:`TemporalPosition`). Both
are small closed enums rather than prose scores, because "is this material?" has a defensible
answer and "how material, out of one?" does not.

**What is absent, and why absence is the safeguard.** ``extra="forbid"`` makes any of these a
validation error rather than a silently dropped key:

``confidence``
    computed by versioned code over evidence features (``CLAUDE.md`` 15). A model asked for a
    number produces a fluent one, and a fluent number is indistinguishable from a calibrated
    one by the time a reader sees it. :class:`Materiality` is what an agent may say instead.
``severity``
    the judge's, and a judgement about business impact the corpus does not contain
    (checkpoint 9.4).
``divergence_class`` / ``relationship`` / ``relation_id`` / ``rule_id``
    **the deterministic classification, and there is nowhere here to write it.** Class A
    means "produced by the deterministic resolver", so an agent that could restate the class
    could turn a stale consumer into a peer contradiction, or a legitimate override into a
    governance defect, without performing the derivation that decides either. The agent
    addresses an item by a local handle and never by a domain identifier, so it cannot name
    a Class A relation at all.
``status``
    same reasoning as the consequence-hunter: absent a recorded outcome the runtime records
    what is missing rather than the model asserting a level.

**The only direction that reduces what a reader sees is the one that must be justified.**
Calling a deterministically-detected divergence ``immaterial`` is the single judgement here
that argues *against* a finding the resolver already stands behind, so the validator below
requires a reason for it. Calling one ``material`` requires naming who is affected, because
"this matters" with no affected party is not an assessment.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

#: Bumped when this shape changes, because it changes what the model was asked for and
#: therefore what its answer means. Pinned onto every admitted assessment.
ASSESSMENT_SCHEMA_VERSION: Final = "contradiction-assessment-1"

#: The fields an agent may never supply. Enforced structurally by ``extra="forbid"``; named
#: here so a test asserts the list rather than trusting the class body to stay right. The
#: last four are the deterministic classification: an agent that could write any of them
#: could relabel a Class A relationship without deriving one.
FORBIDDEN_AGENT_FIELDS: Final = (
    "confidence",
    "severity",
    "status",
    "consequence_class",
    "divergence_class",
    "relationship",
    "relation_id",
    "rule_id",
)

#: Bounded so one call cannot flood the report. The divergence set is finite and supplied,
#: so a batch larger than this is a model repeating itself rather than finding more.
MAX_ASSESSMENTS: Final = 32

Handle = Annotated[str, StringConstraints(min_length=2, max_length=8, strip_whitespace=True)]
Statement = Annotated[str, StringConstraints(min_length=4, max_length=600, strip_whitespace=True)]
Reference = Annotated[str, StringConstraints(min_length=1, max_length=120, strip_whitespace=True)]


class Materiality(StrEnum):
    """Whether a divergence matters. Three values, and no number.

    ``UNDETERMINED`` is a first-class answer rather than a failure. A divergence whose impact
    the corpus cannot establish is exactly the case where a model will otherwise produce a
    confident-sounding guess, and recording the guess is worse than recording the limit.
    """

    MATERIAL = "material"
    IMMATERIAL = "immaterial"
    UNDETERMINED = "undetermined"


class TemporalPosition(StrEnum):
    """Where the two artifacts stand in time, per the agent's reading.

    The resolver already computed authority for the relevant date, so this is not a second
    opinion on who governs. It is the temporal *shape* of the divergence, which decides what
    a reader does about it: a superseded document needs archiving, two live overlapping
    declarations need a decision, and an ambiguous state needs an owner.

    ``AMBIGUOUS`` is reported rather than resolved. Picking the newest source would invent an
    authority the corpus does not establish (checkpoint 5.3), and it would do it invisibly.
    """

    #: One artifact clearly governs at the relevant date.
    AUTHORITY_CLEAR = "authority_clear"
    #: One artifact's validity has ended; the divergence is historical.
    SUPERSEDED = "superseded"
    #: Both are in force at once, so the divergence is live.
    INTERVALS_OVERLAP = "intervals_overlap"
    #: The corpus cannot say which governs. Reported, never resolved by recency.
    AMBIGUOUS = "ambiguous"
    #: The supplied material does not settle the temporal question.
    UNDETERMINED = "undetermined"


class DivergenceAssessment(BaseModel):
    """One divergence, rated. Not a finding, and not a classification.

    ``item_id`` is the local handle from
    :class:`~app.contradictions.divergence.DivergenceItem`. A handle the agent was not given
    is dropped by the runtime and reported, because an assessment of a divergence that was
    never detected is the exact failure mode this stage exists to avoid.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: Handle
    materiality: Materiality
    temporal_position: TemporalPosition

    #: Who is harmed by the divergence, and how. Required for ``material``: the agent
    #: definition asks for it, and a materiality claim with no affected party is a label.
    who_is_affected: Statement | None = None

    #: An exception clause that reconciles the two artifacts, if one does. This is the
    #: legitimate route to ``immaterial`` - the divergence is real and something in the
    #: corpus already resolves it.
    reconciling_exception: Statement | None = None

    #: Which of the artifacts the agent reads as governing at the relevant date. Prose, and
    #: advisory: the resolver's answer is authoritative and this cannot replace it.
    authoritative_reading: Reference | None = None

    #: Evidence the model says supports its reading. **Claimed, not verified**: the runtime
    #: keeps only ids it actually offered for this item.
    cited_evidence_ids: tuple[Reference, ...] = ()

    #: Concise, reader-facing reasoning. Never chain-of-thought (``CLAUDE.md`` 14).
    rationale: Statement | None = None

    @model_validator(mode="after")
    def _material_names_who_is_affected(self) -> Self:
        if self.materiality is Materiality.MATERIAL and self.who_is_affected is None:
            raise ValueError(
                f"{self.item_id}: materiality='material' must name who is affected and how. "
                "A materiality claim with no affected party is a label, not an assessment."
            )
        return self

    @model_validator(mode="after")
    def _immaterial_is_justified(self) -> Self:
        """The one judgement that argues against a deterministic finding needs a reason.

        Every other value here adds to what a reader sees. ``immaterial`` subtracts, against
        a divergence the resolver detected by exact comparison - so it costs either a
        reconciling exception from the corpus or an explicit rationale. Bare dismissal has
        no representation.
        """
        if self.materiality is Materiality.IMMATERIAL and not (
            self.reconciling_exception or self.rationale
        ):
            raise ValueError(
                f"{self.item_id}: materiality='immaterial' must give a reconciling exception "
                "or a rationale. It argues against a divergence the resolver detected by "
                "exact comparison, so it cannot be a bare dismissal."
            )
        return self


class AssessmentBatch(BaseModel):
    """Everything one contradiction-hunter call returns.

    There is no field for a *new* divergence, and that is requirement rather than oversight:
    a corpus with no deterministic divergence must produce no contradiction findings, and the
    cheapest way to guarantee that is a schema with nowhere to put one. The agent definition's
    inferred-contradiction half (SOP versus observed behaviour, conflicting definitions) has
    no runtime path yet; when it gets one it gets its own schema and its own slice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessments: tuple[DivergenceAssessment, ...] = Field(default=(), max_length=MAX_ASSESSMENTS)

    #: Instruction-shaped text the model noticed in the evidence it was shown. Mirrors the
    #: extractor's field: retrieved corpus text is untrusted, and a span that tried to
    #: instruct the model is a security observation rather than a curiosity to drop.
    observed_instructions: tuple[Statement, ...] = ()

    #: Questions the supplied material could not answer. Raw material for an ``EvidenceGap``
    #: through :mod:`app.application.orchestration.evidence_gaps`; this agent does not
    #: construct one, because a gap needs the retrieval trace it does not hold.
    unanswered_questions: tuple[Statement, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The schema handed to the provider, so the ask matches what is enforced."""
        return cls.model_json_schema()
