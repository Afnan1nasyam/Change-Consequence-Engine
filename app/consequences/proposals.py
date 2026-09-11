"""What the consequence-hunter is allowed to say. Untrusted input, refused by shape.

The counterpart of ``app/assumptions/proposals.py``, and it exists for the same reason: the
model's output is **untrusted input to a deterministic gate**, not a finding. This module is
the shape that input must take, and the fields it deliberately does not have are the whole
of its value.

**Three fields are absent, and their absence is the safeguard.** ``extra="forbid"`` means a
proposal carrying any of them is a validation error rather than a silently dropped key:

``confidence``
    computed by versioned code over evidence features (``CLAUDE.md`` 15, ``IDEA.md`` 13). A
    model asked for a number produces a fluent one, and a fluent number is indistinguishable
    from a calibrated one at the point a reader sees it.
``severity``
    a judgment about business impact the corpus does not contain, so the judge assigns it
    (checkpoint 9.4). The asymmetry with confidence is deliberate: one is a function of
    measurable evidence properties, the other is not.
``consequence_class``
    **pipeline-assigned** (``CLAUDE.md`` 22). Classification is by *producer*: Class A means
    "produced by the deterministic resolver", so a hypothesis that could self-declare Class A
    would be claiming a derivation it did not perform. There is no field here to declare it
    with, which is stronger than a rule saying not to.

**Evidence is proposed, never asserted.** ``cited_evidence_ids`` are ids the model claims
support its claim; :mod:`app.consequences.hunter` keeps only those that were actually offered
to it and drops the rest. A hypothesis whose every citation was fabricated does not become a
defect - it becomes a hypothesis with no evidence, which is a legitimate thing to be. What
must never happen is the reverse: an unsupported claim presented as established.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

#: Bumped when this shape changes, because it changes what the model was asked for and
#: therefore what its output means. Pinned onto every admitted hypothesis.
HYPOTHESIS_SCHEMA_VERSION: Final = "class-b-proposal-1"

#: The fields an agent may never supply. Enforced structurally by ``extra="forbid"``; named
#: here so a test can assert the list rather than trusting the class body to stay right.
FORBIDDEN_AGENT_FIELDS: Final = ("confidence", "severity", "consequence_class", "status")

#: A mechanism with fewer links than this is a restatement of the claim, not a mechanism.
MIN_MECHANISM_LINKS: Final = 2
MAX_MECHANISM_LINKS: Final = 8

#: Bounded so one call cannot flood the report. A hunter returning forty hypotheses has not
#: found forty consequences; it has stopped discriminating (``CLAUDE.md``, alert load).
MAX_HYPOTHESES: Final = 12

Claim = Annotated[str, StringConstraints(min_length=8, max_length=600, strip_whitespace=True)]
Link = Annotated[str, StringConstraints(min_length=4, max_length=300, strip_whitespace=True)]
Entity = Annotated[str, StringConstraints(min_length=1, max_length=120, strip_whitespace=True)]


class HypothesisProposal(BaseModel):
    """One Class B candidate, as the model proposed it.

    Not a :class:`~app.domain.entities.claims.ConsequenceHypothesis`. That type carries
    provenance and admitted evidence and is constructed by the runtime after the checks in
    :mod:`app.consequences.hunter` pass. Keeping the two apart is what stops model output
    reaching a store in the shape of a finding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: What is claimed to happen. A behavioural or quantitative effect - the Class B
    #: territory no assumption binding can express.
    claim: Claim
    #: The causal chain, one link per element. Required to have at least two links: a
    #: one-link "mechanism" is the claim again in different words.
    mechanism_chain: tuple[Link, ...] = Field(
        min_length=MIN_MECHANISM_LINKS, max_length=MAX_MECHANISM_LINKS
    )
    #: Which link the author considers least supported. Required, because a mechanism with
    #: no acknowledged weak point is usually an unexamined one.
    weakest_link: Link

    #: Artifacts, teams, queues or metrics the effect lands on. Prose, not identifiers:
    #: the hunter is not resolving references.
    affected_entities: tuple[Entity, ...] = ()
    #: Registry keys the hypothesis touches, where it touches any. Used by the deterministic
    #: Class A suppression step, and never used to *create* a binding.
    assumption_keys: tuple[Entity, ...] = ()

    #: Evidence the model says supports this. **Claimed, not verified**: the runtime keeps
    #: only ids it actually offered.
    cited_evidence_ids: tuple[Entity, ...] = ()

    #: The source version this effect lands on, if the model can name one. Supplied so the
    #: deterministic suppression rule can compare against the Class A set, which is keyed on
    #: ``(assumption_key, affected_source_version_id)``. A proposal that names none is
    #: retained rather than suppressed: ambiguity demotes to Class B (checkpoint 7.4).
    affected_source_version_id: Entity | None = None

    #: A past change or incident with a **recorded observed outcome**. Without one, the
    #: hypothesis cannot exceed `speculative` (checkpoint 9.2), which the runtime records
    #: rather than the model asserting.
    historical_analogue_id: Entity | None = None

    #: Concise, reader-facing reasoning. Never chain-of-thought (``CLAUDE.md`` 14).
    rationale: Claim | None = None

    @model_validator(mode="after")
    def _the_weakest_link_is_one_of_the_links(self) -> Self:
        """The named weak point must be a link in the chain it weakens.

        Otherwise "weakest link" degrades into a second, vaguer summary - which is exactly
        what it exists to prevent.
        """
        if self.weakest_link not in self.mechanism_chain:
            raise ValueError(
                "weakest_link must be one of the mechanism_chain links verbatim, got "
                f"{self.weakest_link!r} which is not among {len(self.mechanism_chain)} links"
            )
        return self


class HypothesisBatch(BaseModel):
    """Everything one consequence-hunter call returns.

    ``observed_instructions`` mirrors the extractor's field for the same reason: retrieved
    corpus text is untrusted, and a span that tried to instruct the model is a security
    observation worth surfacing rather than a curiosity to drop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypotheses: tuple[HypothesisProposal, ...] = Field(default=(), max_length=MAX_HYPOTHESES)

    #: Instruction-shaped text the model noticed in the evidence it was shown.
    observed_instructions: tuple[Claim, ...] = ()

    #: Questions the corpus could not answer. Raw material for an ``EvidenceGap``; the
    #: hunter does not construct one, because a gap is a claim about the corpus and needs
    #: the retrieval trace this agent does not hold.
    unanswered_questions: tuple[Claim, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The schema handed to the provider, so the ask matches what is enforced."""
        return cls.model_json_schema()
