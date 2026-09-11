"""What the blind-spot hunter is allowed to say. Untrusted input, refused by shape.

Two output kinds, deliberately separate types rather than one with a flag: a **consequence**
claims something will happen, and **corpus silence** claims the corpus does not say. Letting
one schema express both is how a gap gets dressed up as a discovery.

The four fields a Class B proposal must not carry are absent here for the same reasons as in
:mod:`app.consequences.proposals`, and ``extra="forbid"`` makes each a validation error rather
than a dropped key. ``partition_slice`` is accepted and then **ignored**: attribution is
derived from the cited spans by :mod:`app.blindspots.hunter`, so a claimed yield cannot
outrun the evidence it came from.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

#: Bumped when this shape changes, because it changes what the model was asked for.
BLIND_SPOT_SCHEMA_VERSION: Final = "blind-spot-proposal-1"

#: Enforced structurally by ``extra="forbid"``; named so a test can assert the list.
FORBIDDEN_AGENT_FIELDS: Final = ("confidence", "severity", "consequence_class", "status")

MIN_MECHANISM_LINKS: Final = 2
MAX_MECHANISM_LINKS: Final = 8

#: Lower than the main hunter's budget, on purpose. This agent is rewarded for useful
#: discoveries, not for filling a quota, and a wide cap is an invitation to pad.
MAX_FINDINGS: Final = 8
MAX_SILENCE: Final = 6

Claim = Annotated[str, StringConstraints(min_length=8, max_length=600, strip_whitespace=True)]
Link = Annotated[str, StringConstraints(min_length=4, max_length=300, strip_whitespace=True)]
Entity = Annotated[str, StringConstraints(min_length=1, max_length=120, strip_whitespace=True)]


class BlindSpotProposal(BaseModel):
    """One consequence the main analysis did not report, as the model proposed it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: Claim
    mechanism_chain: tuple[Link, ...] = Field(
        min_length=MIN_MECHANISM_LINKS, max_length=MAX_MECHANISM_LINKS
    )
    weakest_link: Link

    affected_entities: tuple[Entity, ...] = ()
    assumption_keys: tuple[Entity, ...] = ()
    cited_evidence_ids: tuple[Entity, ...] = ()
    affected_source_version_id: Entity | None = None
    historical_analogue_id: Entity | None = None

    #: The slice the model believes produced this. Recorded for comparison with the derived
    #: attribution, and never used as the attribution itself.
    partition_slice: Entity | None = None

    #: The model's own admission that it has drifted back into the main analysis. Honoured:
    #: a self-reported restatement is dropped rather than padding the count.
    restates_existing: bool = False

    rationale: Claim | None = None

    @model_validator(mode="after")
    def _the_weakest_link_is_one_of_the_links(self) -> Self:
        if self.weakest_link not in self.mechanism_chain:
            raise ValueError(
                "weakest_link must be one of the mechanism_chain links verbatim, got "
                f"{self.weakest_link!r} which is not among {len(self.mechanism_chain)} links"
            )
        return self


class CorpusSilenceProposal(BaseModel):
    """Something the change clearly touches that the partition shows the corpus silent on.

    Not a consequence, and it cannot become one: there is no claim field, no mechanism and no
    citation list. What it carries is a question and the source that would answer it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: Claim
    why_it_matters: Claim
    #: The kind of artifact that would settle it. Prose, because the acquisition decision is
    #: the reader's and naming a store here would imply the system can go and get it.
    recommended_source: Entity
    partition_slice: Entity | None = None


class BlindSpotBatch(BaseModel):
    """Everything one blind-spot call returns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consequences: tuple[BlindSpotProposal, ...] = Field(default=(), max_length=MAX_FINDINGS)
    corpus_silence: tuple[CorpusSilenceProposal, ...] = Field(default=(), max_length=MAX_SILENCE)

    #: Slices the model could not work - empty, unreadable, or irrelevant to the change.
    #: Reported, because "this slice yielded nothing" is the honest good outcome.
    unusable_slices: tuple[Entity, ...] = ()

    observed_instructions: tuple[Claim, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The schema handed to the provider, so the ask matches what is enforced."""
        return cls.model_json_schema()
