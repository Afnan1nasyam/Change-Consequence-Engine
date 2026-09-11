"""What a specialist is allowed to say. Untrusted input, refused by shape.

The same pattern as every other agent schema in this system, and the same four absences. A
specialist observation is an **evidence-backed statement about its slice** - not a consequence,
not a class, not a number:

``confidence``
    computed by versioned code over evidence features (``CLAUDE.md`` 15). Every specialist
    definition already says the runtime decides the number; ``extra="forbid"`` is what makes
    that true rather than requested.
``severity``
    a judgment about business impact the corpus does not contain (checkpoint 9.4).
``status``
    the judge's, after the frozen gates.
``consequence_class``
    pipeline-assigned (``CLAUDE.md`` 22). Nothing a specialist returns can be Class A: Class A
    means *produced by the deterministic resolver*, and no field here could claim that.

``basis`` is the one axis the definitions do ask for - "mark every finding as direct evidence
or inference" - and it is a **ceiling**, not a label: :mod:`app.specialists.analyst` will not
let an inference reach ``supported``.

``observed_bindings`` exists because the definitions ask specialists to report propositions
their slice commits to. They are recorded and **never admitted**: the key registry is closed
and admission happens at ingestion behind the deterministic gate (``CLAUDE.md`` 20, 21). There
is no field here for an admission decision, and no code path from one of these to the coupling
index.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

#: Bumped when this shape changes, because it changes what the model was asked for.
OBSERVATION_SCHEMA_VERSION: Final = "specialist-observation-1"

#: Enforced structurally by ``extra="forbid"``; named here so a test asserts the list rather
#: than trusting the class body to stay right.
FORBIDDEN_AGENT_FIELDS: Final = ("confidence", "severity", "status", "consequence_class")

#: Bounded per specialist. Seven analysts each returning twelve observations is a report
#: nobody reads, and alert load is a correctness constraint (``CLAUDE.md``).
MAX_OBSERVATIONS: Final = 6
MAX_BINDINGS: Final = 8

#: Analogues per call. Low deliberately: one well-evidenced precedent with a measured outcome
#: is worth more than four narratives, and the definition asks for precision over coverage.
MAX_ANALOGUES: Final = 4

Claim = Annotated[str, StringConstraints(min_length=8, max_length=600, strip_whitespace=True)]
Entity = Annotated[str, StringConstraints(min_length=1, max_length=120, strip_whitespace=True)]
Note = Annotated[str, StringConstraints(min_length=4, max_length=400, strip_whitespace=True)]


class ObservedBinding(BaseModel):
    """A proposition the specialist's slice commits to. Recorded, never admitted.

    Reported because the definitions ask for it and because a key nothing in the registry
    covers is a real signal about registry coverage. It reaches no index: the closed registry
    and the admission gate are ingestion-time and deterministic, and a binding admitted from
    an analysis-time model call would be exactly the promotion ``CLAUDE.md`` 20 forbids.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: Entity
    #: Rendered, not typed. Parsing into a ``TypedValue`` is the admission gate's job, and
    #: doing it here would be the first half of a promotion.
    value: Entity
    evidence_id: Entity


class ObservationProposal(BaseModel):
    """One evidence-backed statement about the specialist's own slice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: What the slice shows, and why it bears on the change.
    claim: Claim
    #: ``direct_evidence`` means a cited span says this. ``inference`` means the specialist
    #: reasoned to it from cited spans. The runtime caps what each may assert.
    basis: Literal["direct_evidence", "inference"]
    #: Ids the specialist claims support the statement. **Claimed, not verified**: the runtime
    #: keeps only ids that were in this specialist's slice.
    cited_evidence_ids: tuple[Entity, ...] = ()
    #: Artifacts, teams, queues or metrics the observation is about. Prose, not identifiers.
    affected_entities: tuple[Entity, ...] = ()
    #: Registry keys the observation touches, where it touches any. Used for the deterministic
    #: Class A duplicate check, and never to create a binding.
    assumption_keys: tuple[Entity, ...] = ()
    rationale: Note | None = None


class AnalogueProposal(BaseModel):
    """A prior change or incident the historical analyst found, as it proposed it.

    Asked of the ``historical-analyst`` only - its definition makes it "the anchor for Class B
    claims", and a Class B hypothesis needs an analogue with a recorded outcome to exceed
    ``speculative``. :mod:`app.specialists.analyst` refuses this field from the other six.

    **There is no id field, and that absence is the safeguard.** The event id is derived by
    :func:`app.ingestion.identity.event_id_for` from the spans cited, so a model cannot mint
    one - and a well-formed invented ``he_...`` has nowhere to arrive.

    ``observed_outcome`` is the field that decides whether this anchors anything. Its own
    definition says it: "an analogue with no recorded outcome does not qualify as an anchor -
    say so". Left empty, the event is still recorded and still cannot anchor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: What happened, in a few words. Never part of the event's identity.
    title: Entity
    #: When. Compared against the analysis's ``as_of``: an "analogue" in the future is a
    #: parsing or reading error, not a precedent.
    occurred_at: dt.datetime
    #: The **measured** result, if the source records one. Not intent, not a retrospective
    #: narrative - the definition asks for measured outcomes over opinion.
    observed_outcome: Note | None = None
    #: Spans that record the event. Claimed, not verified: the runtime keeps only ids that
    #: were in this specialist's slice, and an event with none cannot anchor.
    cited_evidence_ids: tuple[Entity, ...] = ()
    #: How the prior case differs from the present one. Required, because its definition says
    #: a weak analogue presented as a strong one is worse than none.
    differs_how: Note


class ObservationBatch(BaseModel):
    """Everything one specialist call returns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: tuple[ObservationProposal, ...] = Field(default=(), max_length=MAX_OBSERVATIONS)
    observed_bindings: tuple[ObservedBinding, ...] = Field(default=(), max_length=MAX_BINDINGS)

    #: Prior changes and incidents, from the ``historical-analyst`` only. Refused from any
    #: other specialist, so the one agent whose definition asks for them is the one that can
    #: supply them.
    historical_events: tuple[AnalogueProposal, ...] = Field(default=(), max_length=MAX_ANALOGUES)

    #: What this slice could not answer. The definitions require a specialist to ask rather
    #: than infer, so this is the sanctioned alternative to filling a gap.
    unresolved_questions: tuple[Claim, ...] = ()
    #: Instruction-shaped text noticed in the slice. Corpus text is untrusted.
    observed_instructions: tuple[Claim, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The schema handed to the provider, so the ask matches what is enforced."""
        return cls.model_json_schema()
