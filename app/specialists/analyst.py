"""The specialist runtime: one call over one slice, then the same narrow gate.

One class for all seven analysts, constructed per agent. They differ by *definition and
slice*, which is the whole design (``CLAUDE.md`` 17) - a class per persona would be seven
copies of this gate waiting to drift apart.

The gate, and what each step can and cannot do
----------------------------------------------

**Citations are checked against this specialist's slice.** Not the analysis's evidence, not
another specialist's. An id the specialist was not shown is dropped exactly like a fabricated
one, and a specialist that cited only ids from somebody else's slice ends up with no evidence -
which the position ceiling then caps.

**The position is a ceiling, never a label.** ``EvidencePosition`` is computed from two
deterministic facts - whether any citation survived, and whether the specialist called the
observation an inference - and it can only weaken:

============================  =========================  =========================
admitted citations             ``basis``                  position
============================  =========================  =========================
none                          either                     ``unresolved``
at least one                  ``inference``              ``weakly_supported``
at least one                  ``direct_evidence``        ``supported``
============================  =========================  =========================

``unresolved`` rather than ``unsupported`` for an uncited observation: "nobody produced
evidence for this" is not the same claim as "the evidence is against it", and the second is a
finding the specialist did not make.

**Four fields have nowhere to arrive.** ``confidence``, ``severity``, ``status`` and
``consequence_class`` are absent from the proposal schema under ``extra="forbid"``, so a
specialist emitting one gets a validation error and the batch is discarded rather than
partially trusted. :class:`~app.domain.entities.findings.SpecialistFinding` has no field for
any of them either, which is the stronger guarantee.

**Observed bindings are recorded and never admitted.** They leave on the outcome as proposals
for registry review. There is no code path from one to the coupling index, and that is
deliberate: the registry is closed and admission is an ingestion-time deterministic gate
(``CLAUDE.md`` 20, 21).

**A failure is one analyst's, not the stage's.** The outcome records the failure with the
provenance the call would have carried, and the fan-out keeps every other specialist's result.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Final

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import SpecialistFinding
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    ProviderError,
    StructuredOutputError,
)
from app.domain.value_objects.enums import EvidencePosition
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import ModelProvenance
from app.specialists.analogues import AnalogueOutcome, admit
from app.specialists.prompting import (
    SPECIALIST_PROMPT_VERSION,
    build_specialist_prompt,
)
from app.specialists.proposals import (
    OBSERVATION_SCHEMA_VERSION,
    ObservationBatch,
    ObservationProposal,
    ObservedBinding,
)
from app.specialists.roster import BY_AGENT, Specialist, SpecialistSlice

logger = logging.getLogger(__name__)

#: The position an observation may reach, by ``basis``, once it has at least one admitted
#: citation. Frozen, and the only place the mapping exists.
POSITION_FOR_BASIS: Final[dict[str, EvidencePosition]] = {
    "direct_evidence": EvidencePosition.SUPPORTED,
    "inference": EvidencePosition.WEAKLY_SUPPORTED,
}

#: An observation with no admitted citation. Not ``unsupported``: nobody produced evidence
#: either way, and saying the evidence is against it would be a finding nobody made.
POSITION_WITHOUT_EVIDENCE: Final = EvidencePosition.UNRESOLVED


def specialist_version(definition: AgentDefinition) -> str:
    """Prompt, contract shape and schema together, as every other agent stamps it."""
    return (
        f"specialist-{definition.name}-{definition.version}:"
        f"{SPECIALIST_PROMPT_VERSION}:{OBSERVATION_SCHEMA_VERSION}"
    )


@dataclass(frozen=True)
class SpecialistOutcome:
    """What one specialist call produced, and what producing it revealed."""

    agent: str
    findings: tuple[SpecialistFinding, ...] = ()
    #: Propositions the slice commits to. **Proposals for registry review, never admitted.**
    observed_bindings: tuple[ObservedBinding, ...] = ()
    #: Prior changes and incidents, after deterministic admission. Empty for every specialist
    #: but the historical analyst, whose definition carries the anchor obligation.
    analogues: AnalogueOutcome = field(default_factory=AnalogueOutcome)
    unresolved_questions: tuple[str, ...] = ()
    #: Cited ids that were not in this specialist's slice.
    fabricated_evidence_ids: tuple[str, ...] = ()
    #: Observations that reached no admitted citation, by finding id.
    uncited_finding_ids: tuple[str, ...] = ()
    observed_instructions: tuple[str, ...] = ()
    secrets_redacted: str = "none"
    attempted_delimiter_breakout: bool = False
    calls_made: int = 0
    failure: str | None = None
    provenance: ModelProvenance | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    @property
    def evidence_ids(self) -> frozenset[str]:
        """Every id the retained findings rest on."""
        return frozenset(i for f in self.findings for i in f.evidence_ids)

    @property
    def anchor_ids(self) -> frozenset[str]:
        """Event ids a Class B hypothesis may cite as its analogue."""
        return self.analogues.anchor_ids


class SpecialistAnalyst:
    """Invokes one specialist over its slice and admits what it returns."""

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        specialist = BY_AGENT.get(definition.name)
        if specialist is None:
            raise ValueError(
                f"{definition.name!r} is not on the specialist roster "
                f"({sorted(BY_AGENT)}); this runtime dispatches the seven analysts only"
            )
        self._completions = completions
        self._definition = definition
        self._specialist: Specialist = specialist

    @property
    def agent(self) -> str:
        return self._specialist.agent

    @property
    def version(self) -> str:
        return specialist_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        """Attribution for anything this specialist proposes.

        Resolved without a model call, so a failed call still carries full provenance and the
        outcome can say which prompt and which binding failed.
        """
        binding = self._completions.resolve(self.agent)
        return ModelProvenance(
            agent=self.agent,
            prompt_version=self._definition.version,
            prompt_sha256=self._definition.prompt_sha256,
            provider=binding.provider,
            model=binding.model,
            tier=binding.tier,
        )

    async def observe(
        self,
        *,
        change: ChangeSpec,
        slice_: SpecialistSlice,
        known_claims: tuple[str, ...] = (),
        tenant_id: str = "",
        as_of: dt.datetime | None = None,
        correlation_id: str | None = None,
    ) -> SpecialistOutcome:
        """One call, then the gate."""
        if slice_.agent != self.agent:
            raise ValueError(
                f"{self.agent} was handed {slice_.agent}'s slice. A specialist reading "
                "another's evidence is the correlated-findings failure the slice design "
                "exists to prevent."
            )

        prompt = build_specialist_prompt(
            system_prompt=self._definition.prompt,
            change=change,
            slice_=slice_,
            known_claims=known_claims,
            nonce=uuid.uuid4().hex[:12],
        )
        provenance = self._provenance()

        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "slice evidence attempted to impersonate the untrusted-content delimiter",
                extra={"change_id": change.change_id, "agent": self.agent},
            )

        base = SpecialistOutcome(
            agent=self.agent,
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                self.agent,
                prompt.messages,
                json_schema=ObservationBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            logger.warning(
                "specialist call failed; no observations from this analyst this run",
                extra={"change_id": change.change_id, "agent": self.agent, "error": str(exc)},
            )
            return replace(base, failure=f"{type(exc).__name__}: {exc}", calls_made=1)

        try:
            batch = ObservationBatch.model_validate(payload)
        except ValidationError as exc:
            # Includes the four that matter most: a specialist emitting `confidence`,
            # `severity`, `status` or `consequence_class`. `extra="forbid"` makes each a
            # rejection instead of a dropped key, and the batch is discarded rather than
            # partially trusted.
            logger.warning(
                "specialist output violated the observation schema; discarded",
                extra={
                    "change_id": change.change_id,
                    "agent": self.agent,
                    "errors": exc.error_count(),
                },
            )
            return replace(base, failure=f"schema: {exc.error_count()} error(s)", calls_made=1)

        return self._admit(
            batch,
            base=base,
            slice_=slice_,
            tenant_id=tenant_id,
            as_of=as_of or dt.datetime.now(tz=dt.UTC),
        )

    # -- the gate --------------------------------------------------------

    def _admit(
        self,
        batch: ObservationBatch,
        *,
        base: SpecialistOutcome,
        slice_: SpecialistSlice,
        tenant_id: str,
        as_of: dt.datetime,
    ) -> SpecialistOutcome:
        """Turn proposals into findings. Every step weakens; none strengthens."""
        assert base.provenance is not None  # set by `observe` before the call

        findings: list[SpecialistFinding] = []
        fabricated: list[str] = []
        uncited: list[str] = []

        for proposal in batch.observations:
            kept, invented = _partition_citations(proposal, offered=slice_.evidence_ids)
            fabricated.extend(invented)

            finding_id = new_id("finding")
            if not kept:
                uncited.append(finding_id)
            findings.append(
                SpecialistFinding(
                    finding_id=finding_id,
                    agent=self.agent,
                    claim=proposal.claim,
                    # Fixed per specialist, not proposed: a label the agent could choose is a
                    # label that tells a reader nothing.
                    category=self._specialist.category,
                    affected_entities=proposal.affected_entities,
                    evidence_ids=kept,
                    evidence_position=position_for(proposal.basis, has_evidence=bool(kept)),
                    assumption_keys=proposal.assumption_keys,
                    produced_by=base.provenance,
                )
            )

        if fabricated:
            logger.warning(
                "specialist cited evidence ids outside its slice; dropped",
                extra={"agent": self.agent, "count": len(fabricated)},
            )
        if batch.observed_bindings:
            logger.info(
                "specialist reported assumption bindings; recorded for review, not admitted",
                extra={"agent": self.agent, "count": len(batch.observed_bindings)},
            )

        return replace(
            base,
            findings=tuple(findings),
            observed_bindings=batch.observed_bindings,
            # The id is derived from the cited spans, so the model has no field to put one in.
            # Refused entirely for any specialist but the historical analyst.
            analogues=admit(
                batch.historical_events,
                agent=self.agent,
                tenant_id=tenant_id,
                offered_evidence_ids=slice_.evidence_ids,
                as_of=as_of,
            ),
            unresolved_questions=batch.unresolved_questions,
            fabricated_evidence_ids=tuple(sorted(set(fabricated))),
            uncited_finding_ids=tuple(uncited),
            observed_instructions=batch.observed_instructions,
            calls_made=1,
        )


def position_for(basis: str, *, has_evidence: bool) -> EvidencePosition:
    """The frozen ceiling. Two deterministic inputs, neither of them the model's opinion.

    There is no parameter for how sure the specialist sounded, and no branch that raises a
    position - which is why an inference cannot be promoted by wording it confidently.
    """
    if not has_evidence:
        return POSITION_WITHOUT_EVIDENCE
    return POSITION_FOR_BASIS[basis]


def _partition_citations(
    proposal: ObservationProposal, *, offered: frozenset[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split cited ids into those in this slice and those outside it.

    Exact string match, no normalization: an id that differs by a character is not a near miss
    to be repaired, it is an id the specialist did not receive. Repairing one would be the
    system inventing a citation on the model's behalf.
    """
    kept: list[str] = []
    invented: list[str] = []
    for cited in proposal.cited_evidence_ids:
        (kept if cited in offered else invented).append(cited)
    return tuple(sorted(set(kept))), tuple(invented)
