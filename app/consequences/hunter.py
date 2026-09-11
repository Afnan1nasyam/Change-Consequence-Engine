"""The consequence-hunter runtime: one call, then a deterministic gate over its output.

The Class B counterpart of ``app/assumptions/extraction.py``, and the same split: the model
**proposes**, deterministic code **admits**. Nothing the model returns reaches a store in the
shape it was returned.

Class B is the part of this system most likely to be wrong and most likely to sound
impressive, because it has no deterministic derivation (``docs/IDEA.md`` 15). So the gate
below is narrow and its refusals are all one-directional - every check can only *weaken* a
claim, never strengthen one:

**Fabricated citations are dropped, and dropping them does not fail the hypothesis.** Only
ids that were actually offered survive. A hypothesis whose every citation was invented
becomes a hypothesis with no evidence - which
:class:`~app.domain.entities.claims.ConsequenceHypothesis` already models correctly, because
its ``support`` is ``MECHANISM_ONLY`` by construction. There is no code path by which an
unsupported claim becomes a supported one, and that is a property of the type rather than of
this module's care.

**Confidence, severity, status and consequence_class have nowhere to arrive.** The proposal
schema forbids extra fields and declares none of the four, so a model emitting one gets a
validation error and the batch is discarded rather than partially trusted. The hypothesis
type then hard-codes ``derivation=INFERRED`` and ``support=MECHANISM_ONLY`` as
``Literal``s - so even a caller constructing one by hand cannot promote it.

**A provider or schema failure yields no hypotheses, never a partial set.** The failure is
reported on the outcome with the provenance the call would have carried, exactly as the
extractor does: an analysis that silently lost its Class B stage is worse than one that says
the stage failed.

What this module does not do: it does not run the Class A suppression rule (that is
``app/application/use_cases/hunt_consequences.py``, using the existing
``merge_class_b``), it does not score, it does not rank, and it does not construct an
``EvidenceGap`` - a gap is a claim about the corpus and needs the retrieval trace this agent
does not hold.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Final

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.consequences.prompting import (
    HUNTER_PROMPT_VERSION,
    AdmissibleAnalogue,
    EvidenceForHunting,
    SpecialistObservation,
    build_hunter_prompt,
)
from app.consequences.proposals import (
    HYPOTHESIS_SCHEMA_VERSION,
    HypothesisBatch,
    HypothesisProposal,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.entities.class_a import ClassAFinding
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    ProviderError,
    StructuredOutputError,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import ModelProvenance

logger = logging.getLogger(__name__)

#: The agent this runtime invokes. One name, so a mismatched definition is caught at
#: construction rather than producing findings attributed to the wrong prompt.
HUNTER_AGENT: Final = "consequence-hunter"


def hunter_version(definition: AgentDefinition) -> str:
    """The version stamped on every hypothesis: prompt, contract shape and schema together.

    All three, because each changes what the output means. A prompt edit changes what was
    asked, a contract change changes how it was asked, and a schema change changes what
    could be answered.
    """
    return f"hunter-{definition.version}:{HUNTER_PROMPT_VERSION}:{HYPOTHESIS_SCHEMA_VERSION}"


@dataclass(frozen=True)
class HuntOutcome:
    """What one consequence-hunter call produced, and what producing it revealed.

    ``failure`` is a string rather than an exception so the outcome stays serializable and
    an analysis can record *that* the stage failed alongside everything that succeeded.
    """

    hypotheses: tuple[ConsequenceHypothesis, ...] = ()
    #: Where the hypothesis's effect lands, by hypothesis id. Assembled here because the
    #: model supplies it and :class:`ConsequenceHypothesis` has no field for it - it names
    #: entities in prose. Handed to the deterministic suppression step, which is keyed on
    #: ``(assumption_key, affected_source_version_id)``.
    affected_artifact_of: dict[str, str] = field(default_factory=dict)
    #: Ids the model cited that it was never offered. Reported, because a model inventing
    #: citations is a signal about the model and not only about one hypothesis.
    fabricated_evidence_ids: tuple[str, ...] = ()
    #: Hypotheses with no admitted evidence at all. Legitimate, and worth counting: a stage
    #: producing only these has found nothing the corpus supports.
    mechanism_only_ids: tuple[str, ...] = ()
    #: Hypotheses with no historical analogue. Cannot exceed `speculative` (checkpoint 9.2).
    without_analogue_ids: tuple[str, ...] = ()
    #: Analogue ids the model claimed that were not admissible. Dropped, and counted: a model
    #: reaching for an anchor it does not have is a signal about the model, not one hypothesis.
    fabricated_analogue_ids: tuple[str, ...] = ()
    observed_instructions: tuple[str, ...] = ()
    unanswered_questions: tuple[str, ...] = ()
    secrets_redacted: str = "none"
    attempted_delimiter_breakout: bool = False
    calls_made: int = 0
    failure: str | None = None
    provenance: ModelProvenance | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


class ConsequenceHunter:
    """Invokes ``consequence-hunter`` and admits what it returns.

    Holds the completion port and the agent definition. The port is injected, so this class
    never names a provider or a model: ``AgentCompletionPort`` resolves tier, provider and
    model from configuration, which is what makes a provider-diversity experiment a config
    change (``ARCHITECTURE.md`` 11.2).
    """

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        if definition.name != HUNTER_AGENT:
            raise ValueError(f"expected the {HUNTER_AGENT!r} definition, got {definition.name!r}")
        self._completions = completions
        self._definition = definition

    @property
    def version(self) -> str:
        return hunter_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        """Attribution for anything this hunter proposes.

        Resolved without a model call, so a failed call still carries full provenance and
        the outcome can say which prompt and which binding failed.
        """
        binding = self._completions.resolve(HUNTER_AGENT)
        return ModelProvenance(
            agent=HUNTER_AGENT,
            prompt_version=self._definition.version,
            prompt_sha256=self._definition.prompt_sha256,
            provider=binding.provider,
            model=binding.model,
            tier=binding.tier,
        )

    async def hunt(
        self,
        *,
        change: ChangeSpec,
        evidence: tuple[EvidenceForHunting, ...],
        class_a: tuple[ClassAFinding, ...] = (),
        specialists: tuple[SpecialistObservation, ...] = (),
        analogues: tuple[AdmissibleAnalogue, ...] = (),
        correlation_id: str | None = None,
    ) -> HuntOutcome:
        """One call, then the gate.

        ``class_a`` is shown to the model so it does not spend the call re-deriving what the
        resolver already found. It is **not** the suppression step - that is deterministic,
        runs afterwards, and does not trust the model to have cooperated.

        ``specialists`` are the domain fan-out's verified observations. They change what the
        hunter *reads*, never what it may return: the gate below is identical with and without
        them, so a specialist cannot widen what a hypothesis is allowed to claim.

        ``analogues`` is the **closed set** of historical anchors a hypothesis may cite. Each
        one's ``event_id`` was derived from evidence spans by the runtime, so the gate can check
        a claimed id against something the model could not have authored. Passing none - the
        default - means every claimed anchor is dropped and every Class B hypothesis caps at
        ``speculative``, which is the honest result when nothing evidenced a precedent.
        """
        prompt = build_hunter_prompt(
            system_prompt=self._definition.prompt,
            change=change,
            evidence=evidence,
            class_a=class_a,
            specialists=specialists,
            analogues=analogues,
            nonce=uuid.uuid4().hex[:12],
        )
        provenance = self._provenance()

        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "retrieved evidence attempted to impersonate the untrusted-content delimiter",
                extra={"change_id": change.change_id, "agent": HUNTER_AGENT},
            )
        if prompt.redaction.found_any:
            logger.info(
                "redacted credential-shaped spans before dispatch",
                extra={"change_id": change.change_id, "secrets": prompt.redaction.summary},
            )

        base = HuntOutcome(
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                HUNTER_AGENT,
                prompt.messages,
                json_schema=HypothesisBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            # The existing error path. A failed stage is reported, never inferred from an
            # empty result: "the hunter found nothing" and "the hunter did not run" are
            # different facts about an analysis.
            logger.warning(
                "consequence-hunter call failed; no Class B hypotheses this run",
                extra={"change_id": change.change_id, "error": str(exc)},
            )
            return replace(base, failure=f"{type(exc).__name__}: {exc}", calls_made=1)

        try:
            batch = HypothesisBatch.model_validate(payload)
        except ValidationError as exc:
            # Includes the cases that matter most: a hunter emitting `confidence`,
            # `severity`, `status` or `consequence_class`. `extra="forbid"` turns each into
            # a rejection instead of a dropped key, and the batch is discarded rather than
            # partially trusted.
            logger.warning(
                "consequence-hunter output violated the proposal schema; discarded",
                extra={"change_id": change.change_id, "errors": exc.error_count()},
            )
            return replace(base, failure=f"schema: {exc.error_count()} error(s)", calls_made=1)

        return self._admit(
            batch,
            base=base,
            prompt_ids=prompt.offered_evidence_ids,
            anchor_ids=frozenset(item.event_id for item in analogues),
        )

    # -- the gate --------------------------------------------------------

    def _admit(
        self,
        batch: HypothesisBatch,
        *,
        base: HuntOutcome,
        prompt_ids: frozenset[str],
        anchor_ids: frozenset[str] = frozenset(),
    ) -> HuntOutcome:
        """Turn proposals into hypotheses, keeping only what the input supports.

        Every step here can weaken a claim and none can strengthen one. That asymmetry is
        the design: there is no branch in this method that adds evidence, raises support, or
        assigns a class.
        """
        assert base.provenance is not None  # set by `hunt` before the call

        hypotheses: list[ConsequenceHypothesis] = []
        affected: dict[str, str] = {}
        fabricated: list[str] = []
        invented_anchors: list[str] = []
        mechanism_only: list[str] = []
        without_analogue: list[str] = []

        for proposal in batch.hypotheses:
            kept, invented = _partition_citations(proposal, offered=prompt_ids)
            fabricated.extend(invented)

            # The anchor gate. A claimed id outside the admissible set is dropped, so the
            # hypothesis lands at `speculative` exactly as it would have with none - which is
            # what makes inventing one pointless rather than merely forbidden.
            anchor = proposal.historical_analogue_id
            if anchor is not None and anchor not in anchor_ids:
                invented_anchors.append(anchor)
                anchor = None

            hypothesis_id = new_id("hypothesis")
            hypotheses.append(
                ConsequenceHypothesis(
                    hypothesis_id=hypothesis_id,
                    claim=proposal.claim,
                    # The chain, joined - the domain type holds one mechanism statement,
                    # and the chain is how the model was asked to think rather than a
                    # second field to store.
                    mechanism=" -> ".join(proposal.mechanism_chain),
                    weakest_link=proposal.weakest_link,
                    affected_entities=proposal.affected_entities,
                    assumption_keys=proposal.assumption_keys,
                    evidence_ids=kept,
                    historical_analogue_id=anchor,
                    proposed_by=base.provenance,
                )
            )

            if proposal.affected_source_version_id is not None:
                affected[hypothesis_id] = proposal.affected_source_version_id
            if not kept:
                mechanism_only.append(hypothesis_id)
            if anchor is None:
                without_analogue.append(hypothesis_id)

        if fabricated:
            logger.warning(
                "consequence-hunter cited evidence ids it was not offered; dropped",
                extra={"agent": HUNTER_AGENT, "count": len(fabricated)},
            )
        if invented_anchors:
            logger.warning(
                "consequence-hunter claimed a historical analogue it was not offered; dropped",
                extra={"agent": HUNTER_AGENT, "count": len(invented_anchors)},
            )

        return replace(
            base,
            hypotheses=tuple(hypotheses),
            affected_artifact_of=affected,
            fabricated_evidence_ids=tuple(sorted(set(fabricated))),
            fabricated_analogue_ids=tuple(sorted(set(invented_anchors))),
            mechanism_only_ids=tuple(mechanism_only),
            without_analogue_ids=tuple(without_analogue),
            observed_instructions=batch.observed_instructions,
            unanswered_questions=batch.unanswered_questions,
            calls_made=1,
        )


def _partition_citations(
    proposal: HypothesisProposal, *, offered: frozenset[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split cited ids into those that were offered and those that were invented.

    Exact string match against the offered set, with no normalization: an id that differs by
    a character is not a near miss to be repaired, it is an id the model did not receive.
    Repairing one would be the system inventing a citation on the model's behalf.
    """
    kept: list[str] = []
    invented: list[str] = []
    for cited in proposal.cited_evidence_ids:
        (kept if cited in offered else invented).append(cited)
    # Deduplicated and ordered, so two runs over one proposal produce one hypothesis.
    return tuple(sorted(set(kept))), tuple(invented)
