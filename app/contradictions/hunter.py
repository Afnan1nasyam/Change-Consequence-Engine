"""The contradiction-hunter runtime: rate a closed set, then a gate over what came back.

The Class A annotation counterpart of :mod:`app.consequences.hunter`, and the split is the
same: the model **proposes**, deterministic code **admits**. What differs is how little the
model is proposing. The divergence set arrives detected and classified; this agent adds
materiality and temporal position and can add nothing else.

Four properties make that structural rather than instructed:

**No search, and no way to search.** This class holds a completion port and an agent
definition. It has no retriever, no store and no corpus access, so "it must not independently
search for contradictions" is not a rule it follows - it is a capability it does not have. It
rates the items it is handed.

**An empty divergence set costs zero calls.** If the resolver found nothing, there is nothing
to rate, and :meth:`ContradictionHunter.assess` returns before it reaches the provider. That
is the strongest form of "a corpus with no deterministic divergence produces no contradiction
findings": not a filter over the model's output, but a model that was never asked.

**An assessment of a divergence that was not detected is dropped.** Handles are verified
against exactly the set the prompt offered. A model that invents ``d9`` has not found a ninth
divergence; it has produced an assessment with no subject.

**Nothing here can reclassify, suppress or promote.** The class travels on the
:class:`~app.contradictions.divergence.DivergenceItem`, which is frozen and built from the
deterministic record. The assessment schema has no field for a class, a relationship or a
relation id. So an admitted assessment is strictly *additive* to a Class A finding - it can
say a divergence is immaterial and it cannot make the divergence stop existing.

Every check below can only weaken a claim the model made. There is no branch that adds
evidence, raises a materiality, or changes a class.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import Final

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.contradictions.divergence import (
    DivergenceClass,
    DivergenceItem,
    by_handle,
)
from app.contradictions.prompting import (
    CONTRADICTION_PROMPT_VERSION,
    ContradictionPrompt,
    EvidenceForContradiction,
    build_contradiction_prompt,
)
from app.contradictions.proposals import (
    ASSESSMENT_SCHEMA_VERSION,
    AssessmentBatch,
    DivergenceAssessment,
    Materiality,
)
from app.domain.entities.change import ChangeSpec
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    ProviderError,
    StructuredOutputError,
)
from app.domain.value_objects.provenance import ModelProvenance

logger = logging.getLogger(__name__)

#: The agent this runtime invokes. One name, so a mismatched definition is caught at
#: construction rather than producing assessments attributed to the wrong prompt.
CONTRADICTION_AGENT: Final = "contradiction-hunter"


def contradiction_hunter_version(definition: AgentDefinition) -> str:
    """The version stamped on every assessment: prompt, contract shape and schema together.

    All three, because each changes what the output means. A prompt edit changes what was
    asked, a contract change changes how it was asked, and a schema change changes what
    could be answered.
    """
    return (
        f"contradiction-{definition.version}:"
        f"{CONTRADICTION_PROMPT_VERSION}:{ASSESSMENT_SCHEMA_VERSION}"
    )


@dataclass(frozen=True)
class AssessedDivergence:
    """One deterministic divergence and, where the model supplied one, its assessment.

    Paired rather than merged. ``item`` is the resolver's conclusion and ``assessment`` is
    the model's annotation of it, and keeping them separate is what makes "Class A remains
    authoritative" visible in the type: a consumer reading ``divergence_class`` never passes
    through anything the model wrote.

    ``assessment`` is ``None`` for an item the model did not rate. That is a reportable
    absence, not a default: a divergence nobody assessed is still a divergence.
    """

    item: DivergenceItem
    assessment: DivergenceAssessment | None = None
    #: Cited ids that were actually offered **for this item**. A subset of the model's
    #: claim, never a superset.
    admitted_evidence_ids: tuple[str, ...] = ()

    # -- the deterministic side, delegated so a caller cannot reach past it --

    @property
    def item_id(self) -> str:
        return self.item.item_id

    @property
    def divergence_class(self) -> DivergenceClass:
        """The resolver's classification. Never influenced by ``assessment``."""
        return self.item.divergence_class

    @property
    def is_contradiction(self) -> bool:
        """Whether a reader may see this described as a contradiction.

        Reads the deterministic class only. A model rating a legitimate override
        ``material`` cannot make this true, and a model rating a real contradiction
        ``immaterial`` cannot make it false.
        """
        return self.item.reportable_as_contradiction

    @property
    def is_governance_defect(self) -> bool:
        return self.item.is_governance_defect

    @property
    def materiality(self) -> Materiality | None:
        return self.assessment.materiality if self.assessment else None

    @property
    def was_assessed(self) -> bool:
        return self.assessment is not None


@dataclass(frozen=True)
class ContradictionOutcome:
    """What one contradiction-hunter call produced, and what producing it revealed.

    ``failure`` is a string rather than an exception so the outcome stays serializable and
    an analysis can record *that* the stage failed alongside everything that succeeded.
    """

    #: Every supplied divergence, assessed or not. The set is never shortened: an item the
    #: model ignored still reaches the reader, because the resolver found it.
    assessed: tuple[AssessedDivergence, ...] = ()
    #: Handles the model rated that it was never offered. Reported, because a model
    #: inventing subjects is a signal about the model and not only about one assessment.
    invented_item_ids: tuple[str, ...] = ()
    #: Handles the model rated more than once. The first assessment wins; the rest are
    #: dropped, because two ratings of one divergence cannot both be the assessment.
    duplicate_item_ids: tuple[str, ...] = ()
    #: Cited ids that were not offered for the item they were cited against - whether
    #: invented outright or borrowed from another divergence.
    fabricated_evidence_ids: tuple[str, ...] = ()
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

    @property
    def contradictions(self) -> tuple[AssessedDivergence, ...]:
        """Items a reader may see as contradictions. Filtered on the deterministic class."""
        return tuple(a for a in self.assessed if a.is_contradiction)

    @property
    def governance_defects(self) -> tuple[AssessedDivergence, ...]:
        return tuple(a for a in self.assessed if a.is_governance_defect)

    @property
    def unassessed(self) -> tuple[AssessedDivergence, ...]:
        return tuple(a for a in self.assessed if not a.was_assessed)

    @property
    def of_class(self) -> dict[DivergenceClass, tuple[AssessedDivergence, ...]]:
        """The five buckets, for a reader who needs them apart."""
        return {
            divergence_class: tuple(
                a for a in self.assessed if a.divergence_class is divergence_class
            )
            for divergence_class in DivergenceClass
        }


class ContradictionHunter:
    """Invokes ``contradiction-hunter`` over a closed divergence set and admits the ratings.

    Holds the completion port and the agent definition, and **nothing else**. The absence of
    a retriever here is the point: this agent cannot search the corpus because it has no way
    to reach it. The port is injected, so this class never names a provider or a model -
    ``AgentCompletionPort`` resolves tier, provider and model from configuration.
    """

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        if definition.name != CONTRADICTION_AGENT:
            raise ValueError(
                f"expected the {CONTRADICTION_AGENT!r} definition, got {definition.name!r}"
            )
        self._completions = completions
        self._definition = definition

    @property
    def version(self) -> str:
        return contradiction_hunter_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        """Attribution for anything this agent asserts.

        Resolved without a model call, so a failed call still carries full provenance and
        the outcome can say which prompt and which binding failed.
        """
        binding = self._completions.resolve(CONTRADICTION_AGENT)
        return ModelProvenance(
            agent=CONTRADICTION_AGENT,
            prompt_version=self._definition.version,
            prompt_sha256=self._definition.prompt_sha256,
            provider=binding.provider,
            model=binding.model,
            tier=binding.tier,
        )

    async def assess(
        self,
        *,
        change: ChangeSpec,
        divergences: tuple[DivergenceItem, ...],
        evidence: tuple[EvidenceForContradiction, ...] = (),
        correlation_id: str | None = None,
    ) -> ContradictionOutcome:
        """Rate the supplied divergences. One call, then the gate.

        ``divergences`` is the deterministic set and is **closed**: this method assesses
        exactly what it is given, adds nothing, and removes nothing.
        """
        provenance = self._provenance()

        if not divergences:
            # No divergence, no call, no findings. The provider is never reached, so there
            # is no output to filter and no opportunity for a model to supply a
            # contradiction the resolver did not find.
            logger.info(
                "no deterministic divergence for this change; contradiction-hunter not called",
                extra={"change_id": change.change_id, "agent": CONTRADICTION_AGENT},
            )
            return ContradictionOutcome(provenance=provenance, calls_made=0)

        # Refuses a duplicate handle before the call, because an ambiguous handle would make
        # an assessment unattributable and the model cannot be blamed for that.
        offered = by_handle(divergences)

        prompt = build_contradiction_prompt(
            system_prompt=self._definition.prompt,
            change=change,
            divergences=divergences,
            evidence=evidence,
            nonce=uuid.uuid4().hex[:12],
        )
        self._log_prompt_observations(prompt, change=change)

        base = ContradictionOutcome(
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                CONTRADICTION_AGENT,
                prompt.messages,
                json_schema=AssessmentBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            # The existing error path. A failed stage is reported, never inferred from an
            # empty result - and the divergence set survives it unassessed rather than
            # disappearing, because the resolver's findings do not depend on this call.
            logger.warning(
                "contradiction-hunter call failed; divergences reported without assessment",
                extra={"change_id": change.change_id, "error": str(exc)},
            )
            return replace(
                base,
                assessed=_unassessed(divergences),
                failure=f"{type(exc).__name__}: {exc}",
                calls_made=1,
            )

        try:
            batch = AssessmentBatch.model_validate(payload)
        except ValidationError as exc:
            # Includes the cases that matter most: an agent emitting `confidence`,
            # `severity`, or a restatement of `divergence_class` / `relationship`.
            # `extra="forbid"` turns each into a rejection instead of a dropped key.
            logger.warning(
                "contradiction-hunter output violated the assessment schema; discarded",
                extra={"change_id": change.change_id, "errors": exc.error_count()},
            )
            return replace(
                base,
                assessed=_unassessed(divergences),
                failure=f"schema: {exc.error_count()} error(s)",
                calls_made=1,
            )

        return self._admit(batch, base=base, offered=offered, prompt=prompt)

    def _log_prompt_observations(self, prompt: ContradictionPrompt, *, change: ChangeSpec) -> None:
        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "divergence evidence attempted to impersonate the untrusted-content delimiter",
                extra={"change_id": change.change_id, "agent": CONTRADICTION_AGENT},
            )
        if prompt.redaction.found_any:
            logger.info(
                "redacted credential-shaped spans before dispatch",
                extra={"change_id": change.change_id, "secrets": prompt.redaction.summary},
            )

    # -- the gate --------------------------------------------------------

    def _admit(
        self,
        batch: AssessmentBatch,
        *,
        base: ContradictionOutcome,
        offered: dict[str, DivergenceItem],
        prompt: ContradictionPrompt,
    ) -> ContradictionOutcome:
        """Attach each assessment to the divergence it claims to be about.

        Attachment is by exact handle match. Every step can weaken what the model said and
        none can strengthen it: there is no branch here that adds an item, changes a class,
        or admits a citation the prompt did not offer for that item.
        """
        accepted: dict[str, DivergenceAssessment] = {}
        admitted_evidence: dict[str, tuple[str, ...]] = {}
        invented: list[str] = []
        duplicated: list[str] = []
        fabricated: list[str] = []

        for assessment in batch.assessments:
            handle = assessment.item_id
            if handle not in offered:
                # Not a ninth divergence. An assessment with no subject.
                invented.append(handle)
                continue
            if handle in accepted:
                # First wins. Two ratings of one divergence cannot both be the assessment,
                # and choosing between them would be this module deciding materiality.
                duplicated.append(handle)
                continue

            kept, borrowed = _partition_citations(
                assessment, admissible=prompt.offered_evidence_ids.get(handle, frozenset())
            )
            fabricated.extend(borrowed)
            accepted[handle] = assessment
            admitted_evidence[handle] = kept

        if invented:
            logger.warning(
                "contradiction-hunter assessed divergences it was not shown; dropped",
                extra={"agent": CONTRADICTION_AGENT, "count": len(invented)},
            )
        if fabricated:
            logger.warning(
                "contradiction-hunter cited evidence not admissible for the item; dropped",
                extra={"agent": CONTRADICTION_AGENT, "count": len(fabricated)},
            )

        assessed = tuple(
            AssessedDivergence(
                item=item,
                assessment=accepted.get(handle),
                admitted_evidence_ids=admitted_evidence.get(handle, ()),
            )
            for handle, item in offered.items()
        )

        return replace(
            base,
            assessed=assessed,
            invented_item_ids=tuple(sorted(set(invented))),
            duplicate_item_ids=tuple(sorted(set(duplicated))),
            fabricated_evidence_ids=tuple(sorted(set(fabricated))),
            observed_instructions=batch.observed_instructions,
            unanswered_questions=batch.unanswered_questions,
            calls_made=1,
        )


def _unassessed(divergences: tuple[DivergenceItem, ...]) -> tuple[AssessedDivergence, ...]:
    """Every divergence, carried through with no assessment.

    Used on both failure paths. The deterministic findings do not depend on the model call,
    so losing the call must not lose them: a failed stage costs the *rating*, not the
    divergence.
    """
    return tuple(AssessedDivergence(item=item) for item in divergences)


def _partition_citations(
    assessment: DivergenceAssessment, *, admissible: frozenset[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split cited ids into those admissible **for this item** and those that are not.

    Exact string match, no normalization: an id that differs by a character is not a near
    miss to be repaired, it is an id the model did not receive for this divergence, and
    repairing one would be the system inventing a citation on the model's behalf.

    The per-item check matters as much as the existence check. A real evidence id belonging
    to a different divergence would support an unrelated claim, which is a fabricated
    citation assembled out of genuine parts.
    """
    kept: list[str] = []
    borrowed: list[str] = []
    for cited in assessment.cited_evidence_ids:
        (kept if cited in admissible else borrowed).append(cited)
    # Deduplicated and ordered, so two runs over one assessment produce one result.
    return tuple(sorted(set(kept))), tuple(borrowed)
