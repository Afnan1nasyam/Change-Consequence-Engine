"""The blind-spot hunter runtime: one call over the partition, then the same narrow gate.

Structurally the main hunter's twin, and the differences are the point:

**Citations are checked against the partition, not against the analysis.** An id the main
consequence-hunter was offered is dropped here exactly like a fabricated one. That is what
makes "independently represented" a property of the output rather than a claim about the
prompt: a blind-spot finding rests on evidence the main analysis did not see, or on nothing.

**Attribution is derived, never accepted.** The slice credited to a finding comes from the
frozen priority order over the spans it actually cited. A model that names a slice it did not
work has its claim recorded beside the derived answer and discarded.

**A self-reported restatement is dropped.** The agent is asked to flag when its partition led
it back into the main analysis; when it does, the finding does not survive. Dropping it is
cheaper than a duplicate a reader has to recognise, and it keeps yield honest.

**Corpus silence stays a separate kind.** It has no claim, no mechanism and no citations, so
there is no code path by which one becomes a consequence. It is also **not** an
:class:`~app.domain.entities.claims.EvidenceGap`: a gap carries negative provenance from an
executed query, and this agent holds no trace.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Final

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.blindspots.partition import SLICE_PRIORITY, EvidencePartition, PartitionSlice
from app.blindspots.prompting import (
    BLIND_SPOT_PROMPT_VERSION,
    build_blind_spot_prompt,
)
from app.blindspots.proposals import (
    BLIND_SPOT_SCHEMA_VERSION,
    BlindSpotBatch,
    BlindSpotProposal,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.claims import ConsequenceHypothesis
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    ProviderError,
    StructuredOutputError,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import ModelProvenance

logger = logging.getLogger(__name__)

BLIND_SPOT_AGENT: Final = "blind-spot-hunter"


def blind_spot_version(definition: AgentDefinition) -> str:
    """Prompt, contract shape and schema together, as every other agent stamps it."""
    return f"blindspot-{definition.version}:{BLIND_SPOT_PROMPT_VERSION}:{BLIND_SPOT_SCHEMA_VERSION}"


@dataclass(frozen=True)
class SilenceRecord:
    """Something the change touches that the partition shows the corpus silent on.

    Carries no id and no evidence, because it asserts neither. It leaves this layer as an
    unresolved question plus the source that would answer it.
    """

    question: str
    why_it_matters: str
    recommended_source: str
    #: The slice the model worked when it found nothing. Unverifiable - there is no citation
    #: to derive it from - so it is recorded as claimed and labelled that way.
    claimed_slice: str | None = None


@dataclass(frozen=True)
class BlindSpotOutcome:
    """What one blind-spot call produced, and what producing it revealed."""

    hypotheses: tuple[ConsequenceHypothesis, ...] = ()
    silences: tuple[SilenceRecord, ...] = ()
    #: Derived slice per retained hypothesis id. The attribution a yield figure may use.
    slice_of: dict[str, PartitionSlice] = field(default_factory=dict)
    #: Slices the model claimed that the derived attribution disagreed with.
    misattributed_ids: tuple[str, ...] = ()
    affected_artifact_of: dict[str, str] = field(default_factory=dict)
    #: Cited ids that were not in the partition. Includes ids the main analysis was offered,
    #: which is the case worth separating out.
    fabricated_evidence_ids: tuple[str, ...] = ()
    reused_main_evidence_ids: tuple[str, ...] = ()
    #: Findings the model itself flagged as restatements of the main analysis. Dropped.
    restated_claims: tuple[str, ...] = ()
    #: Analogue ids this hunter claimed. Always dropped - see :meth:`BlindSpotHunter._admit`.
    fabricated_analogue_ids: tuple[str, ...] = ()
    mechanism_only_ids: tuple[str, ...] = ()
    without_analogue_ids: tuple[str, ...] = ()
    unusable_slices: tuple[str, ...] = ()
    observed_instructions: tuple[str, ...] = ()
    secrets_redacted: str = "none"
    attempted_delimiter_breakout: bool = False
    calls_made: int = 0
    failure: str | None = None
    provenance: ModelProvenance | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    def by_slice(self, slice_: PartitionSlice) -> tuple[ConsequenceHypothesis, ...]:
        return tuple(h for h in self.hypotheses if self.slice_of.get(h.hypothesis_id) is slice_)

    @property
    def yield_by_slice(self) -> dict[str, int]:
        """Findings per slice, so blind-spot yield can be attributed rather than assumed."""
        counts = {s.value: 0 for s in SLICE_PRIORITY}
        for slice_ in self.slice_of.values():
            counts[slice_.value] += 1
        return counts


class BlindSpotHunter:
    """Invokes ``blind-spot-hunter`` over a partition and admits what it returns."""

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        if definition.name != BLIND_SPOT_AGENT:
            raise ValueError(
                f"expected the {BLIND_SPOT_AGENT!r} definition, got {definition.name!r}"
            )
        self._completions = completions
        self._definition = definition

    @property
    def version(self) -> str:
        return blind_spot_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        binding = self._completions.resolve(BLIND_SPOT_AGENT)
        return ModelProvenance(
            agent=BLIND_SPOT_AGENT,
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
        partition: EvidencePartition,
        existing_claims: tuple[str, ...] = (),
        correlation_id: str | None = None,
    ) -> BlindSpotOutcome:
        """One call over the partition, then the gate.

        An empty partition still calls: the agent's honest answer there is "every span for
        this change was already read", and corpus silence is a reportable finding that needs
        no evidence. What it cannot do is produce a cited consequence, because there is
        nothing admissible to cite.
        """
        prompt = build_blind_spot_prompt(
            system_prompt=self._definition.prompt,
            change=change,
            partition=partition,
            existing_claims=existing_claims,
            nonce=uuid.uuid4().hex[:12],
        )
        provenance = self._provenance()

        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "partition evidence attempted to impersonate the untrusted-content delimiter",
                extra={"change_id": change.change_id, "agent": BLIND_SPOT_AGENT},
            )

        base = BlindSpotOutcome(
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                BLIND_SPOT_AGENT,
                prompt.messages,
                json_schema=BlindSpotBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            logger.warning(
                "blind-spot-hunter call failed; no independent findings this run",
                extra={"change_id": change.change_id, "error": str(exc)},
            )
            return replace(base, failure=f"{type(exc).__name__}: {exc}", calls_made=1)

        try:
            batch = BlindSpotBatch.model_validate(payload)
        except ValidationError as exc:
            logger.warning(
                "blind-spot-hunter output violated the proposal schema; discarded",
                extra={"change_id": change.change_id, "errors": exc.error_count()},
            )
            return replace(base, failure=f"schema: {exc.error_count()} error(s)", calls_made=1)

        return self._admit(batch, base=base, partition=partition)

    # -- the gate --------------------------------------------------------

    def _admit(
        self,
        batch: BlindSpotBatch,
        *,
        base: BlindSpotOutcome,
        partition: EvidencePartition,
    ) -> BlindSpotOutcome:
        """Turn proposals into hypotheses. Every step can weaken a claim; none can strengthen.

        **A claimed historical analogue is always dropped here.** This hunter is handed no
        admissible anchor set: its partition is disjoint from the main analysis by design, so
        the historical analyst's evidence is not in it and nothing could have evidenced a
        precedent for it. Every blind-spot hypothesis therefore caps at ``speculative`` - which
        is the honest position for a finding built on the thinnest evidence in the analysis,
        and the alternative was a well-formed invented id lifting exactly those findings.
        """
        assert base.provenance is not None  # set by `hunt` before the call

        hypotheses: list[ConsequenceHypothesis] = []
        slices: dict[str, PartitionSlice] = {}
        affected: dict[str, str] = {}
        fabricated: list[str] = []
        reused: list[str] = []
        restated: list[str] = []
        misattributed: list[str] = []
        invented_anchors: list[str] = []
        mechanism_only: list[str] = []
        without_analogue: list[str] = []

        for proposal in batch.consequences:
            if proposal.restates_existing:
                restated.append(proposal.claim)
                continue

            kept, invented = _partition_citations(proposal, partition=partition)
            fabricated.extend(invented)
            reused.extend(i for i in invented if i in partition.main_offered_ids)

            if proposal.historical_analogue_id is not None:
                invented_anchors.append(proposal.historical_analogue_id)

            hypothesis_id = new_id("hypothesis")
            hypotheses.append(
                ConsequenceHypothesis(
                    hypothesis_id=hypothesis_id,
                    claim=proposal.claim,
                    mechanism=" -> ".join(proposal.mechanism_chain),
                    weakest_link=proposal.weakest_link,
                    affected_entities=proposal.affected_entities,
                    assumption_keys=proposal.assumption_keys,
                    evidence_ids=kept,
                    historical_analogue_id=None,
                    proposed_by=base.provenance,
                )
            )

            derived = _derive_slice(kept, partition=partition)
            if derived is not None:
                slices[hypothesis_id] = derived
            if proposal.partition_slice is not None and (
                derived is None or proposal.partition_slice != derived.value
            ):
                misattributed.append(hypothesis_id)

            if proposal.affected_source_version_id is not None:
                affected[hypothesis_id] = proposal.affected_source_version_id
            if not kept:
                mechanism_only.append(hypothesis_id)
            without_analogue.append(hypothesis_id)

        if fabricated:
            logger.warning(
                "blind-spot-hunter cited ids outside its partition; dropped",
                extra={
                    "agent": BLIND_SPOT_AGENT,
                    "count": len(fabricated),
                    "from_main_analysis": len(reused),
                },
            )

        return replace(
            base,
            hypotheses=tuple(hypotheses),
            silences=tuple(
                SilenceRecord(
                    question=item.question,
                    why_it_matters=item.why_it_matters,
                    recommended_source=item.recommended_source,
                    claimed_slice=item.partition_slice,
                )
                for item in batch.corpus_silence
            ),
            slice_of=slices,
            misattributed_ids=tuple(misattributed),
            affected_artifact_of=affected,
            fabricated_evidence_ids=tuple(sorted(set(fabricated))),
            reused_main_evidence_ids=tuple(sorted(set(reused))),
            restated_claims=tuple(restated),
            fabricated_analogue_ids=tuple(sorted(set(invented_anchors))),
            mechanism_only_ids=tuple(mechanism_only),
            without_analogue_ids=tuple(without_analogue),
            unusable_slices=batch.unusable_slices,
            observed_instructions=batch.observed_instructions,
            calls_made=1,
        )


def _partition_citations(
    proposal: BlindSpotProposal, *, partition: EvidencePartition
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split cited ids into those in the partition and those outside it.

    Exact string match, no normalization, and the *partition* is the admissible set rather
    than the analysis's evidence: an id the main hunter was offered is outside it.
    """
    admissible = partition.evidence_ids
    kept: list[str] = []
    invented: list[str] = []
    for cited in proposal.cited_evidence_ids:
        (kept if cited in admissible else invented).append(cited)
    return tuple(sorted(set(kept))), tuple(invented)


def _derive_slice(kept: tuple[str, ...], *, partition: EvidencePartition) -> PartitionSlice | None:
    """The slice this finding is credited to: the highest-priority slice it actually cited."""
    found = {partition.slice_of(evidence_id) for evidence_id in kept}
    for slice_ in SLICE_PRIORITY:
        if slice_ in found:
            return slice_
    return None
