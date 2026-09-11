"""The evidence arbiter: a qualitative evidence position per candidate, and a ceiling on it.

Runs between the challenge and the judgment. It answers one question - *does the evidence
carry this claim?* - and answers it on the evidence axis
(:class:`~app.domain.value_objects.enums.EvidencePosition`), never on the status axis. Status
is the judge's, severity is the judge's, and the number is
:mod:`app.domain.services.confidence`'s. There is no field here for any of the three.

Independence is the factor that gets faked, so it is the one the model is not trusted with.
The arbiter may *report* a corroboration count; what survives is the intersection of its
claim with the set :mod:`app.judgment.confidence_inputs` already measured from the executed
retrieval outcomes. Overclaim is recorded and discarded, and the **ceiling** on the position
is computed from the measured count only. So the shape of the attack - several agents citing
one document, or one document reached twice - cannot raise a position, because:

* the schema has no field for an agent, an agent count or an agreement level;
* the ceiling reads ``len(measured distinct source versions)``, which is a set of
  ``source_version_id`` and therefore cannot double-count a version or a strategy;
* ``no_refutation_found`` has no parameter in :func:`ceiling_for`, exactly as it has none in
  ``clamp``. Nobody finding contrary evidence raises nothing, by absence.

Disagreements are carried out, never resolved. Every upstream disagreement the dossier
arrived with plus anything the arbiter adds leaves on the ruling, and the orchestrator feeds
them to the judge through ``dossiers_from(extra_disagreements=...)`` - which has no trimming
parameter.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.assumptions.prompting import wrap_untrusted
from app.domain.entities.change import ChangeSpec
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    Message,
    ProviderError,
    Role,
    StructuredOutputError,
)
from app.domain.value_objects.enums import EvidencePosition
from app.domain.value_objects.provenance import ModelProvenance
from app.ingestion.redaction import RedactionResult
from app.judgment.confidence_inputs import AssembledInputs
from app.judgment.dossier import CandidateDossier, by_handle
from app.judgment.prompting import EvidenceForJudgment

logger = logging.getLogger(__name__)

ARBITER_AGENT: Final = "evidence-arbiter"

ARBITRATION_PROMPT_VERSION: Final = "arbitration-prompt-1"
ARBITRATION_SCHEMA_VERSION: Final = "arbitration-proposal-1"

EVIDENCE_LABEL: Final = "candidate-evidence"

#: Distinct ``source_version_id`` values required before a claim may be called ``supported``.
#: Two, because one source is a single point of failure however authoritative it is - and
#: because "genuine corroboration means distinct source versions" is the arbiter's own rule.
CORROBORATION_FOR_SUPPORTED: Final = 2

#: How much each position asserts. A clamp may only move a position **down** this scale.
ASSERTIVENESS: Final[dict[EvidencePosition, int]] = {
    EvidencePosition.UNRESOLVED: 0,
    EvidencePosition.UNSUPPORTED: 1,
    EvidencePosition.DISPUTED: 2,
    EvidencePosition.WEAKLY_SUPPORTED: 3,
    EvidencePosition.SUPPORTED: 4,
}

#: Enforced structurally by ``extra="forbid"``; named so a test can assert the list.
FORBIDDEN_AGENT_FIELDS: Final = ("confidence", "severity", "status", "consequence_class")

MAX_EVIDENCE_SPANS: Final = 40
SPAN_CHARS: Final = 600


class ClampReason(StrEnum):
    """Why a proposed position was lowered. Every value describes a missing input."""

    NO_SUPPORTING_EVIDENCE = "no_supporting_evidence"
    CONTRADICTING_EVIDENCE = "contradicting_evidence"
    REFUTED_BY_CHALLENGE = "refuted_by_challenge"
    INSUFFICIENT_INDEPENDENT_CORROBORATION = "insufficient_independent_corroboration"


# ---------------------------------------------------------------------------
# What the arbiter may say
# ---------------------------------------------------------------------------


class RulingProposal(BaseModel):
    """One candidate's evidence position, as the model proposed it.

    ``corroborating_source_version_ids`` is a claim to be checked, not a measurement: the
    runtime intersects it with what retrieval actually returned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    handle: str = Field(min_length=2, max_length=8)
    position: Literal["supported", "weakly_supported", "disputed", "unsupported", "unresolved"]

    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    #: Distinct source versions behind the corroboration claim, so the scorer can verify it.
    corroborating_source_version_ids: tuple[str, ...] = ()

    #: Why the corroboration count is defensible on the distinct-source-version definition.
    independence_note: str | None = Field(default=None, min_length=4, max_length=400)
    #: What remains unresolved on the evidence. Preserved verbatim.
    residual_disagreement: str | None = Field(default=None, min_length=4, max_length=400)
    rationale: str | None = Field(default=None, min_length=4, max_length=400)


class RulingBatch(BaseModel):
    """Everything one evidence-arbiter call returns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rulings: tuple[RulingProposal, ...] = ()
    observed_instructions: tuple[str, ...] = ()
    unanswered_questions: tuple[str, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# What survives the gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRuling:
    """One admitted ruling: the position after the ceiling, and everything it was built from."""

    handle: str
    consequence_id: str
    proposed_position: EvidencePosition
    position: EvidencePosition
    clamp_reasons: tuple[ClampReason, ...] = ()

    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    #: Claimed versions that the measurement confirms. The intersection, never the claim.
    corroborating_source_version_ids: tuple[str, ...] = ()
    #: Versions the model claimed that no executed strategy returned for this candidate.
    overclaimed_source_version_ids: tuple[str, ...] = ()
    #: The measured count the ceiling was computed from.
    measured_corroboration: int = 0
    fabricated_evidence_ids: tuple[str, ...] = ()
    #: Upstream disagreements plus anything the arbiter added. Never trimmed.
    disagreements: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def was_clamped(self) -> bool:
        return self.position is not self.proposed_position

    @property
    def carries_evidence(self) -> bool:
        return bool(self.supporting_evidence_ids)


@dataclass(frozen=True)
class ArbitrationOutcome:
    """What one arbitration produced, and what it could not rule on."""

    rulings: tuple[EvidenceRuling, ...] = ()
    #: Candidates the model returned no ruling for. Reported, never defaulted to a position.
    unruled: tuple[CandidateDossier, ...] = ()
    #: Proposals naming a handle that was not offered.
    refused_unknown_handles: tuple[str, ...] = ()
    unanswered_questions: tuple[str, ...] = ()
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
    def by_consequence(self) -> dict[str, EvidenceRuling]:
        return {ruling.consequence_id: ruling for ruling in self.rulings}

    @property
    def clamped(self) -> tuple[EvidenceRuling, ...]:
        return tuple(r for r in self.rulings if r.was_clamped)

    @property
    def overclaimed_corroboration(self) -> tuple[EvidenceRuling, ...]:
        return tuple(r for r in self.rulings if r.overclaimed_source_version_ids)

    @property
    def positions(self) -> dict[str, str]:
        return {r.consequence_id: r.position.value for r in self.rulings}

    @property
    def preserved_disagreements(self) -> dict[str, tuple[str, ...]]:
        """Per-consequence disagreements, in the shape ``dossiers_from`` consumes."""
        return {r.consequence_id: r.disagreements for r in self.rulings if r.disagreements}


# ---------------------------------------------------------------------------
# The deterministic ceiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionCeiling:
    """The strongest position the evidence admits, and why it is not stronger."""

    position: EvidencePosition
    reasons: tuple[ClampReason, ...] = ()


@dataclass(frozen=True)
class EvidenceRulingClamp:
    """The position after the ceiling, and the reasons it moved."""

    position: EvidencePosition
    reasons: tuple[ClampReason, ...] = ()


def ceiling_for(
    *,
    has_supporting_evidence: bool,
    has_contradicting_evidence: bool,
    was_refuted: bool,
    measured_corroboration: int,
) -> PositionCeiling:
    """The frozen ceiling. Four inputs, all deterministic, none of them an agent.

    There is deliberately no parameter for ``no_refutation_found``, for the number of agents
    that agreed, or for the arbiter's own confidence in itself. A challenger looking and
    finding nothing therefore cannot raise a ceiling, because there is nowhere to say it.
    """
    ceilings: list[tuple[int, EvidencePosition, ClampReason]] = []
    if not has_supporting_evidence:
        ceilings.append(
            (
                ASSERTIVENESS[EvidencePosition.UNSUPPORTED],
                EvidencePosition.UNSUPPORTED,
                ClampReason.NO_SUPPORTING_EVIDENCE,
            )
        )
    if has_contradicting_evidence:
        ceilings.append(
            (
                ASSERTIVENESS[EvidencePosition.DISPUTED],
                EvidencePosition.DISPUTED,
                ClampReason.CONTRADICTING_EVIDENCE,
            )
        )
    if was_refuted:
        ceilings.append(
            (
                ASSERTIVENESS[EvidencePosition.DISPUTED],
                EvidencePosition.DISPUTED,
                ClampReason.REFUTED_BY_CHALLENGE,
            )
        )
    if measured_corroboration < CORROBORATION_FOR_SUPPORTED:
        ceilings.append(
            (
                ASSERTIVENESS[EvidencePosition.WEAKLY_SUPPORTED],
                EvidencePosition.WEAKLY_SUPPORTED,
                ClampReason.INSUFFICIENT_INDEPENDENT_CORROBORATION,
            )
        )

    if not ceilings:
        return PositionCeiling(position=EvidencePosition.SUPPORTED)

    lowest = min(rank for rank, _position, _reason in ceilings)
    position = next(p for rank, p, _reason in ceilings if rank == lowest)
    # Every reason is reported, not only the binding one: a claim that is both uncorroborated
    # and contradicted has two problems, and a reader shown one of them is misled about which
    # evidence would fix it.
    return PositionCeiling(
        position=position, reasons=tuple(reason for _rank, _p, reason in ceilings)
    )


def apply_ceiling(proposed: EvidencePosition, ceiling: PositionCeiling) -> EvidenceRulingClamp:
    """Lower ``proposed`` to the ceiling where it exceeds it. Never raises it."""
    if ASSERTIVENESS[proposed] <= ASSERTIVENESS[ceiling.position]:
        return EvidenceRulingClamp(position=proposed, reasons=())
    return EvidenceRulingClamp(position=ceiling.position, reasons=ceiling.reasons)


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------


def arbiter_version(definition: AgentDefinition) -> str:
    return f"arbiter-{definition.version}:{ARBITRATION_PROMPT_VERSION}:{ARBITRATION_SCHEMA_VERSION}"


@dataclass(frozen=True)
class ArbitrationPrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = ARBITRATION_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


class EvidenceArbiter:
    """Invokes ``evidence-arbiter`` over the candidate set and admits what it returns."""

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        if definition.name != ARBITER_AGENT:
            raise ValueError(f"expected the {ARBITER_AGENT!r} definition, got {definition.name!r}")
        self._completions = completions
        self._definition = definition

    @property
    def version(self) -> str:
        return arbiter_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        binding = self._completions.resolve(ARBITER_AGENT)
        return ModelProvenance(
            agent=ARBITER_AGENT,
            prompt_version=self._definition.version,
            prompt_sha256=self._definition.prompt_sha256,
            provider=binding.provider,
            model=binding.model,
            tier=binding.tier,
        )

    async def arbitrate(
        self,
        *,
        change: ChangeSpec,
        dossiers: tuple[CandidateDossier, ...],
        evidence: tuple[EvidenceForJudgment, ...] = (),
        measured: Mapping[str, AssembledInputs] | None = None,
        now: dt.datetime | None = None,
        correlation_id: str | None = None,
    ) -> ArbitrationOutcome:
        """One call, then the ceiling.

        ``measured`` is the deterministic confidence assembly keyed on ``consequence_id``. It
        is the only source of the corroboration count the ceiling uses; a candidate with no
        measurement gets a ceiling of ``weakly_supported`` rather than the benefit of doubt.
        """
        _ = now  # rulings carry no timestamp of their own; the analysis provides one
        provenance = self._provenance()
        measurements = dict(measured or {})

        if not dossiers:
            logger.info(
                "no candidates to arbitrate; evidence-arbiter not called",
                extra={"change_id": change.change_id, "agent": ARBITER_AGENT},
            )
            return ArbitrationOutcome(provenance=provenance, calls_made=0)

        offered = by_handle(dossiers)
        prompt = _build_prompt(
            system_prompt=self._definition.prompt,
            change=change,
            dossiers=dossiers,
            evidence=evidence,
            measured=measurements,
            nonce=uuid.uuid4().hex[:12],
        )

        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "candidate evidence attempted to impersonate the untrusted-content delimiter",
                extra={"change_id": change.change_id, "agent": ARBITER_AGENT},
            )

        base = ArbitrationOutcome(
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                ARBITER_AGENT,
                prompt.messages,
                json_schema=RulingBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            logger.warning(
                "evidence-arbiter call failed; candidates reported unruled",
                extra={"change_id": change.change_id, "error": str(exc)},
            )
            return replace(
                base,
                unruled=dossiers,
                failure=f"{type(exc).__name__}: {exc}",
                calls_made=1,
            )

        try:
            batch = RulingBatch.model_validate(payload)
        except ValidationError as exc:
            logger.warning(
                "evidence-arbiter output violated the proposal schema; discarded",
                extra={"change_id": change.change_id, "errors": exc.error_count()},
            )
            return replace(
                base,
                unruled=dossiers,
                failure=f"schema: {exc.error_count()} error(s)",
                calls_made=1,
            )

        return _admit(batch, base=base, offered=offered, measured=measurements)


def _admit(
    batch: RulingBatch,
    *,
    base: ArbitrationOutcome,
    offered: dict[str, CandidateDossier],
    measured: Mapping[str, AssembledInputs],
) -> ArbitrationOutcome:
    """Apply the ceiling, verify the corroboration claim, and keep every disagreement."""
    rulings: list[EvidenceRuling] = []
    refused: list[str] = []
    ruled: set[str] = set()

    for proposal in batch.rulings:
        dossier = offered.get(proposal.handle)
        if dossier is None:
            refused.append(proposal.handle)
            continue
        if proposal.handle in ruled:
            # One ruling per candidate. A second is a repair artefact, not a second opinion.
            refused.append(proposal.handle)
            continue
        ruled.add(proposal.handle)
        rulings.append(_build(proposal, dossier=dossier, measured=measured))

    if refused:
        logger.warning(
            "evidence-arbiter ruled on handles it was not offered, or twice; refused",
            extra={"agent": ARBITER_AGENT, "handles": sorted(set(refused))},
        )

    return replace(
        base,
        rulings=tuple(rulings),
        unruled=tuple(d for d in offered.values() if d.handle not in ruled),
        refused_unknown_handles=tuple(sorted(set(refused))),
        unanswered_questions=batch.unanswered_questions,
        observed_instructions=batch.observed_instructions,
        calls_made=1,
    )


def _build(
    proposal: RulingProposal,
    *,
    dossier: CandidateDossier,
    measured: Mapping[str, AssembledInputs],
) -> EvidenceRuling:
    """Turn one proposal into a ruling. Every step can only lower what it asserts."""
    admissible = dossier.citable_evidence_ids
    support, fabricated_support = _split(proposal.supporting_evidence_ids, admissible)
    against, fabricated_against = _split(proposal.contradicting_evidence_ids, admissible)

    assembled = measured.get(dossier.consequence_id)
    confirmed_versions = (
        frozenset(assembled.inputs.corroborating_source_version_ids) if assembled else frozenset()
    )
    claimed = tuple(dict.fromkeys(proposal.corroborating_source_version_ids))
    admitted_versions = tuple(v for v in claimed if v in confirmed_versions)
    overclaimed = tuple(v for v in claimed if v not in confirmed_versions)

    proposed = EvidencePosition(proposal.position)
    clamp = apply_ceiling(
        proposed,
        ceiling_for(
            has_supporting_evidence=bool(support),
            has_contradicting_evidence=bool(against),
            was_refuted=dossier.was_refuted,
            # The measured set, not the claimed one. This is the line the attack has to cross
            # and cannot: a model cannot add a source version retrieval never returned.
            measured_corroboration=len(confirmed_versions),
        ),
    )

    notes = tuple(
        note for note in (proposal.independence_note, proposal.rationale) if note is not None
    )
    residual = (proposal.residual_disagreement,) if proposal.residual_disagreement else ()
    disagreements = tuple(dict.fromkeys((*residual, *dossier.upstream_disagreements)))

    return EvidenceRuling(
        handle=dossier.handle,
        consequence_id=dossier.consequence_id,
        proposed_position=proposed,
        position=clamp.position,
        clamp_reasons=clamp.reasons,
        supporting_evidence_ids=support,
        contradicting_evidence_ids=against,
        corroborating_source_version_ids=admitted_versions,
        overclaimed_source_version_ids=overclaimed,
        measured_corroboration=len(confirmed_versions),
        fabricated_evidence_ids=tuple(sorted(set(fabricated_support + fabricated_against))),
        disagreements=disagreements,
        notes=notes,
    )


def _split(
    cited: Sequence[str], admissible: frozenset[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Exact match against the ids offered for this candidate. No normalization."""
    kept: list[str] = []
    invented: list[str] = []
    for item in cited:
        (kept if item in admissible else invented).append(item)
    return tuple(sorted(set(kept))), tuple(invented)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def _build_prompt(
    *,
    system_prompt: str,
    change: ChangeSpec,
    dossiers: tuple[CandidateDossier, ...],
    evidence: tuple[EvidenceForJudgment, ...],
    measured: Mapping[str, AssembledInputs],
    nonce: str,
) -> ArbitrationPrompt:
    rendered, redaction, breakout = wrap_untrusted(
        _render_evidence(evidence), nonce=nonce, label=EVIDENCE_LABEL
    )
    task = "\n\n".join(
        (
            _render_change(change),
            _render_candidates(dossiers, measured=measured),
            rendered,
            _render_instructions(),
        )
    )
    return ArbitrationPrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        redaction=redaction,
        attempted_delimiter_breakout=breakout,
        nonce=nonce,
    )


def _render_change(change: ChangeSpec) -> str:
    payload = {
        "change_id": change.change_id,
        "subject_entities": list(change.subject_entities),
        "old_state": change.old_state,
        "new_state": change.new_state,
    }
    return "## The proposed change\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_candidates(
    dossiers: tuple[CandidateDossier, ...], *, measured: Mapping[str, AssembledInputs]
) -> str:
    lines = ["## Candidates"]
    for dossier in dossiers:
        assembled = measured.get(dossier.consequence_id)
        versions = assembled.inputs.corroborating_source_version_ids if assembled else ()
        strategies = assembled.inputs.corroborating_strategies if assembled else ()
        unmeasured = assembled.coverage.unmeasured if assembled else ("all factors",)
        citable = ", ".join(f"`{i}`" for i in sorted(dossier.citable_evidence_ids)) or "(none)"

        lines.extend(
            (
                "",
                f"### `{dossier.handle}` - {dossier.consequence_class.value}",
                f"claim: {dossier.claim}",
                f"mechanism: {dossier.mechanism or '(deterministically derived)'}",
                f"challenged: {'refuted' if dossier.was_refuted else 'no refutation admitted'}",
                (
                    "measured distinct source versions: "
                    f"{len(set(versions))} ({', '.join(sorted(set(versions))) or 'none'})"
                ),
                f"strategies that reached them: {', '.join(sorted(set(strategies))) or 'none'}",
                f"factors nothing could measure: {', '.join(unmeasured) or 'none'}",
                f"evidence you may cite for this candidate: {citable}",
            )
        )
        for item in dossier.upstream_disagreements:
            lines.append(f"upstream disagreement: {item}")
    return "\n".join(lines)


def _render_evidence(evidence: tuple[EvidenceForJudgment, ...]) -> str:
    if not evidence:
        return "(no spans were retrieved for these candidates)"
    blocks = []
    for span in evidence:
        handles = ", ".join(f"`{h}`" for h in span.handles)
        blocks.append(
            f"[evidence_id: {span.evidence_id}] "
            f"(source: {span.source_title}, authority: {span.authority_class}, "
            f"bears on: {handles})\n{span.text[:SPAN_CHARS]}"
        )
    return "\n\n".join(blocks)


def _render_instructions() -> str:
    return "\n".join(
        (
            "## Your task",
            "",
            "For each candidate, state the evidence position: `supported`, "
            "`weakly_supported`, `disputed`, `unsupported` or `unresolved`.",
            "",
            "Cite only evidence ids listed for that candidate. Report the distinct "
            "`source_version_id` values behind any corroboration claim - not a count. The "
            "runtime keeps only the versions retrieval actually returned, and computes the "
            "ceiling on your position from those.",
            "",
            "Several agents citing one document is one source. The same claim reached by two "
            "strategies is one source found twice. There is no field in the schema for an "
            "agent, an agent count or an agreement level.",
            "",
            "Do **not** emit a confidence number, a severity, a status or a consequence "
            "class. The schema will reject all four.",
            "",
            "Where the evidence leaves something unresolved, say so in "
            "`residual_disagreement`. It is preserved verbatim and reaches the judge.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )


def positions_by_consequence(outcome: ArbitrationOutcome) -> dict[str, EvidencePosition]:
    """Positions keyed on ``consequence_id``, for a caller reporting them beside a status."""
    return {r.consequence_id: r.position for r in outcome.rulings}
