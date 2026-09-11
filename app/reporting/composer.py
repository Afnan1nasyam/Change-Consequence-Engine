"""The report-composer runtime: narrow the foreground, order it, gate the prose.

Three admitted effects, and each can only make the report *shorter* or *better ordered*:

**Narrowing.** The final foreground is the deterministic eligible set intersected with what
the composer proposed - so a handle it was not offered cannot be added. Findings the surface
rule makes mandatory (``supported`` or ``likely`` at HIGH or CRITICAL) are foregrounded
whatever the composer says, which is the floor that stops "reduce alert load" from becoming
"hide the alarming ones".

**Ordering.** A preference over the foreground. Anything the composer omits from its order is
appended rather than dropped, so an omission cannot remove a finding from the surface.

**Narration.** Advisory only, and refused whole if any identifier-shaped token in it is not
already in the finalized set. Refused whole rather than repaired: a summary with one invented
citation removed is a summary whose other sentences were written by the same process.

Confidence and severity are carried through by the *caller*, not by this module - it never
sees a number it could round and has no field to put one in.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Final

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.assumptions.prompting import wrap_untrusted
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    Message,
    ProviderError,
    Role,
    StructuredOutputError,
)
from app.domain.value_objects.provenance import ModelProvenance
from app.ingestion.redaction import RedactionResult
from app.reporting.proposals import (
    COMPOSITION_SCHEMA_VERSION,
    CompositionProposal,
)

logger = logging.getLogger(__name__)

COMPOSER_AGENT: Final = "report-composer"

COMPOSITION_PROMPT_VERSION: Final = "report-composition-prompt-1"

FINDINGS_LABEL: Final = "finalized-findings"

HANDLE_PREFIX: Final = "r"

#: Any token shaped like one of this system's ids. The narration gate checks every match
#: against the finalized set, so a plausible-looking id that was never produced is caught.
_ID_TOKEN: Final = re.compile(r"\b((?:ev|cq|eg|dr|sv|src|an|chg|ca|hy|cl|ft|sf)_[0-9a-f]{32})\b")

#: Why a narration was refused. Reported, so a reader knows prose was withheld and why.
REFUSAL_INVENTED_ID: Final = "cited an identifier absent from the finalized findings"
REFUSAL_SCHEMA: Final = "output violated the composition schema"


@dataclass(frozen=True)
class CompositionCandidate:
    """One finalized finding as the composer sees it. A flat projection, deliberately.

    No mechanism, no evidence text and no provenance: the composer is not reviewing the
    finding, it is deciding whether a reader sees it first. ``mandatory`` is computed by the
    caller from the frozen surface rule, and is the floor this agent cannot lower.
    """

    handle: str
    consequence_id: str
    title: str
    status: str
    severity: str
    consequence_class: str
    confidence: float | None = None
    owner: str | None = None
    has_historical_analogue: bool = False
    is_blind_spot: bool = False
    mandatory: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComposedNarration:
    """Prose that passed the id gate. Advisory; never the report's executive summary."""

    executive_summary: str | None = None
    section_notes: tuple[tuple[str, str], ...] = ()
    empty_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComposedLayout:
    """What the composer changed about the surface, and what it was refused."""

    #: Consequence ids a reader sees first, in reading order.
    foreground: tuple[str, ...] = ()
    #: Everything else, reachable but not foregrounded.
    reachable: tuple[str, ...] = ()
    narration: ComposedNarration | None = None
    #: Handles the composer proposed that it was not offered.
    refused_handles: tuple[str, ...] = ()
    #: Mandatory findings the composer tried to drop from the surface.
    restored_handles: tuple[str, ...] = ()
    narration_refusal: str | None = None
    #: Ids in the narration that were not in the finalized set.
    invented_ids: tuple[str, ...] = ()
    secrets_redacted: str = "none"
    attempted_delimiter_breakout: bool = False
    calls_made: int = 0
    failure: str | None = None
    provenance: ModelProvenance | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    @property
    def narrowed_by(self) -> int:
        return len(self.reachable)


def composer_version(definition: AgentDefinition) -> str:
    return (
        f"composer-{definition.version}:{COMPOSITION_PROMPT_VERSION}:{COMPOSITION_SCHEMA_VERSION}"
    )


@dataclass(frozen=True)
class CompositionPrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = COMPOSITION_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


class ReportComposer:
    """Invokes ``report-composer`` over the finalized findings and admits what it returns."""

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        if definition.name != COMPOSER_AGENT:
            raise ValueError(f"expected the {COMPOSER_AGENT!r} definition, got {definition.name!r}")
        self._completions = completions
        self._definition = definition

    @property
    def version(self) -> str:
        return composer_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        binding = self._completions.resolve(COMPOSER_AGENT)
        return ModelProvenance(
            agent=COMPOSER_AGENT,
            prompt_version=self._definition.version,
            prompt_sha256=self._definition.prompt_sha256,
            provider=binding.provider,
            model=binding.model,
            tier=binding.tier,
        )

    async def compose(
        self,
        *,
        candidates: Sequence[CompositionCandidate],
        eligible: Sequence[str],
        finalized_ids: frozenset[str],
        counts: dict[str, int],
        correlation_id: str | None = None,
    ) -> ComposedLayout:
        """One call, then the three gates.

        ``eligible`` are the handles the deterministic surface rule already allows to be
        foregrounded. ``finalized_ids`` is every identifier the analysis produced, and is what
        the narration is checked against.
        """
        provenance = self._provenance()
        by_handle = {c.handle: c for c in candidates}
        mandatory = tuple(c.handle for c in candidates if c.mandatory)

        if not candidates:
            return ComposedLayout(provenance=provenance, calls_made=0)

        prompt = _build_prompt(
            system_prompt=self._definition.prompt,
            candidates=candidates,
            eligible=eligible,
            counts=counts,
            nonce=uuid.uuid4().hex[:12],
        )
        base = ComposedLayout(
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                COMPOSER_AGENT,
                prompt.messages,
                json_schema=CompositionProposal.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            logger.warning(
                "report-composer call failed; the deterministic surface stands",
                extra={"agent": COMPOSER_AGENT, "error": str(exc)},
            )
            return replace(base, failure=f"{type(exc).__name__}: {exc}", calls_made=1)

        try:
            proposal = CompositionProposal.model_validate(payload)
        except ValidationError as exc:
            logger.warning(
                "report-composer output violated the composition schema; discarded",
                extra={"agent": COMPOSER_AGENT, "errors": exc.error_count()},
            )
            return replace(base, failure=f"schema: {exc.error_count()} error(s)", calls_made=1)

        return _admit(
            proposal,
            base=base,
            by_handle=by_handle,
            eligible=tuple(eligible),
            mandatory=mandatory,
            finalized_ids=finalized_ids,
        )


def _admit(
    proposal: CompositionProposal,
    *,
    base: ComposedLayout,
    by_handle: dict[str, CompositionCandidate],
    eligible: tuple[str, ...],
    mandatory: tuple[str, ...],
    finalized_ids: frozenset[str],
) -> ComposedLayout:
    """Intersect, restore, order, then gate the prose."""
    proposed = tuple(dict.fromkeys(proposal.foreground_handles))
    refused = tuple(h for h in proposed if h not in by_handle)

    kept = [h for h in eligible if h in set(proposed)]
    restored = tuple(h for h in mandatory if h not in kept)
    # Mandatory findings go back on the surface in the deterministic order, ahead of the
    # composer's preference. A HIGH `supported` finding a composer wanted buried is exactly
    # the case the floor exists for.
    foreground_handles = [*restored, *kept]

    order = [h for h in dict.fromkeys(proposal.reading_order) if h in foreground_handles]
    order.extend(h for h in foreground_handles if h not in order)

    reachable = tuple(
        by_handle[h].consequence_id for h in by_handle if h not in set(foreground_handles)
    )

    narration, refusal, invented = _gate_narration(proposal, finalized_ids=finalized_ids)

    if refused:
        logger.warning(
            "report-composer foregrounded handles it was not offered; dropped",
            extra={"agent": COMPOSER_AGENT, "handles": sorted(set(refused))},
        )
    if restored:
        logger.warning(
            "report-composer omitted mandatory findings from the surface; restored",
            extra={"agent": COMPOSER_AGENT, "handles": sorted(restored)},
        )

    return replace(
        base,
        foreground=tuple(by_handle[h].consequence_id for h in order),
        reachable=tuple(sorted(reachable)),
        narration=narration,
        refused_handles=tuple(sorted(set(refused))),
        restored_handles=restored,
        narration_refusal=refusal,
        invented_ids=invented,
        calls_made=1,
    )


def _gate_narration(
    proposal: CompositionProposal, *, finalized_ids: frozenset[str]
) -> tuple[ComposedNarration | None, str | None, tuple[str, ...]]:
    """Keep the prose only if every id-shaped token in it was already finalized."""
    prose = [proposal.executive_summary or ""]
    prose.extend(note.note for note in proposal.section_notes)

    invented = sorted(
        {token for line in prose for token in _ID_TOKEN.findall(line) if token not in finalized_ids}
    )
    if invented:
        logger.warning(
            "report-composer narration cited ids absent from the finalized findings; refused",
            extra={"agent": COMPOSER_AGENT, "count": len(invented)},
        )
        return None, REFUSAL_INVENTED_ID, tuple(invented)

    if proposal.executive_summary is None and not proposal.section_notes:
        return None, None, ()

    return (
        ComposedNarration(
            executive_summary=proposal.executive_summary,
            section_notes=tuple((n.section, n.note) for n in proposal.section_notes),
            empty_sections=proposal.empty_sections,
        ),
        None,
        (),
    )


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def _build_prompt(
    *,
    system_prompt: str,
    candidates: Sequence[CompositionCandidate],
    eligible: Sequence[str],
    counts: dict[str, int],
    nonce: str,
) -> CompositionPrompt:
    rendered, redaction, breakout = wrap_untrusted(
        _render_candidates(candidates), nonce=nonce, label=FINDINGS_LABEL
    )
    task = "\n\n".join(
        (
            "## The finalized analysis\n\n```json\n"
            + json.dumps(counts, indent=2, sort_keys=True)
            + "\n```",
            rendered,
            _render_instructions(candidates, eligible),
        )
    )
    return CompositionPrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        redaction=redaction,
        attempted_delimiter_breakout=breakout,
        nonce=nonce,
    )


def _render_candidates(candidates: Sequence[CompositionCandidate]) -> str:
    lines: list[str] = []
    for candidate in candidates:
        confidence = "unscored" if candidate.confidence is None else f"{candidate.confidence:.3f}"
        lines.extend(
            (
                f"### `{candidate.handle}` - {candidate.consequence_class}",
                f"title: {candidate.title}",
                (
                    f"status: {candidate.status} | severity: {candidate.severity} | "
                    f"confidence: {confidence}"
                ),
                f"owner: {candidate.owner or '(none recorded)'}",
                f"historical analogue: {'yes' if candidate.has_historical_analogue else 'no'}",
                f"from the blind-spot partition: {'yes' if candidate.is_blind_spot else 'no'}",
                (f"must stay on the surface: {'yes' if candidate.mandatory else 'no'}"),
                "",
            )
        )
    return "\n".join(lines) or "(no findings were finalized for this change)"


def _render_instructions(
    candidates: Sequence[CompositionCandidate], eligible: Sequence[str]
) -> str:
    allowed = ", ".join(f"`{h}`" for h in eligible) or "(none)"
    return "\n".join(
        (
            "## Your task",
            "",
            "Decide what a reader sees first. Return the subset of the eligible handles that "
            "belongs on the default surface, and a reading order over it.",
            "",
            f"Eligible handles: {allowed}",
            "",
            "A handle marked 'must stay on the surface' will be foregrounded whatever you "
            "return. A handle you were not offered will be dropped. A handle you leave out of "
            "the reading order is appended, not removed.",
            "",
            "Contradictions and corpus-silence findings are usually the most immediately "
            f"useful output. There are {len(candidates)} findings in total; a surface of "
            "eighteen where five matter trains the reader to stop reading.",
            "",
            "You may also return an advisory `executive_summary` and section notes. They are "
            "carried beside the report, never inside it: the report's own summary is composed "
            "deterministically. Any identifier you mention that the analysis did not produce "
            "causes the whole narration to be discarded.",
            "",
            "Do **not** return a confidence number, a severity or a status. Carry nothing "
            "through that you were not given, and add no analysis - you have no retrieval.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )
