"""Assemble one specialist's task contract over one specialist's slice.

Every span is corpus text, so it goes through the same per-call nonce envelope the extractor
and every other agentic stage uses - one implementation, from
:mod:`app.assumptions.prompting`, rather than a seventh copy that would drift from it.

The prompt tells the specialist three things it cannot work out for itself:

* **which ids it may cite**, verbatim. The runtime verifies against exactly this set, so the
  list is a courtesy to a cooperative model and never the enforcement.
* **that its slice is a slice.** Its definition already says so; repeating the source types it
  received keeps the agent from speculating about what it was not shown.
* **what is already known deterministically**, as a do-not-restate list. A specialist
  re-deriving a Class A finding produces a duplicate the suppression step then absorbs, and
  every absorbed duplicate is a call paid for and discarded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final

from app.assumptions.prompting import wrap_untrusted
from app.domain.entities.change import ChangeSpec
from app.domain.interfaces.llm import Message, Role
from app.ingestion.redaction import RedactionResult
from app.specialists.roster import SpecialistSlice

#: Bumped when the assembled contract changes shape, because that changes what was asked.
SPECIALIST_PROMPT_VERSION: Final = "specialist-prompt-1"

#: Label on the untrusted envelope. Names what the content *is*, so a model that reads an
#: instruction inside it can recognise the instruction as data.
SLICE_LABEL: Final = "evidence-slice"

SPAN_CHARS: Final = 700


@dataclass(frozen=True)
class SpecialistPrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    #: The ids this specialist is permitted to cite. The runtime verifies against exactly this.
    offered_evidence_ids: frozenset[str]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = SPECIALIST_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


def build_specialist_prompt(
    *,
    system_prompt: str,
    change: ChangeSpec,
    slice_: SpecialistSlice,
    known_claims: tuple[str, ...] = (),
    nonce: str,
) -> SpecialistPrompt:
    """Build the one message pair this specialist is called with."""
    rendered, redaction, breakout = wrap_untrusted(
        _render_slice(slice_), nonce=nonce, label=SLICE_LABEL
    )
    task = "\n\n".join(
        (
            _render_change(change),
            _render_basis(slice_),
            rendered,
            _render_known(known_claims),
            _render_instructions(slice_),
        )
    )
    return SpecialistPrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        offered_evidence_ids=slice_.evidence_ids,
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
        "scope": change.scope.render(),
        "affected_domains": list(change.affected_domains),
        "constraints": list(change.constraints),
        "assumption_deltas": [
            {
                "key": delta.key,
                "delta_kind": delta.delta_kind.value,
                "old_value": delta.old_value.render() if delta.old_value else None,
                "new_value": delta.new_value.render() if delta.new_value else None,
            }
            for delta in change.assumption_deltas
        ],
    }
    return "## The proposed change\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_basis(slice_: SpecialistSlice) -> str:
    """What this slice is and is not, so the agent does not reason about what it cannot see."""
    types = ", ".join(sorted(t.value for t in slice_.specialist.source_types))
    lines = [
        "## Your evidence slice",
        "",
        f"Focus for this analysis: {slice_.specialist.focus}.",
        "",
        f"Retrieved from source types: {types}. "
        f"{len(slice_.spans)} span(s) offered"
        + (f", {slice_.truncated} beyond the budget and dropped." if slice_.truncated else "."),
        "",
        "This is **your** slice, not the shared bundle. Other specialists read other source "
        "types. Reason from what is here; where it is insufficient, ask an unresolved "
        "question rather than inferring past the evidence.",
    ]
    return "\n".join(lines)


def _render_slice(slice_: SpecialistSlice) -> str:
    if slice_.is_empty:  # pragma: no cover - an empty slice is never invoked
        return "(your slice is empty)"
    blocks = []
    for span in slice_.spans:
        keys = ", ".join(span.assumption_keys) or "(no registered key)"
        blocks.append(
            f"[evidence_id: {span.evidence_id}] "
            f"(source: {span.source_title}, source_type: {span.source_type}, "
            f"authority: {span.authority_class}, keys: {keys})\n{span.text[:SPAN_CHARS]}"
        )
    return "\n\n".join(blocks)


def _render_known(known_claims: tuple[str, ...]) -> str:
    """Deterministic findings, as a do-not-restate list."""
    if not known_claims:
        return (
            "## Already found deterministically\n\n"
            "None. The resolver derived no assumption-mediated consequence for this change, "
            "which does not mean there is nothing to observe."
        )
    lines = [
        "## Already found deterministically (do not restate)",
        "",
        "These are evidence-backed and already in the report. Restating one produces a "
        "duplicate that will be suppressed.",
        "",
    ]
    lines.extend(f"- {claim}" for claim in known_claims)
    return "\n".join(lines)


def _render_instructions(slice_: SpecialistSlice) -> str:
    ids = ", ".join(f"`{span.evidence_id}`" for span in slice_.spans) or "(none)"
    return "\n".join(
        (
            "## Your task",
            "",
            "Report what your slice shows about this change, in your domain. Each observation "
            "is a statement about the evidence - not a consequence, not a severity, not a "
            "status.",
            "",
            f"You may cite **only** these evidence ids, verbatim: {ids}",
            "",
            "An id you did not receive will be dropped, and an observation left with no "
            "admitted citation cannot be reported as supported.",
            "",
            "Set `basis` to `direct_evidence` when a cited span says it, and `inference` when "
            "you reasoned to it from cited spans. An inference is capped below a directly "
            "evidenced observation by the runtime, so labelling one as the other gains "
            "nothing.",
            "",
            "Where a span commits to a proposition - a threshold, a duration, a window, a "
            "rate, an ownership claim - report it in `observed_bindings` with its key, its "
            "rendered value and the evidence id. These are recorded for registry review and "
            "are **not** admitted into the coupling index by this call.",
            "",
            "Do **not** return a confidence number, a severity, a status or a consequence "
            "class. The runtime computes or assigns all four, and the schema will reject them.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )
