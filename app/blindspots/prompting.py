"""Assemble the blind-spot hunter's task contract. Partition in, existing claims for dedup.

The prompt renders the partition **grouped by slice** and names each slice, so the model can
work one evidence basis at a time and say which one yielded nothing. It also renders the
claims already in the analysis - as a do-not-restate list, not as material to critique. That
is the line the agent definition draws: shown the findings it would become a critic, denied
them it would be a second consequence hunter, and neither is the point.

Evidence is corpus text, so every span goes through the same per-call nonce envelope the
extractor and the main hunter use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final

from app.assumptions.prompting import wrap_untrusted
from app.blindspots.partition import SLICE_PRIORITY, EvidencePartition
from app.domain.entities.change import ChangeSpec
from app.domain.interfaces.llm import Message, Role
from app.ingestion.redaction import RedactionResult

BLIND_SPOT_PROMPT_VERSION: Final = "blind-spot-prompt-1"

PARTITION_LABEL: Final = "partition-evidence"

#: What each slice is, in the reader-facing wording the prompt uses. Kept here rather than in
#: the agent definition because it describes *this run's* partition, not the agent's role.
SLICE_DESCRIPTION: Final = {
    "single_holder": (
        "assumption keys bound by exactly one artifact - undocumented coupling, nothing "
        "corroborating and nothing depending on it in writing"
    ),
    "unwritten_rule": (
        "assumption keys declared only by code, configuration or informal notes, with no "
        "policy, standard or runbook stating them"
    ),
    "unqueried_source_type": (
        "source types the main retrieval returned nothing from for this change"
    ),
    "low_ranked": (
        "the corpus region the main query reached but ranked below the spans it handed to the "
        "consequence-hunter"
    ),
}


@dataclass(frozen=True)
class BlindSpotPrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    offered_evidence_ids: frozenset[str]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = BLIND_SPOT_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


def build_blind_spot_prompt(
    *,
    system_prompt: str,
    change: ChangeSpec,
    partition: EvidencePartition,
    existing_claims: tuple[str, ...],
    nonce: str,
) -> BlindSpotPrompt:
    """Build the one message pair this agent is called with."""
    rendered, redaction, breakout = wrap_untrusted(
        _render_partition(partition), nonce=nonce, label=PARTITION_LABEL
    )

    task = "\n\n".join(
        (
            _render_change(change),
            _render_coverage(partition),
            rendered,
            _render_existing(existing_claims),
            _render_instructions(partition),
        )
    )

    return BlindSpotPrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        offered_evidence_ids=partition.evidence_ids,
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
        "assumption_deltas": [
            {"key": delta.key, "delta_kind": delta.delta_kind.value}
            for delta in change.assumption_deltas
        ],
    }
    return "## The proposed change\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_coverage(partition: EvidencePartition) -> str:
    """Which slices this run's partition could fill, and which it could not."""
    lines = ["## Your evidence partition", ""]
    for slice_ in SLICE_PRIORITY:
        spans = partition.by_slice(slice_)
        state = f"{len(spans)} span(s)" if spans else "empty"
        lines.append(f"- `{slice_.value}` ({state}): {SLICE_DESCRIPTION[slice_.value]}")
    lines.extend(
        (
            "",
            "This partition deliberately excludes every span the consequence-hunter was "
            f"shown ({partition.excluded_as_already_seen} returned and dropped for that "
            "reason). An empty slice is a finding about the corpus, not a failure - name it "
            "in `unusable_slices`.",
        )
    )
    return "\n".join(lines)


def _render_partition(partition: EvidencePartition) -> str:
    if partition.is_empty:
        return (
            "(the partition is empty: every span retrievable for this change was already "
            "shown to the main analysis)"
        )

    blocks: list[str] = []
    for slice_ in SLICE_PRIORITY:
        spans = partition.by_slice(slice_)
        if not spans:
            continue
        blocks.append(f"### slice: {slice_.value}")
        for span in spans:
            keys = ", ".join(span.assumption_keys) or "(no registered key)"
            blocks.append(
                f"[evidence_id: {span.evidence_id}] "
                f"(source_type: {span.source_type}, authority: {span.authority_class}, "
                f"keys: {keys})\n{span.text}"
            )
    return "\n\n".join(blocks)


def _render_existing(claims: tuple[str, ...]) -> str:
    """The analysis so far, as a do-not-restate list.

    Claims only - no evidence, no status, no confidence. Enough to recognise a duplicate, not
    enough to review one: this agent is a partition, not a critic.
    """
    if not claims:
        return (
            "## Already reported\n\nNothing. The main analysis produced no findings for this "
            "change, which does not license a weaker standard of evidence here."
        )
    lines = ["## Already reported (do not restate)", ""]
    lines.extend(f"- {claim}" for claim in claims)
    lines.append("")
    lines.append(
        "If your partition leads you back to one of these, set `restates_existing` on that "
        "finding and say so. It will be dropped, which is the correct outcome."
    )
    return "\n".join(lines)


def _render_instructions(partition: EvidencePartition) -> str:
    ids = ", ".join(f"`{span.evidence_id}`" for span in partition.spans) or "(none)"
    return "\n".join(
        (
            "## Your task",
            "",
            "Work the partition above. Report **Class B** consequences the main analysis did "
            "not reach, and report separately where the partition shows the corpus silent on "
            "something this change touches.",
            "",
            "Keep the two apart. A question the corpus cannot answer goes in "
            "`corpus_silence`, never in `consequences`.",
            "",
            f"You may cite **only** these evidence ids, verbatim: {ids}",
            "",
            "An id you did not receive will be dropped, including any id the main analysis "
            "cited. Few findings plus a clear statement of what your partition could not "
            "cover is a good outcome.",
            "",
            "Do **not** return a confidence number, a severity, a status or a consequence "
            "class. The runtime computes or assigns all four, and the schema will reject them.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )
