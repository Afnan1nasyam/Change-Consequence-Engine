"""Assemble the contradiction-hunter's task contract. The divergence set is closed.

Three things are shown: the change, the **already-classified** divergence set, and the
evidence spans behind it. The first two the system built and can vouch for. The third is
corpus text, and the corpus contains a document that tries to issue instructions
(``docs/EVALUATION.md``, prompt injection), so every span goes inside the same per-call nonce
envelope the extractor uses - reused from :mod:`app.assumptions.prompting` rather than
reimplemented, because a second envelope would drift from the first.

What this prompt does differently from the consequence-hunter's:

**It states the class of every item and forbids changing it.** Each item arrives labelled
with the checkpoint row that produced it. The schema has no field for a class, so a model
that tries to relabel one gets a rejected batch; saying so in the prompt keeps the ask and
the enforcement aligned and stops a cooperative model wasting a repair round.

**It says the divergence set is closed.** The agent is not searching. Every item it may
assess is in the list, and there is no field in which to report a new one - so the prompt's
job is to make that legible rather than to enforce it. A model told it may only rate what it
was given does not spend the call hunting.

**It scopes out the agent definition's inferred half.** ``.claude/agents/contradiction-hunter.md``
describes contradictions no key comparison can express - SOP versus observed behaviour,
conflicting definitions, exception clauses that quietly invalidate a broad claim. That half of
the contract has no runtime path yet and no schema, so the task contract names the boundary
explicitly. Leaving it unsaid would ask for output that is guaranteed to be rejected.

**Evidence is offered per item, not as one pool.** A citation only means something against
the divergence it is offered for: an evidence id from an unrelated key would be a real id
supporting an unrelated claim, which is a fabricated citation assembled out of genuine parts.

**Contextual evidence is admissible where the caller says it is.** The agent definition asks
whether an exception clause reconciles the two artifacts, and such a clause usually lives in
neither of them - so a span beyond the participating bindings has to be citable or that
question cannot be answered. It becomes citable by the caller supplying it and naming the
items it bears on, in :attr:`EvidenceForContradiction.item_ids`. The trust boundary is
unchanged: the *caller* performed any retrieval and decides admissibility, and the model still
cannot cite an id nobody offered for the item it cited it against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final

from app.assumptions.prompting import wrap_untrusted
from app.contradictions.divergence import DivergenceClass, DivergenceItem
from app.domain.entities.change import ChangeSpec
from app.domain.interfaces.llm import Message, Role
from app.ingestion.redaction import RedactionResult

#: Bumped when the assembled contract changes shape, because that changes what the model was
#: asked and therefore what its answer means.
CONTRADICTION_PROMPT_VERSION: Final = "contradiction-prompt-1"

#: Label on the untrusted envelope. Names what the content *is*, so a model that reads an
#: instruction inside it can recognise the instruction as data.
EVIDENCE_LABEL: Final = "divergence-evidence"

#: What each class means to a reader, rendered into the prompt. Stated so the agent rates
#: the divergence in front of it rather than the one it expected.
CLASS_GUIDANCE: Final[dict[DivergenceClass, str]] = {
    DivergenceClass.MATERIAL_CONTRADICTION: (
        "two peer declarations at the same scope, both in force, with divergent values. "
        "Rate who is harmed and which one governs."
    ),
    DivergenceClass.STALE_INVALIDATION: (
        "a consuming artifact holds a value that is not the authoritative one for its scope. "
        "The consumer is out of date; the declarer is not wrong."
    ),
    DivergenceClass.UNAUTHORIZED_OVERRIDE: (
        "a narrower scope overrides a broader one it does not outrank. A governance defect "
        "about who was permitted to decide - it stays one whatever its business impact."
    ),
    DivergenceClass.LEGITIMATE_SCOPED_OVERRIDE: (
        "a narrower scope overrides a broader one it does outrank. **Correct behaviour and "
        "not a finding.** Shown so your assessment covers the whole picture. Reporting one "
        "as a contradiction is the fastest way to make this system untrusted, because the "
        "reader can see it is correct."
    ),
    DivergenceClass.NON_MATERIAL_DIVERGENCE: (
        "two consumers disagreeing, one document disagreeing with itself, or incomparable "
        "units. Neither artifact is an authority, so neither can be said to be wrong. Needs "
        "scope review, and is not two authorities in conflict."
    ),
}


@dataclass(frozen=True)
class EvidenceForContradiction:
    """One span as the agent sees it, and which divergence it was offered for.

    A flattened projection rather than the stored evidence: the model needs an id it can
    cite and text it can read. ``item_ids`` records which divergences the span is admissible
    for, so the runtime can refuse a citation that is real but attached to the wrong claim.
    """

    evidence_id: str
    text: str
    source_title: str
    authority_class: str
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContradictionPrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    #: The handles the model may assess. The runtime verifies against exactly this.
    offered_item_ids: frozenset[str]
    #: Per item, the evidence ids admissible for it. Not one flat pool: see the module
    #: docstring on citations assembled out of genuine parts.
    offered_evidence_ids: dict[str, frozenset[str]]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = CONTRADICTION_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


def build_contradiction_prompt(
    *,
    system_prompt: str,
    change: ChangeSpec,
    divergences: tuple[DivergenceItem, ...],
    evidence: tuple[EvidenceForContradiction, ...],
    nonce: str,
) -> ContradictionPrompt:
    """Build the one message pair this agent is called with.

    ``nonce`` is supplied by the caller rather than derived here, so the runtime controls
    per-call uniqueness and a test can pin it.
    """
    rendered, redaction, breakout = wrap_untrusted(
        _render_evidence(evidence), nonce=nonce, label=EVIDENCE_LABEL
    )

    task = "\n\n".join(
        (
            _render_change(change),
            _render_divergences(divergences, evidence),
            rendered,
            _render_instructions(divergences),
        )
    )

    return ContradictionPrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        offered_item_ids=frozenset(item.item_id for item in divergences),
        offered_evidence_ids=_admissible_per_item(divergences, evidence),
        redaction=redaction,
        attempted_delimiter_breakout=breakout,
        nonce=nonce,
    )


def _admissible_per_item(
    divergences: tuple[DivergenceItem, ...],
    evidence: tuple[EvidenceForContradiction, ...],
) -> dict[str, frozenset[str]]:
    """Which evidence ids each item may be cited with.

    The union of two sources, and both are the caller's: the participating bindings' own
    spans, which the resolver named, and any contextual span the caller supplied and marked
    admissible for that item. Nothing here comes from the model.
    """
    admissible = {item.item_id: set(item.evidence_ids) for item in divergences}
    for span in evidence:
        for item_id in span.item_ids:
            if item_id in admissible:
                admissible[item_id].add(span.evidence_id)
    return {item_id: frozenset(ids) for item_id, ids in admissible.items()}


def _citable_for(
    item: DivergenceItem, evidence: tuple[EvidenceForContradiction, ...]
) -> tuple[str, ...]:
    """The ids listed under one item in the prompt. Same union the runtime enforces."""
    extra = {span.evidence_id for span in evidence if item.item_id in span.item_ids}
    return tuple(sorted(set(item.evidence_ids) | extra))


def _render_change(change: ChangeSpec) -> str:
    """The change under analysis, as structured data rather than prose."""
    payload = {
        "change_id": change.change_id,
        "subject_entities": list(change.subject_entities),
        "old_state": change.old_state,
        "new_state": change.new_state,
        "scope": change.scope.render(),
        "affected_domains": list(change.affected_domains),
    }
    return "## The proposed change\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_divergences(
    divergences: tuple[DivergenceItem, ...],
    evidence: tuple[EvidenceForContradiction, ...] = (),
) -> str:
    """The closed, already-classified set.

    ``relation_id`` and ``rule_id`` are deliberately **not** rendered. The agent has no
    legitimate use for a domain identifier it cannot write to, and omitting it means a model
    cannot echo one back as though it had addressed the relation itself.
    """
    if not divergences:
        return (
            "## The divergence set\n\n"
            "Empty. The deterministic resolver found no divergence for this change, so there "
            "is nothing to assess and no contradiction to report."
        )

    lines = [
        "## The divergence set (already detected, already classified)",
        "",
        "Each item below was found by exact comparison of assumption bindings and classified "
        "by the resolver. **The classification is authoritative and you cannot change it.** "
        "Assess materiality and temporal position only.",
        "",
    ]
    for item in divergences:
        lines.append(
            f"### `{item.item_id}` - {item.divergence_class.value}\n\n"
            f"- assumption key: `{item.assumption_key}`\n"
            f"- scope: `{item.scope}`\n"
            f"- as of: {item.as_of.isoformat()}\n"
            f"- what the resolver concluded: {CLASS_GUIDANCE[item.divergence_class]}\n"
            f"- detail: {item.detail}\n"
            f"- evidence you may cite for this item: "
            f"{', '.join(f'`{e}`' for e in _citable_for(item, evidence)) or '(none)'}"
        )
    return "\n\n".join(lines)


def _render_evidence(evidence: tuple[EvidenceForContradiction, ...]) -> str:
    if not evidence:
        return "(no evidence spans were supplied for this divergence set)"

    blocks = []
    for span in evidence:
        blocks.append(
            f"[evidence_id: {span.evidence_id}] "
            f"(source: {span.source_title}, authority: {span.authority_class}, "
            f"admissible for: {', '.join(span.item_ids)})\n"
            f"{span.text}"
        )
    return "\n\n".join(blocks)


def _render_instructions(divergences: tuple[DivergenceItem, ...]) -> str:
    """What to return, and the boundaries that will be enforced if crossed."""
    handles = ", ".join(f"`{item.item_id}`" for item in divergences) or "(none)"
    return "\n".join(
        (
            "## Your task",
            "",
            "For each divergence above, return an assessment of:",
            "",
            "1. **materiality** - `material`, `immaterial`, or `undetermined`. `material` "
            "must name who is affected and how. `immaterial` must give a reconciling "
            "exception from the corpus or a rationale: it argues against a divergence found "
            "by exact comparison, so it cannot be a bare dismissal.",
            "2. **temporal_position** - `authority_clear`, `superseded`, "
            "`intervals_overlap`, `ambiguous`, or `undetermined`. Report `ambiguous` rather "
            "than resolving it by picking the newest source.",
            "",
            f"You may assess **only** these items, verbatim: {handles}",
            "",
            "Cite only evidence ids listed as admissible for that specific item. A real id "
            "attached to the wrong divergence is a fabricated citation assembled out of "
            "genuine parts, and it will be dropped.",
            "",
            "You are **not searching**. The divergence set is closed and complete for this "
            "change; the schema has no field for a new contradiction. Contradictions that no "
            "key comparison can express - SOP versus observed behaviour, conflicting "
            "definitions, exception clauses that invalidate a broad claim - are real and are "
            "a later stage's work. Do not report them here; there is nowhere to put them.",
            "",
            "Do **not** return a confidence number, a severity, a status, a consequence "
            "class, or any restatement of an item's class, relationship or rule. The runtime "
            "computes or has already assigned all of them, and the schema will reject them.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )
