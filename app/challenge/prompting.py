"""Assemble the adversarial challenger's task contract. Two candidate kinds, two rulebooks.

The challenger is shown the finalized candidates, each with its own closed evidence set, and
the corpus spans behind them. The candidates the system built and can vouch for. The spans are
corpus text, and the corpus contains a document that tries to issue instructions
(``docs/EVALUATION.md``, prompt injection), so every span goes inside the same per-call nonce
envelope the extractor uses - reused from :mod:`app.assumptions.prompting` rather than
reimplemented, because a second envelope would drift from the first.

What this prompt does that the other two agentic contracts do not:

**It states a different rulebook per candidate.** A Class B hypothesis may be challenged on
anything, including its mechanism. A Class A finding may be challenged only on the four
:class:`~app.domain.value_objects.enums.ClassAGround` members, and the prompt lists them
against that candidate with the inadmissible case named explicitly. Mixing the two rulebooks
into one paragraph is how a model ends up applying the looser one.

**It does not show a Class A candidate's mechanism.** The mechanism is precisely what may not
be challenged on a derivation, and putting it in front of the model would invite the one
objection the runtime must reject. What is shown instead is the *inputs*: the assumption key,
the scope, the effective date and the evidence - the things a challenge may actually dispute.

**It prices every outcome.** ``refuted`` and ``weakened`` cost a citation;
``no_refutation_found`` costs a falsification test. Saying so keeps the ask and the
enforcement aligned, so a cooperative model does not spend a repair round on an objection the
schema will reject.

**It says that finding nothing is a good result.** Not politeness - calibration. A challenger
that returns objections at the rate it is asked for them is worse than useless, and the honest
framing of ``no_refutation_found`` is the cheapest defence against that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final

from app.assumptions.prompting import wrap_untrusted
from app.challenge.candidates import ChallengeTarget
from app.domain.entities.change import ChangeSpec
from app.domain.interfaces.llm import Message, Role
from app.domain.value_objects.enums import ClassAGround
from app.ingestion.redaction import RedactionResult

#: Bumped when the assembled contract changes shape, because that changes what the model was
#: asked and therefore what its answer means.
CHALLENGE_PROMPT_VERSION: Final = "challenge-prompt-1"

#: Label on the untrusted envelope. Names what the content *is*, so a model that reads an
#: instruction inside it can recognise the instruction as data.
EVIDENCE_LABEL: Final = "candidate-evidence"

#: Each admissible ground, in the words a challenger needs to use it. Rendered against every
#: Class A candidate, because a ground named without its meaning gets chosen by vibe.
GROUND_GUIDANCE: Final[dict[ClassAGround, str]] = {
    ClassAGround.DISPUTED_SPAN: (
        "the cited span does not say what the binding claims - a misread value, a "
        "conditional read as absolute, a heading read as a rule"
    ),
    ClassAGround.DISPUTED_KEY_OR_SCOPE: (
        "the binding was filed under the wrong assumption key, or at the wrong scope, so the "
        "two artifacts are not in fact talking about the same variable"
    ),
    ClassAGround.DISPUTED_TEMPORAL_VALIDITY: (
        "the binding was not in force at the effective date, so the divergence is historical "
        "rather than live"
    ),
    ClassAGround.SUPERSEDING_VERSION: (
        "a newer version of the source changes the binding, so the derivation ran on a stale input"
    ),
}


@dataclass(frozen=True)
class EvidenceForChallenge:
    """One span as the challenger sees it, and which candidates it may be cited against.

    A flattened projection rather than the stored evidence: the model needs an id it can cite
    and text it can read. ``target_ids`` records which candidates the span is admissible for,
    so the runtime can refuse a citation that is real but attached to the wrong claim.
    """

    evidence_id: str
    text: str
    source_title: str
    authority_class: str
    target_ids: tuple[str, ...]
    #: Set when this span comes from a version newer than the one the candidate used. The
    #: challenger's single most valuable finding, so it is flagged rather than left to be
    #: noticed.
    supersedes: str | None = None


@dataclass(frozen=True)
class ChallengePrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    #: The handles the model may return a verdict for. The runtime verifies against this.
    offered_target_ids: frozenset[str]
    #: Per candidate, the evidence ids admissible against it. Not one flat pool: a real id
    #: attached to the wrong candidate would support an unrelated claim.
    offered_evidence_ids: dict[str, frozenset[str]]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = CHALLENGE_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


def build_challenge_prompt(
    *,
    system_prompt: str,
    change: ChangeSpec,
    targets: tuple[ChallengeTarget, ...],
    evidence: tuple[EvidenceForChallenge, ...],
    nonce: str,
) -> ChallengePrompt:
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
            _render_targets(targets, evidence),
            rendered,
            _render_instructions(targets),
        )
    )

    return ChallengePrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        offered_target_ids=frozenset(target.target_id for target in targets),
        offered_evidence_ids=_admissible_per_target(targets, evidence),
        redaction=redaction,
        attempted_delimiter_breakout=breakout,
        nonce=nonce,
    )


def _admissible_per_target(
    targets: tuple[ChallengeTarget, ...],
    evidence: tuple[EvidenceForChallenge, ...],
) -> dict[str, frozenset[str]]:
    """Which evidence ids each candidate may be challenged with.

    The union of two caller-owned sources: the candidate's own evidence set, and any span the
    caller retrieved and marked admissible against it. Nothing here comes from the model.
    """
    admissible = {target.target_id: set(target.evidence_ids) for target in targets}
    for span in evidence:
        for target_id in span.target_ids:
            if target_id in admissible:
                admissible[target_id].add(span.evidence_id)
    return {target_id: frozenset(ids) for target_id, ids in admissible.items()}


def _citable_for(
    target: ChallengeTarget, evidence: tuple[EvidenceForChallenge, ...]
) -> tuple[str, ...]:
    """The ids listed under one candidate in the prompt. Same union the runtime enforces."""
    extra = {span.evidence_id for span in evidence if target.target_id in span.target_ids}
    return tuple(sorted(set(target.evidence_ids) | extra))


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


def _render_targets(
    targets: tuple[ChallengeTarget, ...],
    evidence: tuple[EvidenceForChallenge, ...] = (),
) -> str:
    """The candidates, each under its own rulebook.

    Domain identifiers - ``finding_id``, ``relation_id``, ``rule_id``, ``hypothesis_id`` - are
    deliberately **not** rendered. The model has no legitimate use for an identifier it cannot
    write to, and omitting them means its output cannot name a Class A record at all.
    """
    if not targets:
        return (
            "## The candidates\n\n"
            "Empty. There are no finalized candidate findings for this change, so there is "
            "nothing to challenge."
        )

    blocks = ["## The candidates", ""]
    for target in targets:
        citable = ", ".join(f"`{e}`" for e in _citable_for(target, evidence)) or "(none)"
        lines = [
            f"### `{target.target_id}` - {target.consequence_class.value}",
            "",
            f"- claim: {target.claim}",
        ]
        if target.assumption_key:
            lines.append(f"- assumption key: `{target.assumption_key}`")
        if target.scope:
            lines.append(f"- scope: `{target.scope}`")
        if target.as_of:
            lines.append(f"- effective date: {target.as_of.isoformat()}")
        if target.mechanism:
            lines.append(f"- mechanism (challengeable): {target.mechanism}")
        lines.append(f"- evidence you may cite against this candidate: {citable}")
        lines.append("")
        lines.append(_rulebook(target))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _rulebook(target: ChallengeTarget) -> str:
    """The rules for this candidate, stated against this candidate.

    Per target rather than once at the top, because a reader - human or model - applies the
    rule they are looking at, and the looser rulebook is the one that gets applied when the
    two are adjacent in one paragraph.
    """
    if not target.requires_class_a_ground:
        return (
            "**This is a Class B hypothesis: a model proposed it.** Everything is fair - the "
            "mechanism, the analogue, the magnitude, whether the cited evidence supports the "
            "claim made or only a weaker nearby claim. A citation is still required for "
            "`refuted` and `weakened`."
        )

    grounds = "\n".join(
        f"  - `{ground.value}` - {GROUND_GUIDANCE[ground]}" for ground in ClassAGround
    )
    return (
        "**This is a Class A finding: the resolver derived it deterministically.** A "
        "derivation is only ever rule-wrong or input-wrong, so to refute or weaken it you "
        "must name exactly one of these grounds:\n\n"
        f"{grounds}\n\n"
        "  Disagreeing with the *conclusion* is **inadmissible** here and will be rejected "
        "by the runtime, not by this instruction. If the rule ran correctly on correct "
        "inputs, there is nothing for you to cite and the answer is `no_refutation_found`. "
        "The same applies to an authority comparison: you may dispute the inputs, you may "
        "not dispute the arithmetic."
    )


def _render_evidence(evidence: tuple[EvidenceForChallenge, ...]) -> str:
    if not evidence:
        return "(no evidence spans were supplied for these candidates)"

    blocks = []
    for span in evidence:
        marker = f", SUPERSEDES: {span.supersedes}" if span.supersedes else ""
        blocks.append(
            f"[evidence_id: {span.evidence_id}] "
            f"(source: {span.source_title}, authority: {span.authority_class}, "
            f"admissible against: {', '.join(span.target_ids)}{marker})\n"
            f"{span.text}"
        )
    return "\n\n".join(blocks)


def _render_instructions(targets: tuple[ChallengeTarget, ...]) -> str:
    """What to return, and what each outcome costs."""
    handles = ", ".join(f"`{t.target_id}`" for t in targets) or "(none)"
    return "\n".join(
        (
            "## Your task",
            "",
            "Assume each candidate above may be wrong, and try to falsify it. Return exactly "
            "one verdict per candidate:",
            "",
            "1. `refuted` - at least one cited evidence id whose content actually "
            "contradicts the claim, plus what specifically it contradicts.",
            "2. `weakened` - at least one cited evidence id for a compensating control, an "
            "exception clause, or a newer source version, plus how the scope narrows.",
            "3. `no_refutation_found` - **no** evidence ids, plus the observation that would "
            "falsify the claim.",
            "",
            "There is no fourth option. An objection with no citation is not representable "
            "in the schema and will be discarded, so do not spend effort on arguments you "
            "cannot cite.",
            "",
            f"You may return a verdict **only** for these candidates, verbatim: {handles}",
            "",
            "Cite only evidence ids listed as admissible against that specific candidate. A "
            "real id attached to the wrong candidate is a fabricated citation assembled out "
            "of genuine parts, and it will be dropped.",
            "",
            "For a Class A candidate, `refuted` and `weakened` must also name one of the "
            "four admissible grounds shown against it. A verdict with no ground is "
            "inadmissible and will be refused. For a Class B candidate, do not name a "
            "ground - there is no derivation to dispute the inputs of.",
            "",
            "**`no_refutation_found` on a well-evidenced candidate is the correct answer and "
            "a useful result.** Your value is measured by how many false positives you "
            "remove, not by how many objections you raise. An invented objection costs the "
            "reader more than a missed one.",
            "",
            "Do **not** return a confidence number, a severity, a status, a consequence "
            "class, an origin, or any identifier. The runtime computes or assigns all of "
            "them, and the schema will reject them.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )
