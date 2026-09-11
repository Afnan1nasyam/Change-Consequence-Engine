"""Assemble the consequence judge's task contract. One dossier per candidate.

The judge sees more than any other stage: the derivation or the hypothesis, the challenger's
verdict on it, the divergences touching its assumption key, the known holes in the corpus, and
the questions nobody could answer. All of it the system built, except the evidence spans - and
those go inside the same per-call nonce envelope the extractor uses, reused from
:mod:`app.assumptions.prompting`, because the corpus contains a document that tries to issue
instructions.

What this prompt does that the earlier agentic contracts do not:

**It states the status gates and says they are applied in code afterwards.** Not to excuse a
sloppy proposal - to stop the model spending effort re-deriving a rule it will not be trusted
on. A model told "no evidence caps at speculative, and we enforce it" proposes honestly and
spends its reasoning on severity instead.

**It shows the challenger's verdict and says what it does and does not mean.** A
``no_refutation_found`` in front of a model is an invitation to read it as corroboration. It
is not: nobody finding contrary evidence is strictly weaker than evidence of correctness
(checkpoint 9.2), and the prompt says so against the candidate rather than in a footnote.

**It shows the disagreements the runtime has already collected.** They travel into the
arbitration whether or not the model repeats them, so showing them is honest about what the
reader will see - and it stops a model "resolving" a disagreement it thinks it is expected to
tidy away.

**It asks for severity and forbids confidence in the same breath.** The asymmetry is the one
thing about this stage most likely to be got wrong in the direction of a plausible number.

Domain identifiers are deliberately absent. The judge addresses a candidate by handle, so its
output cannot name a ``consequence_id``, a ``relation_id`` or a ``challenge_id`` - which is
the first step to reattributing one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final

from app.assumptions.prompting import wrap_untrusted
from app.domain.entities.change import ChangeSpec
from app.domain.interfaces.llm import Message, Role
from app.domain.value_objects.enums import ChallengeOutcome, Severity, Status
from app.ingestion.redaction import RedactionResult
from app.judgment.dossier import CandidateDossier

#: Bumped when the assembled contract changes shape, because that changes what the model was
#: asked and therefore what its answer means.
JUDGMENT_PROMPT_VERSION: Final = "judgment-prompt-1"

#: Label on the untrusted envelope. Names what the content *is*, so a model that reads an
#: instruction inside it can recognise the instruction as data.
EVIDENCE_LABEL: Final = "candidate-evidence"

#: What each challenge outcome means for a disposition, stated against the candidate.
OUTCOME_GUIDANCE: Final[dict[ChallengeOutcome, str]] = {
    ChallengeOutcome.REFUTED: (
        "the challenger cited evidence that contradicts this claim. The runtime will record "
        "`rejected`; rate the severity it would have had and say what the refutation showed."
    ),
    ChallengeOutcome.WEAKENED: (
        "the challenger cited a control, exception, or newer version that narrows this claim. "
        "It cannot exceed `disputed`."
    ),
    ChallengeOutcome.NO_REFUTATION_FOUND: (
        "the challenger looked and found nothing against this claim. **This is not "
        "corroboration** - it is strictly weaker than evidence of correctness, and it must "
        "not raise the status by itself."
    ),
}


@dataclass(frozen=True)
class EvidenceForJudgment:
    """One span as the judge sees it, and which candidates it may be cited for.

    A flattened projection rather than the stored evidence: the model needs an id it can cite
    and text it can read. ``handles`` records which candidates the span is admissible for, so
    the runtime can refuse a citation that is real but attached to the wrong claim.
    """

    evidence_id: str
    text: str
    source_title: str
    authority_class: str
    handles: tuple[str, ...]


@dataclass(frozen=True)
class JudgmentPrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    #: The handles the model may disposition. The runtime verifies against exactly this.
    offered_handles: frozenset[str]
    #: Per candidate, the evidence ids citable for it - supporting or contradicting.
    offered_evidence_ids: dict[str, frozenset[str]]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = JUDGMENT_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


def build_judgment_prompt(
    *,
    system_prompt: str,
    change: ChangeSpec,
    dossiers: tuple[CandidateDossier, ...],
    evidence: tuple[EvidenceForJudgment, ...],
    nonce: str,
) -> JudgmentPrompt:
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
            _render_dossiers(dossiers, evidence),
            rendered,
            _render_instructions(dossiers),
        )
    )

    return JudgmentPrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        offered_handles=frozenset(d.handle for d in dossiers),
        offered_evidence_ids=_admissible_per_handle(dossiers, evidence),
        redaction=redaction,
        attempted_delimiter_breakout=breakout,
        nonce=nonce,
    )


def _admissible_per_handle(
    dossiers: tuple[CandidateDossier, ...],
    evidence: tuple[EvidenceForJudgment, ...],
) -> dict[str, frozenset[str]]:
    """Which evidence ids each candidate may be cited with.

    The union of the candidate's own citable set - its support plus whatever the challenger
    cited against it - and any span the caller supplied and marked admissible for it. All
    three sources are the caller's or an upstream stage's; none comes from the judge.
    """
    admissible = {d.handle: set(d.citable_evidence_ids) for d in dossiers}
    for span in evidence:
        for handle in span.handles:
            if handle in admissible:
                admissible[handle].add(span.evidence_id)
    return {handle: frozenset(ids) for handle, ids in admissible.items()}


def _citable_for(
    dossier: CandidateDossier, evidence: tuple[EvidenceForJudgment, ...]
) -> tuple[str, ...]:
    """The ids listed under one candidate in the prompt. Same union the runtime enforces."""
    extra = {span.evidence_id for span in evidence if dossier.handle in span.handles}
    return tuple(sorted(dossier.citable_evidence_ids | extra))


def _render_change(change: ChangeSpec) -> str:
    payload = {
        "change_id": change.change_id,
        "subject_entities": list(change.subject_entities),
        "old_state": change.old_state,
        "new_state": change.new_state,
        "scope": change.scope.render(),
        "affected_domains": list(change.affected_domains),
    }
    return "## The proposed change\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_dossiers(
    dossiers: tuple[CandidateDossier, ...],
    evidence: tuple[EvidenceForJudgment, ...] = (),
) -> str:
    """One block per candidate: the claim, its class, its review, and its disagreements."""
    if not dossiers:
        return (
            "## The candidates\n\n"
            "Empty. This analysis produced no candidate consequences, so there is nothing to "
            "arbitrate."
        )

    blocks = ["## The candidates", ""]
    for dossier in dossiers:
        citable = ", ".join(f"`{e}`" for e in _citable_for(dossier, evidence)) or "(none)"
        lines = [
            f"### `{dossier.handle}` - {dossier.consequence_class.value}",
            "",
            f"- claim: {dossier.claim}",
            f"- mechanism: {dossier.mechanism}",
            f"- affected: {dossier.affected_entity}",
        ]
        if dossier.assumption_keys:
            lines.append(
                f"- assumption keys: {', '.join(f'`{k}`' for k in dossier.assumption_keys)}"
            )
        if dossier.is_class_a:
            lines.append(
                "- **produced by the deterministic resolver.** You do not assign its class, "
                "and you cannot downgrade it on mechanism grounds - a derivation is only ever "
                "rule-wrong or input-wrong, and the challenger has already had the only "
                "admissible say."
            )
        else:
            analogue = (
                "present" if dossier.historical_analogue_id else "**absent - caps at speculative**"
            )
            lines.append(f"- historical analogue: {analogue}")
        lines.append(f"- evidence you may cite for this candidate: {citable}")
        lines.extend(_render_review(dossier))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _render_review(dossier: CandidateDossier) -> list[str]:
    """The challenger's verdict and the collected disagreements, per candidate."""
    lines: list[str] = []
    for challenge in dossier.challenges:
        lines.append(
            f"- challenge outcome `{challenge.outcome.value}`: "
            f"{OUTCOME_GUIDANCE[challenge.outcome]}"
        )
    if dossier.upstream_disagreements:
        lines.append("- disagreements already recorded (they reach the reader regardless):")
        lines.extend(f"  - {item}" for item in dossier.upstream_disagreements)
    if dossier.evidence_gap_ids:
        lines.append(
            f"- {len(dossier.evidence_gap_ids)} known gap(s) in the corpus for this analysis. "
            "A gap is a statement about the corpus, not a consequence - do not disposition one."
        )
    return lines


def _render_evidence(evidence: tuple[EvidenceForJudgment, ...]) -> str:
    if not evidence:
        return "(no evidence spans were supplied for these candidates)"

    blocks = []
    for span in evidence:
        blocks.append(
            f"[evidence_id: {span.evidence_id}] "
            f"(source: {span.source_title}, authority: {span.authority_class}, "
            f"admissible for: {', '.join(span.handles)})\n"
            f"{span.text}"
        )
    return "\n\n".join(blocks)


def _render_instructions(dossiers: tuple[CandidateDossier, ...]) -> str:
    """What to return, what is enforced afterwards, and what will be rejected."""
    handles = ", ".join(f"`{d.handle}`" for d in dossiers) or "(none)"
    statuses = ", ".join(f"`{s.value}`" for s in Status)
    severities = ", ".join(f"`{s.value}`" for s in Severity)
    return "\n".join(
        (
            "## Your task",
            "",
            f"Return one disposition per candidate, for exactly these handles: {handles}",
            "",
            f"- **status**: one of {statuses}",
            f"- **severity**: one of {severities}",
            "- **rationale**: why, concisely",
            "- **supporting** and **contradicting** evidence ids, kept apart",
            "- **preserved_disagreements**: material disagreement, in your words",
            "- **validation_actions**: what source or check would settle an unresolved finding",
            "- **in_default_view**: whether a reviewer should see this first",
            "",
            "### Severity is yours. Confidence is not.",
            "",
            "Confidence is computed by a deterministic versioned function over measurable "
            "evidence properties, and the schema has no field for it. Severity is a judgment "
            "about business impact the corpus does not contain, so it is yours. **Never blend "
            "them**: HIGH severity with LOW confidence is a legitimate and important output, "
            "and nothing you write moves one because of the other.",
            "",
            "### These gates are applied in code after you answer",
            "",
            "Propose honestly rather than pre-applying them, and spend your reasoning on "
            "severity instead:",
            "",
            "1. No admitted supporting evidence caps the status - `supported` and `likely` "
            "are unreachable. Absence of contrary evidence is not support.",
            "2. A Class B candidate with no historical analogue caps at `speculative`.",
            "3. A refuted candidate becomes `rejected`; a weakened one cannot exceed `disputed`.",
            "4. `no_refutation_found` raises nothing.",
            "",
            "### Other rules that override the urge to produce a clean answer",
            "",
            "- **Corroboration counts sources, not agents.** Several specialists reading one "
            "span are one source.",
            "- Material disagreement is **preserved**, never resolved by averaging. Anything "
            "already recorded above reaches the reader whether you repeat it or not.",
            "- An `unauthorized_override` is a deterministic governance finding. Rate its "
            "severity; do not re-decide whether it occurred.",
            "- An EvidenceGap is a statement about the **corpus**. It is never a consequence "
            "and there is no handle for one.",
            "- If the evidence is inadequate, `unresolved` is the correct answer. Fail closed.",
            "",
            "Cite only evidence ids listed as admissible for that specific candidate. A real "
            "id attached to the wrong candidate is a fabricated citation assembled out of "
            "genuine parts, and it will be dropped.",
            "",
            "Do **not** return a confidence number, a consequence class, an origin, or any "
            "identifier. The runtime computes or has already assigned all of them, and the "
            "schema will reject them.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )
