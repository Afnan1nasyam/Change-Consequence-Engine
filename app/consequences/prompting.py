"""Assemble the consequence-hunter's task contract. Evidence is untrusted content.

The hunter is shown three things: the ChangeSpec, the retrieved evidence, and the Class A
findings the resolver already derived. Two of those are trustworthy because the system built
them. The third - the evidence - is **corpus text**, and the corpus contains a document that
tries to issue instructions (``docs/EVALUATION.md``, prompt injection). So every span is
wrapped in the same per-call nonce envelope the extractor uses, from
:mod:`app.assumptions.prompting`, rather than a second implementation that would drift from
it.

Two things the prompt does that are easy to get wrong:

**It lists the evidence ids the model may cite, and says they are the only ones.** A
fabricated citation is the most damaging output this system can produce, and the cheapest
defence is to make the legal set explicit. The runtime still verifies - the prompt is a
courtesy to a cooperative model, never the enforcement.

**It shows the Class A findings and tells the model they are already found.** Not to be
polite: a hunter that re-derives them produces duplicates that the deterministic suppression
step then has to absorb, and every absorbed duplicate is a call paid for and discarded.
Showing them is cheaper than filtering them, and the filter runs anyway.

The prompt asks for **no confidence, no severity and no class**. It cannot ask for them
usefully in any case - the schema has nowhere to put them - but saying so keeps the ask and
the enforcement aligned, which is what stops a model wasting a repair round on a field that
will be rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final

from app.assumptions.prompting import wrap_untrusted
from app.domain.entities.change import ChangeSpec
from app.domain.entities.class_a import ClassAFinding
from app.domain.interfaces.llm import Message, Role
from app.ingestion.redaction import RedactionResult

#: Bumped when the assembled contract changes shape, because that changes what the model was
#: asked and therefore what its answer means.
HUNTER_PROMPT_VERSION: Final = "class-b-prompt-1"

#: Label on the untrusted envelope. Names what the content *is*, so a model that reads an
#: instruction inside it can recognise the instruction as data.
EVIDENCE_LABEL: Final = "retrieved-evidence"


@dataclass(frozen=True)
class EvidenceForHunting:
    """One span as the hunter sees it: its id, its text, and where it came from.

    A flattened projection rather than the ``EvidenceCandidate`` itself, deliberately: the
    model needs an id it can cite and text it can read, and handing it the full candidate
    would put document metadata into the prompt that the reader of a citation cannot see in
    the quoted span.
    """

    evidence_id: str
    text: str
    source_type: str
    authority_class: str


@dataclass(frozen=True)
class SpecialistObservation:
    """One specialist's observation, as the hunter sees it.

    A flat projection rather than the :class:`~app.domain.entities.findings.SpecialistFinding`
    itself, for the reason :class:`EvidenceForHunting` is one: this layer may not import
    ``app.specialists`` (``tests/contract/test_layering.py``), and the hunter needs a claim it
    can read, the ids behind it and who said it - not a domain entity.

    ``evidence_position`` is carried through unchanged. The hunter must be able to tell a
    directly evidenced observation from an inference, and it is not this layer's business to
    re-decide which it was.
    """

    agent: str
    category: str
    claim: str
    evidence_position: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissibleAnalogue:
    """One historical anchor the hunter may cite, as it sees it.

    A flat projection for the reason :class:`EvidenceForHunting` is one: this layer may not
    import ``app.specialists``. The ``event_id`` was *derived* from spans by the runtime, never
    proposed by a model, and the runtime checks any id the hunter claims against exactly this
    set - so a well-formed invention is dropped rather than lifting a status.
    """

    event_id: str
    title: str
    observed_outcome: str
    differs_how: str = ""


@dataclass(frozen=True)
class HunterPrompt:
    """The assembled contract, plus what assembling it observed."""

    messages: tuple[Message, ...]
    #: The ids the model is permitted to cite. The runtime verifies against exactly this.
    offered_evidence_ids: frozenset[str]
    redaction: RedactionResult
    attempted_delimiter_breakout: bool
    prompt_version: str = HUNTER_PROMPT_VERSION
    nonce: str = field(default="", repr=False)


def build_hunter_prompt(
    *,
    system_prompt: str,
    change: ChangeSpec,
    evidence: tuple[EvidenceForHunting, ...],
    class_a: tuple[ClassAFinding, ...],
    nonce: str,
    specialists: tuple[SpecialistObservation, ...] = (),
    analogues: tuple[AdmissibleAnalogue, ...] = (),
) -> HunterPrompt:
    """Build the one message pair this agent is called with.

    ``nonce`` is supplied by the caller rather than derived here, so the runtime controls
    per-call uniqueness and a test can pin it.

    ``specialists`` is the domain fan-out's output. Default-empty, so the hunter still runs
    alone - and when it is supplied the hunter is doing what its own definition describes,
    "combining independent evidence from multiple domains", over evidence a domain analyst has
    actually read rather than over one undifferentiated bundle.
    """
    rendered, redaction, breakout = wrap_untrusted(
        _render_evidence(evidence), nonce=nonce, label=EVIDENCE_LABEL
    )

    task = "\n\n".join(
        (
            _render_change(change),
            _render_class_a(class_a),
            _render_specialists(specialists),
            _render_analogues(analogues),
            rendered,
            _render_instructions(evidence),
        )
    )

    return HunterPrompt(
        messages=(
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ),
        offered_evidence_ids=frozenset(span.evidence_id for span in evidence),
        redaction=redaction,
        attempted_delimiter_breakout=breakout,
        nonce=nonce,
    )


def _render_change(change: ChangeSpec) -> str:
    """The change under analysis, as structured data rather than prose."""
    payload = {
        "change_id": change.change_id,
        "subject_entities": list(change.subject_entities),
        "old_state": change.old_state,
        "new_state": change.new_state,
        "scope": change.scope.render(),
        "constraints": list(change.constraints),
        "affected_domains": list(change.affected_domains),
        "analyst_caveats": list(change.analyst_caveats),
        "ambiguities": list(change.ambiguities),
        "assumption_deltas": [
            {
                "key": delta.key,
                "delta_kind": delta.delta_kind.value,
                "scope": delta.scope.render(),
                "old_value": delta.old_value.render() if delta.old_value else None,
                "new_value": delta.new_value.render() if delta.new_value else None,
            }
            for delta in change.assumption_deltas
        ],
    }
    return "## The proposed change\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_class_a(findings: tuple[ClassAFinding, ...]) -> str:
    """The consequences already derived deterministically.

    Shown so the hunter does not spend the call re-deriving them. The deterministic
    suppression step runs regardless of whether the model cooperates.
    """
    if not findings:
        return (
            "## Already found deterministically\n\n"
            "None. The resolver derived no Class A consequences for this change, which does "
            "not mean there are none to find - it means no *assumption-mediated* one exists."
        )

    lines = [
        "## Already found deterministically (Class A - do not re-derive)",
        "",
        "These are evidence-backed and already in the report. Proposing them again produces "
        "a duplicate that will be suppressed.",
        "",
    ]
    for finding in findings:
        lines.append(
            f"- `{finding.rule_id}` on `{finding.assumption_key}` @ {finding.scope.render()} "
            f"-> {finding.affected_artifact}: {finding.summary}"
        )
    return "\n".join(lines)


def _render_specialists(observations: tuple[SpecialistObservation, ...]) -> str:
    """The domain analysts' observations, attributed and outside the untrusted envelope.

    Outside it deliberately: these are *this system's* gated output, not corpus text. Every
    one already passed its own citation check, so wrapping them as untrusted would say
    something false about where they came from - while the spans they cite stay inside the
    envelope, because those are still documents.
    """
    if not observations:
        return (
            "## Domain analyst observations\n\n"
            "None. No specialist had evidence in its slice for this change, so the domains "
            "below are yours alone to read."
        )

    lines = [
        "## Domain analyst observations",
        "",
        "Each was produced by one analyst over its own evidence slice and has already had its "
        "citations verified. Combine them; do not re-derive them, and do not treat two "
        "analysts citing one document as two sources.",
        "",
    ]
    for item in observations:
        ids = ", ".join(f"`{i}`" for i in item.evidence_ids) or "(no admitted citation)"
        lines.append(
            f"- **{item.agent}** ({item.category}, {item.evidence_position}): {item.claim} [{ids}]"
        )
    return "\n".join(lines)


def _render_analogues(analogues: tuple[AdmissibleAnalogue, ...]) -> str:
    """The historical anchors, and the fact that this list is the whole of what may be cited.

    Worth telling the model plainly: a Class B hypothesis with no anchor caps at
    ``speculative``, so the incentive to invent one is real - and there is nothing to gain,
    because an id outside this list is dropped and the hypothesis lands where it would have
    anyway.
    """
    if not analogues:
        return (
            "## Historical anchors\n\n"
            "None. The historical analyst found no prior case with a recorded outcome in its "
            "slice, so **leave `historical_analogue_id` null**. Every hypothesis you propose "
            "will cap at `speculative`, which is the correct result rather than a problem to "
            "work around - an id that is not in this list is dropped."
        )

    lines = [
        "## Historical anchors (the only ids you may put in `historical_analogue_id`)",
        "",
        "Each is a prior case the historical analyst evidenced, with a **recorded observed "
        "outcome**. Citing one is what lets a behavioural claim exceed `speculative`; citing "
        "anything else is dropped.",
        "",
    ]
    for item in analogues:
        lines.append(
            f"- `{item.event_id}` - {item.title}. Outcome: {item.observed_outcome}"
            + (f" Differs: {item.differs_how}" if item.differs_how else "")
        )
    return "\n".join(lines)


def _render_evidence(evidence: tuple[EvidenceForHunting, ...]) -> str:
    if not evidence:
        return "(no evidence was retrieved for this change)"

    blocks = []
    for span in evidence:
        blocks.append(
            f"[evidence_id: {span.evidence_id}] "
            f"(source_type: {span.source_type}, authority: {span.authority_class})\n"
            f"{span.text}"
        )
    return "\n\n".join(blocks)


def _render_instructions(evidence: tuple[EvidenceForHunting, ...]) -> str:
    """What to return, and the three things that will be rejected if returned.

    The forbidden fields are named even though the schema has nowhere to put them: a model
    that is told will not waste a repair round producing one.
    """
    ids = ", ".join(f"`{span.evidence_id}`" for span in evidence) or "(none)"
    return "\n".join(
        (
            "## Your task",
            "",
            "Propose **Class B** consequences only: behavioural and quantitative effects that "
            "no assumption binding can express - load and capacity shifts, timing and queue "
            "effects, escalation volume, reconciliation drift, second- and third-order "
            "operational chain reactions.",
            "",
            "For each one give the claim, the mechanism as a chain of at least two links, and "
            "which link is weakest. Name the source version the effect lands on if you can.",
            "",
            f"You may cite **only** these evidence ids, verbatim: {ids}",
            "",
            "An id you did not receive will be dropped. A hypothesis with no evidence is a "
            "legitimate result - say so and give the mechanism. An unsupported claim written "
            "as though established is not.",
            "",
            "Do **not** return a confidence number, a severity, a status or a consequence "
            "class. The runtime computes or assigns all four, and the schema will reject them.",
            "",
            "Return only JSON matching the supplied schema.",
        )
    )
