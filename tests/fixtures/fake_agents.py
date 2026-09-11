"""A fake agent that answers from its prompt. The only honest way to run a negative control.

The default ``FakeProvider`` responder returns a digest, which every agentic schema rejects -
so a pipeline run over it would be "quiet" because every stage *failed*, which proves nothing
about the pipeline. A scripted response is no better: pre-writing "no hypotheses" for the
negative control and "one hypothesis" for the positive one would be asserting the script.

So this responder is **maximally cooperative given its input**. It parses the handles and the
admissible evidence ids out of the task contract it was sent and answers for every one of
them, at the strongest position the schema allows. That makes the difference between a
positive and a negative control come from *what the pipeline put in the prompt* - which is
exactly the thing under test.

Two consequences worth stating:

* on a change the corpus knows nothing about, retrieval offers no spans, so the hunter has
  nothing to cite and proposes nothing. The quiet result is produced by the pipeline, not by
  this file.
* an agent this cooperative is the **adversarial** case for every gate upstream: it cites
  everything it is offered and claims the strongest status available. If a finding still comes
  out unsupported, the gates did it.

It parses prompts, which couples it to their wording. That is deliberate: a prompt change that
stops offering evidence would silently make this responder unhelpful, and the end-to-end tests
would fail loudly rather than passing over a degraded prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Final

from app.domain.interfaces.llm import CompletionRequest

#: Handle shapes, one per agent. Rendered by each prompt as ``` `d1` ```.
_HANDLE = {
    "contradiction-hunter": re.compile(r"`(d\d+)`"),
    "adversarial-challenger": re.compile(r"`(c\d+)`"),
    "consequence-judge": re.compile(r"`(j\d+)`"),
    "evidence-arbiter": re.compile(r"`(j\d+)`"),
    "report-composer": re.compile(r"`(r\d+)`"),
}

_EVIDENCE = re.compile(r"`(ev_[0-9a-f]{32})`")
_SOURCE_VERSION = re.compile(r"(sv_[0-9a-f]{32})")

#: A per-candidate block header, e.g. ``### `c1` - class_a_assumption_mediated``.
_BLOCK = re.compile(r"^### `([a-z]\d+)` - (\S+)", re.MULTILINE)

#: The line each prompt uses to list what a candidate may be cited with.
_CITABLE = re.compile(r"evidence you may cite (?:for|against) this (?:item|candidate): (.+)$")

#: The arbiter's prompt reports the deterministically measured corroboration per candidate.
_MEASURED = re.compile(r"^measured distinct source versions: \d+ \((.*)\)$")

#: The blind-spot prompt names each slice and whether it is empty.
_EMPTY_SLICE = re.compile(r"^- `([a-z_]+)` \(empty\)")

#: The composer's prompt names the handles it may foreground.
_ELIGIBLE = re.compile(r"^Eligible handles: (.+)$")

CLASS_A_MARKER: Final = "class_a_assumption_mediated"

#: A well-formed source version that no corpus contains. The arbiter fake claims it, which is
#: how the corroboration gate is exercised by the pipeline rather than by a unit fixture.
INVENTED_VERSION: Final = "sv_" + "0" * 32

#: A well-formed evidence id that no analysis produced. Used by ``composer_invents_evidence``.
INVENTED_EVIDENCE: Final = "ev_" + "1" * 32

#: A well-formed event id the runtime never derived. The consequence-hunter claims it whenever
#: it was offered no real anchor, which is how the anchor gate is exercised by the pipeline
#: rather than by a unit fixture: a Class B finding must still come out `speculative`.
INVENTED_ANALOGUE: Final = "he_" + "2" * 32

#: The line the hunter's prompt uses to list the anchors a hypothesis may cite.
_ANCHOR = re.compile(r"^- `(he_[0-9a-f]{32})` - ")

#: The seven domain analysts. One responder serves all of them: they differ by slice, so a
#: per-agent script would be asserting the difference this file exists to *not* assert.
SPECIALISTS: Final = (
    "policy-analyst",
    "process-analyst",
    "technology-analyst",
    "people-analyst",
    "data-analyst",
    "risk-analyst",
    "historical-analyst",
)


def cooperative_responder(
    *,
    challenger_verdict: str = "no_refutation_found",
    composer_invents_evidence: bool = False,
) -> Callable[[CompletionRequest], str]:
    """Build a responder that answers every agent from its own prompt.

    ``challenger_verdict`` is a *behaviour* rather than a payload: ``no_refutation_found`` is
    the honest default on a corpus with no planted defect, and ``refuted`` exercises the path
    where a challenge succeeds and the judge must reject.

    ``composer_invents_evidence`` makes the report composer cite an id no analysis produced,
    which is the case the narration gate exists for. Off by default, because the default
    responder has to be able to demonstrate the accepting path too.
    """

    def respond(request: CompletionRequest) -> str:
        task = request.messages[-1].content
        match request.agent:
            case "consequence-hunter":
                return _hunter(task)
            case "contradiction-hunter":
                return _contradiction_hunter(task)
            case "adversarial-challenger":
                return _challenger(task, verdict=challenger_verdict)
            case "blind-spot-hunter":
                return _blind_spot_hunter(task)
            case "evidence-arbiter":
                return _arbiter(task)
            case "consequence-judge":
                return _judge(task)
            case "report-composer":
                return _composer(task, invent=composer_invents_evidence)
            case agent if agent in SPECIALISTS:
                return _specialist(task, agent=agent)
            case _:
                return json.dumps({})

    return respond


# ---------------------------------------------------------------------------
# Per-agent answers
# ---------------------------------------------------------------------------


def _hunter(task: str) -> str:
    """One hypothesis citing everything offered, or none when nothing was offered.

    The unanswered question is unconditional: an agent that read the evidence and still could
    not settle a question is the normal case, and it is what feeds the EvidenceGap stage. On a
    quiet change it is the *only* thing produced, which is the honest shape of "we looked and
    the corpus does not say".
    """
    offered = _offered_evidence(task)
    anchors = _offered_anchors(task)
    payload: dict[str, Any] = {
        "hypotheses": [],
        "unanswered_questions": ["What is the observed weekly volume for the affected process?"],
    }
    if offered:
        payload["hypotheses"] = [
            {
                # Always claims an anchor: the real one when the historical analyst evidenced
                # a precedent, an invented one otherwise. So the gate is exercised on every
                # run, and the difference in the resulting status is the pipeline's doing.
                "historical_analogue_id": anchors[0] if anchors else INVENTED_ANALOGUE,
                "claim": "Downstream handling volume shifts materially after the change.",
                "mechanism_chain": [
                    "the change moves the threshold",
                    "more items cross the line",
                    "downstream capacity is unchanged",
                ],
                "weakest_link": "downstream capacity is unchanged",
                "affected_entities": ["downstream handling queue"],
                "cited_evidence_ids": list(offered[:3]),
                "rationale": "Every cited span describes the affected process.",
            }
        ]
    return json.dumps(payload)


def _contradiction_hunter(task: str) -> str:
    """Rate every divergence material, which is the adversarial case for the gate.

    A cooperative agent claiming every divergence matters is exactly what the deterministic
    classification has to survive: a legitimate override rated ``material`` must still not
    become a contradiction.
    """
    assessments = [
        {
            "item_id": handle,
            "materiality": "material",
            "temporal_position": "intervals_overlap",
            "who_is_affected": "the teams operating the affected artifacts",
            "rationale": "Both artifacts are in force and their values differ.",
        }
        for handle, _class in _blocks(task, "contradiction-hunter")
    ]
    return json.dumps({"assessments": assessments})


def _challenger(task: str, *, verdict: str) -> str:
    """A verdict per candidate, in the shape the candidate's class admits.

    ``refuted`` needs a citation and, for a Class A candidate, an admissible ground. Both are
    taken from the prompt, so a candidate offering no citable evidence falls back to
    ``no_refutation_found`` rather than producing an uncited objection the gate would refuse.
    """
    citable = _citable_by_handle(task)
    challenges: list[dict[str, Any]] = []
    for handle, klass in _blocks(task, "adversarial-challenger"):
        available = citable.get(handle, ())
        if verdict == "refuted" and available:
            entry: dict[str, Any] = {
                "handle_placeholder": None,
                "target_id": handle,
                "outcome": "refuted",
                "contradicting_evidence_ids": [available[0]],
                "contradicts_what": "The cited span states a different value than the claim.",
            }
            entry.pop("handle_placeholder")
            if klass == CLASS_A_MARKER:
                entry["class_a_ground"] = "disputed_span"
            challenges.append(entry)
            continue
        challenges.append(
            {
                "target_id": handle,
                "outcome": "no_refutation_found",
                "falsification_observation": (
                    "Re-read the cited spans against the current source revision."
                ),
                "data_source_needed": "the document store",
            }
        )
    return json.dumps({"challenges": challenges})


def _blind_spot_hunter(task: str) -> str:
    """One consequence over the partition, plus a corpus-silence finding, always.

    Cites only what the partition offered, which is the whole point: on a corpus where every
    retrievable span already went to the main hunter, the partition is empty and this agent
    produces silence and nothing else. That result comes from the pipeline, not from here.
    """
    offered = _offered_evidence(task)
    payload: dict[str, Any] = {
        "consequences": [],
        "corpus_silence": [
            {
                "question": (
                    "Which team owns the downstream reconciliation step for the affected process?"
                ),
                "why_it_matters": ("An unowned step has no one to notify when the change lands."),
                "recommended_source": "org ownership metadata",
            }
        ],
        "unusable_slices": _empty_slices(task),
    }
    if offered:
        payload["consequences"] = [
            {
                "claim": (
                    "An undocumented dependency on the changed value goes unnoticed until it "
                    "fails in operation."
                ),
                "mechanism_chain": [
                    "only one artifact records the dependency",
                    "no policy or runbook repeats it",
                    "the change ships without anybody consulting that artifact",
                ],
                "weakest_link": "no policy or runbook repeats it",
                "affected_entities": ["the single holder of the assumption"],
                "cited_evidence_ids": list(offered[:2]),
                # Claims an anchor it was never offered - this hunter is given no admissible
                # set at all, so the runtime must drop it unconditionally and the finding must
                # stay `speculative`.
                "historical_analogue_id": INVENTED_ANALOGUE,
                "rationale": "Every cited span lies outside the main analysis's evidence.",
            }
        ]
    return json.dumps(payload)


def _arbiter(task: str) -> str:
    """The strongest position for every candidate, with an inflated corroboration claim.

    Deliberately over-claiming on the one factor that gets faked: it reports the measured
    source versions **plus** one that no corpus contains. Whatever corroboration survives was
    confirmed by the measurement, not conceded by a cautious fake.
    """
    citable = _citable_by_handle(task)
    measured = _measured_versions(task)
    rulings = [
        {
            "handle": handle,
            "position": "supported",
            "supporting_evidence_ids": list(citable.get(handle, ())),
            "corroborating_source_version_ids": [
                *measured.get(handle, ()),
                INVENTED_VERSION,
            ],
            "independence_note": (
                "Distinct source versions, each reached by the executed retrieval."
            ),
            "residual_disagreement": (
                "The effective date of the governing artifact is not stated in any span."
            ),
        }
        for handle, _class in _blocks(task, "evidence-arbiter")
    ]
    return json.dumps({"rulings": rulings})


def _composer(task: str, *, invent: bool) -> str:
    """Reverse the eligible surface, add a handle it was not offered, and narrate.

    Three adversarial moves in one answer: an unoffered handle (must be dropped), an order
    that disagrees with the deterministic one (must be honoured for the handles it may
    reorder), and - with ``invent`` - a citation to an id the analysis never produced, which
    must discard the whole narration.
    """
    eligible = _eligible_handles(task)
    summary = "The change touches several governed artifacts; review the foregrounded items."
    if invent:
        summary = f"{summary} See evidence {INVENTED_EVIDENCE} for the reconciliation step."
    return json.dumps(
        {
            "foreground_handles": [*reversed(eligible), "r99"],
            "reading_order": list(reversed(eligible)),
            "executive_summary": summary,
            "section_notes": [
                {
                    "section": "contradictions",
                    "note": "Verifiable on delivery; read these before the hypotheses.",
                }
            ],
        }
    )


def _specialist(task: str, *, agent: str) -> str:
    """Two observations over whatever slice this specialist was given, citing all of it.

    Adversarial for every gate this stage has: one observation claims ``direct_evidence`` and
    cites everything offered, the other claims ``direct_evidence`` while citing an id from *no*
    slice - so the citation check and the position ceiling both have something to do. The claim
    text is identical across specialists on purpose, which is what makes the cross-specialist
    duplicate rule observable: two analysts whose slices overlap produce the same claim on the
    same evidence, and only one may survive.
    """
    offered = _offered_evidence(task)
    observations: list[dict[str, Any]] = [
        {
            "claim": (
                "The change alters a value this domain's artifacts depend on without restating it."
            ),
            "basis": "direct_evidence",
            "cited_evidence_ids": list(offered),
            "affected_entities": [f"{agent} scope"],
            "rationale": "Every cited span is in this analyst's own slice.",
        },
        {
            # Cites nothing admissible, so the ceiling must drop it to `unresolved` however
            # confidently it is worded.
            "claim": "A downstream effect is likely although this slice does not evidence it.",
            "basis": "direct_evidence",
            "cited_evidence_ids": [INVENTED_EVIDENCE],
            "rationale": "Asserted without an admissible citation.",
        },
    ]
    payload: dict[str, Any] = {
        "observations": observations,
        "unresolved_questions": [f"What does {agent} need that this slice does not contain?"],
    }
    if offered:
        payload["observed_bindings"] = [
            {"key": "retention.window", "value": "90 d", "evidence_id": offered[0]}
        ]
        # Every specialist proposes an analogue, not just the historical analyst. That is the
        # adversarial case for the agent gate: six of the seven must be refused, and the
        # seventh must still have to evidence and date its precedent.
        payload["historical_events"] = [
            {
                "title": "Prior threshold change with a recorded queue effect",
                "occurred_at": "2024-06-03T00:00:00+00:00",
                "observed_outcome": (
                    "Approval queue depth tripled for three weeks; median turnaround 4h to 19h."
                ),
                "cited_evidence_ids": list(offered[:2]),
                "differs_how": "That change lowered a refund threshold rather than a retention window.",
            },
            {
                # No recorded outcome: kept as a precedent, structurally unable to anchor.
                "title": "An earlier change with no measured result on record",
                "occurred_at": "2023-01-15T00:00:00+00:00",
                "cited_evidence_ids": list(offered[:1]),
                "differs_how": "The source records intent only.",
            },
            {
                # Cites nothing admissible: refused outright, never recorded.
                "title": "A precedent nobody evidenced",
                "occurred_at": "2022-05-01T00:00:00+00:00",
                "observed_outcome": "Asserted without a citation.",
                "cited_evidence_ids": [INVENTED_EVIDENCE],
                "differs_how": "Unknown.",
            },
        ]
    return json.dumps(payload)


def _judge(task: str) -> str:
    """The strongest status the schema allows, citing everything admissible.

    Deliberately over-claiming: ``likely`` on every candidate, with every citable id supplied.
    Whatever comes out below that was lowered by the frozen gates, not by a cautious fake.
    """
    citable = _citable_by_handle(task)
    dispositions = [
        {
            "handle": handle,
            "status": "likely",
            "severity": "MEDIUM",
            "rationale": "The cited evidence describes the affected artifact directly.",
            "supporting_evidence_ids": list(citable.get(handle, ())[:3]),
            "in_default_view": True,
        }
        for handle, _class in _blocks(task, "consequence-judge")
    ]
    return json.dumps({"dispositions": dispositions})


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------


def _offered_anchors(task: str) -> tuple[str, ...]:
    """Event ids the hunter's prompt listed as admissible anchors, in prompt order."""
    return tuple(
        dict.fromkeys(
            match.group(1) for line in task.splitlines() if (match := _ANCHOR.match(line))
        )
    )


def _offered_evidence(task: str) -> tuple[str, ...]:
    """Evidence ids from the hunter's "you may cite only these" line.

    Scoped to that line rather than the whole prompt: the evidence block above it also
    contains the ids, and matching there would let this responder cite spans in an order the
    prompt did not offer them in.

    Shared with the blind-spot hunter and the seven specialists, because all three prompts
    render the same "you may cite only these" line. That coupling is deliberate: a prompt that
    stopped offering evidence would make this responder silently unhelpful, and the end-to-end
    tests would fail loudly rather than passing over a degraded prompt.
    """
    for line in task.splitlines():
        if "You may cite **only** these evidence ids" in line:
            return tuple(dict.fromkeys(_EVIDENCE.findall(line)))
    return ()


def _empty_slices(task: str) -> list[str]:
    """Partition slices the prompt reported as empty, read off the coverage block."""
    return [match.group(1) for line in task.splitlines() if (match := _EMPTY_SLICE.match(line))]


def _eligible_handles(task: str) -> list[str]:
    """Handles the composer is permitted to foreground, from its own instruction line."""
    for line in task.splitlines():
        eligible = _ELIGIBLE.match(line)
        if eligible:
            return list(dict.fromkeys(_HANDLE["report-composer"].findall(eligible.group(1))))
    return []


def _measured_versions(task: str) -> dict[str, tuple[str, ...]]:
    """Per-candidate measured source versions, read from each block's measurement line."""
    found: dict[str, tuple[str, ...]] = {}
    current: str | None = None
    for line in task.splitlines():
        header = _BLOCK.match(line)
        if header:
            current = header.group(1)
            continue
        if current is None:
            continue
        measured = _MEASURED.match(line)
        if measured:
            found[current] = tuple(dict.fromkeys(_SOURCE_VERSION.findall(measured.group(1))))
    return found


def _blocks(task: str, agent: str) -> tuple[tuple[str, str], ...]:
    """Every ``### `handle` - class`` block, in prompt order.

    Cross-checked against the "verbatim" list the prompt also renders, so a handle appearing
    in one and not the other is dropped - which keeps this responder from inventing a handle
    the runtime would refuse and masking a real prompt defect as a gate success.
    """
    offered = set(_HANDLE[agent].findall(task))
    return tuple((handle, klass) for handle, klass in _BLOCK.findall(task) if handle in offered)


def _citable_by_handle(task: str) -> dict[str, tuple[str, ...]]:
    """Per-candidate admissible evidence, read from each block's citable line.

    Per candidate rather than one pool, matching what the runtime enforces: a real id
    attached to the wrong candidate is a fabricated citation assembled out of genuine parts,
    and this responder must not produce one by accident.
    """
    found: dict[str, tuple[str, ...]] = {}
    current: str | None = None
    for line in task.splitlines():
        header = _BLOCK.match(line)
        if header:
            current = header.group(1)
            continue
        if current is None:
            continue
        citable = _CITABLE.search(line)
        if citable:
            found[current] = tuple(dict.fromkeys(_EVIDENCE.findall(citable.group(1))))
    return found
