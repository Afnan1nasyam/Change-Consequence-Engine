"""The consequence-judge runtime: one call, then the frozen gates, then the domain records.

The last agentic stage, and the only one whose output a reader acts on directly. That makes
its gate different in kind from the earlier ones. The hunters' gate refuses claims the model
could not support; the challenger's refuses objections it could not prove. This one refuses
*statuses the evidence does not reach*, and it does so after the model has answered, because
the rules involved are arithmetic rather than judgment.

What is structural here:

**Confidence is computed, never read from the model.** :mod:`app.judgment.proposals` has no
field for it, and the number on the :class:`ConsequenceScore` comes from
:func:`app.domain.services.confidence.compute` over evidence properties. The judge is not
asked, so it cannot be believed.

**Severity is the model's and is not touched.** No branch below moves a severity - not for a
low confidence, not for a weak status. HIGH severity with LOW confidence is a legitimate
output (checkpoint 9.4), and a runtime that quietly reconciled them would destroy the one
signal the judge exists to add.

**Status is proposed, then clamped downward only.** :mod:`app.judgment.status_rules` applies
the checkpoint 9.2 gates. Its signature has no parameter through which a
``no_refutation_found`` could raise anything, so rule 3 holds by construction.

**Disagreements are added to, never trimmed.** Whatever the model preserved is unioned with
what the runtime collected from upstream. There is no code path that removes a disagreement,
which is the only way "do not hide them to make the report cleaner" survives contact with a
model that would rather produce a clean report.

**The identity is inherited.** Every record - ``Consequence``, ``ArbitrationDecision``,
``ConsequenceScore`` - carries the ``consequence_id`` the dossier arrived with. Nothing here
mints one.

**Class A is untouched.** The ``ClassAFinding`` is read for its claim, its rule and its
evidence, and is never constructed, copied or re-provenanced. Its ``rule_id``,
``rule_version``, ``registry_version``, ``coupling_index_version`` and
``authority_model_version`` are the resolver's; the judge's attribution lives on the
``ArbitrationDecision`` instead.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, replace
from typing import Final

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import (
    ArbitrationDecision,
    Consequence,
    ConsequenceScore,
)
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    ProviderError,
    StructuredOutputError,
)
from app.domain.services import confidence as confidence_fn
from app.domain.value_objects.enums import ConsequenceClass, Severity, Status
from app.domain.value_objects.provenance import ModelProvenance, RunProvenance
from app.judgment.dossier import CandidateDossier, by_handle
from app.judgment.prompting import (
    JUDGMENT_PROMPT_VERSION,
    EvidenceForJudgment,
    JudgmentPrompt,
    build_judgment_prompt,
)
from app.judgment.proposals import (
    DISPOSITION_SCHEMA_VERSION,
    DispositionBatch,
    DispositionProposal,
)
from app.judgment.status_rules import (
    ClampedStatus,
    ClampReason,
    clamp,
    in_default_view,
)

logger = logging.getLogger(__name__)

#: The agent this runtime invokes. One name, so a mismatched definition is caught at
#: construction rather than producing dispositions attributed to the wrong prompt.
JUDGE_AGENT: Final = "consequence-judge"

#: Severities that may foreground a Class B finding in the default view.
HIGH_SEVERITIES: Final = frozenset({Severity.HIGH, Severity.CRITICAL})

#: Why a disposition was refused. Counted rather than collapsed: two of these are facts about
#: the model and one is a fact about the pipeline's wiring.
REFUSAL_INVENTED: Final = "invented_handle"
REFUSAL_DUPLICATE: Final = "duplicate_handle"

#: Truncation width for a derived title. ``Consequence.title`` is ``ShortText``.
TITLE_CHARS: Final = 120


def judge_version(definition: AgentDefinition) -> str:
    """The version stamped on every disposition: prompt, contract shape and schema together."""
    return f"judge-{definition.version}:{JUDGMENT_PROMPT_VERSION}:{DISPOSITION_SCHEMA_VERSION}"


@dataclass(frozen=True)
class ArbitratedConsequence:
    """One candidate, arbitrated: the finding, the disposition, and the computed score.

    Three records rather than one merged object, because they have three different authors.
    The ``Consequence`` is the pipeline's, the ``ArbitrationDecision`` is the judge's, and the
    ``ConsequenceScore`` is the confidence function's - and a reader auditing any one of them
    needs to know which.

    All three carry the same ``consequence_id``, inherited from the dossier.
    """

    dossier: CandidateDossier
    consequence: Consequence
    decision: ArbitrationDecision
    score: ConsequenceScore
    #: What the frozen gates did to the judge's proposed status, if anything.
    clamp: ClampedStatus

    def __post_init__(self) -> None:
        ids = {
            self.dossier.consequence_id,
            self.consequence.consequence_id,
            self.decision.consequence_id,
            self.score.consequence_id,
        }
        if len(ids) != 1:
            # The failure this guards is silent: a second minted id would leave the score and
            # the decision pointing at different things, and the report would render both.
            raise ValueError(
                f"one candidate must yield one consequence_id, got {sorted(ids)}. The "
                "identity established upstream is inherited, never re-minted."
            )

    @property
    def consequence_id(self) -> str:
        return self.dossier.consequence_id

    @property
    def status(self) -> Status:
        return self.consequence.status

    @property
    def severity(self) -> Severity:
        return self.consequence.severity

    @property
    def confidence(self) -> float:
        return self.score.confidence


@dataclass(frozen=True)
class JudgmentOutcome:
    """What one consequence-judge call produced, and what producing it revealed."""

    #: One entry per candidate the judge dispositioned and the gate admitted.
    arbitrated: tuple[ArbitratedConsequence, ...] = ()
    #: Candidates with no admitted disposition. Reported, never silently dropped: a finding
    #: nobody arbitrated has not been cleared, and it is not absent either.
    undecided: tuple[CandidateDossier, ...] = ()
    #: Handles the judge dispositioned that it was never offered, with the reason.
    refused: tuple[tuple[str, str], ...] = ()
    #: Cited ids that were not admissible for the candidate they were cited for.
    fabricated_evidence_ids: tuple[str, ...] = ()
    observed_instructions: tuple[str, ...] = ()
    unanswered_questions: tuple[str, ...] = ()
    secrets_redacted: str = "none"
    attempted_delimiter_breakout: bool = False
    calls_made: int = 0
    failure: str | None = None
    provenance: ModelProvenance | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    @property
    def decisions(self) -> tuple[ArbitrationDecision, ...]:
        return tuple(a.decision for a in self.arbitrated)

    @property
    def consequences(self) -> tuple[Consequence, ...]:
        return tuple(a.consequence for a in self.arbitrated)

    @property
    def scores(self) -> tuple[ConsequenceScore, ...]:
        return tuple(a.score for a in self.arbitrated)

    @property
    def default_view(self) -> tuple[ArbitratedConsequence, ...]:
        return tuple(a for a in self.arbitrated if a.decision.in_default_view)

    @property
    def clamped(self) -> tuple[ArbitratedConsequence, ...]:
        """Candidates whose proposed status the gates lowered.

        Surfaced because a clamp and a cautious judge look identical in the output otherwise,
        and they mean different things: one is the corpus being thin, the other is an opinion.
        """
        return tuple(a for a in self.arbitrated if a.clamp.was_clamped)

    def of_status(self, status: Status) -> tuple[ArbitratedConsequence, ...]:
        return tuple(a for a in self.arbitrated if a.status is status)


class ConsequenceJudge:
    """Invokes ``consequence-judge`` over the assembled dossiers and admits the dispositions.

    Holds the completion port and the agent definition. No store, no retriever, and no
    confidence function of its own - the score comes from the domain service, so this class
    cannot be given a different one by configuration.
    """

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        if definition.name != JUDGE_AGENT:
            raise ValueError(f"expected the {JUDGE_AGENT!r} definition, got {definition.name!r}")
        self._completions = completions
        self._definition = definition

    @property
    def version(self) -> str:
        return judge_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        """The judge's own attribution. Never a candidate's.

        Resolved without a model call, so a failed call still carries full provenance. A
        ``Consequence`` derived by the resolver keeps the resolver's version labels; this is
        what produced the *disposition*, and it lives on the ``ArbitrationDecision``.
        """
        binding = self._completions.resolve(JUDGE_AGENT)
        return ModelProvenance(
            agent=JUDGE_AGENT,
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
        run_provenance: RunProvenance,
        confidence_inputs: dict[str, confidence_fn.ConfidenceInputs] | None = None,
        now: dt.datetime | None = None,
        correlation_id: str | None = None,
    ) -> JudgmentOutcome:
        """Disposition every candidate. One call, then the frozen gates.

        ``confidence_inputs`` maps ``consequence_id`` to the measured evidence properties for
        that candidate. Supplied by the caller because measuring them needs the store, and
        defaulted to an empty input rather than skipped: a consequence with no score is a
        consequence a reader cannot rank, and an honest zero says the inputs were not
        measured rather than hiding that they were not.
        """
        provenance = self._provenance()
        computed_at = now or dt.datetime.now(tz=dt.UTC)

        if not dossiers:
            logger.info(
                "no candidate consequences for this change; consequence-judge not called",
                extra={"change_id": change.change_id, "agent": JUDGE_AGENT},
            )
            return JudgmentOutcome(provenance=provenance, calls_made=0)

        offered = by_handle(dossiers)

        prompt = build_judgment_prompt(
            system_prompt=self._definition.prompt,
            change=change,
            dossiers=dossiers,
            evidence=evidence,
            nonce=uuid.uuid4().hex[:12],
        )
        self._log_prompt_observations(prompt, change=change)

        base = JudgmentOutcome(
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                JUDGE_AGENT,
                prompt.messages,
                json_schema=DispositionBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            logger.warning(
                "consequence-judge call failed; candidates reported undecided",
                extra={"change_id": change.change_id, "error": str(exc)},
            )
            return replace(
                base,
                undecided=dossiers,
                failure=f"{type(exc).__name__}: {exc}",
                calls_made=1,
            )

        try:
            batch = DispositionBatch.model_validate(payload)
        except ValidationError as exc:
            # Includes the cases that matter most: a judge emitting `confidence`, a
            # `consequence_class`, or an unresolved high-severity finding with no action.
            logger.warning(
                "consequence-judge output violated the disposition schema; discarded",
                extra={"change_id": change.change_id, "errors": exc.error_count()},
            )
            return replace(
                base,
                undecided=dossiers,
                failure=f"schema: {exc.error_count()} error(s)",
                calls_made=1,
            )

        return self._admit(
            batch,
            base=base,
            offered=offered,
            prompt=prompt,
            change=change,
            run_provenance=run_provenance,
            confidence_inputs=confidence_inputs or {},
            computed_at=computed_at,
        )

    def _log_prompt_observations(self, prompt: JudgmentPrompt, *, change: ChangeSpec) -> None:
        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "candidate evidence attempted to impersonate the untrusted-content delimiter",
                extra={"change_id": change.change_id, "agent": JUDGE_AGENT},
            )
        if prompt.redaction.found_any:
            logger.info(
                "redacted credential-shaped spans before dispatch",
                extra={"change_id": change.change_id, "secrets": prompt.redaction.summary},
            )

    # -- the gate --------------------------------------------------------

    def _admit(
        self,
        batch: DispositionBatch,
        *,
        base: JudgmentOutcome,
        offered: dict[str, CandidateDossier],
        prompt: JudgmentPrompt,
        change: ChangeSpec,
        run_provenance: RunProvenance,
        confidence_inputs: dict[str, confidence_fn.ConfidenceInputs],
        computed_at: dt.datetime,
    ) -> JudgmentOutcome:
        """Turn dispositions into domain records, applying the frozen gates on the way."""
        assert base.provenance is not None  # set by `arbitrate` before the call

        arbitrated: dict[str, ArbitratedConsequence] = {}
        refused: list[tuple[str, str]] = []
        fabricated: list[str] = []

        for proposal in batch.dispositions:
            handle = proposal.handle
            dossier = offered.get(handle)
            if dossier is None:
                refused.append((handle, REFUSAL_INVENTED))
                continue
            if handle in arbitrated:
                refused.append((handle, REFUSAL_DUPLICATE))
                continue

            supporting, contradicting, borrowed = _partition_citations(
                proposal, admissible=prompt.offered_evidence_ids.get(handle, frozenset())
            )
            fabricated.extend(borrowed)

            arbitrated[handle] = self._build(
                proposal,
                dossier=dossier,
                supporting=supporting,
                contradicting=contradicting,
                provenance=base.provenance,
                change=change,
                run_provenance=run_provenance,
                inputs=confidence_inputs.get(
                    dossier.consequence_id, confidence_fn.ConfidenceInputs()
                ),
                computed_at=computed_at,
            )

        if refused:
            logger.warning(
                "consequence-judge dispositions refused by the gate",
                extra={"agent": JUDGE_AGENT, "count": len(refused)},
            )
        if fabricated:
            logger.warning(
                "consequence-judge cited evidence not admissible for the candidate; dropped",
                extra={"agent": JUDGE_AGENT, "count": len(fabricated)},
            )

        return replace(
            base,
            arbitrated=tuple(arbitrated[h] for h in offered if h in arbitrated),
            undecided=tuple(d for h, d in offered.items() if h not in arbitrated),
            refused=tuple(sorted(set(refused))),
            fabricated_evidence_ids=tuple(sorted(set(fabricated))),
            observed_instructions=batch.observed_instructions,
            unanswered_questions=batch.unanswered_questions,
            calls_made=1,
        )

    def _build(
        self,
        proposal: DispositionProposal,
        *,
        dossier: CandidateDossier,
        supporting: tuple[str, ...],
        contradicting: tuple[str, ...],
        provenance: ModelProvenance,
        change: ChangeSpec,
        run_provenance: RunProvenance,
        inputs: confidence_fn.ConfidenceInputs,
        computed_at: dt.datetime,
    ) -> ArbitratedConsequence:
        """Construct the three records for one candidate.

        Order matters: clamp the status first, because both the ``Consequence`` and the
        ``ArbitrationDecision`` must carry the same one - two records disagreeing about
        whether a claim holds is worse than either being wrong.
        """
        clamped = clamp(
            proposal.status,
            consequence_class=dossier.consequence_class,
            has_supporting_evidence=bool(supporting),
            has_historical_analogue=dossier.historical_analogue_id is not None,
            was_refuted=dossier.was_refuted,
            was_weakened=dossier.was_weakened,
        )
        status = clamped.status

        # Added to, never trimmed. The runtime's collection is unioned with the judge's, so a
        # disagreement survives a model that would rather not mention it.
        disagreements = _merge_disagreements(
            proposal.preserved_disagreements, dossier.upstream_disagreements
        )

        # A clamp can move a status *into* `unresolved`, and the domain requires an
        # unresolved high-severity finding to name what would settle it. The model proposed a
        # different status, so it was never asked - which makes the action the runtime's to
        # supply, exactly as the clamp was the runtime's to apply.
        actions = _validation_actions(proposal, clamped)

        foreground = in_default_view(
            proposed=proposal.in_default_view,
            consequence_class=dossier.consequence_class,
            severity_is_high=proposal.severity in HIGH_SEVERITIES,
            has_historical_analogue=dossier.historical_analogue_id is not None,
            status=status,
        )

        consequence = Consequence(
            # Inherited. Nothing here mints one.
            consequence_id=dossier.consequence_id,
            change_id=change.change_id,
            title=dossier.claim[:TITLE_CHARS],
            description=dossier.claim,
            # Derived from the producer, not proposed: the rule for Class A, the behavioural
            # label for Class B.
            category=dossier.rule_id or dossier.consequence_class.value,
            # Pipeline-assigned, from what produced the finding.
            consequence_class=dossier.consequence_class,
            origin=dossier.origin,
            affected_entity=dossier.affected_entity,
            mechanism=dossier.mechanism,
            derived_relation_ids=(
                (dossier.finding.relation_id,) if dossier.finding is not None else ()
            ),
            hypothesis_ids=(
                (dossier.hypothesis.hypothesis_id,) if dossier.hypothesis is not None else ()
            ),
            merged_hypothesis_ids=(
                dossier.finding.merged_hypothesis_ids if dossier.finding is not None else ()
            ),
            status=status,
            # The judge's, untouched. No branch here moves it.
            severity=proposal.severity,
            evidence_ids=supporting,
            contradicting_evidence_ids=contradicting,
            assumption_keys=dossier.assumption_keys,
            # Type E travels as a reference. A gap is a claim about the corpus and never a
            # consequence, so it appears here and nowhere else.
            evidence_gap_ids=dossier.evidence_gap_ids,
            historical_analogue_id=dossier.historical_analogue_id,
            recommended_validation=actions,
            agent_sources=_agent_sources(dossier),
            run_provenance=run_provenance,
        )

        decision = ArbitrationDecision(
            consequence_id=dossier.consequence_id,
            status=status,
            severity=proposal.severity,
            rationale=_rationale(proposal, clamped),
            supporting_evidence_ids=supporting,
            contradicting_evidence_ids=contradicting,
            preserved_disagreements=disagreements,
            validation_actions=actions,
            in_default_view=foreground,
            produced_by=provenance,
        )

        score = confidence_fn.compute(
            inputs,
            consequence_id=dossier.consequence_id,
            computed_at=computed_at,
        )

        return ArbitratedConsequence(
            dossier=dossier,
            consequence=consequence,
            decision=decision,
            score=score,
            clamp=clamped,
        )


#: What would settle a finding the gates lowered, per reason. Mechanical rather than
#: invented: each entry restates the clamp reason as the thing that would remove it, which is
#: what the domain asks for when it requires an unresolved high-risk finding to name a next
#: step. The judge's own actions always come first; these are only added when it supplied
#: none, because it proposed a status that owed nothing.
_CLAMP_ACTIONS: Final[dict[str, str]] = {
    ClampReason.NO_SUPPORTING_EVIDENCE: (
        "locate a span in the corpus that supports this claim and is retrievable under this "
        "analysis's filters, or record why none exists"
    ),
    ClampReason.CLASS_B_WITHOUT_ANALOGUE: (
        "find a past change or incident with a recorded observed outcome that this effect is "
        "analogous to, or leave the claim speculative"
    ),
    ClampReason.WEAKENED_BY_CHALLENGER: (
        "confirm the scope the challenger's narrowing leaves in force, and whether it still "
        "covers this artifact"
    ),
    ClampReason.REFUTED_BY_CHALLENGER: (
        "confirm the refuting source is current and governs this artifact before closing the "
        "finding"
    ),
}


def _validation_actions(proposal: DispositionProposal, clamped: ClampedStatus) -> tuple[str, ...]:
    """The judge's next steps, plus one derived from any clamp that owes the reader one.

    The domain refuses an unresolved high-severity finding with no action, and a clamp can
    put a record in that state after the model has answered - it proposed something that owed
    nothing, so it was never asked. Deriving the action from the clamp reason is not the
    runtime inventing a deliverable: the reason *is* the thing to fix, and restating it is
    mechanical. The alternative - failing the batch - would discard a whole arbitration
    because the model could not anticipate a gate.
    """
    actions = list(proposal.validation_actions)
    if not clamped.was_clamped:
        return tuple(actions)
    for reason in clamped.reasons:
        derived = _CLAMP_ACTIONS.get(reason)
        if derived and derived not in actions:
            actions.append(derived)
    return tuple(actions)


def _rationale(proposal: DispositionProposal, clamped: ClampedStatus) -> str:
    """The judge's reasoning, with any clamp recorded alongside it.

    Appended rather than substituted: the judge's opinion is data even where the gates
    overrode it, and a reader comparing two runs needs to see that the disposition was
    lowered rather than that the judge became cautious.
    """
    if not clamped.was_clamped:
        return proposal.rationale
    reasons = ", ".join(clamped.reasons)
    note = f" [status lowered from {clamped.proposed.value} to {clamped.status.value} by {reasons}]"
    return (proposal.rationale + note)[:2000]


def _merge_disagreements(
    from_judge: tuple[str, ...], from_upstream: tuple[str, ...]
) -> tuple[str, ...]:
    """Union, judge first, order preserved, duplicates removed.

    There is deliberately no parameter through which a caller could ask for fewer. The
    requirement is that disagreements survive, and a trimming option is how that requirement
    quietly stops holding once a report looks crowded.
    """
    merged: list[str] = []
    for item in (*from_judge, *from_upstream):
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def _agent_sources(dossier: CandidateDossier) -> tuple[str, ...]:
    """Which agents contributed to this candidate.

    Recorded for audit, and deliberately **not** an input to anything scored: corroboration
    counts distinct source versions, never agreeing agents (checkpoint 9.3). It cannot leak
    into the score because ``ConfidenceInputs`` has no field it would fit.
    """
    sources: list[str] = [JUDGE_AGENT]
    if dossier.hypothesis is not None:
        sources.append(dossier.hypothesis.proposed_by.agent)
    else:
        sources.append("resolver")
    if dossier.challenges:
        sources.append("adversarial-challenger")
    if dossier.divergences:
        sources.append("contradiction-hunter")
    return tuple(dict.fromkeys(sources))


def _partition_citations(
    proposal: DispositionProposal, *, admissible: frozenset[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split cited ids into admitted support, admitted contradiction, and the rest.

    Exact string match against the ids offered **for this candidate**, with no
    normalization. An id that differs by a character is not a near miss to be repaired, and a
    real id belonging to a different candidate would support an unrelated claim - a
    fabricated citation assembled out of genuine parts.
    """
    supporting = tuple(sorted({e for e in proposal.supporting_evidence_ids if e in admissible}))
    contradicting = tuple(
        sorted({e for e in proposal.contradicting_evidence_ids if e in admissible})
    )
    borrowed = tuple(
        e
        for e in (*proposal.supporting_evidence_ids, *proposal.contradicting_evidence_ids)
        if e not in admissible
    )
    return supporting, contradicting, borrowed


#: Re-exported so a caller can build the inputs without importing the domain service path.
ConfidenceInputs = confidence_fn.ConfidenceInputs

__all__ = [
    "HIGH_SEVERITIES",
    "JUDGE_AGENT",
    "REFUSAL_DUPLICATE",
    "REFUSAL_INVENTED",
    "TITLE_CHARS",
    "ArbitratedConsequence",
    "ConfidenceInputs",
    "ConsequenceClass",
    "ConsequenceJudge",
    "JudgmentOutcome",
    "judge_version",
]
