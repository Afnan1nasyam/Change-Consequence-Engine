"""The adversarial-challenger runtime: one call, then a gate that only ever weakens a verdict.

The third agentic stage, and the one with the most inverted incentive. A model told to
disagree will disagree, fluently, at whatever rate it is asked. So this runtime is built
around a single principle: **an objection is a proof obligation, and an unproven objection
is discarded rather than softened.**

Five properties make that structural rather than instructed:

**It cannot search.** This class holds a completion port and an agent definition. No
retriever, no store. The candidates and every citable span are handed to it, and the layering
contract forbids ``app/challenge`` from importing ``app.retrieval`` at all.

**An empty candidate set costs zero calls.** Nothing finalized, nothing to falsify, and the
provider is never reached.

**A verdict on a candidate nobody proposed is dropped.** Handles are checked against exactly
the offered set. A model that returns a verdict for ``c9`` has not found a ninth candidate; it
has produced a review of nothing - and admitting it would let the one stage nothing reviews
introduce findings.

**An inadmissible Class A objection produces no ChallengeResult at all.** Not a downgrade to
``no_refutation_found``: that outcome asserts "I looked and found nothing", which is not what
happened. The model found something it was not permitted to say, and manufacturing the wrong
outcome would misreport the run *and* fabricate the falsification test that outcome owes. It
is refused and counted instead.

**A citation that survives nothing kills the verdict, not the citation.** ``refuted`` and
``weakened`` require at least one admitted id. If every cited id was fabricated or borrowed
from another candidate, the objection is uncited and is refused - because the alternative is
constructing a ``ChallengeResult`` the domain would reject, one layer later, from the wrong
place.

**Nothing here touches a candidate.** The ``ClassAFinding`` is read for its claim and its
evidence and is never constructed, copied or re-provenanced. A ``ChallengeResult`` carries its
*own* ``produced_by``, resolved from the agent binding, so the challenger's attribution and
the resolver's cannot be confused for one another.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import Final

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.challenge.candidates import ChallengeTarget, by_handle
from app.challenge.prompting import (
    CHALLENGE_PROMPT_VERSION,
    ChallengePrompt,
    EvidenceForChallenge,
    build_challenge_prompt,
)
from app.challenge.proposals import (
    CHALLENGE_SCHEMA_VERSION,
    ChallengeBatch,
    ChallengeProposal,
)
from app.domain.entities.change import ChangeSpec
from app.domain.entities.findings import ChallengeResult, FalsificationTest
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    ProviderError,
    StructuredOutputError,
)
from app.domain.value_objects.enums import ChallengeOutcome
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.provenance import ModelProvenance

logger = logging.getLogger(__name__)

#: The agent this runtime invokes. One name, so a mismatched definition is caught at
#: construction rather than producing verdicts attributed to the wrong prompt.
CHALLENGER_AGENT: Final = "adversarial-challenger"


def challenger_version(definition: AgentDefinition) -> str:
    """The version stamped on every verdict: prompt, contract shape and schema together.

    All three, because each changes what the output means. A prompt edit changes what was
    asked, a contract change changes how it was asked, and a schema change changes what
    could be answered.
    """
    return f"challenge-{definition.version}:{CHALLENGE_PROMPT_VERSION}:{CHALLENGE_SCHEMA_VERSION}"


#: Why a verdict was refused. Counted rather than collapsed, because the three mean different
#: things about the model and only one of them is about the corpus.
REFUSAL_INVENTED: Final = "invented_target"
REFUSAL_INADMISSIBLE: Final = "inadmissible_class_a_ground"
REFUSAL_UNCITED: Final = "no_admitted_citation"
REFUSAL_GROUND_ON_CLASS_B: Final = "class_a_ground_on_class_b"


@dataclass(frozen=True)
class RefusedChallenge:
    """A verdict the gate would not admit, and the candidate it was aimed at.

    Kept rather than dropped silently. "The challenger tried to refute this on inadmissible
    grounds" is a fact about the model worth surfacing, and it is a different fact from "the
    challenger found nothing".
    """

    target_id: str
    outcome: ChallengeOutcome
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class ChallengedCandidate:
    """One candidate and, where the gate admitted one, its verdict.

    Paired rather than merged. ``target`` is the finalized candidate as it was handed in and
    ``result`` is the challenger's verdict on it - and keeping them separate is what makes
    "the challenger never modifies the finding" visible in the type: a consumer reading the
    candidate never passes through anything the model wrote.

    ``result`` is ``None`` for a candidate that was not challenged, or whose verdict the gate
    refused. That is a reportable absence: an unchallenged candidate has not survived review.
    """

    target: ChallengeTarget
    result: ChallengeResult | None = None
    #: The falsification test minted for a ``no_refutation_found`` verdict.
    falsification_test: FalsificationTest | None = None

    @property
    def target_id(self) -> str:
        return self.target.target_id

    @property
    def consequence_id(self) -> str:
        return self.target.consequence_id

    @property
    def outcome(self) -> ChallengeOutcome | None:
        return self.result.outcome if self.result else None

    @property
    def was_challenged(self) -> bool:
        return self.result is not None

    @property
    def survived(self) -> bool:
        """Whether review left the candidate standing.

        ``no_refutation_found`` counts as surviving, and an unchallenged candidate does not.
        Note what this is **not**: corroboration. Nobody finding contrary evidence is
        strictly weaker than evidence of correctness and must never raise a status
        (checkpoint 9.2).
        """
        return self.outcome is ChallengeOutcome.NO_REFUTATION_FOUND


@dataclass(frozen=True)
class ChallengeOutcomeSet:
    """What one adversarial-challenger call produced, and what producing it revealed.

    ``failure`` is a string rather than an exception so the outcome stays serializable and an
    analysis can record *that* the stage failed alongside everything that succeeded.
    """

    #: Every supplied candidate, challenged or not. The set is never shortened: a candidate
    #: the model ignored still reaches the reader, flagged as unchallenged.
    challenged: tuple[ChallengedCandidate, ...] = ()
    #: Verdicts the gate would not admit, with the reason.
    refused: tuple[RefusedChallenge, ...] = ()
    #: Handles the model returned more than once. The first verdict wins; the rest are
    #: dropped, because two verdicts on one candidate cannot both be the review.
    duplicate_target_ids: tuple[str, ...] = ()
    #: Cited ids that were not offered against the candidate they were cited for - whether
    #: invented outright or borrowed from another candidate.
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
    def refuted(self) -> tuple[ChallengedCandidate, ...]:
        return self._of(ChallengeOutcome.REFUTED)

    @property
    def weakened(self) -> tuple[ChallengedCandidate, ...]:
        return self._of(ChallengeOutcome.WEAKENED)

    @property
    def unrefuted(self) -> tuple[ChallengedCandidate, ...]:
        return self._of(ChallengeOutcome.NO_REFUTATION_FOUND)

    @property
    def unchallenged(self) -> tuple[ChallengedCandidate, ...]:
        return tuple(c for c in self.challenged if not c.was_challenged)

    @property
    def falsification_tests(self) -> tuple[FalsificationTest, ...]:
        """The deliverable ``no_refutation_found`` paid for."""
        return tuple(
            c.falsification_test for c in self.challenged if c.falsification_test is not None
        )

    @property
    def refusals_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for refusal in self.refused:
            counts[refusal.reason] = counts.get(refusal.reason, 0) + 1
        return counts

    def _of(self, outcome: ChallengeOutcome) -> tuple[ChallengedCandidate, ...]:
        return tuple(c for c in self.challenged if c.outcome is outcome)


class AdversarialChallenger:
    """Invokes ``adversarial-challenger`` over finalized candidates and admits the verdicts.

    Holds the completion port and the agent definition, and nothing else. The absence of a
    retriever is the point: the challenger reviews what it is handed. The port is injected,
    so this class never names a provider or a model - ``AgentCompletionPort`` resolves tier,
    provider and model from configuration.
    """

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
    ) -> None:
        if definition.name != CHALLENGER_AGENT:
            raise ValueError(
                f"expected the {CHALLENGER_AGENT!r} definition, got {definition.name!r}"
            )
        self._completions = completions
        self._definition = definition

    @property
    def version(self) -> str:
        return challenger_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        """Attribution for anything this challenger asserts.

        Resolved without a model call, so a failed call still carries full provenance and the
        outcome can say which prompt and which binding failed. This is the challenger's own
        attribution and never touches a candidate's: a ``ChallengeResult`` says who
        challenged, the finding still says who derived.
        """
        binding = self._completions.resolve(CHALLENGER_AGENT)
        return ModelProvenance(
            agent=CHALLENGER_AGENT,
            prompt_version=self._definition.version,
            prompt_sha256=self._definition.prompt_sha256,
            provider=binding.provider,
            model=binding.model,
            tier=binding.tier,
        )

    async def challenge(
        self,
        *,
        change: ChangeSpec,
        targets: tuple[ChallengeTarget, ...],
        evidence: tuple[EvidenceForChallenge, ...] = (),
        correlation_id: str | None = None,
    ) -> ChallengeOutcomeSet:
        """Review the supplied candidates. One call, then the gate."""
        provenance = self._provenance()

        if not targets:
            # Nothing finalized, so nothing to falsify. The provider is never reached, which
            # means there is no output to filter and no opportunity for a review stage to
            # introduce a finding.
            logger.info(
                "no finalized candidates for this change; adversarial-challenger not called",
                extra={"change_id": change.change_id, "agent": CHALLENGER_AGENT},
            )
            return ChallengeOutcomeSet(provenance=provenance, calls_made=0)

        offered = by_handle(targets)

        prompt = build_challenge_prompt(
            system_prompt=self._definition.prompt,
            change=change,
            targets=targets,
            evidence=evidence,
            nonce=uuid.uuid4().hex[:12],
        )
        self._log_prompt_observations(prompt, change=change)

        base = ChallengeOutcomeSet(
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                CHALLENGER_AGENT,
                prompt.messages,
                json_schema=ChallengeBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            # The existing error path. A failed stage is reported, never inferred from an
            # empty result - and the candidates survive it unchallenged rather than
            # disappearing. Note what this must not become: an outage is not review, so
            # these candidates are `unchallenged`, not `survived`.
            logger.warning(
                "adversarial-challenger call failed; candidates reported unchallenged",
                extra={"change_id": change.change_id, "error": str(exc)},
            )
            return replace(
                base,
                challenged=_unchallenged(targets),
                failure=f"{type(exc).__name__}: {exc}",
                calls_made=1,
            )

        try:
            batch = ChallengeBatch.model_validate(payload)
        except ValidationError as exc:
            # Includes the cases that matter most: an uncited `refuted`, a
            # `no_refutation_found` with no falsification observation, and any attempt at
            # `confidence`, `severity` or `status`. `extra="forbid"` and the per-outcome
            # validators turn each into a rejection instead of a partially trusted verdict.
            logger.warning(
                "adversarial-challenger output violated the challenge schema; discarded",
                extra={"change_id": change.change_id, "errors": exc.error_count()},
            )
            return replace(
                base,
                challenged=_unchallenged(targets),
                failure=f"schema: {exc.error_count()} error(s)",
                calls_made=1,
            )

        return self._admit(batch, base=base, offered=offered, prompt=prompt)

    def _log_prompt_observations(self, prompt: ChallengePrompt, *, change: ChangeSpec) -> None:
        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "candidate evidence attempted to impersonate the untrusted-content delimiter",
                extra={"change_id": change.change_id, "agent": CHALLENGER_AGENT},
            )
        if prompt.redaction.found_any:
            logger.info(
                "redacted credential-shaped spans before dispatch",
                extra={"change_id": change.change_id, "secrets": prompt.redaction.summary},
            )

    # -- the gate --------------------------------------------------------

    def _admit(
        self,
        batch: ChallengeBatch,
        *,
        base: ChallengeOutcomeSet,
        offered: dict[str, ChallengeTarget],
        prompt: ChallengePrompt,
    ) -> ChallengeOutcomeSet:
        """Turn proposals into ``ChallengeResult`` records, keeping only what is provable.

        Four checks, applied in this order and for this reason: a verdict on a candidate that
        does not exist cannot be assessed for admissibility; an inadmissible objection should
        be reported as inadmissible rather than as uncited; and only then does it matter
        whether the citation survives.

        Every step can weaken what the model said and none can strengthen it. There is no
        branch here that admits a citation the prompt did not offer, supplies a ground the
        model omitted, or invents the falsification observation that ``no_refutation_found``
        owes.
        """
        assert base.provenance is not None  # set by `challenge` before the call

        accepted: dict[str, tuple[ChallengeResult, FalsificationTest | None]] = {}
        refused: list[RefusedChallenge] = []
        duplicated: list[str] = []
        fabricated: list[str] = []

        for proposal in batch.challenges:
            handle = proposal.target_id
            target = offered.get(handle)

            if target is None:
                refused.append(
                    RefusedChallenge(
                        target_id=handle,
                        outcome=proposal.outcome,
                        reason=REFUSAL_INVENTED,
                        detail="no candidate with this handle was supplied",
                    )
                )
                continue
            if handle in accepted:
                duplicated.append(handle)
                continue

            inadmissible = _inadmissibility(proposal, target)
            if inadmissible is not None:
                refused.append(inadmissible)
                continue

            kept, borrowed = _partition_citations(
                proposal, admissible=prompt.offered_evidence_ids.get(handle, frozenset())
            )
            fabricated.extend(borrowed)

            if proposal.owes_a_citation and not kept:
                # Every cited id was fabricated or borrowed. The objection is uncited, and an
                # uncited objection is not representable - so it is refused here rather than
                # allowed to fail domain construction one layer later.
                refused.append(
                    RefusedChallenge(
                        target_id=handle,
                        outcome=proposal.outcome,
                        reason=REFUSAL_UNCITED,
                        detail=(
                            f"all {len(proposal.contradicting_evidence_ids)} cited id(s) were "
                            "not admissible against this candidate"
                        ),
                    )
                )
                continue

            accepted[handle] = _build(
                proposal, target=target, admitted=kept, provenance=base.provenance
            )

        self._log_refusals(refused, fabricated)

        challenged = tuple(
            ChallengedCandidate(
                target=target,
                result=accepted[handle][0] if handle in accepted else None,
                falsification_test=accepted[handle][1] if handle in accepted else None,
            )
            for handle, target in offered.items()
        )

        return replace(
            base,
            challenged=challenged,
            refused=tuple(refused),
            duplicate_target_ids=tuple(sorted(set(duplicated))),
            fabricated_evidence_ids=tuple(sorted(set(fabricated))),
            observed_instructions=batch.observed_instructions,
            unanswered_questions=batch.unanswered_questions,
            calls_made=1,
        )

    def _log_refusals(self, refused: list[RefusedChallenge], fabricated: list[str]) -> None:
        by_reason: dict[str, int] = {}
        for refusal in refused:
            by_reason[refusal.reason] = by_reason.get(refusal.reason, 0) + 1
        if by_reason:
            logger.warning(
                "adversarial-challenger verdicts refused by the gate",
                extra={"agent": CHALLENGER_AGENT, "by_reason": by_reason},
            )
        if fabricated:
            logger.warning(
                "adversarial-challenger cited evidence not admissible for the candidate",
                extra={"agent": CHALLENGER_AGENT, "count": len(fabricated)},
            )


def _inadmissibility(
    proposal: ChallengeProposal, target: ChallengeTarget
) -> RefusedChallenge | None:
    """Whether this verdict is admissible against this candidate, per checkpoint 8.1.

    Two asymmetric rules, and the asymmetry is the whole guard:

    A **Class A** candidate refuted or weakened without one of the four grounds is refused. A
    derivation is only ever rule-wrong or input-wrong, and a verdict that names no ground is
    a verdict about the conclusion - "I do not think that follows" - which is inadmissible.
    Note that a mechanism objection has no other shape available: the enum has no mechanism
    member, so a model attempting one either omits the ground and lands here, or invents an
    enum value and fails schema validation.

    A **Class B** candidate carrying a ground is also refused. There is no derivation whose
    inputs could be disputed, so the ground is a category error - and admitting it would put
    a Class A vocabulary on a model-proposed claim, which is how a reader comes to believe a
    hypothesis was derived.
    """
    if not proposal.owes_a_citation:
        # `no_refutation_found` refutes nothing, so it needs no ground either way. Requiring
        # one would make "I found nothing" cost an admissibility argument.
        if proposal.class_a_ground is not None:
            return RefusedChallenge(
                target_id=proposal.target_id,
                outcome=proposal.outcome,
                reason=REFUSAL_INADMISSIBLE,
                detail="no_refutation_found refutes nothing and must not name a ground",
            )
        return None

    if target.requires_class_a_ground:
        if proposal.class_a_ground is None:
            return RefusedChallenge(
                target_id=proposal.target_id,
                outcome=proposal.outcome,
                reason=REFUSAL_INADMISSIBLE,
                detail=(
                    f"{proposal.outcome.value} against a deterministic derivation must name "
                    f"one of {sorted(g.value for g in target.admissible_grounds)}. "
                    "Disagreeing with the conclusion is inadmissible."
                ),
            )
        if proposal.class_a_ground not in target.admissible_grounds:
            return RefusedChallenge(
                target_id=proposal.target_id,
                outcome=proposal.outcome,
                reason=REFUSAL_INADMISSIBLE,
                detail=f"{proposal.class_a_ground.value} is not admissible for this candidate",
            )
        return None

    if proposal.class_a_ground is not None:
        return RefusedChallenge(
            target_id=proposal.target_id,
            outcome=proposal.outcome,
            reason=REFUSAL_GROUND_ON_CLASS_B,
            detail=(
                "a Class B hypothesis has no derivation whose inputs could be disputed, so "
                "a class_a_ground is a category error"
            ),
        )
    return None


def _build(
    proposal: ChallengeProposal,
    *,
    target: ChallengeTarget,
    admitted: tuple[str, ...],
    provenance: ModelProvenance,
) -> tuple[ChallengeResult, FalsificationTest | None]:
    """Construct the domain records. Identifiers are minted here, never by the model.

    The ``consequence_id`` comes off the target, so a verdict is attached to the candidate the
    runtime projected and not to anything the model named. The ``class_a_ground`` is carried
    through only for a Class A candidate - it has already been checked admissible - and is
    dropped for a Class B one, where it has no meaning.
    """
    test: FalsificationTest | None = None
    if proposal.outcome is ChallengeOutcome.NO_REFUTATION_FOUND:
        # The price of the outcome, and a real deliverable: it feeds the validation
        # checklist. The observation is the model's; the identity is not.
        assert proposal.falsification_observation is not None  # schema-guaranteed
        test = FalsificationTest(
            test_id=new_id("test"),
            consequence_id=target.consequence_id,
            observation=proposal.falsification_observation,
            data_source_needed=proposal.data_source_needed,
        )

    result = ChallengeResult(
        challenge_id=new_id("challenge"),
        consequence_id=target.consequence_id,
        outcome=proposal.outcome,
        contradicting_evidence_ids=admitted,
        contradicts_what=proposal.contradicts_what,
        scope_narrowing=proposal.scope_narrowing,
        falsification_test_id=test.test_id if test else None,
        class_a_ground=proposal.class_a_ground if target.requires_class_a_ground else None,
        produced_by=provenance,
    )
    return result, test


def _unchallenged(targets: tuple[ChallengeTarget, ...]) -> tuple[ChallengedCandidate, ...]:
    """Every candidate, carried through with no verdict.

    Used on both failure paths. The candidates do not depend on the model call, so losing the
    call must not lose them - and it must not silently promote them either: an unchallenged
    candidate is reported as unchallenged, never as one that survived review.
    """
    return tuple(ChallengedCandidate(target=target) for target in targets)


def _partition_citations(
    proposal: ChallengeProposal, *, admissible: frozenset[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split cited ids into those admissible **against this candidate** and those that are not.

    Exact string match, no normalization: an id that differs by a character is not a near
    miss to be repaired, it is an id the model did not receive, and repairing one would be the
    system inventing a citation on the model's behalf.

    The per-candidate check matters as much as the existence check. A real evidence id
    belonging to a different candidate would support an unrelated claim, which is a fabricated
    citation assembled out of genuine parts.
    """
    kept: list[str] = []
    borrowed: list[str] = []
    for cited in proposal.contradicting_evidence_ids:
        (kept if cited in admissible else borrowed).append(cited)
    # Deduplicated and ordered, so two runs over one proposal produce one result.
    return tuple(sorted(set(kept))), tuple(borrowed)
