"""Who the specialists are, what evidence each reads, and which of them a change wakes up.

Deterministic. No model call decides the fan-out, because "which specialists are relevant"
would otherwise be a judgement the system pays for before it has any evidence to judge with.

The slice, not the persona
--------------------------

``CLAUDE.md`` 17: *analysis diversity comes from differing inputs, not differing personas.*
Every specialist definition says the same thing in its own words - "you receive **your own
evidence slice**... several agents reasoning over identical context produce correlated findings
that then masquerade as corroboration". So the only thing that makes the seven different here
is :attr:`Specialist.source_types`: the source types their slice is biased toward.

Overlap is real and is not hidden. ``risk-analyst`` and ``historical-analyst`` share incidents
and tickets, because that genuinely is both of their evidence bases - the risk analyst asks
what control was missing, the historical analyst asks what outcome was recorded. What makes
that safe is not this table: corroboration counts distinct ``source_version_id`` values, never
agreeing agents (checkpoint 9.3), so two specialists citing one document is one source however
the roster is drawn.

Selection
---------

A specialist is invoked **iff its slice is non-empty**. That is the whole rule, and it is a
fact about the authorized evidence rather than a guess about the change:

* a specialist with nothing to cite can only speculate, which its own definition forbids;
* paying for a provider call to be told "my slice is empty" is the cost the cheap gate exists
  to avoid (``CLAUDE.md``, cost discipline).

``ChangeSpec.affected_domains`` is deliberately **not** a gate. It is analyst-supplied prose,
so gating on it would let a mislabelled change silence a specialist that had evidence - and the
evidence is the harder test anyway. It is passed to the specialists as context.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from app.domain.value_objects.enums import SourceType

#: Bumped when the roster or the selection rule changes: both change which agents ran and
#: therefore what an analysis contains without any agent changing.
ROSTER_VERSION: Final = "specialist-roster-1"

#: Spans handed to one specialist. Bounded for the reason every other slice is: an agent shown
#: the whole corpus is summarizing it rather than reasoning about a change.
MAX_SLICE_SPANS: Final = 12

#: How deep each specialist's own query reaches. Its slice is source-type filtered, so the
#: same ``top_k`` as the main query would usually return everything of that type - which is
#: what a domain specialist wants.
SLICE_TOP_K: Final = 12


@dataclass(frozen=True)
class Specialist:
    """One analyst: its agent name, its evidence basis, and the category it reports under.

    ``category`` lands on :attr:`~app.domain.entities.findings.SpecialistFinding.category`, so
    a reader can group findings by domain without parsing a claim. It is fixed per specialist
    rather than proposed, for the same reason ``consequence_class`` is pipeline-assigned: a
    label the agent could choose is a label that tells you nothing.
    """

    agent: str
    category: str
    source_types: frozenset[SourceType]
    #: What this analyst is for, one line, rendered into its task contract. Not a persona -
    #: the definition already carries the role; this is the *ask* for this analysis.
    focus: str


#: The seven, in the order the fan-out reports them. Source types are grounded in the corpus
#: the system actually ingests rather than in the full enum: a specialist pointed at a source
#: type nothing produces would always be un-selected, which reads like a defect.
ROSTER: Final = (
    Specialist(
        agent="policy-analyst",
        category="policy_governance",
        source_types=frozenset({SourceType.POLICY}),
        focus="policy, governance, authorization and effective-date consequences",
    ),
    Specialist(
        agent="process-analyst",
        category="process_operations",
        source_types=frozenset({SourceType.SOP, SourceType.RUNBOOK, SourceType.KNOWLEDGE_DOC}),
        focus="SOP steps, approval chains, handoffs, SLAs and queue effects",
    ),
    Specialist(
        agent="technology-analyst",
        category="technology_contracts",
        source_types=frozenset({SourceType.CODE, SourceType.CONFIG, SourceType.UI_TEXT}),
        focus="configuration, integration contracts, automation and code/document mismatches",
    ),
    Specialist(
        agent="people-analyst",
        category="people_responsibilities",
        source_types=frozenset(
            {SourceType.TRAINING, SourceType.ORG_METADATA, SourceType.EMAIL_TEMPLATE}
        ),
        focus="roles, approval rights, training material and instructions that go stale",
    ),
    Specialist(
        agent="data-analyst",
        category="data_reporting",
        source_types=frozenset({SourceType.DASHBOARD}),
        focus="metric definitions, reporting logic, retention and historical comparability",
    ),
    Specialist(
        agent="risk-analyst",
        category="risk_controls",
        source_types=frozenset({SourceType.INCIDENT, SourceType.TICKET}),
        focus="control gaps, failure modes and compensating controls",
    ),
    Specialist(
        agent="historical-analyst",
        category="historical_analogue",
        source_types=frozenset({SourceType.INCIDENT, SourceType.TICKET, SourceType.CHANGE_REQUEST}),
        focus="prior changes and incidents with a **recorded observed outcome**",
    ),
)

#: The agent names, for a caller that needs to load exactly these definitions.
SPECIALIST_AGENTS: Final = tuple(s.agent for s in ROSTER)

BY_AGENT: Final = {s.agent: s for s in ROSTER}

#: Why a specialist was not invoked. Reported per specialist, because "did not run" and
#: "ran and found nothing" are different facts about an analysis.
NOT_SELECTED_EMPTY_SLICE: Final = "no authorized evidence of this specialist's source types"
NOT_SELECTED_NO_DEFINITION: Final = "no agent definition was supplied for this specialist"


@dataclass(frozen=True)
class SpecialistSlice:
    """One specialist's evidence, and where it came from.

    Held as ids plus text rather than as ``EvidenceCandidate``s for the reason every other
    prompt projection is: the agent needs an id it can cite and text it can read, and a match
    must be attributable to words inside the citation.
    """

    specialist: Specialist
    spans: tuple[SliceSpan, ...] = ()
    #: Spans the specialist's own query returned beyond the span budget.
    truncated: int = 0

    @property
    def agent(self) -> str:
        return self.specialist.agent

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(span.evidence_id for span in self.spans)

    @property
    def is_empty(self) -> bool:
        return not self.spans


@dataclass(frozen=True)
class SliceSpan:
    """One span as a specialist sees it."""

    evidence_id: str
    text: str
    source_type: str
    authority_class: str
    source_title: str
    assumption_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class FanOut:
    """The selection decision, per specialist, with the reason for every exclusion."""

    selected: tuple[SpecialistSlice, ...] = ()
    #: ``(agent, reason)`` for every specialist that will not be invoked.
    not_selected: tuple[tuple[str, str], ...] = ()
    roster_version: str = ROSTER_VERSION

    @property
    def agents(self) -> tuple[str, ...]:
        return tuple(item.agent for item in self.selected)

    @property
    def slice_sizes(self) -> dict[str, int]:
        return {item.agent: len(item.spans) for item in self.selected}

    @property
    def reasons(self) -> dict[str, str]:
        return dict(self.not_selected)

    @property
    def shared_evidence_ids(self) -> frozenset[str]:
        """Ids more than one selected specialist can cite.

        Reported rather than prevented. Two specialists citing one document is one source -
        ``ConfidenceInputs`` counts distinct source versions - so overlap costs a little
        duplicated reading and cannot manufacture corroboration.
        """
        seen: set[str] = set()
        shared: set[str] = set()
        for item in self.selected:
            for evidence_id in item.evidence_ids:
                (shared if evidence_id in seen else seen).add(evidence_id)
        return frozenset(shared)


def select(slices: Iterable[SpecialistSlice], *, available: Sequence[str] | None = None) -> FanOut:
    """Invoke the specialists whose slice has something in it, and say why the rest did not.

    ``available`` is the set of agent names a definition was loaded for. A specialist with
    evidence but no definition is excluded with its own reason rather than crashing the stage:
    a renamed ``.md`` should cost one analyst, not the analysis.
    """
    loadable = None if available is None else set(available)
    chosen: list[SpecialistSlice] = []
    excluded: list[tuple[str, str]] = []

    by_agent = {item.agent: item for item in slices}
    for specialist in ROSTER:
        found = by_agent.get(specialist.agent)
        if found is None or found.is_empty:
            excluded.append((specialist.agent, NOT_SELECTED_EMPTY_SLICE))
            continue
        if loadable is not None and specialist.agent not in loadable:
            excluded.append((specialist.agent, NOT_SELECTED_NO_DEFINITION))
            continue
        chosen.append(found)

    return FanOut(selected=tuple(chosen), not_selected=tuple(excluded))
