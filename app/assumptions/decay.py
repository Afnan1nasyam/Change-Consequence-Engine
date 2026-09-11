"""The knowledge-decay report, built from the coupling index and nothing else.

``IDEA.md`` 15.5 makes this a first-class deliverable rather than a side effect, and the
argument is economic: it takes **no change request as input**, predicts nothing about the
future, and is verifiable on delivery. It is also the earliest honest test of whether the
corpus can support the product at all.

The report is a **projection**. Every row that accuses an artifact of being wrong is backed
by a rule the resolver already ran; this module never decides that something is
inconsistent, it only groups and renders what the resolver derived. That is why
:class:`DecayEntry` refuses an inconsistency row without a ``rule_id`` and refuses a
missing-knowledge row *with* one.

Sections, and the three-way separation they preserve:

=================================== ==========================================
``stale_consumer``                  known inconsistency - a dependent is wrong
``unauthorized_override``           known inconsistency - a governance defect
``contradictory_declarations``      known inconsistency - two authorities clash
``ambiguous_authority``             known ambiguity - the corpus cannot decide
``consumer_divergence``             known ambiguity - neither side is authority
``registry_gap``                    missing knowledge - unregistered vocabulary
``single_holder``                   missing knowledge - one point of knowledge
``unwritten_rule``                  missing knowledge - config without policy
=================================== ==========================================

Absence is never reported as contradiction. A key with one holder is not a conflict, an
unresolvable authority is not a defect in either candidate, and an unregistered key is a gap
in *our* vocabulary rather than an error in the document.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.assumptions.coupling import CouplingIndex
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.coupling import AuthorityResolution, ResolvedBinding
from app.domain.entities.decay import DecayEntry, DecayParticipant, KnowledgeDecayReport
from app.domain.entities.registry import KeyRegistryGap
from app.domain.value_objects.enums import (
    AmbiguityReason,
    AnomalyKind,
    AuthorityClass,
    BindingRole,
    DecayCategory,
)
from app.domain.value_objects.rules import RuleId
from app.domain.value_objects.scope import ScopeSelector

#: Authority classes that constitute *governance*. A key bound only outside this set is an
#: unwritten rule: the system behaves a certain way and no policy or runbook says so.
_GOVERNING: frozenset[AuthorityClass] = frozenset(
    {
        AuthorityClass.REGULATORY,
        AuthorityClass.POLICY,
        AuthorityClass.STANDARD,
        AuthorityClass.SOP_RUNBOOK,
    }
)

#: Which decay category each inconsistency rule reports under.
_RULE_CATEGORY = {
    RuleId.STALE_CONSUMER: DecayCategory.STALE_CONSUMER,
    RuleId.UNAUTHORIZED_OVERRIDE: DecayCategory.UNAUTHORIZED_OVERRIDE,
    RuleId.SAME_SCOPE_CONTRADICTION: DecayCategory.CONTRADICTORY_DECLARATIONS,
}


def build_report(
    index: CouplingIndex,
    *,
    tenant_id: str,
    findings: Sequence[ClassAFinding] = (),
    gaps: Iterable[KeyRegistryGap] = (),
) -> KnowledgeDecayReport:
    """Assemble the standing report. Deterministic, and free of any model call.

    Reproducible from the same index, the same findings and the same gap set: entries are
    emitted in a fixed order so two reports over identical inputs are byte-identical, and a
    diff between two runs is therefore attributable to a registry expansion or an
    authority-ordering change rather than to iteration order.
    """
    entries: list[DecayEntry] = []

    entries.extend(_inconsistencies(index, findings))
    entries.extend(_ambiguities(index))
    entries.extend(_consumer_anomalies(index))
    entries.extend(_single_holders(index))
    entries.extend(_unwritten_rules(index))
    entries.extend(_registry_gaps(gaps))

    return KnowledgeDecayReport(
        tenant_id=tenant_id,
        as_of=index.as_of,
        entries=tuple(entries),
        keys_examined=len(index.keys),
        bindings_examined=index.binding_count,
        registry_version=index.registry_version,
        coupling_index_version=index.coupling_index_version,
        authority_model_version=index.authority_model_version,
    )


# ---------------------------------------------------------------------------
# Known inconsistencies - derived, and each row names its rule
# ---------------------------------------------------------------------------


def _inconsistencies(
    index: CouplingIndex, findings: Sequence[ClassAFinding]
) -> tuple[DecayEntry, ...]:
    """Sections A, B and C: rows where a deterministic rule says an artifact is wrong.

    Grouped by ``relation_id`` rather than by finding. A symmetric ``conflicts_with``
    produces two findings - one per declarer, so either author sees it - but it is **one**
    inconsistency in the corpus, and reporting it twice would double the apparent defect
    count.
    """
    by_assumption = {b.assumption_id: b for b in index.bindings}
    by_relation: dict[str, list[ClassAFinding]] = {}
    for finding in findings:
        by_relation.setdefault(finding.relation_id, []).append(finding)

    entries: list[DecayEntry] = []
    for relation_id in sorted(by_relation):
        group = sorted(by_relation[relation_id], key=lambda f: f.finding_id)
        head = group[0]
        category = _RULE_CATEGORY.get(head.rule_id)
        if category is None:
            # A delta invalidation is a consequence of a *proposed change*, not standing
            # corpus decay. The report answers "what is already wrong", so it stays out.
            continue

        entries.append(
            DecayEntry(
                category=category,
                key=head.assumption_key,
                scope=head.scope,
                summary=head.summary,
                participants=tuple(
                    _participant(by_assumption[record_id])
                    for record_id in sorted(head.input_record_ids)
                    if record_id in by_assumption
                ),
                finding_id=head.finding_id,
                rule_id=head.rule_id,
                requires_human_decision=False,
            )
        )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Known ambiguity - no winner is named
# ---------------------------------------------------------------------------


def _ambiguities(index: CouplingIndex) -> tuple[DecayEntry, ...]:
    """Section D. Report the candidates and what would resolve them; pick nobody.

    Naming a winner here would be newest-wins by another route. The report's job is to hand
    the decision to a human with everything they need to make it.
    """
    entries: list[DecayEntry] = []
    for resolution in index.ambiguous_resolutions:
        entries.append(
            DecayEntry(
                category=DecayCategory.AMBIGUOUS_AUTHORITY,
                key=resolution.key,
                scope=resolution.scope,
                summary=_ambiguity_summary(resolution),
                participants=tuple(
                    _participant(candidate) for candidate in resolution.tied_candidates
                ),
                requires_human_decision=True,
                what_would_resolve=_ambiguity_resolution(resolution),
            )
        )
    return tuple(entries)


def _ambiguity_summary(resolution: AuthorityResolution) -> str:
    values = ", ".join(
        f"{c.source_title} ({c.authority_class.value}) = {c.value.render()}"
        for c in resolution.tied_candidates
    )
    return (
        f"No artifact can be shown to govern {resolution.key} at "
        f"{resolution.scope.render()} on {resolution.as_of.date().isoformat()}: {values}. "
        "The corpus does not establish an authority, and none has been chosen."
    )


def _ambiguity_resolution(resolution: AuthorityResolution) -> str:
    if resolution.ambiguity_reason is AmbiguityReason.INCOMPARABLE_VALUES:
        return (
            "The candidate values cannot be compared. Confirm the registry entry declares "
            "the right type and unit for this key, then re-ingest the affected sources."
        )
    return (
        "Two or more declarers share the top authority class with overlapping validity and "
        "divergent values. Resolve by narrowing one document's scope, ending one "
        "document's validity interval, or re-registering one holder at a different "
        "authority class. The system will not choose between them."
    )


def _consumer_anomalies(index: CouplingIndex) -> tuple[DecayEntry, ...]:
    """Two consumers disagreeing with no authority above them.

    Reported as ambiguity, not inconsistency: neither artifact declares, so neither can be
    said to be wrong. Requires scope review (checkpoint 3.3).
    """
    by_assumption = {b.assumption_id: b for b in index.bindings}
    entries: list[DecayEntry] = []
    for anomaly in index.anomalies:
        if anomaly.kind is not AnomalyKind.CONSUMER_DIVERGENCE:
            continue
        participants = [by_assumption[pid] for pid in anomaly.participants if pid in by_assumption]
        entries.append(
            DecayEntry(
                category=DecayCategory.CONSUMER_DIVERGENCE,
                key=anomaly.key,
                scope=(participants[0].scope if participants else ScopeSelector.global_scope()),
                summary=anomaly.detail,
                participants=tuple(_participant(p) for p in participants),
                requires_human_decision=True,
                what_would_resolve=(
                    "Establish which scope each consumer belongs to, or register a "
                    "declaring artifact so the value has an authority. Until then neither "
                    "consumer can be shown to be the stale one."
                ),
            )
        )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Missing knowledge - recorded, never derived
# ---------------------------------------------------------------------------


def _single_holders(index: CouplingIndex) -> tuple[DecayEntry, ...]:
    """Keys exactly one artifact knows about.

    Undocumented coupling and a single point of knowledge. Not a defect and not a
    contradiction - which is why the row carries no rule id and no finding.
    """
    entries: list[DecayEntry] = []
    for key in index.keys:
        holders = [b for b in index.bindings if b.key == key and b.is_in_force(index.as_of)]
        if len(holders) != 1:
            continue
        only = holders[0]
        entries.append(
            DecayEntry(
                category=DecayCategory.SINGLE_HOLDER,
                key=key,
                scope=only.scope,
                summary=(
                    f"{key} is bound by exactly one artifact, {only.source_title} "
                    f"({only.authority_class.value}, role {only.role.value}). Nothing "
                    "corroborates it and nothing depends on it in writing, so a change "
                    "here would leave no trace anywhere else in the corpus."
                ),
                participants=(_participant(only),),
                requires_human_decision=False,
            )
        )
    return tuple(entries)


def _unwritten_rules(index: CouplingIndex) -> tuple[DecayEntry, ...]:
    """Keys bound by configuration, code or informal notes but by no governing artifact.

    The system behaves a particular way and no policy, standard or runbook says it should.
    """
    entries: list[DecayEntry] = []
    for key in index.keys:
        holders = [b for b in index.bindings if b.key == key and b.is_in_force(index.as_of)]
        if not holders:
            continue
        if any(b.authority_class in _GOVERNING for b in holders):
            continue
        if not any(b.role is BindingRole.DECLARES for b in holders):
            # Nothing declares it at all, so there is no rule to be unwritten - only a
            # dependency with no source, which the single-holder row already covers.
            continue
        entries.append(
            DecayEntry(
                category=DecayCategory.UNWRITTEN_RULE,
                key=key,
                scope=holders[0].scope,
                summary=(
                    f"{key} is declared only by non-governing artifacts "
                    f"({', '.join(sorted({b.authority_class.value for b in holders}))}). "
                    "The behaviour is real but no policy, standard or runbook states it."
                ),
                participants=tuple(
                    _participant(b) for b in sorted(holders, key=lambda x: x.assumption_id)
                ),
                requires_human_decision=False,
            )
        )
    return tuple(entries)


def _registry_gaps(gaps: Iterable[KeyRegistryGap]) -> tuple[DecayEntry, ...]:
    """Section E. Where the closed registry needs to grow.

    Gap volume is a product signal, not an error log (checkpoint 4.1.7). Each row is a
    candidate for registration, and registering one promotes its consequences to Class A on
    the next run - the intended growth path.
    """
    entries: list[DecayEntry] = []
    for gap in sorted(gaps, key=lambda g: (g.proposed_key, g.gap_id)):
        entries.append(
            DecayEntry(
                category=DecayCategory.REGISTRY_GAP,
                key=gap.proposed_key,
                summary=(
                    f"{gap.proposed_key} was proposed {gap.occurrence_count} time(s) but is "
                    f"not in registry {gap.registry_version}. It is quarantined and takes "
                    "no part in coupling. Registering it would promote its consequences to "
                    "Class A on the next run."
                ),
                participants=(
                    DecayParticipant(
                        source_version_id=gap.holder_source_version_id,
                        artifact=gap.proposed_key,
                        rendered_value=gap.proposed_value.render(),
                    ),
                ),
                requires_human_decision=False,
            )
        )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _participant(binding: ResolvedBinding) -> DecayParticipant:
    return DecayParticipant(
        source_version_id=binding.source_version_id,
        artifact=binding.source_title,
        assumption_id=binding.assumption_id,
        role=binding.role,
        authority_class=binding.authority_class,
        rendered_value=binding.value.render(),
        effective_from=binding.record.effective_from,
        effective_to=binding.record.effective_to,
    )
