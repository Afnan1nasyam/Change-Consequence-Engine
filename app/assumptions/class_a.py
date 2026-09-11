"""Class A generation by producer, and the Class B boundary that protects it.

Checkpoint 7.2 states the decision rule in one line:

    **A consequence is Class A iff the deterministic resolver produced it.**

Classification is by producer, not by subject matter. That is the whole defence against the
agent layer rediscovering Class A consequences merely because they are easy to verbalize:
*sounding derivable is not a derivation*. This module is the only producer of
:class:`ClassAFinding`, it takes no agent input, and ``consequence_class`` is a ``Literal``
on the entity - so there is no path, prompt-mediated or otherwise, by which a model can
self-declare Class A.

**Evidence materialization, and the line it must not cross.** A ``DerivedRelation`` carries
no ``evidence_ids`` of its own, and the schema refuses them. Its evidence set is the
*computed transitive union* of its input records' evidence, materialized here onto the
finding and flagged ``evidence_is_derived_union=True``. The distinction is not pedantry:
nominating an arbitrary input span as direct support for a derivation would manufacture
false provenance, which is the failure this project exists to prevent (checkpoint 2.4).

**Class B deduplication is structural, in three layers** (checkpoint 7.5). Prompt wording
would not hold this line:

1. ``consequence_class`` is pipeline-assigned - see above;
2. output-side **merge**, not append: a hypothesis whose ``(assumption_key,
   affected_artifact)`` pair is already in the Class A set is absorbed and re-attributed to
   the resolver, its mechanism text kept as articulation;
3. a visible ``class_b_rediscovery_rate``. A high value is a **warning** that the hunter's
   evidence slice is misconfigured, not evidence that it is being productive.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

from app.domain.entities.claims import ConsequenceHypothesis, DerivedRelation
from app.domain.entities.class_a import ClassAFinding, ClassBMergeOutcome
from app.domain.entities.coupling import CouplingStrength, ResolvedBinding
from app.domain.value_objects.enums import AUTHORITY_MODEL_VERSION
from app.domain.value_objects.rules import COUPLING_INDEX_VERSION, RuleId
from app.domain.value_objects.taxonomy import Relationship
from app.ingestion.identity import class_a_finding_id_for


def materialize_evidence(
    relation: DerivedRelation, by_assumption: dict[str, ResolvedBinding]
) -> tuple[str, ...]:
    """The transitive union of the input records' evidence, in a stable order.

    Computed on read rather than stored on the relation. The relation itself must stay
    evidence-free: a derivation's support is its rule and its inputs, and a citation
    attached directly to it would claim a span that says something the relation does not.
    """
    found: set[str] = set()
    for record_id in relation.input_record_ids:
        binding = by_assumption.get(record_id)
        if binding is not None:
            found.add(binding.evidence_id)
    return tuple(sorted(found))


def derive_class_a(
    relations: Iterable[DerivedRelation],
    *,
    bindings: Sequence[ResolvedBinding],
    tenant_id: str,
    registry_version: str,
    strengths: dict[str, CouplingStrength] | None = None,
) -> tuple[ClassAFinding, ...]:
    """Turn derived relations into Class A findings, one per affected artifact.

    A symmetric ``conflicts_with`` yields **two** findings, one per declarer: both authors
    have something to reconcile, and a reviewer filtering by document must see the conflict
    from either side. The directed relations yield one, aimed at the artifact that is
    actually wrong - the narrower deviating document for an override, the stale consumer for
    an invalidation.

    No ``severity`` and no ``confidence`` are produced anywhere in this function. Severity
    is judgment the corpus does not contain (checkpoint 9.4) and confidence is computed by a
    versioned function elsewhere; a resolver stamping either would launder a guess into the
    trust-bearing tier.
    """
    by_assumption = {b.assumption_id: b for b in bindings}
    findings: dict[str, ClassAFinding] = {}

    for relation in relations:
        evidence = materialize_evidence(relation, by_assumption)
        if not evidence:
            # Every input record has gone from the pool. Emitting a finding whose evidence
            # set is empty would assert a derivation nobody can check.
            continue

        # Resolve the two sides from *this relation's own input records*, not from a
        # global version map. One document version routinely holds several bindings - the
        # EMEA notes in the sample corpus hold a global refund threshold and an EU
        # retention window - so a version-keyed lookup silently attaches the wrong scope
        # and the wrong value to the finding.
        participants = {
            by_assumption[record_id].source_version_id: by_assumption[record_id]
            for record_id in relation.input_record_ids
            if record_id in by_assumption
        }

        for affected_id, counterpart_id in _affected_pairs(relation):
            affected = participants.get(affected_id)
            if affected is None:
                continue
            counterpart = participants.get(counterpart_id) if counterpart_id else None

            finding_id = class_a_finding_id_for(
                relation_id=relation.relation_id, affected_version_id=affected_id
            )
            findings[finding_id] = ClassAFinding(
                finding_id=finding_id,
                tenant_id=tenant_id,
                relation_id=relation.relation_id,
                relationship=relation.relationship,
                rule_id=relation.provenance.rule_id,
                assumption_key=str(relation.provenance.detail.get("assumption_key", "")),
                scope=affected.scope,
                as_of=_as_of_of(relation),
                affected_source_version_id=affected.source_version_id,
                affected_source_id=affected.source_id,
                affected_artifact=affected.source_title,
                counterpart_source_version_id=(
                    counterpart.source_version_id if counterpart else None
                ),
                counterpart_artifact=counterpart.source_title if counterpart else None,
                summary=_summary(relation, affected=affected, counterpart=counterpart),
                observed_value=affected.value,
                authoritative_value=counterpart.value if counterpart else None,
                input_record_ids=relation.input_record_ids,
                delta_id=_delta_id_of(relation),
                materialized_evidence_ids=evidence,
                strength=(strengths or {}).get(relation.relation_id),
                rule_version=relation.provenance.rule_version,
                registry_version=registry_version,
                coupling_index_version=COUPLING_INDEX_VERSION,
                authority_model_version=(
                    relation.provenance.authority_model_version or AUTHORITY_MODEL_VERSION
                ),
            )

    return tuple(findings[key] for key in sorted(findings))


def merge_class_b(
    hypotheses: Iterable[ConsequenceHypothesis],
    findings: Sequence[ClassAFinding],
    *,
    affected_artifact_of: dict[str, str] | None = None,
) -> tuple[ClassBMergeOutcome, tuple[ClassAFinding, ...]]:
    """Reconcile agentic hypotheses against the Class A set (checkpoint 7.5.2).

    A hypothesis is suppressed when its ``(assumption_key, affected_artifact)`` pair already
    exists in the Class A set. It is **merged**, not dropped: the finding records the
    absorbed hypothesis id so the model's articulated mechanism stays available. What must
    not survive is a second, independent finding claiming the same thing at a trust level
    the model did not earn.

    ``affected_artifact_of`` maps a hypothesis id to the source version it concerns.
    Hypotheses have no such field of their own - they name entities in prose - so the caller
    that assembled the hypothesis supplies the mapping. A hypothesis with no mapping is
    retained: without knowing which artifact it concerns, we cannot know it is a duplicate,
    and the default rule is that ambiguity demotes to Class B (checkpoint 7.4).
    """
    # One artifact can be both a peer in a same-scope contradiction and the deviating side
    # of an unauthorized override on the same key, so a `(key, artifact)` pair is not unique
    # across findings. The lowest finding id wins, deterministically: without a stable tie
    # break the absorbing finding would depend on iteration order, and two runs over the
    # same store would attribute the same mechanism text to different findings.
    resolved_pairs: dict[tuple[str, str], str] = {}
    for finding in sorted(findings, key=lambda f: f.finding_id):
        resolved_pairs.setdefault(finding.dedupe_pair, finding.finding_id)

    lookup = affected_artifact_of or {}

    suppressed: list[str] = []
    retained: list[str] = []
    merged_into: dict[str, str] = {}
    absorbed: dict[str, list[str]] = {}

    for hypothesis in hypotheses:
        artifact = lookup.get(hypothesis.hypothesis_id)
        target: str | None = None
        if artifact is not None:
            for key in hypothesis.assumption_keys:
                candidate = resolved_pairs.get((key, artifact))
                if candidate is not None:
                    target = candidate
                    break

        if target is None:
            retained.append(hypothesis.hypothesis_id)
            continue

        suppressed.append(hypothesis.hypothesis_id)
        merged_into[hypothesis.hypothesis_id] = target
        absorbed.setdefault(target, []).append(hypothesis.hypothesis_id)

    outcome = ClassBMergeOutcome(
        suppressed_hypothesis_ids=tuple(suppressed),
        retained_hypothesis_ids=tuple(retained),
        merged_into=merged_into,
        considered=len(suppressed) + len(retained),
    )

    updated = tuple(
        finding.model_copy(
            update={
                "merged_hypothesis_ids": tuple(
                    sorted(set(finding.merged_hypothesis_ids) | set(absorbed[finding.finding_id]))
                )
            }
        )
        if finding.finding_id in absorbed
        else finding
        for finding in findings
    )
    return outcome, updated


# ---------------------------------------------------------------------------
# Per-relationship shaping
# ---------------------------------------------------------------------------


def _affected_pairs(relation: DerivedRelation) -> tuple[tuple[str, str | None], ...]:
    """``(affected artifact, counterpart)`` pairs this relation produces findings for."""
    match relation.relationship:
        case Relationship.CONFLICTS_WITH:
            # Symmetric: neither is the subject, so both authors are told.
            return (
                (relation.subject_ref, relation.object_ref),
                (relation.object_ref, relation.subject_ref),
            )
        case Relationship.UNAUTHORIZED_OVERRIDE:
            # Directed narrower -> broader. The narrower artifact is the one deviating.
            return ((relation.subject_ref, relation.object_ref),)
        case Relationship.INVALIDATES_ASSUMPTION:
            # Directed authority/change -> dependent. The dependent is what needs fixing.
            subject = relation.subject_ref
            counterpart = None if subject.startswith("dl_") else subject
            return ((relation.object_ref, counterpart),)
        case _:  # pragma: no cover - DerivedRelation admits only the three above
            return ()


def _summary(
    relation: DerivedRelation,
    *,
    affected: ResolvedBinding,
    counterpart: ResolvedBinding | None,
) -> str:
    """One sentence a reviewer can act on, built from recorded facts only."""
    detail = relation.provenance.detail
    key = detail.get("assumption_key", "this key")

    match relation.provenance.rule_id:
        case RuleId.SAME_SCOPE_CONTRADICTION:
            other = counterpart.source_title if counterpart else "another declaration"
            other_value = counterpart.value.render() if counterpart else "a different value"
            return (
                f"{affected.source_title} ({affected.authority_class.value}) declares "
                f"{key} = {affected.value.render()} at {affected.scope.render()}, while "
                f"{other} declares {other_value} at the same scope with overlapping "
                "validity. Two authorities disagree; one must be corrected."
            )
        case RuleId.UNAUTHORIZED_OVERRIDE:
            return (
                f"{affected.source_title} ({detail.get('narrower_authority_class')}) narrows "
                f"{key} to {affected.value.render()} at {affected.scope.render()}, "
                f"overriding {detail.get('broader_artifact')} "
                f"({detail.get('broader_authority_class')}) which declares "
                f"{detail.get('broader_value')} at {detail.get('broader_scope')}. The "
                "narrower artifact does not carry the authority to override the broader one."
            )
        case RuleId.STALE_CONSUMER:
            return (
                f"{affected.source_title} depends on {key} = {affected.value.render()}, but "
                f"{detail.get('authoritative_artifact')} "
                f"({detail.get('authoritative_authority_class')}) governs "
                f"{affected.scope.render()} and declares "
                f"{detail.get('authoritative_value')}. The dependent artifact is stale."
            )
        case RuleId.DELTA_INVALIDATION:
            return (
                f"{affected.source_title} binds {key} = {affected.value.render()} "
                f"({affected.role.value}) at {affected.scope.render()}, which the proposed "
                f"change to {detail.get('new_value')} invalidates: {detail.get('reason')}."
            )
        case _:  # pragma: no cover
            return f"{affected.source_title} is affected by a derived relation on {key}."


def _as_of_of(relation: DerivedRelation) -> dt.datetime:
    """The evaluation date recorded on the derivation.

    Read back from provenance rather than taken from the caller so that a persisted finding
    and a freshly derived one agree: the date that produced the finding is the date the
    finding reports.
    """
    raw = relation.provenance.detail.get("as_of")
    if isinstance(raw, str):
        return dt.datetime.fromisoformat(raw)
    if isinstance(raw, dt.datetime):
        return raw
    return relation.created_at  # pragma: no cover - every rule records as_of


def _delta_id_of(relation: DerivedRelation) -> str | None:
    raw = relation.provenance.detail.get("delta_id")
    return raw if isinstance(raw, str) else None
