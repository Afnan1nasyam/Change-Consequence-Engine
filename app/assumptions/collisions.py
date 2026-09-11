"""Role-and-scope-aware collision detection. The checkpoint 3.3 matrix, in code.

This module is where the product's central claim either holds or collapses into noise. The
matrix it implements:

======================================= ==================================================
``declares`` vs ``declares``, divergent three-way by scope and authority (5.4)
``declares`` vs ``consumes``, divergent staleness -> ``invalidates_assumption``
``consumes`` vs ``consumes``, divergent **not** a contradiction; a reportable anomaly
anything vs ``describes``                **no finding**, ever
======================================= ==================================================

Three decisions in here are the ones most easily got wrong, so they are stated rather than
left to the reader:

**Staleness is evaluated against resolved authority, not pairwise.** The naive reading of
"declares vs consumes, divergent" would compare a consumer against *every* declarer. On the
sample corpus that reports the training deck stale because a lower-authority operations
standard disagrees with the policy the deck correctly follows - a false positive in the
highest-trust tier, produced by comparing against a document that does not govern. So a
consumer is stale iff its value differs from the value ``resolve_authority`` returns for
its scope. The rule is still deterministic and its provenance still names exactly two
records.

**Ambiguous authority produces no staleness finding.** If the corpus cannot say what the
value is, it cannot say a consumer has the wrong one. The ambiguity is reported instead.
Manufacturing a finding out of ambiguity is the same error as calling absence a
contradiction.

**A legitimate override emits nothing.** It is recorded (:class:`OverrideRecord`) so a
later invalidation pass can ask for confirmation rather than report breakage, but it
produces no relation and no edge. Reporting correct behaviour as a defect is the fastest
way to lose a reader who can see it is correct.

**A single source cannot manufacture a cross-source contradiction.** Two divergent
declarations inside one source version are reported as an intra-document inconsistency -
an anomaly, not a ``conflicts_with`` relation. This is a security property, not tidiness:
without it a planted document that states a value twice would surface in the highest-trust
tier as a governance conflict between two independent authorities
(``docs/SECURITY.md``).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.assumptions.temporal import resolve_authority
from app.domain.entities.claims import DerivedRelation
from app.domain.entities.coupling import (
    AuthorityResolution,
    CouplingAnomaly,
    OverrideRecord,
    ResolvedBinding,
)
from app.domain.services import scope_lattice
from app.domain.services.temporal_intervals import overlaps
from app.domain.services.value_comparison import are_divergent, is_blocked
from app.domain.value_objects.enums import (
    AUTHORITY_MODEL_VERSION,
    AnomalyKind,
    BindingRole,
)
from app.domain.value_objects.provenance import DerivationProvenance
from app.domain.value_objects.rules import COUPLING_INDEX_VERSION, RuleId
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from app.ingestion.identity import relation_id_for


@dataclass(frozen=True)
class KeyCollisions:
    """Everything the resolver deterministically concluded about one key at one date."""

    key: str
    as_of: dt.datetime
    relations: tuple[DerivedRelation, ...] = ()
    overrides: tuple[OverrideRecord, ...] = ()
    anomalies: tuple[CouplingAnomaly, ...] = ()
    #: One resolution per distinct scope observed for this key, keyed by rendered scope.
    authority: dict[str, AuthorityResolution] = field(default_factory=dict)

    def relations_of(self, relationship: Relationship) -> tuple[DerivedRelation, ...]:
        return tuple(r for r in self.relations if r.relationship is relationship)


def collide_key(
    bindings: Iterable[ResolvedBinding],
    *,
    key: str,
    as_of: dt.datetime,
    registry_version: str,
) -> KeyCollisions:
    """Apply the full matrix to every binding of ``key`` that is in force at ``as_of``."""
    live = tuple(
        sorted(
            (b for b in bindings if b.key == key and b.is_in_force(as_of)),
            key=lambda b: b.assumption_id,
        )
    )
    if not live:
        return KeyCollisions(key=key, as_of=as_of)

    authority = {
        scope.render(): resolve_authority(live, key=key, scope=scope, as_of=as_of)
        for scope in _observed_scopes(live)
    }

    declarer_relations, overrides, declarer_anomalies = _declarer_collisions(
        live, key=key, as_of=as_of, registry_version=registry_version
    )
    stale_relations = _stale_consumers(
        live, key=key, as_of=as_of, registry_version=registry_version, authority=authority
    )
    consumer_anomalies = _consumer_divergence(live, key=key, as_of=as_of)

    return KeyCollisions(
        key=key,
        as_of=as_of,
        relations=_dedupe(declarer_relations + stale_relations),
        overrides=overrides,
        anomalies=declarer_anomalies + consumer_anomalies,
        authority=authority,
    )


def _observed_scopes(live: Sequence[ResolvedBinding]) -> tuple[ScopeSelector, ...]:
    """Distinct scopes among the live bindings, narrowest last for stable output."""
    seen: dict[str, ScopeSelector] = {}
    for binding in live:
        seen.setdefault(binding.scope.render(), binding.scope)
    return tuple(seen[name] for name in sorted(seen))


# ---------------------------------------------------------------------------
# declares vs declares - the three-way rule (checkpoint 5.4)
# ---------------------------------------------------------------------------


def _declarer_collisions(
    live: Sequence[ResolvedBinding],
    *,
    key: str,
    as_of: dt.datetime,
    registry_version: str,
) -> tuple[tuple[DerivedRelation, ...], tuple[OverrideRecord, ...], tuple[CouplingAnomaly, ...]]:
    declarers = [b for b in live if b.role is BindingRole.DECLARES]
    relations: list[DerivedRelation] = []
    overrides: list[OverrideRecord] = []
    anomalies: list[CouplingAnomaly] = []

    for index, left in enumerate(declarers):
        for right in declarers[index + 1 :]:
            outcome = _classify_declarer_pair(left, right)
            match outcome:
                case ("blocked", _, _):
                    anomalies.append(
                        CouplingAnomaly(
                            kind=AnomalyKind.BLOCKED_BY_UNIT_MISMATCH,
                            key=key,
                            as_of=as_of,
                            participants=(left.assumption_id, right.assumption_id),
                            detail=(
                                f"{left.value.render()} and {right.value.render()} are not "
                                "comparable; coupling is blocked, not weakened"
                            ),
                            coupling_index_version=COUPLING_INDEX_VERSION,
                        )
                    )
                case ("conflict", a, b) if a.source_version_id == b.source_version_id:
                    # One document disagreeing with itself. Reported, never suppressed -
                    # but never as two artifacts disagreeing, because a single source must
                    # not be able to manufacture a cross-source contradiction
                    # (``docs/SECURITY.md``). An injected file that states a value twice
                    # would otherwise appear in the highest-trust tier as a governance
                    # conflict between two independent authorities.
                    anomalies.append(
                        CouplingAnomaly(
                            kind=AnomalyKind.INTRA_DOCUMENT_DIVERGENCE,
                            key=key,
                            as_of=as_of,
                            participants=(a.assumption_id, b.assumption_id),
                            detail=(
                                f"{a.source_title} declares {key} as both "
                                f"{a.value.render()} and {b.value.render()} at "
                                f"{a.scope.render()}. One source version disagreeing with "
                                "itself is an intra-document inconsistency, not two "
                                "authorities in conflict."
                            ),
                            coupling_index_version=COUPLING_INDEX_VERSION,
                        )
                    )
                case ("conflict", a, b):
                    relations.append(
                        _conflict_relation(
                            a, b, key=key, as_of=as_of, registry_version=registry_version
                        )
                    )
                case ("legitimate", narrower, broader):
                    overrides.append(
                        OverrideRecord(
                            key=key,
                            narrower=narrower,
                            broader=broader,
                            as_of=as_of,
                            authority_model_version=AUTHORITY_MODEL_VERSION,
                        )
                    )
                case ("unauthorized", narrower, broader):
                    relations.append(
                        _unauthorized_override_relation(
                            narrower,
                            broader,
                            key=key,
                            as_of=as_of,
                            registry_version=registry_version,
                        )
                    )
                case _:
                    pass  # not coupled for contradiction

    return tuple(relations), tuple(overrides), tuple(anomalies)


def _classify_declarer_pair(
    left: ResolvedBinding, right: ResolvedBinding
) -> tuple[str, ResolvedBinding, ResolvedBinding]:
    """The checkpoint 5.4.4 table, evaluated once per pair.

    Returns a tag plus the two bindings in rule order - for the override cases that is
    ``(narrower, broader)``, because the directed edge runs narrower to broader: the
    narrower artifact is the one deviating.
    """
    # Precondition 2: overlapping validity. Non-overlapping means superseded, not
    # conflicting, and adjacency is not overlap.
    if not overlaps(left.interval, right.interval):
        return ("superseded", left, right)

    # Precondition 5: compatible units. A mismatch blocks coupling entirely.
    if is_blocked(left.value, right.value):
        return ("blocked", left, right)

    # Precondition 4: values unequal under unit-aware comparison.
    if not are_divergent(left.value, right.value):
        return ("consistent", left, right)

    if scope_lattice.equals(left.scope, right.scope):
        return ("conflict", left, right)

    narrower = scope_lattice.narrower_of(left.scope, right.scope)
    if narrower is None:
        # Disjoint scopes: not coupled for contradiction. Two regions may differ freely.
        return ("disjoint", left, right)

    narrow_binding, broad_binding = (
        (left, right) if left.scope.path == narrower.path else (right, left)
    )
    if narrow_binding.outranks_or_matches(broad_binding):
        return ("legitimate", narrow_binding, broad_binding)
    return ("unauthorized", narrow_binding, broad_binding)


def _conflict_relation(
    left: ResolvedBinding,
    right: ResolvedBinding,
    *,
    key: str,
    as_of: dt.datetime,
    registry_version: str,
) -> DerivedRelation:
    """``conflicts_with``: symmetric, so the pair is stored in one canonical order."""
    first, second = sorted((left, right), key=lambda b: b.assumption_id)
    inputs = (first.assumption_id, second.assumption_id)
    return DerivedRelation(
        relation_id=relation_id_for(
            rule_id=RuleId.SAME_SCOPE_CONTRADICTION,
            key=key,
            scope=first.scope.render(),
            input_ids=inputs,
        ),
        relationship=Relationship.CONFLICTS_WITH,
        subject_ref=first.source_version_id,
        object_ref=second.source_version_id,
        input_record_ids=inputs,
        provenance=DerivationProvenance(
            rule_id=RuleId.SAME_SCOPE_CONTRADICTION,
            rule_version=COUPLING_INDEX_VERSION,
            registry_version=registry_version,
            coupling_index_version=COUPLING_INDEX_VERSION,
            authority_model_version=AUTHORITY_MODEL_VERSION,
            detail={
                "assumption_key": key,
                "scope": first.scope.render(),
                "as_of": as_of.isoformat(),
                "subject_value": first.value.render(),
                "object_value": second.value.render(),
                "subject_authority_class": first.authority_class.value,
                "object_authority_class": second.authority_class.value,
                "subject_artifact": first.source_title,
                "object_artifact": second.source_title,
            },
        ),
    )


def _unauthorized_override_relation(
    narrower: ResolvedBinding,
    broader: ResolvedBinding,
    *,
    key: str,
    as_of: dt.datetime,
    registry_version: str,
) -> DerivedRelation:
    """``unauthorized_override``: directed narrower -> broader, authority comparison recorded.

    The two ``*_authority_class`` detail keys are mandatory - ``DerivedRelation`` refuses
    the relationship without them - because the checkpoint requires the comparison to be
    *recorded*, not merely applied. A reviewer must be able to audit the conclusion without
    re-running the resolver.
    """
    inputs = (narrower.assumption_id, broader.assumption_id)
    return DerivedRelation(
        relation_id=relation_id_for(
            rule_id=RuleId.UNAUTHORIZED_OVERRIDE,
            key=key,
            scope=narrower.scope.render(),
            input_ids=inputs,
        ),
        relationship=Relationship.UNAUTHORIZED_OVERRIDE,
        subject_ref=narrower.source_version_id,
        object_ref=broader.source_version_id,
        input_record_ids=inputs,
        provenance=DerivationProvenance(
            rule_id=RuleId.UNAUTHORIZED_OVERRIDE,
            rule_version=COUPLING_INDEX_VERSION,
            registry_version=registry_version,
            coupling_index_version=COUPLING_INDEX_VERSION,
            authority_model_version=AUTHORITY_MODEL_VERSION,
            detail={
                "assumption_key": key,
                "as_of": as_of.isoformat(),
                "narrower_authority_class": narrower.authority_class.value,
                "broader_authority_class": broader.authority_class.value,
                "narrower_authority_rank": narrower.authority_rank,
                "broader_authority_rank": broader.authority_rank,
                "authority_ordering": "narrower_rank < broader_rank",
                "narrower_scope": narrower.scope.render(),
                "broader_scope": broader.scope.render(),
                "narrower_value": narrower.value.render(),
                "broader_value": broader.value.render(),
                "narrower_artifact": narrower.source_title,
                "broader_artifact": broader.source_title,
                "temporal_overlap": True,
                "registry_version": registry_version,
            },
        ),
    )


# ---------------------------------------------------------------------------
# declares vs consumes - staleness (checkpoint 3.3)
# ---------------------------------------------------------------------------


def _stale_consumers(
    live: Sequence[ResolvedBinding],
    *,
    key: str,
    as_of: dt.datetime,
    registry_version: str,
    authority: dict[str, AuthorityResolution],
) -> tuple[DerivedRelation, ...]:
    """A consumer binding a value the governing declarer does not state.

    Compared against resolved authority rather than pairwise - see the module docstring for
    why the pairwise reading produces false positives.
    """
    relations: list[DerivedRelation] = []

    for consumer in (b for b in live if b.role is BindingRole.CONSUMES):
        resolution = authority.get(consumer.scope.render())
        if resolution is None or not resolution.is_resolved:
            # No governing declarer, or the corpus cannot say which one governs. Either
            # way there is no established value for the consumer to be wrong about.
            continue

        governing = resolution.authoritative
        assert governing is not None  # guaranteed by is_resolved
        if not are_divergent(consumer.value, governing.value):
            continue
        if not overlaps(consumer.interval, governing.interval):
            continue  # pragma: no cover - both are in force at as_of

        inputs = (governing.assumption_id, consumer.assumption_id)
        relations.append(
            DerivedRelation(
                relation_id=relation_id_for(
                    rule_id=RuleId.STALE_CONSUMER,
                    key=key,
                    scope=consumer.scope.render(),
                    input_ids=inputs,
                ),
                relationship=Relationship.INVALIDATES_ASSUMPTION,
                subject_ref=governing.source_version_id,
                object_ref=consumer.source_version_id,
                input_record_ids=inputs,
                provenance=DerivationProvenance(
                    rule_id=RuleId.STALE_CONSUMER,
                    rule_version=COUPLING_INDEX_VERSION,
                    registry_version=registry_version,
                    coupling_index_version=COUPLING_INDEX_VERSION,
                    authority_model_version=AUTHORITY_MODEL_VERSION,
                    detail={
                        "assumption_key": key,
                        "scope": consumer.scope.render(),
                        "as_of": as_of.isoformat(),
                        "authoritative_value": governing.value.render(),
                        "consumed_value": consumer.value.render(),
                        "authoritative_artifact": governing.source_title,
                        "authoritative_authority_class": governing.authority_class.value,
                        "stale_artifact": consumer.source_title,
                        "stale_role": consumer.role.value,
                    },
                ),
            )
        )

    return tuple(relations)


# ---------------------------------------------------------------------------
# consumes vs consumes - an anomaly, never a contradiction (checkpoint 3.3)
# ---------------------------------------------------------------------------


def _consumer_divergence(
    live: Sequence[ResolvedBinding], *, key: str, as_of: dt.datetime
) -> tuple[CouplingAnomaly, ...]:
    """Two consumers disagreeing. Reported for scope review, never as a Class A finding.

    Neither artifact is an authority, so neither can be said to be wrong. Possibly they sit
    at different scopes; possibly both are stale. The checkpoint assigns no edge to this
    case, and inventing one would recreate the catch-all the taxonomy excludes.
    """
    consumers = [b for b in live if b.role is BindingRole.CONSUMES]
    anomalies: list[CouplingAnomaly] = []

    for index, left in enumerate(consumers):
        for right in consumers[index + 1 :]:
            if not overlaps(left.interval, right.interval):
                continue
            if not scope_lattice.intersects(left.scope, right.scope):
                continue
            if is_blocked(left.value, right.value) or not are_divergent(left.value, right.value):
                continue
            anomalies.append(
                CouplingAnomaly(
                    kind=AnomalyKind.CONSUMER_DIVERGENCE,
                    key=key,
                    as_of=as_of,
                    participants=(left.assumption_id, right.assumption_id),
                    detail=(
                        f"{left.source_title} consumes {left.value.render()} while "
                        f"{right.source_title} consumes {right.value.render()}; neither "
                        "declares, so neither is authoritative. Requires scope review."
                    ),
                    coupling_index_version=COUPLING_INDEX_VERSION,
                )
            )

    return tuple(anomalies)


def _dedupe(relations: Sequence[DerivedRelation]) -> tuple[DerivedRelation, ...]:
    """One relation per id, in id order.

    Relation ids are derived from the rule plus the sorted input records, so a pair reached
    twice - from either side, or via two overlapping coupling sets - collapses to one
    finding rather than double-counting the contradiction.
    """
    unique = {relation.relation_id: relation for relation in relations}
    return tuple(unique[key] for key in sorted(unique))
