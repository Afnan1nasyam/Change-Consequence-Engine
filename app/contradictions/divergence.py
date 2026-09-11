"""The deterministic divergence set, projected for one agent to rate. Nothing is decided here.

The contradiction-hunter does not search for contradictions. Divergence between artifacts
binding the same assumption key is detected by **exact comparison** in
:mod:`app.assumptions.collisions` and resolved by the checkpoint 3.3 / 5.4 matrix *before*
the agent runs. This module is the projection of that already-decided result into the shape
the agent is shown, and its whole purpose is that the classification travels as **data the
agent receives** rather than a question the agent is asked.

Five classes, and each one is a different deterministic bucket rather than a judgement:

============================= ===================================== ==============================
class                         produced by                           reader's action
============================= ===================================== ==============================
``material_contradiction``    ``conflicts_with`` relation           two peers disagree; pick one
``stale_invalidation``        ``invalidates_assumption`` relation   the consumer is out of date
``unauthorized_override``     ``unauthorized_override`` relation    a governance defect
``legitimate_scoped_override`` :class:`OverrideRecord`              **none** - correct behaviour
``non_material_divergence``   :class:`CouplingAnomaly`              scope review, not a conflict
============================= ===================================== ==============================

**The agent cannot move an item between classes, because the class is not a field it can
write.** :class:`DivergenceItem` is built from the deterministic record, and the assessment
schema in :mod:`app.contradictions.proposals` has no slot for a class, a relationship, a
relation id or a rule id. So the guarantee is a property of the *shapes* rather than of the
prompt's wording, which is the only kind of guarantee worth having against a model nobody
has evaluated.

**The handle the model cites is deliberately local.** ``item_id`` is ``d1``, ``d2``, ... and
never a ``relation_id``, so a model's output cannot name a Class A relation at all - not to
relabel it, not to reference it, not to reattribute it. It also gives the two non-Class-A
buckets, which have no relation id, the same citable shape as the three that do.

**A legitimate override is included, and stays not-a-finding.** Excluding it would be cheaper,
but the agent is required to *distinguish* the five and a reader benefits from seeing that a
scope narrowing was checked and found correct. Reporting one as a contradiction is the fastest
way to make this system untrusted, because the reader can see it is correct behaviour - so
:attr:`DivergenceItem.reportable_as_contradiction` is ``False`` for it by construction and no
model output can flip that.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.assumptions.collisions import KeyCollisions
from app.domain.entities.claims import DerivedRelation
from app.domain.entities.coupling import CouplingAnomaly, OverrideRecord, ResolvedBinding
from app.domain.value_objects.enums import AnomalyKind
from app.domain.value_objects.taxonomy import Relationship

#: Bumped when this projection changes, because it changes what the agent was shown and
#: therefore what its assessment refers to.
DIVERGENCE_PROJECTION_VERSION: Final = "divergence-projection-1"

#: A handle is short on purpose. The model cites it dozens of times in one response, and a
#: 36-character relation id would be both expensive and a name for something the model must
#: not be able to address.
HANDLE_PREFIX: Final = "d"


class DivergenceClass(StrEnum):
    """What the deterministic layer already concluded about one divergence.

    Not a spectrum and not a severity. Each member names the row of the checkpoint matrix
    that produced the item, so "which class is this?" is answered by *where it came from*
    and never by reading the artifacts again.
    """

    #: Two ``declares`` bindings, same scope, overlapping validity, divergent values.
    MATERIAL_CONTRADICTION = "material_contradiction"
    #: A ``consumes`` binding holding a value that is not the authoritative one for its scope.
    STALE_INVALIDATION = "stale_invalidation"
    #: A narrower scope overriding a broader one it does not outrank. A governance defect.
    UNAUTHORIZED_OVERRIDE = "unauthorized_override"
    #: A narrower scope overriding a broader one it does outrank. Correct behaviour.
    LEGITIMATE_SCOPED_OVERRIDE = "legitimate_scoped_override"
    #: Two consumers disagreeing, one document disagreeing with itself, or a unit mismatch.
    #: Reportable, but not two authorities in conflict.
    NON_MATERIAL_DIVERGENCE = "non_material_divergence"


#: The deterministic relationship each Class A divergence class is derived from. The mapping
#: is one-to-one and total over the three relationships the collision matrix emits: a
#: relation this table does not know is refused rather than dropped, because silently
#: skipping one would hide a Class A finding from the reader.
RELATIONSHIP_CLASSES: Final[dict[Relationship, DivergenceClass]] = {
    Relationship.CONFLICTS_WITH: DivergenceClass.MATERIAL_CONTRADICTION,
    Relationship.INVALIDATES_ASSUMPTION: DivergenceClass.STALE_INVALIDATION,
    Relationship.UNAUTHORIZED_OVERRIDE: DivergenceClass.UNAUTHORIZED_OVERRIDE,
}

#: Classes backed by a ``DerivedRelation``, and therefore by a Class A finding. The agent
#: annotates these; it cannot create, suppress or reclassify one.
CLASS_A_CLASSES: Final = frozenset(RELATIONSHIP_CLASSES.values())

#: Classes a reader may see described as a contradiction. Exactly one, deliberately.
CONTRADICTION_CLASSES: Final = frozenset({DivergenceClass.MATERIAL_CONTRADICTION})

#: A governance defect rather than a peer conflict. Retained whatever the model concludes
#: about materiality: an unauthorized override is a fact about who was allowed to decide,
#: and no assessment of business impact makes it stop being one.
GOVERNANCE_CLASSES: Final = frozenset({DivergenceClass.UNAUTHORIZED_OVERRIDE})

#: Classes that must never reach a reader as a contradiction, however the model rates them.
NOT_CONTRADICTIONS: Final = frozenset(DivergenceClass) - CONTRADICTION_CLASSES


class DivergenceProjectionError(ValueError):
    """The deterministic input cannot be projected as written.

    A ``ValueError`` because nothing was attempted and no model was involved. Raised rather
    than skipped: every case it covers would otherwise drop a deterministic result on the
    floor, and a Class A finding the agent never saw is worse than a crash.
    """


@dataclass(frozen=True)
class DivergenceItem:
    """One already-decided divergence, as the agent sees it.

    Frozen, and the class is set at construction from the record that produced it. The
    ``__post_init__`` checks below are not defensive coding: each of them is a way a caller
    could otherwise assemble an item whose class disagrees with its provenance, which is the
    one thing this module exists to make impossible.
    """

    #: The local handle the model cites. Never a domain identifier.
    item_id: str
    divergence_class: DivergenceClass
    assumption_key: str
    #: Rendered, because the agent reads it and never computes with it.
    scope: str
    as_of: dt.datetime
    #: The bindings that produced the divergence. At least two: a divergence between one
    #: binding and itself is not a divergence.
    participants: tuple[str, ...]
    #: Evidence for the participating bindings, resolved by the caller from the store. The
    #: closed set the model may cite for this item.
    evidence_ids: tuple[str, ...]
    detail: str

    #: Present iff the class is Class A backed. Carried so the application layer can join
    #: an assessment back to the finding; **never** rendered into the prompt.
    relation_id: str | None = None
    relationship: Relationship | None = None
    rule_id: str | None = None
    anomaly_kind: AnomalyKind | None = None

    def __post_init__(self) -> None:
        if len(self.participants) < 2:
            raise DivergenceProjectionError(
                f"{self.item_id}: a divergence needs at least two participating bindings, "
                f"got {len(self.participants)}"
            )
        if self.divergence_class in CLASS_A_CLASSES:
            if self.relationship is None or self.relation_id is None:
                raise DivergenceProjectionError(
                    f"{self.item_id}: {self.divergence_class.value!r} is derived by the "
                    "resolver and must name the relation it reports"
                )
            expected = RELATIONSHIP_CLASSES.get(self.relationship)
            if expected is not self.divergence_class:
                # The structural form of "the agent may never turn one Class A relationship
                # into another": even the code assembling the input cannot mislabel one.
                raise DivergenceProjectionError(
                    f"{self.item_id}: relationship {self.relationship.value!r} classifies as "
                    f"{expected.value if expected else 'nothing'!r}, not "
                    f"{self.divergence_class.value!r}. The deterministic classification is "
                    "authoritative and is not re-derivable here."
                )
        elif self.relation_id is not None or self.relationship is not None:
            raise DivergenceProjectionError(
                f"{self.item_id}: {self.divergence_class.value!r} is not produced by the "
                "resolver and must not name a derived relation"
            )

    @property
    def is_class_a(self) -> bool:
        """Whether a ``DerivedRelation`` stands behind this item."""
        return self.divergence_class in CLASS_A_CLASSES

    @property
    def reportable_as_contradiction(self) -> bool:
        """Whether a reader may be shown this as a contradiction. One class qualifies."""
        return self.divergence_class in CONTRADICTION_CLASSES

    @property
    def is_governance_defect(self) -> bool:
        return self.divergence_class in GOVERNANCE_CLASSES


def divergences_from(
    collisions: Iterable[KeyCollisions],
    *,
    bindings: Mapping[str, ResolvedBinding],
) -> tuple[DivergenceItem, ...]:
    """Project every deterministic divergence in ``collisions`` into agent-visible items.

    ``bindings`` maps assumption id to the resolved binding, so evidence ids can be attached
    without this module touching a store. A participant the mapping does not cover is an
    error rather than an item with thinner evidence: the model would then be asked to rate a
    divergence it cannot cite, and its only honest answer would be unsupported.

    Ordering is canonical and independent of input order, because the handles are positional
    and two runs over one corpus must produce the same handle for the same divergence.
    """
    drafts: list[tuple[tuple[str, str, str, tuple[str, ...]], dict[str, object]]] = []

    for group in collisions:
        for relation in group.relations:
            drafts.append(_draft_from_relation(relation, group=group, bindings=bindings))
        for override in group.overrides:
            drafts.append(_draft_from_override(override, bindings=bindings))
        for anomaly in group.anomalies:
            drafts.append(_draft_from_anomaly(anomaly, bindings=bindings))

    drafts.sort(key=lambda pair: pair[0])
    return tuple(
        DivergenceItem(item_id=f"{HANDLE_PREFIX}{index}", **fields)  # type: ignore[arg-type]
        for index, (_sort_key, fields) in enumerate(drafts, start=1)
    )


def by_handle(items: Iterable[DivergenceItem]) -> dict[str, DivergenceItem]:
    """Index items by the handle the model cites. Refuses a duplicate handle."""
    indexed: dict[str, DivergenceItem] = {}
    for item in items:
        if item.item_id in indexed:
            raise DivergenceProjectionError(
                f"duplicate divergence handle {item.item_id!r}; a handle must address one "
                "divergence or an assessment cannot be attributed"
            )
        indexed[item.item_id] = item
    return indexed


# ---------------------------------------------------------------------------
# One draft per deterministic record
# ---------------------------------------------------------------------------


def _draft_from_relation(
    relation: DerivedRelation,
    *,
    group: KeyCollisions,
    bindings: Mapping[str, ResolvedBinding],
) -> tuple[tuple[str, str, str, tuple[str, ...]], dict[str, object]]:
    divergence_class = RELATIONSHIP_CLASSES.get(relation.relationship)
    if divergence_class is None:
        # A derivable relationship the collision matrix does not emit, or a new one added
        # without a class. Refused rather than skipped: dropping it would hide a Class A
        # finding from the only stage that rates materiality.
        raise DivergenceProjectionError(
            f"relation {relation.relation_id} is a {relation.relationship.value!r}, which is "
            f"not a divergence this projection knows. Known: "
            f"{sorted(r.value for r in RELATIONSHIP_CLASSES)}."
        )

    participants = tuple(relation.input_record_ids)
    scope = _scope_of(participants, bindings=bindings, fallback=_sole_scope(group))
    return (
        (divergence_class.value, group.key, scope, participants),
        {
            "divergence_class": divergence_class,
            "assumption_key": group.key,
            "scope": scope,
            "as_of": group.as_of,
            "participants": participants,
            "evidence_ids": _evidence_of(participants, bindings=bindings),
            "detail": f"{relation.subject_ref} -> {relation.object_ref}",
            "relation_id": relation.relation_id,
            "relationship": relation.relationship,
            "rule_id": relation.provenance.rule_id,
        },
    )


def _draft_from_override(
    override: OverrideRecord,
    *,
    bindings: Mapping[str, ResolvedBinding],
) -> tuple[tuple[str, str, str, tuple[str, ...]], dict[str, object]]:
    """A legitimate override: recorded, shown, and never a finding.

    It carries no ``relation_id`` because the resolver deliberately emits no relation for
    it. That absence is the reason the class cannot be confused with the unauthorized case
    downstream - the two do not have the same shape, not merely the same shape with a
    different label.
    """
    participants = (override.narrower.assumption_id, override.broader.assumption_id)
    scope = override.narrower.scope.render()
    detail = (
        f"{override.narrower.source_title} narrows {override.key} to {scope} with "
        f"{override.narrower.authority_class.value} authority, which matches or outranks "
        f"{override.broader.source_title} ({override.broader.authority_class.value}) at "
        f"{override.broader.scope.render()}. Correct behaviour, not a defect."
    )
    return (
        (DivergenceClass.LEGITIMATE_SCOPED_OVERRIDE.value, override.key, scope, participants),
        {
            "divergence_class": DivergenceClass.LEGITIMATE_SCOPED_OVERRIDE,
            "assumption_key": override.key,
            "scope": scope,
            "as_of": override.as_of,
            "participants": participants,
            "evidence_ids": _evidence_of(participants, bindings=bindings),
            "detail": detail,
        },
    )


def _draft_from_anomaly(
    anomaly: CouplingAnomaly,
    *,
    bindings: Mapping[str, ResolvedBinding],
) -> tuple[tuple[str, str, str, tuple[str, ...]], dict[str, object]]:
    participants = tuple(anomaly.participants)
    scope = _scope_of(participants, bindings=bindings, fallback="(mixed)")
    return (
        (DivergenceClass.NON_MATERIAL_DIVERGENCE.value, anomaly.key, scope, participants),
        {
            "divergence_class": DivergenceClass.NON_MATERIAL_DIVERGENCE,
            "assumption_key": anomaly.key,
            "scope": scope,
            "as_of": anomaly.as_of,
            "participants": participants,
            "evidence_ids": _evidence_of(participants, bindings=bindings),
            "detail": f"{anomaly.kind.value}: {anomaly.detail}",
            "anomaly_kind": anomaly.kind,
        },
    )


def _evidence_of(
    participants: tuple[str, ...], *, bindings: Mapping[str, ResolvedBinding]
) -> tuple[str, ...]:
    """The evidence spans behind the participating bindings, deduplicated and ordered."""
    found: list[str] = []
    for assumption_id in participants:
        binding = bindings.get(assumption_id)
        if binding is None:
            raise DivergenceProjectionError(
                f"binding {assumption_id} participates in a divergence but was not supplied. "
                "Rating a divergence whose evidence cannot be cited would leave the agent no "
                "honest answer but 'unsupported'."
            )
        found.append(binding.evidence_id)
    return tuple(sorted(set(found)))


def _scope_of(
    participants: tuple[str, ...],
    *,
    bindings: Mapping[str, ResolvedBinding],
    fallback: str,
) -> str:
    """The scope the participants share, or ``fallback`` when they differ.

    Deliberately does not pick one when they differ. The scope of an override or a
    cross-scope anomaly is a property of the pair, and naming one side's scope as the
    item's would misdescribe which readers it affects.
    """
    scopes = {bindings[p].scope.render() for p in participants if p in bindings}
    return scopes.pop() if len(scopes) == 1 else fallback


def _sole_scope(group: KeyCollisions) -> str:
    """The one scope observed for the key, or a marker when several were."""
    return next(iter(group.authority)) if len(group.authority) == 1 else "(mixed)"
