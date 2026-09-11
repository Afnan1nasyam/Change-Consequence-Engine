"""Class A findings: what the deterministic resolver produces, before any judgment.

**Why this is not a `Consequence`.** A ``Consequence`` requires a ``severity``, and
checkpoint 9.4 is explicit that severity is *judgment* - a function of business impact the
corpus does not contain - and is therefore assigned by the judge, attributed to a
prompt/provider/model, and reviewer-overridable. The resolver has no model and no business
context. Having it stamp a severity would either fabricate a judgment or hard-code one, and
both would launder a guess into the trust-bearing tier.

So the resolver emits a :class:`ClassAFinding`: fully derived, fully evidenced, carrying no
score and no judgment. The judge converts it into a ``Consequence`` in a later phase by
adding exactly the field it is qualified to add. See ``docs/STATUS.md`` for the open review
item this raises against the checkpoint's entity map.

Two invariants are structural here:

* ``consequence_class`` and ``origin`` are ``Literal``s, so this object cannot be
  constructed as anything but resolver-produced Class A, and no agent-authored payload can
  deserialize into one (checkpoint 7.2).
* the evidence set is **explicitly marked** as the computed union of the inputs'
  evidence. The underlying ``DerivedRelation`` still carries none of its own, because
  nominating an arbitrary input span as direct support for a derivation is precisely the
  fabricated provenance this project exists to prevent (checkpoint 2.4).
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.entities.coupling import CouplingStrength
from app.domain.value_objects.enums import (
    ConsequenceClass,
    Origin,
)
from app.domain.value_objects.identifiers import (
    AssumptionId,
    AssumptionKey,
    ClassAFindingId,
    DeltaId,
    EvidenceId,
    HypothesisId,
    RelationId,
    SourceId,
    SourceVersionId,
    TenantId,
    VersionLabel,
)
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import DERIVED_RELATIONSHIPS, Relationship
from app.domain.value_objects.typed_value import TypedValue

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
Statement = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class ClassAFinding(TimestampedModel):
    """A consequence produced by the deterministic resolver.

    Class A means *the resolver produced it* - classification is by producer, not by
    subject matter. Sounding derivable is not a derivation (checkpoint 7.2).
    """

    finding_id: ClassAFindingId
    tenant_id: TenantId

    #: Pipeline-assigned and unwritable. There is no agent-authored path to this value.
    consequence_class: Literal[ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED] = (
        ConsequenceClass.CLASS_A_ASSUMPTION_MEDIATED
    )
    origin: Literal[Origin.RESOLVER] = Origin.RESOLVER

    #: The type C record this finding reports. Required: Class A *is* being derived.
    relation_id: RelationId
    relationship: Relationship
    rule_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]

    assumption_key: AssumptionKey
    scope: ScopeSelector
    as_of: dt.datetime

    #: The artifact the reader has to go and fix.
    affected_source_version_id: SourceVersionId
    affected_source_id: SourceId
    affected_artifact: ShortText

    #: The other side of the coupling - the authority, or the peer declarer. Absent only
    #: for a delta-triggered invalidation whose corpus holds no declaration to supersede:
    #: there the counterpart is the proposed change, named by ``delta_id``.
    counterpart_source_version_id: SourceVersionId | None = None
    counterpart_artifact: ShortText | None = None

    summary: Statement
    observed_value: TypedValue | None = None
    authoritative_value: TypedValue | None = None

    input_record_ids: tuple[AssumptionId, ...] = Field(min_length=1)
    #: The delta this finding was derived against, when the trigger was a proposed change
    #: rather than a standing divergence between two artifacts. It counts as the second
    #: input to the derivation - it simply is not an ``AssumptionRecord``, so it cannot
    #: live in ``input_record_ids``.
    delta_id: DeltaId | None = None

    #: The computed transitive union of the input records' evidence, materialized for the
    #: reader. Marked as derived below so it can never be mistaken for direct support.
    materialized_evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    evidence_is_derived_union: Literal[True] = True

    strength: CouplingStrength | None = None
    #: Class B hypotheses absorbed by this finding. The model's mechanism text is worth
    #: keeping even when the discovery was not the model's (checkpoint 7.5).
    merged_hypothesis_ids: tuple[HypothesisId, ...] = ()

    rule_version: VersionLabel
    registry_version: VersionLabel
    coupling_index_version: VersionLabel
    authority_model_version: VersionLabel

    @model_validator(mode="after")
    def _relationship_is_derivable(self) -> Self:
        if self.relationship not in DERIVED_RELATIONSHIPS:
            raise ValueError(
                f"finding {self.finding_id}: a Class A finding may only report a derivable "
                f"relationship, got {self.relationship.value!r}. Derivable relationships "
                f"are {sorted(r.value for r in DERIVED_RELATIONSHIPS)}."
            )
        return self

    @model_validator(mode="after")
    def _derivation_has_two_inputs(self) -> Self:
        """A derivation over one input is not a derivation.

        Checkpoint 2 defines type C as computed from **>= 2** records by a named rule. A
        single-input "derivation" would be a restatement of one record wearing derived
        provenance, which is the shape this whole ontology exists to make impossible.

        The second input is not always an ``AssumptionRecord``. A delta-triggered
        invalidation is derived from *(the proposed change, the affected binding)*, so
        ``delta_id`` satisfies the requirement in exactly that case. A standing divergence
        between two artifacts has no delta and must name two records.
        """
        distinct = set(self.input_record_ids)
        if len(distinct) >= 2:
            return self
        if self.delta_id is not None:
            return self
        raise ValueError(
            f"finding {self.finding_id}: a derived relation needs two inputs. Got one "
            f"record ({sorted(distinct)}) and no delta_id, which would make this a "
            "restatement of a single binding rather than a derivation."
        )

    @property
    def dedupe_pair(self) -> tuple[str, str]:
        """The ``(assumption_key, affected_artifact)`` pair Class B merges against.

        Keyed on the affected *source version*, not its title: two documents may share a
        title, and merging on a display string would suppress an unrelated hypothesis
        (checkpoint 7.5.2).
        """
        return (self.assumption_key, self.affected_source_version_id)


class ClassBMergeOutcome(DomainModel):
    """The result of reconciling agentic hypotheses against the Class A set.

    Merged, never dropped: the articulated mechanism is worth keeping even when the
    discovery was not the model's. What must not survive is a *second, independent*
    finding claiming the same thing at a trust level the model did not earn.

    ``rediscovery_rate`` is a visible metric and a **warning**, not an achievement. A high
    value means the hunter's prompt or its evidence slice is misconfigured - it is being
    fed what the resolver already knows (checkpoint 7.5.3).
    """

    #: Hypotheses suppressed because a Class A finding already covers the pair.
    suppressed_hypothesis_ids: tuple[HypothesisId, ...] = ()
    #: Hypotheses that survived as genuine Class B candidates.
    retained_hypothesis_ids: tuple[HypothesisId, ...] = ()
    #: Which Class A finding absorbed each suppressed hypothesis.
    merged_into: dict[str, str] = Field(default_factory=dict)
    considered: int = Field(ge=0)

    @model_validator(mode="after")
    def _accounting_balances(self) -> Self:
        """Every hypothesis is either suppressed or retained. None may vanish."""
        total = len(self.suppressed_hypothesis_ids) + len(self.retained_hypothesis_ids)
        if total != self.considered:
            raise ValueError(
                f"merge accounting does not balance: considered={self.considered} but "
                f"{len(self.suppressed_hypothesis_ids)} suppressed + "
                f"{len(self.retained_hypothesis_ids)} retained = {total}. A hypothesis that "
                "is neither suppressed nor retained has been silently discarded."
            )
        return self

    @property
    def class_b_rediscovery_rate(self) -> float:
        """Share of Class B hypotheses suppressed as Class A duplicates.

        Zero when nothing was considered - an empty run has no rate, and reporting 0.0 for
        "no hypotheses" versus "no rediscovery" would conflate two different situations, so
        callers should check :attr:`considered` before reading this.
        """
        if self.considered == 0:
            return 0.0
        return len(self.suppressed_hypothesis_ids) / self.considered
