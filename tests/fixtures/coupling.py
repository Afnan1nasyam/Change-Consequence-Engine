"""Builders for the coupling phase.

:func:`binding` is the workhorse. It exists so that a checkpoint scenario reads as one
line of test code:

    binding(role=DECLARES, value=days(90), scope="global", authority=POLICY)
    binding(role=DECLARES, value=days(30), scope="region:eu", authority=REGULATORY)

Two properties matter for the tests built on it. Identifiers are **derived from the
arguments**, so two bindings that differ in any respect get different ids without the
caller managing a counter - and two calls with identical arguments get the *same* id, which
is what lets a reproducibility test rebuild an equivalent pool from scratch. And every
binding is admitted by default, because an unadmitted one cannot legally enter the index at
all; the tests that check that gate set the flags off explicitly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

from app.domain.entities.assumption import AssumptionDelta, AssumptionRecord
from app.domain.entities.class_a import ClassAFinding, ClassBMergeOutcome
from app.domain.entities.coupling import (
    AuthorityCandidate,
    AuthorityResolution,
    CouplingAnomaly,
    CouplingStrength,
    OverrideRecord,
    ResolvedBinding,
)
from app.domain.entities.decay import DecayEntry, DecayParticipant, KnowledgeDecayReport
from app.domain.value_objects.enums import (
    AUTHORITY_MODEL_VERSION,
    AmbiguityReason,
    AnomalyKind,
    AuthorityClass,
    AuthorityOutcome,
    BindingRole,
    DecayCategory,
    DeltaAssertedBy,
    DeltaKind,
    Explicitness,
    ValueType,
)
from app.domain.value_objects.provenance import DerivationProvenance
from app.domain.value_objects.rules import COUPLING_INDEX_VERSION, RuleId
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from app.domain.value_objects.typed_value import TypedValue
from tests.fixtures.ids import AS_OF, make_id

TENANT = make_id("tenant")
REGISTRY_VERSION = "registry-1"
KEY = "retention.window"

#: A date every default fixture interval contains, so a test that does not care about time
#: does not have to think about it.
NOW = AS_OF


def days(count: float = 90) -> TypedValue:
    return TypedValue(value_type=ValueType.DURATION, raw=count, unit="d")


def usd(amount: float = 10000) -> TypedValue:
    return TypedValue(value_type=ValueType.SCALAR, raw=amount, unit="USD")


def score(value: float = 0.85) -> TypedValue:
    """A unitless-kind value that must never couple with a USD one."""
    return TypedValue(value_type=ValueType.SCALAR, raw=value, unit="score")


def _derived_id(kind: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return f"{'as' if kind == 'assumption' else kind}_{digest[:32]}"


def binding(
    *,
    key: str = KEY,
    value: TypedValue | None = None,
    scope: str = "global",
    role: BindingRole = BindingRole.DECLARES,
    authority: AuthorityClass = AuthorityClass.POLICY,
    explicitness: Explicitness = Explicitness.EXPLICIT,
    effective_from: dt.datetime | None = None,
    effective_to: dt.datetime | None = None,
    title: str | None = None,
    span_verified: bool = True,
    value_parsed: bool = True,
    key_resolved: bool = True,
    tenant_id: str = TENANT,
    tag: str = "",
) -> ResolvedBinding:
    """One admitted binding joined to its holder's authority.

    ``tag`` distinguishes two bindings that are otherwise identical - two peers declaring
    the same value from different documents, for instance.
    """
    payload = value if value is not None else days(90)
    label = title or f"{authority.value}-{scope}-{payload.render()}{f'-{tag}' if tag else ''}"

    seed = (key, scope, role.value, payload.render(), authority.value, tag)
    version_id = _derived_id("sv", "version", *seed)
    record = AssumptionRecord(
        assumption_id=_derived_id("assumption", *seed),
        tenant_id=tenant_id,
        key=key,
        registry_version=REGISTRY_VERSION,
        scope=ScopeSelector.parse(scope),
        value=payload,
        role=role,
        explicitness=explicitness,
        holder_source_version_id=version_id,
        evidence_id=_derived_id("ev", "evidence", *seed),
        effective_from=effective_from,
        effective_to=effective_to,
        extractor_version="extractor-1",
        span_verified=span_verified,
        value_parsed=value_parsed,
        key_resolved=key_resolved,
    )
    return ResolvedBinding(
        record=record,
        authority_class=authority,
        source_id=_derived_id("src", "source", *seed),
        source_title=label,
    )


def delta(**over: Any) -> AssumptionDelta:
    return AssumptionDelta(
        **{
            "delta_id": make_id("delta"),
            "key": KEY,
            "scope": ScopeSelector.global_scope(),
            "old_value": days(90),
            "new_value": days(180),
            "delta_kind": DeltaKind.VALUE_CHANGE,
            "asserted_by": DeltaAssertedBy.CHANGE_REQUEST,
            "registry_version": REGISTRY_VERSION,
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Entity samples, for the contract-test inventory
# ---------------------------------------------------------------------------


def resolved_binding(**over: Any) -> ResolvedBinding:
    base = binding()
    return base.model_copy(update=over) if over else base


def authority_candidate(**over: Any) -> AuthorityCandidate:
    return AuthorityCandidate(**{"binding": binding(), "selected": True, **over})


def authority_resolution(**over: Any) -> AuthorityResolution:
    winner = binding()
    return AuthorityResolution(
        **{
            "key": KEY,
            "scope": ScopeSelector.global_scope(),
            "as_of": NOW,
            "outcome": AuthorityOutcome.RESOLVED,
            "authoritative": winner,
            "candidates": (AuthorityCandidate(binding=winner, selected=True),),
            "authority_model_version": AUTHORITY_MODEL_VERSION,
            "coupling_index_version": COUPLING_INDEX_VERSION,
            **over,
        }
    )


def ambiguous_resolution(**over: Any) -> AuthorityResolution:
    left = binding(value=days(90), tag="a")
    right = binding(value=days(45), tag="b")
    return authority_resolution(
        outcome=AuthorityOutcome.AMBIGUOUS,
        authoritative=None,
        ambiguity_reason=AmbiguityReason.EQUAL_AUTHORITY_DIVERGENCE,
        candidates=(
            AuthorityCandidate(binding=left),
            AuthorityCandidate(binding=right),
        ),
        **over,
    )


def override_record(**over: Any) -> OverrideRecord:
    return OverrideRecord(
        **{
            "key": KEY,
            "narrower": binding(scope="region:eu", authority=AuthorityClass.REGULATORY),
            "broader": binding(scope="global", authority=AuthorityClass.POLICY),
            "as_of": NOW,
            "authority_model_version": AUTHORITY_MODEL_VERSION,
            **over,
        }
    )


def coupling_anomaly(**over: Any) -> CouplingAnomaly:
    return CouplingAnomaly(
        **{
            "kind": AnomalyKind.CONSUMER_DIVERGENCE,
            "key": KEY,
            "as_of": NOW,
            "participants": (make_id("assumption", 1), make_id("assumption", 2)),
            "detail": "two consumers disagree and neither declares",
            "coupling_index_version": COUPLING_INDEX_VERSION,
            **over,
        }
    )


def coupling_strength(**over: Any) -> CouplingStrength:
    return CouplingStrength(
        **{
            "strength": 0.75,
            "blocked": False,
            "factor_inputs": {"role_composition": 1.0, "explicitness": 1.0},
            "strength_fn_version": "strength-1",
            **over,
        }
    )


def class_a_finding(**over: Any) -> ClassAFinding:
    return ClassAFinding(
        **{
            "finding_id": make_id("class_a_finding"),
            "tenant_id": TENANT,
            "relation_id": make_id("relation"),
            "relationship": Relationship.UNAUTHORIZED_OVERRIDE,
            "rule_id": RuleId.UNAUTHORIZED_OVERRIDE,
            "assumption_key": KEY,
            "scope": ScopeSelector.parse("region:eu"),
            "as_of": NOW,
            "affected_source_version_id": make_id("source_version", 1),
            "affected_source_id": make_id("source", 1),
            "affected_artifact": "EMEA Support Team Notes",
            "counterpart_source_version_id": make_id("source_version", 2),
            "counterpart_artifact": "Data Retention Policy",
            "summary": "a wiki page narrows a policy it does not outrank",
            "observed_value": days(45),
            "authoritative_value": days(90),
            "input_record_ids": (make_id("assumption", 1), make_id("assumption", 2)),
            "materialized_evidence_ids": (make_id("evidence", 1), make_id("evidence", 2)),
            "rule_version": COUPLING_INDEX_VERSION,
            "registry_version": REGISTRY_VERSION,
            "coupling_index_version": COUPLING_INDEX_VERSION,
            "authority_model_version": AUTHORITY_MODEL_VERSION,
            **over,
        }
    )


def class_b_merge_outcome(**over: Any) -> ClassBMergeOutcome:
    return ClassBMergeOutcome(
        **{
            "suppressed_hypothesis_ids": (make_id("hypothesis", 1),),
            "retained_hypothesis_ids": (make_id("hypothesis", 2),),
            "merged_into": {make_id("hypothesis", 1): make_id("class_a_finding")},
            "considered": 2,
            **over,
        }
    )


def decay_participant(**over: Any) -> DecayParticipant:
    return DecayParticipant(
        **{
            "source_version_id": make_id("source_version"),
            "artifact": "Data Retention Policy",
            "rendered_value": "90d",
            **over,
        }
    )


def decay_entry(**over: Any) -> DecayEntry:
    return DecayEntry(
        **{
            "category": DecayCategory.UNAUTHORIZED_OVERRIDE,
            "key": KEY,
            "scope": ScopeSelector.parse("region:eu"),
            "summary": "a wiki page narrows a policy it does not outrank",
            "participants": (decay_participant(),),
            "rule_id": RuleId.UNAUTHORIZED_OVERRIDE,
            **over,
        }
    )


def knowledge_decay_report(**over: Any) -> KnowledgeDecayReport:
    return KnowledgeDecayReport(
        **{
            "tenant_id": TENANT,
            "as_of": NOW,
            "entries": (decay_entry(),),
            "keys_examined": 9,
            "bindings_examined": 21,
            "registry_version": REGISTRY_VERSION,
            "coupling_index_version": COUPLING_INDEX_VERSION,
            "authority_model_version": AUTHORITY_MODEL_VERSION,
            **over,
        }
    )


def derivation_provenance(**over: Any) -> DerivationProvenance:
    return DerivationProvenance(
        **{
            "rule_id": RuleId.SAME_SCOPE_CONTRADICTION,
            "rule_version": COUPLING_INDEX_VERSION,
            "registry_version": REGISTRY_VERSION,
            "coupling_index_version": COUPLING_INDEX_VERSION,
            "authority_model_version": AUTHORITY_MODEL_VERSION,
            **over,
        }
    )


#: One valid instance of every entity this phase added, merged into the Phase 1 inventory
#: by ``tests.fixtures.domain``. An entity with no sample escapes every cross-cutting
#: contract test, so the inventory test fails rather than letting that happen quietly.
COUPLING_SAMPLES: dict[str, Any] = {
    "AuthorityCandidate": authority_candidate(),
    "AuthorityResolution": authority_resolution(),
    "ClassAFinding": class_a_finding(),
    "ClassBMergeOutcome": class_b_merge_outcome(),
    "CouplingAnomaly": coupling_anomaly(),
    "CouplingStrength": coupling_strength(),
    "DecayEntry": decay_entry(),
    "DecayParticipant": decay_participant(),
    "KnowledgeDecayReport": knowledge_decay_report(),
    "OverrideRecord": override_record(),
    "ResolvedBinding": resolved_binding(),
}
