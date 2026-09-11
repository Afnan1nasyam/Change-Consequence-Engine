"""The named deterministic rules, and the versions pinned onto their output.

Checkpoint 7.1 defines Class A as a derivation chain composed only of registry-resolved
verified bindings, scope-compatible coupling, and **named deterministic rules**. A rule
with no name is not auditable, so every rule that can emit a finding is enumerated here and
its id is stamped onto the :class:`DerivationProvenance` of whatever it produced.

Three version labels travel with every derivation, because each can change a finding
without any source document changing:

* ``COUPLING_INDEX_VERSION`` - the coupling and collision logic itself;
* ``registry_version`` - which keys were admissible (pinned per record, passed through);
* ``authority_model_version`` - the authority ordering used by the override test.

Bumping any of them is expected to change output. That is what makes two runs diffable and
the diff attributable (checkpoint 4.3, 13.21).
"""

from __future__ import annotations

from typing import Final

#: The coupling index, collision matrix and invalidation logic, taken together.
COUPLING_INDEX_VERSION: Final = "coupling-1"

#: The coupling-strength function. Versioned separately because retuning weights must not
#: look like a change to the collision rules.
COUPLING_STRENGTH_VERSION: Final = "strength-1"

#: The confidence function (checkpoint 9.1, 13.17). Versioned separately again, and for the
#: sharpest reason of the three: confidence is the only number a reader is likely to compare
#: across runs, so a reweighting that looked like a change in the evidence would be the most
#: misleading kind of silent change this system could make.
CONFIDENCE_FN_VERSION: Final = "confidence-1"


class RuleId:
    """Rule identifiers. Strings, deliberately, because they are persisted data.

    Each rule is one row of a checkpoint decision table, and the docstring points at the
    section it implements so a reviewer can check the code against the frozen contract
    rather than against intuition.
    """

    #: Checkpoint 5.4.1 - two declarers, equal scope, overlapping validity, values differ.
    SAME_SCOPE_CONTRADICTION: Final = "same_scope_contradiction"

    #: Checkpoint 5.4.2 - narrower scope, ``narrower.authority >= broader.authority``.
    #: Emits no finding; recorded so later passes can reason about the override.
    LEGITIMATE_OVERRIDE: Final = "legitimate_override"

    #: Checkpoint 5.4.3 - narrower scope, ``narrower.authority < broader.authority``.
    UNAUTHORIZED_OVERRIDE: Final = "unauthorized_override"

    #: Checkpoint 3.3 - a consumer binds a value that is not the authoritative one for its
    #: scope. Evaluated against ``resolve_authority``, never pairwise against an arbitrary
    #: declarer, so a consumer agreeing with the governing policy is not reported stale
    #: merely because some lower-authority document disagrees.
    STALE_CONSUMER: Final = "stale_consumer"

    #: Checkpoint 5.5 - a delta invalidates a binding that intersects its scope.
    DELTA_INVALIDATION: Final = "delta_invalidation"

    #: Checkpoint 5.5 - a delta reaches a scope holding a *legitimate* narrower override.
    #: Requires confirmation; the override is never labelled contradictory.
    SCOPE_INHERITANCE: Final = "scope_inheritance"


#: Every rule id, for the test that asserts each one is reachable. A rule that can never
#: fire is either dead code or an unimplemented decision-table row, and both are defects.
ALL_RULE_IDS: Final = (
    RuleId.SAME_SCOPE_CONTRADICTION,
    RuleId.LEGITIMATE_OVERRIDE,
    RuleId.UNAUTHORIZED_OVERRIDE,
    RuleId.STALE_CONSUMER,
    RuleId.DELTA_INVALIDATION,
    RuleId.SCOPE_INHERITANCE,
)

#: Rule ids that emit a ``DerivedRelation`` (ontology type C) and therefore a Class A
#: finding. ``LEGITIMATE_OVERRIDE`` is deliberately absent: it is recorded, not reported.
CLASS_A_EMITTING_RULES: Final = frozenset(
    {
        RuleId.SAME_SCOPE_CONTRADICTION,
        RuleId.UNAUTHORIZED_OVERRIDE,
        RuleId.STALE_CONSUMER,
        RuleId.DELTA_INVALIDATION,
    }
)
