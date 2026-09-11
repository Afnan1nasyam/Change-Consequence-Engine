"""The knowledge-decay report: a standing corpus-health deliverable.

``IDEA.md`` 15.5 treats this as a first-class product, not a side effect, and the reason is
economic: it needs no change request, predicts nothing about the future, and is verifiable
on delivery. It is also the earliest honest proof that the corpus supports the product at
all - if the assumption store yields nothing here, no amount of agent reasoning downstream
will help.

The report's central obligation is to keep three things apart, because collapsing them is
exactly how a corpus-health tool loses its reader's trust:

============================ =========================================================
**known inconsistency**      two artifacts disagree, and one of them is wrong
**known ambiguity**          the corpus cannot say which artifact governs
**missing knowledge**        nobody wrote it down
============================ =========================================================

Absence is never reported as contradiction. A key with a single holder is not a conflict;
an unresolvable authority is not a defect in either candidate; an unregistered key is a
gap in *our vocabulary*, not an error in the document.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from app.domain.entities.base import DomainModel
from app.domain.value_objects.enums import (
    KNOWN_AMBIGUITY,
    KNOWN_INCONSISTENCY,
    MISSING_KNOWLEDGE,
    AuthorityClass,
    BindingRole,
    DecayCategory,
)
from app.domain.value_objects.identifiers import (
    AssumptionId,
    AssumptionKey,
    ClassAFindingId,
    SourceVersionId,
    TenantId,
    VersionLabel,
)
from app.domain.value_objects.scope import ScopeSelector

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
Statement = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class DecayParticipant(DomainModel):
    """One artifact's side of a decay entry, with everything a reviewer needs to act.

    ``rendered_value`` rather than a ``TypedValue`` because this is report surface: the
    reviewer needs to read ``90d`` and ``45d`` side by side. The machine-readable values
    stay on the underlying records, which the ids here point at.
    """

    source_version_id: SourceVersionId
    artifact: ShortText
    assumption_id: AssumptionId | None = None
    role: BindingRole | None = None
    authority_class: AuthorityClass | None = None
    rendered_value: ShortText | None = None
    effective_from: dt.datetime | None = None
    effective_to: dt.datetime | None = None


class DecayEntry(DomainModel):
    """One row of the report.

    ``requires_human_decision`` separates "go and fix this document" from "somebody has to
    decide what the truth is". An ambiguity handed to a reviewer as though it were a defect
    wastes their time; a defect handed over as an open question never gets fixed.
    """

    category: DecayCategory
    key: AssumptionKey
    scope: ScopeSelector = Field(default_factory=ScopeSelector.global_scope)
    summary: Statement
    participants: tuple[DecayParticipant, ...] = ()

    #: Present when this row is backed by a deterministic Class A finding. Absent for
    #: ambiguity, gaps, and the single-holder / unwritten-rule observations, which are
    #: claims about the corpus rather than derived relations between artifacts.
    finding_id: ClassAFindingId | None = None
    rule_id: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = None

    requires_human_decision: bool = False
    what_would_resolve: Statement | None = None

    @model_validator(mode="after")
    def _inconsistency_is_derived(self) -> Self:
        """A row that accuses an artifact of being wrong must name the rule that says so.

        Without this an inconsistency could be asserted by the report itself, which would
        make the report a second source of truth rather than a projection of the resolver.
        """
        if self.category in KNOWN_INCONSISTENCY and self.rule_id is None:
            raise ValueError(
                f"decay entry for {self.key} in category {self.category.value!r} asserts an "
                "inconsistency and must cite the deterministic rule that derived it"
            )
        return self

    @model_validator(mode="after")
    def _absence_is_not_derived(self) -> Self:
        """A gap may not carry a rule id, because no rule derived it.

        This is the structural form of "do not call absence a contradiction". An
        ``EvidenceGap`` can never become an edge (checkpoint 2.4); the same discipline
        applies to the report rows that describe missing knowledge.
        """
        if self.category in MISSING_KNOWLEDGE and self.rule_id is not None:
            raise ValueError(
                f"decay entry for {self.key} in category {self.category.value!r} describes "
                f"missing knowledge and must not cite rule {self.rule_id!r}: absence is "
                "recorded, never derived"
            )
        if self.category in MISSING_KNOWLEDGE and self.finding_id is not None:
            raise ValueError(
                f"decay entry for {self.key} in category {self.category.value!r} must not "
                "reference a Class A finding: nothing was derived from an absence"
            )
        return self

    @model_validator(mode="after")
    def _ambiguity_states_its_resolution(self) -> Self:
        """An unresolvable authority is only useful if it says what would resolve it."""
        if self.category in KNOWN_AMBIGUITY:
            if not self.requires_human_decision:
                raise ValueError(
                    f"decay entry for {self.key} in category {self.category.value!r} is an "
                    "ambiguity and requires a human decision by definition"
                )
            if self.what_would_resolve is None:
                raise ValueError(
                    f"decay entry for {self.key} in category {self.category.value!r} must "
                    "state what human validation would resolve it"
                )
        return self


class KnowledgeDecayReport(DomainModel):
    """The whole standing report for one tenant at one ``as_of``.

    Reproducible from the store state plus the version labels recorded here. Two reports
    over the same store and the same versions are byte-identical, which is what makes a
    diff between two runs attributable to a registry expansion or an authority-ordering
    change rather than to nondeterminism.
    """

    tenant_id: TenantId
    as_of: dt.datetime
    entries: tuple[DecayEntry, ...] = ()

    keys_examined: int = Field(ge=0)
    bindings_examined: int = Field(ge=0)

    registry_version: VersionLabel
    coupling_index_version: VersionLabel
    authority_model_version: VersionLabel

    def of_category(self, category: DecayCategory) -> tuple[DecayEntry, ...]:
        return tuple(e for e in self.entries if e.category is category)

    @property
    def counts_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.category.value] = counts.get(entry.category.value, 0) + 1
        return counts

    @property
    def known_inconsistencies(self) -> tuple[DecayEntry, ...]:
        """Rows where an artifact is wrong and a deterministic rule says which."""
        return tuple(e for e in self.entries if e.category in KNOWN_INCONSISTENCY)

    @property
    def known_ambiguities(self) -> tuple[DecayEntry, ...]:
        """Rows where the corpus cannot decide. No winner is named."""
        return tuple(e for e in self.entries if e.category in KNOWN_AMBIGUITY)

    @property
    def missing_knowledge(self) -> tuple[DecayEntry, ...]:
        """Rows where nobody wrote something down. Never a contradiction."""
        return tuple(e for e in self.entries if e.category in MISSING_KNOWLEDGE)

    @property
    def is_quiet(self) -> bool:
        """True when the corpus is healthy.

        A report that is never quiet has not been shown to detect anything, which is the
        negative-control bar in ``CLAUDE.md``.
        """
        return not self.entries
