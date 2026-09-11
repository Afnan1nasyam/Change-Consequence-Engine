"""ChangeRequest and ChangeSpec."""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from pydantic import Field, StringConstraints

from app.domain.entities.assumption import AssumptionDelta
from app.domain.entities.base import DomainModel, TimestampedModel
from app.domain.value_objects.identifiers import ChangeId, EvidenceId, TenantId, VersionLabel
from app.domain.value_objects.scope import ScopeSelector

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
LongText = Annotated[str, StringConstraints(min_length=1, max_length=20000)]


class ChangeRequest(TimestampedModel):
    """The raw proposed change, as submitted."""

    change_id: ChangeId
    tenant_id: TenantId
    title: ShortText
    description: LongText
    requested_by: ShortText | None = None
    effective_date: dt.datetime | None = None
    scope: ScopeSelector = Field(default_factory=ScopeSelector.global_scope)
    source_refs: tuple[str, ...] = ()


class ChangeSpec(DomainModel):
    """The structured interpretation produced by the change-decomposer.

    ``analyst_caveats`` was previously named ``assumptions``. The rename is not cosmetic:
    the old name collided with the ``AssumptionRecord`` vocabulary and would eventually
    have been wired into the coupling index, where it does not belong. These are caveats
    about the *change request*, closer to ontology type E than to a binding.

    Carries no ``confidence`` (it is LLM-authored) and no ``consequence_class`` (that is
    pipeline-assigned).
    """

    change_id: ChangeId
    subject_entities: tuple[ShortText, ...] = ()
    old_state: ShortText | None = None
    new_state: ShortText | None = None
    scope: ScopeSelector = Field(default_factory=ScopeSelector.global_scope)
    constraints: tuple[ShortText, ...] = ()

    analyst_caveats: tuple[ShortText, ...] = ()
    ambiguities: tuple[ShortText, ...] = ()
    affected_domains: tuple[ShortText, ...] = ()

    #: The deterministic resolver's input. Precision matters more than coverage here: a
    #: wrong key propagates to every downstream consequence, a missing key merely narrows
    #: the analysis.
    assumption_deltas: tuple[AssumptionDelta, ...] = ()

    #: Spans of the change request itself. The decomposer runs before retrieval, so it has
    #: no corpus evidence to cite and must never invent an id to fill this.
    evidence_ids: tuple[EvidenceId, ...] = ()

    version: VersionLabel
