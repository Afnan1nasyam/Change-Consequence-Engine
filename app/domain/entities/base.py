"""Shared base models.

``extra="forbid"`` is the workhorse invariant here, not a style preference. It is how
"no agent-authored model carries a confidence field" becomes structural: an agent that
emits ``{"confidence": 0.8}`` into a SpecialistFinding gets a validation error rather than
a silently-ignored key. Absence of the field plus a permissive config would enforce
nothing.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


class DomainModel(BaseModel):
    """Mutable-by-replacement domain object.

    Frozen so that a value cannot drift after validation - which matters most for anything
    the confidence function reads, since a recomputable score depends on its inputs being
    the ones that were persisted.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )


class TimestampedModel(DomainModel):
    created_at: dt.datetime = Field(default_factory=utc_now)


#: Bumped when any contract in this package changes shape. Persisted on findings via
#: RunProvenance.schema_version so two runs are explainable.
DOMAIN_SCHEMA_VERSION = "domain-1"
