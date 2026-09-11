"""Provenance records.

Two kinds, deliberately separate:

* :class:`ModelProvenance` - which prompt, provider and model produced a model-authored
  artifact. Recorded per finding because provider identity is an experimental variable in
  the diversity study, not merely an operational detail.
* :class:`DerivationProvenance` - which rule, inputs and configuration versions produced a
  deterministically derived artifact.

A type C claim carries the second and **never** the first: it had no model. That separation
is the machine-readable form of "the LLM did not find this".
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.domain.value_objects.identifiers import VersionLabel

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ModelProvenance(BaseModel):
    """Attribution for a model-authored artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    prompt_version: int = Field(ge=1)
    prompt_sha256: Sha256
    provider: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    model: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    tier: Annotated[str, StringConstraints(min_length=1, max_length=16)]


class DerivationProvenance(BaseModel):
    """Attribution for a deterministically derived artifact.

    ``detail`` carries the rule's inputs. For an override legitimacy test that means both
    ``authority_class`` values and the ordering used - the checkpoint requires the authority
    comparison to be recorded, not merely applied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    rule_version: VersionLabel
    registry_version: VersionLabel
    coupling_index_version: VersionLabel
    authority_model_version: VersionLabel | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class RunProvenance(BaseModel):
    """Everything needed to explain why two analysis runs differ.

    ``registry_version`` and ``authority_model_version`` are included because a registry
    expansion or an authority-ordering change legitimately produces new Class A
    derivations, and two runs differing only in those must be diffable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: VersionLabel
    confidence_fn_version: VersionLabel | None = None
    coupling_index_version: VersionLabel | None = None
    registry_version: VersionLabel | None = None
    authority_model_version: VersionLabel | None = None
    retrieval_config_version: VersionLabel | None = None
    permission_scope_hash: Sha256 | None = None
