"""Agent runtime bindings loaded from ``agents.yaml``.

Bindings are runtime configuration (tier, optional provider/model override, budget), kept
separate from agent behaviour, which lives in ``.claude/agents/*.md``. See
``docs/FOLDER_STRUCTURE.md``, "Separation of prompt from runtime binding".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.domain.value_objects.tiers import CapabilityTier


class AgentBinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: CapabilityTier = CapabilityTier.STANDARD
    max_output_tokens: int = Field(default=2048, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    #: Optional explicit provider override. Enables the F2-F5 diversity conditions in
    #: docs/EVALUATION.md without touching code.
    provider: str | None = None

    #: Env var holding the model identifier when overriding the tier default. Never a
    #: literal model name: those belong in the environment.
    model_env: str | None = None

    def resolve_model_override(self) -> str | None:
        if not self.model_env:
            return None
        value = os.environ.get(self.model_env)
        return value or None


class AgentBindings(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 1
    defaults: AgentBinding = AgentBinding()
    bindings: dict[str, AgentBinding] = Field(default_factory=dict)

    def for_agent(self, agent: str) -> AgentBinding:
        """Binding for ``agent``, falling back to ``defaults``.

        Unknown agents resolve to defaults rather than raising: an agent may legitimately
        exist as a prompt before it is given a bespoke binding.
        """
        return self.bindings.get(agent, self.defaults)

    @classmethod
    def load(cls, path: Path) -> AgentBindings:
        if not path.is_file():
            raise FileNotFoundError(f"agent binding config not found: {path}")

        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")

        defaults = AgentBinding.model_validate(raw.get("defaults") or {})

        bindings: dict[str, AgentBinding] = {}
        # `agents` and `internal` share one namespace at runtime: internal roles such as
        # the assumption extractor consume the gateway exactly as agents do.
        for section in ("agents", "internal"):
            entries = raw.get(section) or {}
            if not isinstance(entries, dict):
                raise ValueError(f"{path}: '{section}' must be a mapping")
            for name, spec in entries.items():
                merged = {**defaults.model_dump(mode="json"), **(spec or {})}
                bindings[str(name)] = AgentBinding.model_validate(merged)

        return cls(
            version=int(raw.get("version") or 1),
            defaults=defaults,
            bindings=bindings,
        )
