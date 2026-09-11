"""Loads agent definitions from ``.claude/agents/*.md``.

Those Markdown files are the single source of agent *behaviour*: role, responsibilities,
reasoning constraints, evidence requirements, output requirements. No Python class exists
merely to hold a prompt (``CLAUDE.md`` principle 12).

Runtime *binding* - tier, provider, model, budget - comes from ``app/config/agents.yaml``
instead. The ``model:`` key in the Markdown frontmatter is honoured by the Claude Code
interactive harness only; the production runtime resolves models through the gateway and
deliberately ignores it. It is surfaced here as ``harness_model_hint`` so the distinction
is visible rather than confusing.

Every definition carries a content hash so a finding can be attributed to the exact prompt
text that produced it, satisfying the prompt-versioning requirement without moving prompts
out of ``.claude/agents/``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from app.config.settings import REPO_ROOT

AGENTS_DIR = REPO_ROOT / ".claude" / "agents"

_FRONTMATTER_DELIM = "---"


class AgentDefinitionError(ValueError):
    """A definition file is malformed. Fail loudly: a silently skipped agent is worse."""


class AgentDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    version: int
    prompt: str
    prompt_sha256: str
    source_path: Path

    #: Frontmatter ``model:``. Interactive-harness only; not used for resolution.
    harness_model_hint: str | None = None


def parse_agent_markdown(text: str, *, source_path: Path) -> AgentDefinition:
    frontmatter, body = _split_frontmatter(text, source_path=source_path)

    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        raise AgentDefinitionError(f"{source_path}: frontmatter 'name' is required")

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        raise AgentDefinitionError(f"{source_path}: frontmatter 'description' is required")

    stem = source_path.stem
    if name != stem:
        raise AgentDefinitionError(
            f"{source_path}: frontmatter name '{name}' does not match filename '{stem}'"
        )

    prompt = body.strip()
    if not prompt:
        raise AgentDefinitionError(f"{source_path}: agent body is empty")

    raw_version = frontmatter.get("version", 1)
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise AgentDefinitionError(
            f"{source_path}: frontmatter 'version' must be an integer, got {raw_version!r}"
        ) from exc

    model_hint = frontmatter.get("model")

    return AgentDefinition(
        name=name,
        description=description.strip(),
        version=version,
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        source_path=source_path,
        harness_model_hint=model_hint if isinstance(model_hint, str) else None,
    )


def _split_frontmatter(text: str, *, source_path: Path) -> tuple[dict[str, Any], str]:
    stripped = text.lstrip("﻿").lstrip()
    if not stripped.startswith(_FRONTMATTER_DELIM):
        raise AgentDefinitionError(f"{source_path}: missing YAML frontmatter")

    parts = stripped.split(_FRONTMATTER_DELIM, 2)
    # parts[0] is empty, parts[1] is the frontmatter, parts[2] is the body.
    if len(parts) < 3:
        raise AgentDefinitionError(f"{source_path}: unterminated YAML frontmatter")

    try:
        loaded = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise AgentDefinitionError(f"{source_path}: invalid YAML frontmatter: {exc}") from exc

    if not isinstance(loaded, dict):
        raise AgentDefinitionError(f"{source_path}: frontmatter must be a mapping")

    return loaded, parts[2]


def load_agent_definitions(directory: Path | None = None) -> dict[str, AgentDefinition]:
    """Load every agent definition, keyed by name.

    Raises if the directory is missing or a file is malformed.
    """
    target = directory or AGENTS_DIR
    if not target.is_dir():
        raise AgentDefinitionError(f"agent definition directory not found: {target}")

    definitions: dict[str, AgentDefinition] = {}
    for path in sorted(target.glob("*.md")):
        definition = parse_agent_markdown(path.read_text(encoding="utf-8"), source_path=path)
        if definition.name in definitions:
            raise AgentDefinitionError(f"duplicate agent name '{definition.name}' at {path}")
        definitions[definition.name] = definition

    if not definitions:
        raise AgentDefinitionError(f"no agent definitions found in {target}")

    return definitions
