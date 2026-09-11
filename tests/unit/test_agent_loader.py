from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.runtime.agent_loader import (
    AgentDefinition,
    AgentDefinitionError,
    load_agent_definitions,
    parse_agent_markdown,
)

VALID = """---
name: sample-agent
description: A sample agent.
version: 3
model: sonnet
---

# Sample Agent

Body text.
"""


def _parse(text: str, stem: str = "sample-agent") -> AgentDefinition:
    return parse_agent_markdown(text, source_path=Path(f"{stem}.md"))


def test_parses_frontmatter_and_body() -> None:
    definition = _parse(VALID)
    assert definition.name == "sample-agent"
    assert definition.description == "A sample agent."
    assert definition.version == 3
    assert definition.prompt.startswith("# Sample Agent")
    assert "---" not in definition.prompt


def test_frontmatter_model_is_only_a_harness_hint() -> None:
    """Production resolves models through the gateway and ignores this field."""
    assert _parse(VALID).harness_model_hint == "sonnet"


def test_prompt_hash_is_stable_and_content_addressed() -> None:
    first = _parse(VALID)
    second = _parse(VALID)
    assert first.prompt_sha256 == second.prompt_sha256
    assert len(first.prompt_sha256) == 64

    changed = _parse(VALID.replace("Body text.", "Body text, revised."))
    assert changed.prompt_sha256 != first.prompt_sha256


def test_version_defaults_to_one_when_absent() -> None:
    text = VALID.replace("version: 3\n", "")
    assert _parse(text).version == 1


def test_rejects_name_filename_mismatch() -> None:
    with pytest.raises(AgentDefinitionError, match="does not match filename"):
        _parse(VALID, stem="different-name")


def test_rejects_missing_frontmatter() -> None:
    with pytest.raises(AgentDefinitionError, match="missing YAML frontmatter"):
        _parse("# No frontmatter\n")


def test_rejects_unterminated_frontmatter() -> None:
    with pytest.raises(AgentDefinitionError, match="unterminated"):
        _parse("---\nname: sample-agent\n")


def test_rejects_missing_description() -> None:
    text = "---\nname: sample-agent\n---\n\nBody.\n"
    with pytest.raises(AgentDefinitionError, match="'description' is required"):
        _parse(text)


def test_rejects_empty_body() -> None:
    text = "---\nname: sample-agent\ndescription: d\n---\n\n   \n"
    with pytest.raises(AgentDefinitionError, match="body is empty"):
        _parse(text)


def test_rejects_non_integer_version() -> None:
    text = VALID.replace("version: 3", "version: not-a-number")
    with pytest.raises(AgentDefinitionError, match="must be an integer"):
        _parse(text)


def test_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(AgentDefinitionError, match="directory not found"):
        load_agent_definitions(tmp_path / "nope")


def test_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(AgentDefinitionError, match="no agent definitions found"):
        load_agent_definitions(tmp_path)


def test_loads_the_real_agent_directory() -> None:
    definitions = load_agent_definitions()
    assert len(definitions) == 16
    assert all(d.prompt_sha256 for d in definitions.values())
    assert all(d.name == d.source_path.stem for d in definitions.values())
