"""Provider registry and configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.agent_bindings import AgentBindings
from app.config.settings import Settings
from app.domain.interfaces.llm import ProviderUnavailable
from app.domain.value_objects.tiers import CapabilityTier
from app.infrastructure.llm.providers.fake import FakeProvider
from app.infrastructure.llm.registry import PROVIDER_FACTORIES, ProviderRegistry
from tests.conftest import make_settings

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_returns_an_enabled_provider() -> None:
    registry = ProviderRegistry(make_settings(llm_enabled_providers=("fake",)))
    assert registry.get("fake").name == "fake"


def test_caches_provider_instances() -> None:
    registry = ProviderRegistry(make_settings(llm_enabled_providers=("fake",)))
    assert registry.get("fake") is registry.get("fake")


def test_disabled_provider_is_rejected_even_though_an_adapter_exists() -> None:
    """Enablement is explicit; a known adapter is not an enabled one (docs/SECURITY.md)."""
    registry = ProviderRegistry(make_settings(llm_enabled_providers=("fake",)))
    assert "groq" in PROVIDER_FACTORIES
    with pytest.raises(ProviderUnavailable, match="not enabled"):
        registry.get("groq")


def test_unknown_provider_names_the_known_adapters() -> None:
    registry = ProviderRegistry(make_settings(llm_enabled_providers=("nonsense",)))
    with pytest.raises(ProviderUnavailable, match="no adapter"):
        registry.get("nonsense")


def test_enabled_groq_without_a_key_fails_at_construction() -> None:
    """Missing credentials surface as configuration errors, not mid-analysis surprises."""
    registry = ProviderRegistry(make_settings(llm_enabled_providers=("groq",), groq_api_key=None))
    with pytest.raises(ProviderUnavailable, match="no API key"):
        registry.get("groq")


def test_optional_providers_are_absent_by_default() -> None:
    """Gemini, Mistral and Ollama must never be needed to run the project."""
    settings = Settings(_env_file=None)
    assert settings.llm_enabled_providers == ("groq", "fake")
    for optional in ("gemini", "mistral", "ollama"):
        assert optional not in settings.llm_enabled_providers


def test_overrides_bypass_the_factory() -> None:
    fake = FakeProvider()
    registry = ProviderRegistry(make_settings(llm_enabled_providers=()), overrides={"fake": fake})
    assert registry.get("fake") is fake
    assert registry.is_enabled("fake")


async def test_aclose_closes_every_instantiated_provider() -> None:
    fake = FakeProvider()
    registry = ProviderRegistry(make_settings(), overrides={"fake": fake})
    registry.get("fake")
    await registry.aclose()
    assert fake.closed is True


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_no_model_names_are_defaulted() -> None:
    """Model identifiers are vendor inventory; a built-in default would be an outage."""
    settings = Settings(_env_file=None)
    assert settings.model_reasoning is None
    assert settings.model_standard is None
    assert settings.model_fast is None


def test_groq_is_the_default_provider() -> None:
    assert Settings(_env_file=None).llm_default_provider == "groq"


def test_model_for_tier_maps_each_tier() -> None:
    settings = make_settings()
    assert settings.model_for_tier("reasoning") == "test-reasoning-model"
    assert settings.model_for_tier("standard") == "test-standard-model"
    assert settings.model_for_tier("fast") == "test-fast-model"
    assert settings.model_for_tier("nonexistent") is None


def test_api_keys_are_secrets_and_do_not_leak_in_repr() -> None:
    settings = make_settings(groq_api_key="super-secret-value")
    assert "super-secret-value" not in repr(settings)
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "super-secret-value"


def test_env_prefix_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CCE_MODEL_FAST", "from-environment")
    assert Settings(_env_file=None).model_fast == "from-environment"


# ---------------------------------------------------------------------------
# Agent bindings
# ---------------------------------------------------------------------------


def test_loads_the_shipped_binding_file(bindings: AgentBindings) -> None:
    assert bindings.version == 1
    assert bindings.for_agent("consequence-judge").tier is CapabilityTier.REASONING
    assert bindings.for_agent("change-decomposer").tier is CapabilityTier.FAST


def test_unknown_agents_fall_back_to_defaults(bindings: AgentBindings) -> None:
    assert bindings.for_agent("not-an-agent") == bindings.defaults


def test_defaults_merge_into_partial_entries(tmp_path: Path) -> None:
    config = tmp_path / "agents.yaml"
    config.write_text(
        "version: 2\n"
        "defaults:\n"
        "  tier: standard\n"
        "  max_output_tokens: 1234\n"
        "  temperature: 0.1\n"
        "agents:\n"
        "  only-tier-specified:\n"
        "    tier: reasoning\n",
        encoding="utf-8",
    )
    loaded = AgentBindings.load(config)
    binding = loaded.for_agent("only-tier-specified")
    assert binding.tier is CapabilityTier.REASONING
    assert binding.max_output_tokens == 1234  # inherited
    assert binding.temperature == 0.1  # inherited


def test_internal_roles_share_the_agent_namespace(bindings: AgentBindings) -> None:
    """The assumption extractor consumes the gateway exactly as an agent does."""
    assert "assumption-extractor" in bindings.bindings


def test_missing_binding_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AgentBindings.load(tmp_path / "absent.yaml")


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "agents.yaml"
    config.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        AgentBindings.load(config)


def test_invalid_tier_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "agents.yaml"
    config.write_text("agents:\n  x:\n    tier: turbo\n", encoding="utf-8")
    with pytest.raises(ValueError):
        AgentBindings.load(config)
