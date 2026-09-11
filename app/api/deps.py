"""FastAPI dependency wiring.

Composition happens here so that route handlers depend on ports, not on concrete adapters.
"""

from __future__ import annotations

import functools

from app.agents.runtime.agent_loader import AgentDefinition, load_agent_definitions
from app.config.agent_bindings import AgentBindings
from app.config.settings import Settings, get_settings
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.registry import ProviderRegistry
from app.infrastructure.observability.usage import InMemoryUsageLedger


@functools.lru_cache(maxsize=1)
def get_agent_bindings() -> AgentBindings:
    return AgentBindings.load(get_settings().agents_config_path)


@functools.lru_cache(maxsize=1)
def get_agent_definitions() -> dict[str, AgentDefinition]:
    return load_agent_definitions()


@functools.lru_cache(maxsize=1)
def get_usage_ledger() -> InMemoryUsageLedger:
    return InMemoryUsageLedger()


@functools.lru_cache(maxsize=1)
def get_provider_registry() -> ProviderRegistry:
    return ProviderRegistry(get_settings())


@functools.lru_cache(maxsize=1)
def get_gateway() -> LlmGateway:
    return LlmGateway(
        settings=get_settings(),
        registry=get_provider_registry(),
        bindings=get_agent_bindings(),
        ledger=get_usage_ledger(),
    )


def get_app_settings() -> Settings:
    return get_settings()


def reset_dependency_caches() -> None:
    """Clear composition caches. Used by tests that change the environment."""
    for cached in (
        get_agent_bindings,
        get_agent_definitions,
        get_usage_ledger,
        get_provider_registry,
        get_gateway,
    ):
        cached.cache_clear()
    get_settings.cache_clear()
