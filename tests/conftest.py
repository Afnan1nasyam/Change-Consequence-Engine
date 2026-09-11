"""Shared fixtures.

Two invariants this file exists to protect:

1. Tests never read the developer's real ``.env``. Every ``Settings`` instance is built
   with ``_env_file=None`` and explicit values, so a local credential cannot change a
   test outcome.
2. Tests never touch the network. The default provider is ``fake``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config.agent_bindings import AgentBindings
from app.config.settings import REPO_ROOT, Settings
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.providers.fake import FakeProvider
from app.infrastructure.llm.registry import ProviderRegistry
from app.infrastructure.observability.usage import InMemoryUsageLedger


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def make_settings(**overrides: Any) -> Settings:
    """Settings isolated from the ambient environment."""
    defaults: dict[str, Any] = {
        "environment": "ci",
        "llm_default_provider": "fake",
        "llm_enabled_providers": ("fake",),
        "model_reasoning": "test-reasoning-model",
        "model_standard": "test-standard-model",
        "model_fast": "test-fast-model",
        "llm_backoff_base_s": 0.001,
        "llm_backoff_max_s": 0.002,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def bindings() -> AgentBindings:
    """The real ``agents.yaml``. Tests assert against shipped configuration, not a stub."""
    return AgentBindings.load(REPO_ROOT / "app" / "config" / "agents.yaml")


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def ledger() -> InMemoryUsageLedger:
    return InMemoryUsageLedger()


@pytest.fixture
def gateway(
    settings: Settings,
    bindings: AgentBindings,
    fake_provider: FakeProvider,
    ledger: InMemoryUsageLedger,
) -> LlmGateway:
    registry = ProviderRegistry(settings, overrides={"fake": fake_provider})

    async def instant_sleep(_: float) -> None:
        """Retry backoff without wall-clock cost."""
        return None

    return LlmGateway(
        settings=settings,
        registry=registry,
        bindings=bindings,
        ledger=ledger,
        sleeper=instant_sleep,
    )


@pytest.fixture
def client(settings: Settings, fake_provider: FakeProvider) -> Iterator[TestClient]:
    """FastAPI test client wired to the fake provider."""
    from app.api import deps
    from app.main import create_app

    deps.reset_dependency_caches()

    app = create_app(settings)
    app.dependency_overrides[deps.get_app_settings] = lambda: settings

    registry = ProviderRegistry(settings, overrides={"fake": fake_provider})
    test_gateway = LlmGateway(
        settings=settings,
        registry=registry,
        bindings=AgentBindings.load(settings.agents_config_path),
        ledger=InMemoryUsageLedger(),
    )
    app.dependency_overrides[deps.get_gateway] = lambda: test_gateway

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    deps.reset_dependency_caches()
