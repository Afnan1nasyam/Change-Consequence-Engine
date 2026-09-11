"""Gateway behaviour: resolution, budgets, retries, accounting, structured output."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.agent_bindings import AgentBindings
from app.domain.interfaces.llm import (
    Message,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderTransportError,
    ProviderUnavailable,
    Role,
    StructuredOutputError,
    TokenBudgetExceeded,
)
from app.infrastructure.llm.gateway import LlmGateway
from app.infrastructure.llm.providers.fake import FakeProvider
from app.infrastructure.llm.registry import ProviderRegistry
from app.infrastructure.observability.usage import InMemoryUsageLedger
from tests.conftest import make_settings

PROMPT = [Message(role=Role.USER, content="analyse this change")]


async def instant_sleep(_: float) -> None:
    return None


def build_gateway(
    *,
    provider: FakeProvider | None = None,
    ledger: InMemoryUsageLedger | None = None,
    **setting_overrides: object,
) -> tuple[LlmGateway, FakeProvider, InMemoryUsageLedger]:
    settings = make_settings(**setting_overrides)
    fake = provider or FakeProvider()
    used_ledger = ledger or InMemoryUsageLedger()
    registry = ProviderRegistry(settings, overrides={"fake": fake})
    gateway = LlmGateway(
        settings=settings,
        registry=registry,
        bindings=AgentBindings.load(settings.agents_config_path),
        ledger=used_ledger,
        sleeper=instant_sleep,
    )
    return gateway, fake, used_ledger


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolves_agent_to_tier_provider_and_model(gateway: LlmGateway) -> None:
    resolved = gateway.resolve("consequence-judge")
    assert resolved.agent == "consequence-judge"
    assert resolved.tier == "reasoning"
    assert resolved.provider == "fake"
    assert resolved.model == "test-reasoning-model"


def test_tier_selects_a_different_model_per_tier(gateway: LlmGateway) -> None:
    assert gateway.resolve("change-decomposer").model == "test-fast-model"
    assert gateway.resolve("policy-analyst").model == "test-standard-model"
    assert gateway.resolve("consequence-hunter").model == "test-reasoning-model"


def test_agents_never_name_a_model_themselves(gateway: LlmGateway) -> None:
    """Every resolution must come from configuration, i.e. the test's settings."""
    configured = {"test-reasoning-model", "test-standard-model", "test-fast-model"}
    for agent in ("policy-analyst", "consequence-judge", "report-composer"):
        assert gateway.resolve(agent).model in configured


def test_unconfigured_tier_names_the_variable_to_set() -> None:
    gateway, _, _ = build_gateway(model_reasoning=None)
    with pytest.raises(ProviderUnavailable, match="CCE_MODEL_REASONING"):
        gateway.resolve("consequence-judge")


def test_unknown_agent_falls_back_to_defaults(gateway: LlmGateway) -> None:
    resolved = gateway.resolve("agent-that-does-not-exist-yet")
    assert resolved.tier == "standard"
    assert resolved.model == "test-standard-model"


def test_binding_to_a_disabled_provider_is_rejected() -> None:
    settings = make_settings(llm_default_provider="groq", llm_enabled_providers=("fake",))
    gateway = LlmGateway(
        settings=settings,
        registry=ProviderRegistry(settings, overrides={"fake": FakeProvider()}),
        bindings=AgentBindings.load(settings.agents_config_path),
    )
    with pytest.raises(ProviderUnavailable, match="not enabled"):
        gateway.resolve("policy-analyst")


def test_per_agent_provider_override_is_honoured(tmp_path: Path) -> None:
    """Provider diversity (docs/EVALUATION.md F2-F5) must be config-only."""
    config = tmp_path / "agents.yaml"
    config.write_text(
        "version: 1\n"
        "defaults:\n  tier: standard\n"
        "agents:\n"
        "  blind-spot-hunter:\n"
        "    tier: reasoning\n"
        "    provider: fake\n",
        encoding="utf-8",
    )
    settings = make_settings(llm_default_provider="groq", agents_config_path=config)
    gateway = LlmGateway(
        settings=settings,
        registry=ProviderRegistry(settings, overrides={"fake": FakeProvider()}),
        bindings=AgentBindings.load(config),
    )
    assert gateway.resolve("blind-spot-hunter").provider == "fake"


def test_model_env_override_is_read_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "agents.yaml"
    config.write_text(
        "version: 1\n"
        "agents:\n"
        "  blind-spot-hunter:\n"
        "    tier: reasoning\n"
        "    provider: fake\n"
        "    model_env: CCE_MODEL_BLINDSPOT\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CCE_MODEL_BLINDSPOT", "override-model")
    settings = make_settings(agents_config_path=config)
    gateway = LlmGateway(
        settings=settings,
        registry=ProviderRegistry(settings, overrides={"fake": FakeProvider()}),
        bindings=AgentBindings.load(config),
    )
    assert gateway.resolve("blind-spot-hunter").model == "override-model"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def test_completes_and_records_usage() -> None:
    gateway, fake, ledger = build_gateway()
    response = await gateway.complete_for_agent("policy-analyst", PROMPT)

    assert response.provider == "fake"
    assert response.model == "test-standard-model"
    assert fake.call_count == 1

    assert ledger.call_count == 1
    entry = ledger.entries[0]
    assert entry.agent == "policy-analyst"
    assert entry.tier == "standard"
    assert entry.provider == "fake"
    assert entry.status == "ok"
    assert entry.usage.total_tokens > 0


async def test_request_carries_agent_and_correlation_id() -> None:
    gateway, fake, _ = build_gateway()
    await gateway.complete_for_agent("risk-analyst", PROMPT, correlation_id="corr-1")
    assert fake.calls[0].agent == "risk-analyst"
    assert fake.calls[0].correlation_id == "corr-1"


async def test_binding_supplies_max_output_tokens() -> None:
    gateway, fake, _ = build_gateway(llm_max_output_tokens=99_999)
    await gateway.complete_for_agent("report-composer", PROMPT)
    # report-composer is bound to 6144 in the shipped agents.yaml.
    assert fake.calls[0].max_output_tokens == 6144


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


async def test_budget_is_enforced_before_dispatch() -> None:
    gateway, fake, _ = build_gateway(llm_analysis_token_budget=10)
    with pytest.raises(TokenBudgetExceeded):
        await gateway.complete_for_agent("policy-analyst", PROMPT)
    # The point of pre-dispatch enforcement: the provider was never called.
    assert fake.call_count == 0


async def test_spent_tokens_accumulate_and_reset() -> None:
    gateway, _, _ = build_gateway()
    assert gateway.tokens_spent == 0
    await gateway.complete_for_agent("policy-analyst", PROMPT)
    spent = gateway.tokens_spent
    assert spent > 0
    assert gateway.tokens_remaining < 250_000

    gateway.reset_budget()
    assert gateway.tokens_spent == 0


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


async def test_retries_retryable_failures_then_succeeds() -> None:
    fake = FakeProvider(
        failures=[
            ProviderRateLimited("slow down", provider="fake", retry_after_s=0.001),
            ProviderTransportError("connection reset", provider="fake"),
        ]
    )
    gateway, _, ledger = build_gateway(provider=fake, llm_max_attempts=3)

    response = await gateway.complete_for_agent("policy-analyst", PROMPT)

    assert response.provider == "fake"
    assert fake.call_count == 3
    assert ledger.entries[-1].attempts == 3


async def test_does_not_retry_terminal_failures() -> None:
    fake = FakeProvider(failures=[ProviderBadRequest("malformed", provider="fake")])
    gateway, _, ledger = build_gateway(provider=fake, llm_max_attempts=3)

    with pytest.raises(ProviderBadRequest):
        await gateway.complete_for_agent("policy-analyst", PROMPT)

    assert fake.call_count == 1
    assert ledger.failures()[0].status == "ProviderBadRequest"


async def test_gives_up_after_max_attempts() -> None:
    fake = FakeProvider(
        failures=[ProviderTransportError(f"boom {i}", provider="fake") for i in range(5)]
    )
    gateway, _, _ = build_gateway(provider=fake, llm_max_attempts=2)

    with pytest.raises(ProviderTransportError):
        await gateway.complete_for_agent("policy-analyst", PROMPT)

    assert fake.call_count == 2


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


async def test_complete_json_parses_valid_json() -> None:
    fake = FakeProvider(responses=['{"findings": []}'])
    gateway, _, _ = build_gateway(provider=fake)

    parsed, _ = await gateway.complete_json("policy-analyst", PROMPT)
    assert parsed == {"findings": []}
    assert fake.call_count == 1


async def test_complete_json_tolerates_a_markdown_fence() -> None:
    fake = FakeProvider(responses=['```json\n{"ok": true}\n```'])
    gateway, _, _ = build_gateway(provider=fake)

    parsed, _ = await gateway.complete_json("policy-analyst", PROMPT)
    assert parsed == {"ok": True}
    assert fake.call_count == 1


async def test_complete_json_repairs_once() -> None:
    fake = FakeProvider(responses=["not json at all", '{"repaired": true}'])
    gateway, _, _ = build_gateway(provider=fake)

    parsed, _ = await gateway.complete_json("policy-analyst", PROMPT)
    assert parsed == {"repaired": True}
    assert fake.call_count == 2
    # The repair turn must show the model its own bad output plus an instruction.
    assert fake.calls[1].messages[-1].role is Role.USER
    assert len(fake.calls[1].messages) == len(PROMPT) + 2


async def test_complete_json_discards_after_second_failure() -> None:
    fake = FakeProvider(responses=["still not json", "nope"])
    gateway, _, _ = build_gateway(provider=fake)

    with pytest.raises(StructuredOutputError, match="invalid JSON twice"):
        await gateway.complete_json("policy-analyst", PROMPT)

    assert fake.call_count == 2


async def test_json_request_flags_schema_to_the_provider() -> None:
    fake = FakeProvider(responses=["{}"])
    gateway, _, _ = build_gateway(provider=fake)

    await gateway.complete_json("policy-analyst", PROMPT, json_schema={"type": "object"})
    assert fake.calls[0].json_schema == {"type": "object"}
