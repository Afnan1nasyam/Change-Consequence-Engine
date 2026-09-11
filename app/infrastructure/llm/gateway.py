"""LLM gateway.

The single point through which every model call passes. It owns the concerns that would
otherwise be duplicated across adapters or leaked into agents:

* agent -> tier -> provider/model resolution, from configuration only
* token budgets, enforced *before* dispatch
* retries with exponential backoff and jitter, retryable failures only
* usage and cost accounting attributed to agent and analysis
* structured-output validation with one bounded repair attempt

Adapters stay dumb on purpose: one method, vendor translation only.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from app.config.agent_bindings import AgentBindings
from app.config.settings import Settings
from app.domain.interfaces.llm import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderUnavailable,
    ResolvedBinding,
    Role,
    StructuredOutputError,
    TokenBudgetExceeded,
    UsageLedgerPort,
    UsageRecord,
)
from app.infrastructure.llm.registry import ProviderRegistry
from app.infrastructure.observability.logging import get_logger

logger = get_logger(__name__)

Sleeper = Callable[[float], Awaitable[None]]

_REPAIR_INSTRUCTION = (
    "Your previous response was not valid JSON. Return the same content as a single "
    "valid JSON object with no surrounding prose, no markdown fence, and no commentary."
)


class LlmGateway:
    """Implements :class:`~app.domain.interfaces.llm.AgentCompletionPort`."""

    def __init__(
        self,
        *,
        settings: Settings,
        registry: ProviderRegistry,
        bindings: AgentBindings,
        ledger: UsageLedgerPort | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._bindings = bindings
        self._ledger = ledger
        # Injectable so retry tests do not spend real wall-clock time.
        self._sleep: Sleeper = sleeper or asyncio.sleep
        self._tokens_spent = 0

    # -- budget -----------------------------------------------------------

    @property
    def tokens_spent(self) -> int:
        return self._tokens_spent

    @property
    def tokens_remaining(self) -> int:
        return max(0, self._settings.llm_analysis_token_budget - self._tokens_spent)

    def reset_budget(self) -> None:
        self._tokens_spent = 0

    # -- resolution -------------------------------------------------------

    def resolve(self, agent: str) -> ResolvedBinding:
        """Resolve agent -> tier -> provider/model with no model call.

        Raises :class:`ProviderUnavailable` with the exact environment variable to set when
        a tier has no configured model. Model names are never defaulted or guessed.
        """
        binding = self._bindings.for_agent(agent)
        tier = binding.tier.value

        provider = binding.provider or self._settings.llm_default_provider

        model = binding.resolve_model_override()
        if model is None:
            model = self._settings.model_for_tier(tier)

        if not model:
            expected = binding.model_env or f"CCE_MODEL_{tier.upper()}"
            raise ProviderUnavailable(
                f"no model configured for agent '{agent}' (tier '{tier}'); set {expected}. "
                "Model identifiers come from configuration only - none are built in.",
                provider=provider,
            )

        if not self._registry.is_enabled(provider):
            raise ProviderUnavailable(
                f"agent '{agent}' is bound to provider '{provider}', which is not enabled; "
                f"enabled providers are {list(self._registry.enabled)}",
                provider=provider,
            )

        return ResolvedBinding(
            agent=agent,
            tier=tier,
            provider=provider,
            model=model,
            max_output_tokens=binding.max_output_tokens,
            temperature=binding.temperature,
        )

    # -- completion -------------------------------------------------------

    async def complete_for_agent(
        self,
        agent: str,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        correlation_id: str | None = None,
    ) -> CompletionResponse:
        binding = self.resolve(agent)

        cap = min(
            max_output_tokens or binding.max_output_tokens,
            self._settings.llm_max_output_tokens
            if max_output_tokens is None
            else max_output_tokens,
        )

        # Enforced before dispatch: a budget checked after the fact is not a budget.
        if self._tokens_spent + cap > self._settings.llm_analysis_token_budget:
            raise TokenBudgetExceeded(
                f"analysis token budget exhausted for agent '{agent}': "
                f"spent {self._tokens_spent}, requested up to {cap}, "
                f"budget {self._settings.llm_analysis_token_budget}",
                provider=binding.provider,
            )

        request = CompletionRequest(
            model=binding.model,
            messages=list(messages),
            max_output_tokens=cap,
            temperature=binding.temperature if temperature is None else temperature,
            json_schema=json_schema,
            correlation_id=correlation_id,
            agent=agent,
        )

        return await self._dispatch(binding, request, correlation_id)

    async def _dispatch(
        self,
        binding: ResolvedBinding,
        request: CompletionRequest,
        correlation_id: str | None,
    ) -> CompletionResponse:
        provider = self._registry.get(binding.provider)
        attempts = 0
        last_error: ProviderError | None = None

        while attempts < self._settings.llm_max_attempts:
            attempts += 1
            try:
                response = await provider.complete(request)
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempts >= self._settings.llm_max_attempts:
                    self._record(
                        binding,
                        None,
                        attempts=attempts,
                        status=type(exc).__name__,
                        correlation_id=correlation_id,
                    )
                    raise
                delay = self._backoff_delay(attempts, exc)
                logger.warning(
                    "provider call failed, retrying",
                    extra={
                        "agent": binding.agent,
                        "provider": binding.provider,
                        "attempt": attempts,
                        "delay_s": round(delay, 3),
                        "error": type(exc).__name__,
                        "correlation_id": correlation_id,
                    },
                )
                await self._sleep(delay)
                continue

            self._tokens_spent += response.usage.total_tokens
            self._record(
                binding, response, attempts=attempts, status="ok", correlation_id=correlation_id
            )
            return response

        # Unreachable: the loop either returns or raises.
        raise last_error or ProviderError("dispatch failed", provider=binding.provider)

    def _backoff_delay(self, attempt: int, error: ProviderError) -> float:
        retry_after = getattr(error, "retry_after_s", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            return min(float(retry_after), self._settings.llm_backoff_max_s)

        base = self._settings.llm_backoff_base_s * (2 ** (attempt - 1))
        capped = min(base, self._settings.llm_backoff_max_s)
        # Full jitter: avoids a retry stampede when several agents are rate-limited at once.
        return random.uniform(0.0, capped)

    def _record(
        self,
        binding: ResolvedBinding,
        response: CompletionResponse | None,
        *,
        attempts: int,
        status: str,
        correlation_id: str | None,
    ) -> None:
        if self._ledger is None:
            return
        from app.domain.interfaces.llm import TokenUsage

        self._ledger.record(
            UsageRecord(
                agent=binding.agent,
                tier=binding.tier,
                provider=binding.provider,
                model=binding.model,
                usage=response.usage if response else TokenUsage(),
                latency_ms=response.latency_ms if response else None,
                attempts=attempts,
                status=status,
                correlation_id=correlation_id,
            )
        )

    # -- structured output ------------------------------------------------

    async def complete_json(
        self,
        agent: str,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> tuple[Any, CompletionResponse]:
        """Completion that must yield JSON, with exactly one repair attempt.

        On second failure the caller gets :class:`StructuredOutputError` and the finding is
        discarded. Malformed output is never coerced into a partially populated finding
        (``docs/ARCHITECTURE.md`` 12).
        """
        response = await self.complete_for_agent(
            agent, messages, json_schema=json_schema, correlation_id=correlation_id
        )

        parsed = _try_parse_json(response.text)
        if parsed is not None:
            return parsed, response

        logger.warning(
            "structured output invalid, attempting single repair",
            extra={"agent": agent, "correlation_id": correlation_id},
        )
        repair_messages = [
            *messages,
            Message(role=Role.ASSISTANT, content=response.text),
            Message(role=Role.USER, content=_REPAIR_INSTRUCTION),
        ]
        repaired = await self.complete_for_agent(
            agent, repair_messages, json_schema=json_schema, correlation_id=correlation_id
        )

        parsed = _try_parse_json(repaired.text)
        if parsed is None:
            raise StructuredOutputError(
                f"agent '{agent}' produced invalid JSON twice; finding discarded",
                provider=repaired.provider,
            )
        return parsed, repaired


def _try_parse_json(text: str) -> Any | None:
    """Parse JSON, tolerating a markdown fence but nothing more inventive."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else ""
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[: -len("```")]
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None
