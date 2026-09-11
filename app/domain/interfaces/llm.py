"""LLM provider port.

This module is the *only* contract the application layer uses to reach a language model.
It must never import a vendor SDK, an HTTP client, or a framework: adapters live in
``app/infrastructure/llm/providers/`` and are the sole holders of vendor knowledge.

The layering rule is asserted by ``tests/contract/test_layering.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """A single provider-agnostic model call.

    ``model`` is resolved by the gateway from configuration before the request reaches an
    adapter. Callers upstream of the gateway supply an agent name or a tier, not a model.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    messages: Sequence[Message]
    max_output_tokens: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stop: Sequence[str] | None = None

    # When set, the adapter must ask the provider for JSON output. Adapters that cannot
    # enforce this natively degrade to instruction-based JSON and the gateway validates.
    json_schema: Mapping[str, Any] | None = None

    # Propagated for tracing and cost attribution; never sent to the provider.
    correlation_id: str | None = None
    agent: str | None = None


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CompletionResponse(BaseModel):
    """Provider output plus the provenance needed to explain a finding after the fact.

    ``provider`` and ``model`` are carried on every response because provider identity is
    an experimental variable in the diversity study (``docs/EVALUATION.md``), not merely an
    operational detail.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    provider: str
    model: str
    usage: TokenUsage = TokenUsage()
    finish_reason: str | None = None
    latency_ms: int | None = None
    raw: Mapping[str, Any] | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base for every provider failure.

    The application layer catches these; it never catches a vendor exception type.
    """

    retryable: bool = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class ProviderUnavailable(ProviderError):
    """Provider is not configured, lacks credentials, or its dependency is absent.

    Raised at resolution time rather than call time so that a missing optional provider is
    a clear configuration error and never a mid-analysis surprise.
    """

    retryable = False


class ProviderRateLimited(ProviderError):
    retryable = True

    def __init__(
        self, message: str, *, provider: str | None = None, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after_s = retry_after_s


class ProviderTransportError(ProviderError):
    """Network-level failure: timeout, connection reset, 5xx."""

    retryable = True


class ProviderBadRequest(ProviderError):
    """The request itself is wrong. Retrying will not help."""

    retryable = False


class TokenBudgetExceeded(ProviderError):
    """Budget enforcement rejected the call *before* dispatch."""

    retryable = False


class StructuredOutputError(ProviderError):
    """Output did not validate against the requested schema after bounded repair.

    The finding is discarded and a parse failure recorded. Malformed output is never
    coerced into a partially populated finding.
    """

    retryable = False


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


@runtime_checkable
class ProviderPort(Protocol):
    """What every LLM adapter must implement.

    Deliberately one method. A wider port would tempt adapters into carrying orchestration
    concerns that belong in the gateway (budgets, retries, accounting, caching).
    """

    @property
    def name(self) -> str:
        """Stable provider identifier, e.g. ``"groq"``. Recorded on every finding."""
        ...

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute one completion, translating vendor failures into ``ProviderError``."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...


@runtime_checkable
class EmbeddingPort(Protocol):
    """Embeddings sit behind their own port, under the same no-vendor-above-infra rule."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class ResolvedBinding(BaseModel):
    """The outcome of agent -> tier -> provider/model resolution.

    Returned by the gateway so that resolution can be asserted in tests and logged without
    performing a model call.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    tier: str
    provider: str
    model: str
    max_output_tokens: int
    temperature: float


@runtime_checkable
class AgentCompletionPort(Protocol):
    """What the orchestrator and agent runtime depend on.

    Callers name an *agent*. Tier, provider and model are configuration concerns resolved
    below this line, which is what makes a provider-diversity experiment a config change
    rather than a code change.
    """

    def resolve(self, agent: str) -> ResolvedBinding:
        """Resolve an agent's runtime binding without calling a model."""
        ...

    async def complete_for_agent(
        self,
        agent: str,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        correlation_id: str | None = None,
    ) -> CompletionResponse: ...

    async def complete_json(
        self,
        agent: str,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> tuple[Any, CompletionResponse]:
        """Completion that must yield JSON, with bounded repair.

        Part of the port rather than a gateway convenience: every production caller wants
        structured output, and one that reached for `complete_for_agent` and parsed the text
        itself would be re-implementing the repair-then-discard rule that keeps malformed
        output from being coerced into a partial finding.
        """
        ...


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


class UsageRecord(BaseModel):
    """One model invocation, for cost accounting and provenance.

    ``provider`` and ``model`` are recorded per call because provider identity is an
    experimental variable, not just an operational detail (``docs/DATA_MODEL.md``).
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    tier: str
    provider: str
    model: str
    usage: TokenUsage
    latency_ms: int | None = None
    attempts: int = 1
    status: str = "ok"
    correlation_id: str | None = None


@runtime_checkable
class UsageLedgerPort(Protocol):
    def record(self, entry: UsageRecord) -> None: ...
