"""Provider adapter tests.

The OpenAI-compatible adapter is exercised through ``httpx.MockTransport``: real request
mapping, real response parsing, real status-code translation, no network. This is where
the Groq wire contract is pinned.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.domain.interfaces.llm import (
    CompletionRequest,
    Message,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTransportError,
    ProviderUnavailable,
    Role,
)
from app.infrastructure.llm.providers.groq import GroqProvider
from app.infrastructure.llm.providers.ollama import OllamaProvider

REQUEST = CompletionRequest(
    model="configured-model",
    messages=[
        Message(role=Role.SYSTEM, content="you are a policy analyst"),
        Message(role=Role.USER, content="what breaks"),
    ],
    max_output_tokens=512,
    temperature=0.0,
    agent="policy-analyst",
)

OK_BODY: dict[str, Any] = {
    "model": "configured-model",
    "choices": [{"message": {"role": "assistant", "content": '{"ok":1}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 31, "completion_tokens": 7},
}


Handler = Callable[[httpx.Request], httpx.Response]


def provider_with(handler: Handler) -> GroqProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqProvider(api_key="test-key", client=client)


async def test_maps_a_successful_response() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    provider = provider_with(handler)
    response = await provider.complete(REQUEST)

    assert response.text == '{"ok":1}'
    assert response.provider == "groq"
    assert response.model == "configured-model"
    assert response.usage.input_tokens == 31
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 38
    assert response.finish_reason == "stop"
    assert response.latency_ms is not None

    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "configured-model"
    assert captured["body"]["max_tokens"] == 512
    assert captured["body"]["messages"][0] == {
        "role": "system",
        "content": "you are a policy analyst",
    }

    await provider.aclose()


async def test_json_schema_requests_json_object_response_format() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    provider = provider_with(handler)
    await provider.complete(REQUEST.model_copy(update={"json_schema": {"type": "object"}}))
    assert captured["body"]["response_format"] == {"type": "json_object"}
    await provider.aclose()


async def test_does_not_send_internal_fields_to_the_provider() -> None:
    """`agent` and `correlation_id` are for tracing, not for the vendor."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    provider = provider_with(handler)
    await provider.complete(REQUEST.model_copy(update={"correlation_id": "corr-9"}))
    assert "agent" not in captured["body"]
    assert "correlation_id" not in captured["body"]
    await provider.aclose()


@pytest.mark.parametrize(
    "status,expected",
    [
        (429, ProviderRateLimited),
        (500, ProviderTransportError),
        (503, ProviderTransportError),
        (401, ProviderUnavailable),
        (403, ProviderUnavailable),
        (400, ProviderBadRequest),
        (404, ProviderBadRequest),
    ],
)
async def test_translates_status_codes(status: int, expected: type[ProviderError]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "upstream detail"}})

    provider = provider_with(handler)
    with pytest.raises(expected, match="upstream detail"):
        await provider.complete(REQUEST)
    await provider.aclose()


async def test_retry_after_header_is_surfaced() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "2.5"}, json={"error": "slow"})

    provider = provider_with(handler)
    with pytest.raises(ProviderRateLimited) as exc_info:
        await provider.complete(REQUEST)
    assert exc_info.value.retry_after_s == 2.5
    await provider.aclose()


async def test_timeout_becomes_a_retryable_transport_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    provider = provider_with(handler)
    with pytest.raises(ProviderTransportError) as exc_info:
        await provider.complete(REQUEST)
    assert exc_info.value.retryable is True
    await provider.aclose()


async def test_empty_choices_is_an_error_not_an_empty_success() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    provider = provider_with(handler)
    with pytest.raises(ProviderError, match="no choices"):
        await provider.complete(REQUEST)
    await provider.aclose()


async def test_non_json_body_is_an_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    provider = provider_with(handler)
    with pytest.raises(ProviderError, match="non-JSON"):
        await provider.complete(REQUEST)
    await provider.aclose()


def test_missing_api_key_fails_at_construction() -> None:
    """A configuration error must surface up front, not mid-analysis."""
    with pytest.raises(ProviderUnavailable, match="no API key"):
        GroqProvider(api_key=None)


def test_ollama_needs_no_credential_and_normalises_its_base_url() -> None:
    provider = OllamaProvider(base_url="http://localhost:11434")
    assert provider.name == "ollama"
    assert provider._base_url.endswith("/v1")

    already = OllamaProvider(base_url="http://localhost:11434/v1/")
    assert already._base_url.endswith("/v1")
    assert not already._base_url.endswith("/v1/v1")
