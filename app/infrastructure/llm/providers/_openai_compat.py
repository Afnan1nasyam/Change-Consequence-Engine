"""Shared adapter for providers exposing an OpenAI-compatible chat-completions endpoint.

Groq, Mistral and Ollama all speak this shape, so they share one implementation rather
than three near-identical ones. Providers with a genuinely different wire format get their
own adapter.

Deliberately built on ``httpx`` rather than a vendor SDK. Four remote providers behind one
HTTP client keeps the adapter surface uniform, keeps vendor SDKs out of the dependency
tree entirely, and avoids the framework lock-in that ``CLAUDE.md`` principle 11 warns
about.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.domain.interfaces.llm import (
    CompletionRequest,
    CompletionResponse,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTransportError,
    ProviderUnavailable,
    TokenUsage,
)


class OpenAICompatibleProvider:
    """Adapter for ``POST {base_url}/chat/completions``.

    Subclasses set :attr:`provider_name` and pass credentials/base URL.
    """

    provider_name: str = "openai-compatible"

    #: Some deployments (Ollama) need no credential at all.
    requires_api_key: bool = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if self.requires_api_key and not api_key:
            raise ProviderUnavailable(
                f"{self.provider_name} is enabled but no API key is configured; "
                f"set CCE_{self.provider_name.upper()}_API_KEY",
                provider=self.provider_name,
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @property
    def name(self) -> str:
        return self.provider_name

    # -- request/response mapping ----------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.json_schema is not None:
            # Only json_object is portable across these providers. Schema conformance is
            # verified by the gateway, which is where the repair loop lives anyway.
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self._base_url}/chat/completions"
        started = time.perf_counter()
        try:
            response = await self._client.post(
                url, headers=self._headers(), json=self._payload(request)
            )
        except httpx.TimeoutException as exc:
            raise ProviderTransportError(
                f"{self.provider_name} request timed out after {self._timeout_s}s",
                provider=self.provider_name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderTransportError(
                f"{self.provider_name} transport failure: {exc}", provider=self.provider_name
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        self._raise_for_status(response)
        return self._parse(response, request, latency_ms)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        detail = self._error_detail(response)

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimited(
                f"{self.provider_name} rate limited: {detail}",
                provider=self.provider_name,
                retry_after_s=_as_float(retry_after),
            )
        if response.status_code >= 500:
            raise ProviderTransportError(
                f"{self.provider_name} server error {response.status_code}: {detail}",
                provider=self.provider_name,
            )
        if response.status_code in (401, 403):
            # Credential problems are configuration errors, not transient failures.
            raise ProviderUnavailable(
                f"{self.provider_name} rejected credentials ({response.status_code}): {detail}",
                provider=self.provider_name,
            )
        raise ProviderBadRequest(
            f"{self.provider_name} rejected request {response.status_code}: {detail}",
            provider=self.provider_name,
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text[:500]
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    return message
            if isinstance(error, str):
                return error
        return str(body)[:500]

    def _parse(
        self, response: httpx.Response, request: CompletionRequest, latency_ms: int
    ) -> CompletionResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.provider_name} returned non-JSON body", provider=self.provider_name
            ) from exc

        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(
                f"{self.provider_name} returned no choices", provider=self.provider_name
            )

        message = choices[0].get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            raise ProviderError(
                f"{self.provider_name} returned no message content",
                provider=self.provider_name,
            )

        usage_raw = body.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("prompt_tokens") or 0),
            output_tokens=int(usage_raw.get("completion_tokens") or 0),
        )

        return CompletionResponse(
            text=text,
            provider=self.provider_name,
            model=str(body.get("model") or request.model),
            usage=usage,
            finish_reason=choices[0].get("finish_reason"),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
