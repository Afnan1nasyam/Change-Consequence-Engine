"""Google Gemini adapter - optional provider.

Not required for development, tests, or CI.

Gemini does not speak the OpenAI chat-completions shape, so this adapter maps the wire
format itself rather than reusing ``OpenAICompatibleProvider``. That is deliberate: a port
that only ever fronted one wire format would not have been shown to abstract anything.

UNVERIFIED against the live API - implemented from the documented ``generateContent``
contract but never executed, because no Gemini credential exists in this environment. See
docs/STATUS.md. Treat the mapping as reviewed-but-untested code.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.domain.interfaces.llm import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTransportError,
    ProviderUnavailable,
    Role,
    TokenUsage,
)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider:
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderUnavailable(
                "gemini is enabled but no API key is configured; set CCE_GEMINI_API_KEY",
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

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """Gemini carries system text in ``systemInstruction``, not in ``contents``."""
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        rest = [m for m in messages if m.role is not Role.SYSTEM]
        return ("\n\n".join(system_parts) or None), rest

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        system, turns = self._split_system(list(request.messages))

        generation_config: dict[str, Any] = {
            "maxOutputTokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if request.stop:
            generation_config["stopSequences"] = list(request.stop)
        if request.json_schema is not None:
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": [
                {
                    # Gemini's assistant role is "model".
                    "role": "model" if m.role is Role.ASSISTANT else "user",
                    "parts": [{"text": m.content}],
                }
                for m in turns
            ],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self._base_url}/models/{request.model}:generateContent"
        started = time.perf_counter()
        try:
            response = await self._client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key or "",
                },
                json=self._payload(request),
            )
        except httpx.TimeoutException as exc:
            raise ProviderTransportError(
                f"gemini request timed out after {self._timeout_s}s",
                provider=self.provider_name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderTransportError(
                f"gemini transport failure: {exc}", provider=self.provider_name
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)

        if not response.is_success:
            detail = response.text[:500]
            if response.status_code == 429:
                raise ProviderRateLimited(
                    f"gemini rate limited: {detail}", provider=self.provider_name
                )
            if response.status_code >= 500:
                raise ProviderTransportError(
                    f"gemini server error {response.status_code}: {detail}",
                    provider=self.provider_name,
                )
            if response.status_code in (401, 403):
                raise ProviderUnavailable(
                    f"gemini rejected credentials ({response.status_code}): {detail}",
                    provider=self.provider_name,
                )
            raise ProviderBadRequest(
                f"gemini rejected request {response.status_code}: {detail}",
                provider=self.provider_name,
            )

        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            # A prompt blocked by safety filtering lands here; surface it as a bad request
            # rather than an empty success, so the finding is discarded not silently lost.
            raise ProviderBadRequest(
                f"gemini returned no candidates: {str(body.get('promptFeedback'))[:300]}",
                provider=self.provider_name,
            )

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

        usage_raw = body.get("usageMetadata") or {}
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("promptTokenCount") or 0),
            output_tokens=int(usage_raw.get("candidatesTokenCount") or 0),
        )

        if not text:
            raise ProviderError("gemini returned empty text", provider=self.provider_name)

        return CompletionResponse(
            text=text,
            provider=self.provider_name,
            model=request.model,
            usage=usage,
            finish_reason=candidates[0].get("finishReason"),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
