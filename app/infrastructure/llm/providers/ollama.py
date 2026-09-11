"""Ollama adapter - optional local provider.

Not required for development, tests, or CI.

Ollama is the only correct choice for a corpus that may not leave the network boundary
(docs/SECURITY.md, multi-provider section). It needs no credential, which is why
``requires_api_key`` is False.

Unverified against a live Ollama instance; see docs/STATUS.md.
"""

from __future__ import annotations

import httpx

from app.infrastructure.llm.providers._openai_compat import OpenAICompatibleProvider

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(OpenAICompatibleProvider):
    provider_name = "ollama"
    requires_api_key = False

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Ollama exposes its OpenAI-compatible surface under /v1.
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        super().__init__(base_url=normalized, api_key=None, timeout_s=timeout_s, client=client)
