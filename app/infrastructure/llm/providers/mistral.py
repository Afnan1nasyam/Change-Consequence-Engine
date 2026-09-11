"""Mistral adapter - optional provider.

Not required for development, tests, or CI. Absence is never an error: the provider is only
constructed when it appears in ``CCE_LLM_ENABLED_PROVIDERS``.

Mistral's chat-completions endpoint is OpenAI-compatible, so this is a base-URL binding
only. Unverified against the live API; see docs/STATUS.md.
"""

from __future__ import annotations

import httpx

from app.infrastructure.llm.providers._openai_compat import OpenAICompatibleProvider

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"


class MistralProvider(OpenAICompatibleProvider):
    provider_name = "mistral"
    requires_api_key = True

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, timeout_s=timeout_s, client=client)
