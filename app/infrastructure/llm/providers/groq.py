"""Groq adapter - the default remote provider.

Groq serves an OpenAI-compatible chat-completions API, so this adapter only pins the
provider identity and base URL. Model identifiers come from configuration
(``CCE_MODEL_REASONING`` / ``_STANDARD`` / ``_FAST``); none are hard-coded here, because
vendor inventory changes and a stale identifier is an outage.
"""

from __future__ import annotations

import httpx

from app.infrastructure.llm.providers._openai_compat import OpenAICompatibleProvider

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAICompatibleProvider):
    provider_name = "groq"
    requires_api_key = True

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            client=client,
        )
