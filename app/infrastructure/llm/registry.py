"""Provider registry.

Providers are constructed lazily and only when explicitly enabled. Enablement comes from
``CCE_LLM_ENABLED_PROVIDERS`` and is never inferred from which API keys happen to be
present: silently activating a provider because a key exists would widen the data-egress
surface without anyone deciding to (docs/SECURITY.md).
"""

from __future__ import annotations

from collections.abc import Callable

from app.config.settings import Settings
from app.domain.interfaces.llm import ProviderPort, ProviderUnavailable

ProviderFactory = Callable[[Settings], ProviderPort]


def _build_groq(settings: Settings) -> ProviderPort:
    from app.infrastructure.llm.providers.groq import GroqProvider

    return GroqProvider(
        api_key=settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,
        base_url=settings.groq_base_url,
        timeout_s=settings.llm_timeout_s,
    )


def _build_gemini(settings: Settings) -> ProviderPort:
    from app.infrastructure.llm.providers.gemini import GeminiProvider

    return GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None,
        base_url=settings.gemini_base_url,
        timeout_s=settings.llm_timeout_s,
    )


def _build_mistral(settings: Settings) -> ProviderPort:
    from app.infrastructure.llm.providers.mistral import MistralProvider

    return MistralProvider(
        api_key=settings.mistral_api_key.get_secret_value() if settings.mistral_api_key else None,
        base_url=settings.mistral_base_url,
        timeout_s=settings.llm_timeout_s,
    )


def _build_ollama(settings: Settings) -> ProviderPort:
    from app.infrastructure.llm.providers.ollama import OllamaProvider

    return OllamaProvider(base_url=settings.ollama_base_url, timeout_s=settings.llm_timeout_s)


def _build_fake(_: Settings) -> ProviderPort:
    from app.infrastructure.llm.providers.fake import FakeProvider

    return FakeProvider()


#: Every provider CCE knows how to build. Membership here does not enable a provider.
PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "groq": _build_groq,
    "gemini": _build_gemini,
    "mistral": _build_mistral,
    "ollama": _build_ollama,
    "fake": _build_fake,
}


class ProviderRegistry:
    """Lazily instantiates and caches enabled providers."""

    def __init__(
        self,
        settings: Settings,
        *,
        overrides: dict[str, ProviderPort] | None = None,
    ) -> None:
        self._settings = settings
        self._enabled = tuple(settings.llm_enabled_providers)
        # Pre-built instances win over factories. This is how tests inject a FakeProvider
        # and how a future failover policy could pin a specific client.
        self._instances: dict[str, ProviderPort] = dict(overrides or {})

    @property
    def enabled(self) -> tuple[str, ...]:
        return self._enabled

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled or name in self._instances

    def get(self, name: str) -> ProviderPort:
        if name in self._instances:
            return self._instances[name]

        if name not in self._enabled:
            raise ProviderUnavailable(
                f"provider '{name}' is not enabled; "
                f"enabled providers are {list(self._enabled)}. "
                f"Add it to CCE_LLM_ENABLED_PROVIDERS to allow it.",
                provider=name,
            )

        factory = PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise ProviderUnavailable(
                f"provider '{name}' has no adapter; "
                f"known adapters are {sorted(PROVIDER_FACTORIES)}",
                provider=name,
            )

        # Construction may itself raise ProviderUnavailable (missing credential), which is
        # correct: a configuration error should surface here, not mid-analysis.
        instance = factory(self._settings)
        self._instances[name] = instance
        return instance

    async def aclose(self) -> None:
        for provider in self._instances.values():
            await provider.aclose()
        self._instances.clear()
