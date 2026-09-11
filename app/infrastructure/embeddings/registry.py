"""Which embedder retrieval uses. Selection by configuration, enablement by decision.

The embedding counterpart of ``app/infrastructure/llm/registry.py``, and the same rule is
load-bearing: **enablement is explicit and is never inferred from which credentials happen
to be present.** A remote embedder sends corpus text - including spans from documents under
an ACL - to a third party. Activating one because a key appeared in the environment would
widen the data-egress surface without anyone deciding to (``docs/SECURITY.md``).

So membership in :data:`EMBEDDING_FACTORIES` means "CCE knows how to build this". It does
not mean the adapter can be used. That needs it in ``CCE_EMBEDDING_ENABLED_PROVIDERS``,
which ships as ``("hashing", "replay")`` - the two that reach no network.

Construction is lazy, and a missing credential raises :class:`ProviderUnavailable` **here**
rather than at call time, so a misconfiguration surfaces when the embedder is selected and
never in the middle of a run.

Nothing in this module performs I/O, and nothing above the infrastructure layer imports it:
callers are handed an ``EmbeddingPort`` and cannot tell which adapter answered.
"""

from __future__ import annotations

from collections.abc import Callable

from app.config.settings import Settings
from app.domain.interfaces.llm import EmbeddingPort, ProviderUnavailable

EmbeddingFactory = Callable[[Settings], EmbeddingPort]

#: Enabled by default, because ``ARCHITECTURE.md`` 11.4 requires the default embedder to
#: need no API key. Both of these reach no network.
OFFLINE_PROVIDERS: tuple[str, ...] = ("hashing", "replay")


def _require_dimensions(settings: Settings, provider: str) -> int:
    if settings.embedding_dimensions is None:
        raise ProviderUnavailable(
            f"embedding provider {provider!r} needs an expected vector width; "
            "set CCE_EMBEDDING_DIMENSIONS. It is validated per vector, because nothing in "
            "a provider's response proves the numbers came from the model that was asked "
            "for.",
            provider=provider,
        )
    return settings.embedding_dimensions


def _build_hashing(settings: Settings) -> EmbeddingPort:
    """The offline default. Deterministic, credential-free, and not semantic.

    Its width comes from ``CCE_EMBEDDING_DIMENSIONS`` when set and otherwise from the
    embedder's own default - the one case where a default width is right, because this
    embedder *is* its own model and there is no remote contract to disagree with.
    """
    from app.infrastructure.embeddings.hashing import DEFAULT_DIMENSIONS, HashingEmbedder

    return HashingEmbedder(dimensions=settings.embedding_dimensions or DEFAULT_DIMENSIONS)


def _build_replay(settings: Settings) -> EmbeddingPort:
    """Committed provider vectors. The seam that makes a semantic measurement possible."""
    from app.infrastructure.embeddings.replay import ReplayEmbedder, load_recorded

    path = settings.embedding_fixture_path
    if not path.exists():
        raise ProviderUnavailable(
            f"the replay embedder needs a recorded fixture at {path}, and none is "
            "committed. Recording one takes a single run against a real provider; until "
            "then no semantic claim about vector retrieval is supported.",
            provider="replay",
        )
    return ReplayEmbedder(load_recorded(path))


def _build_http(settings: Settings) -> EmbeddingPort:
    """An OpenAI-compatible remote embedder. **Not enabled by default, and never called.**

    Every value it needs comes from configuration or the environment: the model identifier
    from ``CCE_MODEL_EMBEDDING``, the credential from ``CCE_EMBEDDING_API_KEY``, the
    endpoint from ``CCE_EMBEDDING_BASE_URL``, the expected width from
    ``CCE_EMBEDDING_DIMENSIONS``. None has a default, and the adapter raises rather than
    guessing at any of them.
    """
    from app.infrastructure.embeddings.http import HttpEmbeddingProvider

    return HttpEmbeddingProvider(
        model=settings.model_embedding or "",
        dimensions=_require_dimensions(settings, "http"),
        base_url=settings.embedding_base_url or "",
        api_key=(
            settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None
        ),
        expects_unit_vectors=settings.embedding_expects_unit_vectors,
        max_batch_size=settings.embedding_max_batch_size,
        timeout_s=settings.embedding_timeout_s,
    )


def _build_gemini(settings: Settings) -> EmbeddingPort:
    """``gemini-embedding-001`` at the configured width. **Not enabled by default.**

    Same rule as the OpenAI-compatible adapter: model identifier, credential, endpoint and
    expected width all come from configuration or the environment with no defaults, and the
    adapter raises rather than guessing at any of them. It has never made a live call.
    """
    from app.infrastructure.embeddings.gemini import (
        GEMINI_DEFAULT_DIMENSIONS,
        GeminiEmbeddingProvider,
    )

    return GeminiEmbeddingProvider(
        model=settings.model_embedding or "",
        base_url=settings.embedding_base_url or "",
        api_key=(
            settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None
        ),
        dimensions=settings.embedding_dimensions or GEMINI_DEFAULT_DIMENSIONS,
        task_type=settings.embedding_task_type,
        expects_unit_vectors=settings.embedding_expects_unit_vectors,
        timeout_s=settings.embedding_timeout_s,
    )


#: Every embedding adapter CCE knows how to build. Membership does **not** enable one.
EMBEDDING_FACTORIES: dict[str, EmbeddingFactory] = {
    "hashing": _build_hashing,
    "replay": _build_replay,
    "http": _build_http,
    "gemini": _build_gemini,
}


class EmbeddingRegistry:
    """Lazily instantiates and caches the enabled embedders."""

    def __init__(
        self,
        settings: Settings,
        *,
        overrides: dict[str, EmbeddingPort] | None = None,
    ) -> None:
        self._settings = settings
        self._enabled = tuple(settings.embedding_enabled_providers)
        # Pre-built instances win over factories: how a test injects a stub, and how a
        # recorded fixture could be pinned for one run without touching configuration.
        self._instances: dict[str, EmbeddingPort] = dict(overrides or {})

    @property
    def enabled(self) -> tuple[str, ...]:
        return self._enabled

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled or name in self._instances

    def get(self, name: str) -> EmbeddingPort:
        if name in self._instances:
            return self._instances[name]

        if name not in self._enabled:
            raise ProviderUnavailable(
                f"embedding provider {name!r} is not enabled; enabled providers are "
                f"{list(self._enabled)}. Add it to CCE_EMBEDDING_ENABLED_PROVIDERS to "
                "allow it - a remote embedder sends corpus text to a third party, so that "
                "is an operator's decision.",
                provider=name,
            )

        factory = EMBEDDING_FACTORIES.get(name)
        if factory is None:
            raise ProviderUnavailable(
                f"embedding provider {name!r} has no adapter; "
                f"known adapters are {sorted(EMBEDDING_FACTORIES)}",
                provider=name,
            )

        instance = factory(self._settings)
        self._instances[name] = instance
        return instance

    def default(self) -> EmbeddingPort:
        """The configured embedder, ``CCE_EMBEDDING_PROVIDER``."""
        return self.get(self._settings.embedding_provider)

    async def aclose(self) -> None:
        for embedder in self._instances.values():
            closer = getattr(embedder, "aclose", None)
            if closer is not None:
                await closer()
