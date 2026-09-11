"""Adapter for providers exposing an OpenAI-compatible ``POST {base_url}/embeddings``.

The transport seam for a real embedding provider. It is the embedding counterpart of
``app/infrastructure/llm/providers/_openai_compat.py`` and follows it deliberately: one
``httpx`` client, no vendor SDK, vendor failures translated into the ``ProviderError``
hierarchy at the boundary so nothing above catches a vendor exception type.

**No live call has ever been made through this class, and it is not enabled.** It is absent
from ``CCE_EMBEDDING_ENABLED_PROVIDERS`` by default, so constructing it requires an operator
to say so. Every test exercises it through ``httpx.MockTransport``, which means the wire
*mapping* is verified and the wire *contract* is not: a real provider may name its fields
differently, page its responses, or return a width its documentation does not mention. Until
one run has happened against a real endpoint, treat this as untested integration code -
``docs/STATUS.md`` carries that as an open issue rather than implying otherwise.

Three configuration rules it enforces rather than documents:

**The model identifier comes from configuration and has no default.** A default model would
mean a recall figure attributable to whatever the default happened to be that month.

**The API key comes from the environment only.** It is read from ``Settings``, which reads
``CCE_EMBEDDING_API_KEY``; it is never a literal, never a constructor default, and never
inferred from another provider's credential.

**The expected width comes from configuration too, and is checked per vector.** A provider
is asked for a model and answers with numbers; nothing in the response proves the numbers
came from the model that was asked for. Declaring the width and validating it is the cheapest
available guard against a mistyped model identifier silently producing a different index.

Deliberately absent: retry and backoff (a gateway concern embeddings do not yet share),
caching (needs a key design and a measurement), and any per-provider special-casing. A
provider whose wire format genuinely differs gets its own adapter rather than a flag here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import httpx

from app.domain.interfaces.llm import (
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTransportError,
    ProviderUnavailable,
)
from app.infrastructure.embeddings.base import (
    EmbeddingAdapter,
    EmbeddingMetadata,
)

#: The wire shape this adapter speaks. Named so that a second embedding wire format is a
#: second adapter with its own identifier rather than a branch in this one.
HTTP_EMBEDDING_PROVIDER: Final = "openai-compatible"

DEFAULT_BATCH_SIZE: Final = 64
DEFAULT_TIMEOUT_S: Final = 60.0


class HttpEmbeddingProvider(EmbeddingAdapter):
    """``POST {base_url}/embeddings`` with a bearer token.

    ``client`` is injectable, which is how every test drives this class through
    ``httpx.MockTransport`` with no network. An injected client is **not** closed by
    :meth:`aclose`: whoever created it owns its lifetime.
    """

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        base_url: str,
        api_key: str | None = None,
        provider: str = HTTP_EMBEDDING_PROVIDER,
        expects_unit_vectors: bool = False,
        max_batch_size: int = DEFAULT_BATCH_SIZE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model:
            raise ProviderUnavailable(
                "no embedding model is configured; set CCE_MODEL_EMBEDDING. There is no "
                "default, because a vector index has to name the model that produced it.",
                provider=provider,
            )
        if not base_url:
            raise ProviderUnavailable(
                f"no base URL is configured for embedding provider {provider!r}; "
                "set CCE_EMBEDDING_BASE_URL",
                provider=provider,
            )
        if not api_key:
            # A configuration error, raised at construction rather than at call time, so a
            # missing credential surfaces when the adapter is selected and never mid-run.
            raise ProviderUnavailable(
                f"embedding provider {provider!r} is enabled but no API key is configured; "
                "set CCE_EMBEDDING_API_KEY",
                provider=provider,
            )

        super().__init__(
            metadata=EmbeddingMetadata(
                provider=provider,
                model=model,
                dimensions=dimensions,
                expects_unit_vectors=expects_unit_vectors,
            ),
            max_batch_size=max_batch_size,
        )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    # -- the wire --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _payload(self, texts: Sequence[str]) -> dict[str, Any]:
        return {"model": self._metadata.model, "input": list(texts)}

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = await post_json(
            self._client,
            f"{self._base_url}/embeddings",
            headers=self._headers(),
            payload=self._payload(texts),
            provider=self._metadata.provider,
            timeout_s=self._timeout_s,
        )
        raise_for_status(response, provider=self._metadata.provider)
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> list[list[float]]:
        """Read ``{"data": [{"index": n, "embedding": [...]}, ...]}``.

        **Sorted by the provider's own ``index``** rather than trusting document order. The
        field exists precisely because the wire format does not promise order, and the
        failure it guards against is the one that cannot be detected later: vectors
        attached to the wrong texts produce a plausible ranking, not an error. Where an
        entry carries no index, its position is used - which is the only available reading
        and is why a missing index is not itself an error.
        """
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self._metadata.provider} returned a non-JSON embedding body",
                provider=self._metadata.provider,
            ) from exc

        if not isinstance(body, dict):
            raise ProviderError(
                f"{self._metadata.provider} returned {type(body).__name__} rather than an object",
                provider=self._metadata.provider,
            )

        data = body.get("data")
        if not isinstance(data, list):
            raise ProviderError(
                f"{self._metadata.provider} returned no embedding data array",
                provider=self._metadata.provider,
            )

        indexed: list[tuple[int, list[float]]] = []
        for position, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ProviderError(
                    f"{self._metadata.provider} embedding entry {position} is not an object",
                    provider=self._metadata.provider,
                )
            vector = entry.get("embedding")
            if not isinstance(vector, list):
                raise ProviderError(
                    f"{self._metadata.provider} embedding entry {position} carries no vector",
                    provider=self._metadata.provider,
                )
            raw_index = entry.get("index")
            order = raw_index if isinstance(raw_index, int) else position
            indexed.append((order, [float(component) for component in vector]))

        indexed.sort(key=lambda row: row[0])
        return [vector for _order, vector in indexed]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _error_detail(response: httpx.Response) -> str:
    """The provider's own message, when it gave one, bounded so a log stays readable."""
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


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shared by every HTTP embedding adapter
# ---------------------------------------------------------------------------
#
# Extracted rather than duplicated when the Gemini adapter landed. The two speak different
# wire formats - different endpoint, different auth header, different body and response
# shape - but a 429 means the same thing to both, and so does a timeout. Error translation
# is the part an adapter must not get creatively wrong, so there is one of it.


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    provider: str,
    timeout_s: float,
) -> httpx.Response:
    """POST JSON, translating transport failures into ``ProviderTransportError``."""
    try:
        return await client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise ProviderTransportError(
            f"{provider} embedding request timed out after {timeout_s}s", provider=provider
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderTransportError(
            f"{provider} embedding transport failure: {exc}", provider=provider
        ) from exc


def raise_for_status(response: httpx.Response, *, provider: str) -> None:
    """Translate a status code into the error the application layer already handles.

    The same split the completion adapter makes, and for the same reason: a 429 or a 5xx is
    worth another attempt, a 401 is an operator's problem, and a 400 is ours.
    """
    if response.is_success:
        return

    detail = _error_detail(response)

    if response.status_code == 429:
        raise ProviderRateLimited(
            f"{provider} rate limited: {detail}",
            provider=provider,
            retry_after_s=_as_float(response.headers.get("retry-after")),
        )
    if response.status_code >= 500:
        raise ProviderTransportError(
            f"{provider} server error {response.status_code}: {detail}", provider=provider
        )
    if response.status_code in (401, 403):
        raise ProviderUnavailable(
            f"{provider} rejected credentials ({response.status_code}): {detail}",
            provider=provider,
        )
    raise ProviderBadRequest(
        f"{provider} rejected embedding request {response.status_code}: {detail}",
        provider=provider,
    )
