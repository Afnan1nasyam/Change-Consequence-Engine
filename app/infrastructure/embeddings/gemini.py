"""Gemini embeddings: ``gemini-embedding-001`` behind the existing adapter base.

Request/response mapping only. **No live call has been made through this class**, here or
anywhere: it was written and tested against ``httpx.MockTransport`` for a run planned on a
personal laptop. Until that run happens this is unverified integration code, and
``docs/STATUS.md`` says so rather than implying otherwise.

Why it is a separate adapter rather than a flag on ``HttpEmbeddingProvider``: the wire format
genuinely differs in four places, and a branch per difference is how one adapter becomes
unreadable.

=========  ==========================================  ===========================================
           OpenAI-compatible                           Gemini
=========  ==========================================  ===========================================
endpoint   ``POST /embeddings``                        ``POST /models/{m}:batchEmbedContents``
auth       ``Authorization: Bearer <key>``             ``x-goog-api-key: <key>``
request    ``{"model":.., "input": [text, ..]}``       ``{"requests": [{"content": {"parts":..}}]}``
response   ``{"data": [{"index":n, "embedding":..}]}``  ``{"embeddings": [{"values": [..]}]}``
=========  ==========================================  ===========================================

What they share - the error translation, the transport wrapping, and every obligation in
``EmbeddingAdapter`` (per-vector width validation, finiteness, batching, provenance) - is
shared as code, not copied.

**The response carries no index, so ordering is positional and unguarded.** This is the one
place Gemini is weaker than the OpenAI shape: ``batchEmbedContents`` returns
``embeddings[]`` and nothing that identifies which request each entry answers. The pairing
therefore rests entirely on the provider preserving request order, and no validation can
detect a permutation - a permuted response would produce a plausible wrong ranking rather
than an error. The count is checked, which catches a dropped or extra vector; order is
trusted, which is documented here because it cannot be enforced.

**Two things to verify on the first real run, not to assume:**

*Normalization.* ``gemini-embedding-001`` is a Matryoshka model: 3072 is its native width and
1536/768 are truncations. Google documents truncated output as **not** unit length, needing
L2 normalization by the caller. So :data:`GEMINI_EXPECTS_UNIT_VECTORS` is ``False`` and the
adapter does **not** normalize - cosine is scale-invariant, so ranking is identical either
way, and transforming a provider's numbers inside a mapping adapter would make the recorded
fixture something other than what the provider returned. If the first run shows unit-length
vectors at 768, the flag can be turned on and nothing else changes.

*Task type.* Gemini accepts ``taskType``, and for retrieval Google recommends the
**asymmetric** pair ``RETRIEVAL_DOCUMENT`` for corpus text and ``RETRIEVAL_QUERY`` for
queries. ``EmbeddingPort.embed`` takes an undifferentiated list of texts and cannot express
that distinction, and widening it is a domain-contract change. So this adapter sends one
symmetric ``taskType`` (default ``SEMANTIC_SIMILARITY``), which is the honest thing the
current interface can do - and it probably costs recall. Recorded as an open issue rather
than smuggled in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import httpx

from app.domain.interfaces.llm import ProviderError, ProviderUnavailable
from app.infrastructure.embeddings.base import (
    EmbeddingAdapter,
    EmbeddingMetadata,
)
from app.infrastructure.embeddings.http import post_json, raise_for_status

#: Provider identifier, recorded in the fixture and in ``scorer_version``.
GEMINI_EMBEDDING_PROVIDER: Final = "gemini"

#: The model this adapter was written for. Configuration supplies the identifier; this is
#: here to be asserted against, not to be defaulted to.
GEMINI_EMBEDDING_MODEL: Final = "gemini-embedding-001"

#: The width planned for the experiment. ``gemini-embedding-001`` is Matryoshka: 3072 native,
#: with 1536 and 768 as supported truncations. 768 keeps a committed fixture of 51 vectors
#: reviewable in a diff and is a width the model documents as supported - it is **not** a
#: tuned choice, and no measurement has compared it with 1536 or 3072.
GEMINI_DEFAULT_DIMENSIONS: Final = 768
GEMINI_SUPPORTED_DIMENSIONS: Final = (768, 1536, 3072)

#: False, deliberately: Google documents sub-3072 output as unnormalized. See the module
#: docstring - cosine is scale-invariant so ranking is unaffected, and the adapter does not
#: transform what the provider returned.
GEMINI_EXPECTS_UNIT_VECTORS: Final = False

#: Symmetric, because ``EmbeddingPort`` cannot express the asymmetric retrieval pair.
GEMINI_DEFAULT_TASK_TYPE: Final = "SEMANTIC_SIMILARITY"

#: ``batchEmbedContents`` caps a batch. 100 is the documented limit; the 51-text recording
#: fits in one request either way.
GEMINI_MAX_BATCH_SIZE: Final = 100

DEFAULT_TIMEOUT_S: Final = 60.0

#: Gemini names models ``models/<id>``. Configuration may carry either form.
_MODEL_PREFIX: Final = "models/"


def qualified_model(model: str) -> str:
    """``gemini-embedding-001`` -> ``models/gemini-embedding-001``, idempotently.

    Accepting both forms because an operator will reasonably write either into
    ``CCE_MODEL_EMBEDDING``, and a 404 from a missing prefix is a poor way to learn that.
    """
    return model if model.startswith(_MODEL_PREFIX) else f"{_MODEL_PREFIX}{model}"


def bare_model(model: str) -> str:
    """The identifier without the prefix, for the URL path and for provenance."""
    return model.removeprefix(_MODEL_PREFIX)


class GeminiEmbeddingProvider(EmbeddingAdapter):
    """``POST {base_url}/models/{model}:batchEmbedContents`` with ``x-goog-api-key``.

    ``client`` is injectable, which is how every test drives this class through
    ``httpx.MockTransport`` with no network. An injected client is not closed by
    :meth:`aclose`: whoever created it owns its lifetime.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        dimensions: int = GEMINI_DEFAULT_DIMENSIONS,
        task_type: str = GEMINI_DEFAULT_TASK_TYPE,
        expects_unit_vectors: bool = GEMINI_EXPECTS_UNIT_VECTORS,
        max_batch_size: int = GEMINI_MAX_BATCH_SIZE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model:
            raise ProviderUnavailable(
                "no embedding model is configured; set CCE_MODEL_EMBEDDING to "
                f"{GEMINI_EMBEDDING_MODEL!r}. There is no default, because a vector index "
                "has to name the model that produced it.",
                provider=GEMINI_EMBEDDING_PROVIDER,
            )
        if not base_url:
            raise ProviderUnavailable(
                "no base URL is configured for Gemini embeddings; set CCE_EMBEDDING_BASE_URL",
                provider=GEMINI_EMBEDDING_PROVIDER,
            )
        if not api_key:
            # A configuration error, raised when the adapter is selected rather than
            # mid-run. The key is read from the environment and never defaulted.
            raise ProviderUnavailable(
                "Gemini embeddings are enabled but no API key is configured; set "
                "CCE_EMBEDDING_API_KEY",
                provider=GEMINI_EMBEDDING_PROVIDER,
            )
        if dimensions not in GEMINI_SUPPORTED_DIMENSIONS:
            # Refused rather than sent: an unsupported width is either a 400 or, worse, a
            # silently different vector, and both are cheaper to catch here.
            raise ProviderUnavailable(
                f"{model} supports {list(GEMINI_SUPPORTED_DIMENSIONS)} output dimensions, "
                f"not {dimensions}; set CCE_EMBEDDING_DIMENSIONS to one of them",
                provider=GEMINI_EMBEDDING_PROVIDER,
            )

        super().__init__(
            metadata=EmbeddingMetadata(
                provider=GEMINI_EMBEDDING_PROVIDER,
                model=bare_model(model),
                dimensions=dimensions,
                expects_unit_vectors=expects_unit_vectors,
            ),
            max_batch_size=max_batch_size,
        )
        self._model = qualified_model(model)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._task_type = task_type
        self._timeout_s = timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @property
    def task_type(self) -> str:
        return self._task_type

    # -- the wire --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Gemini authenticates with a header, never a query parameter.

        The ``?key=`` form is equivalent to the API and much worse operationally: query
        strings land in access logs, proxy logs and error reports, and a credential in a URL
        is a credential in somebody's log aggregator.
        """
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

    def _payload(self, texts: Sequence[str]) -> dict[str, Any]:
        """One ``requests`` entry per text, each naming the model and the output width.

        ``model`` is repeated inside every entry as well as in the URL because
        ``batchEmbedContents`` requires it per request - a batch may in principle mix
        models, which this adapter does not do.
        """
        return {
            "requests": [
                {
                    "model": self._model,
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self._metadata.dimensions,
                    "taskType": self._task_type,
                }
                for text in texts
            ]
        }

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{self._base_url}/{self._model}:batchEmbedContents"
        response = await post_json(
            self._client,
            url,
            headers=self._headers(),
            payload=self._payload(texts),
            provider=self._metadata.provider,
            timeout_s=self._timeout_s,
        )
        raise_for_status(response, provider=self._metadata.provider)
        return self._parse(response, expected=len(texts))

    def _parse(self, response: httpx.Response, *, expected: int) -> list[list[float]]:
        """Read ``{"embeddings": [{"values": [...]}, ...]}``, in request order.

        ``expected`` is checked here as well as in :meth:`EmbeddingAdapter.embed`, because
        this response has **no index field**: order is the only thing tying a vector to its
        text, so a count mismatch is the only detectable pairing failure and it is worth
        catching at the parse site with the provider's own shape in view.
        """
        provider = self._metadata.provider
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"{provider} returned a non-JSON embedding body", provider=provider
            ) from exc

        if not isinstance(body, dict):
            raise ProviderError(
                f"{provider} returned {type(body).__name__} rather than an object",
                provider=provider,
            )

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise ProviderError(f"{provider} returned no embeddings array", provider=provider)
        if len(embeddings) != expected:
            raise ProviderError(
                f"{provider} returned {len(embeddings)} embeddings for {expected} texts; "
                "batchEmbedContents carries no index, so order is the only thing pairing a "
                "vector to its text and a count mismatch cannot be recovered from",
                provider=provider,
            )

        vectors: list[list[float]] = []
        for position, entry in enumerate(embeddings):
            if not isinstance(entry, dict):
                raise ProviderError(
                    f"{provider} embedding {position} is not an object", provider=provider
                )
            values = entry.get("values")
            if not isinstance(values, list):
                raise ProviderError(
                    f"{provider} embedding {position} carries no values array",
                    provider=provider,
                )
            vectors.append([float(component) for component in values])

        return vectors

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
