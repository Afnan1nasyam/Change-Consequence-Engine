"""The Gemini embedding adapter, against ``httpx.MockTransport``. No network, no key.

``gemini-embedding-001`` at 768 dimensions, prepared for a run planned on a personal laptop.
**No live call has been made** here or anywhere, so what these tests establish is the wire
*mapping* - endpoint, auth header, request body, response shape, error translation and
validation - and not the wire *contract*. The real endpoint may disagree with any of it, and
the first live run should be read as a debugging exercise (``docs/STATUS.md`` 41).

Three tests carry most of the weight, each guarding something that produces a plausible
wrong answer rather than an error:

* :func:`TestRequestShape.test_the_credential_goes_in_a_header_never_the_query_string` - a
  key in a URL is a key in somebody's log aggregator.
* :func:`TestResponseShape.test_a_count_mismatch_is_refused` - ``batchEmbedContents`` carries
  no index, so order is the only thing pairing a vector to its text.
* :func:`TestDimensions.test_a_response_at_the_wrong_width_is_refused` - the guard against a
  mistyped model or an ignored ``outputDimensionality``.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest

from app.domain.interfaces.llm import (
    EmbeddingPort,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTransportError,
    ProviderUnavailable,
)
from app.infrastructure.embeddings.base import EmbeddingAdapter, EmbeddingValidationError
from app.infrastructure.embeddings.gemini import (
    GEMINI_DEFAULT_DIMENSIONS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_EXPECTS_UNIT_VECTORS,
    GEMINI_MAX_BATCH_SIZE,
    GEMINI_SUPPORTED_DIMENSIONS,
    GeminiEmbeddingProvider,
    bare_model,
    qualified_model,
)
from app.infrastructure.embeddings.registry import EMBEDDING_FACTORIES, EmbeddingRegistry
from tests.conftest import make_settings

Handler = Callable[[httpx.Request], httpx.Response]

MODEL = GEMINI_EMBEDDING_MODEL
DIMS = GEMINI_DEFAULT_DIMENSIONS
KEY = "test-key-not-a-real-credential"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def vector(seed: float = 1.0, *, width: int = DIMS) -> list[float]:
    """An arbitrary, deliberately **unnormalized** vector of the right width.

    Unnormalized on purpose: Google documents sub-3072 Matryoshka output as not unit length,
    so this is the shape the adapter is expected to accept without complaint.
    """
    return [seed + index * 0.01 for index in range(width)]


def body(count: int, *, width: int = DIMS) -> dict[str, Any]:
    return {"embeddings": [{"values": vector(1.0 + i, width=width)} for i in range(count)]}


def echo(width: int = DIMS) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(200, json=body(len(payload["requests"]), width=width))

    return handler


def provider(handler: Handler, **over: Any) -> GeminiEmbeddingProvider:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "base_url": BASE_URL,
        "api_key": KEY,
        "client": httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    }
    kwargs.update(over)
    return GeminiEmbeddingProvider(**kwargs)


def capture() -> tuple[list[httpx.Request], Handler]:
    """Record the requests and answer at the width they asked for.

    Honouring ``outputDimensionality`` rather than a fixed width, because a faithful stand-in
    is the only kind worth asserting request shape against - and because the first version
    of this helper ignored it and was caught by the adapter's own validation.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        payload = json.loads(request.content)
        requests = payload["requests"]
        width = requests[0]["outputDimensionality"] if requests else DIMS
        return httpx.Response(200, json=body(len(requests), width=width))

    return seen, handler


# ---------------------------------------------------------------------------
# Identity and configuration
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_the_planned_experiment_is_768_dimensional(self) -> None:
        """Chosen because it is a documented Matryoshka width and keeps a 51-vector fixture
        reviewable. **Not** a tuned choice - nothing has compared it with 1536 or 3072."""
        assert GEMINI_DEFAULT_DIMENSIONS == 768
        assert provider(echo()).dimensions == 768

    def test_the_metadata_names_gemini_and_the_model(self) -> None:
        embedder = provider(echo())
        assert embedder.metadata.provider == "gemini"
        assert embedder.metadata.model == MODEL
        assert MODEL in embedder.name

    def test_the_model_prefix_is_accepted_in_either_form(self) -> None:
        """An operator will reasonably write either into ``CCE_MODEL_EMBEDDING``, and a 404
        from a missing ``models/`` prefix is a poor way to learn that."""
        assert qualified_model(MODEL) == f"models/{MODEL}"
        assert qualified_model(f"models/{MODEL}") == f"models/{MODEL}"
        assert bare_model(f"models/{MODEL}") == MODEL
        assert bare_model(MODEL) == MODEL

    def test_provenance_records_the_bare_model_identifier(self) -> None:
        """So a fixture says ``gemini-embedding-001`` rather than ``models/...``."""
        assert provider(echo(), model=f"models/{MODEL}").metadata.model == MODEL

    def test_it_does_not_expect_unit_vectors(self) -> None:
        """Google documents sub-3072 output as unnormalized. The adapter does not normalize
        either: cosine is scale-invariant so ranking is identical, and transforming a
        provider's numbers would make the fixture something other than what it returned.
        """
        assert GEMINI_EXPECTS_UNIT_VECTORS is False
        assert provider(echo()).metadata.expects_unit_vectors is False

    def test_it_sends_one_symmetric_task_type(self) -> None:
        """``EmbeddingPort`` takes an undifferentiated list of texts, so the asymmetric
        ``RETRIEVAL_DOCUMENT``/``RETRIEVAL_QUERY`` pair Google recommends for retrieval
        cannot be expressed. One symmetric value is the honest alternative, and it probably
        costs recall (``docs/STATUS.md`` 44).
        """
        assert provider(echo()).task_type == "SEMANTIC_SIMILARITY"


class TestConfigurationIsRequired:
    def test_no_model_is_a_configuration_error(self) -> None:
        with pytest.raises(ProviderUnavailable, match="CCE_MODEL_EMBEDDING"):
            provider(echo(), model="")

    def test_no_api_key_is_a_configuration_error(self) -> None:
        with pytest.raises(ProviderUnavailable, match="CCE_EMBEDDING_API_KEY"):
            provider(echo(), api_key=None)

    def test_no_base_url_is_a_configuration_error(self) -> None:
        with pytest.raises(ProviderUnavailable, match="CCE_EMBEDDING_BASE_URL"):
            provider(echo(), base_url="")

    @pytest.mark.parametrize("width", sorted(GEMINI_SUPPORTED_DIMENSIONS))
    def test_every_documented_width_is_accepted(self, width: int) -> None:
        assert provider(echo(width), dimensions=width).dimensions == width

    @pytest.mark.parametrize("width", [1, 512, 1024, 3073])
    def test_an_undocumented_width_is_refused_before_any_request(self, width: int) -> None:
        """Refused rather than sent: an unsupported width is either a 400 or, worse, a
        silently different vector."""
        with pytest.raises(ProviderUnavailable, match="output dimensions"):
            provider(echo(), dimensions=width)

    def test_the_batch_limit_matches_the_documented_one(self) -> None:
        assert provider(echo()).max_batch_size == GEMINI_MAX_BATCH_SIZE == 100


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_it_posts_to_batch_embed_contents(self) -> None:
        seen, handler = capture()
        asyncio.run(provider(handler).embed_one("a span"))
        assert str(seen[0].url) == f"{BASE_URL}/models/{MODEL}:batchEmbedContents"
        assert seen[0].method == "POST"

    def test_a_trailing_slash_in_the_base_url_does_not_double(self) -> None:
        seen, handler = capture()
        asyncio.run(provider(handler, base_url=f"{BASE_URL}/").embed_one("a"))
        assert str(seen[0].url) == f"{BASE_URL}/models/{MODEL}:batchEmbedContents"

    def test_the_credential_goes_in_a_header_never_the_query_string(self) -> None:
        """The ``?key=`` form is equivalent to the API and much worse operationally: query
        strings land in access logs, proxy logs and error reports."""
        seen, handler = capture()
        asyncio.run(provider(handler).embed_one("a"))

        assert seen[0].headers["x-goog-api-key"] == KEY
        assert "authorization" not in seen[0].headers
        assert KEY not in str(seen[0].url)
        assert seen[0].url.query == b""

    def test_the_body_is_one_request_entry_per_text(self) -> None:
        seen, handler = capture()
        asyncio.run(provider(handler).embed(["first span", "second span"]))
        payload = json.loads(seen[0].content)

        assert payload == {
            "requests": [
                {
                    "model": f"models/{MODEL}",
                    "content": {"parts": [{"text": "first span"}]},
                    "outputDimensionality": DIMS,
                    "taskType": "SEMANTIC_SIMILARITY",
                },
                {
                    "model": f"models/{MODEL}",
                    "content": {"parts": [{"text": "second span"}]},
                    "outputDimensionality": DIMS,
                    "taskType": "SEMANTIC_SIMILARITY",
                },
            ]
        }

    def test_the_output_width_is_requested_explicitly(self) -> None:
        """Without ``outputDimensionality`` the model returns its native 3072 and the
        configured width would be wrong for every vector."""
        seen, handler = capture()
        asyncio.run(provider(handler, dimensions=1536).embed_one("a"))
        payload = json.loads(seen[0].content)
        assert payload["requests"][0]["outputDimensionality"] == 1536

    def test_a_configured_task_type_reaches_the_request(self) -> None:
        seen, handler = capture()
        asyncio.run(provider(handler, task_type="RETRIEVAL_DOCUMENT").embed_one("a"))
        payload = json.loads(seen[0].content)
        assert payload["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"

    def test_the_single_text_path_sends_one_entry(self) -> None:
        seen, handler = capture()
        result = asyncio.run(provider(handler).embed_one("a"))
        assert len(json.loads(seen[0].content)["requests"]) == 1
        assert len(result) == DIMS

    def test_a_long_request_is_split_at_the_batch_limit(self) -> None:
        seen, handler = capture()
        asyncio.run(provider(handler, max_batch_size=4).embed([f"span {i}" for i in range(10)]))

        sizes = [len(json.loads(request.content)["requests"]) for request in seen]
        assert sizes == [4, 4, 2]

    def test_an_empty_request_reaches_no_provider(self) -> None:
        seen, handler = capture()
        assert asyncio.run(provider(handler).embed([])) == []
        assert seen == []


# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------


class TestResponseShape:
    def test_values_are_read_in_request_order(self) -> None:
        """``batchEmbedContents`` carries no index, so position is the only pairing there
        is. This asserts the adapter preserves it; nothing can verify the provider did.
        """

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        {"values": vector(1.0)},
                        {"values": vector(2.0)},
                    ]
                },
            )

        vectors = asyncio.run(provider(handler).embed(["a", "b"]))
        assert vectors == [vector(1.0), vector(2.0)]

    def test_a_count_mismatch_is_refused(self) -> None:
        """The only detectable pairing failure, and the message says why."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body(1))

        with pytest.raises(ProviderError, match="carries no index"):
            asyncio.run(provider(handler).embed(["a", "b"]))

    def test_an_unnormalized_vector_is_accepted(self) -> None:
        """The expected shape at 768. If a real run returns unit vectors instead, the flag
        can be turned on and nothing else changes."""
        vectors = asyncio.run(provider(echo()).embed_one("a"))
        norm = math.sqrt(sum(component * component for component in vectors))
        assert norm != pytest.approx(1.0)

    def test_a_non_json_body_is_a_provider_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>")

        with pytest.raises(ProviderError, match="non-JSON"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_json_array_body_is_a_provider_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[1, 2, 3])

        with pytest.raises(ProviderError, match="rather than an object"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_missing_embeddings_array_is_a_provider_error(self) -> None:
        """Including the case where a caller pointed the adapter at the OpenAI-shaped
        endpoint by mistake: ``{"data": ...}`` has no ``embeddings``."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

        with pytest.raises(ProviderError, match="no embeddings array"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_an_entry_that_is_not_an_object_is_a_provider_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": ["not an object"]})

        with pytest.raises(ProviderError, match="is not an object"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_an_entry_without_values_is_a_provider_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [{"statistics": {}}]})

        with pytest.raises(ProviderError, match="no values array"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_integer_components_are_read_as_floats(self) -> None:
        """JSON does not distinguish, and a mixed list would otherwise reach cosine as
        ints."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [{"values": [0, 1] * (DIMS // 2)}]})

        result = asyncio.run(provider(handler).embed_one("a"))
        assert all(isinstance(component, float) for component in result)


class TestDimensions:
    def test_a_response_at_the_wrong_width_is_refused(self) -> None:
        """The guard against a mistyped model, or a provider ignoring
        ``outputDimensionality`` and returning its native 3072."""
        with pytest.raises(EmbeddingValidationError, match="3072 components"):
            asyncio.run(provider(echo(width=3072)).embed_one("a"))

    def test_the_error_names_the_configured_model_and_width(self) -> None:
        with pytest.raises(EmbeddingValidationError) as raised:
            asyncio.run(provider(echo(width=1536)).embed_one("a"))

        message = str(raised.value)
        assert MODEL in message
        assert "768-dimensional" in message

    def test_a_non_finite_component_is_refused(self) -> None:
        """A NaN compares false against everything and would poison every cosine it
        enters. JSON has no NaN literal, so it arrives as the bare token."""

        def handler(_: httpx.Request) -> httpx.Response:
            values = [1.0] * (DIMS - 1)
            return httpx.Response(
                200,
                content=json.dumps({"embeddings": [{"values": [*values, float("nan")]}]}).encode(),
                headers={"Content-Type": "application/json"},
            )

        with pytest.raises(EmbeddingValidationError, match="non-finite"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_every_returned_vector_is_validated_not_only_the_first(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        {"values": vector(1.0)},
                        {"values": vector(2.0, width=DIMS - 1)},
                    ]
                },
            )

        with pytest.raises(EmbeddingValidationError, match="vector 1"):
            asyncio.run(provider(handler).embed(["a", "b"]))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_a_google_shaped_error_message_is_surfaced(self) -> None:
        """Gemini nests its detail under ``error.message``, the same place the shared
        translator already looks."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": 400,
                        "message": "Invalid value at 'requests[0].output_dimensionality'",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )

        with pytest.raises(ProviderBadRequest, match="output_dimensionality"):
            asyncio.run(provider(handler).embed_one("a"))

    @pytest.mark.parametrize("status", [500, 503])
    def test_a_server_error_is_a_transport_error(self, status: int) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "backend unavailable"}})

        with pytest.raises(ProviderTransportError, match="backend unavailable"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_rate_limit_carries_retry_after_and_is_retryable(self) -> None:
        """The quota error a free-tier key will produce first."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"retry-after": "31"},
                json={"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
            )

        with pytest.raises(ProviderRateLimited) as raised:
            asyncio.run(provider(handler).embed_one("a"))

        assert raised.value.retry_after_s == 31.0
        assert raised.value.retryable is True

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_rejected_key_is_unavailable_and_not_retryable(self, status: int) -> None:
        """A configuration error - retrying cannot fix a wrong key."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "API key not valid"}})

        with pytest.raises(ProviderUnavailable) as raised:
            asyncio.run(provider(handler).embed_one("a"))
        assert raised.value.retryable is False

    def test_a_timeout_is_a_transport_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("too slow")

        with pytest.raises(ProviderTransportError, match="timed out"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_connection_failure_is_a_transport_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        with pytest.raises(ProviderTransportError, match="transport failure"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_no_vendor_exception_ever_escapes(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("reset")

        with pytest.raises(ProviderError):
            asyncio.run(provider(handler).embed_one("a"))

    def test_the_credential_is_never_in_an_error_message(self) -> None:
        """An error message is one of the places a secret leaks into a log."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": {"message": "API key not valid"}})

        with pytest.raises(ProviderUnavailable) as raised:
            asyncio.run(provider(handler).embed_one("a"))
        assert KEY not in str(raised.value)


# ---------------------------------------------------------------------------
# Selection and lifecycle
# ---------------------------------------------------------------------------


class TestSelection:
    def test_the_adapter_is_known_but_not_enabled(self) -> None:
        """A remote embedder sends corpus text to a third party, so enabling it is an
        operator's decision and not a consequence of a key existing."""
        assert "gemini" in EMBEDDING_FACTORIES

        settings = make_settings()
        assert "gemini" not in settings.embedding_enabled_providers
        with pytest.raises(ProviderUnavailable, match="not enabled"):
            EmbeddingRegistry(settings).get("gemini")

    def test_enabling_it_without_configuration_still_fails(self) -> None:
        settings = make_settings(embedding_enabled_providers=("gemini",))
        with pytest.raises(ProviderUnavailable):
            EmbeddingRegistry(settings).get("gemini")

    def test_the_registry_builds_it_from_configuration_alone(self) -> None:
        """The shape of the planned run, with a placeholder credential."""
        settings = make_settings(
            embedding_provider="gemini",
            embedding_enabled_providers=("hashing", "replay", "gemini"),
            model_embedding=MODEL,
            embedding_api_key=KEY,
            embedding_base_url=BASE_URL,
            embedding_dimensions=768,
        )
        embedder = EmbeddingRegistry(settings).default()

        assert isinstance(embedder, GeminiEmbeddingProvider)
        assert embedder.dimensions == 768
        assert embedder.metadata.model == MODEL

    def test_it_is_a_live_provider_as_far_as_the_recorder_is_concerned(self) -> None:
        """So ``record_embeddings`` accepts it, and refuses to record through an offline
        embedder that would produce a fixture claiming a provider ran."""
        from scripts.record_embeddings import LIVE_PROVIDERS

        assert "gemini" in LIVE_PROVIDERS

    def test_the_default_width_is_used_when_configuration_omits_one(self) -> None:
        settings = make_settings(
            embedding_provider="gemini",
            embedding_enabled_providers=("gemini",),
            model_embedding=MODEL,
            embedding_api_key=KEY,
            embedding_base_url=BASE_URL,
        )
        assert EmbeddingRegistry(settings).default().dimensions == 768


class TestConformance:
    def test_it_satisfies_the_embedding_port_and_the_adapter_base(self) -> None:
        embedder = provider(echo())
        assert isinstance(embedder, EmbeddingPort)
        assert isinstance(embedder, EmbeddingAdapter)

    def test_an_injected_client_is_not_closed_by_the_adapter(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(echo()))
        embedder = provider(echo(), client=client)

        asyncio.run(embedder.aclose())
        assert not client.is_closed
        asyncio.run(client.aclose())


class TestNothingElseChanged:
    def test_the_openai_shaped_adapter_still_speaks_its_own_format(self) -> None:
        """The shared error translation was extracted, not merged: the two adapters keep
        different endpoints, auth and bodies."""
        from app.infrastructure.embeddings.http import HttpEmbeddingProvider

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            texts: Sequence[str] = json.loads(request.content)["input"]
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": [0.1, 0.2]} for i, _ in enumerate(texts)]},
            )

        embedder = HttpEmbeddingProvider(
            model="other-model",
            dimensions=2,
            base_url="https://other.example/v1",
            api_key=KEY,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        asyncio.run(embedder.embed_one("a"))

        assert str(seen[0].url).endswith("/embeddings")
        assert seen[0].headers["authorization"] == f"Bearer {KEY}"

    def test_the_offline_default_is_untouched(self) -> None:
        """The vector benchmark is pinned to ``hashing-1`` at 256 dimensions."""
        from app.infrastructure.embeddings.hashing import HashingEmbedder

        embedder = HashingEmbedder()
        assert embedder.name == "hashing-1"
        assert embedder.dimensions == 256
        assert make_settings().embedding_provider == "hashing"
