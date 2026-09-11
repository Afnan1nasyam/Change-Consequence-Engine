"""The embedding adapter foundation: validation, batching, provenance, selection.

The HTTP adapter is exercised through ``httpx.MockTransport``, so real requests are built
and real responses are parsed with **no network and no credentials** - the same technique
``tests/unit/test_providers.py`` uses for completions. What that verifies is the wire
*mapping*; it cannot verify the wire *contract*, and no live call has ever been made through
this adapter.

The tests worth reading first, because each guards a failure that produces a plausible wrong
answer rather than an error:

* :func:`TestPairing.test_vectors_are_ordered_by_the_providers_own_index` - a permuted
  response attaches every vector to the wrong span.
* :func:`TestDimensionValidation.test_a_vector_of_the_wrong_width_is_refused` - a mistyped
  model identifier otherwise builds a different index in silence.
* :func:`TestEnablement.test_the_remote_adapter_is_not_enabled_by_default` - a remote
  embedder sends corpus text to a third party, including spans from ACL-restricted
  documents.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence

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
from app.infrastructure.embeddings.base import (
    EMBEDDING_ADAPTER_VERSION,
    MAX_BATCH_SIZE,
    UNIT_NORM_TOLERANCE,
    EmbeddingAdapter,
    EmbeddingMetadata,
    EmbeddingValidationError,
    batches,
    validate_vectors,
)
from app.infrastructure.embeddings.hashing import HashingEmbedder
from app.infrastructure.embeddings.http import HttpEmbeddingProvider
from app.infrastructure.embeddings.registry import (
    EMBEDDING_FACTORIES,
    OFFLINE_PROVIDERS,
    EmbeddingRegistry,
)
from tests.conftest import make_settings

Handler = Callable[[httpx.Request], httpx.Response]

MODEL = "acme-embed-small"
DIMENSIONS = 4
KEY = "test-key-not-a-real-credential"
BASE_URL = "https://embeddings.example/v1"


def unit(*components: float) -> list[float]:
    norm = math.sqrt(sum(c * c for c in components))
    return [c / norm for c in components]


def body(vectors: Sequence[Sequence[float]], *, with_index: bool = True) -> dict[str, object]:
    return {
        "object": "list",
        "model": MODEL,
        "data": [
            ({"index": i, "embedding": list(v)} if with_index else {"embedding": list(v)})
            for i, v in enumerate(vectors)
        ],
    }


def provider(handler: Handler, **over: object) -> HttpEmbeddingProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs: dict[str, object] = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "base_url": BASE_URL,
        "api_key": KEY,
        "client": client,
    }
    kwargs.update(over)
    return HttpEmbeddingProvider(**kwargs)  # type: ignore[arg-type]


def echo(vectors: Sequence[Sequence[float]]) -> Handler:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body(vectors))

    return handler


def metadata(**over: object) -> EmbeddingMetadata:
    kwargs: dict[str, object] = {
        "provider": "acme",
        "model": MODEL,
        "dimensions": DIMENSIONS,
    }
    kwargs.update(over)
    return EmbeddingMetadata(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_the_name_carries_provider_model_and_adapter_version(self) -> None:
        """Every component that would change the vectors is in the identity the retriever
        records, so a re-pointed model cannot look like the same index."""
        name = metadata().name
        assert EMBEDDING_ADAPTER_VERSION in name
        assert "acme" in name
        assert MODEL in name

    def test_a_different_model_is_a_different_name(self) -> None:
        assert metadata().name != metadata(model="acme-embed-large").name

    def test_a_missing_model_identifier_is_refused(self) -> None:
        """It comes from configuration and has no default: a figure measured under an
        unnamed model cannot be reproduced."""
        with pytest.raises(ValueError, match="model identifier"):
            metadata(model="")

    def test_a_missing_provider_is_refused(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            metadata(provider="")

    def test_a_non_positive_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            metadata(dimensions=0)

    def test_the_adapter_exposes_its_metadata_without_widening_the_port(self) -> None:
        """Provenance is on the adapter, not on ``EmbeddingPort``: retrieval needs a name
        and a width, and putting vendor-shaped detail in the domain contract would serve no
        caller."""
        embedder = provider(echo([unit(1, 0, 0, 0)]))
        assert embedder.metadata.model == MODEL
        assert not hasattr(EmbeddingPort, "metadata")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    def test_a_correctly_shaped_batch_passes(self) -> None:
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        assert validate_vectors(vectors, expected_count=2, metadata=metadata()) == vectors

    def test_a_vector_of_the_wrong_width_is_refused(self) -> None:
        """The guard against a mistyped model identifier building a different index."""
        with pytest.raises(EmbeddingValidationError, match="3 components"):
            validate_vectors([[1.0, 0.0, 0.0]], expected_count=1, metadata=metadata())

    def test_the_error_names_the_configured_model_and_width(self) -> None:
        with pytest.raises(EmbeddingValidationError) as raised:
            validate_vectors([[1.0]], expected_count=1, metadata=metadata())
        message = str(raised.value)
        assert MODEL in message
        assert "4-dimensional" in message

    def test_the_wrong_number_of_vectors_is_refused_first(self) -> None:
        """Count before width, because a wrong count means the *pairing* is broken - and a
        broken pairing produces a plausible wrong ranking rather than an error."""
        with pytest.raises(EmbeddingValidationError, match="pairing"):
            validate_vectors([[1.0, 0.0, 0.0, 0.0]], expected_count=2, metadata=metadata())

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_component_is_refused(self, bad: float) -> None:
        """A NaN compares false against everything and would silently poison every cosine
        it takes part in."""
        with pytest.raises(EmbeddingValidationError, match="non-finite"):
            validate_vectors([[bad, 0.0, 0.0, 0.0]], expected_count=1, metadata=metadata())

    def test_validation_is_an_error_the_application_layer_already_catches(self) -> None:
        """A ``ProviderError``, not a bare ``ValueError``: the request succeeded and the
        answer is unusable, which is a provider failure."""
        assert issubclass(EmbeddingValidationError, ProviderError)
        assert EmbeddingValidationError("x").retryable is False


class TestNormalizationValidation:
    def test_an_unnormalized_vector_passes_when_nothing_was_promised(self) -> None:
        """``VectorRetriever`` computes true cosine, so it survives either convention. The
        check exists for the adapters that *declare* unit vectors."""
        vectors = [[3.0, 4.0, 0.0, 0.0]]
        assert validate_vectors(vectors, expected_count=1, metadata=metadata()) == vectors

    def test_an_unnormalized_vector_is_refused_when_unit_vectors_are_expected(self) -> None:
        with pytest.raises(EmbeddingValidationError, match="expects unit vectors"):
            validate_vectors(
                [[3.0, 4.0, 0.0, 0.0]],
                expected_count=1,
                metadata=metadata(expects_unit_vectors=True),
            )

    def test_a_unit_vector_passes(self) -> None:
        vectors = [unit(3, 4, 0, 0)]
        assert (
            validate_vectors(
                vectors, expected_count=1, metadata=metadata(expects_unit_vectors=True)
            )
            == vectors
        )

    def test_float_round_tripping_is_tolerated(self) -> None:
        """Wide enough for float32 through JSON, narrow enough to catch a provider that
        stopped normalizing."""
        nudged = [1.0 - UNIT_NORM_TOLERANCE / 2, 0.0, 0.0, 0.0]
        assert validate_vectors(
            [nudged], expected_count=1, metadata=metadata(expects_unit_vectors=True)
        )

    def test_a_zero_vector_is_refused_when_unit_vectors_are_expected(self) -> None:
        """A zero vector has no direction. Where a provider promised unit length, one is
        evidence the provider failed rather than that the text carried no signal."""
        with pytest.raises(EmbeddingValidationError, match=r"norm 0\.000000"):
            validate_vectors(
                [[0.0, 0.0, 0.0, 0.0]],
                expected_count=1,
                metadata=metadata(expects_unit_vectors=True),
            )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_texts_are_split_into_consecutive_chunks(self) -> None:
        assert batches(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]

    def test_a_short_request_is_one_batch(self) -> None:
        assert batches(["a"], 64) == [["a"]]

    def test_no_texts_is_no_batches(self) -> None:
        assert batches([], 8) == []

    @pytest.mark.parametrize("size", [0, -1, MAX_BATCH_SIZE + 1])
    def test_an_out_of_range_batch_size_is_refused(self, size: int) -> None:
        """An unbounded batch is a request nobody sized and a cost nobody predicted."""
        with pytest.raises(ValueError, match="batch size"):
            batches(["a"], size)

    def test_a_request_larger_than_the_batch_size_makes_several_calls(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            payload = json.loads(request.content)
            count = len(payload["input"])
            calls.append(count)
            return httpx.Response(200, json=body([unit(1, 0, 0, 0)] * count))

        embedder = provider(handler, max_batch_size=2)
        vectors = asyncio.run(embedder.embed(["a", "b", "c", "d", "e"]))

        assert calls == [2, 2, 1]
        assert len(vectors) == 5

    def test_batches_are_concatenated_in_input_order(self) -> None:
        """The vector index pairs spans to vectors positionally, so reassembly order is
        the whole correctness of batching."""
        wanted = [unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0), unit(0, 0, 0, 1)]
        served = iter([wanted[0:2], wanted[2:4]])

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body(next(served)))

        embedder = provider(handler, max_batch_size=2)
        assert asyncio.run(embedder.embed(["a", "b", "c", "d"])) == wanted

    def test_an_empty_request_makes_no_call_at_all(self) -> None:
        """A provider billed per request should not be paid for a question nobody asked -
        and some reject an empty batch with a 400 that would surface as a failure."""
        called = False

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json=body([]))

        assert asyncio.run(provider(handler).embed([])) == []
        assert not called

    def test_an_out_of_range_adapter_batch_size_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            provider(echo([]), max_batch_size=0)


class TestPairing:
    def test_vectors_are_ordered_by_the_providers_own_index(self) -> None:
        """The wire format does not promise order, which is why it carries an index.

        A permuted response would otherwise attach every vector to the wrong span and
        produce a plausible ranking rather than an error.
        """
        first, second = unit(1, 0, 0, 0), unit(0, 1, 0, 0)

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": second},
                        {"index": 0, "embedding": first},
                    ]
                },
            )

        assert asyncio.run(provider(handler).embed(["a", "b"])) == [first, second]

    def test_position_is_used_when_a_provider_sends_no_index(self) -> None:
        """The only available reading, and why a missing index is not itself an error."""
        wanted = [unit(1, 0, 0, 0), unit(0, 1, 0, 0)]

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body(wanted, with_index=False))

        assert asyncio.run(provider(handler).embed(["a", "b"])) == wanted


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------


class TestRequest:
    def test_it_posts_to_the_embeddings_endpoint(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=body([unit(1, 0, 0, 0)]))

        asyncio.run(provider(handler).embed_one("a"))
        assert str(seen[0].url) == f"{BASE_URL}/embeddings"
        assert seen[0].method == "POST"

    def test_a_trailing_slash_in_the_base_url_does_not_double(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=body([unit(1, 0, 0, 0)]))

        asyncio.run(provider(handler, base_url=f"{BASE_URL}/").embed_one("a"))
        assert str(seen[0].url) == f"{BASE_URL}/embeddings"

    def test_the_configured_model_is_sent_and_nothing_is_defaulted(self) -> None:
        import json

        seen: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=body([unit(1, 0, 0, 0)]))

        asyncio.run(provider(handler).embed_one("a span"))
        assert seen[0] == {"model": MODEL, "input": ["a span"]}

    def test_the_credential_is_sent_as_a_bearer_token(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=body([unit(1, 0, 0, 0)]))

        asyncio.run(provider(handler).embed_one("a"))
        assert seen[0].headers["authorization"] == f"Bearer {KEY}"

    def test_the_single_text_path_returns_one_vector(self) -> None:
        wanted = unit(1, 2, 3, 4)
        assert asyncio.run(provider(echo([wanted])).embed_one("a")) == wanted


class TestConfigurationIsRequired:
    def test_no_model_is_a_configuration_error_at_construction(self) -> None:
        """Raised when the adapter is selected, never mid-run."""
        with pytest.raises(ProviderUnavailable, match="CCE_MODEL_EMBEDDING"):
            provider(echo([]), model="")

    def test_no_api_key_is_a_configuration_error_at_construction(self) -> None:
        with pytest.raises(ProviderUnavailable, match="CCE_EMBEDDING_API_KEY"):
            provider(echo([]), api_key=None)

    def test_no_base_url_is_a_configuration_error_at_construction(self) -> None:
        with pytest.raises(ProviderUnavailable, match="CCE_EMBEDDING_BASE_URL"):
            provider(echo([]), base_url="")


class TestErrorMapping:
    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_a_server_error_is_a_transport_error(self, status: int) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "upstream"}})

        with pytest.raises(ProviderTransportError, match="upstream"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_rate_limit_carries_retry_after(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "2.5"}, json={"error": "slow"})

        with pytest.raises(ProviderRateLimited) as raised:
            asyncio.run(provider(handler).embed_one("a"))
        assert raised.value.retry_after_s == 2.5
        assert raised.value.retryable is True

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_credential_rejection_is_unavailable_not_retryable(self, status: int) -> None:
        """A configuration error, not a transient failure - retrying cannot fix it."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "bad key"}})

        with pytest.raises(ProviderUnavailable) as raised:
            asyncio.run(provider(handler).embed_one("a"))
        assert raised.value.retryable is False

    def test_a_client_error_is_a_bad_request(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "input too long"}})

        with pytest.raises(ProviderBadRequest, match="input too long"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_timeout_is_a_transport_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("too slow")

        with pytest.raises(ProviderTransportError, match="timed out"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_transport_failure_is_a_transport_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route")

        with pytest.raises(ProviderTransportError, match="transport failure"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_vendor_exception_never_escapes(self) -> None:
        """The application layer catches ``ProviderError`` and never a vendor type."""

        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route")

        with pytest.raises(ProviderError):
            asyncio.run(provider(handler).embed_one("a"))


class TestMalformedResponses:
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

    def test_a_missing_data_array_is_a_provider_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "list"})

        with pytest.raises(ProviderError, match="no embedding data"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_an_entry_without_a_vector_is_a_provider_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0}]})

        with pytest.raises(ProviderError, match="carries no vector"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_an_entry_that_is_not_an_object_is_a_provider_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": ["not an object"]})

        with pytest.raises(ProviderError, match="not an object"):
            asyncio.run(provider(handler).embed_one("a"))

    def test_a_short_data_array_is_caught_by_validation(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body([unit(1, 0, 0, 0)]))

        with pytest.raises(EmbeddingValidationError, match="pairing"):
            asyncio.run(provider(handler).embed(["a", "b"]))


class TestLifecycle:
    def test_an_injected_client_is_not_closed_by_the_adapter(self) -> None:
        """Whoever created the client owns its lifetime."""
        client = httpx.AsyncClient(transport=httpx.MockTransport(echo([])))
        embedder = provider(echo([]), client=client)

        asyncio.run(embedder.aclose())
        assert not client.is_closed
        asyncio.run(client.aclose())


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestEnablement:
    def test_the_remote_adapter_is_not_enabled_by_default(self) -> None:
        """The security property of the slice.

        A remote embedder sends corpus text - including spans from documents under an ACL -
        to a third party. It is enabled by an operator's decision and never because a key
        happens to be present (``docs/SECURITY.md``).
        """
        settings = make_settings()
        assert "http" not in settings.embedding_enabled_providers
        assert settings.embedding_enabled_providers == OFFLINE_PROVIDERS

    def test_knowing_how_to_build_an_adapter_does_not_enable_it(self) -> None:
        assert "http" in EMBEDDING_FACTORIES
        registry = EmbeddingRegistry(make_settings())
        with pytest.raises(ProviderUnavailable, match="not enabled"):
            registry.get("http")

    def test_the_default_embedder_needs_no_credential_and_no_network(self) -> None:
        """``ARCHITECTURE.md`` 11.4, asserted rather than promised."""
        settings = make_settings()
        assert settings.embedding_provider == "hashing"
        assert isinstance(EmbeddingRegistry(settings).default(), HashingEmbedder)

    def test_no_embedding_credential_is_configured_in_tests(self) -> None:
        assert make_settings().embedding_api_key is None

    def test_an_unknown_adapter_name_is_refused(self) -> None:
        settings = make_settings(embedding_enabled_providers=("nonesuch",))
        with pytest.raises(ProviderUnavailable, match="no adapter"):
            EmbeddingRegistry(settings).get("nonesuch")

    def test_an_enabled_remote_adapter_still_needs_its_configuration(self) -> None:
        """Enabling it is necessary and not sufficient; the missing credential surfaces at
        selection time rather than mid-run."""
        settings = make_settings(embedding_enabled_providers=("http",))
        with pytest.raises(ProviderUnavailable):
            EmbeddingRegistry(settings).get("http")

    def test_the_replay_adapter_reports_the_absent_fixture_clearly(self) -> None:
        """No provider vectors are committed, so the semantic question is still open."""
        registry = EmbeddingRegistry(make_settings())
        with pytest.raises(ProviderUnavailable, match="none is"):
            registry.get("replay")

    def test_an_override_wins_over_a_factory(self) -> None:
        """How a test injects a stub, and how a recorded fixture could be pinned for one
        run without touching configuration."""
        stub = HashingEmbedder(dimensions=8)
        registry = EmbeddingRegistry(make_settings(), overrides={"http": stub})

        assert registry.is_enabled("http")
        assert registry.get("http") is stub

    def test_an_instance_is_built_once_and_cached(self) -> None:
        registry = EmbeddingRegistry(make_settings())
        assert registry.get("hashing") is registry.get("hashing")

    def test_the_configured_width_reaches_the_default_embedder(self) -> None:
        settings = make_settings(embedding_dimensions=32)
        assert EmbeddingRegistry(settings).default().dimensions == 32


class TestPortConformance:
    @pytest.mark.parametrize(
        "embedder",
        [
            HashingEmbedder(dimensions=16),
            HttpEmbeddingProvider(
                model=MODEL,
                dimensions=DIMENSIONS,
                base_url=BASE_URL,
                api_key=KEY,
                client=httpx.AsyncClient(transport=httpx.MockTransport(echo([]))),
            ),
        ],
    )
    def test_every_adapter_satisfies_the_embedding_port(self, embedder: object) -> None:
        assert isinstance(embedder, EmbeddingPort)

    def test_the_adapter_base_satisfies_the_port_without_subclassing_it(self) -> None:
        """``EmbeddingPort`` is a Protocol, so conformance is structural - an adapter owes
        it nothing at import time and cannot drift from it silently either."""
        assert issubclass(HttpEmbeddingProvider, EmbeddingAdapter)
        assert isinstance(provider(echo([])), EmbeddingPort)


class TestNothingUpstreamChanged:
    def test_the_hashing_embedders_identity_is_unchanged(self) -> None:
        """The vector benchmark is pinned to ``hashing-1`` at 256 dimensions. The adapter
        foundation must not have touched it - a changed name would change
        ``scorer_version`` and silently invalidate every measured figure."""
        embedder = HashingEmbedder()
        assert embedder.name == "hashing-1"
        assert embedder.dimensions == 256

    def test_the_hashing_embedder_did_not_adopt_the_new_base(self) -> None:
        """Deliberately not retrofitted. Validation belongs where the data is untrusted,
        and this embedder's output is computed locally and already tested; putting it
        behind the new base would have risked its identity for no guarantee gained.
        """
        assert not issubclass(HashingEmbedder, EmbeddingAdapter)
