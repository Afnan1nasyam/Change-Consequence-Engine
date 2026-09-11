"""The recording path, proven end to end without a live call.

``scripts/record_embeddings.py`` is the mechanism of the real-embedding experiment
(``docs/STATUS.md`` 36), and it is the one thing in the repository that may reach a
third-party endpoint. It cannot be run for real here - this environment has no embedding
credential and no reachable local provider - so what these tests establish is that
**everything except the credential works**: the texts collected are the ones the benchmark
scores, the request is validated, the fixture carries the provenance reproducibility needs,
no credential reaches it, and a fixture written this way replays through ``ReplayEmbedder``
into a scoreable vector benchmark.

The provider is an ``httpx.MockTransport`` returning arbitrary unit vectors. **No number
produced here is a retrieval result** - the vectors are meaningless by construction, and the
benchmark run below is asserted only to *complete and be internally consistent*, never for
its recall. That distinction is the whole reason this file exists rather than a fixture being
committed: a fixture labelled with a provider that never ran would make every downstream
figure a fabrication.

The test worth reading first is :func:`TestFixtureProvenance.test_no_credential_reaches_the_fixture`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from scripts.record_embeddings import collect_texts, preflight, record

from app.domain.value_objects.enums import RetrievalStrategy
from app.evaluation.retrieval_benchmark import load_suite, run_vector_suite
from app.infrastructure.embeddings.base import EMBEDDING_ADAPTER_VERSION
from app.infrastructure.embeddings.http import HttpEmbeddingProvider
from app.infrastructure.embeddings.registry import EmbeddingRegistry
from app.infrastructure.embeddings.replay import (
    ReplayEmbedder,
    load_recorded,
    text_digest,
)
from tests.conftest import make_settings

DIMENSIONS = 16
MODEL = "acme-embed-small"
SECRET = "sk-not-a-real-credential-abcdef123456"
BASE_URL = "https://embeddings.example/v1"

Handler = Callable[[httpx.Request], httpx.Response]


def pseudo_vector(text: str) -> list[float]:
    """A deterministic unit vector for a text. Stands in for a provider; means nothing.

    Derived from the text digest so the stand-in is reproducible, which keeps these tests
    deterministic. It carries no semantics whatsoever and must never be read as one.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [float(digest[i % len(digest)]) - 127.5 for i in range(DIMENSIONS)]
    norm = math.sqrt(sum(component * component for component in raw)) or 1.0
    return [component / norm for component in raw]


def handler_returning_unit_vectors(seen: list[dict[str, Any]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        inputs = payload["input"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": pseudo_vector(text)} for i, text in enumerate(inputs)
                ]
            },
        )

    return handler


def recording_settings(tmp_path: Path, **over: object):  # type: ignore[no-untyped-def]
    """Settings as an operator would set them for the recording run."""
    defaults: dict[str, object] = {
        "embedding_provider": "http",
        "embedding_enabled_providers": ("hashing", "replay", "http"),
        "model_embedding": MODEL,
        "embedding_api_key": SECRET,
        "embedding_base_url": BASE_URL,
        "embedding_dimensions": DIMENSIONS,
        "embedding_expects_unit_vectors": True,
        "embedding_fixture_path": tmp_path / "embeddings.json",
    }
    defaults.update(over)
    return make_settings(**defaults)


def mock_registry(settings, seen: list[dict[str, Any]]) -> EmbeddingRegistry:  # type: ignore[no-untyped-def]
    provider = HttpEmbeddingProvider(
        model=MODEL,
        dimensions=DIMENSIONS,
        base_url=BASE_URL,
        api_key=SECRET,
        expects_unit_vectors=True,
        max_batch_size=settings.embedding_max_batch_size,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler_returning_unit_vectors(seen))
        ),
    )
    return EmbeddingRegistry(settings, overrides={"http": provider})


def run_recording(tmp_path: Path, **over: object) -> tuple[int, list[dict[str, Any]]]:
    settings = recording_settings(tmp_path, **over)
    seen: list[dict[str, Any]] = []
    code = asyncio.run(
        record(settings=settings, dry_run=False, registry=mock_registry(settings, seen))
    )
    return code, seen


# ---------------------------------------------------------------------------
# What gets embedded
# ---------------------------------------------------------------------------


class TestWhatIsSent:
    def test_it_collects_the_spans_and_queries_the_benchmark_scores(self) -> None:
        """The 21 admitted spans and the 35 query texts, and nothing larger.

        This call sends corpus text to a third party, so it sends the minimum that makes
        the benchmark runnable - span texts, not whole documents.
        """
        texts, spans, queries = asyncio.run(collect_texts())
        assert spans == 21
        assert queries == 35
        assert len(texts) == len(set(texts))

    def test_duplicate_texts_are_embedded_once(self) -> None:
        """56 texts, 51 distinct: the version-currency pairs ask one question at two dates
        and the permission pair asks one question under two scopes. Paying to embed the
        same string twice would be waste, and the replay key is the text anyway.
        """
        texts, spans, queries = asyncio.run(collect_texts())
        assert spans + queries == 56
        assert len(texts) == 51

    def test_the_order_is_deterministic_across_runs(self) -> None:
        """Sorted by digest, so two recordings send the same list in the same order and a
        diff of two fixtures is meaningful."""
        first, _, _ = asyncio.run(collect_texts())
        second, _, _ = asyncio.run(collect_texts())
        assert first == second
        assert first == sorted(first, key=text_digest)

    def test_the_request_carries_the_configured_model_and_the_texts(self, tmp_path: Path) -> None:
        code, seen = run_recording(tmp_path)
        assert code == 0
        assert seen
        assert all(payload["model"] == MODEL for payload in seen)
        sent = [text for payload in seen for text in payload["input"]]
        assert len(sent) == 51

    def test_a_long_request_is_batched_and_still_complete(self, tmp_path: Path) -> None:
        code, seen = run_recording(tmp_path, embedding_max_batch_size=8)
        assert code == 0
        assert len(seen) == 7  # 51 texts in batches of 8
        sent = [text for payload in seen for text in payload["input"]]
        assert len(sent) == len(set(sent)) == 51


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_the_shipped_configuration_is_not_ready_and_says_why(self) -> None:
        """The state this repository is actually in: no credential, no model, no endpoint.

        Reported as a list rather than one failure at a time, so an operator setting this
        up sees the whole gap at once.
        """
        problems = preflight(make_settings())
        assert len(problems) == 5
        assert any("CCE_EMBEDDING_API_KEY" in p for p in problems)
        assert any("CCE_MODEL_EMBEDDING" in p for p in problems)
        assert any("CCE_EMBEDDING_BASE_URL" in p for p in problems)
        assert any("CCE_EMBEDDING_DIMENSIONS" in p for p in problems)
        assert any("reaches no provider" in p for p in problems)

    def test_a_fully_configured_run_passes_preflight(self, tmp_path: Path) -> None:
        assert preflight(recording_settings(tmp_path)) == []

    @pytest.mark.parametrize(
        "missing",
        ["model_embedding", "embedding_base_url", "embedding_api_key", "embedding_dimensions"],
    )
    def test_each_required_value_is_required(self, tmp_path: Path, missing: str) -> None:
        problems = preflight(recording_settings(tmp_path, **{missing: None}))
        assert problems

    def test_recording_through_an_offline_embedder_is_refused(self, tmp_path: Path) -> None:
        """A fixture claiming a provider ran when none did would make every downstream
        figure a fabrication."""
        problems = preflight(recording_settings(tmp_path, embedding_provider="hashing"))
        assert any("reaches no provider" in p for p in problems)

    def test_an_unenabled_provider_is_refused(self, tmp_path: Path) -> None:
        problems = preflight(
            recording_settings(tmp_path, embedding_enabled_providers=("hashing", "replay"))
        )
        assert any("ENABLED_PROVIDERS" in p for p in problems)

    def test_no_fixture_is_written_when_preflight_fails(self, tmp_path: Path) -> None:
        settings = recording_settings(tmp_path, embedding_api_key=None)
        code = asyncio.run(record(settings=settings, dry_run=False))
        assert code == 2
        assert not settings.embedding_fixture_path.exists()

    def test_a_dry_run_writes_nothing_and_calls_nothing(self, tmp_path: Path) -> None:
        settings = recording_settings(tmp_path)
        seen: list[dict[str, Any]] = []
        code = asyncio.run(
            record(settings=settings, dry_run=True, registry=mock_registry(settings, seen))
        )
        assert code == 0
        assert seen == []
        assert not settings.embedding_fixture_path.exists()


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------


class TestFixtureProvenance:
    def test_no_credential_reaches_the_fixture(self, tmp_path: Path) -> None:
        """The rule with no exceptions. The key builds one header and goes nowhere else."""
        code, _ = run_recording(tmp_path)
        assert code == 0

        raw = (tmp_path / "embeddings.json").read_text(encoding="utf-8")
        assert SECRET not in raw
        assert "api_key" not in raw
        assert "Authorization" not in raw
        assert "authorization" not in raw

    def test_it_records_everything_reproduction_needs(self, tmp_path: Path) -> None:
        code, _ = run_recording(tmp_path)
        assert code == 0

        fixture = load_recorded(tmp_path / "embeddings.json")
        assert fixture.provider == "openai-compatible"
        assert fixture.model == MODEL
        assert fixture.dimensions == DIMENSIONS
        assert fixture.adapter_version == EMBEDDING_ADAPTER_VERSION
        assert fixture.expects_unit_vectors is True
        assert fixture.recorded_at is not None
        assert fixture.recorded_texts == 51

    def test_the_recording_date_is_part_of_the_replayed_identity(self, tmp_path: Path) -> None:
        """A model identifier is not a version: providers re-train behind a stable name, so
        the date is the only thing separating two recordings of the "same" model."""
        run_recording(tmp_path)
        fixture = load_recorded(tmp_path / "embeddings.json")
        name = ReplayEmbedder(fixture).name

        assert MODEL in name
        assert fixture.recorded_at is not None
        assert fixture.recorded_at.date().isoformat() in name

    def test_every_key_is_the_sha256_of_a_text_that_was_sent(self, tmp_path: Path) -> None:
        code, seen = run_recording(tmp_path)
        assert code == 0

        fixture = load_recorded(tmp_path / "embeddings.json")
        sent = {text for payload in seen for text in payload["input"]}
        assert set(fixture.vectors) == {text_digest(text) for text in sent}

    def test_every_requested_text_has_a_vector(self, tmp_path: Path) -> None:
        """Coverage, not just count: a provider that dropped one cannot produce a fixture."""
        run_recording(tmp_path)
        texts, _, _ = asyncio.run(collect_texts())
        fixture = load_recorded(tmp_path / "embeddings.json")

        assert len(fixture.vectors) == len(texts)
        for text in texts:
            assert text_digest(text) in fixture.vectors

    def test_every_vector_has_the_declared_width_and_is_finite(self, tmp_path: Path) -> None:
        run_recording(tmp_path)
        fixture = load_recorded(tmp_path / "embeddings.json")

        for vector in fixture.vectors.values():
            assert len(vector) == DIMENSIONS
            assert all(math.isfinite(component) for component in vector)

    def test_unit_length_is_verified_when_the_provider_promised_it(self, tmp_path: Path) -> None:
        run_recording(tmp_path)
        fixture = load_recorded(tmp_path / "embeddings.json")

        for vector in fixture.vectors.values():
            norm = math.sqrt(sum(component * component for component in vector))
            assert norm == pytest.approx(1.0, abs=1e-3)

    def test_a_provider_returning_the_wrong_width_writes_no_fixture(self, tmp_path: Path) -> None:
        """The guard against a mistyped model identifier. A partial or wrong recording is
        worse than none, because it looks like a complete one."""
        settings = recording_settings(tmp_path, embedding_dimensions=DIMENSIONS)
        wrong = HttpEmbeddingProvider(
            model=MODEL,
            dimensions=DIMENSIONS,
            base_url=BASE_URL,
            api_key=SECRET,
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        json={
                            "data": [
                                {"index": i, "embedding": [0.1, 0.2]}
                                for i, _ in enumerate(json.loads(request.content)["input"])
                            ]
                        },
                    )
                )
            ),
        )
        code = asyncio.run(
            record(
                settings=settings,
                dry_run=False,
                registry=EmbeddingRegistry(settings, overrides={"http": wrong}),
            )
        )
        assert code == 1
        assert not settings.embedding_fixture_path.exists()

    def test_a_provider_failure_writes_no_fixture(self, tmp_path: Path) -> None:
        settings = recording_settings(tmp_path)
        failing = HttpEmbeddingProvider(
            model=MODEL,
            dimensions=DIMENSIONS,
            base_url=BASE_URL,
            api_key=SECRET,
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(429, json={"error": "slow down"})
                )
            ),
        )
        code = asyncio.run(
            record(
                settings=settings,
                dry_run=False,
                registry=EmbeddingRegistry(settings, overrides={"http": failing}),
            )
        )
        assert code == 1
        assert not settings.embedding_fixture_path.exists()


# ---------------------------------------------------------------------------
# Replaying it
# ---------------------------------------------------------------------------


class TestReplayingARecording:
    def test_a_recorded_fixture_replays_offline(self, tmp_path: Path) -> None:
        run_recording(tmp_path)
        fixture = load_recorded(tmp_path / "embeddings.json")
        embedder = ReplayEmbedder(fixture)
        texts, _, _ = asyncio.run(collect_texts())

        vectors = asyncio.run(embedder.embed(texts))
        assert len(vectors) == len(texts)
        assert vectors[0] == pseudo_vector(texts[0])

    def test_the_registry_selects_it_from_configuration(self, tmp_path: Path) -> None:
        """The switch an operator makes after recording: ``CCE_EMBEDDING_PROVIDER=replay``."""
        run_recording(tmp_path)
        settings = recording_settings(tmp_path, embedding_provider="replay")

        embedder = EmbeddingRegistry(settings).default()
        assert isinstance(embedder, ReplayEmbedder)
        assert embedder.dimensions == DIMENSIONS

    def test_the_vector_benchmark_runs_on_a_recorded_fixture(self, tmp_path: Path) -> None:
        """The end of the pipeline: recording feeds the benchmark with no live call.

        **The recall figures produced here are meaningless** - the stand-in vectors carry
        no semantics. What is asserted is that the run completes and is internally
        consistent: every label resolves, reachability is unchanged, and nothing leaks. A
        real recording would produce figures worth reading, and would need the suite's
        ``vector_scorer_version`` pin updated to match.
        """
        run_recording(tmp_path)
        fixture = load_recorded(tmp_path / "embeddings.json")
        report = asyncio.run(run_vector_suite(load_suite(), embedder=ReplayEmbedder(fixture)))

        assert report.strategy is RetrievalStrategy.VECTOR
        assert report.label_defects == ()
        assert report.leaks == ()
        assert report.metrics.overall.expectations == 62
        assert report.metrics.overall.retrievable == 56
        assert report.metrics.overall.not_admitted == 4
        assert report.metrics.overall.withheld_by_permission == 2
        assert report.corpus_spans == 21

    def test_replaying_a_text_that_was_not_recorded_raises(self, tmp_path: Path) -> None:
        """So a fixture that no longer covers the corpus is a loud failure rather than a
        recall regression nobody can explain."""
        from app.infrastructure.embeddings.replay import EmbeddingCacheMiss

        run_recording(tmp_path)
        embedder = ReplayEmbedder(load_recorded(tmp_path / "embeddings.json"))

        with pytest.raises(EmbeddingCacheMiss):
            embedder.embed_one("a span nobody recorded")


class TestNoFixtureIsCommitted:
    def test_the_repository_ships_no_recorded_embeddings(self) -> None:
        """Stated as a test, because the absence is the honest state of the experiment.

        No embedding credential and no reachable local provider exist in this environment,
        so the live call has not happened and **no semantic claim about vector retrieval is
        supported by anything**. Committing a synthetic fixture to make the pipeline look
        finished is the one thing that would make that worse (``docs/STATUS.md`` 36).
        """
        from app.config.settings import REPO_ROOT

        assert not (REPO_ROOT / "data" / "fixtures" / "embeddings.json").exists()

    def test_the_shipped_configuration_still_uses_the_offline_embedder(self) -> None:
        settings = make_settings()
        assert settings.embedding_provider == "hashing"
        assert "http" not in settings.embedding_enabled_providers
