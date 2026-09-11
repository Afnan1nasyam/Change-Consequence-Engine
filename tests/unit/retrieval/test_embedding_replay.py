"""The replay embedder: recorded vectors, and a loud failure when one is missing.

The test worth reading is :func:`TestCacheMiss.test_a_missing_text_raises_rather_than_
substituting`. Every other behaviour here is bookkeeping; that one is the reason the class
exists. A fallback - embedding the text some other way, or returning zeros - would put two
embedders inside one index, where the cosine between a recorded vector and a substituted one
is meaningless, and the recall figure over it would read as a retrieval result rather than
as a fixture defect.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.infrastructure.embeddings.replay import (
    EmbeddingCacheMiss,
    RecordedEmbeddings,
    ReplayEmbedder,
    load_recorded,
    text_digest,
)

REFUND = "Refunds of 10,000 USD or more require manager approval."
RETENTION = "Personal data is retained for 90 days."


def recorded(**over: object) -> RecordedEmbeddings:
    payload: dict[str, object] = {
        "provider": "acme-embed",
        "model": "acme-embed-small",
        "dimensions": 3,
        "vectors": {
            text_digest(REFUND): (1.0, 0.0, 0.0),
            text_digest(RETENTION): (0.0, 1.0, 0.0),
        },
    }
    payload.update(over)
    return RecordedEmbeddings.model_validate(payload)


class TestFixture:
    def test_it_names_the_provider_and_model_that_produced_it(self) -> None:
        """A vector fixture with no provenance is a pile of numbers nobody can reproduce
        or supersede."""
        embedder = ReplayEmbedder(recorded())
        assert "acme-embed" in embedder.name
        assert "acme-embed-small" in embedder.name

    def test_the_name_says_the_vectors_were_replayed(self) -> None:
        """So a recall figure cannot be read as though the provider ran here."""
        assert ReplayEmbedder(recorded()).name.startswith("embed-replay-1")

    def test_the_declared_width_is_the_ports_width(self) -> None:
        assert ReplayEmbedder(recorded()).dimensions == 3

    def test_a_vector_of_the_wrong_width_is_refused_at_load(self) -> None:
        """A fixture that disagrees with itself must fail loading, not at query time."""
        with pytest.raises(ValueError, match="declares"):
            recorded(vectors={text_digest(REFUND): (1.0, 0.0)})

    def test_a_fixture_without_a_provider_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            recorded(provider="")

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            RecordedEmbeddings.model_validate(
                {
                    "provider": "p",
                    "model": "m",
                    "dimensions": 1,
                    "vectors": {},
                    "temperature": 0.4,
                }
            )


class TestReplay:
    def test_a_recorded_text_returns_its_recorded_vector(self) -> None:
        assert ReplayEmbedder(recorded()).embed_one(REFUND) == [1.0, 0.0, 0.0]

    def test_the_batch_form_preserves_order(self) -> None:
        """The pairing between spans and vectors is positional, so order is not cosmetic."""
        vectors = asyncio.run(ReplayEmbedder(recorded()).embed([RETENTION, REFUND]))
        assert vectors == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]

    def test_replay_is_deterministic(self) -> None:
        embedder = ReplayEmbedder(recorded())
        assert embedder.embed_one(REFUND) == embedder.embed_one(REFUND)

    def test_the_key_is_the_exact_text(self) -> None:
        """A whitespace edit is a new key, so a fixture cannot drift onto a changed span."""
        embedder = ReplayEmbedder(recorded())
        with pytest.raises(EmbeddingCacheMiss):
            embedder.embed_one(REFUND + " ")


class TestCacheMiss:
    def test_a_missing_text_raises_rather_than_substituting(self) -> None:
        with pytest.raises(EmbeddingCacheMiss):
            ReplayEmbedder(recorded()).embed_one("a span nobody recorded")

    def test_the_error_names_the_digest_and_not_the_text(self) -> None:
        """A fixture may cover restricted spans, and an exception message is one of the
        places content leaks into a log."""
        secret = "the salary band for the head of support is confidential"
        with pytest.raises(EmbeddingCacheMiss) as raised:
            ReplayEmbedder(recorded()).embed_one(secret)

        message = str(raised.value)
        assert secret not in message
        assert text_digest(secret)[:12] in message

    def test_the_error_says_what_to_do_about_it(self) -> None:
        """Re-record, never substitute. The message carries the reason because the wrong
        fix produces a plausible-looking number instead of a failure."""
        with pytest.raises(EmbeddingCacheMiss, match="mixing two embedders"):
            ReplayEmbedder(recorded()).embed_one("unrecorded")


class TestLoading:
    def test_a_fixture_round_trips_through_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "vectors.json"
        path.write_text(recorded().model_dump_json(), encoding="utf-8")

        loaded = load_recorded(path)
        assert ReplayEmbedder(loaded).embed_one(REFUND) == [1.0, 0.0, 0.0]

    def test_a_non_object_fixture_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "vectors.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        with pytest.raises(ValueError, match="expected an object"):
            load_recorded(path)

    def test_no_fixture_ships_yet_and_that_is_recorded_not_hidden(self) -> None:
        """There is no committed embedding fixture in the repository.

        Recording one needs one run against a real provider, which needs credentials this
        environment does not have. So the vector benchmark is measured with the hashing
        embedder, every figure says so, and no semantic claim is made anywhere
        (``docs/EVALUATION.md``). This test exists to make the absence deliberate rather
        than something a reader has to notice.
        """
        from app.config.settings import REPO_ROOT

        assert not list((REPO_ROOT / "data" / "fixtures").glob("*embeddings*.json"))
