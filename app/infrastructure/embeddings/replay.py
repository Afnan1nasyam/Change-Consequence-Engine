"""Recorded vectors instead of a live embedding call. The seam a real provider fills.

Same argument as ``app/evaluation/replay.py`` makes for extraction, applied to embeddings.
``docs/EVALUATION.md`` needs runs that differ in *one* variable, and a live embedding
provider makes that impossible: the vectors move when the vendor updates a model, so a
recall figure could change with no code change and no corpus change. Recording the vectors
once and committing them holds the embedder fixed.

It is also the only honest way to measure **semantic** retrieval in this repository. The
default :class:`~app.infrastructure.embeddings.hashing.HashingEmbedder` is a term-overlap
signal in vector form and knows no semantics at all; a real provider does. Running one
provider once, committing the vectors, and pointing this class at them turns the vector
benchmark into a genuine semantic measurement that still needs no credentials and no
network in CI.

**A cache miss raises.** It does not embed the text some other way, and it does not return a
zero vector. Either fallback would silently mix two embedders inside one index, where the
cosine between a recorded vector and a substituted one is meaningless - and the resulting
recall number would look like a retrieval result rather than a fixture defect. The failure
is loud for the same reason a recorded extraction quote that no longer matches is reported
as an extraction failure rather than skipped.

Nothing here imports a vendor SDK. A provider is reached once, offline, by whatever script
records the fixture; this class only reads what that produced.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.infrastructure.embeddings.base import EMBEDDING_ADAPTER_VERSION

REPLAY_EMBEDDER_VERSION: Final = "embed-replay-1"


class EmbeddingCacheMiss(LookupError):
    """A text was asked for that the fixture does not hold.

    Carries the digest rather than the text: a fixture may cover restricted document spans,
    and an exception message is one of the places content leaks into a log.
    """

    def __init__(self, digest: str, *, recorded: int) -> None:
        super().__init__(
            f"no recorded embedding for text sha256:{digest[:12]}... "
            f"({recorded} texts in the fixture). Re-record the fixture rather than "
            "substituting another embedder: mixing two embedders in one index makes every "
            "cosine in it meaningless."
        )
        self.digest = digest


def text_digest(text: str) -> str:
    """The fixture key. SHA-256 of the exact text, so a whitespace edit is a new key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RecordedEmbeddings(BaseModel):
    """A committed embedding fixture: which provider produced it, and the vectors.

    ``provider`` and ``model`` are recorded because they are the whole point - a vector
    fixture with no provenance is a pile of numbers nobody can reproduce or supersede.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)

    #: The adapter contract the vectors passed through on the way in. Part of provenance
    #: because the validation and batching rules are what make two recordings comparable
    #: rather than merely similarly labelled.
    adapter_version: str = EMBEDDING_ADAPTER_VERSION
    #: Whether the recording provider was expected to return unit vectors, carried so a
    #: replayed run can be checked under the same convention it was recorded under.
    expects_unit_vectors: bool = False

    #: When the live call happened, UTC. A model identifier is not a version: providers
    #: silently re-train behind a stable name, so the date is the only thing that
    #: distinguishes two recordings of the "same" model.
    recorded_at: dt.datetime | None = None
    #: What was embedded, for a reader who needs to know whether a fixture still covers the
    #: corpus. Counts only - never the texts, which may include restricted spans.
    recorded_texts: int = Field(default=0, ge=0)

    #: ``sha256(text) -> vector``. The key **is** the source-text hash, so a fixture cannot
    #: drift onto edited text: an altered span is simply a cache miss.
    vectors: dict[str, tuple[float, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _widths_agree_with_the_declaration(self) -> Self:
        """A fixture whose vectors disagree with its declared width cannot be constructed.

        A validator rather than a method somebody has to call: the failure this prevents -
        two widths inside one index - is silent, and an opt-in check is one a caller can
        skip by writing ``RecordedEmbeddings(...)`` directly.
        """
        for digest, vector in self.vectors.items():
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"recorded vector {digest[:12]}... has {len(vector)} components, "
                    f"but the fixture declares {self.dimensions}"
                )
        return self


class ReplayEmbedder:
    """Satisfies ``EmbeddingPort`` from a committed fixture. Never calls a provider."""

    def __init__(self, recorded: RecordedEmbeddings) -> None:
        self._recorded = recorded

    @property
    def name(self) -> str:
        """Names the recorded provider *and* the replay, so a figure is attributable to
        both without pretending the provider ran here."""
        recorded = self._recorded
        stamp = f"@{recorded.recorded_at.date().isoformat()}" if recorded.recorded_at else ""
        return f"{REPLAY_EMBEDDER_VERSION}:{recorded.provider}/{recorded.model}{stamp}"

    @property
    def dimensions(self) -> int:
        return self._recorded.dimensions

    @property
    def size(self) -> int:
        return len(self._recorded.vectors)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        digest = text_digest(text)
        vector = self._recorded.vectors.get(digest)
        if vector is None:
            raise EmbeddingCacheMiss(digest, recorded=self.size)
        return list(vector)


def load_recorded(path: Path) -> RecordedEmbeddings:
    """Load and validate an embedding fixture."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object with provider, model and vectors")
    return RecordedEmbeddings.model_validate(raw)
