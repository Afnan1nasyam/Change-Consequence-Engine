"""The shared shape every embedding adapter has: metadata, validation, batching.

Foundation only. This module makes a *remote* embedding provider possible without changing
anything above it, and it deliberately performs no I/O of its own.

Why a base class at all, when :class:`~app.domain.interfaces.llm.EmbeddingPort` is three
members wide. Because the port describes what a caller needs and says nothing about what an
*adapter* owes, and three of those obligations are exactly where a remote provider goes
wrong in ways that are invisible downstream:

**A vector of the wrong width is a silent catastrophe.** ``VectorRetriever`` refuses a
mis-sized vector at index construction, which is the right last line of defence but a late
one - by then the batch has been paid for and the provider's answer has been trusted. A
provider that silently truncates, or that returns a different model's width because a model
identifier was mistyped, is caught here instead, per vector, with the index it came from.

**A cosine over unnormalized vectors is not the number the benchmark measured.** Some
providers return unit vectors and some do not, and the ones that do occasionally stop.
:class:`VectorRetriever` computes true cosine so it survives either - but a *ranking*
measured under one convention is not comparable with one measured under the other. Where an
adapter declares that it expects unit vectors, that expectation is checked rather than
assumed.

**Provenance has to be attached at the source.** A recall figure belongs to one provider,
one model and one adapter version. :class:`EmbeddingMetadata` carries all three and derives
the ``name`` the retriever folds into its ``scorer_version``, so re-pointing a model
identifier cannot look like the same index.

**Batching is the adapter's problem, not the caller's.** ``EmbeddingPort.embed`` takes a
sequence and returns one vector per text, in order. A remote provider has a request-size
limit, so somebody has to split the call - and if that somebody is the caller, every caller
reimplements it and one of them will reassemble out of order. :meth:`EmbeddingAdapter.embed`
chunks deterministically and concatenates in input order, so the pairing the vector index
relies on holds by construction.

What this module does **not** do: no HTTP, no credentials, no provider selection, no
caching, and no retry. Transport lives in an adapter, selection in
``app/infrastructure/embeddings/registry.py``, and retry/backoff is a gateway concern that
embeddings do not yet share.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.domain.interfaces.llm import ProviderError

#: Bumped when the validation rules or the batching contract change, because either alters
#: what an adapter guarantees about its output. Travels in every adapter's ``name``.
EMBEDDING_ADAPTER_VERSION: Final = "embed-adapter-1"

#: How far from unit length a vector may be and still count as normalized. Wide enough for
#: float32 round-tripping through JSON, narrow enough to catch a provider that stopped
#: normalizing: a genuinely unnormalized vector is off by far more than this.
UNIT_NORM_TOLERANCE: Final = 1e-3

#: Bounds on a batch. An unbounded batch is a request nobody sized and a cost nobody
#: predicted; a batch of one wastes a round trip per span on a corpus of thousands.
MIN_BATCH_SIZE: Final = 1
MAX_BATCH_SIZE: Final = 2048


class EmbeddingValidationError(ProviderError):
    """An adapter returned vectors that cannot be trusted.

    A ``ProviderError`` subclass rather than a bare ``ValueError``, because that is what the
    application layer already catches for provider failures - and because this *is* a
    provider failure: the request succeeded and the answer is unusable. Not retryable: a
    provider returning the wrong width will do it again.
    """

    retryable = False


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Who produced a vector, with what, under which adapter rules.

    All three are needed and none is redundant. ``provider`` and ``model`` identify the
    thing that computed the numbers; ``adapter_version`` identifies the validation and
    batching contract they passed through, which is what makes two runs comparable rather
    than merely similarly labelled.
    """

    #: Adapter/provider identifier, e.g. ``"hashing"`` or ``"openai-compatible"``.
    provider: str
    #: The model identifier, which comes from configuration and never from a default.
    model: str
    dimensions: int
    #: Whether this adapter's provider is expected to return unit vectors. Checked when
    #: true; a provider that stops normalizing is a silent ranking change otherwise.
    expects_unit_vectors: bool = False
    adapter_version: str = EMBEDDING_ADAPTER_VERSION

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("embedding metadata needs a provider")
        if not self.model:
            raise ValueError(
                "embedding metadata needs a model identifier. It comes from configuration "
                "(CCE_MODEL_EMBEDDING) and has no default, because a recall figure measured "
                "under an unnamed model cannot be reproduced."
            )
        if self.dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {self.dimensions}")

    @property
    def name(self) -> str:
        """The ``EmbeddingPort.name`` value, and the identity the retriever records.

        Every component that would change the vectors is in it, so a re-pointed model
        identifier or a changed adapter contract cannot be mistaken for the same index.
        """
        return f"{self.adapter_version}:{self.provider}/{self.model}"


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    metadata: EmbeddingMetadata,
) -> list[list[float]]:
    """Check a provider's answer before anything downstream trusts it.

    Four checks, in the order that localizes a fault fastest: how many vectors came back,
    then per vector its width, then that its components are finite, then - only where the
    adapter says the provider promises it - that it is unit length.

    The count check is first because a wrong count means the *pairing* is broken, and a
    broken pairing attaches every subsequent vector to the wrong span. That produces a
    plausible, wrong ranking rather than an error, which is the worst available outcome.
    """
    if len(vectors) != expected_count:
        raise EmbeddingValidationError(
            f"{metadata.provider} returned {len(vectors)} vectors for {expected_count} texts; "
            "the pairing between texts and vectors is positional and cannot be recovered",
            provider=metadata.provider,
        )

    checked: list[list[float]] = []
    for index, vector in enumerate(vectors):
        if len(vector) != metadata.dimensions:
            raise EmbeddingValidationError(
                f"{metadata.provider} vector {index} has {len(vector)} components, but "
                f"model {metadata.model!r} is configured as {metadata.dimensions}-dimensional. "
                "Mixing widths in one index makes every cosine in it meaningless.",
                provider=metadata.provider,
            )

        for component in vector:
            if not math.isfinite(component):
                raise EmbeddingValidationError(
                    f"{metadata.provider} vector {index} contains a non-finite component; "
                    "a NaN would silently poison every comparison it takes part in",
                    provider=metadata.provider,
                )

        if metadata.expects_unit_vectors:
            norm = math.sqrt(sum(component * component for component in vector))
            if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
                raise EmbeddingValidationError(
                    f"{metadata.provider} vector {index} has norm {norm:.6f}, but this adapter "
                    "expects unit vectors. A ranking measured under one normalization "
                    "convention is not comparable with one measured under the other.",
                    provider=metadata.provider,
                )

        checked.append(list(vector))

    return checked


def batches(texts: Sequence[str], size: int) -> list[Sequence[str]]:
    """Split ``texts`` into consecutive chunks of at most ``size``, order preserved."""
    if not MIN_BATCH_SIZE <= size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be in [{MIN_BATCH_SIZE}, {MAX_BATCH_SIZE}], got {size}")
    return [texts[start : start + size] for start in range(0, len(texts), size)]


class EmbeddingAdapter(ABC):
    """Satisfies ``EmbeddingPort``, and owes validation, batching and provenance.

    A subclass implements exactly one method - :meth:`_embed_batch`, the wire call for one
    batch - and inherits the rest. That split is the point: everything a subclass could get
    subtly wrong in a way nothing downstream would notice is implemented once, here.
    """

    def __init__(self, *, metadata: EmbeddingMetadata, max_batch_size: int) -> None:
        if not MIN_BATCH_SIZE <= max_batch_size <= MAX_BATCH_SIZE:
            raise ValueError(
                f"max_batch_size must be in [{MIN_BATCH_SIZE}, {MAX_BATCH_SIZE}], "
                f"got {max_batch_size}"
            )
        self._metadata = metadata
        self._max_batch_size = max_batch_size

    # -- EmbeddingPort ---------------------------------------------------

    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def dimensions(self) -> int:
        return self._metadata.dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in input order, validated.

        An empty request makes **no call at all** and returns an empty list. That is worth
        stating: a provider billed per request should not be paid for a question nobody
        asked, and some reject an empty batch with a 400 that would surface as a provider
        failure rather than as the non-event it is.
        """
        if not texts:
            return []

        collected: list[list[float]] = []
        for batch in batches(texts, self._max_batch_size):
            raw = await self._embed_batch(batch)
            collected.extend(
                validate_vectors(raw, expected_count=len(batch), metadata=self._metadata)
            )

        # Re-checked over the whole request, not only per batch: a chunking mistake would
        # otherwise pass every individual batch and still return the wrong number.
        if len(collected) != len(texts):
            raise EmbeddingValidationError(
                f"batching produced {len(collected)} vectors for {len(texts)} texts",
                provider=self._metadata.provider,
            )
        return collected

    # -- the adapter's own surface ---------------------------------------

    @property
    def metadata(self) -> EmbeddingMetadata:
        """Provenance, for whatever records which embedder produced a figure.

        Deliberately **not** on ``EmbeddingPort``: the port describes what retrieval needs,
        and retrieval needs a name and a width. Widening the domain contract to carry
        infrastructure provenance would put vendor-shaped detail in the domain layer for no
        caller's benefit.
        """
        return self._metadata

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    async def embed_one(self, text: str) -> list[float]:
        """One text, one vector. The single-text path, so callers do not unwrap a list."""
        vectors = await self.embed([text])
        return vectors[0]

    @abstractmethod
    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one batch, already sized to :attr:`max_batch_size`.

        Return one vector per text, in the order given. Implementations do not validate:
        :meth:`embed` does that, so a subclass cannot forget to.
        """

    async def aclose(self) -> None:
        """Release transport resources. A no-op for adapters that hold none."""
        return None
