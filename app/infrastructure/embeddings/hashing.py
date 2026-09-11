"""A deterministic, offline embedder. Satisfies ``EmbeddingPort`` with no API key.

``ARCHITECTURE.md`` 11.4 requires the default local embedding implementation to need no
credentials, so that ingestion and retrieval are developable and testable offline. This is
that implementation, and it is the one the test suite and the benchmark use.

**Read this before reading a single number produced with it.** This is *feature hashing over
a bag of words*, L2-normalized. It is a vector, and cosine over it is a real similarity, but
it carries **no semantic knowledge whatsoever**: two texts are close only if they share
literal tokens. It cannot know that "sign off" means "approval" or that "45 days" is a
retention window. Anything measured with this embedder is a measurement of the vector
*machinery* - the index, the filters, the ordering, the provenance - and of a
term-overlap signal expressed as cosine. It is **not** a measurement of semantic retrieval,
and the docstring says so here rather than in a footnote because the temptation to read it
that way is the whole risk of shipping a fake.

The honest semantic measurement needs a real provider, embedded once and committed:
:mod:`app.infrastructure.embeddings.replay` is the seam for that, and it deliberately fails
rather than falling back to this class.

Three properties it does have, all of which the machinery needs:

**Determinism across processes and platforms.** Hashing uses :mod:`hashlib`, never the
built-in ``hash()``, which is randomly salted per process - a vector index built with
``hash()`` would rank differently on every run and no benchmark over it could be compared
with its own history.

**Its own tokenizer, deliberately not the lexical one.** Reusing
``app.retrieval.tokens.tokenize`` would be less code and would make the two retrieval
strategies share a preprocessing stage. That matters beyond tidiness: checkpoint 9.3 counts
corroboration across **distinct retrieval strategies**, and two strategies that agree
because they share a tokenizer are closer to one strategy than to two. So this splits on
non-alphanumeric characters, folds case, and stops there - no stopword list, no suffix rule,
no dependency on either.

**A fingerprint that changes when the vectors would.** ``name`` and ``dimensions`` travel
into the retriever's ``scorer_version``, so re-dimensioning or reseeding this class cannot
be mistaken for the same index.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Final

#: Bumped when the vector a given text maps to would change: the tokenizer, the hash, the
#: sign rule or the normalization. Not bumped for a dimension change, which travels
#: separately because it is a configuration choice rather than a different recipe.
HASHING_EMBEDDER_VERSION: Final = "hashing-1"

#: Small enough to keep a committed fixture readable, large enough that collisions between
#: the corpus's ~700 distinct tokens are rare. Chosen for those two reasons and **not**
#: tuned against any retrieval measurement.
DEFAULT_DIMENSIONS: Final = 256

MIN_DIMENSIONS: Final = 8
MAX_DIMENSIONS: Final = 4096

_TERM = re.compile(r"[a-z0-9]+")


def terms(text: str) -> tuple[str, ...]:
    """Fold case and split on anything that is not a letter or digit.

    Deliberately the whole of it. This is not the lexical tokenizer and must not grow into
    a copy of one - see the module docstring on why the two strategies keep their
    preprocessing separate.
    """
    return tuple(_TERM.findall(text.casefold()))


def _bucket_and_sign(term: str, dimensions: int) -> tuple[int, float]:
    """Map a term to a coordinate and a sign, reproducibly on every platform.

    The sign is what keeps unrelated terms from only ever adding to each other: two terms
    colliding on one coordinate cancel about half the time instead of always reinforcing.
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimensions, (1.0 if (value >> 63) & 1 else -1.0)


class HashingEmbedder:
    """Feature hashing over a bag of words. Deterministic, offline, and not semantic."""

    def __init__(self, *, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        if not MIN_DIMENSIONS <= dimensions <= MAX_DIMENSIONS:
            raise ValueError(
                f"dimensions must be in [{MIN_DIMENSIONS}, {MAX_DIMENSIONS}], got {dimensions}"
            )
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return HASHING_EMBEDDER_VERSION

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text. Async to satisfy the port; performs no I/O and cannot fail."""
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        """One text to one vector.

        A text with no terms embeds to the **zero vector** rather than to an arbitrary
        point. That is the honest answer - it says "no signal" - and the retriever scores it
        as no match rather than as a spuriously close one.
        """
        vector = [0.0] * self._dimensions
        for term in terms(text):
            bucket, sign = _bucket_and_sign(term, self._dimensions)
            vector[bucket] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]
