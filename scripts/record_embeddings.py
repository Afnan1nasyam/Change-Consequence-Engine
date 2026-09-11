"""Make the one live embedding call, validate it, and commit the vectors.

    python -m scripts.record_embeddings --check     # preflight only, no call
    python -m scripts.record_embeddings --dry-run   # show exactly what would be sent
    python -m scripts.record_embeddings             # the live call

This is the mechanism of the real-embedding experiment (``docs/STATUS.md`` 36). It exists as
a script rather than a test because it is the **only** thing in the repository that may reach
a third-party endpoint, and that should be a deliberate command an operator runs, never a
side effect of running the suite.

What it embeds: the **21 admitted evidence span texts** and the **35 benchmark query texts**
- the exact inputs the vector benchmark scores. Nothing else, and nothing larger: this call
sends corpus text to a third party, so it sends the minimum that makes the benchmark
runnable and not, for instance, whole documents.

Ordering and coverage are checked rather than assumed. Texts are deduplicated and sorted by
their own digest, so two runs send the same list in the same order; every requested text must
come back with a vector, and the fixture is written only if all of them did.

**The API key is never written anywhere.** It is read from the environment through
``Settings``, used to build one ``Authorization`` header, and never enters the fixture, the
log line, the error text or the repository. What the fixture records is provider, model,
dimensions, adapter version, the recording timestamp, the text count and the per-text
SHA-256 - everything needed to reproduce or supersede the recording, and nothing that
authenticates.

**It refuses to write a fixture it did not measure.** There is no offline mode, no synthetic
fallback and no partial write. A fixture labelled with a provider that never ran would make
every downstream recall figure a fabrication, and ``ReplayEmbedder`` is built to raise rather
than let two embedders share an index for the same reason.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys

from app.config.settings import Settings, get_settings
from app.domain.interfaces.llm import EmbeddingPort, ProviderError, ProviderUnavailable
from app.evaluation.corpus import load_corpus
from app.evaluation.retrieval_benchmark import load_suite
from app.infrastructure.embeddings.base import EMBEDDING_ADAPTER_VERSION, validate_vectors
from app.infrastructure.embeddings.registry import EmbeddingRegistry
from app.infrastructure.embeddings.replay import RecordedEmbeddings, text_digest

RULE = "=" * 92

#: The adapters that reach a provider. Recording through an offline one would produce a
#: fixture that says a provider ran when none did.
LIVE_PROVIDERS = frozenset({"http", "gemini"})


class PreflightError(RuntimeError):
    """Configuration is not ready for a live call. Raised before any request is made."""


def preflight(settings: Settings) -> list[str]:
    """Every reason the live call cannot run yet, in one pass.

    Reported together rather than one at a time: an operator setting this up for the first
    time should see the whole list, not discover it four runs in a row.
    """
    problems: list[str] = []

    if settings.embedding_provider not in LIVE_PROVIDERS:
        problems.append(
            f"CCE_EMBEDDING_PROVIDER is {settings.embedding_provider!r}, which reaches no "
            f"provider. Set it to one of {sorted(LIVE_PROVIDERS)} for the recording run."
        )
    if settings.embedding_provider not in settings.embedding_enabled_providers:
        problems.append(
            f"{settings.embedding_provider!r} is not in CCE_EMBEDDING_ENABLED_PROVIDERS "
            f"{list(settings.embedding_enabled_providers)}. A remote embedder sends corpus "
            "text to a third party, so enabling it is an explicit decision."
        )
    if not settings.model_embedding:
        problems.append("CCE_MODEL_EMBEDDING is unset. A vector index has to name its model.")
    if not settings.embedding_base_url:
        problems.append("CCE_EMBEDDING_BASE_URL is unset.")
    if settings.embedding_api_key is None:
        problems.append("CCE_EMBEDDING_API_KEY is unset.")
    if settings.embedding_dimensions is None:
        problems.append(
            "CCE_EMBEDDING_DIMENSIONS is unset. It is validated per vector, because nothing "
            "in a response proves which model answered."
        )
    return problems


async def collect_texts() -> tuple[list[str], int, int]:
    """The span texts and the query texts, deduplicated and deterministically ordered.

    Sorted by digest rather than by corpus order, so the request is identical between runs
    and a diff of two fixtures is meaningful. Returns the texts plus the two counts, which
    the report prints so a reader can see the corpus was fully covered.
    """
    stores = await load_corpus()
    if stores.extraction_failures:
        raise PreflightError(
            f"the corpus load reported extraction failures, so the spans are not the ones "
            f"the benchmark scores: {stores.extraction_failures}"
        )

    spans = [item.text_span for item in stores.sources.evidence_for_tenant(stores.tenant_id)]
    queries = [query.text for query in load_suite().queries]

    unique = sorted({*spans, *queries}, key=text_digest)
    return unique, len(spans), len(queries)


async def record(
    *,
    settings: Settings,
    dry_run: bool,
    registry: EmbeddingRegistry | None = None,
) -> int:
    """Preflight, collect, embed, validate, write. ``registry`` is injectable so the
    recording path can be proven through a mock transport without a live call."""
    problems = preflight(settings)
    texts, span_count, query_count = await collect_texts()

    print(RULE)
    print("REAL EMBEDDING RECORDING")
    print(RULE)
    print(f"admitted evidence spans   {span_count}")
    print(f"benchmark query texts     {query_count}")
    print(f"distinct texts to embed   {len(texts)}")
    print(f"provider                  {settings.embedding_provider}")
    print(f"model                     {settings.model_embedding or '(unset)'}")
    print(f"base url                  {settings.embedding_base_url or '(unset)'}")
    print(f"dimensions                {settings.embedding_dimensions or '(unset)'}")
    print(f"expects unit vectors      {settings.embedding_expects_unit_vectors}")
    print(f"api key                   {'present' if settings.embedding_api_key else 'ABSENT'}")
    print(f"fixture                   {settings.embedding_fixture_path}")

    if problems:
        print()
        print("PREFLIGHT FAILED - no call was made:")
        for problem in problems:
            print(f"  ! {problem}")
        print(RULE)
        return 2

    if dry_run:
        print()
        print("DRY RUN - no call was made. Digests of the texts that would be sent:")
        for text in texts[:5]:
            print(f"  {text_digest(text)[:16]}...  {len(text)} chars")
        print(f"  ... and {max(0, len(texts) - 5)} more")
        print(RULE)
        return 0

    resolved = registry if registry is not None else EmbeddingRegistry(settings)
    embedder: EmbeddingPort = resolved.get(settings.embedding_provider)

    print()
    print(f"embedding {len(texts)} texts through {embedder.name} ...")
    started = dt.datetime.now(tz=dt.UTC)
    try:
        vectors = await embedder.embed(texts)
    except ProviderError as exc:
        # The adapter already translated the vendor failure. Print it and stop: a partial
        # recording is worse than none, because it looks like a complete one.
        print(f"  ! the live call failed: {exc}", file=sys.stderr)
        print(RULE)
        return 1
    finally:
        await resolved.aclose()

    elapsed = (dt.datetime.now(tz=dt.UTC) - started).total_seconds()
    print(f"  returned {len(vectors)} vectors in {elapsed:.1f}s")

    fixture = _build_fixture(
        settings=settings,
        embedder=embedder,
        texts=texts,
        vectors=vectors,
        recorded_at=started,
    )

    path = settings.embedding_fixture_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fixture.model_dump_json(indent=2), encoding="utf-8")

    print()
    print(f"wrote {len(fixture.vectors)} vectors to {path}")
    print("the fixture records provider, model, dimensions, adapter version, the recording")
    print("date and one SHA-256 per text. It contains no credential.")
    print()
    print("next: set CCE_EMBEDDING_PROVIDER=replay, then")
    print("      python -m scripts.retrieval_benchmark --strategy vector")
    print(RULE)
    return 0


def _build_fixture(
    *,
    settings: Settings,
    embedder: EmbeddingPort,
    texts: list[str],
    vectors: list[list[float]],
    recorded_at: dt.datetime,
) -> RecordedEmbeddings:
    """Validate the provider's answer a second time, then key it by text digest.

    The adapter already validated width, finiteness and normalization per batch. This
    re-runs the same checks over the assembled result and additionally proves **coverage**:
    every requested text has a distinct key, so a provider that returned two identical
    vectors for different texts, or dropped one, cannot produce a fixture.
    """
    dimensions = settings.embedding_dimensions
    if dimensions is None:  # pragma: no cover - preflight rejects this
        raise PreflightError("CCE_EMBEDDING_DIMENSIONS is unset")

    # The adapter's own provider identifier where it has one, so the fixture records
    # `openai-compatible` rather than a nested adapter name that would compose badly into
    # `ReplayEmbedder.name` on the way back out.
    adapter_metadata = getattr(embedder, "metadata", None)
    provider = getattr(adapter_metadata, "provider", None) or embedder.name

    metadata = RecordedEmbeddings(
        provider=provider,
        model=settings.model_embedding or "",
        dimensions=dimensions,
        adapter_version=EMBEDDING_ADAPTER_VERSION,
        expects_unit_vectors=settings.embedding_expects_unit_vectors,
        recorded_at=recorded_at,
        recorded_texts=len(texts),
    )
    checked = validate_vectors(
        vectors,
        expected_count=len(texts),
        metadata=_validation_metadata(metadata),
    )

    keyed = {text_digest(text): tuple(vector) for text, vector in zip(texts, checked, strict=True)}
    if len(keyed) != len(texts):
        raise PreflightError(
            f"{len(texts)} texts collapsed onto {len(keyed)} digests; a text was requested "
            "twice and the deduplication above should have prevented it"
        )

    return metadata.model_copy(update={"vectors": keyed})


def _validation_metadata(fixture: RecordedEmbeddings) -> object:
    from app.infrastructure.embeddings.base import EmbeddingMetadata

    return EmbeddingMetadata(
        provider=fixture.provider,
        model=fixture.model,
        dimensions=fixture.dimensions,
        expects_unit_vectors=fixture.expects_unit_vectors,
        adapter_version=fixture.adapter_version,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether configuration is ready for a live call, then stop",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show exactly what would be embedded without calling a provider",
    )
    args = parser.parse_args()

    settings = get_settings()
    try:
        return asyncio.run(record(settings=settings, dry_run=args.check or args.dry_run))
    except (PreflightError, ProviderUnavailable) as exc:
        print(f"cannot record: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
