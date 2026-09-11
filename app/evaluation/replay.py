"""Replay extractor: recorded proposals instead of a live model call.

Why this exists in the product rather than only in tests. ``docs/EVALUATION.md`` needs runs
that differ in *one* variable - a registry version, an alias approval, a chunker change -
and a live extractor makes that impossible, because the model output moves too. Replaying a
recorded set of proposals holds extraction fixed so the deterministic layer is the only
thing under test.

It is also how the ingestion path is exercised with no credentials and no network: the
parser, chunker, registry, admission gate and stores are all real, and only the model call
is substituted.

Recorded proposals carry a **quote, not a chunk ordinal**. The ordinal is resolved here by
finding the quote in the chunked document, which keeps the fixture valid across a chunker
change and mirrors what the real extractor does - the model never supplies offsets.

A recorded quote that cannot be located is reported as an extraction failure rather than
skipped, because a silently-dropped fixture entry would look like a recall problem in the
deterministic layer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from app.assumptions.extraction import ChunkExtraction, ExtractionOutcome
from app.assumptions.proposals import AssumptionProposal
from app.domain.value_objects.provenance import ModelProvenance
from app.ingestion.chunking import ChunkWithContext

REPLAY_VERSION = "replay-1"


class RecordedProposal(BaseModel):
    """One proposal as recorded, without a chunk ordinal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposed_key: str
    scope: str = "global"
    value_quote: str
    value_type: str
    unit: str | None = None
    role: str
    explicitness: str
    quote: str
    effective_from: str | None = None
    effective_to: str | None = None
    ambiguity: str | None = None

    def to_proposal(self, *, chunk_ordinal: int) -> AssumptionProposal:
        return AssumptionProposal.model_validate(
            {**self.model_dump(exclude_none=True), "chunk_ordinal": chunk_ordinal}
        )


class RecordedExtraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bindings: tuple[RecordedProposal, ...] = ()
    observed_instructions: tuple[str, ...] = ()


class ReplayExtractor:
    """Satisfies the ingestion ``ExtractionPort`` from a recorded fixture."""

    def __init__(
        self,
        *,
        recorded: RecordedExtraction,
        provenance: ModelProvenance,
        version_label: str = REPLAY_VERSION,
    ) -> None:
        self._recorded = recorded
        self._provenance = provenance
        self._version = version_label

    @property
    def version(self) -> str:
        return self._version

    async def extract(
        self,
        chunks: Sequence[ChunkWithContext],
        *,
        correlation_id: str | None = None,
    ) -> ExtractionOutcome:
        by_ordinal: dict[int, list[AssumptionProposal]] = {}
        failures: list[str] = []

        for recorded in self._recorded.bindings:
            ordinal = _find_chunk(chunks, recorded.quote)
            if ordinal is None:
                failures.append(f"recorded quote not present in any chunk: {recorded.quote[:60]!r}")
                continue
            try:
                proposal = recorded.to_proposal(chunk_ordinal=ordinal)
            except ValidationError as exc:
                # A recorded entry that violates the proposal schema is exactly what a
                # misbehaving extractor would emit, and it is refused the same way.
                failures.append(f"recorded proposal invalid: {exc.error_count()} error(s)")
                continue
            by_ordinal.setdefault(ordinal, []).append(proposal)

        extractions: list[ChunkExtraction] = []
        for index, chunk in enumerate(chunks):
            extractions.append(
                ChunkExtraction(
                    chunk=chunk,
                    proposals=tuple(by_ordinal.get(chunk.chunk.ordinal, ())),
                    observed_instructions=(
                        self._recorded.observed_instructions if index == 0 else ()
                    ),
                    provenance=self._provenance,
                    failure="; ".join(failures) if failures and index == 0 else None,
                )
            )

        return ExtractionOutcome(tuple(extractions), calls_made=0)


def _find_chunk(chunks: Sequence[ChunkWithContext], quote: str) -> int | None:
    for chunk in chunks:
        if quote in chunk.chunk.text:
            return chunk.chunk.ordinal
    return None


def load_recordings(path: Path) -> dict[str, RecordedExtraction]:
    """Load a fixture of recorded proposals keyed by corpus-relative path."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object keyed by source path")
    return {key: RecordedExtraction.model_validate(value) for key, value in raw.items()}
