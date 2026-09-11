"""Assumption extraction: the one LLM call in the ingestion path.

Depends on :class:`~app.domain.interfaces.llm.AgentCompletionPort` and nothing else. It
names an *agent* (``assumption-extractor``); tier, provider and model are resolved below
this line from configuration, which is why a provider-diversity experiment is a config
change and why no document can influence which model reads it.

Cost shape (``CLAUDE.md``, cost discipline): the ``fast`` tier is mandatory here, not an
optimization. Extraction runs once per source version over the whole corpus, so this is the
only stage whose cost is proportional to corpus size rather than to analysis count. It is
also amortized - the use case skips the call entirely when a version is already extracted.

Failure behaviour is deliberately lossy in one direction only: a chunk whose output will not
validate is **dropped with a recorded reason**, never partially coerced into proposals. A
half-parsed proposal would enter the admission gate looking like a real one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NamedTuple

from pydantic import ValidationError

from app.agents.runtime.agent_loader import AgentDefinition
from app.assumptions.prompting import PROMPT_TEMPLATE_VERSION, build_extraction_prompt
from app.assumptions.proposals import AssumptionProposal, ExtractionBatch
from app.assumptions.registry import AssumptionKeyRegistry
from app.domain.interfaces.llm import (
    AgentCompletionPort,
    ProviderError,
    StructuredOutputError,
)
from app.domain.value_objects.provenance import ModelProvenance
from app.ingestion.chunking import ChunkWithContext

logger = logging.getLogger(__name__)

EXTRACTOR_AGENT = "assumption-extractor"

#: Bumped when extraction code changes shape. Combined with the prompt hash to form the
#: per-record `extractor_version`, so re-extracting under a new prompt creates new records
#: rather than overwriting ones a past analysis cited.
EXTRACTOR_CODE_VERSION = "extractor-1"


def extractor_version(definition: AgentDefinition) -> str:
    """A version label that changes whenever the prompt or the code changes."""
    return f"{EXTRACTOR_CODE_VERSION}+{PROMPT_TEMPLATE_VERSION}+{definition.prompt_sha256[:8]}"


class ChunkExtraction(NamedTuple):
    """What extraction produced for one chunk, including what it refused to produce."""

    chunk: ChunkWithContext
    proposals: tuple[AssumptionProposal, ...]
    observed_instructions: tuple[str, ...]
    provenance: ModelProvenance
    #: Set when the call or the schema validation failed. The chunk yielded nothing and the
    #: reason is recorded rather than retried indefinitely.
    failure: str | None = None
    secrets_redacted: str = "none"
    attempted_delimiter_breakout: bool = False


class ExtractionOutcome(NamedTuple):
    extractions: tuple[ChunkExtraction, ...]
    calls_made: int

    @property
    def proposals(self) -> tuple[AssumptionProposal, ...]:
        return tuple(p for e in self.extractions for p in e.proposals)

    @property
    def failures(self) -> tuple[ChunkExtraction, ...]:
        return tuple(e for e in self.extractions if e.failure is not None)

    @property
    def observed_instructions(self) -> tuple[str, ...]:
        return tuple(i for e in self.extractions for i in e.observed_instructions)


class AssumptionExtractor:
    """Proposes bindings for chunks. Admits nothing."""

    def __init__(
        self,
        *,
        completions: AgentCompletionPort,
        definition: AgentDefinition,
        registry: AssumptionKeyRegistry,
    ) -> None:
        if definition.name != EXTRACTOR_AGENT:
            raise ValueError(
                f"expected the {EXTRACTOR_AGENT!r} definition, got {definition.name!r}"
            )
        self._completions = completions
        self._definition = definition
        self._registry = registry

    @property
    def version(self) -> str:
        return extractor_version(self._definition)

    def _provenance(self) -> ModelProvenance:
        """Attribution for anything this extractor proposes.

        Resolved without a model call, so a gap or a quarantine record carries full
        provenance even when the call itself failed.
        """
        binding = self._completions.resolve(EXTRACTOR_AGENT)
        return ModelProvenance(
            agent=EXTRACTOR_AGENT,
            prompt_version=self._definition.version,
            prompt_sha256=self._definition.prompt_sha256,
            provider=binding.provider,
            model=binding.model,
            tier=binding.tier,
        )

    async def extract_chunk(
        self, chunk: ChunkWithContext, *, correlation_id: str | None = None
    ) -> ChunkExtraction:
        prompt = build_extraction_prompt(
            system_prompt=self._definition.prompt,
            chunk=chunk,
            registry_keys=self._registry.keys,
            registry_version=self._registry.version,
        )
        provenance = self._provenance()

        if prompt.attempted_delimiter_breakout:
            logger.warning(
                "chunk attempted to impersonate the untrusted-content delimiter",
                extra={
                    "chunk_id": chunk.chunk.chunk_id,
                    "source_version_id": chunk.chunk.source_version_id,
                },
            )
        if prompt.redaction.found_any:
            logger.info(
                "redacted credential-shaped spans before dispatch",
                extra={
                    "chunk_id": chunk.chunk.chunk_id,
                    "secrets": prompt.redaction.summary,
                },
            )

        base = ChunkExtraction(
            chunk=chunk,
            proposals=(),
            observed_instructions=(),
            provenance=provenance,
            secrets_redacted=prompt.redaction.summary,
            attempted_delimiter_breakout=prompt.attempted_delimiter_breakout,
        )

        try:
            payload, _response = await self._completions.complete_json(
                EXTRACTOR_AGENT,
                prompt.messages,
                json_schema=ExtractionBatch.json_schema(),
                correlation_id=correlation_id,
            )
        except (StructuredOutputError, ProviderError) as exc:
            logger.warning(
                "extraction call failed; chunk yields no proposals",
                extra={"chunk_id": chunk.chunk.chunk_id, "error": str(exc)},
            )
            return base._replace(failure=f"{type(exc).__name__}: {exc}")

        try:
            batch = ExtractionBatch.model_validate(payload)
        except ValidationError as exc:
            # Includes the case that matters most: an extractor emitting `confidence`.
            # extra="forbid" turns that into a rejection instead of a dropped key.
            logger.warning(
                "extraction output violated the proposal schema; discarded",
                extra={"chunk_id": chunk.chunk.chunk_id, "errors": exc.error_count()},
            )
            return base._replace(failure=f"schema: {exc.error_count()} error(s)")

        # A proposal naming a chunk other than the one it was shown cannot be verified
        # against that chunk's window, so it is dropped here rather than at the gate.
        kept = tuple(p for p in batch.bindings if p.chunk_ordinal == chunk.chunk.ordinal)
        if len(kept) != len(batch.bindings):
            logger.warning(
                "dropped proposals citing a chunk the extractor was not shown",
                extra={
                    "chunk_id": chunk.chunk.chunk_id,
                    "dropped": len(batch.bindings) - len(kept),
                },
            )

        return base._replace(proposals=kept, observed_instructions=batch.observed_instructions)

    async def extract(
        self,
        chunks: Sequence[ChunkWithContext],
        *,
        correlation_id: str | None = None,
    ) -> ExtractionOutcome:
        """Extract over every chunk of one source version."""
        results: list[ChunkExtraction] = []
        for chunk in chunks:
            results.append(await self.extract_chunk(chunk, correlation_id=correlation_id))
        return ExtractionOutcome(tuple(results), calls_made=len(results))
