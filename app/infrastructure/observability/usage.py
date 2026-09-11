"""Usage and cost accounting.

Phase 0 keeps the ledger in memory and mirrors every entry to the log. Persistence
(``model_calls``, ``token_usage`` in ``docs/ARCHITECTURE.md`` 9) arrives with the database
layer; the port does not change when it does.

Cost is reported as tokens, not currency. Per-token prices are vendor inventory that
changes, and a stale hard-coded price is a misleading number in a report - worse than no
number. Currency conversion belongs in configuration, later.
"""

from __future__ import annotations

from collections import defaultdict

from app.domain.interfaces.llm import UsageRecord
from app.infrastructure.observability.logging import get_logger

logger = get_logger(__name__)


class InMemoryUsageLedger:
    """Implements :class:`~app.domain.interfaces.llm.UsageLedgerPort`."""

    def __init__(self) -> None:
        self.entries: list[UsageRecord] = []

    def record(self, entry: UsageRecord) -> None:
        self.entries.append(entry)
        logger.info(
            "model call",
            extra={
                "agent": entry.agent,
                "tier": entry.tier,
                "provider": entry.provider,
                "model": entry.model,
                "input_tokens": entry.usage.input_tokens,
                "output_tokens": entry.usage.output_tokens,
                "latency_ms": entry.latency_ms,
                "attempts": entry.attempts,
                "status": entry.status,
                "correlation_id": entry.correlation_id,
            },
        )

    # -- aggregates -------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return sum(e.usage.total_tokens for e in self.entries)

    @property
    def call_count(self) -> int:
        return len(self.entries)

    def tokens_by_agent(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            totals[entry.agent] += entry.usage.total_tokens
        return dict(totals)

    def tokens_by_provider(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            totals[entry.provider] += entry.usage.total_tokens
        return dict(totals)

    def failures(self) -> list[UsageRecord]:
        return [e for e in self.entries if e.status != "ok"]
