"""Capability tiers.

Agents request a *tier*, never a model. Configuration binds a tier to a provider and a
model identifier (see ``docs/ARCHITECTURE.md`` 11.1). This is what keeps vendor inventory
out of business logic and makes provider-diversity experiments a config change.
"""

from __future__ import annotations

from enum import StrEnum


class CapabilityTier(StrEnum):
    """Role-declared model capability.

    ``FAST`` matters more than it looks: assumption extraction runs over the entire corpus
    at ingestion time, so binding it to a reasoning-tier model would make ingestion the
    dominant cost of the whole system.
    """

    REASONING = "reasoning"
    STANDARD = "standard"
    FAST = "fast"
