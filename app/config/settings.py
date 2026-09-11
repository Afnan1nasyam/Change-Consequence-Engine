"""Application configuration.

Every model identifier and credential is sourced from the environment. Model names are
deliberately **not** defaulted: they are vendor inventory that changes, and a stale
hard-coded identifier is an outage. An unset tier fails loudly at resolution time with a
message naming the variable to set.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- application ------------------------------------------------------
    app_name: str = "change-consequence-engine"
    environment: Literal["local", "ci", "staging", "production"] = "local"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # -- persistence ------------------------------------------------------
    # No default, and deliberately so. A default carrying `cce:cce` is a credential in
    # source; a default pointing at a host that is not there turns a configuration mistake
    # into a connection error. `app.infrastructure.database.session.database_url()` raises
    # `DatabaseNotConfigured` naming this variable instead.
    database_url: str | None = None
    redis_url: str = "redis://localhost:6379/0"

    # -- llm gateway ------------------------------------------------------
    llm_default_provider: str = "groq"

    # Providers are enabled explicitly, never inferred from which keys happen to be
    # present (docs/SECURITY.md, multi-provider section). Widening this list widens the
    # data-egress surface.
    llm_enabled_providers: tuple[str, ...] = ("groq", "fake")

    agents_config_path: Path = REPO_ROOT / "app" / "config" / "agents.yaml"
    # The closed assumption-key registry. Beside `agents_config_path` because it is the same
    # kind of thing: shipped configuration a composition root has to be able to find.
    assumption_keys_path: Path = REPO_ROOT / "app" / "config" / "assumption_keys.yaml"

    # Tier -> model identifier. No defaults on purpose.
    model_reasoning: str | None = None
    model_standard: str | None = None
    model_fast: str | None = None

    # -- budgets and retries ---------------------------------------------
    llm_max_output_tokens: int = Field(default=2048, gt=0)
    llm_analysis_token_budget: int = Field(default=250_000, gt=0)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1)
    llm_backoff_base_s: float = Field(default=0.5, gt=0)
    llm_backoff_max_s: float = Field(default=8.0, gt=0)

    # -- providers --------------------------------------------------------
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Optional. Absence must never be an error.
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    mistral_api_key: SecretStr | None = None
    mistral_base_url: str = "https://api.mistral.ai/v1"

    ollama_base_url: str = "http://localhost:11434"

    # -- retrieval --------------------------------------------------------
    # Which strategy the orchestration layer selects when a request does not name one.
    # `lexical` is the default because it is the strongest measured strategy and the only
    # one that needs no embedder (docs/EVALUATION.md); `vector` and `hybrid` are selectable
    # and both currently measure worse. Validated at resolution time - an unknown name
    # raises UnknownRetrievalStrategy rather than falling back to anything.
    retrieval_strategy: str = "lexical"

    # -- embeddings -------------------------------------------------------
    # Which embedding adapter retrieval uses. The default needs no credential and no
    # network (ARCHITECTURE.md 11.4), so vector retrieval is developable offline.
    embedding_provider: str = "hashing"

    # Enabled explicitly, exactly as with LLM providers, and for the same reason: a remote
    # embedder sends corpus text to a third party, so it is enabled by an operator's
    # decision and never because a key happens to be present (docs/SECURITY.md).
    embedding_enabled_providers: tuple[str, ...] = ("hashing", "replay")

    # The model identifier, from configuration only. No default: a vector index has to name
    # the model that produced it, and a defaulted model makes a recall figure unreproducible.
    model_embedding: str | None = None

    # From the environment only, never a literal and never inferred from another provider's
    # credential.
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None

    # The width the configured model is expected to return, validated per vector. Nothing
    # in a response proves the numbers came from the model that was asked for, so this is
    # the guard against a mistyped identifier producing a different index in silence.
    embedding_dimensions: int | None = None

    # True only for providers documented to return unit vectors. Checked when set, because
    # a ranking measured under one normalization convention is not comparable with one
    # measured under the other.
    embedding_expects_unit_vectors: bool = False

    # Gemini accepts a taskType. For retrieval Google recommends the *asymmetric* pair
    # RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY, which EmbeddingPort cannot express - it takes an
    # undifferentiated list of texts. So this is one symmetric value, which is the honest
    # thing the current interface can do (docs/STATUS.md 44).
    embedding_task_type: str = "SEMANTIC_SIMILARITY"

    embedding_max_batch_size: int = Field(default=64, ge=1, le=2048)
    embedding_timeout_s: float = Field(default=60.0, gt=0)

    #: Committed provider vectors for the replay embedder, when one has been recorded.
    embedding_fixture_path: Path = REPO_ROOT / "data" / "fixtures" / "embeddings.json"

    def model_for_tier(self, tier: str) -> str | None:
        return {
            "reasoning": self.model_reasoning,
            "standard": self.model_standard,
            "fast": self.model_fast,
        }.get(tier)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached; call ``get_settings.cache_clear()`` in tests."""
    return Settings()
