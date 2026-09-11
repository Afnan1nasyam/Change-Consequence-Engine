"""FastAPI application factory.

Phase 0 exposes health and configuration diagnostics only. Analysis, ingestion and source
routes arrive in later phases (``docs/DEVELOPMENT_PLAN.md``).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.api.deps import get_provider_registry
from app.api.routes import health
from app.config.settings import Settings, get_settings
from app.infrastructure.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    logger.info(
        "starting",
        extra={
            "app": settings.app_name,
            "environment": settings.environment,
            "enabled_providers": list(settings.llm_enabled_providers),
            "default_provider": settings.llm_default_provider,
        },
    )
    try:
        yield
    finally:
        # Providers hold HTTP clients; close them rather than leaking sockets on reload.
        await get_provider_registry().aclose()
        logger.info("stopped", extra={"app": settings.app_name})


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, fmt=resolved.log_format)

    app = FastAPI(
        title="Change Consequence Engine",
        version="0.1.0",
        summary="Multi-agent evidence analysis for latent consequences of organizational change",
        lifespan=lifespan,
        debug=resolved.debug,
    )

    @app.middleware("http")
    async def correlation_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a correlation ID to every request.

        Required by ``docs/ARCHITECTURE.md`` 13: an analysis must be traceable end to end,
        so the ID is established at the edge and reused, never regenerated downstream.
        """
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = correlation_id
        return response

    app.include_router(health.router)
    return app


app = create_app()
