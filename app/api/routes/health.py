"""Health and configuration-diagnostic endpoints.

Readiness deliberately checks only what Phase 0 actually owns: agent definitions parse,
runtime bindings load, and the LLM gateway can resolve. Postgres and Redis are *not*
probed yet, because no code path uses them - claiming to check them would make readiness a
lie. They join readiness in the phase that introduces the persistence layer.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from app.agents.runtime.agent_loader import AgentDefinition
from app.api.deps import (
    get_agent_bindings,
    get_agent_definitions,
    get_app_settings,
    get_gateway,
)
from app.config.agent_bindings import AgentBindings
from app.config.settings import Settings
from app.domain.interfaces.llm import ProviderError
from app.infrastructure.llm.gateway import LlmGateway

router = APIRouter(tags=["health"])

CheckStatus = Literal["ok", "degraded", "fail", "not_configured"]


class ComponentCheck(BaseModel):
    status: CheckStatus
    detail: str | None = None


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded", "fail"]
    checks: dict[str, ComponentCheck]


class AgentBindingView(BaseModel):
    agent: str
    version: int
    prompt_sha256: str
    tier: str
    provider: str | None = None
    model_configured: bool
    resolution: CheckStatus
    detail: str | None = None


@router.get("/health/live", response_model=LivenessResponse, summary="Liveness probe")
def liveness(settings: Settings = Depends(get_app_settings)) -> LivenessResponse:
    return LivenessResponse(app=settings.app_name, environment=settings.environment)


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
def readiness(
    response: Response,
    definitions: dict[str, AgentDefinition] = Depends(get_agent_definitions),
    bindings: AgentBindings = Depends(get_agent_bindings),
    gateway: LlmGateway = Depends(get_gateway),
) -> ReadinessResponse:
    checks: dict[str, ComponentCheck] = {
        "agent_definitions": ComponentCheck(
            status="ok", detail=f"{len(definitions)} definitions loaded"
        ),
        "agent_bindings": ComponentCheck(
            status="ok", detail=f"{len(bindings.bindings)} bindings, version {bindings.version}"
        ),
    }

    # Resolution is config-only: no model call, no network, no cost.
    resolvable = 0
    problems: list[str] = []
    for name in definitions:
        try:
            gateway.resolve(name)
            resolvable += 1
        except ProviderError as exc:
            problems.append(f"{name}: {exc}")

    if resolvable == len(definitions):
        checks["llm_gateway"] = ComponentCheck(
            status="ok", detail=f"all {resolvable} agents resolve to a provider and model"
        )
    elif resolvable == 0:
        checks["llm_gateway"] = ComponentCheck(
            status="not_configured",
            detail=(
                "no agent resolves to a model. Set CCE_MODEL_REASONING / "
                "CCE_MODEL_STANDARD / CCE_MODEL_FAST. "
                f"First problem: {problems[0] if problems else 'unknown'}"
            ),
        )
    else:
        checks["llm_gateway"] = ComponentCheck(
            status="degraded",
            detail=f"{resolvable}/{len(definitions)} agents resolve. {problems[0]}",
        )

    # Phase 0 has no persistence code path, so these are declared, not probed.
    checks["database"] = ComponentCheck(
        status="not_configured", detail="not probed until the persistence layer exists"
    )
    checks["redis"] = ComponentCheck(
        status="not_configured", detail="not probed until the queue layer exists"
    )

    statuses = {c.status for c in checks.values()}
    if "fail" in statuses:
        overall: Literal["ok", "degraded", "fail"] = "fail"
    elif "degraded" in statuses or "not_configured" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    # Degraded is still serving: the app runs without provider credentials on purpose.
    if overall == "fail":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(status=overall, checks=checks)


@router.get(
    "/health/agents",
    response_model=list[AgentBindingView],
    summary="Agent -> tier -> provider/model resolution, without calling a model",
)
def agent_bindings(
    definitions: dict[str, AgentDefinition] = Depends(get_agent_definitions),
    bindings: AgentBindings = Depends(get_agent_bindings),
    gateway: LlmGateway = Depends(get_gateway),
) -> list[AgentBindingView]:
    """Diagnostic view. Answers "why is this agent not running" without spending a token.

    Never returns a credential; only whether a model identifier is configured.
    """
    views: list[AgentBindingView] = []
    for name, definition in sorted(definitions.items()):
        binding = bindings.for_agent(name)
        view: dict[str, Any] = {
            "agent": name,
            "version": definition.version,
            "prompt_sha256": definition.prompt_sha256,
            "tier": binding.tier.value,
            "provider": binding.provider,
            "model_configured": False,
            "resolution": "not_configured",
            "detail": None,
        }
        try:
            resolved = gateway.resolve(name)
            view.update(
                provider=resolved.provider,
                model_configured=True,
                resolution="ok",
            )
        except ProviderError as exc:
            view.update(resolution="not_configured", detail=str(exc))
        views.append(AgentBindingView.model_validate(view))
    return views
