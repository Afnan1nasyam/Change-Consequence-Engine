"""HTTP surface of the Phase 0 application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_liveness_returns_ok(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "change-consequence-engine"
    assert body["environment"] == "ci"


def test_readiness_reports_component_checks(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200

    checks = response.json()["checks"]
    assert checks["agent_definitions"]["status"] == "ok"
    assert checks["agent_bindings"]["status"] == "ok"
    assert checks["llm_gateway"]["status"] == "ok"


def test_readiness_does_not_pretend_to_probe_unbuilt_infrastructure(client: TestClient) -> None:
    """Postgres and Redis have no code path yet; claiming to check them would be a lie."""
    checks = client.get("/health/ready").json()["checks"]
    assert checks["database"]["status"] == "not_configured"
    assert checks["redis"]["status"] == "not_configured"


def test_readiness_is_degraded_not_failed_without_credentials(client: TestClient) -> None:
    """The app must run with no provider credentials, so degraded still serves 200."""
    body = client.get("/health/ready").json()
    assert body["status"] == "degraded"


def test_agent_diagnostics_lists_every_agent(client: TestClient) -> None:
    response = client.get("/health/agents")
    assert response.status_code == 200

    views = response.json()
    assert len(views) == 16

    by_agent = {v["agent"]: v for v in views}
    judge = by_agent["consequence-judge"]
    assert judge["tier"] == "reasoning"
    assert judge["resolution"] == "ok"
    assert judge["model_configured"] is True
    assert len(judge["prompt_sha256"]) == 64


def test_agent_diagnostics_never_returns_a_model_name_or_credential(client: TestClient) -> None:
    """It reports *whether* a model is configured, not what it is."""
    for view in client.get("/health/agents").json():
        assert set(view) == {
            "agent",
            "version",
            "prompt_sha256",
            "tier",
            "provider",
            "model_configured",
            "resolution",
            "detail",
        }
        assert "api_key" not in str(view).lower()


def test_correlation_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Correlation-ID": "given-id"})
    assert response.headers["X-Correlation-ID"] == "given-id"


def test_correlation_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health/live")
    generated = response.headers.get("X-Correlation-ID")
    assert generated
    assert len(generated) == 36  # uuid4


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health/live" in paths
    assert "/health/ready" in paths
    assert "/health/agents" in paths
