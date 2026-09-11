"""Deterministic in-process provider.

This is the provider the test suite runs on. It requires no credentials and performs no
network I/O, which is what allows CI and local development to be green with an empty
``.env`` - a hard requirement in ``CLAUDE.md``.

It is also the seam for scripted failure: queue ``ProviderError`` instances to exercise
the gateway's retry, budget and repair behaviour without touching a vendor.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Sequence

from app.domain.interfaces.llm import (
    CompletionRequest,
    CompletionResponse,
    ProviderError,
    TokenUsage,
)

Responder = Callable[[CompletionRequest], str]


def echo_digest(request: CompletionRequest) -> str:
    """Default responder: stable JSON derived from the request.

    Deterministic by construction, so a test asserting on output is asserting on the
    request that produced it.
    """
    payload = json.dumps(
        {
            "model": request.model,
            "agent": request.agent,
            "messages": [[m.role.value, m.content] for m in request.messages],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return json.dumps({"echo": digest, "agent": request.agent, "model": request.model})


class FakeProvider:
    """In-memory provider with recorded calls and scriptable behaviour."""

    provider_name = "fake"

    def __init__(
        self,
        *,
        responder: Responder | None = None,
        responses: Sequence[str] | None = None,
        failures: Sequence[ProviderError] | None = None,
    ) -> None:
        self._responder = responder or echo_digest
        self._responses: deque[str] = deque(responses or ())
        self._failures: deque[ProviderError] = deque(failures or ())
        self.calls: list[CompletionRequest] = []
        self.closed = False

    @property
    def name(self) -> str:
        return self.provider_name

    def queue_response(self, text: str) -> None:
        self._responses.append(text)

    def queue_failure(self, error: ProviderError) -> None:
        self._failures.append(error)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)

        # Failures are consumed first so a test can script "fail twice, then succeed".
        if self._failures:
            raise self._failures.popleft()

        text = self._responses.popleft() if self._responses else self._responder(request)

        return CompletionResponse(
            text=text,
            provider=self.provider_name,
            model=request.model,
            usage=TokenUsage(
                input_tokens=sum(len(m.content) for m in request.messages) // 4,
                output_tokens=max(1, len(text) // 4),
            ),
            finish_reason="stop",
            latency_ms=0,
        )

    async def aclose(self) -> None:
        self.closed = True
