"""Logging foundation, with emphasis on redaction.

``docs/SECURITY.md`` forbids logging secrets and raw sensitive content. Redaction is
tested rather than trusted, because a leak here is silent.
"""

from __future__ import annotations

import json
import logging

from app.infrastructure.observability.logging import (
    REDACTED,
    JsonFormatter,
    configure_logging,
    get_logger,
)


def format_record(**extra: object) -> dict[str, object]:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event happened",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    payload: dict[str, object] = json.loads(JsonFormatter().format(record))
    return payload


def test_emits_structured_json() -> None:
    payload = format_record()
    assert payload["message"] == "event happened"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert "ts" in payload


def test_passes_through_extra_fields() -> None:
    payload = format_record(agent="policy-analyst", input_tokens=42)
    assert payload["agent"] == "policy-analyst"
    assert payload["input_tokens"] == 42


def test_token_count_metrics_are_not_redacted() -> None:
    """A broad "token" rule would eat the numbers cost accounting depends on.

    Regression guard: usage metrics must survive redaction while credentials do not.
    """
    payload = format_record(
        input_tokens=100,
        output_tokens=250,
        total_tokens=350,
        tokens_spent=1000,
        max_output_tokens=4096,
        prompt_tokens=12,
        completion_tokens=13,
        session_token="must-not-appear",
        access_token="also-must-not-appear",
    )
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 250
    assert payload["total_tokens"] == 350
    assert payload["tokens_spent"] == 1000
    assert payload["max_output_tokens"] == 4096
    assert payload["prompt_tokens"] == 12
    assert payload["completion_tokens"] == 13

    assert payload["session_token"] == REDACTED
    assert payload["access_token"] == REDACTED
    assert "must-not-appear" not in json.dumps(payload)


def test_redacts_sensitive_top_level_keys() -> None:
    payload = format_record(
        groq_api_key="sk-should-never-appear",
        authorization="Bearer nope",
        password="hunter2",
        session_token="abc",
    )
    for key in ("groq_api_key", "authorization", "password", "session_token"):
        assert payload[key] == REDACTED

    serialized = json.dumps(payload)
    assert "sk-should-never-appear" not in serialized
    assert "hunter2" not in serialized


def test_redacts_sensitive_keys_nested_in_structures() -> None:
    payload = format_record(
        context={"provider": "groq", "api_key": "sk-nested", "nested": {"secret": "s3cr3t"}}
    )
    context = payload["context"]
    assert isinstance(context, dict)
    assert context["provider"] == "groq"
    assert context["api_key"] == REDACTED
    nested = context["nested"]
    assert isinstance(nested, dict)
    assert nested["secret"] == REDACTED
    assert "sk-nested" not in json.dumps(payload)


def test_redacts_inside_lists() -> None:
    payload = format_record(items=[{"token": "leak-me"}, {"safe": "ok"}])
    assert "leak-me" not in json.dumps(payload)


def test_truncates_deeply_nested_structures() -> None:
    deep: dict[str, object] = {"level": 0}
    cursor = deep
    for i in range(1, 12):
        nxt: dict[str, object] = {"level": i}
        cursor["child"] = nxt
        cursor = nxt
    assert "[truncated]" in json.dumps(format_record(deep=deep))


def test_includes_exception_text() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="t",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in str(payload["exception"])


def test_configure_logging_is_idempotent() -> None:
    configure_logging(level="INFO", fmt="json")
    first = len(logging.getLogger().handlers)
    configure_logging(level="INFO", fmt="json")
    assert len(logging.getLogger().handlers) == first == 1


def test_console_format_is_available() -> None:
    configure_logging(level="DEBUG", fmt="console")
    handler = logging.getLogger().handlers[0]
    assert not isinstance(handler.formatter, JsonFormatter)
    # Restore JSON for the rest of the suite.
    configure_logging(level="INFO", fmt="json")


def test_get_logger_returns_a_named_logger() -> None:
    assert get_logger("app.test").name == "app.test"
