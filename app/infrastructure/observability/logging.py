"""Logging foundation.

JSON by default so that correlation IDs and token counts are queryable rather than
grep-able. Standard library only: a log formatter is not worth a dependency.

Two rules from ``docs/SECURITY.md`` are enforced here rather than left to discipline:
raw source content is never logged, and known-sensitive keys are redacted even if a caller
passes them by mistake.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

REDACTED = "[redacted]"

#: Substring match, lowercased. Deliberately broad: a false redaction is cheap.
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "password",
    "token",
    "credential",
    "cookie",
)

#: Exempt from the broad match above.
#:
#: "token" appears in every token-count metric, so the broad rule would redact the usage
#: numbers that cost accounting depends on. Broad-by-default with a small explicit
#: allowlist is the right trade: adding a metric here is a deliberate, reviewable act,
#: whereas widening the sensitive list is not.
SAFE_KEY_NAMES = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tokens_spent",
        "tokens_remaining",
        "max_output_tokens",
        "token_budget",
        "token_count",
        "prompt_tokens",
        "completion_tokens",
    }
)

#: Attributes the stdlib puts on every record; anything else came from `extra=`.
_STANDARD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    if lowered in SAFE_KEY_NAMES:
        return False
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _scrub(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_sensitive(str(k)) else _scrub(v, depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(v, depth + 1) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = REDACTED if _is_sensitive(key) else _scrub(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(*, level: str = "INFO", fmt: str = "json") -> None:
    """Install the root handler. Idempotent."""
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own handlers; let records reach ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
