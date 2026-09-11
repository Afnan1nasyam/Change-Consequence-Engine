"""Secret redaction for text leaving the trust boundary.

``docs/SECURITY.md`` 5 and 6 require that credentials are not logged and not shipped to a
provider. The subtlety is *where* to redact.

**Not in the stored canonical text.** Span verification compares a citation against stored
text; redacting it would break every offset and, worse, would make a verified span
unverifiable later. The stored artifact stays byte-faithful and is protected by access
control instead.

**Yes in anything crossing a boundary** - the prompt payload sent to a provider, and log
output. Redaction is therefore applied at the edge, and the spans it found are recorded on
the chunk's metadata so a reviewer can see that a chunk contained a secret without the
secret being reproduced to show them.

The pattern set is deliberately small and high-precision. A greedy matcher that redacted
"10,000 USD" as a credential would silently destroy the corpus's real bindings, which is a
worse outcome than missing an exotic token format.
"""

from __future__ import annotations

import re
from typing import NamedTuple

MASK = "[REDACTED:{kind}]"


class SecretKind(NamedTuple):
    name: str
    pattern: re.Pattern[str]


#: Ordered; earlier patterns win an overlap.
PATTERNS: tuple[SecretKind, ...] = (
    SecretKind(
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    SecretKind("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    SecretKind(
        "jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
    ),
    SecretKind("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    SecretKind("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    SecretKind("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    # Keyword-anchored: only a value introduced by a credential-ish key is redacted, which
    # is what keeps ordinary numbers and prose out of the matcher.
    SecretKind(
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|client[_-]?secret|"
            r"private[_-]?key)\b\s*[:=]\s*[\"']?([^\s\"',;]{6,})[\"']?"
        ),
    ),
    SecretKind("connection_string_password", re.compile(r"(?i)://[^\s:/@]+:([^\s:/@]{4,})@")),
)


class FoundSecret(NamedTuple):
    kind: str
    char_start: int
    char_end: int


class RedactionResult(NamedTuple):
    text: str
    secrets: tuple[FoundSecret, ...]

    @property
    def found_any(self) -> bool:
        return bool(self.secrets)

    @property
    def summary(self) -> str:
        """A loggable description that names kinds and counts but no values."""
        if not self.secrets:
            return "none"
        counts: dict[str, int] = {}
        for secret in self.secrets:
            counts[secret.kind] = counts.get(secret.kind, 0) + 1
        return ", ".join(f"{kind}x{count}" for kind, count in sorted(counts.items()))


def find_secrets(text: str) -> tuple[FoundSecret, ...]:
    """Locate credential-shaped spans. Non-overlapping, earliest-and-longest first."""
    found: list[FoundSecret] = []
    claimed: list[tuple[int, int]] = []

    for kind in PATTERNS:
        for match in kind.pattern.finditer(text):
            # For keyword-anchored patterns, redact the captured value only, so the key
            # name survives and the document stays readable.
            start, end = match.span(1) if match.re.groups and match.group(1) else match.span(0)
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            found.append(FoundSecret(kind.name, start, end))

    return tuple(sorted(found, key=lambda s: s.char_start))


def redact(text: str) -> RedactionResult:
    """Replace credential-shaped spans with a typed mask.

    Length is not preserved: a mask that padded to the original length would leak the
    secret's length, and nothing downstream of redaction uses these offsets for citation.
    """
    secrets = find_secrets(text)
    if not secrets:
        return RedactionResult(text, ())

    out: list[str] = []
    cursor = 0
    for secret in secrets:
        out.append(text[cursor : secret.char_start])
        out.append(MASK.format(kind=secret.kind))
        cursor = secret.char_end
    out.append(text[cursor:])
    return RedactionResult("".join(out), secrets)
