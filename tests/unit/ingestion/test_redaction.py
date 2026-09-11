"""Secret redaction at the trust boundary.

Two failure modes, and the tests weigh them differently on purpose:

* **Missing a secret** ships a credential to a provider. Bad.
* **Redacting an ordinary number** silently destroys a real binding in every future
  analysis of that corpus. Worse, because it is invisible.

So the false-positive tests here are the load-bearing ones.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.ingestion.redaction import find_secrets, redact

ORDINARY = [
    "Refunds of 10,000 USD or more require manager approval.",
    "Personal data is retained for 90 days after closure.",
    "auth.reset_token.ttl_hours: 24",
    "A P1 incident is acknowledged within 15 minutes.",
    "The API returns at most 100 items per page.",
    "Contact secops@example.com for access.",
    "Version 6.1.2 shipped on 2025-04-01.",
    "The password reset flow sends an email.",
    "Our secret sauce is good documentation.",
]


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_corpus_text_is_never_redacted(text: str) -> None:
    """A greedy matcher here would quietly delete the corpus's real assumptions."""
    result = redact(text)
    assert result.text == text
    assert not result.found_any


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("AKIAIOSFODNN7EXAMPLE is the key", "aws_access_key_id"),
        ("ghp_abcdefghijklmnopqrstuvwxyz0123", "github_token"),
        ("xoxb-123456789012-abcdefghijkl", "slack_token"),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345", "bearer_token"),
        ("api_key = sk_test_51H8vB2eZvKYlo2CkQfvT7nQ9pLmXwR4", "credential_assignment"),
        ("password = Sandbox-Winter-2026", "credential_assignment"),
        ("postgres://user:s3cretpw@host/db", "connection_string_password"),
    ],
)
def test_credential_shapes_are_found(text: str, kind: str) -> None:
    secrets = find_secrets(text)
    assert [s.kind for s in secrets] == [kind]


def test_a_jwt_is_found() -> None:
    token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
    )
    assert [s.kind for s in find_secrets(f"token {token} end")] == ["jwt"]


def test_a_private_key_block_is_redacted_whole() -> None:
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...lines...\n-----END RSA PRIVATE KEY-----"
    result = redact(f"before\n{block}\nafter")
    assert "MIIEow" not in result.text
    assert "before" in result.text and "after" in result.text


def test_redaction_replaces_with_a_typed_mask() -> None:
    result = redact("password = Sandbox-Winter-2026")
    assert "Sandbox-Winter-2026" not in result.text
    assert "[REDACTED:credential_assignment]" in result.text
    assert result.text.startswith("password = ")


def test_the_key_name_survives_so_the_document_stays_readable() -> None:
    """The extractor still needs to see that a credential field exists."""
    assert redact("api_key = abcdef123456").text.startswith("api_key = ")


def test_mask_does_not_preserve_length() -> None:
    """Padding to the original length would leak the secret's length."""
    result = redact("password = " + "x" * 64)
    assert len(result.text) < len("password = " + "x" * 64)


def test_the_summary_names_kinds_and_counts_but_no_values() -> None:
    """A log line must be able to say a secret was present without reproducing it."""
    result = redact("password = hunter2000 and api_key = abcdef123456")
    assert "credential_assignment" in result.summary
    assert "hunter2000" not in result.summary
    assert "abcdef123456" not in result.summary


def test_no_secrets_summarizes_as_none() -> None:
    assert redact("nothing here").summary == "none"


def test_overlapping_matches_are_claimed_once() -> None:
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmno"
    spans = find_secrets(text)
    for earlier, later in pairwise(spans):
        assert earlier.char_end <= later.char_start


def test_multiple_secrets_are_all_masked() -> None:
    result = redact("password = one-two-three\nAKIAIOSFODNN7EXAMPLE\n")
    assert "one-two-three" not in result.text
    assert "AKIAIOSFODNN7EXAMPLE" not in result.text
    assert len(result.secrets) == 2


def test_redaction_is_deterministic() -> None:
    text = "password = hunter2000"
    assert redact(text) == redact(text)
