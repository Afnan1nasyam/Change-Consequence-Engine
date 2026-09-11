"""The permission scope: what it grants, and the determinism of its hash.

Two properties carry the weight. **Any single matching label grants read**, and a document
with no labels is unrestricted - that is the access-control model, decided in
``app/domain/value_objects/permissions.py`` rather than left implicit. And the hash is a
function of the *effective* scope, so two spellings of the same access collide and two
different accesses never do; an analysis is keyed on that hash, so a collision would serve
one requester's findings to another (``ARCHITECTURE.md`` 14.1).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.value_objects.permissions import (
    MAX_GRANTS,
    PERMISSION_MODEL_VERSION,
    PermissionScope,
    normalize_label,
    normalize_labels,
)


class TestNormalization:
    def test_labels_are_case_folded_and_stripped(self) -> None:
        assert normalize_label("  HR-Confidential ") == "hr-confidential"

    def test_labels_are_deduplicated_and_sorted(self) -> None:
        assert normalize_labels(["b", "A", "a", " B "]) == ("a", "b")

    def test_blank_labels_are_dropped(self) -> None:
        assert normalize_labels(["", "   ", "payments"]) == ("payments",)

    def test_a_scope_stores_its_grants_canonically(self) -> None:
        assert PermissionScope.of("Payments", "privacy", "payments").grants == (
            "payments",
            "privacy",
        )


class TestPermits:
    def test_an_unlabelled_document_is_readable_by_any_scope(self) -> None:
        """The one fail-open edge, and it is deliberate: an empty label set restricts nothing."""
        assert PermissionScope().permits(())
        assert PermissionScope.of("payments").permits(())

    def test_a_matching_grant_reads_a_labelled_document(self) -> None:
        assert PermissionScope.of("payments").permits(("payments",))

    def test_any_single_matching_label_is_enough(self) -> None:
        """The labels are an ACL, not a clearance requirement: they are read disjunctively."""
        assert PermissionScope.of("privacy").permits(("payments", "privacy"))

    def test_a_scope_with_no_grants_cannot_read_a_labelled_document(self) -> None:
        assert not PermissionScope().permits(("payments",))

    def test_an_unrelated_grant_does_not_read_a_labelled_document(self) -> None:
        assert not PermissionScope.of("facilities").permits(("payments",))

    def test_matching_is_case_insensitive_on_both_sides(self) -> None:
        """An asymmetric comparison would make access depend on where the shift key landed."""
        assert PermissionScope.of("hr-confidential").permits(("HR-Confidential",))
        assert PermissionScope.of("HR-CONFIDENTIAL").permits(("hr-confidential",))

    def test_a_near_miss_label_does_not_match(self) -> None:
        assert not PermissionScope.of("payment").permits(("payments",))

    def test_there_is_no_wildcard_grant(self) -> None:
        """A bypass that exists is a bypass that gets used."""
        for pretender in ("*", "all", "admin", "root", "%"):
            assert not PermissionScope.of(pretender).permits(("payments",))


class TestScopeHash:
    def test_the_hash_is_a_sha256_digest(self) -> None:
        digest = PermissionScope.of("payments").scope_hash
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_the_same_scope_hashes_the_same_across_instances(self) -> None:
        assert (
            PermissionScope.of("payments").scope_hash == PermissionScope.of("payments").scope_hash
        )

    @pytest.mark.parametrize(
        "grants",
        [("b", "a"), ("a", "b"), ("a", "b", "a"), ("A", " b "), (" B ", "a", "")],
    )
    def test_spelling_does_not_change_the_hash(self, grants: tuple[str, ...]) -> None:
        """Deterministic for the *effective* scope: order, case, padding and repeats are noise."""
        assert PermissionScope(grants=grants).scope_hash == PermissionScope.of("a", "b").scope_hash

    def test_different_access_produces_a_different_hash(self) -> None:
        assert PermissionScope.of("a").scope_hash != PermissionScope.of("b").scope_hash
        assert PermissionScope.of("a").scope_hash != PermissionScope.of("a", "b").scope_hash

    def test_the_empty_scope_has_its_own_stable_hash(self) -> None:
        assert PermissionScope().scope_hash == PermissionScope(grants=()).scope_hash
        assert PermissionScope().scope_hash != PermissionScope.of("a").scope_hash

    def test_grants_cannot_be_concatenated_into_another_scopes_hash(self) -> None:
        """The preimage is newline-joined and a grant may not contain a newline, so
        ``{"a\\nb"}`` cannot forge the canonical form of ``{"a", "b"}``."""
        with pytest.raises(ValidationError):
            PermissionScope(grants=("a" + chr(10) + "b",))

    def test_the_model_version_is_in_the_preimage(self) -> None:
        """A change to the matching rule must change every hash, or a stored scope would
        keep claiming an access decision the code no longer makes."""
        assert PERMISSION_MODEL_VERSION
        import hashlib

        expected = hashlib.sha256(f"{PERMISSION_MODEL_VERSION}\na".encode()).hexdigest()
        assert PermissionScope.of("a").scope_hash == expected


class TestValidation:
    def test_a_bare_string_is_rejected(self) -> None:
        """``grants="payments"`` would iterate into single characters: a silent privilege change."""
        with pytest.raises(ValidationError, match="not a single string"):
            PermissionScope(grants="payments")

    def test_too_many_grants_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PermissionScope(grants=tuple(f"g{i}" for i in range(MAX_GRANTS + 1)))

    def test_an_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PermissionScope(grants=(), tenant_id="tn_1")  # type: ignore[call-arg]

    def test_a_scope_is_frozen(self) -> None:
        scope = PermissionScope.of("payments")
        with pytest.raises(ValidationError):
            scope.grants = ("other",)  # type: ignore[misc]

    def test_an_overlong_grant_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PermissionScope.of("g" * 65)


def test_a_scope_renders_without_exposing_a_hash() -> None:
    """``__str__`` is for a log line, so it names the grants the requester already knows."""
    assert str(PermissionScope()) == "scope[unlabelled-only]"
    assert str(PermissionScope.of("payments")) == "scope[payments]"
