"""The closed AssumptionKeyRegistry and its resolution rules.

Resolution is the third admission check and the one that decides whether a proposal can
ever become Class A. Three properties make it trustworthy:

* **Closed.** An unregistered key has no path into the index. Not a low-confidence path, not
  a quarantine-then-retry path - no path. It becomes a ``KeyRegistryGap``.
* **Deterministic.** Resolution is a dictionary lookup plus an approved-alias lookup. There
  is no clustering, no embedding similarity, and no learned normalization
  (``DOMAIN_CHECKPOINT.md`` 4.1.5 and 14). Canonical form is declared per entry.
* **Gated.** A *proposed* alias resolves to nothing. Only ``status = approved`` participates,
  and the distinction is reported rather than hidden, so a reviewer sees a pending gate
  instead of a mysterious gap.

The registry is loaded from configuration, versioned, and pinned onto every record admitted
under it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from app.domain.entities.registry import AssumptionKeyRegistryEntry, KeyAlias
from app.domain.value_objects.scope import ScopeSelector

DEFAULT_REGISTRY_PATH = Path("app/config/assumption_keys.yaml")


class RegistryError(ValueError):
    """The registry file is malformed.

    Fails at load. A registry that silently loses an entry would silently demote every
    consequence that depended on that key from Class A to a gap.
    """


class ResolutionFailure(StrEnum):
    """Why a key did not resolve. Each maps to a different reviewer action."""

    #: Not in the registry. Becomes a KeyRegistryGap; the action is to consider registering it.
    UNREGISTERED = "unregistered"
    #: Registered but retired. The action is to fix the source, not the registry.
    DEPRECATED = "deprecated"
    #: An alias exists but nobody has approved it. The action is to review the proposal.
    ALIAS_NOT_APPROVED = "alias_not_approved"


class KeyResolution(NamedTuple):
    """The outcome of resolving one proposed key."""

    entry: AssumptionKeyRegistryEntry | None
    via_alias: KeyAlias | None = None
    failure: ResolutionFailure | None = None
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.entry is not None

    @property
    def alias_mediated(self) -> bool:
        """Recorded because alias mediation *lowers* confidence (checkpoint 9.1)."""
        return self.via_alias is not None


class AssumptionKeyRegistry:
    """An immutable, versioned view of the closed vocabulary."""

    def __init__(
        self,
        *,
        version: str,
        entries: Iterable[AssumptionKeyRegistryEntry],
        aliases: Iterable[KeyAlias] = (),
    ) -> None:
        self._version = version
        self._entries: dict[str, AssumptionKeyRegistryEntry] = {}
        for entry in entries:
            if entry.key in self._entries:
                raise RegistryError(f"duplicate registry key {entry.key!r}")
            self._entries[entry.key] = entry

        # Only approved aliases are indexed for resolution. Proposals are kept separately
        # so that "an unapproved alias exists" stays reportable.
        self._approved: dict[str, KeyAlias] = {}
        self._unapproved: dict[str, KeyAlias] = {}
        for alias in aliases:
            target = self._approved if alias.participates_in_class_a else self._unapproved
            target[alias.source_key] = alias

        unknown_targets = sorted(
            a.target_key for a in self._approved.values() if a.target_key not in self._entries
        )
        if unknown_targets:
            raise RegistryError(
                f"approved aliases target unregistered keys: {unknown_targets}. "
                "An alias may only point at a key that exists in this registry version."
            )

    # -- identity ---------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({e.domain for e in self._entries.values()}))

    def entry(self, key: str) -> AssumptionKeyRegistryEntry | None:
        return self._entries.get(key)

    def contains(self, key: str) -> bool:
        return key in self._entries

    # -- resolution -------------------------------------------------------

    def resolve(self, proposed_key: str) -> KeyResolution:
        """Resolve a proposed key to a registry entry, or explain why not."""
        normalized = proposed_key.strip()

        direct = self._entries.get(normalized)
        if direct is not None:
            if not direct.is_active:
                return KeyResolution(
                    None,
                    failure=ResolutionFailure.DEPRECATED,
                    detail=(
                        f"key {normalized!r} was deprecated in {direct.deprecated_in_version!r}"
                    ),
                )
            return KeyResolution(direct)

        approved = self._approved.get(normalized)
        if approved is not None:
            target = self._entries[approved.target_key]
            if not target.is_active:
                return KeyResolution(
                    None,
                    failure=ResolutionFailure.DEPRECATED,
                    detail=f"alias target {target.key!r} is deprecated",
                )
            return KeyResolution(target, via_alias=approved)

        pending = self._unapproved.get(normalized)
        if pending is not None:
            return KeyResolution(
                None,
                failure=ResolutionFailure.ALIAS_NOT_APPROVED,
                detail=(
                    f"alias {normalized!r} -> {pending.target_key!r} is "
                    f"{pending.status.value!r}; only an approved alias may create Class A "
                    "coupling. No confidence threshold approves it."
                ),
            )

        return KeyResolution(
            None,
            failure=ResolutionFailure.UNREGISTERED,
            detail=f"{normalized!r} is not in registry {self._version!r}",
        )

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, path: Path, *, aliases: Iterable[KeyAlias] = ()) -> AssumptionKeyRegistry:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RegistryError(f"registry file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise RegistryError(f"invalid registry YAML in {path}: {exc}") from exc
        return cls.from_mapping(raw, source=str(path), aliases=aliases)

    @classmethod
    def from_mapping(
        cls,
        raw: Any,
        *,
        source: str = "<memory>",
        aliases: Iterable[KeyAlias] = (),
    ) -> AssumptionKeyRegistry:
        if not isinstance(raw, Mapping):
            raise RegistryError(f"{source}: registry must be a mapping")

        version = raw.get("registry_version")
        if not isinstance(version, str) or not version.strip():
            raise RegistryError(f"{source}: 'registry_version' is required")

        declared = raw.get("keys")
        if not isinstance(declared, list) or not declared:
            raise RegistryError(f"{source}: 'keys' must be a non-empty list")

        entries: list[AssumptionKeyRegistryEntry] = []
        for index, item in enumerate(declared):
            if not isinstance(item, Mapping):
                raise RegistryError(f"{source}: keys[{index}] must be a mapping")
            try:
                entries.append(AssumptionKeyRegistryEntry(**item))
            except Exception as exc:  # pydantic ValidationError, kept vendor-agnostic
                raise RegistryError(
                    f"{source}: keys[{index}] ({item.get('key')!r}) is invalid: {exc}"
                ) from exc

        return cls(version=version, entries=entries, aliases=aliases)


def scope_is_permitted(entry: AssumptionKeyRegistryEntry, scope: ScopeSelector) -> bool:
    """Every dimension in ``scope`` must be declared by the key's registry entry.

    A scope is *validated*, never invented. Without this, an extractor could file a
    retention binding under ``business_unit:emea`` - a dimension the key has no declared
    meaning for - and the coupling set for that key would fragment along an axis nobody
    designed.

    ``global`` is always permitted: it is the absence of a constraint, not a dimension.
    """
    return all(entry.permits_dimension(dimension) for dimension in sorted(scope.dimensions))
