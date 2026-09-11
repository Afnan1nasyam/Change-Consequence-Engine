"""ScopeSelector - a path over declared scope dimensions.

Scope is the field whose absence would have been the most damaging defect in the model.
Without it, ``retention.window = 90d @global`` and ``= 30d @region:EU`` compare as a
contradiction, and the highest-trust output floods with false positives on any realistic
corpus.

The value object stays deliberately dumb. Lattice operations live in
``app/domain/services/scope_lattice.py``, and validation of *which* dimensions a given key
may use is a deferred service check against the closed registry.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

#: Dimension names and values: lowercase, digits, underscore, hyphen.
_TOKEN = r"^[a-z][a-z0-9_-]*$"

ScopeToken = Annotated[str, StringConstraints(pattern=_TOKEN)]


class ScopeDimension(BaseModel):
    """One ``dimension:value`` step, e.g. ``region:EU`` (normalized to lowercase)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: ScopeToken
    value: ScopeToken

    def __str__(self) -> str:
        return f"{self.dimension}:{self.value}"


class ScopeSelector(BaseModel):
    """An ordered path of dimensions. An empty path means ``global``.

    Ordering is significant for rendering and for containment: ``region:EU / tier:enterprise``
    is contained by ``region:EU``, and the prefix relationship is what containment means.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: tuple[ScopeDimension, ...] = ()

    @model_validator(mode="after")
    def _no_repeated_dimension(self) -> Self:
        seen = [d.dimension for d in self.path]
        if len(seen) != len(set(seen)):
            raise ValueError(
                f"scope path repeats a dimension: {self.render()}. "
                "A dimension may constrain a scope only once."
            )
        return self

    # -- constructors -----------------------------------------------------

    @classmethod
    def global_scope(cls) -> ScopeSelector:
        return cls(path=())

    @classmethod
    def parse(cls, text: str) -> ScopeSelector:
        """Parse ``"region:EU / tier:enterprise"``. ``""`` and ``"global"`` mean global."""
        cleaned = text.strip()
        if not cleaned or cleaned == "global":
            return cls.global_scope()

        steps: list[ScopeDimension] = []
        for raw in cleaned.split("/"):
            token = raw.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"scope step {token!r} is not 'dimension:value' (in {text!r})")
            dimension, _, value = token.partition(":")
            steps.append(
                ScopeDimension(dimension=dimension.strip().lower(), value=value.strip().lower())
            )
        return cls(path=tuple(steps))

    # -- rendering --------------------------------------------------------

    def render(self) -> str:
        return " / ".join(str(d) for d in self.path) if self.path else "global"

    def __str__(self) -> str:
        return self.render()

    # -- shape ------------------------------------------------------------

    @property
    def is_global(self) -> bool:
        return not self.path

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def dimensions(self) -> frozenset[str]:
        return frozenset(d.dimension for d in self.path)
