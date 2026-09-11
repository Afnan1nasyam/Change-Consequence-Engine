"""TypedValue - the payload an artifact binds to an assumption key.

Only *structural* validation lives here. Unit-aware comparison and conversion are deferred
domain services, because a unit mismatch must **block** coupling rather than weaken it
(checkpoint 5.6) and that decision needs the registry entry, not just the value.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from app.domain.value_objects.enums import UNIT_REQUIRED_TYPES, ValueType

Unit = Annotated[str, StringConstraints(min_length=1, max_length=32)]


class TypedValue(BaseModel):
    """A typed, comparable value.

    ``unit`` is mandatory for scalars and durations. A unitless scalar cannot be compared
    safely, and an unsafe comparison is precisely how a false contradiction gets
    manufactured - so the schema refuses to represent one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value_type: ValueType
    raw: Any
    unit: Unit | None = None

    @model_validator(mode="after")
    def _unit_required_where_applicable(self) -> Self:
        if self.value_type in UNIT_REQUIRED_TYPES and not self.unit:
            raise ValueError(
                f"unit is required for value_type={self.value_type.value!r}; "
                "a unitless scalar or duration cannot be compared safely"
            )
        if self.value_type not in UNIT_REQUIRED_TYPES and self.unit:
            raise ValueError(
                f"unit must be omitted for value_type={self.value_type.value!r}, got {self.unit!r}"
            )
        return self

    @model_validator(mode="after")
    def _raw_matches_declared_type(self) -> Self:
        expected: dict[ValueType, tuple[type, ...]] = {
            ValueType.SCALAR: (int, float),
            ValueType.DURATION: (int, float),
            ValueType.BOOLEAN: (bool,),
            ValueType.ENUM: (str,),
            ValueType.REFERENCE: (str,),
            ValueType.INTERVAL: (list, tuple),
        }
        allowed = expected[self.value_type]

        # bool is a subclass of int in Python; keep the two genuinely distinct.
        if self.value_type in (ValueType.SCALAR, ValueType.DURATION) and isinstance(self.raw, bool):
            raise ValueError(
                f"value_type={self.value_type.value!r} received a bool; use value_type='boolean'"
            )

        if not isinstance(self.raw, allowed):
            raise ValueError(
                f"value_type={self.value_type.value!r} expects raw of "
                f"{[t.__name__ for t in allowed]}, got {type(self.raw).__name__}"
            )

        # Re-checked against a literal type union: the check above uses a runtime
        # `allowed` tuple, which tells a type checker nothing about what `raw` is.
        if self.value_type is ValueType.INTERVAL and (
            not isinstance(self.raw, list | tuple) or len(self.raw) != 2
        ):
            raise ValueError("interval raw must have exactly two endpoints")

        return self

    def is_unit_compatible_with(self, other: TypedValue) -> bool:
        """Structural pre-check only.

        Genuine conversion (``h`` to ``min``, ``USD`` to ``EUR``) is a deferred service.
        This answers the cheap question the schema layer can answer: are these even the
        same kind of quantity, nominally?
        """
        return self.value_type is other.value_type and self.unit == other.unit

    def render(self) -> str:
        return f"{self.raw}{self.unit}" if self.unit else str(self.raw)
