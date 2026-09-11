"""The extractor's output contract: proposals, not bindings.

Nothing in this module is trusted. A proposal is what a model said about a chunk; an
:class:`~app.domain.entities.assumption.AssumptionRecord` is what survived deterministic
verification. Keeping them in different types is the whole of *extraction proposes,
verification admits* - there is no field an extractor can set that makes a proposal
admitted, because admission produces a different object.

Three structural refusals here, each blocking a specific failure:

* **No offsets.** The model supplies a verbatim ``quote``; code locates it. A model that
  could supply offsets could supply *plausible* offsets and there would be nothing left to
  verify.
* **No confidence, no severity, no class.** ``extra="forbid"`` makes an extractor that
  emits ``{"confidence": 0.9}`` a validation error rather than a silently dropped key.
* **The value must live inside its own quote.** Enforced by a validator, so a proposal that
  cites a real sentence and attaches an unrelated number cannot be constructed at all.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.domain.value_objects.enums import BindingRole, Explicitness, ValueType

QuoteText = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class AssumptionProposal(BaseModel):
    """One candidate binding, as proposed by the extractor."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=False)

    #: Free text on purpose: this is what the model actually emitted, and an unregistered
    #: key must survive into a KeyRegistryGap rather than failing to parse.
    proposed_key: ShortText

    #: Rendered scope path, e.g. ``"global"`` or ``"region:EU / tier:enterprise"``.
    scope: ShortText = "global"

    #: The value exactly as written in the document ("10,000", "24", "true").
    value_quote: ShortText
    value_type: ValueType
    unit: ShortText | None = None

    role: BindingRole
    explicitness: Explicitness

    #: The span the value was read from. Must contain ``value_quote`` verbatim.
    quote: QuoteText
    #: Which chunk the model was shown. Bounds the search window during verification.
    chunk_ordinal: int = Field(ge=0)

    effective_from: dt.date | None = None
    effective_to: dt.date | None = None

    #: Where the model is unsure. Free text, carried into the quarantine or the record's
    #: audit trail - never a number, and never a reason to admit anything.
    ambiguity: ShortText | None = None

    @model_validator(mode="after")
    def _value_lives_inside_its_quote(self) -> Self:
        if self.value_quote not in self.quote:
            raise ValueError(
                f"value_quote {self.value_quote!r} does not appear in quote "
                f"{self.quote[:80]!r}. The value must be read from the span that is cited "
                "for it."
            )
        return self

    @model_validator(mode="after")
    def _interval_ordered(self) -> Self:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise ValueError(
                f"effective_from ({self.effective_from.isoformat()}) must precede "
                f"effective_to ({self.effective_to.isoformat()})"
            )
        return self


class ExtractionBatch(BaseModel):
    """What one extraction call returns.

    An empty ``bindings`` list is a legitimate and common answer: most chunks of most
    documents bind nothing. An extractor that always finds something has not been shown to
    find anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bindings: tuple[AssumptionProposal, ...] = ()
    #: Instructions or requests found *in the document*, recorded as content so an
    #: injection attempt is visible in the audit trail rather than silently obeyed.
    observed_instructions: tuple[ShortText, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The schema handed to the provider for structured output."""
        return cls.model_json_schema()
