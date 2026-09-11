"""The effective permission scope a retrieval runs under, and its hash.

``docs/SECURITY.md`` 4 requires authorization to be enforced *before* retrieval output
enters an agent, and ``ARCHITECTURE.md`` 14.1 requires an analysis to be scoped to the
permission set it was computed under - a differing scope forces recomputation rather than a
filtered replay of stored findings. Both need one thing this module provides: a canonical,
hashable statement of "who was asking".

The model, decided here rather than left implicit:

* :class:`~app.domain.entities.source.SourceDocument` carries ``permissions``, a set of
  **labels**. The labels answer "who may read this", so they are an access-control list and
  not a clearance requirement: holding **any** one of them grants read.
* A document with **no** labels carries no restriction and is readable by any scope inside
  its tenant. This is the one fail-open edge in the design and it is deliberate - a label
  set is a statement, and an empty statement restricts nothing - but it means a connector
  that forgets to populate ``permissions`` publishes a restricted document to the whole
  tenant. That is recorded in ``docs/STATUS.md`` as a boundary, not hidden here.
* Comparison is normalized on **both** sides. A document labelled ``HR-Confidential`` is
  matched by the grant ``hr-confidential``, because an asymmetric comparison makes access
  depend on which side the capitalization happened to land - the same failure the retrieval
  tokenizer avoids.

Tenant is **not** part of this. Tenant isolation is structural and upstream (a store port
takes a tenant, and the retrieval index is partitioned by it); folding it into a grant would
turn a partition into a filter, which is strictly weaker.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

#: The hashing recipe and the match rule, taken together. Pinned into the hash preimage, so
#: a change to either produces different hashes and cannot be mistaken for the same scope.
PERMISSION_MODEL_VERSION: Final = "permission-1"

#: Bounded on purpose: an unbounded grant list is an unbounded hash preimage.
MAX_GRANTS: Final = 256

#: A grant may not contain a line break. That is what makes the hash preimage below
#: unambiguous: the grants are newline-joined, so a grant able to contain a newline could
#: forge the canonical form of a *different* scope and collide with it.
Grant = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[^\r\n]+$")]


def normalize_label(label: str) -> str:
    """One label to its comparable form. Applied to grants and to document labels alike."""
    return label.strip().casefold()


def normalize_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Deduplicated, sorted, comparable labels. Blank entries are dropped."""
    return tuple(sorted({normalize_label(label) for label in labels if normalize_label(label)}))


class PermissionScope(BaseModel):
    """The grants a requester holds, in canonical form.

    Required wherever it is used rather than optional. An optional scope defaulting to "no
    filtering" would make forgetting the parameter the unrestricted case, and the whole
    point of the design is that the unsafe thing must be the one that is hard to write
    (``docs/SECURITY.md`` 11).

    An empty scope is legal and is the **least** privileged one: it reads unlabelled
    documents and nothing else. It is not an administrator, and there is deliberately no
    wildcard grant - a bypass that exists is a bypass that gets used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grants: tuple[Grant, ...] = Field(default=(), max_length=MAX_GRANTS)

    @field_validator("grants", mode="before")
    @classmethod
    def _canonical(cls, value: Any) -> Any:
        """Normalize on the way in, so the hash cannot depend on how a scope was spelled.

        A bare string is rejected rather than treated as one grant: ``grants="payments"``
        would otherwise iterate into eight single-character grants, which is a silent
        privilege change and exactly the kind of near-miss the strict schemas exist to stop.
        """
        if isinstance(value, str):
            raise ValueError("grants must be a sequence of labels, not a single string")
        if value is None:
            return ()
        if isinstance(value, Iterable):
            labels = list(value)
            if len(labels) > MAX_GRANTS:
                raise ValueError(f"at most {MAX_GRANTS} grants, got {len(labels)}")
            return normalize_labels(labels)
        return value

    @classmethod
    def of(cls, *grants: str) -> PermissionScope:
        """Convenience constructor: ``PermissionScope.of("payments", "privacy")``."""
        return cls(grants=tuple(grants))

    # -- the decision ----------------------------------------------------

    def permits(self, labels: Iterable[str]) -> bool:
        """Whether this scope may read a document carrying ``labels``.

        Any single matching label grants read. No labels means no restriction.
        """
        required = normalize_labels(labels)
        if not required:
            return True
        return bool(set(required) & set(self.grants))

    # -- identity --------------------------------------------------------

    @property
    def scope_hash(self) -> str:
        """A deterministic digest of the *effective* scope.

        Deterministic means two scopes that grant the same access hash the same, however
        they were spelled: ``("b", "a")``, ``("a", "b", "a")`` and ``("A", " b ")`` all
        normalize to one preimage. Two scopes that grant *different* access never collide,
        because the preimage is the newline-joined canonical grant list and a newline cannot
        occur inside a grant.

        This is the value ``RunProvenance.permission_scope_hash`` records, and the value an
        analysis is keyed on so a differing scope forces recomputation instead of a filtered
        replay (``ARCHITECTURE.md`` 14.1).
        """
        preimage = "\n".join((PERMISSION_MODEL_VERSION, *self.grants))
        return hashlib.sha256(preimage.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        return f"scope[{', '.join(self.grants) if self.grants else 'unlabelled-only'}]"
