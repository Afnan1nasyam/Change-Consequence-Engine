"""What the report composer is allowed to say. Ordering and narrowing, plus advisory prose.

The composer is the one agent in this system whose output cannot be verified against
evidence, because prose is not a citation. So its admitted output is **structural**: which of
the eligible findings to foreground, and in what order. Both are checkable - a subset and a
permutation - and both are real work: alert load is a correctness constraint, not
presentation.

The prose it also returns is carried as **advisory** and never becomes
:attr:`~app.domain.entities.findings.AnalysisReport.executive_summary`. That field is composed
deterministically from the finalized findings, which is the only way to guarantee the report
introduces no new factual claim. The narration is gated anyway - every identifier-shaped token
in it must already appear in the finalized set - so a reader who wants it gets a version that
at least cannot cite something that does not exist.

``confidence`` and ``severity`` are absent, and ``extra="forbid"`` makes an attempt to supply
one a validation error: the runtime's number and the judge's severity are carried through
unchanged, never blended or re-described.
"""

from __future__ import annotations

from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

COMPOSITION_SCHEMA_VERSION: Final = "report-composition-1"

#: Enforced structurally by ``extra="forbid"``; named so a test can assert the list.
FORBIDDEN_AGENT_FIELDS: Final = ("confidence", "severity", "status", "consequence_class")

#: Sections the composer may annotate. Closed, so a note cannot invent a section.
ANNOTATABLE_SECTIONS: Final = (
    "high_priority",
    "class_a",
    "class_b",
    "hidden",
    "blind_spots",
    "contradictions",
    "challenge_outcomes",
    "corpus_silence",
    "evidence_gaps",
    "unresolved_questions",
    "validation_actions",
    "reviewer_signals",
)

Handle = Annotated[str, StringConstraints(min_length=2, max_length=8)]
Prose = Annotated[str, StringConstraints(min_length=8, max_length=1200, strip_whitespace=True)]
Note = Annotated[str, StringConstraints(min_length=4, max_length=400, strip_whitespace=True)]


class SectionNote(BaseModel):
    """One advisory note about one named section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section: Note
    note: Note


class CompositionProposal(BaseModel):
    """The composer's whole output: a foreground subset, an order, and advisory prose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: A subset of the handles offered. The runtime intersects; anything else is dropped, and
    #: findings the surface rule makes mandatory are foregrounded whatever this says.
    foreground_handles: tuple[Handle, ...] = ()
    #: A reading order over the foreground. The runtime treats it as a preference and
    #: appends anything the composer left out, so an omission cannot hide a finding.
    reading_order: tuple[Handle, ...] = ()

    #: Advisory. Never becomes the report's executive summary.
    executive_summary: Prose | None = None
    section_notes: tuple[SectionNote, ...] = Field(default=(), max_length=12)
    #: Sections the composer judged empty. Compared against the deterministic answer.
    empty_sections: tuple[Note, ...] = ()

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()
