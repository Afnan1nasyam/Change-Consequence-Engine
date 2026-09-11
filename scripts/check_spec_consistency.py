"""Mechanical consistency check of the specification against DOMAIN_CHECKPOINT.md.

    python -m scripts.check_spec_consistency

Exit code 0 == clean.

The checkpoint is authoritative. This module asserts that the rest of the specification
agrees with it, so documentation drift is a failing check rather than something a reader has
to notice.

It is importable, and ``tests/contract/test_spec_consistency.py`` calls :func:`run_checks`,
because ``CLAUDE.md`` requires architectural rules to be enforced by a test rather than by
convention - and a script somebody has to remember to run is a convention.

Paths resolve against the repository root, not the working directory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "docs/DOMAIN_CHECKPOINT.md"


class Issue(NamedTuple):
    """One disagreement between a document and the checkpoint."""

    rule: str
    path: str
    detail: str


# A retired term may legitimately appear in prose that documents its retirement.
# Anything else is a live contradiction.
RETIREMENT_PROSE = re.compile(
    r"deliberately absent|deliberately excluded|Deliberately excluded|"
    r"\bremoved\b|\babolished\b|\bexcluded\b|\breplaced\b|split into|"
    r"no longer|\bwas undefined\b|previous draft|previously|"
    r"never carried|is wrong|\bretired\b|no combined|\bdropping\b|catch-all|"
    r"not implement|no automatic|superseded|unrepresentable|decided against|"
    r"withdrawn|no key normalizer|none is planned",
    re.I,
)

RETIRED: dict[str, str] = {
    r"\bbinding_kind\b": "split into role + explicitness (checkpoint 3.3/3.4)",
    r"\bextraction_confidence\b": (
        "abolished; explicitness enum + computed extraction_quality (9.1)"
    ),
    r"\blikelihood\b": "removed; a probability by another name (IDEA 13)",
    r"\bhas_collision\b": "removed; CouplingSet.outcome three-way rule (5.4)",
    r"\bpotentially_impacts\b": "excluded from the MVP taxonomy (IDEA 12)",
    r"\bunknowns\b": "replaced by evidence_gap_ids / EvidenceGap (type E)",
    # Phase 1 decision: normalization is registry-declared canonical form, never learned.
    r"key normaliz|normalizer": "no key normalizer in the MVP; registry canonical_form (4.1.5)",
}

# The checkpoint is authoritative; DATA_MODEL must declare the same value sets.
EXPECTED_ENUMS: dict[str, set[str]] = {
    "Derivation": {"observed", "extracted", "derived", "inferred", "absent"},
    "Support": {
        "verified_span",
        "multi_span",
        "rule_over_records",
        "mechanism_only",
        "negative_provenance",
    },
    "Status": {"supported", "likely", "disputed", "speculative", "rejected", "unresolved"},
    "Severity": {"LOW", "MEDIUM", "HIGH", "CRITICAL"},
    "ConsequenceClass": {"class_a_assumption_mediated", "class_b_behavioral"},
}

REQUIRED: dict[str, list[str]] = {
    "docs/DATA_MODEL.md": [
        "AssumptionKeyRegistryEntry",
        "KeyRegistryGap",
        "KeyAlias",
        "ScopeSelector",
        "DerivedRelation",
        "EvidenceGap",
        "ObservedFact",
        "ConsequenceScore",
        "FalsificationTest",
        "GraphNode",
        "GraphEdge",
        "ConsequenceGraph",
        "is_hypothesized",
        "unauthorized_override",
        "authority_class",
        "registry_version",
        "authority_model_version",
        "analyst_caveats",
        "no_refutation_found",
        "origin",
        "rule_id",
        "input_record_ids",
        # Phase 2 ingestion contracts.
        "AssumptionProposal",
        "QuarantinedProposal",
        "document-absolute",
    ],
    "docs/ARCHITECTURE.md": [
        "AssumptionKeyRegistry",
        "KeyRegistryGap",
        "KeyAlias",
        "unauthorized_override",
        "Admission Gate",
        "quarantin",
        "(key, scope, as_of)",
    ],
    "docs/IDEA.md": ["unauthorized_override", "registry", "explicitness", "scope"],
    "CLAUDE.md": ["never promote", "closed", "pure projection", "pipeline-assigned"],
    "docs/SECURITY.md": ["KeyRegistryGap", "KeyAlias", "cross-source"],
    ".claude/agents/adversarial-challenger.md": [
        "disputed_span",
        "no_refutation_found",
        "inadmissible",
    ],
    ".claude/agents/consequence-judge.md": ["do not assign", "severity", "unauthorized_override"],
    ".claude/agents/contradiction-hunter.md": [
        "unauthorized_override",
        "legitimate override",
        "authority",
    ],
    ".claude/agents/evidence-arbiter.md": ["explicit", "derived_from_text", "implied"],
    # The extractor proposes and admits nothing; its prompt must say so.
    ".claude/agents/assumption-extractor.md": [
        "propose",
        "quarantined",
        "never instruction",
        "confidence number",
    ],
}

EDGES: set[str] = {
    "directly_changes",
    "binds_assumption",
    "invalidates_assumption",
    "conflicts_with",
    "unauthorized_override",
    "creates_risk_for",
    "historically_analogous_to",
}


def norm(paths: list[str]) -> list[str]:
    return sorted(q.replace("\\", "/") for q in paths)


def discover() -> tuple[list[str], list[str]]:
    """The documents and agent definitions under review, repo-relative."""
    docs = [
        "CLAUDE.md",
        "README.md",
        *norm([str(q.relative_to(REPO_ROOT)) for q in (REPO_ROOT / "docs").glob("*.md")]),
    ]
    agents = norm(
        [str(q.relative_to(REPO_ROOT)) for q in (REPO_ROOT / ".claude" / "agents").glob("*.md")]
    )
    return docs, agents


def _line_of(body: str, pos: int) -> int:
    return body[:pos].count("\n") + 1


def _line_text(body: str, pos: int) -> str:
    start = body.rfind("\n", 0, pos) + 1
    end = body.find("\n", pos)
    return body[start : end if end != -1 else len(body)]


def run_checks() -> list[Issue]:
    """Every rule, over every document. An empty list means the specification agrees."""
    docs, agents = discover()
    every = docs + agents
    text = {p: (REPO_ROOT / p).read_text(encoding="utf-8") for p in every}
    issues: list[Issue] = []

    def flag(rule: str, path: str, detail: str) -> None:
        issues.append(Issue(rule, path, detail))

    # -- retired vocabulary ------------------------------------------------
    for pattern, why in RETIRED.items():
        for path in every:
            if path == CHECKPOINT:
                continue  # the checkpoint defines what was retired
            for match in re.finditer(pattern, text[path]):
                if RETIREMENT_PROSE.search(_line_text(text[path], match.start())):
                    continue
                flag(
                    "retired-vocab",
                    f"{path}:{_line_of(text[path], match.start())}",
                    f"{match.group(0)} -> {why}",
                )

    # -- enum agreement ----------------------------------------------------
    data_model = text["docs/DATA_MODEL.md"]
    for name, expected in EXPECTED_ENUMS.items():
        section = re.search(rf"^### {re.escape(name)}\s*\n\s*\n(.+?)\n", data_model, re.M | re.S)
        if not section:
            flag("enum-missing", "docs/DATA_MODEL.md", f'no "### {name}" value list')
            continue
        declared = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", section.group(1).replace("`", "")))
        if missing := expected - declared:
            flag("enum-mismatch", "docs/DATA_MODEL.md", f"{name} missing {sorted(missing)}")
        if extra := declared - expected:
            flag("enum-mismatch", "docs/DATA_MODEL.md", f"{name} has extra {sorted(extra)}")

    # -- confidence placement ---------------------------------------------
    # Only ConsequenceScore may carry a confidence field (checkpoint 9.1).
    for match in re.finditer(r"^- `?confidence`?\s*$", data_model, re.M):
        heading = data_model.rfind("\n### ", 0, match.start())
        owner = data_model[heading : data_model.find("\n", heading + 5)].strip("# \n")
        if owner != "ConsequenceScore":
            flag(
                "llm-confidence",
                f"docs/DATA_MODEL.md:{_line_of(data_model, match.start())}",
                f"{owner} carries a confidence field; only ConsequenceScore may (9.1)",
            )

    # -- required concepts -------------------------------------------------
    for path, needles in REQUIRED.items():
        lowered = text[path].lower()
        for needle in needles:
            if needle.lower() not in lowered:
                flag("missing-concept", path, f"expected to mention {needle!r}")

    # -- edge taxonomy -----------------------------------------------------
    for path in ["docs/IDEA.md", "docs/DATA_MODEL.md", CHECKPOINT]:
        if missing_edges := sorted(e for e in EDGES if e not in text[path]):
            flag("edge-taxonomy", path, f"missing edge types: {missing_edges}")

    for path in every:
        for match in re.finditer(r"six (MVP )?edge|edge types \(six\)|the six MVP", text[path]):
            if RETIREMENT_PROSE.search(_line_text(text[path], match.start())):
                continue
            flag(
                "edge-count",
                f"{path}:{_line_of(text[path], match.start())}",
                "taxonomy is seven edges",
            )

    # -- agent-level rules -------------------------------------------------
    for path in agents:
        if "confidence number" not in text[path]:
            flag(
                "agent-confidence",
                path,
                "no instruction against emitting a confidence number",
            )
        if re.search(r"^version:\s*\d+\s*$", text[path], re.M) is None:
            flag("agent-version", path, "no version in frontmatter")

    # -- specific regressions ---------------------------------------------
    if "must have evidence and confidence" in text["CLAUDE.md"]:
        flag(
            "claude-principle-6",
            "CLAUDE.md",
            "principle 6 still demands evidence AND confidence for inferred dependencies",
        )

    analogue = re.search(
        r"\|\s*`historically_analogous_to`\s*\|[^\n]*\|([^\n|]*)\|", text["docs/IDEA.md"]
    )
    if analogue and "inferred" not in analogue.group(0):
        flag(
            "analogue-derivation",
            "docs/IDEA.md",
            "historically_analogous_to must be `inferred`, not `observed`",
        )

    return issues


def main() -> int:
    docs, agents = discover()
    issues = run_checks()

    print("=" * 78)
    if not issues:
        print("CONSISTENCY: CLEAN")
    else:
        print(f"CONSISTENCY: {len(issues)} issue(s)")
        for issue in issues:
            print(f"  [{issue.rule:<17}] {issue.path:<46} {issue.detail}")
    print("=" * 78)
    print(f"checked {len(docs)} docs + {len(agents)} agent files against {CHECKPOINT}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
