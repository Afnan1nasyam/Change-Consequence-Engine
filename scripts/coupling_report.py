"""Ingest the corpus, build the coupling index, and print what the resolver derived.

    python -m scripts.coupling_report
    python -m scripts.coupling_report --as-of 2026-01-01
    python -m scripts.coupling_report --delta refund.approval_threshold=5000USD
    python -m scripts.coupling_report --json

This is the end-to-end demonstration of the central Class A claim. **Nothing in the
coupling path makes a model call**, and the run needs no credentials and no network: the
extractor is the replay harness, so the only thing substituted for a real call is the
proposal text, and every deterministic check downstream is genuine.

The ``--delta`` flag expresses a proposed change as an assumption delta and propagates it,
which is the E1 worked example from the checkpoint: change the refund threshold and watch
the training deck that teaches the old one fall out as invalidated.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys

from app.application.use_cases.resolve_coupling import CouplingAnalysis, ResolveCouplingUseCase
from app.assumptions.class_a import merge_class_b
from app.assumptions.invalidation import InvalidationReport
from app.domain.entities.assumption import AssumptionDelta
from app.domain.entities.decay import DecayParticipant
from app.domain.value_objects.enums import (
    AuthorityOutcome,
    CouplingOutcome,
    DeltaAssertedBy,
    DeltaKind,
    InvalidationOutcome,
    ValueType,
)
from app.domain.value_objects.identifiers import new_id
from app.domain.value_objects.scope import ScopeSelector
from app.domain.value_objects.taxonomy import Relationship
from app.domain.value_objects.typed_value import TypedValue
from app.evaluation.corpus import load_corpus
from app.infrastructure.database.repositories.memory import InMemoryCouplingIndexStore

RULE = "=" * 100
DEFAULT_AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


async def run(*, as_of: dt.datetime, delta_spec: str | None, as_json: bool) -> int:
    stores = await load_corpus()
    if stores.extraction_failures:
        print(f"extraction failures: {stores.extraction_failures}", file=sys.stderr)
        return 1

    index_store = InMemoryCouplingIndexStore()
    use_case = ResolveCouplingUseCase(
        sources=stores.sources,
        assumptions=stores.assumptions,
        gaps=stores.gaps,
        index_store=index_store,
    )
    analysis = use_case.execute(
        tenant_id=stores.tenant_id,
        as_of=as_of,
        registry_version=stores.registry_version,
    )

    invalidation: InvalidationReport | None = None
    if delta_spec is not None:
        delta = _parse_delta(delta_spec, registry_version=stores.registry_version)
        invalidation = use_case.apply_delta(delta, analysis, as_of=as_of)

    if as_json:
        print(json.dumps(_as_json(analysis, invalidation), indent=2, sort_keys=True))
        return 0

    _print_index(analysis)
    _print_authority(analysis)
    _print_findings(analysis)
    _print_decay(analysis)
    if invalidation is not None:
        _print_invalidation(invalidation)
    _print_summary(analysis, invalidation)
    return 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _print_index(analysis: CouplingAnalysis) -> None:
    print(f"\n{RULE}\nCOUPLING INDEX  (key, scope, as_of) -> holders, outcome\n{RULE}")
    for cell in analysis.index.sets:
        marker = "  " if cell.outcome is CouplingOutcome.CONSISTENT else "! "
        print(
            f"{marker}{cell.key:<32} {cell.scope.render():<12} "
            f"holders={cell.holder_count} declarers={cell.declarer_count} "
            f"distinct={cell.distinct_declared_values}  -> {cell.outcome.value}"
        )


def _print_authority(analysis: CouplingAnalysis) -> None:
    print(f"\n{RULE}\nTEMPORAL AUTHORITY  resolve_authority(key, scope, as_of)\n{RULE}")
    for key in analysis.index.keys:
        collisions = analysis.index.collisions[key]
        for scope_name in sorted(collisions.authority):
            resolution = collisions.authority[scope_name]
            if resolution.outcome is AuthorityOutcome.RESOLVED:
                winner = resolution.authoritative
                assert winner is not None
                detail = (
                    f"{winner.value.render():<10} <- {winner.source_title} "
                    f"({winner.authority_class.value})"
                )
            elif resolution.outcome is AuthorityOutcome.AMBIGUOUS:
                assert resolution.ambiguity_reason is not None
                tied = ", ".join(
                    f"{c.source_title}={c.value.render()}" for c in resolution.tied_candidates
                )
                detail = f"AMBIGUOUS ({resolution.ambiguity_reason.value}): {tied}"
            else:
                detail = "no declarer"

            flagged = (
                f"   [flagged narrowings: {len(resolution.unauthorized_narrowings)}]"
                if resolution.unauthorized_narrowings
                else ""
            )
            print(f"  {key:<32} {scope_name:<12} {detail}{flagged}")


def _print_findings(analysis: CouplingAnalysis) -> None:
    print(f"\n{RULE}\nCLASS A FINDINGS  (deterministic, zero model calls)\n{RULE}")
    if not analysis.findings:
        print("  none")
    for finding in analysis.findings:
        strength = (
            f"strength={finding.strength.strength:.3f}" if finding.strength else "strength=n/a"
        )
        print(
            f"\n  [{finding.rule_id}] {finding.relationship.value}  "
            f"{finding.consequence_class.value}  origin={finding.origin.value}  {strength}"
        )
        print(f"    key       {finding.assumption_key} @ {finding.scope.render()}")
        print(f"    affected  {finding.affected_artifact}  ({finding.affected_source_version_id})")
        print(f"    summary   {finding.summary}")
        print(
            f"    evidence  {len(finding.materialized_evidence_ids)} span(s), "
            f"derived union = {finding.evidence_is_derived_union}"
        )
        print(f"    inputs    {', '.join(finding.input_record_ids)}")


def _print_decay(analysis: CouplingAnalysis) -> None:
    report = analysis.report
    print(f"\n{RULE}\nKNOWLEDGE-DECAY REPORT  (no change request as input)\n{RULE}")
    print(
        f"  {report.keys_examined} keys, {report.bindings_examined} bindings, "
        f"registry={report.registry_version}, coupling={report.coupling_index_version}, "
        f"authority={report.authority_model_version}"
    )

    for label, rows in (
        ("KNOWN INCONSISTENCY  (an artifact is wrong)", report.known_inconsistencies),
        ("KNOWN AMBIGUITY  (the corpus cannot decide)", report.known_ambiguities),
        ("MISSING KNOWLEDGE  (nobody wrote it down)", report.missing_knowledge),
    ):
        print(f"\n  -- {label} -- {len(rows)}")
        for row in rows:
            print(f"     [{row.category.value}] {row.key} @ {row.scope.render()}")
            print(f"        {row.summary}")
            for participant in row.participants:
                print(f"          - {_participant_line(participant)}")
            if row.what_would_resolve:
                print(f"        resolve: {row.what_would_resolve}")


def _participant_line(participant: DecayParticipant) -> str:
    """One decay participant, rendered for a terminal.

    Split out of the loop because the inline version was four nested conditionals inside an
    f-string, which is unreadable at exactly the moment somebody is trying to check whether
    the report is telling the truth.
    """
    parts = [participant.artifact]

    qualifiers = [
        participant.authority_class.value if participant.authority_class else None,
        participant.role.value if participant.role else None,
    ]
    stated = [q for q in qualifiers if q]
    if stated:
        parts.append(f"({', '.join(stated)})")

    if participant.rendered_value:
        parts.append(f"= {participant.rendered_value}")
    if participant.effective_from:
        parts.append(f"from {participant.effective_from.date().isoformat()}")

    return " ".join(parts)


def _print_invalidation(report: InvalidationReport) -> None:
    print(f"\n{RULE}\nDELTA PROPAGATION  {report.delta.key}\n{RULE}")
    old = report.delta.old_value.render() if report.delta.old_value else "-"
    new = report.delta.new_value.render() if report.delta.new_value else "-"
    print(f"  {report.delta.delta_kind.value}: {old} -> {new} @ {report.delta.scope.render()}")

    for outcome, label in (
        (InvalidationOutcome.INVALIDATES_ASSUMPTION, "INVALIDATED"),
        (InvalidationOutcome.SCOPE_INHERITANCE_AFFECTED, "CONFIRM (scope inheritance)"),
        (InvalidationOutcome.NOT_AFFECTED, "not affected"),
    ):
        impacts = report.of_outcome(outcome)
        print(f"\n  -- {label} -- {len(impacts)}")
        for impact in impacts:
            print(
                f"     {impact.binding.source_title} "
                f"({impact.binding.role.value}, {impact.binding.scope.render()}) "
                f"= {impact.binding.value.render()}"
            )
            print(f"        {impact.reason}")

    if report.pre_existing_unauthorized_overrides:
        print(
            f"\n  -- PRE-EXISTING, NOT CAUSED BY THIS CHANGE -- "
            f"{len(report.pre_existing_unauthorized_overrides)}"
        )
        for relation in report.pre_existing_unauthorized_overrides:
            detail = relation.provenance.detail
            print(
                f"     unauthorized_override: {detail.get('narrower_artifact')} "
                f"({detail.get('narrower_authority_class')}) over "
                f"{detail.get('broader_artifact')} ({detail.get('broader_authority_class')})"
            )


def _print_summary(analysis: CouplingAnalysis, invalidation: InvalidationReport | None) -> None:
    index = analysis.index
    merge, _ = merge_class_b((), analysis.findings)
    print(f"\n{RULE}")
    print(
        f"{len(index.keys)} keys | {index.binding_count} bindings | {len(index.sets)} index cells"
        f" | {len(index.relations)} derived relations | {len(analysis.findings)} Class A findings"
    )
    print(
        f"{len(index.relations_of(Relationship.CONFLICTS_WITH))} conflicts | "
        f"{len(index.overrides)} legitimate overrides (silent) | "
        f"{len(index.ambiguous_resolutions)} ambiguous | "
        f"{len(index.anomalies)} anomalies | "
        f"{len(analysis.report.entries)} decay entries"
    )
    print(f"class_b_rediscovery_rate: {merge.class_b_rediscovery_rate:.3f} (0 hypotheses run)")
    print("model calls in the coupling path: 0")
    print(RULE)


def _as_json(
    analysis: CouplingAnalysis, invalidation: InvalidationReport | None
) -> dict[str, object]:
    """A machine-readable dump, for diffing two runs."""
    return {
        "as_of": analysis.index.as_of.isoformat(),
        "registry_version": analysis.index.registry_version,
        "coupling_index_version": analysis.index.coupling_index_version,
        "authority_model_version": analysis.index.authority_model_version,
        "coupling_sets": [
            {
                "key": cell.key,
                "scope": cell.scope.render(),
                "holders": cell.holder_count,
                "outcome": cell.outcome.value,
            }
            for cell in analysis.index.sets
        ],
        "findings": [
            {
                "rule_id": finding.rule_id,
                "relationship": finding.relationship.value,
                "key": finding.assumption_key,
                "scope": finding.scope.render(),
                "affected": finding.affected_artifact,
                "evidence": list(finding.materialized_evidence_ids),
            }
            for finding in analysis.findings
        ],
        "decay": analysis.report.counts_by_category,
        "invalidated": [impact.binding.source_title for impact in invalidation.invalidated]
        if invalidation
        else [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_delta(spec: str, *, registry_version: str) -> AssumptionDelta:
    """Parse ``key[@scope]=[old->]new`` into an AssumptionDelta.

    Supplying ``old`` matters for the demonstration: with it the delta is a
    ``value_change`` and the resolver can identify *which* declaration the change edits,
    and therefore exclude it from its own impact report. Without it the change is a
    ``key_added`` and every divergent binding - including the policy being edited - shows
    up as invalidated, which is correct but less informative.

    Deliberately crude: this is a demonstration entry point, not the change-decomposer.
    The real ``ChangeSpec`` path arrives with the agent layer.
    """
    target, _, value_text = spec.partition("=")
    if not value_text:
        raise SystemExit(f"--delta expects key=value, got {spec!r}")

    key, _, scope_text = target.partition("@")
    scope = ScopeSelector.parse(scope_text) if scope_text else ScopeSelector.global_scope()

    old_text, arrow, new_text = value_text.partition("->")
    if not arrow:
        old_text, new_text = "", value_text

    new_value = _parse_value(new_text)
    old_value = _parse_value(old_text) if old_text else None

    return AssumptionDelta(
        delta_id=new_id("delta"),
        key=key,
        scope=scope,
        old_value=old_value,
        new_value=new_value,
        delta_kind=DeltaKind.VALUE_CHANGE if old_value else DeltaKind.KEY_ADDED,
        asserted_by=DeltaAssertedBy.CHANGE_REQUEST,
        registry_version=registry_version,
        note=f"scripts.coupling_report --delta {spec}",
    )


def _parse_value(text: str) -> TypedValue:
    """``"5000USD"`` or ``"180d"`` into a TypedValue. A unit is mandatory."""
    cleaned = text.strip()
    digits = "".join(c for c in cleaned if c.isdigit() or c == ".")
    unit = cleaned[len(digits) :].strip()
    if not digits or not unit:
        raise SystemExit(
            f"--delta value must be a number with a unit, got {text!r}. A unitless scalar "
            "cannot be compared safely."
        )

    number = float(digits) if "." in digits else int(digits)
    value_type = ValueType.DURATION if unit in {"d", "h", "min", "s"} else ValueType.SCALAR
    return TypedValue(value_type=value_type, raw=number, unit=unit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        default=DEFAULT_AS_OF.date().isoformat(),
        help="evaluation date (YYYY-MM-DD); coupling is always evaluated as of a date",
    )
    parser.add_argument(
        "--delta",
        default=None,
        help="propose a change, e.g. refund.approval_threshold=5000USD",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    as_of = dt.datetime.combine(dt.date.fromisoformat(args.as_of), dt.time(tzinfo=dt.UTC))
    return asyncio.run(run(as_of=as_of, delta_spec=args.delta, as_json=args.json))


if __name__ == "__main__":
    sys.exit(main())
