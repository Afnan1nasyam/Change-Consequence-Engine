"""Build the coupling index from the stores, and answer a delta against it.

This is the only place in the phase that touches a store. Everything it calls is a pure
function over values, which is deliberate: the collision matrix, the authority resolver and
the decay report are all testable by constructing bindings, and none of them can be made
nondeterministic by store iteration order.

The join this use case performs is the one the domain cannot: ``authority_class`` is a
property of the holder **document** (checkpoint 3.5a), and it comes from *registration*,
never from document content. Doing the join here, once, keeps that security property in one
auditable place - a planted file cannot claim policy authority because nothing reads
authority from the file (``docs/SECURITY.md``).

There is no model call anywhere in this module, and no parameter through which one could be
supplied. The central Class A claim is that these consequences are found without an LLM;
that claim is enforced by the absence of a seam rather than by a promise.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from app.assumptions.class_a import derive_class_a
from app.assumptions.collisions import KeyCollisions
from app.assumptions.coupling import CouplingIndex, build_coupling_index
from app.assumptions.decay import build_report
from app.assumptions.invalidation import InvalidationReport, propagate
from app.domain.entities.assumption import AssumptionDelta
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.coupling import CouplingStrength, ResolvedBinding
from app.domain.entities.decay import KnowledgeDecayReport
from app.domain.interfaces.repositories import (
    AssumptionStore,
    CouplingIndexStore,
    RegistryGapStore,
    SourceStore,
)
from app.domain.services import coupling_strength

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CouplingAnalysis:
    """Everything the deterministic layer concluded, with nothing merged or averaged."""

    index: CouplingIndex
    findings: tuple[ClassAFinding, ...]
    report: KnowledgeDecayReport
    strengths: dict[str, CouplingStrength]
    #: Bindings that could not be joined to a document, and were therefore excluded.
    unresolvable: tuple[str, ...] = ()

    @property
    def collisions(self) -> dict[str, KeyCollisions]:
        return self.index.collisions


class ResolveCouplingUseCase:
    """Rebuild the coupling index and the standing decay report for one tenant."""

    def __init__(
        self,
        *,
        sources: SourceStore,
        assumptions: AssumptionStore,
        gaps: RegistryGapStore | None = None,
        index_store: CouplingIndexStore | None = None,
    ) -> None:
        self._sources = sources
        self._assumptions = assumptions
        self._gaps = gaps
        self._index_store = index_store

    def execute(
        self,
        *,
        tenant_id: str,
        as_of: dt.datetime,
        registry_version: str,
    ) -> CouplingAnalysis:
        bindings, unresolvable = self.assemble(tenant_id)

        index = build_coupling_index(bindings, as_of=as_of, registry_version=registry_version)
        strengths = _score(index, as_of=as_of)
        findings = derive_class_a(
            index.relations,
            bindings=bindings,
            tenant_id=tenant_id,
            registry_version=registry_version,
            strengths=strengths,
        )
        report = build_report(
            index,
            tenant_id=tenant_id,
            findings=findings,
            gaps=self._gaps.open_gaps() if self._gaps else (),
        )

        if self._index_store is not None:
            self._index_store.put_index(tenant_id, index.sets)
            self._index_store.put_relations(tenant_id, index.relations)
            self._index_store.put_findings(tenant_id, findings)

        logger.info(
            "coupling_index_built",
            extra={
                "tenant_id": tenant_id,
                "as_of": as_of.isoformat(),
                "registry_version": registry_version,
                "coupling_index_version": index.coupling_index_version,
                "keys": len(index.keys),
                "bindings": index.binding_count,
                "coupling_sets": len(index.sets),
                "relations": len(index.relations),
                "class_a_findings": len(findings),
                "legitimate_overrides": len(index.overrides),
                "ambiguous_keys": len(index.ambiguous_resolutions),
                "anomalies": len(index.anomalies),
                "decay_entries": len(report.entries),
                "unresolvable_bindings": len(unresolvable),
            },
        )

        return CouplingAnalysis(
            index=index,
            findings=findings,
            report=report,
            strengths=strengths,
            unresolvable=unresolvable,
        )

    def assemble(self, tenant_id: str) -> tuple[tuple[ResolvedBinding, ...], tuple[str, ...]]:
        """Join every admitted binding to its holder document's authority.

        A binding whose document or version has gone missing is **excluded and reported**,
        never defaulted to some authority class. Guessing an authority would silently
        change which overrides count as legitimate, and the guess would not be visible in
        the finding it produced.
        """
        resolved: list[ResolvedBinding] = []
        unresolvable: list[str] = []

        for record in self._assumptions.all_records(tenant_id):
            version = self._sources.get_version(record.holder_source_version_id)
            if version is None:
                unresolvable.append(record.assumption_id)
                continue
            document = self._sources.get_document(tenant_id, version.source_id)
            if document is None:
                unresolvable.append(record.assumption_id)
                continue

            resolved.append(
                ResolvedBinding(
                    record=record,
                    authority_class=document.authority_class,
                    source_id=document.source_id,
                    source_title=document.title,
                )
            )

        if unresolvable:
            logger.warning(
                "bindings_excluded_from_coupling",
                extra={
                    "tenant_id": tenant_id,
                    "count": len(unresolvable),
                    "reason": "holder document or version not found; authority unknown",
                },
            )

        return tuple(sorted(resolved, key=lambda b: b.assumption_id)), tuple(sorted(unresolvable))

    def apply_delta(
        self,
        delta: AssumptionDelta,
        analysis: CouplingAnalysis,
        *,
        as_of: dt.datetime | None = None,
    ) -> InvalidationReport:
        """Propagate one delta against an already-built index."""
        report = propagate(delta, analysis.index, as_of=as_of)
        logger.info(
            "delta_propagated",
            extra={
                "delta_id": delta.delta_id,
                "assumption_key": delta.key,
                "delta_kind": delta.delta_kind.value,
                "invalidated": len(report.invalidated),
                "needs_confirmation": len(report.needs_confirmation),
                "pre_existing_overrides": len(report.pre_existing_unauthorized_overrides),
            },
        )
        return report


def _score(index: CouplingIndex, *, as_of: dt.datetime) -> dict[str, CouplingStrength]:
    """Coupling strength per derived relation, keyed by relation id.

    Scored over the relation's two input records where both are present. Strength is
    currently *reported* rather than used as a filter: the weights encode only the
    orderings the checkpoint states, and no calibration data exists to tune them.
    """
    by_assumption = {b.assumption_id: b for b in index.bindings}
    scores: dict[str, CouplingStrength] = {}

    for relation in index.relations:
        inputs = [by_assumption[i] for i in relation.input_record_ids if i in by_assumption]
        if len(inputs) < 2:
            continue
        scores[relation.relation_id] = coupling_strength.compute(inputs[0], inputs[1], as_of=as_of)
    return scores
