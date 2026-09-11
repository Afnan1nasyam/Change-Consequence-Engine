"""Trust invariants of the deterministic layer, asserted negatively.

Every test here answers "what would it take to break the Class A promise?" and then
demonstrates that the thing is impossible rather than merely discouraged:

* no module in the coupling path can reach a provider, a gateway or an agent runtime;
* agent identity cannot affect corroboration, because nothing in the scorer's reachable
  type carries one;
* model confidence cannot enter Class A scoring;
* evidence cannot be fabricated - a made-up id fails at the schema boundary;
* a derived finding cannot masquerade as directly evidenced;
* an unadmitted binding cannot enter the index.
"""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from app.config.settings import REPO_ROOT
from app.domain.entities.claims import DerivedRelation
from app.domain.entities.class_a import ClassAFinding
from app.domain.entities.coupling import CouplingStrength, ResolvedBinding
from app.domain.entities.decay import KnowledgeDecayReport
from app.domain.value_objects.enums import Derivation, Support
from app.domain.value_objects.taxonomy import ADMISSIBLE_DERIVATIONS, Relationship
from tests.fixtures.coupling import binding, class_a_finding, days

#: The modules that produce Class A. If any of them could make a model call, the central
#: claim - that these consequences are found *without* an LLM - would be unenforceable.
COUPLING_PATH = (
    "app/assumptions/temporal.py",
    "app/assumptions/collisions.py",
    "app/assumptions/coupling.py",
    "app/assumptions/invalidation.py",
    "app/assumptions/class_a.py",
    "app/assumptions/decay.py",
    "app/domain/services/value_comparison.py",
    "app/domain/services/temporal_intervals.py",
    "app/domain/services/coupling_strength.py",
    "app/domain/services/scope_lattice.py",
    "app/application/use_cases/resolve_coupling.py",
)

#: Anything that could reach a model, directly or through our own abstractions.
MODEL_REACHING = frozenset(
    {
        "groq",
        "google",
        "google_genai",
        "mistralai",
        "ollama",
        "openai",
        "anthropic",
        "langchain",
        "langgraph",
        "httpx",
        "requests",
        "aiohttp",
    }
)

MODEL_REACHING_MODULES = (
    "app.infrastructure.llm",
    "app.agents",
    "app.assumptions.extraction",
    "app.assumptions.prompting",
)


def _imports(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
            modules.add(node.module)
    return roots, modules


# ---------------------------------------------------------------------------
# No model in the Class A path
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.parametrize("relative", COUPLING_PATH)
def test_no_coupling_module_can_reach_a_provider(relative: str) -> None:
    roots, _modules = _imports(REPO_ROOT / relative)
    assert not (roots & MODEL_REACHING), f"{relative} imports {sorted(roots & MODEL_REACHING)}"


@pytest.mark.contract
@pytest.mark.parametrize("relative", COUPLING_PATH)
def test_no_coupling_module_can_reach_the_gateway_or_agent_runtime(relative: str) -> None:
    """Not even through our own abstraction: there must be no seam at all."""
    _roots, modules = _imports(REPO_ROOT / relative)
    offenders = sorted(
        module
        for module in modules
        if any(module.startswith(prefix) for prefix in MODEL_REACHING_MODULES)
    )
    assert not offenders, f"{relative} imports {offenders}"


@pytest.mark.contract
def test_the_coupling_path_files_all_exist() -> None:
    """A renamed module must not silently drop out of the guarded set."""
    missing = [relative for relative in COUPLING_PATH if not (REPO_ROOT / relative).exists()]
    assert not missing, missing


# ---------------------------------------------------------------------------
# Agent identity cannot affect corroboration or scoring
# ---------------------------------------------------------------------------


def _reachable(model: type[BaseModel]) -> set[type[BaseModel]]:
    found: set[type[BaseModel]] = {model}
    for field in model.model_fields.values():
        for candidate in (field.annotation, *getattr(field.annotation, "__args__", ())):
            if (
                isinstance(candidate, type)
                and issubclass(candidate, BaseModel)
                and candidate not in found
            ):
                found |= _reachable(candidate)
    return found


@pytest.mark.contract
def test_a_resolved_binding_carries_no_agent_identity() -> None:
    """The coupling engine's only input type. It cannot see an agent, so it cannot count one."""
    from app.domain.value_objects.provenance import ModelProvenance

    reachable = _reachable(ResolvedBinding)
    assert ModelProvenance not in reachable

    fields = {name for model in reachable for name in model.model_fields}
    offenders = {
        field
        for field in fields
        if any(token in field for token in ("agent", "prompt", "provider", "confidence"))
    }
    assert not offenders, offenders


@pytest.mark.contract
def test_a_class_a_finding_carries_no_model_attribution() -> None:
    """A derivation had no model, so it must have no ModelProvenance path."""
    from app.domain.value_objects.provenance import ModelProvenance

    assert ModelProvenance not in _reachable(ClassAFinding)


@pytest.mark.contract
def test_a_derived_relation_carries_no_model_attribution() -> None:
    from app.domain.value_objects.provenance import ModelProvenance

    assert ModelProvenance not in _reachable(DerivedRelation)


@pytest.mark.contract
def test_the_decay_report_carries_no_model_attribution() -> None:
    from app.domain.value_objects.provenance import ModelProvenance

    assert ModelProvenance not in _reachable(KnowledgeDecayReport)


@pytest.mark.contract
def test_adding_more_agents_cannot_change_a_class_a_finding() -> None:
    """Several agents reading the same evidence are one source (checkpoint 9.3).

    Demonstrated structurally: the finding's inputs are assumption records, and there is no
    field through which an agent count could be supplied.
    """
    sample = class_a_finding()
    for field in ("agent_sources", "agents", "agent_count", "votes", "produced_by"):
        with pytest.raises(ValidationError):
            ClassAFinding(**{**sample.model_dump(), field: ("a", "b")})


# ---------------------------------------------------------------------------
# Model confidence cannot enter Class A scoring
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_no_class_a_entity_accepts_a_confidence_number() -> None:
    from tests.fixtures.coupling import coupling_strength, decay_entry, knowledge_decay_report

    samples = [
        class_a_finding(),
        coupling_strength(),
        decay_entry(),
        knowledge_decay_report(),
        binding(),
    ]
    for sample in samples:
        with pytest.raises(ValidationError):
            type(sample)(**{**sample.model_dump(), "confidence": 0.9})


@pytest.mark.contract
def test_coupling_strength_is_the_only_number_the_resolver_emits() -> None:
    """And it is computed from evidence properties, with its inputs persisted."""
    strength = CouplingStrength(
        strength=0.5,
        factor_inputs={"role_composition": 1.0},
        strength_fn_version="strength-1",
    )
    assert strength.factor_inputs
    assert strength.strength_fn_version
    assert "confidence" not in CouplingStrength.model_fields


# ---------------------------------------------------------------------------
# Evidence cannot be fabricated
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.parametrize("bad", ["EV-12", "evidence_3", "ev_123", "", "ev_" + "z" * 32])
def test_a_fabricated_evidence_id_fails_at_the_schema_boundary(bad: str) -> None:
    """A made-up citation is a parse error, not a detection problem."""
    sample = class_a_finding()
    with pytest.raises(ValidationError):
        ClassAFinding(**{**sample.model_dump(), "materialized_evidence_ids": (bad,)})


@pytest.mark.contract
def test_a_class_a_finding_cannot_have_an_empty_evidence_set() -> None:
    """A derivation nobody can check must not be assertable."""
    sample = class_a_finding()
    with pytest.raises(ValidationError):
        ClassAFinding(**{**sample.model_dump(), "materialized_evidence_ids": ()})


@pytest.mark.contract
def test_a_derived_relation_cannot_be_given_direct_evidence() -> None:
    """Its support is its rule and its inputs; a citation here would be fabricated."""
    from tests.fixtures.domain import derived_relation

    sample = derived_relation()
    with pytest.raises(ValidationError):
        DerivedRelation(**{**sample.model_dump(), "evidence_ids": ("ev_" + "1" * 32,)})


@pytest.mark.contract
def test_a_derived_relation_cannot_claim_span_support() -> None:
    from tests.fixtures.domain import derived_relation

    sample = derived_relation()
    with pytest.raises(ValidationError):
        DerivedRelation(**{**sample.model_dump(), "support": Support.VERIFIED_SPAN})


@pytest.mark.contract
def test_a_derived_relation_cannot_claim_an_inferred_derivation() -> None:
    """`derived` and `inferred` are separate values; collapsing them destroys the audit."""
    from tests.fixtures.domain import derived_relation

    sample = derived_relation()
    with pytest.raises(ValidationError):
        DerivedRelation(**{**sample.model_dump(), "derivation": Derivation.INFERRED})


@pytest.mark.contract
def test_a_class_a_finding_flags_its_evidence_as_a_derived_union() -> None:
    """It must be impossible to read the materialized set as direct support."""
    sample = class_a_finding()
    assert sample.evidence_is_derived_union is True
    with pytest.raises(ValidationError):
        ClassAFinding(**{**sample.model_dump(), "evidence_is_derived_union": False})


# ---------------------------------------------------------------------------
# Only admitted bindings can be coupled
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.parametrize(
    "flags",
    [
        {"span_verified": False},
        {"value_parsed": False},
        {"key_resolved": False},
    ],
    ids=["span_verified", "value_parsed", "key_resolved"],
)
def test_an_unadmitted_binding_cannot_enter_the_index(flags: dict[str, bool]) -> None:
    """A Class A finding derived from unverified input is the one thing Class A forbids."""
    with pytest.raises(ValidationError):
        binding(
            span_verified=flags.get("span_verified", True),
            value_parsed=flags.get("value_parsed", True),
            key_resolved=flags.get("key_resolved", True),
        )


@pytest.mark.contract
def test_a_quarantined_key_cannot_reach_a_finding() -> None:
    """The registry is closed: an unregistered key has no path into coupling.

    Asserted at the coupling layer: the index only ever sees admitted records, and
    admission requires registry resolution, so there is no unregistered key to find here.
    """
    from app.assumptions.coupling import build_coupling_index

    index = build_coupling_index(
        [binding(value=days(90))],
        as_of=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        registry_version="registry-1",
    )
    for held in index.bindings:
        assert held.record.key_resolved is True
        assert held.record.is_admitted


# ---------------------------------------------------------------------------
# Graph admissibility
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_every_relationship_the_resolver_emits_is_admissible_as_a_derived_edge() -> None:
    """A finding must be projectable without relabelling its derivation."""
    for relationship in (
        Relationship.CONFLICTS_WITH,
        Relationship.UNAUTHORIZED_OVERRIDE,
        Relationship.INVALIDATES_ASSUMPTION,
    ):
        assert Derivation.DERIVED in ADMISSIBLE_DERIVATIONS[relationship]


@pytest.mark.contract
def test_no_relationship_admits_an_absent_derivation() -> None:
    """An EvidenceGap can never become an edge: absence is not a relationship."""
    for relationship, admissible in ADMISSIBLE_DERIVATIONS.items():
        assert Derivation.ABSENT not in admissible, relationship


@pytest.mark.contract
def test_a_class_a_finding_cannot_report_a_hypothesised_relationship() -> None:
    sample = class_a_finding()
    for relationship in (Relationship.CREATES_RISK_FOR, Relationship.HISTORICALLY_ANALOGOUS_TO):
        with pytest.raises(ValidationError):
            ClassAFinding(**{**sample.model_dump(), "relationship": relationship.value})


@pytest.mark.contract
def test_a_class_a_finding_cannot_report_a_projection_only_relationship() -> None:
    sample = class_a_finding()
    for relationship in (Relationship.BINDS_ASSUMPTION, Relationship.DIRECTLY_CHANGES):
        with pytest.raises(ValidationError):
            ClassAFinding(**{**sample.model_dump(), "relationship": relationship.value})


# ---------------------------------------------------------------------------
# The index is a projection, not a second source of truth
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_the_coupling_index_store_has_no_per_record_mutator() -> None:
    """A writeable index would be a divergence bug (checkpoint 10.1)."""
    from app.domain.interfaces.repositories import CouplingIndexStore

    forbidden = {"put_edge", "update_outcome", "add_relation", "set_finding", "patch"}
    assert not (forbidden & set(dir(CouplingIndexStore)))


@pytest.mark.contract
def test_writing_the_index_replaces_rather_than_merges() -> None:
    """A merge would leave a stale cell producing a finding no document implies."""
    from app.infrastructure.database.repositories.memory import InMemoryCouplingIndexStore

    store = InMemoryCouplingIndexStore()
    first = (class_a_finding(),)
    store.put_findings("tn_" + "0" * 32, first)
    store.put_findings("tn_" + "0" * 32, ())
    assert store.findings("tn_" + "0" * 32) == ()
