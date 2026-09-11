"""Contract tests over the shipped agent definitions and their runtime bindings.

These guard the specification decisions that are easy to regress by copy-paste, and that
would silently degrade output quality rather than raise an error.
"""

from __future__ import annotations

import pytest

from app.agents.runtime.agent_loader import AgentDefinition, load_agent_definitions
from app.config.agent_bindings import AgentBindings
from app.config.settings import REPO_ROOT
from app.domain.value_objects.tiers import CapabilityTier

EXPECTED_AGENTS = {
    # Ingestion-time. Its prompt lives here like every other, because principle 12 admits
    # no exception and because the loader's content hash is what stamps the prompt identity
    # onto every proposal, gap and quarantine record it produces.
    "assumption-extractor",
    "change-decomposer",
    "policy-analyst",
    "process-analyst",
    "technology-analyst",
    "people-analyst",
    "data-analyst",
    "risk-analyst",
    "historical-analyst",
    "consequence-hunter",
    "contradiction-hunter",
    "adversarial-challenger",
    "blind-spot-hunter",
    "evidence-arbiter",
    "consequence-judge",
    "report-composer",
}


@pytest.fixture(scope="module")
def definitions() -> dict[str, AgentDefinition]:
    return load_agent_definitions()


@pytest.fixture(scope="module")
def shipped_bindings() -> AgentBindings:
    return AgentBindings.load(REPO_ROOT / "app" / "config" / "agents.yaml")


@pytest.mark.contract
def test_all_specified_agents_are_present(definitions: dict[str, AgentDefinition]) -> None:
    assert set(definitions) == EXPECTED_AGENTS


@pytest.mark.contract
def test_every_agent_has_output_discipline(definitions: dict[str, AgentDefinition]) -> None:
    missing = [
        name
        for name, definition in definitions.items()
        if "## Output discipline" not in definition.prompt
    ]
    assert not missing, f"agents without an output discipline section: {missing}"


@pytest.mark.contract
def test_change_decomposer_does_not_demand_evidence_ids(
    definitions: dict[str, AgentDefinition],
) -> None:
    """The decomposer runs before retrieval, so it has no evidence IDs to cite.

    The original definitions carried a copy-pasted "every material claim must cite
    evidence IDs" instruction here, which would induce fabricated citations - the exact
    failure this project exists to prevent. This test stops that regressing.
    """
    prompt = definitions["change-decomposer"].prompt
    assert "no evidence IDs" in prompt
    assert "Every material claim must cite evidence IDs" not in prompt


@pytest.mark.contract
def test_challenger_carries_a_citation_obligation(definitions: dict[str, AgentDefinition]) -> None:
    """An uncited objection must not be expressible (docs/IDEA.md 15.2 step 6)."""
    prompt = definitions["adversarial-challenger"].prompt
    assert "no_refutation_found" in prompt
    assert "Citation obligation" in prompt


@pytest.mark.contract
def test_blind_spot_hunter_is_defined_as_a_partition(
    definitions: dict[str, AgentDefinition],
) -> None:
    """Diversity comes from a disjoint evidence basis, not from a persona."""
    prompt = definitions["blind-spot-hunter"].prompt
    assert "partition" in prompt.lower()


@pytest.mark.contract
def test_no_agent_is_asked_for_a_confidence_number(definitions: dict[str, AgentDefinition]) -> None:
    """Confidence is deterministic, versioned code (CLAUDE.md principle 15)."""
    offenders = [
        name
        for name, definition in definitions.items()
        if "Do not emit a confidence number" not in definition.prompt
        and "Do not emit confidence numbers" not in definition.prompt
    ]
    assert not offenders, f"agents not told to leave confidence to the scorer: {offenders}"


@pytest.mark.contract
def test_every_agent_has_a_runtime_binding(
    definitions: dict[str, AgentDefinition], shipped_bindings: AgentBindings
) -> None:
    unbound = [name for name in definitions if name not in shipped_bindings.bindings]
    assert not unbound, f"agents with no explicit binding in agents.yaml: {unbound}"


@pytest.mark.contract
def test_no_binding_refers_to_a_nonexistent_agent(
    definitions: dict[str, AgentDefinition], shipped_bindings: AgentBindings
) -> None:
    stray = set(shipped_bindings.bindings) - set(definitions)
    assert not stray, f"agents.yaml binds unknown agents: {sorted(stray)}"


@pytest.mark.contract
def test_assumption_extraction_is_bound_to_the_fast_tier(
    shipped_bindings: AgentBindings,
) -> None:
    """Extraction runs over the entire corpus; a reasoning tier here dominates cost.

    See CLAUDE.md, cost discipline.
    """
    binding = shipped_bindings.for_agent("assumption-extractor")
    assert binding.tier is CapabilityTier.FAST


@pytest.mark.contract
def test_adversarial_and_judging_agents_get_the_reasoning_tier(
    shipped_bindings: AgentBindings,
) -> None:
    """Weak models agree; falsification and arbitration must not be cheap."""
    for agent in ("adversarial-challenger", "consequence-judge", "consequence-hunter"):
        assert shipped_bindings.for_agent(agent).tier is CapabilityTier.REASONING, agent


@pytest.mark.contract
def test_bindings_never_hardcode_a_model_name(shipped_bindings: AgentBindings) -> None:
    for name, binding in shipped_bindings.bindings.items():
        assert not hasattr(binding, "model") or getattr(binding, "model", None) is None, name
        if binding.model_env is not None:
            assert binding.model_env.startswith("CCE_"), name
