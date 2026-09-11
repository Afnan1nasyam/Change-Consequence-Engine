"""Architectural contract: dependency direction.

``CLAUDE.md`` requires ``domain <- application <- infrastructure`` and forbids any provider
SDK or agent framework above the infrastructure layer. That rule is worth nothing as a
convention, so it is asserted here by parsing imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.config.settings import REPO_ROOT

# Vendor SDKs and agent frameworks. A provider is reached only through an infrastructure
# adapter; business logic must not know who served a request.
FORBIDDEN_ROOTS = frozenset(
    {
        "groq",
        "google",
        "google_genai",
        "mistralai",
        "ollama",
        "openai",
        "anthropic",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langgraph",
        "llama_index",
    }
)

# Layers that must stay vendor-free, and the internal packages each may import.
#
# `app/ingestion` and `app/assumptions` hold the deterministic ingestion mechanism. They are
# guarded for the same reason the domain is: the admission gate and the closed registry are
# the trust boundary, and a vendor SDK reaching into them would mean the trusted layer could
# make a network call. They may import the domain and each other's ingestion primitives.
#
# `app/assumptions` may also import `app.agents`: the extractor consumes a prompt definition
# from the agent runtime, which is a pure loader over `.claude/agents/*.md` with no vendor or
# transport dependency of its own. That is how the extractor obtains the prompt hash it
# stamps onto every proposal's provenance.
# `app/retrieval` is guarded too. Lexical retrieval is arithmetic over stored spans, and it
# must stay that way: the moment a retriever can reach a provider, "retrieval before
# reasoning" stops being enforceable and an embedding call could hide inside a filter. When
# vector search lands it reaches its provider through `EmbeddingPort` in `app.domain`, which
# is what this rule leaves open and a direct SDK import does not.
# `app/consequences` is guarded for the reason `app/assumptions` is: it is the gate over an
# agent's output, and a vendor SDK or HTTP client reaching into it would mean the code that
# decides what a model is allowed to have said could itself make a network call. It reaches
# its provider only through `AgentCompletionPort` in `app.domain`, exactly as the extractor
# does, and it may import `app.assumptions` because it reuses the untrusted-content envelope
# rather than writing a second one.
# `app/contradictions` is guarded for the same reason, and one more of its own: it is the
# only agentic layer whose input is a *deterministic Class A conclusion*, so a network call
# reaching into it would put a provider between the resolver and the reader. It may import
# `app.assumptions` because the divergence set it projects is that module's output - and
# notably it may **not** import `app.retrieval`, which is what makes "the contradiction
# hunter does not search the corpus" a property of the layer rather than of its docstring.
# `app/challenge` is guarded for the same reasons and carries the sharpest version of the
# retrieval ban: the adversarial challenger's job is to try to falsify a deterministic
# derivation, so a layer that could reach a provider or a store on its own would be able to
# go looking for the objection it wants. It reviews what it is handed.
# `app/judgment` is the last guarded layer and the one whose output a reader acts on
# directly. It may import the three agentic layers whose findings it arbitrates, and it may
# not import `app.retrieval` - but the sharper rule is the one it does not need a line for:
# the confidence number it attaches comes from `app.domain.services.confidence`, so a
# provider reaching into this layer could not supply one even if it were here.
# `app/graph` is the most tightly guarded of all, and the omissions are the point: it may
# import the domain and the ingestion identity recipes and **nothing else** - not
# `app.agents`, not any of the four agentic layers, not `app.retrieval`. Checkpoint 10.2 says
# no agent writes an edge, and a projector that cannot import an agent runtime cannot be
# handed one. It takes admitted domain records and returns a projection.
GUARDED_LAYERS = {
    "app/domain": {"app.domain"},
    "app/ingestion": {"app.domain", "app.ingestion"},
    "app/assumptions": {"app.domain", "app.assumptions", "app.ingestion", "app.agents"},
    "app/retrieval": {"app.domain", "app.retrieval"},
    "app/consequences": {
        "app.domain",
        "app.consequences",
        "app.assumptions",
        "app.ingestion",
        "app.agents",
    },
    "app/contradictions": {
        "app.domain",
        "app.contradictions",
        "app.assumptions",
        "app.ingestion",
        "app.agents",
    },
    "app/challenge": {
        "app.domain",
        "app.challenge",
        "app.assumptions",
        "app.ingestion",
        "app.agents",
    },
    # `app/blindspots` is the gate over a second Class B agent, so it is guarded for the
    # reason `app/consequences` is. It may import `app.assumptions` because it reuses the
    # untrusted-content envelope and reads the existing decay categories rather than
    # re-deriving what "undocumented" means.
    "app/blindspots": {
        "app.domain",
        "app.blindspots",
        "app.assumptions",
        "app.ingestion",
        "app.agents",
    },
    # `app/specialists` is the gate over seven agents at once, so it is guarded for the reason
    # `app/consequences` is - and for one more of its own: the specialists are the widest
    # agentic surface in the system, and a vendor SDK reaching into the layer that decides
    # what seven models are allowed to have said would be the single worst place for one. It
    # may import `app.assumptions` because it reuses the untrusted-content envelope rather
    # than writing a seventh copy.
    "app/specialists": {
        "app.domain",
        "app.specialists",
        "app.assumptions",
        "app.ingestion",
        "app.agents",
    },
    # `app/reporting` is guarded because it is the last gate before a reader: the code that
    # decides whether generated prose reaches a report must not be able to fetch anything.
    # It has no retriever and no store, and this rule is what keeps it that way.
    "app/reporting": {
        "app.domain",
        "app.reporting",
        "app.assumptions",
        "app.ingestion",
        "app.agents",
    },
    "app/graph": {
        "app.domain",
        "app.graph",
        "app.ingestion",
    },
    "app/judgment": {
        "app.domain",
        "app.judgment",
        "app.assumptions",
        "app.challenge",
        "app.consequences",
        "app.contradictions",
        "app.ingestion",
        "app.agents",
    },
    "app/application": {
        "app.domain",
        "app.application",
        "app.assumptions",
        "app.blindspots",
        "app.challenge",
        "app.consequences",
        "app.contradictions",
        "app.graph",
        "app.ingestion",
        "app.judgment",
        "app.reporting",
        "app.retrieval",
        "app.specialists",
    },
}

# Transport clients belong in infrastructure too: an HTTP client in the domain layer is a
# vendor dependency wearing a disguise.
FORBIDDEN_TRANSPORT = frozenset({"httpx", "requests", "aiohttp", "urllib3"})


def _python_files(relative_dir: str) -> list[Path]:
    root = REPO_ROOT / relative_dir
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py")


def _imported_roots(path: Path) -> set[str]:
    """Top-level module names imported by ``path``, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            # A relative import stays inside its own package, so it cannot be a vendor SDK.
            roots.add(node.module.split(".")[0])
    return roots


def _imported_modules(path: Path) -> set[str]:
    """Fully qualified module names imported by ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return modules


@pytest.mark.contract
@pytest.mark.parametrize("layer", sorted(GUARDED_LAYERS))
def test_layer_imports_no_vendor_sdk(layer: str) -> None:
    offenders: list[str] = []
    for path in _python_files(layer):
        forbidden = _imported_roots(path) & (FORBIDDEN_ROOTS | FORBIDDEN_TRANSPORT)
        if forbidden:
            offenders.append(f"{path.relative_to(REPO_ROOT)} imports {sorted(forbidden)}")

    assert not offenders, (
        "vendor SDK / transport client imported above the infrastructure layer:\n"
        + "\n".join(offenders)
    )


@pytest.mark.contract
@pytest.mark.parametrize("layer,allowed", sorted((k, v) for k, v in GUARDED_LAYERS.items()))
def test_layer_respects_dependency_direction(layer: str, allowed: set[str]) -> None:
    offenders: list[str] = []
    for path in _python_files(layer):
        for module in _imported_modules(path):
            if not module.startswith("app."):
                continue
            # `app.config` is shared configuration, not a layer; it imports nothing upward.
            if module.startswith("app.config"):
                continue
            if not any(module == pkg or module.startswith(pkg + ".") for pkg in allowed):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {module}")

    assert not offenders, (
        f"{layer} may only import {sorted(allowed)} (plus app.config):\n" + "\n".join(offenders)
    )


@pytest.mark.contract
def test_domain_does_not_import_application_or_infrastructure() -> None:
    """The innermost layer depends on nothing of ours but itself."""
    offenders: list[str] = []
    for path in _python_files("app/domain"):
        for module in _imported_modules(path):
            if module.startswith(("app.application", "app.infrastructure", "app.api")):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert not offenders, "domain layer reaches outward:\n" + "\n".join(offenders)


@pytest.mark.contract
def test_no_hardcoded_provider_model_names() -> None:
    """Model identifiers must come from configuration, never from source.

    Guards the rule in ``CLAUDE.md``: vendor inventory changes, and a stale hard-coded
    identifier is an outage. Matches the common vendor naming shapes rather than a
    specific catalogue.
    """
    import re

    # e.g. llama-3.1-70b-versatile, gpt-4o-mini, claude-opus-4-20250514, gemini-1.5-pro
    patterns = [
        re.compile(r"\bllama[-_]?\d"),
        re.compile(r"\bmixtral[-_]?\d"),
        re.compile(r"\bgemma[-_]?\d"),
        re.compile(r"\bgpt-[0-9]"),
        re.compile(r"\bclaude-[a-z]+-[0-9]"),
        re.compile(r"\bgemini-[0-9]"),
        re.compile(r"\bmistral-(large|small|medium)"),
        re.compile(r"\bqwen[-_]?\d"),
        re.compile(r"\bdeepseek-"),
    ]

    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "app").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)!r}")

    assert not offenders, (
        "model identifiers found in source; move them to configuration:\n" + "\n".join(offenders)
    )
