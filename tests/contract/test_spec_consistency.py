"""The specification must agree with the domain checkpoint.

``scripts/check_spec_consistency.py`` was previously only a script, which meant the CLEAN
result depended on somebody remembering to run it. ``CLAUDE.md`` requires architectural
rules to be enforced by a test rather than by convention, and documentation drift is exactly
the kind of rule that rots quietly: nothing breaks, the docs simply stop describing the
system.

Running it here makes a drift a failing build.
"""

from __future__ import annotations

import pytest
from scripts.check_spec_consistency import discover, run_checks


@pytest.mark.contract
def test_the_specification_agrees_with_the_checkpoint() -> None:
    issues = run_checks()
    formatted = "\n".join(f"  [{i.rule}] {i.path}: {i.detail}" for i in issues)
    assert not issues, f"specification drift against DOMAIN_CHECKPOINT.md:\n{formatted}"


@pytest.mark.contract
def test_the_checker_actually_reads_the_corpus_of_documents() -> None:
    """A checker that silently found no files would report CLEAN forever."""
    docs, agents = discover()
    assert "CLAUDE.md" in docs
    assert "docs/DOMAIN_CHECKPOINT.md" in docs
    assert len(docs) >= 10
    assert len(agents) >= 15
    assert ".claude/agents/assumption-extractor.md" in agents


@pytest.mark.contract
def test_running_the_checks_twice_gives_the_same_answer() -> None:
    """It holds no module-level state, so it is safe to call from a test and a CLI."""
    assert run_checks() == run_checks()
