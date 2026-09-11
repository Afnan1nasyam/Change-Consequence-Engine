"""Print agent -> tier -> provider/model resolution.

Answers "why is this agent not running" without spending a token: resolution is pure
configuration. Never prints a credential, and prints a model identifier only because it
came from the operator's own environment.

    python -m scripts.show_agents
"""

from __future__ import annotations

import sys

from app.agents.runtime.agent_loader import load_agent_definitions
from app.api.deps import get_agent_bindings, get_gateway
from app.config.settings import get_settings
from app.domain.interfaces.llm import ProviderError


def main() -> int:
    settings = get_settings()
    definitions = load_agent_definitions()
    bindings = get_agent_bindings()
    gateway = get_gateway()

    print(f"default provider : {settings.llm_default_provider}")
    print(f"enabled providers: {', '.join(settings.llm_enabled_providers)}")
    print(f"binding config   : {settings.agents_config_path}")
    print()

    header = f"{'agent':<26} {'tier':<10} {'provider':<10} {'model':<28} status"
    print(header)
    print("-" * len(header))

    unresolved: list[tuple[str, str]] = []
    for name in sorted(definitions):
        tier = bindings.for_agent(name).tier.value
        try:
            resolved = gateway.resolve(name)
        except ProviderError as exc:
            print(f"{name:<26} {tier:<10} {'-':<10} {'-':<28} NOT CONFIGURED")
            unresolved.append((name, str(exc)))
        else:
            print(
                f"{resolved.agent:<26} {resolved.tier:<10} "
                f"{resolved.provider:<10} {resolved.model:<28} ok"
            )

    if unresolved:
        print(f"\n{len(unresolved)} agent(s) cannot run:\n")
        # One representative reason is enough; they nearly always share a root cause.
        seen: set[str] = set()
        for name, reason in unresolved:
            if reason in seen:
                continue
            seen.add(reason)
            print(f"  {name}: {reason}")
        return 1

    print("\nall agents resolve to a provider and model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
