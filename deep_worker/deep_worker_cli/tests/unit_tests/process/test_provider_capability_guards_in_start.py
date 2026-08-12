"""Structural guards for the provider capability guards in
``AgentProcess.start()`` codex/grok passthrough blocks (B2).

Capability-guard contract: the codex (``NATIVE_TOOLS``) and grok
(``build_passthrough_tools``) wiring inside ``AgentProcess.start()`` must
resolve the vendored provider through
``provider_paths.ensure_provider_importable`` and raise ``CapabilityError``
when unavailable — instead of a raw ``from deep_worker.providers...``
import that would surface ``ModuleNotFoundError`` on a wheel without the
provider installed.

A full behavioral test would require standing up a real AgentProcess
(start() daemon-forks, opens PID files, initializes sessions, spawns
rpyc servers) — infeasible in a unit test. These are textual guards over
the same surface protected by ``test_agent_process_deepagents_path.py``:
the codex and grok blocks must resolve before importing, and must not
fall back to sys.path hacks.
"""

from __future__ import annotations

from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_AGENT_PROCESS = (
    _THIS_FILE.parent.parent.parent.parent  # → deep_worker_cli/
    / "deep_worker_cli" / "process" / "agent_process.py"
)


def test_agent_process_exists() -> None:
    assert _AGENT_PROCESS.exists(), f"not found: {_AGENT_PROCESS}"


def test_codex_block_resolves_before_importing_native_tools() -> None:
    """The codex passthrough block MUST call ``ensure_provider_importable``
    and guard on ``resolution.unavailable`` BEFORE the
    ``from deep_worker.providers.codex_provider.native_tools import NATIVE_TOOLS``
    import, and raise CapabilityError on failure."""
    text = _AGENT_PROCESS.read_text(encoding="utf-8")
    # Resolve call happens before the import (line order in source).
    resolve_idx = text.index('ensure_provider_importable(\n                        "deep_worker.providers.codex_provider"')
    import_idx = text.index("from deep_worker.providers.codex_provider.native_tools import NATIVE_TOOLS")
    assert resolve_idx < import_idx, (
        "codex provider must be resolved via ensure_provider_importable "
        "BEFORE importing NATIVE_TOOLS"
    )
    # The guard between resolve and import.
    between = text[resolve_idx:import_idx]
    assert "resolution.unavailable" in between, (
        "codex block must check resolution.unavailable"
    )
    assert "CapabilityError" in between, (
        "codex block must raise CapabilityError when unavailable"
    )


def test_grok_block_resolves_before_importing_passthrough() -> None:
    """The grok passthrough block MUST call ``ensure_provider_importable``
    and guard on ``resolution.unavailable`` BEFORE the
    ``from deep_worker.providers.grok_provider.builtin_tool_passthrough import ...``
    import, and raise CapabilityError on failure."""
    text = _AGENT_PROCESS.read_text(encoding="utf-8")
    resolve_idx = text.index('ensure_provider_importable(\n                        "deep_worker.providers.grok_provider"')
    import_idx = text.index(
        "from deep_worker.providers.grok_provider.builtin_tool_passthrough import"
    )
    assert resolve_idx < import_idx, (
        "grok provider must be resolved via ensure_provider_importable "
        "BEFORE importing build_passthrough_tools"
    )
    between = text[resolve_idx:import_idx]
    assert "resolution.unavailable" in between, (
        "grok block must check resolution.unavailable"
    )
    assert "CapabilityError" in between, (
        "grok block must raise CapabilityError when unavailable"
    )


def test_no_sys_path_hack_in_codex_grok_blocks() -> None:
    """B2 removed the sys.path hacks that used to make the vendored
    providers importable. Neither block may MUTATE sys.path (comments
    may mention it — provider_paths itself manages sys.path in source
    mode, but the call sites must not)."""
    text = _AGENT_PROCESS.read_text(encoding="utf-8")
    # Find the region spanning both blocks (from the codex resolve to the
    # end of the grok build_passthrough_tools() call).
    start_idx = text.index('ensure_provider_importable(\n                        "deep_worker.providers.codex_provider"')
    end_idx = text.index("web_tools.extend(build_passthrough_tools())")
    region = text[start_idx:end_idx]
    for mutation in ("sys.path.insert", "sys.path.append", "sys.path[0:0]", "sys.path += "):
        assert mutation not in region, (
            f"codex/grok blocks must NOT mutate sys.path via {mutation!r} "
            f"(B2 removes the hacks)"
        )
