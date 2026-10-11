"""The migration's whole point: an agent must not carry the memory stack (§14).

Before the shared service, offering ``project_memory`` or
``history_memory`` meant importing llama-index, LanceDB and — under them —
sentence-transformers, into every agent process. That is hundreds of
megabytes of RSS per agent for a capability most turns never use.

Two things this file had to get right, both learned the hard way.

*The measurement must be differential.* ``langchain.agents`` itself
imports ``torch`` and ``transformers`` on this platform, before any of
our code runs. An absolute "torch must not be in sys.modules" assertion
therefore fails for a reason the service can do nothing about, and it
failed exactly that way when first written. What the migration actually
promises is that *building an agent adds nothing on top of the framework
baseline* — so the probe measures the baseline first, then builds the
agent, and asserts the delta is empty. ``llama_index`` and ``lancedb``
are checked absolutely too: those are ours, nothing else pulls them, and
their presence would be unambiguous.

*The proof must be two-sided.* ``sys.modules`` names what the interpreter
believes it imported; ``/proc/<pid>/maps`` names the shared objects
really mapped in. A module can be listed without its native library
mapped, and a mapping can survive a module table someone tampered with.
The heavy-stack check reads both.

The agent is built through the real ``create_cli_agent`` with history
memory enabled — the configuration that used to import the indexer
eagerly. The service is deliberately NOT started: registering a tool must
not require one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .conftest import BUDGET_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(BUDGET_S)]

#: Ours: nothing but the memory stack drags these in, so their presence
#: in an agent process is proof the migration regressed.
_OURS = ("llama_index", "lancedb", "sentence_transformers")

#: The framework's: langchain imports these regardless of our code, so
#: they are only meaningful as a delta against the baseline.
_FRAMEWORK = ("torch", "transformers")

#: Mapped shared objects that betray a real torch load. Also measured as
#: a delta — langchain's torch maps them before we build anything.
_MAPPINGS = ("libtorch", "libc10", "libtorch_cpu")

_PROBE = textwrap.dedent(
    """
    import json, os, sys

    def snapshot():
        watched = OURS + FRAMEWORK
        modules = {m.split(".")[0] for m in sys.modules if m.split(".")[0] in watched}
        try:
            with open(f"/proc/{os.getpid()}/maps") as fh:
                text = fh.read()
            mappings = {name for name in MAPPINGS if name in text}
        except OSError:
            mappings = {"<maps unreadable>"}
        return modules, mappings

    # Baseline: the framework alone, before any deep_worker import.
    from langchain.agents import create_agent  # noqa: F401
    base_modules, base_mappings = snapshot()

    # Now the thing under test.
    from deep_worker_cli.agent import create_cli_agent

    agent, backend = create_cli_agent(
        model="claude-sonnet-5",
        assistant_id="torch_probe_agent",
        enable_history_memory=True,
        project_memory_dir=Path(os.getcwd()),
        enable_shell=False,
        enable_skills=False,
        checkpointer=None,
    )

    after_modules, after_mappings = snapshot()

    # Tools live in the compiled graph's "tools" node, not on the agent.
    # A probe that silently registered nothing would "prove" the absence
    # of the heavy stack by proving the absence of the feature.
    node = agent.get_graph().nodes.get("tools")
    tool_names = sorted(
        getattr(getattr(node, "data", None), "tools_by_name", {}) or {}
    )

    print("PROBE " + json.dumps({
        "tools": tool_names,
        "ours_loaded": sorted(m for m in after_modules if m in OURS),
        "added_modules": sorted(after_modules - base_modules),
        "added_mappings": sorted(after_mappings - base_mappings),
        "baseline_modules": sorted(base_modules),
    }))
    """
)


def _probe_env(tree_root: Path) -> dict[str, str]:
    """Environment that resolves the packages under test, not another tree.

    The operator's PYTHONPATH may point at a different worktree; a probe
    that imported *that* tree's agent would report on code nobody is
    reviewing.
    """
    env = dict(os.environ)
    roots = [
        str(tree_root / "deep_worker" / "deep_worker_cli"),
        str(tree_root / "deep_worker" / "deepagents"),
        str(tree_root / "deep_worker"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        roots + [p for p in existing.split(os.pathsep) if p]
    )
    return env


def _run_probe(tree_root: Path) -> dict[str, list[str]]:
    """Build an agent in a fresh interpreter and report what it added."""
    preamble = (
        "from pathlib import Path\n"
        f"OURS = {_OURS!r}\n"
        f"FRAMEWORK = {_FRAMEWORK!r}\n"
        f"MAPPINGS = {_MAPPINGS!r}\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", preamble + _PROBE],
        capture_output=True,
        text=True,
        timeout=BUDGET_S - 10,
        cwd=str(tree_root),
        env=_probe_env(tree_root),
    )
    assert proc.returncode == 0, (
        f"agent probe failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )

    for line in proc.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[len("PROBE ") :])
    raise AssertionError(f"probe printed no result:\n{proc.stdout}\n{proc.stderr}")


def test_agent_with_memory_tools_loads_no_heavy_stack() -> None:
    """An agent offering both memory tools carries none of their weight."""
    tree_root = Path(__file__).resolve().parents[5]

    report = _run_probe(tree_root)

    assert "history_memory" in report["tools"], (
        f"history_memory was not registered; tools={report['tools']}"
    )
    assert "project_memory" in report["tools"], (
        f"project_memory was not registered; tools={report['tools']}"
    )

    assert not report["ours_loaded"], (
        "agent process imported the memory stack: "
        f"{report['ours_loaded']} — the service exists so it would not"
    )
    assert not report["added_modules"], (
        "building the agent imported modules the framework had not: "
        f"{report['added_modules']} (baseline: {report['baseline_modules']})"
    )
    assert not report["added_mappings"], (
        "building the agent mapped native libraries the framework had not: "
        f"{report['added_mappings']}"
    )


def test_probe_detects_the_stack_when_it_is_loaded() -> None:
    """The detector is not vacuously green.

    Without this, a probe that mis-spelled every module name would pass
    the test above forever. Here the child imports one of *our* modules on
    purpose and must be caught by the same snapshot logic.
    """
    preamble = (
        f"OURS = {_OURS!r}\n"
        f"FRAMEWORK = {_FRAMEWORK!r}\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            preamble
            + textwrap.dedent(
                """
                import json, sys
                import llama_index.core  # the thing the service now owns
                watched = OURS + FRAMEWORK
                loaded = sorted({
                    m.split(".")[0] for m in sys.modules
                    if m.split(".")[0] in watched
                })
                print("PROBE " + json.dumps({"modules": loaded}))
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=BUDGET_S - 10,
    )
    if proc.returncode != 0:
        pytest.skip("llama_index is not installed in this environment")

    payload = next(
        json.loads(line[len("PROBE ") :])
        for line in proc.stdout.splitlines()
        if line.startswith("PROBE ")
    )
    assert "llama_index" in payload["modules"], (
        "the detector missed a module that was definitely imported — "
        "the negative result in the test above would be meaningless"
    )
