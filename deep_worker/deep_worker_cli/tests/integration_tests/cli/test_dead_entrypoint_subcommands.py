"""Integration test for CLI subcommand wiring — on the REAL binary.

Context
-------
The ``mcp``/``skills``/``threads`` subcommands were registered ONLY in
``deep_worker_cli/main.py`` — a dead entry point. The real launcher runs
``python3 -m deep_worker_cli.cli.agent_command``, so the commands did not
work:

    $ deep-worker mcp list
    deep-worker: error: unrecognized arguments: list

The same defect previously affected ``lsp`` (fixed in an earlier commit).
It is only caught by running the real binary: unit tests that drive the
parser directly pass even with the broken wiring.

Policy "integration tests must not hang": a hard timeout on each run, and
the process is guaranteed to be killed in ``finally``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# tests/integration_tests/cli/ → tests/integration_tests → tests → package
_PKG_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER = _PKG_ROOT / "deep-worker"

_CMD_BUDGET_SEC = 60.0

_ARGPARSE_REJECTIONS = ("unrecognized arguments", "invalid choice")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    assert _LAUNCHER.is_file(), f"launcher not found: {_LAUNCHER}"

    env = {**os.environ}
    env.pop("DEEP_WORKER_DEV_TREE", None)
    env.pop("DEEP_WORKER_TREE_ROOT", None)

    proc = subprocess.Popen(
        ["bash", str(_LAUNCHER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=_CMD_BUDGET_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate(timeout=10)
        pytest.fail(
            f"'deep-worker {' '.join(args)}' exceeded {_CMD_BUDGET_SEC}s — "
            f"treat as a hang, not a flake.\nstdout={out}\nstderr={err}"
        )
    finally:
        if proc.poll() is None:  # pragma: no cover — defensive
            proc.kill()
            proc.wait(timeout=10)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def _assert_not_rejected_by_argparse(
    res: subprocess.CompletedProcess[str], what: str
) -> None:
    combined = res.stdout + res.stderr
    for marker in _ARGPARSE_REJECTIONS:
        assert marker not in combined, f"{what}: argparse rejected it → {combined}"


def test_mcp_list_runs_via_real_binary() -> None:
    """``deep-worker mcp list`` must work, not fail with
    ``unrecognized arguments: list``."""
    res = _run(["mcp", "list"])
    _assert_not_rejected_by_argparse(res, "mcp list")
    assert res.returncode == 0, res.stdout + res.stderr


def test_mcp_appears_in_help() -> None:
    res = _run(["--help"])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "mcp" in res.stdout, res.stdout


def test_skills_list_runs_via_real_binary() -> None:
    res = _run(["skills", "list"])
    _assert_not_rejected_by_argparse(res, "skills list")


def test_threads_list_runs_via_real_binary() -> None:
    res = _run(["threads", "list"])
    _assert_not_rejected_by_argparse(res, "threads list")


def test_lsp_still_works() -> None:
    """A previously fixed command must not be broken by the refactor."""
    res = _run(["lsp", "status"])
    _assert_not_rejected_by_argparse(res, "lsp status")
    assert res.returncode == 0, res.stdout + res.stderr


def test_normal_agent_launch_still_parses() -> None:
    """REGRESSION LOCK.

    The subcommand dispatcher must not break normal agent launch: the main
    parser has a positional ``name`` (``nargs='?'``).
    """
    res = _run(["--list", "some_agent_name"])
    _assert_not_rejected_by_argparse(res, "--list some_agent_name")
    combined = res.stdout + res.stderr
    assert "Cannot specify agent name" in combined, combined


@pytest.mark.parametrize("name", ["mcpagent", "skillsbot", "threadsrunner", "lspagent"])
def test_agent_names_resembling_subcommands_are_not_swallowed(name: str) -> None:
    """An agent name that merely STARTS like a subcommand must not be
    intercepted: the dispatcher must compare the token IN FULL."""
    res = _run(["--list", name])
    combined = res.stdout + res.stderr
    assert "Cannot specify agent name" in combined, combined
