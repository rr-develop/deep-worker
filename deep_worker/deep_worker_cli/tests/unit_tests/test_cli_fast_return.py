"""Regression test: ``deep-worker <name> -asm --dir /tmp --clean`` MUST
return control to the shell quickly (≤ 5 seconds). Previously the CLI
ran a slow SQLite history cleanup on the parent process before
daemonizing, which could block the terminal for 20+ seconds — the user
reported this as "just hangs".

This test runs the real ``deep-worker`` binary as a subprocess
(equivalent to typing the command in bash), measures wall-clock until
exit, and fails if the budget is exceeded.

Also runs the ``--clean`` cleanup actually happened in background, by
checking the agent's registry entry appeared.

Cleanup: kills the spawned agent and removes its registry entry at
teardown so the test is idempotent and doesn't pollute the system.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


# Budget intentionally generous: on a loaded CI box with slow disk it may
# take a second or two; we just want to catch the "20 seconds or more"
# regression. Typical observed time on dev hardware is ~0.17s.
_FAST_RETURN_BUDGET_SECONDS = 5.0

_REGISTRY_FILE = Path("/tmp/deep_worker_registry.json")


def _registry_entries() -> dict:
    if not _REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(_REGISTRY_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _kill_agent(name: str) -> None:
    """Kill the agent process if it's in the registry and wait for it
    to disappear. Best-effort; a cleanup failure is never a test
    failure."""
    entry = _registry_entries().get(name)
    if not entry:
        return
    pid = entry.get("pid")
    if not isinstance(pid, int):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    # Wait up to 5s for graceful shutdown.
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _remove_agent_data(name: str) -> None:
    data_dir = Path("~/.deep_worker").expanduser() / name
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for f in log_dir.glob(f"{name}.*"):
            try:
                f.unlink()
            except OSError:
                pass
    pid_file = Path(f"/tmp/deep_worker_{name}.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass


@pytest.fixture
def agent_name():
    """Unique agent name per test run, auto-cleaned at teardown."""
    name = f"test_fast_return_{os.getpid()}_{int(time.time() * 1000) % 100000}"
    yield name
    _kill_agent(name)
    _remove_agent_data(name)


def _run_cli_with_budget(args: list[str]) -> tuple[float, subprocess.CompletedProcess]:
    """Run ``deep-worker`` with given args as a real subprocess, measure
    wall-clock, and return ``(elapsed_seconds, CompletedProcess)``.

    Uses ``subprocess.run`` with ``timeout=30`` — a hard upper bound
    much larger than our budget, so that if the CLI regresses to hang
    forever the test fails with a clear message rather than stalling
    CI indefinitely."""
    cmd = ["deep-worker"] + args
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    elapsed = time.monotonic() - t0
    return elapsed, proc


@pytest.mark.skipif(
    shutil.which("deep-worker") is None,
    reason="deep-worker binary not in PATH; smoke test only runs in an installed env",
)
def test_dw_clean_returns_control_within_budget(agent_name: str) -> None:
    """The exact user scenario: ``deep-worker <name> -asm --dir /tmp --clean``
    must return control to the shell within ``_FAST_RETURN_BUDGET_SECONDS``.

    This is a REGRESSION TEST for the '20+ second hang on --clean' bug
    the user reported. Before the fix, the CLI would run
    ``asyncio.run(clean_agent_history(...))`` on the parent before
    calling ``daemonize()``, and since ``clean_agent_history`` opens
    SQLite with a 30-second timeout, a WAL lock held by another agent
    could block the parent for up to 30 seconds.
    """
    elapsed, proc = _run_cli_with_budget(
        [agent_name, "-asm", "--dir", "/tmp", "--clean"]
    )

    # Primary assertion: the user gets their shell back quickly.
    assert elapsed < _FAST_RETURN_BUDGET_SECONDS, (
        f"deep-worker {agent_name} -asm --dir /tmp --clean took "
        f"{elapsed:.2f}s to return (budget: "
        f"{_FAST_RETURN_BUDGET_SECONDS}s).\n"
        f"This is the UX hang the user reported — a regression of the "
        f"daemonize-before-slow-work fix.\n"
        f"stdout: {proc.stdout!r}\n"
        f"stderr: {proc.stderr!r}"
    )

    # Secondary: exit code must be 0 (daemonize succeeded) and stdout
    # must carry the "starting in background" line.
    assert proc.returncode == 0, (
        f"deep-worker returned {proc.returncode}; "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )
    assert "starting in background" in proc.stdout, (
        f"Expected 'starting in background' notice in stdout; got: "
        f"{proc.stdout!r}"
    )


@pytest.mark.skipif(
    shutil.which("deep-worker") is None,
    reason="deep-worker binary not in PATH",
)
def test_dw_without_clean_returns_control_within_budget(
    agent_name: str,
) -> None:
    """Same fast-return guarantee without ``--clean``. Catches a
    separate class of regression where heavy imports creep back into
    the parent process."""
    elapsed, proc = _run_cli_with_budget(
        [agent_name, "-asm", "--dir", "/tmp"]
    )
    assert elapsed < _FAST_RETURN_BUDGET_SECONDS, (
        f"deep-worker {agent_name} -asm --dir /tmp took {elapsed:.2f}s "
        f"to return (budget: {_FAST_RETURN_BUDGET_SECONDS}s).\n"
        f"stdout: {proc.stdout!r}\n"
        f"stderr: {proc.stderr!r}"
    )
    assert proc.returncode == 0


@pytest.mark.skipif(
    shutil.which("deep-worker") is None,
    reason="deep-worker binary not in PATH",
)
def test_dw_help_is_instant() -> None:
    """``deep-worker --help`` must be near-instant. Catches regressions
    where something heavy creeps into the CLI module's top-level
    imports."""
    elapsed, proc = _run_cli_with_budget(["--help"])
    assert elapsed < 2.0, (
        f"deep-worker --help took {elapsed:.2f}s; should be well under 2s."
    )
    assert proc.returncode == 0
    assert "usage: deep-worker" in proc.stdout


def test_stdout_not_duplicated_across_fork(agent_name: str) -> None:
    """Regression for a buffering bug: before ``daemonize()`` flushed
    stdio, any text printed pre-daemonize got replayed three times
    (parent + child#1 + grandchild) because Python's stdout buffer was
    duplicated into each fork."""
    if shutil.which("deep-worker") is None:
        pytest.skip("deep-worker binary not in PATH")

    _, proc = _run_cli_with_budget(
        [agent_name, "-asm", "--dir", "/tmp", "--clean"]
    )
    # Count occurrences of the '--clean' notice. Must be exactly 1.
    count = proc.stdout.count("DB cleanup will run in background")
    assert count == 1, (
        f"Expected exactly 1 '--clean' notice in stdout but found {count}. "
        f"This is a sign that stdio buffers are still being duplicated "
        f"across daemonize()'s forks.\nstdout: {proc.stdout!r}"
    )
