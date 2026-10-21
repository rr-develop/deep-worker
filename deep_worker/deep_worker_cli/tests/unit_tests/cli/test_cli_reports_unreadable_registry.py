"""An unreadable registry must reach the CLI user as a message, not a stack.

Review finding 2026-10-19 (agent_h). Making ``_load_registry`` raise on
EACCES fixed a silent "0 agents" in the web, but it also turned
``deep-worker-cli --list`` from a graceful "No running agents found" into an
unhandled ``RegistryReadError`` traceback.

That regression hits exactly the person the fix is for: someone whose
registry permissions are broken runs the listing first, and the exception
message — which carries the path and the ``ls -l`` remedy — gets buried in
a Python stack trace instead of being read.

Loud is right. A traceback is not loud, it is noise.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.cli import client_command
from deep_worker_cli.exceptions import RegistryReadError
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


@pytest.fixture
def registry_with_agent(tmp_path: Path) -> tuple[ProcessRegistry, Path]:
    """A readable registry holding one live-looking agent."""
    registry_file = tmp_path / "registry.json"
    registry = ProcessRegistry(
        registry_file=registry_file, lock_file=tmp_path / "registry.lock"
    )
    registry.register(
        AgentProcessInfo(
            name="ghost",
            pid=os.getpid(),
            pid_file=tmp_path / "ghost.pid",
            working_dir=str(tmp_path),
            started_at=datetime.now(),
            rpyc_port=18861,
        )
    )
    return registry, registry_file


def _patched(registry: ProcessRegistry):
    """Point client_command at this test's registry, not the host's."""
    return patch.object(client_command, "ProcessRegistry", lambda: registry)


def test_list_agents_prints_the_listing_when_readable(
    registry_with_agent: tuple[ProcessRegistry, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive control.

    Without it, a test asserting "no traceback" would also pass against a
    command that silently prints nothing at all.
    """
    registry, _ = registry_with_agent
    with _patched(registry):
        rc = asyncio.run(client_command.list_agents())

    out = capsys.readouterr().out
    assert rc == 0
    assert "ghost" in out


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permission bits")
def test_list_agents_reports_unreadable_registry_without_raising(
    registry_with_agent: tuple[ProcessRegistry, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure must be a message on stderr and a non-zero exit code."""
    registry, registry_file = registry_with_agent
    os.chmod(registry_file, 0o000)
    try:
        with _patched(registry):
            # The assertion that matters: this call must not raise.
            rc = asyncio.run(client_command.list_agents())
    finally:
        os.chmod(registry_file, 0o666)

    captured = capsys.readouterr()
    assert rc == 1, "an unreadable registry must not report success"
    # The remedy has to reach the user, which is the whole point.
    assert str(registry_file) in captured.err
    assert "ls -l" in captured.err
    # And it must not be mistaken for an empty registry.
    assert "No running agents found" not in captured.out


def test_connect_interactive_lists_peers_when_registry_is_readable(
    registry_with_agent: tuple[ProcessRegistry, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive control for the second consumer.

    ``connect_interactive`` prints an "Available agents" listing as part of
    its "agent not found" message. Pin that the listing works before
    asserting anything about how it degrades.
    """
    registry, _ = registry_with_agent
    with _patched(registry):
        rc = asyncio.run(client_command.connect_interactive("nosuchagent"))

    out = capsys.readouterr().out
    assert rc == 1
    assert "ghost" in out


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permission bits")
def test_connect_interactive_reports_unreadable_registry_without_raising(
    registry_with_agent: tuple[ProcessRegistry, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The secondary listing must degrade, not crash the primary error.

    Here the unreadable registry is hit twice: ``lookup`` cannot find the
    agent, and the follow-up ``list_all`` cannot enumerate alternatives. The
    user still has to learn that their registry is unreadable.
    """
    registry, registry_file = registry_with_agent
    os.chmod(registry_file, 0o000)
    try:
        with _patched(registry):
            rc = asyncio.run(client_command.connect_interactive("nosuchagent"))
    finally:
        os.chmod(registry_file, 0o666)

    captured = capsys.readouterr()
    assert rc == 1
    combined = captured.out + captured.err
    assert str(registry_file) in combined
    assert "ls -l" in combined
    # "(none)" would claim the registry is empty, which is a different fact.
    assert "(none)" not in captured.out


def test_stop_agent_reports_a_missing_agent_when_registry_is_readable(
    registry_with_agent: tuple[ProcessRegistry, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive control for ``stop_agent``'s registry read."""
    registry, _ = registry_with_agent
    with _patched(registry):
        rc = asyncio.run(client_command.stop_agent("nosuchagent"))

    assert rc == 1
    assert "not found" in capsys.readouterr().err


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permission bits")
def test_stop_agent_reports_unreadable_registry_without_a_traceback(
    registry_with_agent: tuple[ProcessRegistry, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """``stop_agent`` has a blanket ``except Exception`` that also dumps a
    stack trace. Catching the message alone is not enough here: the oracle
    has to be "no traceback", otherwise the blanket handler makes this test
    pass while the user still gets a wall of Python frames.
    """
    registry, registry_file = registry_with_agent
    os.chmod(registry_file, 0o000)
    try:
        with _patched(registry):
            rc = asyncio.run(client_command.stop_agent("ghost"))
    finally:
        os.chmod(registry_file, 0o666)

    captured = capsys.readouterr()
    assert rc == 1
    assert str(registry_file) in captured.err
    assert "ls -l" in captured.err
    assert "Traceback (most recent call last)" not in captured.err
    assert "RegistryReadError" not in captured.err


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permission bits")
def test_unreadable_registry_still_raises_at_the_registry_layer(
    registry_with_agent: tuple[ProcessRegistry, Path]
) -> None:
    """Pin that the CLI handles the error rather than the layer dropping it.

    If a later change made ``list_all`` swallow EACCES again, the CLI test
    above would still pass while the silent-empty defect was back. This
    anchors the contract the CLI is handling.
    """
    registry, registry_file = registry_with_agent
    assert [a.name for a in registry.list_all()] == ["ghost"]

    os.chmod(registry_file, 0o000)
    try:
        with pytest.raises(RegistryReadError):
            registry.list_all()
    finally:
        os.chmod(registry_file, 0o666)
