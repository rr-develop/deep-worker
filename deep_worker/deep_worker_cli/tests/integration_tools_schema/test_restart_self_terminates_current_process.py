"""Critical contract: restart_self hands termination to the DRIVER and
returns a real tool result — it must never hard-exit its own process.

History (two incidents, in order):

1. 2026-06-12 — the original implementation called ``subprocess.Popen``
   and returned. The old process kept running, the LLM finalised the
   assistant message, the turn ended, and the new process's auto-
   continuation was treated as a *separate* idle turn ("after this you are in
   idle, which is WRONG"). The fix at the time was ``os._exit(0)``.

2. 2026-08-28 — that hard-exit turned out to destroy the **in-process**
   MCP server (``create_sdk_mcp_server`` runs inside this very process)
   before the tool result could reach the CLI. The CLI saw a ``tool_use``
   whose result never arrived, stamped ``toolDenialKind: "user-rejected"``
   and synthesised "The user doesn't want to proceed with this tool use"
   plus "[Request interrupted by user for tool use]". Both are replayed
   into the next process on resume, so the model reads a fabricated
   statement about the user. 5 of 5 historical restart_self calls were
   mis-recorded this way; no user ever denied one.

The current semantics keep BOTH properties:

  * No hard-exit. The tool returns ``{"success": True, ...}`` through the
    ordinary path (ToolNode -> ``resolve_call`` -> CLI), so the transcript
    records what actually happened.
  * The spawned ``deep-worker --restart <name> --self-requested`` driver
    stops this process. ``--self-requested`` waives the "actively running"
    guard in ``stop_agent`` — an agent calling restart_self is by
    construction mid-turn, so without it every self-restart is refused.
  * The 2026-06-12 invariant is carried by ``.restart_notice.json``, NOT
    by killing the turn: ``AgentProcess.start`` schedules
    ``_auto_continue_after_restart`` on the notice's presence alone and
    never inspects whether the previous turn committed.

These tests pin the contract:

  1. On successful Popen the tool returns a success result and does not
     hard-exit.
  2. The driver is invoked with ``--self-requested``.
  3. The result never reads as a user denial.
  4. Failure paths (notice write error, deep-worker not on PATH, Popen
     OSError) return a structured error and spawn nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_barrier():
    """The delivery barrier is module-global state.

    A test that arms a restart and never fires it would otherwise leak that
    arm into the next test, where a ``_deliver()`` call would spawn a driver
    the test never asked for. Reset on both sides so each test starts and
    leaves the barrier disarmed.
    """
    from deep_worker_cli.process.self_restart_barrier import (
        reset_armed_self_restart_for_test,
        set_self_restart_delivery_available_for_test,
    )

    reset_armed_self_restart_for_test()
    set_self_restart_delivery_available_for_test(True)
    yield
    reset_armed_self_restart_for_test()
    set_self_restart_delivery_available_for_test(False)


def _install_fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from deep_worker_cli import tools as tools_mod

    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)
    return fake_home


def _install_fake_deep_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[list]:
    """Put a fake ``deep-worker`` on PATH and capture the driver argv.

    The spawn now belongs to the delivery barrier, not to the tool, so the
    barrier's ``subprocess`` is the one that must be patched. ``tools`` is
    patched too, so a regression that reintroduces an inline spawn is still
    captured (and then caught by the assertions that ``popen_calls`` is
    empty before delivery).
    """
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.process import self_restart_barrier as barrier

    barrier.reset_armed_self_restart_for_test()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_dw = fake_bin / "deep-worker"
    fake_dw.write_text("#!/bin/sh\nexit 0\n")
    fake_dw.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    popen_calls: list[list] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            popen_calls.append(args)
            self.pid = 4242

    monkeypatch.setattr(barrier.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    return popen_calls


async def _ainvoke(tool, call_id: str = "toolu_test"):
    """Invoke as LangGraph does, so the InjectedToolCallId is populated.

    ``restart_self`` arms itself bound to this id, so the barrier can only
    be fired by a commit that actually carries it.
    """
    result = await tool.ainvoke(
        {"args": {}, "id": call_id, "name": tool.name, "type": "tool_call"}
    )
    content = getattr(result, "content", result)
    return json.loads(content) if isinstance(content, str) else content


def _deliver(call_id: str = "toolu_test") -> bool:
    """Deliver THIS call's result the way production does.

    Delivery is the checkpoint commit: the checkpointer's commit listener
    hands the barrier the ``tool_call_id`` of every ToolMessage in the
    committed checkpoint. The tool armed itself already bound to its own
    ``InjectedToolCallId``, so there is no separate binding step.
    """
    from deep_worker_cli.process.self_restart_barrier import (
        fire_armed_self_restart,
    )

    return fire_armed_self_restart(frozenset({call_id}))


@pytest.mark.asyncio
async def test_restart_self_returns_result_and_does_not_hard_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real result must come back; the process must survive to deliver it.

    If ``os._exit`` were reintroduced this test would not merely fail — the
    pytest process itself would vanish. The structural guarantee is that
    ``tools.py`` no longer imports ``os`` at all.
    """
    import deep_worker_cli.tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_noexit_{os.getpid()}"
    _install_fake_home(tmp_path, monkeypatch)
    popen_calls = _install_fake_deep_worker(tmp_path, monkeypatch)

    assert not hasattr(tools_mod, "os"), (
        "tools.py must not import os — the only historical use was the "
        "os._exit(0) that caused the fabricated user-denial"
    )

    tool = make_restart_self_tool(agent_name)
    result = await _ainvoke(tool)

    assert result["success"] is True
    assert "restart_notice" in result

    # The driver must NOT have been spawned yet: while this result is
    # unrecorded, a driver is free to SIGTERM us and strand the tool_use.
    assert popen_calls == [], (
        "the restart driver must not be spawned before the tool result is "
        "delivered"
    )

    # ...and once delivery is confirmed, the restart really does happen.
    assert _deliver() is True
    assert len(popen_calls) == 1


@pytest.mark.asyncio
async def test_restart_self_requests_self_sanctioned_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driver must be told the stop is self-requested — and prove it.

    Without ``--self-requested`` the "actively running" guard refuses the
    stop (the caller is mid-turn by construction), ``--restart`` returns 1
    and the agent is never replaced. The flag alone is not enough either:
    it is a claim any shell could make, so the driver must also carry the
    one-time capability token the agent minted in its own private data dir.
    """
    from deep_worker_cli.process.self_restart_token import TOKEN_FILENAME
    from deep_worker_cli.storage_paths import agent_root
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_sanction_{os.getpid()}"
    _install_fake_home(tmp_path, monkeypatch)
    popen_calls = _install_fake_deep_worker(tmp_path, monkeypatch)

    tool = make_restart_self_tool(agent_name)
    await _ainvoke(tool)
    assert _deliver() is True

    argv = popen_calls[0]
    assert "--restart" in argv
    assert agent_name in argv
    assert "--self-requested" in argv

    token_path = agent_root(agent_name) / TOKEN_FILENAME
    assert token_path.exists(), "restart_self must mint a capability token"
    on_disk = json.loads(token_path.read_text())["token"]
    assert "--self-requested-token" in argv
    assert argv[argv.index("--self-requested-token") + 1] == on_disk, (
        "the driver must carry the token that is actually on disk, or "
        "stop_agent will refuse the restart"
    )


@pytest.mark.asyncio
async def test_restart_self_result_contains_no_denial_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin for the exact fabricated strings."""
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_lang_{os.getpid()}"
    _install_fake_home(tmp_path, monkeypatch)
    _install_fake_deep_worker(tmp_path, monkeypatch)

    tool = make_restart_self_tool(agent_name)
    result = await _ainvoke(tool)

    blob = json.dumps(result).lower()
    assert "doesn't want to proceed" not in blob
    assert "request interrupted by user" not in blob
    assert "rejected" not in blob


@pytest.mark.asyncio
async def test_restart_self_does_not_exit_when_deep_worker_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If deep-worker is not on PATH the tool must NOT hard-exit — there
    is no replacement coming. Return a structured error instead."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_nodw_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)

    monkeypatch.setenv("PATH", "/nonexistent_directory")

    popen_calls: list[list] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            popen_calls.append(args)

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)

    tool = make_restart_self_tool(agent_name)
    result = await _ainvoke(tool)

    assert result["success"] is False
    assert result["error"] == "command_unavailable"
    assert popen_calls == []


@pytest.mark.asyncio
async def test_restart_self_does_not_exit_on_popen_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable driver is reported in the RESULT, not just a log line.

    Deferring the spawn to the delivery barrier also defers any spawn
    error past the point where this tool can report it — by the time the
    barrier fires, the result the model sees has already been written.
    ``restart_self`` therefore pre-flights the driver, so the ordinary
    failure (missing or non-executable binary) still comes back as a
    structured error instead of a cheerful "success" followed by silence.

    A non-executable binary is rejected by ``shutil.which`` first, so the
    reported code is ``command_unavailable``; the pre-flight covers the
    narrower case where the execute bit is lost after ``which`` succeeded.
    Either way the contract that matters holds: a structured failure, no
    spawn, and nothing left armed.
    """
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_popenfail_{os.getpid()}"
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: fake_home)

    # On PATH (so ``shutil.which`` finds it) but not executable.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dw = fake_bin / "deep-worker"
    fake_dw.write_text("#!/bin/sh\nexit 0\n")
    fake_dw.chmod(0o644)
    monkeypatch.setenv("PATH", str(fake_bin))

    popen_calls: list[list] = []
    monkeypatch.setattr(
        tools_mod.subprocess, "Popen",
        lambda *a, **k: popen_calls.append(a),
    )

    tool = make_restart_self_tool(agent_name)
    result = await _ainvoke(tool)

    assert result["success"] is False
    assert result["error"] in {"spawn_failed", "command_unavailable"}
    assert popen_calls == []

    # Nothing may be left armed — a later turn must not fire a restart the
    # model was told had failed.
    assert _deliver() is False


@pytest.mark.asyncio
async def test_restart_self_preflights_a_driver_that_lost_its_exec_bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-flight covers what ``shutil.which`` cannot.

    ``which`` runs before the spawn is armed; if the binary is unusable by
    the time we would launch it, the deferred spawn would fail where the
    model can no longer be told. Exercised directly by making ``which``
    succeed on a path that is not executable.
    """
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_preflight_{os.getpid()}"
    _install_fake_home(tmp_path, monkeypatch)
    popen_calls = _install_fake_deep_worker(tmp_path, monkeypatch)

    dud = tmp_path / "bin" / "deep-worker-dud"
    dud.write_text("#!/bin/sh\nexit 0\n")
    dud.chmod(0o644)
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _n: str(dud))

    tool = make_restart_self_tool(agent_name)
    result = await _ainvoke(tool)

    assert result["success"] is False
    assert result["error"] == "spawn_failed"
    assert popen_calls == []
    assert _deliver() is False, "a failed pre-flight must leave nothing armed"


@pytest.mark.asyncio
async def test_barrier_spawn_failure_does_not_leave_restart_armed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A driver that fails to exec at fire time must not stay armed.

    The tool already reported success by then, so the barrier is the last
    place this can surface. It must log loudly and disarm rather than
    retry on every subsequent checkpoint commit.
    """
    from deep_worker_cli.process import self_restart_barrier as barrier
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_firefail_{os.getpid()}"
    _install_fake_home(tmp_path, monkeypatch)
    _install_fake_deep_worker(tmp_path, monkeypatch)

    def _failing_popen(*args, **kwargs):
        raise OSError("simulated spawn failure")

    monkeypatch.setattr(barrier.subprocess, "Popen", _failing_popen)

    tool = make_restart_self_tool(agent_name)
    result = await _ainvoke(tool, "toolu_firefail")
    assert result["success"] is True

    assert barrier.fire_armed_self_restart(frozenset({"toolu_firefail"})) is False
    assert barrier.self_restart_armed() is False
    assert barrier.fire_armed_self_restart(frozenset({"toolu_firefail"})) is False


@pytest.mark.asyncio
async def test_restart_self_does_not_exit_when_notice_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If we cannot even write the notice file the tool must NOT hard-exit
    — the replacement process would have no way to know it was restarted."""
    from deep_worker_cli import tools as tools_mod
    from deep_worker_cli.tools import make_restart_self_tool

    agent_name = f"test_noticefail_{os.getpid()}"
    # Point HOME at a path we cannot write to.
    bad_home = tmp_path / "ro_home"
    bad_home.mkdir(mode=0o500)
    monkeypatch.setattr(Path, "home", lambda: bad_home)
    monkeypatch.setattr(tools_mod.Path, "home", lambda: bad_home)

    popen_calls: list[list] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            popen_calls.append(args)

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)

    tool = make_restart_self_tool(agent_name)
    result = await _ainvoke(tool)

    # Restore perms so tmp_path can be cleaned up.
    bad_home.chmod(0o700)

    assert result["success"] is False
    assert result["error"] == "notice_write_error"
    assert popen_calls == []
