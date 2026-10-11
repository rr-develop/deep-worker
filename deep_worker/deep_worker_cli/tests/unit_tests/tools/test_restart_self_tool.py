"""Tests for restart_self tool: make_restart_self_tool factory.

The tool ARMS a restart and returns a real result; it does NOT hard-exit.
``os._exit(0)`` used to run right after ``subprocess.Popen``, severing
the turn before the result could be recorded — the CLI then wrote a
fabricated ``user-rejected`` denial (incidents 2026-06-12, 2026-08-28).
``tools.py`` no longer imports ``os`` at all, which is asserted below.

The driver is spawned by the delivery barrier, on the checkpointer's
commit listener: a result counts as delivered when, and only when, it has
been persisted to the checkpoint. Tests therefore invoke the tool the way
LangGraph does (so the ``InjectedToolCallId`` is populated) and then fire
the barrier with the committed id. There is no flag-file, no
"already_restarting" state, and no "after the turn completes" semantics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from deep_worker_cli.storage_paths import deep_worker_home
_CALL_ID = "toolu_test"


def _checkpoint_with_tool_calls(call_ids: list[str]) -> dict:
    """A checkpoint carrying one ToolMessage per tool_call_id.

    Mirrors what LangGraph's ToolNode persists, so the commit listener
    extracts the ids from the same place it does in production.
    """
    import uuid

    from langchain_core.messages import ToolMessage

    messages = [
        ToolMessage(content="ok", tool_call_id=cid, name="restart_self")
        for cid in call_ids
    ]
    return {
        "v": 1,
        "id": uuid.uuid4().hex[:16],
        "ts": "2026-08-28T12:00:00",
        "channel_values": {"messages": messages, "working_dir": "/tmp"},
        "channel_versions": {"messages": f"{len(messages):032}.0"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["messages"],
    }


async def _ainvoke(tool, call_id: str = _CALL_ID, **kwargs):
    """Invoke as LangGraph does, so InjectedToolCallId is populated."""
    result = await tool.ainvoke(
        {"args": kwargs, "id": call_id, "name": tool.name, "type": "tool_call"}
    )
    content = getattr(result, "content", result)
    return json.loads(content) if isinstance(content, str) else content


class TestRestartSelfFactory:
    def test_factory_returns_tool_with_expected_name(self):
        from deep_worker_cli.tools import make_restart_self_tool

        tool = make_restart_self_tool("alpha")
        assert tool.name == "restart_self"

    def test_tool_exposes_no_model_facing_args(self):
        """The model still calls ``restart_self()`` with no arguments.

        ``tool_call_id`` is an ``InjectedToolCallId``: LangGraph fills it
        in, and ``tool.args`` (what the model is shown) must stay empty —
        otherwise the model starts inventing call ids.
        """
        from deep_worker_cli.tools import make_restart_self_tool

        tool = make_restart_self_tool("alpha")
        assert tool.args == {}, tool.args


class TestRestartSelfBehavior:
    def _setup(self, tmp_path, monkeypatch):
        """Make the tool see tmp_path as $HOME and find a fake deep-worker.

        No ``os._exit`` patch is needed any more: the tool returns
        normally and lets the driver stop the process, so the test process
        is never at risk. Returns ``(popen_calls, [])``; the second element
        is retained so existing unpacking keeps working and stays empty as
        proof no hard-exit happens.

        ``popen_calls`` records what the DELIVERY BARRIER spawns, not what
        the tool spawns — the tool only arms the restart, and the driver is
        launched once the tool result has been committed to the checkpoint.
        Each helper that needs the driver argv therefore fires the barrier
        explicitly (see ``_deliver``), which is the production sequence.
        """
        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.process import self_restart_barrier as barrier

        barrier.reset_armed_self_restart_for_test()
        barrier.set_self_restart_delivery_available_for_test(True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(tools_mod.Path, "home", lambda: tmp_path)

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir(exist_ok=True)
        fake_dw = fake_bin / "deep-worker"
        fake_dw.write_text("#!/bin/sh\nexit 0\n")
        fake_dw.chmod(0o755)
        monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

        popen_calls: list[list] = []

        class _FakePopen:
            def __init__(self, args, **kwargs):
                popen_calls.append(args)
                self.pid = 99999
            def poll(self):
                return 0

        # The spawn now lives in the barrier. Patch it there AND in tools
        # so a regression that reintroduces an inline spawn is still
        # captured (and then caught by the "must not Popen" assertions).
        monkeypatch.setattr(barrier.subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)

        exit_calls: list[int] = []
        return popen_calls, exit_calls

    @staticmethod
    def _deliver(call_id: str = _CALL_ID) -> bool:
        """Deliver THIS call's result the way production does.

        Delivery is the checkpoint commit: the checkpointer's commit
        listener hands the barrier the tool_call_ids that were committed.
        """
        from deep_worker_cli.process.self_restart_barrier import (
            fire_armed_self_restart,
        )

        return fire_armed_self_restart(frozenset({call_id}))

    async def test_restart_self_writes_notice_file(self, tmp_path, monkeypatch):
        """Tool writes .restart_notice.json in agent's data dir."""
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_self_{os.getpid()}"
        self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        notice_path = deep_worker_home(tmp_path) / "agents" / agent_name / ".restart_notice.json"
        assert notice_path.exists()
        notice = json.loads(notice_path.read_text())
        assert notice["cause"] == "self"
        assert "ts" in notice

    async def test_restart_self_spawns_deep_worker_synchronously(
        self, tmp_path, monkeypatch
    ):
        """One tool call spawns ONE driver — but only after delivery.

        The spawn must NOT happen while the result is in flight: a driver
        racing the result can SIGTERM this process before the CLI has the
        result, which is the dangling-``tool_use`` defect this whole fix
        exists to remove.
        """
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_spawn_{os.getpid()}"
        popen_calls, _exit_calls = self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        assert popen_calls == [], (
            "the driver must not be spawned while the tool result is still "
            "travelling to the CLI"
        )

        assert self._deliver() is True
        assert len(popen_calls) == 1
        argv = popen_calls[0]
        assert "--restart" in argv
        assert agent_name in argv

    async def test_restart_self_returns_result_without_hard_exit(
        self, tmp_path, monkeypatch
    ):
        """The tool MUST return normally — no ``os._exit``.

        The hard-exit killed the in-process MCP server before the result
        could reach the CLI. The CLI then recorded the dangling call as
        ``toolDenialKind: "user-rejected"`` and fabricated "The user
        doesn't want to proceed with this tool use" — a statement about
        the user that the user never made. Returning a real result is
        what keeps that fabrication out of the transcript.
        """
        import deep_worker_cli.tools as tools_mod
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_exit_{os.getpid()}"
        popen_calls, exit_calls = self._setup(tmp_path, monkeypatch)

        # Not patched by _setup any more: a surviving os._exit would kill
        # the test runner outright, so trip a flag instead of dying.
        assert not hasattr(tools_mod, "os"), (
            "tools.py must not import os — a live os._exit here would "
            "sever the turn and resurrect the false-denial bug"
        )

        tool = make_restart_self_tool(agent_name)
        result = await _ainvoke(tool)

        assert exit_calls == [], "restart_self must not hard-exit"
        assert result["success"] is True
        assert "restart_notice" in result

        # The restart is real — it is merely deferred until the result has
        # been handed to the CLI.
        assert self._deliver() is True
        assert len(popen_calls) == 1

    async def test_restart_self_passes_self_requested_to_driver(
        self, tmp_path, monkeypatch
    ):
        """``--self-requested`` + its token waive the "actively running" guard.

        An agent calling restart_self is BY CONSTRUCTION mid-turn
        (status="running"). Without the flag ``stop_agent`` refuses, the
        driver returns 1, and nothing restarts. Without the token the flag
        is refused too — it is a claim anyone could make from a shell, so
        the driver must carry the capability the agent minted in its own
        private data dir.
        """
        from deep_worker_cli.process.self_restart_token import TOKEN_FILENAME
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_selfreq_{os.getpid()}"
        popen_calls, _ = self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)
        assert self._deliver() is True

        argv = popen_calls[0]
        assert "--self-requested" in argv
        assert "--restart" in argv
        assert agent_name in argv

        # The token on the command line must be the one on disk, or
        # stop_agent will refuse the restart.
        token_path = (
            deep_worker_home(tmp_path) / "agents" / agent_name / TOKEN_FILENAME
        )
        assert token_path.exists(), "restart_self must mint a capability token"
        on_disk = json.loads(token_path.read_text())["token"]

        assert "--self-requested-token" in argv
        assert argv[argv.index("--self-requested-token") + 1] == on_disk

    async def test_restart_self_token_is_private(self, tmp_path, monkeypatch):
        """The capability is a secret — 0600, like the notice."""
        from deep_worker_cli.process.self_restart_token import TOKEN_FILENAME
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_tokperm_{os.getpid()}"
        self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        token_path = (
            deep_worker_home(tmp_path) / "agents" / agent_name / TOKEN_FILENAME
        )
        mode = token_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    async def test_restart_self_result_is_not_a_denial_message(
        self, tmp_path, monkeypatch
    ):
        """The returned result must not read as a user denial.

        Regression pin for the exact fabricated strings the CLI used to
        synthesise when the process died mid-call.
        """
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_nodenial_{os.getpid()}"
        self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        result = await _ainvoke(tool)

        blob = json.dumps(result).lower()
        assert "doesn't want to proceed" not in blob
        assert "request interrupted by user" not in blob
        assert "rejected" not in blob

    async def test_restart_self_notice_written_with_private_perms(
        self, tmp_path, monkeypatch
    ):
        """Notice file is created with private permissions (0o600)."""
        from deep_worker_cli.tools import make_restart_self_tool

        agent_name = f"test_perm_{os.getpid()}"
        self._setup(tmp_path, monkeypatch)

        tool = make_restart_self_tool(agent_name)
        await _ainvoke(tool)

        notice_path = deep_worker_home(tmp_path) / "agents" / agent_name / ".restart_notice.json"
        assert notice_path.exists()
        mode = notice_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    async def test_production_provider_matrix(
        self, tmp_path, monkeypatch
    ):
        """EVERY production provider restarts exactly once after delivery.

        Delivery means the result was recorded in the model's context,
        which is the checkpointer's commit. The checkpointer is shared by
        all providers, so there is no provider gate: singling out
        ``claude-code-sdk`` was a property of the old transport hook, not
        of delivery, and pinning it here would re-freeze that defect.
        """
        from deep_worker_cli.process import self_restart_barrier as barrier
        from deep_worker_cli.runtime.provider_chain import all_provider_names
        from deep_worker_cli.tools import make_restart_self_tool

        popen_calls, _ = self._setup(tmp_path, monkeypatch)
        names = all_provider_names()
        assert "claude-code-sdk" in names
        assert "claude-code" in names

        for provider in names:
            barrier.reset_armed_self_restart_for_test()
            barrier.set_self_restart_delivery_available_for_test(True)
            agent_name = f"matrix_{provider}"
            call_id = f"toolu_{provider}"
            before = len(popen_calls)

            result = await _ainvoke(
                make_restart_self_tool(agent_name), call_id=call_id
            )

            assert result["success"] is True, provider
            assert barrier.self_restart_armed() is True, provider
            assert popen_calls[before:] == [], (
                f"{provider}: must not spawn before the commit"
            )

            assert self._deliver(call_id) is True, provider
            assert len(popen_calls) == before + 1, provider
            argv = popen_calls[-1]
            assert "--restart" in argv
            assert agent_name in argv

    async def test_real_checkpoint_commit_fires_restart_for_every_provider(
        self, tmp_path, monkeypatch
    ):
        """End-to-end on the REAL checkpointer, for every provider.

        The matrix above fires the barrier directly. This one proves the
        wiring the owner actually specified: a genuine
        ``DeltaSqliteCheckpointer.aput`` — real SQLite, real ``commit()``
        — is what releases the restart, and it does so identically for
        every provider because the checkpointer is provider-agnostic.
        """
        import aiosqlite

        from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer
        from deep_worker_cli.process import self_restart_barrier as barrier
        from deep_worker_cli.runtime.provider_chain import all_provider_names
        from deep_worker_cli.tools import make_restart_self_tool

        popen_calls, _ = self._setup(tmp_path, monkeypatch)

        for provider in all_provider_names():
            barrier.reset_armed_self_restart_for_test()
            barrier.set_self_restart_delivery_available_for_test(True)
            agent_name = f"e2e_{provider}"
            call_id = f"toolu_e2e_{provider}"
            before = len(popen_calls)

            db_path = str(tmp_path / f"cp_{provider}.db")
            async with aiosqlite.connect(db_path) as conn:
                cp = DeltaSqliteCheckpointer(conn, snapshot_interval=0)
                cp.add_commit_listener(barrier.fire_armed_self_restart)

                result = await _ainvoke(
                    make_restart_self_tool(agent_name), call_id=call_id
                )
                assert result["success"] is True, provider
                assert popen_calls[before:] == [], (
                    f"{provider}: armed but not yet committed — no spawn"
                )

                # A commit that does NOT contain our call id must not fire.
                await cp.aput(
                    {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                    _checkpoint_with_tool_calls(["toolu_someone_else"]),
                    {},
                    {},
                )
                assert popen_calls[before:] == [], (
                    f"{provider}: a foreign tool result must not fire ours"
                )
                assert barrier.self_restart_armed() is True, provider

                # The commit carrying our call id releases the restart.
                await cp.aput(
                    {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                    _checkpoint_with_tool_calls([call_id]),
                    {},
                    {},
                )

            assert len(popen_calls) == before + 1, (
                f"{provider}: exactly one restart after the real commit"
            )
            argv = popen_calls[-1]
            assert "--restart" in argv
            assert agent_name in argv
            assert barrier.self_restart_armed() is False, provider

    async def test_missing_commit_signal_fails_before_side_effects(
        self, tmp_path, monkeypatch
    ):
        """No committing checkpointer => fail closed, before any side effect.

        Under ``--memory-only`` the saver never commits, so the barrier
        could never confirm the result reached the context. Reporting
        success there would promise a restart that silently never fires.
        """
        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.process import self_restart_barrier as barrier
        from deep_worker_cli.tools import make_restart_self_tool

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(tools_mod.Path, "home", lambda: tmp_path)
        barrier.reset_armed_self_restart_for_test()
        barrier.set_self_restart_delivery_available_for_test(False)

        agent_name = "missing_commit_signal"
        result = await _ainvoke(make_restart_self_tool(agent_name))
        assert result["success"] is False
        assert result["error"] == "no_delivery_signal"
        assert barrier.self_restart_armed() is False
        assert not (deep_worker_home(tmp_path) / "agents" / agent_name).exists()
