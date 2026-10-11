"""Self-requested stop: narrow waiver, and the invariant it must not break.

``restart_self`` used to kill its own process with ``os._exit(0)`` because
``stop_agent`` refuses to stop an agent whose status is ``running`` — and an
agent calling ``restart_self`` is mid-turn by construction, so it is ALWAYS
``running``. That hard-exit destroyed the in-process MCP server before the
tool result could reach the CLI, which then recorded the dangling call as
``toolDenialKind: "user-rejected"`` and fabricated "The user doesn't want to
proceed with this tool use". The user had denied nothing.

The fix gives ``stop_agent`` an explicit ``self_requested`` waiver. These
tests pin two things that must both hold:

  * the waiver is NARROW — it only bypasses the "actively running" guard,
    and only when the caller opts in (an external ``--stop`` of a busy
    agent is refused exactly as before);
  * the 2026-06-12 invariant survives — the replacement process must
    CONTINUE the pending task rather than wake up idle. That invariant is
    carried by ``.restart_notice.json``, NOT by severing the turn, so
    removing the hard-exit cannot regress it.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deep_worker_cli.process.agent_stop import stop_agent


def _write_status(data_dir: Path, status: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "session_metadata.json").write_text(json.dumps({"status": status}))


def _registry_with(pid: int) -> MagicMock:
    registry = MagicMock()
    registry.lookup.return_value = MagicMock(pid=pid)
    return registry


class TestSelfRequestedWaiverIsNarrow:
    """The waiver must not weaken the guard for anyone else."""

    @pytest.mark.asyncio
    async def test_external_stop_of_running_agent_still_refused(self, tmp_path):
        """Default behaviour is unchanged: a third party cannot interrupt."""
        _write_status(tmp_path, "running")

        with pytest.raises(RuntimeError, match="actively running"):
            await stop_agent(
                "busy_agent", _registry_with(424242), data_dir=tmp_path
            )

    @pytest.mark.asyncio
    async def test_self_requested_stop_of_running_agent_is_allowed(
        self, tmp_path, monkeypatch
    ):
        """The agent is never a third party to its own turn.

        The waiver requires the capability token as well as the flag — see
        ``TestSelfRequestedRequiresCapability`` for why the flag alone is
        not enough. Here we exercise the sanctioned path.
        """
        from deep_worker_cli.process.self_restart_token import (
            mint_self_restart_token,
        )

        _write_status(tmp_path, "running")
        token = mint_self_restart_token(tmp_path)

        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop.os.kill",
            lambda pid, sig: killed.append((pid, sig)),
        )
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop._wait_for_exit",
            _immediate_exit,
        )

        registry = _registry_with(424242)
        result = await stop_agent(
            "self_agent",
            registry,
            data_dir=tmp_path,
            self_requested=True,
            self_requested_token=token,
        )

        assert result is True
        assert killed, "self-requested stop must actually signal the process"

    @pytest.mark.asyncio
    async def test_waiver_does_not_apply_to_other_statuses(
        self, tmp_path, monkeypatch
    ):
        """An idle agent was always stoppable — that path is untouched."""
        _write_status(tmp_path, "idle")

        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop.os.kill", lambda pid, sig: None
        )
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop._wait_for_exit",
            _immediate_exit,
        )

        assert await stop_agent(
            "idle_agent", _registry_with(424242), data_dir=tmp_path
        ) is True


class TestDriverPlumbing:
    """The sanction must survive the trip CLI -> driver -> stop_agent."""

    def test_flag_parses_and_defaults_to_false(self):
        from deep_worker_cli.cli.agent_command import create_parser

        parser = create_parser()

        assert parser.parse_args(
            ["a", "--restart", "--self-requested"]
        ).self_requested is True
        assert parser.parse_args(["a", "--restart"]).self_requested is False

    def test_flag_is_not_advertised_in_help(self):
        """Internal plumbing — an operator has no reason to pass it."""
        from deep_worker_cli.cli.agent_command import create_parser

        assert "--self-requested" not in create_parser().format_help()

    @pytest.mark.asyncio
    async def test_driver_forwards_the_sanction_to_stop_agent(
        self, tmp_path, monkeypatch
    ):
        """``--restart --self-requested`` must reach ``stop_agent``.

        Behavioural, not a source match: pinning the literal
        ``self_requested=`` would still pass if someone hard-coded
        ``self_requested=False``, which silently restores the bug.
        """
        from deep_worker_cli.cli import agent_command

        seen: dict = {}

        async def _spy_stop_agent(name, registry, **kwargs):
            seen.update(kwargs)
            return True

        # ``stop_agent`` is imported INSIDE restart_agent, so patch it at
        # its definition site rather than on the cli module.
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop.stop_agent", _spy_stop_agent
        )
        monkeypatch.setattr(
            agent_command, "load_launch_config", lambda n: {"provider": "x"}
        )
        monkeypatch.setattr(
            agent_command, "merge_restart_config", lambda saved, args: saved
        )
        monkeypatch.setattr(
            agent_command, "_get_agent_data_dir", lambda n: tmp_path
        )

        class _Registry:
            def lookup(self, name):
                return MagicMock(pid=4242)

        monkeypatch.setattr(
            "deep_worker_cli.process.registry.ProcessRegistry", _Registry
        )

        parser = agent_command.create_parser()
        args = parser.parse_args(["ag", "--restart", "--self-requested"])

        # Everything past the stop is out of scope; let it fail however it
        # likes, the assertion below only cares that the sanction arrived.
        try:
            await agent_command.restart_agent(args)
        except Exception:  # noqa: BLE001
            pass

        assert seen.get("self_requested") is True, (
            "the --restart driver must forward the REAL sanction to "
            "stop_agent; a hard-coded False refuses every self-restart"
        )


class TestRestartContinuationInvariant:
    """Incident 2026-06-12: the new process must not wake up idle."""

    def test_auto_continue_is_keyed_to_the_notice_not_the_turn(self, tmp_path):
        """The invariant rides on ``.restart_notice.json``.

        This is what makes removing ``os._exit`` safe: the decision to
        continue is taken from the notice file alone. Nothing inspects
        whether the previous process committed its turn, so a cleanly
        terminated predecessor continues to produce a continuation — not
        an idle agent.
        """
        from deep_worker_cli.tools import write_restart_notice

        notice_path = write_restart_notice(
            tmp_path, cause="self", message="restarted by its own tool call"
        )

        notice = json.loads(notice_path.read_text())
        assert notice["cause"] == "self"
        assert "ts" in notice
        assert notice["message"]

    def test_restart_self_still_writes_the_notice_before_spawning(self):
        """Order matters: no notice => the replacement would go idle.

        Pinned by reading the source rather than by running the tool, so
        the ordering cannot silently invert during a refactor.
        """
        import inspect

        from deep_worker_cli import tools as tools_mod

        src = inspect.getsource(tools_mod.make_restart_self_tool)
        gate_at = src.index("no_delivery_signal")
        notice_at = src.index("write_restart_notice")
        arm_at = src.index("arm_self_restart(")
        assert gate_at < notice_at < arm_at, (
            "a process with no committing checkpointer must fail BEFORE the "
            "notice is written or the driver is armed, and the notice must "
            "still precede the arm so a restart cannot wake idle "
            "(incident 2026-06-12)"
        )

    def test_restart_self_is_not_gated_on_the_provider(self):
        """No provider gate: the barrier fires on the checkpoint commit.

        Delivery means the result was recorded in the model's context,
        which is the checkpointer's commit — and the checkpointer is
        shared by every provider. Passing a provider in here would
        reintroduce the transport-shaped gate that made restart_self
        silently unavailable outside claude-code-sdk.
        """
        import inspect

        from deep_worker_cli import tools as tools_mod
        from deep_worker_cli.process import agent_process

        sig = inspect.signature(tools_mod.make_restart_self_tool)
        assert "provider" not in sig.parameters, sig

        source = Path(agent_process.__file__).read_text()
        idx = source.index("restart_self_tool_factory(")
        window = source[idx:idx + 240]
        assert "provider=" not in window, window

    def test_restart_self_arms_instead_of_spawning(self):
        """The tool must not spawn the driver itself.

        A ``subprocess.Popen`` inside the handler is the transport race:
        the driver would be free to SIGTERM this process while the tool
        result is still travelling to the CLI. The spawn belongs to the
        delivery barrier, which fires only after the result is out.
        """
        import inspect

        from deep_worker_cli import tools as tools_mod

        src = inspect.getsource(tools_mod.make_restart_self_tool)
        offenders = [
            ln.strip() for ln in src.splitlines()
            if "Popen" in ln and not ln.strip().startswith("#")
        ]
        assert offenders == [], (
            "restart_self must ARM the restart, not spawn it — spawning "
            f"here races the tool result to the CLI: {offenders}"
        )

    def test_restart_self_no_longer_severs_its_own_turn(self):
        """No hard-exit anywhere in the tool module.

        This is a SOURCE check on purpose. A behavioural assertion cannot
        catch a reintroduced ``os._exit``: it would kill the pytest process
        itself, and a runner that vanishes mid-file exits 0 and reads as a
        pass. Verified — a mutation adding a function-local
        ``import os as _os; _os._exit(0)`` produced no failure report at
        all. Matching the source keeps the check honest regardless of how
        the import is spelled (module-level, aliased, or function-local).
        """
        import inspect

        from deep_worker_cli import tools as tools_mod

        src = inspect.getsource(tools_mod)
        offenders = [
            ln.strip() for ln in src.splitlines()
            if "_exit(" in ln and not ln.strip().startswith("#")
        ]
        assert offenders == [], f"hard-exit reintroduced: {offenders}"

        # Also reject the import itself: ``os`` has no other use here, so
        # its mere presence means someone is preparing to hard-exit.
        imports = [
            ln.strip() for ln in src.splitlines()
            if ln.strip().startswith(("import os", "from os "))
            or ("import os as" in ln and not ln.strip().startswith("#"))
        ]
        assert imports == [], f"os imported into tools.py: {imports}"


class TestSelfRequestedRequiresCapability:
    """``--self-requested`` is a claim; only the token authorizes it.

    Reviewer finding: the flag alone made the waiver a publicly reachable
    capability — any shell could run
    ``deep-worker --restart <busy> --self-requested`` and bypass the
    active-turn guard. Hiding it from ``--help`` is obscurity, not
    authorization. The sanction must be proven, not asserted.
    """

    @pytest.mark.asyncio
    async def test_external_self_requested_without_token_is_refused(
        self, tmp_path
    ):
        """The attack the reviewer described, verbatim."""
        _write_status(tmp_path, "running")

        with pytest.raises(RuntimeError, match="actively running"):
            await stop_agent(
                "busy_agent",
                _registry_with(424242),
                data_dir=tmp_path,
                self_requested=True,          # claimed...
                self_requested_token=None,    # ...but not proven
            )

    @pytest.mark.asyncio
    async def test_forged_token_is_refused(self, tmp_path):
        """A guessed token must not open the waiver."""
        from deep_worker_cli.process.self_restart_token import (
            mint_self_restart_token,
        )

        _write_status(tmp_path, "running")
        mint_self_restart_token(tmp_path)

        with pytest.raises(RuntimeError, match="actively running"):
            await stop_agent(
                "busy_agent",
                _registry_with(424242),
                data_dir=tmp_path,
                self_requested=True,
                self_requested_token="0" * 64,
            )

    @pytest.mark.asyncio
    async def test_genuine_self_origin_token_is_accepted(
        self, tmp_path, monkeypatch
    ):
        """The real self-restart path still works end to end."""
        from deep_worker_cli.process.self_restart_token import (
            mint_self_restart_token,
        )

        _write_status(tmp_path, "running")
        token = mint_self_restart_token(tmp_path)

        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop.os.kill",
            lambda pid, sig: killed.append((pid, sig)),
        )
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop._wait_for_exit",
            _immediate_exit,
        )

        assert await stop_agent(
            "self_agent",
            _registry_with(424242),
            data_dir=tmp_path,
            self_requested=True,
            self_requested_token=token,
        ) is True
        assert killed, "a sanctioned self-stop must signal the process"

    @pytest.mark.asyncio
    async def test_token_is_consumed_once(self, tmp_path, monkeypatch):
        """A replayed token must not authorize a second stop.

        Without consume-once a token left on disk would let any later
        external ``--self-requested`` reuse it.
        """
        from deep_worker_cli.process.self_restart_token import (
            mint_self_restart_token,
        )

        _write_status(tmp_path, "running")
        token = mint_self_restart_token(tmp_path)

        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop.os.kill", lambda pid, sig: None
        )
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop._wait_for_exit",
            _immediate_exit,
        )

        assert await stop_agent(
            "self_agent",
            _registry_with(424242),
            data_dir=tmp_path,
            self_requested=True,
            self_requested_token=token,
        ) is True

        _write_status(tmp_path, "running")
        with pytest.raises(RuntimeError, match="actively running"):
            await stop_agent(
                "self_agent",
                _registry_with(424242),
                data_dir=tmp_path,
                self_requested=True,
                self_requested_token=token,   # same token, second use
            )

    @pytest.mark.asyncio
    async def test_unconsumable_token_is_refused(self, tmp_path, monkeypatch):
        """If the token cannot be DELETED, it must not authorize.

        Consume-once is the whole guarantee. Authorizing a token that
        survives on disk leaves a replayable capability, so a failed
        ``unlink`` has to deny even when the value matches — and the next
        attempt must be denied too.
        """
        from deep_worker_cli.process import self_restart_token as tok_mod
        from deep_worker_cli.process.self_restart_token import (
            TOKEN_FILENAME,
            mint_self_restart_token,
        )

        _write_status(tmp_path, "running")
        token = mint_self_restart_token(tmp_path)

        def _unlink_fails(self, *a, **k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(tok_mod.Path, "unlink", _unlink_fails)

        killed: list = []
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop.os.kill",
            lambda pid, sig: killed.append((pid, sig)),
        )
        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop._wait_for_exit",
            _immediate_exit,
        )

        with pytest.raises(RuntimeError, match="actively running"):
            await stop_agent(
                "self_agent",
                _registry_with(424242),
                data_dir=tmp_path,
                self_requested=True,
                self_requested_token=token,
            )

        assert killed == [], "an unconsumable token must not signal anything"
        assert (tmp_path / TOKEN_FILENAME).exists(), (
            "precondition: the token really did survive on disk"
        )

        # The surviving token must not work on a second attempt either.
        with pytest.raises(RuntimeError, match="actively running"):
            await stop_agent(
                "self_agent",
                _registry_with(424242),
                data_dir=tmp_path,
                self_requested=True,
                self_requested_token=token,
            )
        assert killed == []

    def test_token_file_is_private(self, tmp_path):
        """The token is a secret; 0600 keeps other uids out."""
        from deep_worker_cli.process.self_restart_token import (
            TOKEN_FILENAME,
            mint_self_restart_token,
        )

        mint_self_restart_token(tmp_path)
        mode = (tmp_path / TOKEN_FILENAME).stat().st_mode & 0o777
        assert mode == 0o600, f"token file is {oct(mode)}, must be 0o600"

    def test_token_is_created_exclusive_at_0600(self, tmp_path, monkeypatch):
        """Path.write_text + later chmod leaves a umask window.

        The capability must be born 0600 via O_CREAT|O_EXCL, not chmod'd
        after a world-readable create.
        """
        import os

        from deep_worker_cli.process import self_restart_token as tok_mod
        from deep_worker_cli.process.self_restart_token import (
            mint_self_restart_token,
        )

        seen: list[tuple[int, int]] = []
        real_open = tok_mod.os.open

        def _spy(path, flags, mode=0o777, *a, **k):
            seen.append((flags, mode))
            return real_open(path, flags, mode, *a, **k)

        monkeypatch.setattr(tok_mod.os, "open", _spy)
        mint_self_restart_token(tmp_path)
        assert seen, "mint must create the token via os.open"
        flags, mode = seen[0]
        assert flags & os.O_CREAT
        assert flags & os.O_EXCL
        assert flags & getattr(os, "O_NOFOLLOW", 0)
        assert mode == 0o600

    @pytest.mark.asyncio
    async def test_driver_forwards_the_token_to_stop_agent(
        self, tmp_path, monkeypatch
    ):
        """The token must survive the trip CLI -> driver -> stop_agent.

        Behavioural: a hard-coded ``self_requested_token=None`` would
        refuse every real self-restart, and this catches it.
        """
        from deep_worker_cli.cli import agent_command

        seen: dict = {}

        async def _spy_stop_agent(name, registry, **kwargs):
            seen.update(kwargs)
            return True

        monkeypatch.setattr(
            "deep_worker_cli.process.agent_stop.stop_agent", _spy_stop_agent
        )
        monkeypatch.setattr(
            agent_command, "load_launch_config", lambda n: {"provider": "x"}
        )
        monkeypatch.setattr(
            agent_command, "merge_restart_config", lambda saved, args: saved
        )
        monkeypatch.setattr(
            agent_command, "_get_agent_data_dir", lambda n: tmp_path
        )

        class _Registry:
            def lookup(self, name):
                return MagicMock(pid=4242)

        monkeypatch.setattr(
            "deep_worker_cli.process.registry.ProcessRegistry", _Registry
        )

        parser = agent_command.create_parser()
        args = parser.parse_args([
            "ag", "--restart", "--self-requested",
            "--self-requested-token", "abc123",
        ])

        try:
            await agent_command.restart_agent(args)
        except Exception:  # noqa: BLE001
            pass

        assert seen.get("self_requested_token") == "abc123", (
            "the --restart driver must forward the capability token; "
            "dropping it refuses every genuine self-restart"
        )


class TestDeliveryBarrier:
    """The driver must not be spawned until the result has been delivered.

    Delivery means the result has been recorded in the model's context.
    Returning from the tool handler establishes no happens-before edge with
    that. Spawning the driver inside the handler lets a separate process
    SIGTERM us while the result is still unrecorded — reproducing the
    dangling ``tool_use`` and the fabricated denial through a narrower
    window.
    """

    def setup_method(self):
        from deep_worker_cli.process.self_restart_barrier import (
            reset_armed_self_restart_for_test,
        )

        reset_armed_self_restart_for_test()

    teardown_method = setup_method

    def test_arming_does_not_spawn(self, monkeypatch):
        """Arming is inert — that is the whole point."""
        from deep_worker_cli.process import self_restart_barrier as barrier

        spawned: list = []
        monkeypatch.setattr(
            barrier.subprocess, "Popen",
            lambda *a, **k: spawned.append(a) or MagicMock(),
        )

        barrier.arm_self_restart(["/bin/true"], "toolu_1")

        assert spawned == [], (
            "arming must NOT spawn the driver; spawning here races the "
            "tool result to the CLI"
        )
        assert barrier.self_restart_armed() is True

    def test_firing_spawns_exactly_once(self, monkeypatch):
        """The barrier fires the armed driver, and only once."""
        from deep_worker_cli.process import self_restart_barrier as barrier

        spawned: list = []
        monkeypatch.setattr(
            barrier.subprocess, "Popen",
            lambda *a, **k: spawned.append(a[0]) or MagicMock(),
        )

        barrier.arm_self_restart(["/bin/true", "--restart", "ag"], "toolu_1")
        assert barrier.fire_armed_self_restart(frozenset({"toolu_1"})) is True
        assert barrier.fire_armed_self_restart(frozenset({"toolu_1"})) is False, (
            "double-fire must be a no-op — two drivers would race to stop "
            "the same process"
        )
        assert spawned == [["/bin/true", "--restart", "ag"]]

    def test_firing_without_arming_is_a_noop(self, monkeypatch):
        """Every delivery calls this; it must be cheap."""
        from deep_worker_cli.process import self_restart_barrier as barrier

        spawned: list = []
        monkeypatch.setattr(
            barrier.subprocess, "Popen",
            lambda *a, **k: spawned.append(a) or MagicMock(),
        )

        assert barrier.fire_armed_self_restart(frozenset({"toolu_x"})) is False
        assert spawned == []

    def test_unrelated_response_must_not_fire_the_restart(self, monkeypatch):
        """THE concurrency case: two tool calls in flight.

        The barrier is called on EVERY delivery. If a concurrent tool's
        delivery fires the restart, this process is stopped while
        ``restart_self``'s own result is still unrecorded — the transcript
        then holds a dangling ``tool_use`` and the CLI fabricates the
        user-rejected denial. Only the armed call's id may fire.
        """
        from deep_worker_cli.process import self_restart_barrier as barrier

        spawned: list = []
        monkeypatch.setattr(
            barrier.subprocess, "Popen",
            lambda *a, **k: spawned.append(a[0]) or MagicMock(),
        )

        barrier.arm_self_restart(
            ["/bin/true", "--restart", "ag"], "toolu_restart"
        )

        # Someone else's result is committed first.
        assert barrier.fire_armed_self_restart(frozenset({"toolu_other"})) is False
        assert spawned == [], (
            "an unrelated tool's delivery must NOT spawn the restart "
            "driver — restart_self's own result is still unrecorded"
        )
        assert barrier.self_restart_armed() is True, (
            "the restart must stay armed for the result it belongs to"
        )

        # ...then ours, which is the one that may fire.
        assert barrier.fire_armed_self_restart(frozenset({"toolu_restart"})) is True
        assert spawned == [["/bin/true", "--restart", "ag"]]

    def test_a_commit_without_our_id_cannot_fire(self):
        """Correlation is mandatory: no id, no restart.

        ``restart_self`` now arms already bound to its own
        ``InjectedToolCallId``, so the old "armed but unbound" window is
        gone by construction. What remains is the invariant that matters:
        a commit that does not carry OUR id — including an empty one, or
        no id set at all — must leave the restart armed.
        """
        from deep_worker_cli.process import self_restart_barrier as barrier

        barrier.arm_self_restart(["/bin/true"], "toolu_mine")
        assert barrier.armed_self_restart_call_id() == "toolu_mine"
        assert barrier.fire_armed_self_restart(frozenset({"toolu_other"})) is False
        assert barrier.fire_armed_self_restart(frozenset()) is False
        assert barrier.fire_armed_self_restart(None) is False
        assert barrier.self_restart_armed() is True

    def test_spawn_failure_is_reported_not_swallowed(self, monkeypatch):
        """The tool already told the model success; this is the last place
        a spawn failure can surface, so it must be loud."""
        from deep_worker_cli.process import self_restart_barrier as barrier

        def _boom(*a, **k):
            raise OSError("no exec for you")

        monkeypatch.setattr(barrier.subprocess, "Popen", _boom)
        barrier.arm_self_restart(["/bin/true"], "toolu_1")

        assert barrier.fire_armed_self_restart(frozenset({"toolu_1"})) is False
        assert barrier.self_restart_armed() is False

    def test_end_of_turn_disarms_and_never_spawns(self, monkeypatch):
        """An undelivered result must NOT be turned into a restart.

        Reaching end of turn still armed means the result never reached
        the CLI. Spawning then stops the process on exactly the path where
        the transcript may already hold a dangling ``tool_use`` — which is
        how the fabricated denial gets written. Disarm loudly instead.
        """
        from deep_worker_cli.process import self_restart_barrier as barrier

        spawned: list = []
        monkeypatch.setattr(
            barrier.subprocess, "Popen",
            lambda *a, **k: spawned.append(a) or MagicMock(),
        )

        barrier.arm_self_restart(["/bin/true", "--restart", "ag"], "toolu_1")

        assert barrier.disarm_undelivered_self_restart() is True
        assert spawned == [], (
            "the end-of-turn path must never spawn the driver — the result "
            "was never confirmed delivered"
        )
        assert barrier.self_restart_armed() is False
        # And a late commit of that id must not resurrect it.
        assert barrier.fire_armed_self_restart(frozenset({"toolu_1"})) is False
        assert spawned == []

    def test_disarm_is_a_noop_when_nothing_is_armed(self):
        """Runs at the end of every turn; must be silent when idle."""
        from deep_worker_cli.process import self_restart_barrier as barrier

        assert barrier.disarm_undelivered_self_restart() is False

    def test_agent_process_turn_teardown_disarms_and_never_fires(self):
        """Teardown may only disarm; firing belongs to the commit listener.

        ``fire_armed_self_restart`` does now appear in agent_process — it
        is handed to ``add_commit_listener`` — so the invariant is checked
        structurally: the only mention is that registration, and it is
        never *called* on the end-of-turn path.
        """
        import ast

        from deep_worker_cli.process import agent_process

        source = Path(agent_process.__file__).read_text()
        assert source.count("disarm_undelivered_self_restart()") == 1

        # The name may only be referenced, never invoked: a bare
        # ``fire_armed_self_restart(...)`` call anywhere in this module
        # would mean something other than a durable commit can spawn the
        # driver.
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                assert name != "fire_armed_self_restart", (
                    f"agent_process must not call fire_armed_self_restart "
                    f"directly (line {node.lineno}) — only the checkpointer's "
                    f"commit listener may fire the restart"
                )


async def _immediate_exit(pid, timeout):  # noqa: ARG001
    return True
