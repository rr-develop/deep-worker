"""Unit tests for SHARED AcpClient (review item 6).

The main agent model and the TaskManager model each build an AcpChatModel.
Without sharing each would spawn its OWN CLI subprocess — two adapter
processes for one agent. ``get_global_acp_client`` keys one client per
(provider, cmd) identity; every model in the process resolves to the same
client, so ``start()`` (one spawn) and the initialize/authenticate handshake
run exactly once. ACP sessions remain per-thread (distinct session ids over
the same stdio channel).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deep_worker.providers.acp.acp_client import (
    get_global_acp_client,
    reset_global_acp_clients,
)
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.profile import AcpProfile


def _profile(**overrides) -> AcpProfile:
    kwargs: dict = dict(
        provider="claude-code",
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
        requires_authenticate=False,
    )
    kwargs.update(overrides)
    return AcpProfile(**kwargs)


def _model(profile: AcpProfile, client_state: AcpClientState) -> "object":
    """AcpChatModel WITHOUT constructor side effects; only the fields used by
    ``_ensure_acp_started`` are set."""
    from deep_worker.providers.acp.chat_model import AcpChatModel

    model = object.__new__(AcpChatModel)
    model._profile = profile
    model._client_state = client_state
    model._acp_client = None
    model._acp_started = False
    return model


class TestSharedClientRegistry:
    def test_same_profile_resolves_to_same_client(self) -> None:
        reset_global_acp_clients()
        profile = _profile()
        cs = AcpClientState()
        assert get_global_acp_client(profile, cs) is get_global_acp_client(profile, cs)

    def test_different_profile_gets_different_client(self) -> None:
        reset_global_acp_clients()
        cs = AcpClientState()
        a = get_global_acp_client(_profile(provider="claude-code"), cs)
        b = get_global_acp_client(_profile(provider="other-acp"), cs)
        assert a is not b

    def test_models_share_client_and_handshake_once(self) -> None:
        """Main + TaskManager models resolve to ONE client; the handshake
        runs exactly once (the second model must not re-initialize)."""
        async def _run() -> None:
            reset_global_acp_clients()
            profile = _profile()
            cs = AcpClientState()
            m1 = _model(profile, cs)
            m2 = _model(profile, cs)

            client = get_global_acp_client(profile, cs)
            # start()/initialize() are idempotent via _proc/_initialized;
            # emulate those guards so the mocks reflect the real behaviour.
            spawns: list[int] = []
            handshakes: list[int] = []

            async def _fake_start() -> None:
                if client._proc is None:
                    client._proc = object()
                    spawns.append(1)

            async def _fake_initialize() -> dict:
                if not client._initialized:
                    client._initialized = True
                    handshakes.append(1)
                return {}

            client.start = AsyncMock(side_effect=_fake_start)
            client.initialize = AsyncMock(side_effect=_fake_initialize)
            client.authenticate = AsyncMock(return_value={})

            await m1._ensure_acp_started()
            await m2._ensure_acp_started()

            assert m1._acp_client is client
            assert m2._acp_client is client
            assert len(spawns) == 1  # one spawn, not two
            assert len(handshakes) == 1  # one initialize round-trip
            client.authenticate.assert_not_awaited()

        asyncio.run(_run())

    def test_distinct_threads_keep_distinct_session_ids(self) -> None:
        """Session identity is per-thread (ThreadState), never shared even
        though the client is. Two threads get two different session slots."""
        reset_global_acp_clients()
        cs = AcpClientState()
        st_main = cs.get_or_create("agent-main")
        st_tm = cs.get_or_create("agent-main:tm")
        st_main.session_id = "session-A"
        st_tm.session_id = "session-B"
        assert st_main.session_id != st_tm.session_id

    def test_eof_reuse_respawns_same_shared_client(self) -> None:
        """After an EOF the shared client resets in place (process dropped,
        handshake flags cleared) so a later start() re-spawns — and the
        registry keeps returning the SAME client object."""
        async def _run() -> None:
            reset_global_acp_clients()
            profile = _profile()
            cs = AcpClientState()
            client = get_global_acp_client(profile, cs)

            client._initialized = True
            client._authenticated = True
            client._proc = None  # not running
            client._reader_task = None
            client._on_eof()

            assert client._initialized is False
            assert client._authenticated is False
            # Registry still hands out the same (now reusable) client.
            assert get_global_acp_client(profile, cs) is client

        asyncio.run(_run())

    def test_same_model_recovers_after_eof_with_single_respawn(self) -> None:
        """EOF on the SHARED client must NOT wedge an already-started model.

        Review item: startup must depend on the shared client's liveness, not
        a stale per-model ``_acp_started`` flag. After ``_on_eof()`` resets the
        client (process + handshake dropped), the SAME model's next
        ``_ensure_acp_started()`` must re-spawn and re-handshake exactly once;
        the second shared model rides the recovered client with NO third
        spawn and NO second handshake.
        """
        async def _run() -> None:
            reset_global_acp_clients()
            profile = _profile()
            cs = AcpClientState()
            m1 = _model(profile, cs)
            m2 = _model(profile, cs)

            client = get_global_acp_client(profile, cs)
            spawns: list[int] = []
            handshakes: list[int] = []

            async def _fake_start() -> None:
                if client._proc is None:
                    client._proc = object()
                    spawns.append(1)

            async def _fake_initialize() -> dict:
                if not client._initialized:
                    client._initialized = True
                    handshakes.append(1)
                return {}

            client.start = AsyncMock(side_effect=_fake_start)
            client.initialize = AsyncMock(side_effect=_fake_initialize)
            client.authenticate = AsyncMock(return_value={})

            # First start: ONE spawn + ONE handshake across BOTH models.
            await m1._ensure_acp_started()
            await m2._ensure_acp_started()
            assert m1._acp_client is client
            assert m2._acp_client is client
            assert len(spawns) == 1
            assert len(handshakes) == 1

            # EOF: shared client resets in place (process + handshake gone).
            client._on_eof()
            assert client._proc is None
            assert client._initialized is False
            assert client._authenticated is False

            # SAME model m1 invokes again → exactly ONE respawn + re-handshake.
            await m1._ensure_acp_started()
            assert len(spawns) == 2
            assert len(handshakes) == 2

            # m2 (also already-started) rides the recovered client — NO third
            # spawn, NO second handshake.
            await m2._ensure_acp_started()
            assert len(spawns) == 2
            assert len(handshakes) == 2

        asyncio.run(_run())

    def test_concurrent_first_startup_single_spawn(self) -> None:
        """Review round 3 / hard constraint #5: two models entering startup
        CONCURRENTLY (asyncio.gather) must produce exactly ONE spawn and ONE
        handshake.

        The fake ``start()`` models the check-then-act race: without the
        client's single-flight lock BOTH callers would pass the ``_proc is
        None`` check before either sets it, so both would "spawn". The lock
        lets the first caller own the section; the second waits and finds the
        state already up.
        """
        async def _run() -> None:
            reset_global_acp_clients()
            profile = _profile()
            cs = AcpClientState()
            m1 = _model(profile, cs)
            m2 = _model(profile, cs)

            client = get_global_acp_client(profile, cs)
            spawns: list[int] = []
            handshakes: list[int] = []

            # Barrier semantics: the first caller signals it is INSIDE the
            # start() race window and blocks; we give the second caller time
            # to reach the same window (which, without the lock, it would),
            # then release both. With the lock only ONE caller ever enters
            # start(), so the barrier is tripped exactly once.
            entered = asyncio.Event()
            release = asyncio.Event()

            async def _fake_start() -> None:
                entered.set()
                await release.wait()
                client._proc = object()
                spawns.append(1)

            async def _fake_initialize() -> dict:
                handshakes.append(1)
                client._initialized = True
                return {}

            client.start = AsyncMock(side_effect=_fake_start)
            client.initialize = AsyncMock(side_effect=_fake_initialize)
            client.authenticate = AsyncMock(return_value={})

            gather = asyncio.gather(
                m1._ensure_acp_started(), m2._ensure_acp_started()
            )
            await asyncio.wait_for(entered.wait(), timeout=5.0)
            # Give the second caller a real chance to enter start() — with a
            # lock-free ensure_started it would be waiting in start() right
            # now; with the lock it is parked on _start_lock instead.
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.wait_for(gather, timeout=5.0)

            assert m1._acp_client is client
            assert m2._acp_client is client
            assert len(spawns) == 1  # ONE spawn, not two
            assert len(handshakes) == 1  # ONE initialize round-trip
            client.authenticate.assert_not_awaited()

        asyncio.run(_run())

    def test_concurrent_respawn_after_eof_single_spawn(self) -> None:
        """After an EOF, two already-started models re-entering startup
        CONCURRENTLY still produce exactly ONE respawn + ONE re-handshake."""
        async def _run() -> None:
            reset_global_acp_clients()
            profile = _profile()
            cs = AcpClientState()
            m1 = _model(profile, cs)
            m2 = _model(profile, cs)

            client = get_global_acp_client(profile, cs)
            spawns: list[int] = []
            handshakes: list[int] = []

            entered = asyncio.Event()
            release = asyncio.Event()

            async def _fake_start() -> None:
                entered.set()
                await release.wait()
                client._proc = object()
                spawns.append(1)

            async def _fake_initialize() -> dict:
                handshakes.append(1)
                client._initialized = True
                return {}

            client.start = AsyncMock(side_effect=_fake_start)
            client.initialize = AsyncMock(side_effect=_fake_initialize)
            client.authenticate = AsyncMock(return_value={})

            # First startup: plain (no barrier — release is already set).
            release.set()
            await m1._ensure_acp_started()
            await m2._ensure_acp_started()
            assert len(spawns) == 1
            assert len(handshakes) == 1
            client._on_eof()
            assert client._proc is None
            assert client._initialized is False

            # Concurrent re-entry after EOF → exactly ONE respawn.
            entered.clear()
            release.clear()
            gather = asyncio.gather(
                m1._ensure_acp_started(), m2._ensure_acp_started()
            )
            await asyncio.wait_for(entered.wait(), timeout=5.0)
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.wait_for(gather, timeout=5.0)

            assert len(spawns) == 2  # one initial + one respawn
            assert len(handshakes) == 2

        asyncio.run(_run())

    def test_failed_first_startup_retries_not_poisoned(self) -> None:
        """A FAILED first startup must not poison the client: the next call
        retries cleanly (lock released, flags still cleared) instead of
        inheriting a stuck future."""
        async def _run() -> None:
            reset_global_acp_clients()
            profile = _profile()
            cs = AcpClientState()
            m1 = _model(profile, cs)

            client = get_global_acp_client(profile, cs)
            start_attempts: list[int] = []

            async def _flaky_start() -> None:
                start_attempts.append(1)
                if len(start_attempts) == 1:
                    raise OSError("spawn failed")
                client._proc = object()

            async def _fake_initialize() -> dict:
                client._initialized = True
                return {}

            client.start = AsyncMock(side_effect=_flaky_start)
            client.initialize = AsyncMock(side_effect=_fake_initialize)
            client.authenticate = AsyncMock(return_value={})

            with pytest.raises(OSError, match="spawn failed"):
                await m1._ensure_acp_started()
            # State unchanged after the failure — nothing half-initialized.
            assert client._proc is None
            assert client._initialized is False

            # Next call genuinely retries and succeeds (the lock was released,
            # there is no cached failed future to inherit).
            await m1._ensure_acp_started()
            assert len(start_attempts) == 2
            assert client._proc is not None
            assert client._initialized is True

        asyncio.run(_run())

    def test_start_spawns_single_subprocess_for_two_models(self) -> None:
        """End-to-end: two models starting sequentially result in ONE
        create_subprocess_exec call. The handshake is mocked (a real reply
        would need a live CLI); the spawn itself goes through the real
        ``start()`` so its ``_proc`` idempotence guard is exercised."""
        async def _run() -> None:
            reset_global_acp_clients()
            profile = _profile()
            cs = AcpClientState()
            m1 = _model(profile, cs)
            m2 = _model(profile, cs)

            client = get_global_acp_client(profile, cs)
            client.initialize = AsyncMock(return_value={})
            client.authenticate = AsyncMock(return_value={})

            with patch(
                "deep_worker.providers.acp.acp_client.asyncio.create_subprocess_exec",
                new=AsyncMock(),
            ) as spawn:
                await m1._ensure_acp_started()
                await m2._ensure_acp_started()
                if client._reader_task is not None:
                    client._reader_task.cancel()

            assert spawn.await_count == 1

        asyncio.run(_run())
