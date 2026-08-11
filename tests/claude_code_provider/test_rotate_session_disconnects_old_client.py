# pyright: reportPrivateUsage=false
"""Regression test for rotate_session() orphaning ClaudeSDKClient subprocesses.

Bug:
    rotate_session() used to bump epoch and reset sent_count but never
    called disconnect() on the old ClaudeSDKClient stored in
    self.clients[thread_id]. The next get_or_create_client() then
    overwrote the slot and the old client's subprocess became unreachable
    with its pipes still open. Each summarization round leaked one
    'claude' process; several leaked processes eventually hit the pipe-I/O
    deadlock (see the internal sandbox pipe-deadlock incident report).

Fix:
    rotate_session() is now async; it pops the old client first and
    awaits client.disconnect() (best-effort) before bumping the epoch.

Additional fix (Issue #119):
    rotate_session() now acquires ``_creation_locks[thread_id]`` so that
    pop+disconnect is serialised against concurrent get_or_create_client()
    for the same thread_id. Without the lock, a concurrent ainvoke could
    read a client from ``self.clients`` (line 263), then have it popped AND
    disconnected by rotate_session before the first coroutine's subsequent
    ``client.query()`` / ``client.receive_response()`` — raising

        CLIConnectionError: Not connected. Call connect() first.

These tests use a hand-written fake ClaudeSDKClient - NOT a Mock - so
the assertions are about real state transitions, not stub recordings.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from deep_worker.providers.claude_code_provider.client_manager import ClientManager
from deep_worker.providers.claude_code_provider.session_state import load_session_state, save_session_state


class FakeClient:
    """Hand-rolled test double: records disconnect() invocations."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


class FakeRaisingClient(FakeClient):
    """A double whose disconnect() raises - exercises the suppress branch."""

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        raise RuntimeError("simulated SDK shutdown failure")


class TestRotateSessionDisconnectsOldClient:
    """rotate_session() must disconnect & drop any prior ClaudeSDKClient."""

    @pytest.mark.asyncio
    async def test_disconnects_old_client_on_rotation(self) -> None:
        manager = ClientManager()
        old = FakeClient("old")
        manager.clients["t1"] = (old, object())  # type: ignore[assignment]

        await manager.rotate_session("t1")

        assert old.disconnect_calls == 1
        assert "t1" not in manager.clients
        assert manager.get_session_epoch("t1") == 1

    @pytest.mark.asyncio
    async def test_rotation_with_no_prior_client_is_noop_then_bump(self) -> None:
        manager = ClientManager()
        new_sid = await manager.rotate_session("fresh")
        assert manager.get_session_epoch("fresh") == 1
        assert new_sid == "fresh__epoch_1"

    @pytest.mark.asyncio
    async def test_disconnect_failure_does_not_block_rotation(self) -> None:
        """A broken disconnect() must NOT stop the epoch from bumping."""
        manager = ClientManager()
        bad = FakeRaisingClient("bad")
        manager.clients["t2"] = (bad, object())  # type: ignore[assignment]

        await manager.rotate_session("t2")

        assert bad.disconnect_calls == 1
        assert "t2" not in manager.clients
        assert manager.get_session_epoch("t2") == 1

    @pytest.mark.asyncio
    async def test_no_leak_across_many_rotations(self) -> None:
        """5 rotations against 5 distinct clients -> 5 disconnects, 0 leftovers."""
        manager = ClientManager()
        seen: list[FakeClient] = []
        for i in range(5):
            c = FakeClient(f"c{i}")
            seen.append(c)
            manager.clients["t3"] = (c, object())  # type: ignore[assignment]
            await manager.rotate_session("t3")

        assert all(c.disconnect_calls == 1 for c in seen)
        assert "t3" not in manager.clients
        assert manager.get_session_epoch("t3") == 5


class FakeSlowClient(FakeClient):
    """Double whose disconnect() blocks until an event is set, so we can
    prove the creation lock serialises rotate_session vs get_or_create_client."""

    def __init__(
        self, name: str, hold: asyncio.Event, entered: asyncio.Event | None = None
    ) -> None:
        super().__init__(name)
        self._hold = hold
        self._entered = entered

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self._entered is not None:
            self._entered.set()
        await self._hold.wait()  # block inside the lock


class TestRotateSessionLock:
    """rotate_session() must acquire _creation_locks to prevent concurrent
    client creation/rotation races."""

    @pytest.mark.asyncio
    async def test_lock_serialises_rotate_and_get_or_create(self) -> None:
        """Concurrent rotate_session + get_or_create_client for the same
        thread_id must be serialised by the creation lock."""
        manager = ClientManager()

        hold = asyncio.Event()
        old = FakeSlowClient("old", hold)
        manager.clients["tX"] = (old, object())  # type: ignore[assignment]

        # Start rotate_session — it will block inside disconnect().
        rotate_task = asyncio.create_task(
            manager.rotate_session("tX")
        )
        # Yield control so rotate_task enters the lock and starts disconnect.
        await asyncio.sleep(0)

        # Before the lock is released, get_or_create_client must NOT proceed
        # into the creation critical section. We prove it by timing out.
        create_task = asyncio.create_task(
            manager.get_or_create_client(
                "tX",
                tools_metadata=[],  # type: ignore[arg-type]
            )
        )
        await asyncio.sleep(0.05)  # ample time for a non-locked task

        # get_or_create_client should still be blocked (not yet in creation).
        # The cache was already popped by rotate_session, so the first check
        # (line 262) returns None. After acquiring the lock (blocked by
        # rotate_task), the double-check (line 271) also finds None.
        assert not create_task.done(), (
            "get_or_create_client must be blocked by the creation lock "
            "while rotate_session is inside the critical section"
        )

        # Release the hold → rotate_task finishes → lock released.
        hold.set()
        await rotate_task
        epoch = manager.get_session_epoch("tX")
        assert epoch == 1

        # Now get_or_create_client can proceed, lock is free.
        client, proxy = await create_task
        assert client is not None
        assert proxy is not None
        # The old client was disconnected exactly once.
        assert old.disconnect_calls == 1

    @pytest.mark.asyncio
    async def test_persisted_invalidation_is_atomic_with_rotation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        manager = ClientManager()
        hold = asyncio.Event()
        entered = asyncio.Event()
        old = FakeSlowClient("poisoned", hold, entered)
        manager.clients["tR"] = (old, object())  # type: ignore[assignment]
        save_session_state("tR", sent_count=9, session_uuid="poisoned-uuid")

        rotate_task = asyncio.create_task(
            manager.rotate_session("tR", invalidate_persisted=True)
        )
        await entered.wait()

        async def create_and_persist_clean_session() -> None:
            async with manager._creation_locks["tR"]:
                save_session_state(
                    "tR", sent_count=1, session_uuid="clean-generation-2"
                )

        writer_task = asyncio.create_task(create_and_persist_clean_session())
        assert manager._creation_locks["tR"].locked()
        hold.set()
        await asyncio.gather(rotate_task, writer_task)

        assert load_session_state("tR") == {
            "sent_count": 1,
            "session_uuid": "clean-generation-2",
        }

    @pytest.mark.asyncio
    async def test_fresh_manager_after_poison_reset_has_no_resume_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("tRestart", sent_count=7, session_uuid="poisoned-uuid")
        manager = ClientManager()

        await manager.rotate_session("tRestart", invalidate_persisted=True)

        restarted_manager = ClientManager()
        assert restarted_manager.get_session_epoch("tRestart") == 0
        assert load_session_state("tRestart") is None
