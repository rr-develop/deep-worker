"""Unit tests for the claude-code §5.2 preflight (review item 7).

Covers:
  * node >= 22 enforcement (cached check — one quick subprocess, never a
    blocking 30s probe during async agent init),
  * adapter version read from trusted package.json metadata (NO node spawn
    in ``build_profile`` — nothing blocks the event loop at init),
  * async live smoke helper (non-blocking, for diagnostics),
  * describe_preflight never spawns node.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker.providers.claude_code_provider import profile as prof


@pytest.fixture(autouse=True)
def _clear_node_cache():
    prof._NODE_MAJOR_CACHE = None
    prof._ASYNC_PREFLIGHT = None
    yield
    prof._NODE_MAJOR_CACHE = None
    prof._ASYNC_PREFLIGHT = None


class TestNodeVersionCheck:
    def test_node_22_passes(self) -> None:
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "v22.22.2\n"
        with patch("subprocess.run", return_value=proc) as run:
            assert prof._node_version_ok() is True
        run.assert_called_once()  # cached after first call
        # Second call hits the cache — no extra subprocess.
        with patch("subprocess.run", return_value=proc) as run2:
            assert prof._node_version_ok() is True
        run2.assert_not_called()

    def test_node_18_fails(self) -> None:
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "v18.20.4\n"
        with patch("subprocess.run", return_value=proc):
            assert prof._node_version_ok() is False

    def test_node_missing_fails(self) -> None:
        with patch(
            "subprocess.run", side_effect=FileNotFoundError("node")
        ):
            assert prof._node_version_ok() is False

    def test_node_timeout_fails_fast(self) -> None:
        with patch("subprocess.run", side_effect=TimeoutError):
            assert prof._node_version_ok() is False

    def test_garbage_version_fails(self) -> None:
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "not-a-version\n"
        with patch("subprocess.run", return_value=proc):
            assert prof._node_version_ok() is False


class TestBuildProfilePreflight:
    def test_ok_path_with_pinned_metadata(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        with patch.object(prof, "_node_version_ok", return_value=True), patch.object(
            prof, "ADAPTER_ENTRY", entry
        ), patch.object(
            prof, "_adapter_version_from_package", return_value=prof.PINNED_ADAPTER_VERSION
        ), patch.object(
            prof, "subprocess"
        ) as subp:
            profile = prof.build_profile()
            assert profile.provider == "claude-code"
            # No node spawn during init — subprocess never touched.
            subp.run.assert_not_called()

    def test_old_node_raises_when_cache_warm(self) -> None:
        # The node gate is enforced sync ONLY when a previous sync probe
        # warmed the cache (fail-fast in sync contexts); with a cold cache
        # it is deferred to the async startup preflight.
        entry = MagicMock()
        entry.is_file.return_value = True
        prof._NODE_MAJOR_CACHE = (18,)
        with patch.object(prof, "ADAPTER_ENTRY", entry), patch.object(
            prof, "_adapter_version_from_package", return_value=prof.PINNED_ADAPTER_VERSION
        ):
            with pytest.raises(prof.AcpAdapterError, match="node >= 22"):
                prof.build_profile()

    def test_cold_cache_build_profile_never_spawns_node(self) -> None:
        """Review item 7 core: with a COLD node cache (async agent init),
        build_profile must NOT spawn node — the node gate is deferred to the
        async preflight at the first ainvoke."""
        entry = MagicMock()
        entry.is_file.return_value = True
        with patch.object(prof, "ADAPTER_ENTRY", entry), patch.object(
            prof, "_adapter_version_from_package", return_value=prof.PINNED_ADAPTER_VERSION
        ), patch(
            "deep_worker.providers.claude_code_provider.profile.subprocess.run",
            side_effect=AssertionError("node must not be spawned in build_profile"),
        ) as run:
            profile = prof.build_profile()
            assert profile.provider == "claude-code"
            run.assert_not_called()

    def test_missing_entry_raises_install_diagnostics(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = False
        with patch.object(prof, "_node_version_ok", return_value=True), patch.object(
            prof, "ADAPTER_ENTRY", entry
        ):
            with pytest.raises(prof.AcpAdapterError, match="npm install"):
                prof.build_profile()

    def test_version_mismatch_raises(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        with patch.object(prof, "_node_version_ok", return_value=True), patch.object(
            prof, "ADAPTER_ENTRY", entry
        ), patch.object(
            prof, "_adapter_version_from_package", return_value="9.9.9"
        ):
            with pytest.raises(prof.AcpAdapterError, match="version mismatch"):
                prof.build_profile()

    def test_verify_false_skips_version_check_only(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        with patch.object(prof, "_node_version_ok", return_value=False), patch.object(
            prof, "ADAPTER_ENTRY", entry
        ):
            # verify=False still requires the entry; node/version skipped.
            profile = prof.build_profile(verify=False)
            assert profile.provider == "claude-code"


class TestAsyncSmoke:
    @pytest.mark.integration  # needs the claude-agent-acp adapter installed
    def test_async_smoke_uses_subprocess_exec(self) -> None:
        """The live adapter --version probe is ASYNC (create_subprocess_exec)
        — it never blocks the event loop with a sync subprocess.run."""

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"0.68.0\n", b"")

        proc = _FakeProc()
        with patch(
            "deep_worker.providers.claude_code_provider.profile.asyncio."
            "create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as spawn:
            version = asyncio.run(prof._smoke_check_version_async())
        assert version == "0.68.0"
        spawn.assert_awaited_once()


class TestDescribePreflight:
    def test_never_spawns_node(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        prof._NODE_MAJOR_CACHE = (22,)  # warm cache from a prior probe
        with patch.object(
            prof, "ADAPTER_ENTRY", entry
        ), patch.object(
            prof, "_adapter_version_from_package", return_value=prof.PINNED_ADAPTER_VERSION
        ), patch(
            "deep_worker.providers.claude_code_provider.profile.subprocess.run",
            side_effect=AssertionError("describe_preflight must not spawn node"),
        ) as subp:
            info = prof.describe_preflight()
            assert info["ok"] is True
            assert info["reported_version"] == prof.PINNED_ADAPTER_VERSION
            assert info["node_major"] == 22
            subp.run.assert_not_called()  # trusted metadata + cache only

    def test_cold_cache_reports_node_unchecked(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        with patch.object(prof, "ADAPTER_ENTRY", entry), patch.object(
            prof, "_adapter_version_from_package", return_value=prof.PINNED_ADAPTER_VERSION
        ), patch(
            "deep_worker.providers.claude_code_provider.profile.subprocess.run",
            side_effect=AssertionError("describe_preflight must not spawn node"),
        ):
            info = prof.describe_preflight()
            assert info["node_major"] is None
            assert info["ok"] is True  # node gate deferred, not failed


class TestLimitMarkersStillWired:
    def test_extra_limit_markers_match_claude_banner(self) -> None:
        assert any(
            p.search("You've hit your session limit · resets 3:00 AM")
            for p in prof.EXTRA_LIMIT_MARKERS
        )


class _SlowFakeProc:
    """Fake subprocess whose communicate() yields to the event loop (sleeps),
    so a responsiveness test can observe other tasks ticking meanwhile."""

    def __init__(self, stdout: bytes = b"v22.22.2\n", *, delay: float = 0.05) -> None:
        self.stdout_bytes = stdout
        self.delay = delay
        self.returncode = 0

    async def communicate(self):
        await asyncio.sleep(self.delay)
        return (self.stdout_bytes, b"")


class TestAsyncPreflight:
    """Review item 7: the startup node probe and the full §5.2 preflight run
    ASYNC (create_subprocess_exec) — never a blocking subprocess.run."""

    def test_node_major_async_uses_subprocess_exec_and_caches(self) -> None:
        proc = _SlowFakeProc()
        with patch(
            "deep_worker.providers.claude_code_provider.profile.asyncio."
            "create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as spawn:
            major = asyncio.run(prof._node_major_async())
        assert major == 22
        spawn.assert_awaited_once()
        # Second call hits the shared cache — no second spawn.
        with patch(
            "deep_worker.providers.claude_code_provider.profile.asyncio."
            "create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as spawn2:
            assert asyncio.run(prof._node_version_ok_async()) is True
        spawn2.assert_not_awaited()

    def test_node_async_timeout_counts_as_missing(self) -> None:
        with patch(
            "deep_worker.providers.claude_code_provider.profile.asyncio."
            "create_subprocess_exec",
            side_effect=FileNotFoundError("node"),
        ):
            assert asyncio.run(prof._node_version_ok_async()) is False

    def test_preflight_async_full_ok_and_cached(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        # First spawn is the node probe (v22), second is the adapter smoke
        # (0.68.0) — side_effect hands them out in call order.
        procs = [
            _SlowFakeProc(stdout=b"v22.22.2\n"),
            _SlowFakeProc(stdout=b"0.68.0\n"),
        ]
        with patch.object(prof, "ADAPTER_ENTRY", entry), patch(
            "deep_worker.providers.claude_code_provider.profile.asyncio."
            "create_subprocess_exec",
            new=AsyncMock(side_effect=procs),
        ) as spawn:
            asyncio.run(prof.preflight_async())
            asyncio.run(prof.preflight_async())  # cached — no second probe
        assert spawn.await_count == 2  # node + adapter smoke, once total

    def test_preflight_async_fails_on_bad_node(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        proc = _SlowFakeProc(stdout=b"v18.20.4\n")
        with patch.object(prof, "ADAPTER_ENTRY", entry), patch(
            "deep_worker.providers.claude_code_provider.profile.asyncio."
            "create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            with pytest.raises(prof.AcpAdapterError, match="node >= 22"):
                asyncio.run(prof.preflight_async())
        # Failure resets the cache so the next start retries.
        assert prof._ASYNC_PREFLIGHT is None

    def test_preflight_async_keeps_event_loop_responsive(self) -> None:
        """While the async preflight awaits its subprocess, other asyncio
        tasks must keep running (a sync spawn would freeze them)."""
        entry = MagicMock()
        entry.is_file.return_value = True
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.002)

        async def _run() -> None:
            t = asyncio.create_task(_ticker())
            await asyncio.sleep(0.01)
            ticks_before = ticks
            with patch.object(prof, "ADAPTER_ENTRY", entry), patch(
                "deep_worker.providers.claude_code_provider.profile.asyncio."
                "create_subprocess_exec",
                new=AsyncMock(return_value=_SlowFakeProc(delay=0.2)),
            ):
                await prof.preflight_async()
            await asyncio.sleep(0.01)
            t.cancel()
            assert ticks > ticks_before, "event loop froze during async preflight"

        asyncio.run(_run())


class TestModelInitResponsive:
    """Review item 7 end-to-end: constructing ClaudeCodeAcpChatModel inside
    a running event loop must not block it (no sync node spawn)."""

    def _model_ctor(self):
        from deep_worker.providers.claude_code_provider import (
            ClaudeCodeAcpChatModel,
        )

        return ClaudeCodeAcpChatModel(model=None, auto_approve=False)

    def test_construction_never_spawns_node_sync(self) -> None:
        entry = MagicMock()
        entry.is_file.return_value = True
        with patch.object(prof, "ADAPTER_ENTRY", entry), patch.object(
            prof, "_adapter_version_from_package", return_value=prof.PINNED_ADAPTER_VERSION
        ), patch(
            "deep_worker.providers.claude_code_provider.profile.subprocess.run",
            side_effect=AssertionError("model init must not spawn node"),
        ) as run:
            model = self._model_ctor()
        assert model._profile.provider == "claude-code"
        run.assert_not_called()

    def test_event_loop_stays_responsive_during_init(self) -> None:
        """A ticker task keeps ticking while the (sync, fast) model
        constructor runs inside the event loop — and the constructor does
        not take the ~seconds a blocking node spawn would."""
        entry = MagicMock()
        entry.is_file.return_value = True
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.002)

        async def _run() -> None:
            t = asyncio.create_task(_ticker())
            await asyncio.sleep(0.01)
            with patch.object(prof, "ADAPTER_ENTRY", entry), patch.object(
                prof, "_adapter_version_from_package", return_value=prof.PINNED_ADAPTER_VERSION
            ):
                model = self._model_ctor()  # sync ctor, must be fast
            await asyncio.sleep(0.01)
            t.cancel()
            assert model is not None
            assert ticks > 0, "event loop froze during model construction"

        asyncio.run(_run())
