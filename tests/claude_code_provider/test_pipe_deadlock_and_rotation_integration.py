# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""End-to-end integration test for the pipe-deadlock + rotation fixes.

NO MOCKS of either the SDK transport or the ClientManager. We use real
fake-claude subprocesses (a Python script masquerading as the CLI) plus the
real ``SubprocessCLITransport`` (patched in the local ``claude_sdk``
source).

We deliberately drop ``ClaudeSDKClient`` here because its ``connect()`` does
a control-protocol initialize handshake that the fake CLI would have to
emulate; the test would then exercise a lot of unrelated SDK plumbing.
Instead, we plug a ``SubprocessCLITransport`` directly into a tuple
``(transport, sentinel_proxy)`` and store it in ``manager.clients`` as if
it were a real ``ClaudeSDKClient``. Since ``rotate_session()`` only calls
``client.disconnect()`` on whatever is in the slot, we model that by
giving the stored object an async ``disconnect()`` that closes the
transport. The transport — the part actually exercised by Fix A — is real.

What this proves end-to-end:
  - SDK Fix A: closing a transport whose subprocess (a) has a full stdout
    pipe and (b) ignores SIGTERM must return in well under 2 s.
  - Fix B: ``rotate_session()`` actually drives that
    ``disconnect()`` instead of orphaning the subprocess; after N
    rotations zero PIDs remain in the process table.
"""

from __future__ import annotations

import os
import stat
import sys
import textwrap
import time
from pathlib import Path

import pytest

_SDK_SRC = Path(__file__).parent.parent.parent.parent / "claude_sdk" / "src"
if _SDK_SRC.exists() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from claude_agent_sdk._internal.transport.subprocess_cli import (  # noqa: E402
    SubprocessCLITransport,
)
from claude_agent_sdk.types import ClaudeAgentOptions  # noqa: E402


_FAKE_CLI_BODY = textwrap.dedent(
    """
    import os, signal, sys
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.stdout.write('{"type":"system","subtype":"init"}\\n')
    sys.stdout.flush()
    try:
        sys.stdin.read()
    except Exception:
        pass
    payload = ('{"type":"text","text":"' + ('x' * 1023) + '"}\\n') * 256
    try:
        sys.stdout.write(payload)
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)
    sys.exit(0)
    """
)


def _write_fake_cli(tmp_path: Path) -> Path:
    cli = tmp_path / "fake_claude"
    cli.write_text(f"#!{sys.executable}\n" + _FAKE_CLI_BODY)
    cli.chmod(
        stat.S_IRWXU
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    return cli


@pytest.fixture(autouse=True)
def _skip_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")


class _TransportBackedClient:
    """Test double: a real transport wearing a ClaudeSDKClient-shaped hat.

    Only the surface that ``ClientManager.rotate_session()`` touches —
    ``async def disconnect(self)`` — is implemented. Inside, it really
    closes the real transport, so SDK Fix A is exercised for real.
    """

    def __init__(self, transport: SubprocessCLITransport) -> None:
        self._transport = transport
        self.pid: int | None = None

    async def connect(self) -> None:
        await self._transport.connect()
        # Record the spawned subprocess PID so the test can check for orphans.
        self.pid = self._transport._process.pid  # type: ignore[union-attr]

    async def disconnect(self) -> None:
        await self._transport.close()


@pytest.mark.asyncio
async def test_transport_close_is_fast_under_pipe_deadlock(
    tmp_path: Path,
) -> None:
    """Direct SDK Fix A check at the integration level (no mocks)."""
    cli = _write_fake_cli(tmp_path)

    transport = SubprocessCLITransport(
        prompt="ignored", options=ClaudeAgentOptions(cli_path=str(cli))
    )
    await transport.connect()

    t0 = time.monotonic()
    await transport.close()
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, (
        f"transport.close() took {elapsed:.2f}s — SDK Fix A "
        f"(close stdout pipe FD before process.wait) is missing or regressed."
    )


@pytest.mark.asyncio
async def test_rotation_disconnects_and_reaps_every_subprocess(
    tmp_path: Path,
) -> None:
    """Fix B end-to-end: 5 rotations → 0 orphan subprocesses."""
    from deep_worker.providers.claude_code_provider.client_manager import ClientManager

    cli = _write_fake_cli(tmp_path)
    manager = ClientManager()
    thread_id = "integ_thread"
    pids: list[int] = []

    for _ in range(5):
        transport = SubprocessCLITransport(
            prompt="ignored", options=ClaudeAgentOptions(cli_path=str(cli))
        )
        client = _TransportBackedClient(transport)
        await client.connect()
        assert client.pid is not None
        pids.append(client.pid)
        manager.clients[thread_id] = (client, object())  # type: ignore[assignment]

        await manager.rotate_session(thread_id)

    # Give the OS a moment to reap each subprocess via its parent's wait().
    # Without Fix B those subprocesses would never have been wait()-ed on;
    # without Fix A they would still be in S-state on a blocked write().
    # Either failure mode shows up as a still-alive PID below.
    survivors: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            survivors.append(pid)
        except ProcessLookupError:
            pass

    assert not survivors, (
        f"{len(survivors)}/{len(pids)} fake-claude subprocesses survived "
        f"rotation — Fix A or Fix B is missing/regressed. PIDs: {survivors}"
    )
    assert manager.get_session_epoch(thread_id) == 5
    assert thread_id not in manager.clients
