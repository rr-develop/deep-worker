"""Regression: concurrent SessionManager._persist must not race on a shared
temp file and raise FileNotFoundError.

Incident (observed 2026-05-22): the deepseek agent logged many
``Error in StatusTrackingCallback.on_tool_start/on_tool_end callback:
FileNotFoundError(2, 'No such file or directory')``. Root cause:
``_persist`` wrote to a SHARED temp path ``session_metadata.tmp`` (no per-call
uniqueness, no lock). on_tool_start + on_tool_end + the main turn all fire
``set_status`` → ``_persist`` concurrently (each via ``asyncio.to_thread``).
Two concurrent persists race on the same ``.tmp``: one's ``.replace()`` /
``.unlink()`` removes it under the other → ``FileNotFoundError``. (Swallowed
by LangChain's callback handler, so it was log noise — but it points at a
real atomicity bug in the metadata writer.)

The fix must make concurrent persists safe: each writes to its OWN temp file
(unique name) so they never clobber each other, and the final metadata is
always a complete, valid JSON document.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


def _make_manager() -> SessionManager:
    tmp = tempfile.mkdtemp(prefix="dwtest_persist_race_")
    return SessionManager(data_dir=Path(tmp))


@pytest.mark.asyncio
async def test_concurrent_persist_does_not_raise() -> None:
    """Many overlapping set_status (each persists) must not raise
    FileNotFoundError from a shared temp-file race."""
    sm = _make_manager()

    statuses = [
        AgentStatus.RUNNING,
        AgentStatus.WAITING_INPUT,
        AgentStatus.RUNNING,
        AgentStatus.IDLE,
    ] * 15  # 60 rapid overlapping persists

    # gather drives them concurrently — exactly the on_tool_start/on_tool_end/
    # main-turn overlap that triggered the race.
    await asyncio.gather(*(sm.set_status(s) for s in statuses))

    # Final file must exist and be valid JSON (not half-written / missing).
    data = json.loads(sm.metadata_file.read_text())
    assert data["status"] in {s.value for s in AgentStatus}


@pytest.mark.asyncio
async def test_no_leftover_temp_files() -> None:
    """After concurrent persists, no stray ``.tmp`` files should linger in
    the data dir (each unique temp is consumed by its own replace)."""
    sm = _make_manager()

    await asyncio.gather(*(sm.set_status(AgentStatus.RUNNING) for _ in range(40)))
    # Give any trailing writes a moment.
    await asyncio.sleep(0.05)

    leftovers = list(sm.data_dir.glob("*.tmp"))
    assert leftovers == [], f"leftover temp files: {leftovers}"


@pytest.mark.asyncio
async def test_persist_still_writes_correct_content() -> None:
    """Sanity: a single set_status persists the right status (no regression
    from the atomicity change)."""
    sm = _make_manager()
    await sm.set_status(AgentStatus.WAITING_INPUT)
    data = json.loads(sm.metadata_file.read_text())
    assert data["status"] == AgentStatus.WAITING_INPUT.value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
