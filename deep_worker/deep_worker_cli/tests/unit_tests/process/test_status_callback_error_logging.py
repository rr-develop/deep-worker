from __future__ import annotations

import re
from uuid import UUID
from unittest.mock import AsyncMock

import pytest

from deep_worker_cli.process.status_callback import StatusTrackingCallback
from deep_worker_cli.types import AgentStatus


@pytest.mark.asyncio
async def test_tool_error_log_includes_utc_timestamp_and_run_ids(capsys: pytest.CaptureFixture[str]) -> None:
    session_manager = AsyncMock()
    callback = StatusTrackingCallback(session_manager=session_manager)
    run_id = UUID("11111111-1111-1111-1111-111111111111")
    parent_run_id = UUID("22222222-2222-2222-2222-222222222222")

    await callback.on_tool_error(
        RuntimeError("tool exploded"),
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    stderr = capsys.readouterr().err
    assert re.search(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00\] \[TOOL_ERROR\]", stderr)
    assert "error=tool exploded" in stderr
    assert f"run_id={run_id}" in stderr
    assert f"parent_run_id={parent_run_id}" in stderr
    session_manager.set_status.assert_awaited_once_with(AgentStatus.ERROR)
