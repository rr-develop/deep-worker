from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from deep_worker_cli import storage_paths
from deep_worker_cli.infrastructure_feedback import make_infrastructure_feedback_tool


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_tool_writes_local_append_only_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    root = home / ".deep_worker"
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    monkeypatch.setattr(storage_paths, "deep_worker_home", lambda _home=None: root)

    for agent_name in ("integration-agent", "integration-agent:tm"):
        result = await make_infrastructure_feedback_tool(agent_name).ainvoke(
            {
                "kind": "problem",
                "component": "lsp.serena",
                "summary": "symbol lookup returned no result",
                "details": "waited 90 seconds; api_key=must-not-survive",
                "expected_behavior": "return a result or a bounded error",
            }
        )
        assert result["success"] is True

    directory = root / "infrastructure_feedback"
    report_file = directory / "reports.jsonl"
    records = [json.loads(line) for line in report_file.read_text().splitlines()]
    assert len(records) == 2
    assert [record["agent_name"] for record in records] == [
        "integration-agent",
        "integration-agent:tm",
    ]
    assert all("must-not-survive" not in record["details"] for record in records)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(report_file.stat().st_mode) == 0o600
