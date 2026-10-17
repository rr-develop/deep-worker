import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


def service_with_agent(started):
    service = AgentService(registry=MockProcessRegistry())
    info = AgentInfo(
        name="probe",
        pid=12345,
        status=AgentStatus.IDLE,
        provider="codex",
        model="unknown",
        rpyc_port=18861,
        working_dir="/tmp",
        started_at=started,
    )
    service._agents["probe"] = info
    return service, info


@pytest.mark.asyncio
async def test_provider_switch_clears_old_web_model_until_the_provider_reports():
    service, info = service_with_agent(datetime.now(timezone.utc))
    info.provider = "claude-code"
    info.model = "claude-opus-5"
    with patch("rpyc_async.utils.helpers.fire_and_forget_call") as call:
        await service._fetch_agent_metadata("probe", MagicMock())
        await call.call_args.kwargs["success_callback"](
            json.dumps(
                {
                    "success": True,
                    "metadata": {
                        "provider": "codex",
                        "model": None,
                        "model_display": "provider default (codex)",
                    },
                }
            )
        )
    assert info.provider == "codex"
    assert info.model == "provider default (codex)"
    with patch("rpyc_async.utils.helpers.fire_and_forget_call") as call:
        await service._fetch_agent_metadata("probe", MagicMock())
        await call.call_args.kwargs["success_callback"](
            json.dumps(
                {
                    "success": True,
                    "metadata": {"provider": "codex", "model": "gpt-current"},
                }
            )
        )
    assert info.model == "gpt-current"


@pytest.mark.parametrize("stale", ["provider", "previous_launch", "task_manager"])
def test_web_stats_model_belongs_to_the_current_launch(tmp_path, monkeypatch, stale):
    from deep_worker_cli.storage_paths import agent_root

    monkeypatch.setenv("DEEP_WORKER_HOME", str(tmp_path))
    started = datetime.now(timezone.utc)
    service, info = service_with_agent(started)
    record = {
        "provider": "codex",
        "kind": "agent",
        "model": "gpt-old",
        "ts": (started + timedelta(seconds=1)).isoformat(),
    }
    if stale == "provider":
        record.update(provider="claude-code", model="claude-opus-5")
    elif stale == "previous_launch":
        record["ts"] = (started - timedelta(seconds=1)).isoformat()
    else:
        record["kind"] = "tm"
    path = agent_root("probe") / "token_stats.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert service._latest_model_from_token_stats("probe") is None
    service._context_model_cache.clear()
    record.update(
        provider="codex",
        kind="agent",
        model="gpt-current",
        ts=(started + timedelta(seconds=1)).isoformat(),
    )
    path.write_text(json.dumps(record) + "\n")
    assert service._latest_model_from_token_stats("probe") == "gpt-current"
    info.started_at = started + timedelta(seconds=2)
    assert service._latest_model_from_token_stats("probe") is None
