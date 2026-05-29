from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.rpyc_service.agent_service import AgentService


@pytest.mark.asyncio
async def test_get_metadata_uses_dynamic_display_model(tmp_path: Path) -> None:
    config = AgentConfig(working_dir=tmp_path, provider="codex", model=None)
    config.set_dynamic_display_model("gpt-dynamic-default")
    process = SimpleNamespace(
        config=config,
        started_at=datetime.now(),
        rpyc_port=18861,
    )
    service = AgentService()
    service._agent_process = process

    response = json.loads(await service.exposed_get_metadata())

    assert response["success"] is True
    assert response["metadata"]["provider"] == "codex"
    assert response["metadata"]["model"] == "gpt-dynamic-default"
