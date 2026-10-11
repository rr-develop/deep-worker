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


@pytest.mark.asyncio
async def test_get_metadata_hides_cross_provider_config_model(tmp_path: Path) -> None:
    config = AgentConfig(working_dir=tmp_path, provider="codex", model="opus")
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
    assert response["metadata"]["model"] is None


@pytest.mark.asyncio
async def test_get_metadata_hides_cross_provider_dynamic_model(tmp_path: Path) -> None:
    config = AgentConfig(working_dir=tmp_path, provider="codex", model=None)
    config.set_dynamic_display_model("opus")
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
    assert response["metadata"]["model"] is None


# ---------------------------------------------------------------------------
# Past incident: Claude Code aliases were scrubbed from metadata.
#
# ``exposed_get_metadata`` filters config.model / dynamic_display_model /
# resolved_model through ``model_can_belong_to_provider``. While that
# predicate carried the stale four-token claude-code-sdk whitelist, an agent
# genuinely running ``fable`` reported model=None — the UI showed nothing,
# or a stale value, for a perfectly healthy agent.
#
# These exercise the REAL service method (not a copy of its filter chain),
# so they still catch it if the precedence order inside get_metadata changes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metadata_surfaces_claude_code_fable(tmp_path: Path) -> None:
    """The exact display symptom from that incident."""
    config = AgentConfig(working_dir=tmp_path, provider="claude-code-sdk", model="fable")
    process = SimpleNamespace(
        config=config,
        started_at=datetime.now(),
        rpyc_port=18861,
    )
    service = AgentService()
    service._agent_process = process

    response = json.loads(await service.exposed_get_metadata())

    assert response["success"] is True
    assert response["metadata"]["provider"] == "claude-code-sdk"
    assert response["metadata"]["model"] == "fable", (
        "an agent running fable must report it, not a blank/stale model"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alias",
    ["fable", "best", "default", "opusplan", "sonnet[1m]", "opus[1m]",
     "claude-fable-5"],
)
async def test_get_metadata_surfaces_every_documented_alias(
    tmp_path: Path, alias: str
) -> None:
    config = AgentConfig(working_dir=tmp_path, provider="claude-code-sdk", model=alias)
    process = SimpleNamespace(
        config=config,
        started_at=datetime.now(),
        rpyc_port=18861,
    )
    service = AgentService()
    service._agent_process = process

    response = json.loads(await service.exposed_get_metadata())

    assert response["metadata"]["model"] == alias


@pytest.mark.asyncio
async def test_get_metadata_surfaces_dynamic_fable(tmp_path: Path) -> None:
    """A fable picked up at runtime (``/model fable``) must also show."""
    config = AgentConfig(working_dir=tmp_path, provider="claude-code-sdk", model=None)
    config.set_dynamic_display_model("fable")
    process = SimpleNamespace(
        config=config,
        started_at=datetime.now(),
        rpyc_port=18861,
    )
    service = AgentService()
    service._agent_process = process

    response = json.loads(await service.exposed_get_metadata())

    assert response["metadata"]["model"] == "fable"


@pytest.mark.asyncio
async def test_get_metadata_claude_code_without_model_is_none(tmp_path: Path) -> None:
    """No pinned model ⇒ None (the claude CLI resolves its own default).

    Regression guard for the default change: metadata must NOT invent
    ``sonnet`` for an agent that never asked for it.
    """
    config = AgentConfig(working_dir=tmp_path, provider="claude-code-sdk", model=None)
    process = SimpleNamespace(
        config=config,
        started_at=datetime.now(),
        rpyc_port=18861,
    )
    service = AgentService()
    service._agent_process = process

    response = json.loads(await service.exposed_get_metadata())

    assert response["metadata"]["model"] is None


@pytest.mark.asyncio
async def test_get_metadata_still_hides_foreign_model_on_claude_code(
    tmp_path: Path,
) -> None:
    """Widening the alias set must not disable the cross-provider filter."""
    config = AgentConfig(
        working_dir=tmp_path, provider="claude-code-sdk", model="grok-composer-2.5-fast"
    )
    process = SimpleNamespace(
        config=config,
        started_at=datetime.now(),
        rpyc_port=18861,
    )
    service = AgentService()
    service._agent_process = process

    response = json.loads(await service.exposed_get_metadata())

    assert response["metadata"]["model"] is None
