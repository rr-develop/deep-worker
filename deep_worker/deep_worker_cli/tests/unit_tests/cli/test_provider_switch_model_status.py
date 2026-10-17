import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.cli.agent_command import (
    build_launch_config,
    create_parser,
    merge_restart_config,
)
from deep_worker_cli.process.agent_process import _make_resolved_model_handler
from deep_worker_cli.rpyc_service.agent_service import AgentService
from deep_worker_cli.session.session_manager import SessionManager


@pytest.mark.asyncio
@pytest.mark.parametrize("failover", [False, True])
async def test_provider_switch_reports_and_persists_the_provider_answer(
    tmp_path, monkeypatch, failover
):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    argv = ["probe", "--restart", "--provider", "codex"]
    if failover:
        argv.append("--no-update-preferred")
    args = create_parser().parse_args(argv)
    saved = {
        "provider": "claude-code",
        "model": "claude-opus-5",
        "preferred_provider": "claude-code",
        "preferred_model": "claude-opus-5",
    }
    effective = merge_restart_config(saved, args, explicit={"provider"})
    launch = build_launch_config(effective)
    config = AgentConfig(
        working_dir=tmp_path, provider=effective.provider, model=effective.model
    )
    events = []

    async def notify(payload):
        events.append(payload)

    manager = SessionManager(tmp_path / "probe", launch_config=launch, notifier=notify)
    handler = _make_resolved_model_handler(config=config, session_manager=manager)
    await handler("gpt-provider-default")
    service = AgentService()
    service._agent_process = SimpleNamespace(
        name="probe",
        config=config,
        launch_config=launch,
        session_manager=manager,
        started_at=datetime.now(),
        rpyc_port=18861,
    )
    metadata = json.loads(await service.exposed_get_metadata())["metadata"]
    stored = json.loads(manager.metadata_file.read_text())["launch_config"]
    assert metadata["model"] == "gpt-provider-default"
    assert stored["model"] is None
    assert stored.get("resolved_model") == "gpt-provider-default", (
        "the provider answer must be persisted separately from model=None"
    )
    assert stored.get("resolved_model_provider") == "codex"
    assert events[-1]["model"] == "gpt-provider-default"
    if failover:
        assert stored["preferred_model"] == "claude-opus-5"


def test_unpinned_failover_drops_a_compatible_but_unrequested_model(monkeypatch):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    args = create_parser().parse_args(
        ["probe", "--restart", "--provider", "codex", "--no-update-preferred"]
    )
    saved = {"provider": "openai", "model": "gpt-old", "preferred_provider": "openai"}
    effective = merge_restart_config(saved, args, explicit={"provider"})
    assert effective.model is None, (
        "an unpinned failover must use the new provider's default"
    )
    assert build_launch_config(effective)["model"] is None


def test_explicit_compatible_model_survives_provider_switch():
    args = create_parser().parse_args(
        ["probe", "--restart", "--provider", "codex", "--model", "gpt-explicit"]
    )
    effective = merge_restart_config(
        {"provider": "claude-code", "model": "claude-opus-5"},
        args,
        explicit={"provider", "model"},
    )
    assert effective.model == "gpt-explicit"
    assert build_launch_config(effective)["model"] == "gpt-explicit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale", ["provider", "previous_launch", "task_manager", "other_agent"]
)
async def test_status_never_reuses_a_foreign_or_previous_launch_model(
    tmp_path, monkeypatch, stale
):
    from deep_worker_cli import storage_paths

    monkeypatch.setenv("DEEP_WORKER_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    config = AgentConfig(working_dir=tmp_path, provider="codex", model=None)
    started = datetime.now(timezone.utc)
    record = {
        "provider": "codex",
        "model": "gpt-old",
        "kind": "agent",
        "ts": (started + timedelta(seconds=1)).isoformat(),
    }
    if stale == "provider":
        record.update(provider="claude-code", model="claude-opus-5")
    elif stale == "previous_launch":
        record["ts"] = (started - timedelta(seconds=1)).isoformat()
    elif stale == "task_manager":
        record["kind"] = "tm"
    name = "another" if stale == "other_agent" else "probe"
    root = storage_paths.agent_root(name)
    (root / "token_stats.jsonl").write_text(json.dumps(record) + "\n")
    legacy = tmp_path / ".deep_worker" / name
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "token_stats.jsonl").write_text(json.dumps(record) + "\n")
    service = AgentService()
    service._agent_process = SimpleNamespace(
        name="probe",
        config=config,
        started_at=started,
        rpyc_port=18861,
        launch_config={"provider": "codex", "model": None},
    )
    metadata = json.loads(await service.exposed_get_metadata())["metadata"]
    assert metadata["model"] is None, (
        "old/foreign/TM statistics are not this launch's model"
    )
    assert metadata["model_source"] == "provider_default"


@pytest.mark.asyncio
async def test_status_accepts_current_agent_provider_statistics(tmp_path, monkeypatch):
    from deep_worker_cli import storage_paths

    monkeypatch.setenv("DEEP_WORKER_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    config = AgentConfig(working_dir=tmp_path, provider="codex", model=None)
    started = datetime.now(timezone.utc)
    record = {
        "provider": "codex",
        "model": "gpt-current",
        "kind": "agent",
        "ts": (started + timedelta(seconds=1)).isoformat(),
    }
    (storage_paths.agent_root("probe") / "token_stats.jsonl").write_text(
        json.dumps(record) + "\n"
    )
    service = AgentService()
    service._agent_process = SimpleNamespace(
        name="probe",
        config=config,
        started_at=started,
        rpyc_port=18861,
        launch_config={"provider": "codex", "model": None},
    )
    metadata = json.loads(await service.exposed_get_metadata())["metadata"]
    assert metadata["model"] == "gpt-current"
