"""``AgentProcess(turn_ending_tool_specs=[...])`` — fail-fast in ``__init__``.

See ``deep_worker_cli/docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md`` §3.5.

The constructor parses the raw spec strings SYNCHRONOUSLY. Doing it in
``.start()`` instead would put the failure inside the swallowing init
``try`` (``agent_process.py``), which flips the agent to
``AgentStatus.ERROR`` and keeps it registered — turning a config typo
into a ghost process rather than a clean startup failure.
"""

from pathlib import Path

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.middleware.send_message_completion import TOOL_NAME
from deep_worker_cli.middleware.tool_completion import (
    DuplicateTurnEndingToolError,
    TurnEndingTool,
    TurnEndingToolFlagError,
)
from deep_worker_cli.process.agent_process import AgentProcess


def _make(tmp_path: Path, **kwargs) -> AgentProcess:
    return AgentProcess(
        name="test_turn_ending_wiring",
        config=AgentConfig(model="deepseek-chat", working_dir=tmp_path),
        rpyc_port=False,
        memory_only=True,
        data_dir=tmp_path / "data",
        pid_file=tmp_path / "agent.pid",
        **kwargs,
    )


def test_ctor_default_is_empty_list(tmp_path):
    agent = _make(tmp_path)
    assert agent._turn_ending_tools == []
    assert agent._turn_ending_tool_specs_raw == []


def test_ctor_accepts_turn_ending_tool_specs(tmp_path):
    agent = _make(tmp_path, turn_ending_tool_specs=["foo"])
    (spec,) = agent._turn_ending_tools
    assert isinstance(spec, TurnEndingTool)
    assert spec.tool_name == "foo"
    assert spec.echo_field is None


def test_ctor_stores_raw_alongside_parsed(tmp_path):
    raw = ["foo", "bar:done"]
    agent = _make(tmp_path, turn_ending_tool_specs=raw)
    assert agent._turn_ending_tool_specs_raw == raw
    assert [s.tool_name for s in agent._turn_ending_tools] == ["foo", "bar"]


def test_ctor_multiple_distinct_specs_survive(tmp_path):
    agent = _make(tmp_path, turn_ending_tool_specs=["foo", "bar:done"])
    assert len(agent._turn_ending_tools) == 2
    bar = agent._turn_ending_tools[1]
    assert bar.is_success({"done": True}) is True
    assert bar.is_success({"done": 1}) is False


# ----------------------------------------------------------------------
# Fail-fast: no resource allocation on a bad spec
# ----------------------------------------------------------------------


@pytest.fixture
def allocation_spies(monkeypatch):
    """Spies on everything ``__init__``/``start()`` would allocate."""
    calls: dict[str, int] = {"registry": 0, "mcp_pool": 0, "checkpointer": 0}

    import deep_worker_cli.process.agent_process as ap

    original_registry = ap.ProcessRegistry

    class _SpyRegistry(original_registry):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            calls["registry"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ap, "ProcessRegistry", _SpyRegistry)

    async def _spy_pool(self):
        calls["mcp_pool"] += 1
        return None

    monkeypatch.setattr(
        AgentProcess, "_start_external_mcp_pool", _spy_pool, raising=False
    )

    if hasattr(ap, "get_checkpointer"):
        def _spy_checkpointer(*args, **kwargs):
            calls["checkpointer"] += 1
            raise AssertionError("checkpointer must not be created")

        monkeypatch.setattr(ap, "get_checkpointer", _spy_checkpointer)

    return calls


def test_ctor_invalid_spec_raises_before_resource_allocation(
    tmp_path, allocation_spies
):
    with pytest.raises(TurnEndingToolFlagError) as exc:
        _make(tmp_path, turn_ending_tool_specs=["bad:spec:trap"])
    assert "bad:spec:trap" in str(exc.value)
    assert allocation_spies == {"registry": 0, "mcp_pool": 0, "checkpointer": 0}


def test_ctor_duplicate_specs_raises_before_resource_allocation(
    tmp_path, allocation_spies
):
    with pytest.raises(DuplicateTurnEndingToolError) as exc:
        _make(tmp_path, turn_ending_tool_specs=["foo", "foo:x"])
    assert "foo" in str(exc.value)
    assert allocation_spies == {"registry": 0, "mcp_pool": 0, "checkpointer": 0}


def test_ctor_builtin_name_raises_before_resource_allocation(
    tmp_path, allocation_spies
):
    with pytest.raises(DuplicateTurnEndingToolError) as exc:
        _make(tmp_path, turn_ending_tool_specs=[TOOL_NAME])
    message = str(exc.value)
    assert TOOL_NAME in message
    assert "built-in" in message
    assert allocation_spies == {"registry": 0, "mcp_pool": 0, "checkpointer": 0}


def test_ctor_builtin_name_with_field_still_rejected(tmp_path):
    """A field does not 'un-collide' a spec from the built-in name."""
    with pytest.raises(DuplicateTurnEndingToolError):
        _make(tmp_path, turn_ending_tool_specs=[f"{TOOL_NAME}:success"])


def test_ctor_multiple_invalid_specs_reports_first(tmp_path):
    with pytest.raises(TurnEndingToolFlagError) as exc:
        _make(tmp_path, turn_ending_tool_specs=["a:b:c", "d:e:f"])
    assert "a:b:c" in str(exc.value)
    assert "d:e:f" not in str(exc.value)
