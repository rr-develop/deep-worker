"""Web-service tests for the two introspection Full-Dialog modes (§9):

``AgentService.get_agent_dialog_formatted`` must, for
``verbosity="system_prompt"`` and ``verbosity="tools"``, bypass the history
fetch and call the dedicated RPyC methods, then render via the pure
formatters. RPyC root is mocked.
"""

from __future__ import annotations

import json

import pytest

from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


class _AwaitableValue:
    """Wrap a value so ``root.method(...)`` returns an awaitable, matching
    the RPyC async-netref call shape used in production
    (``await root.get_history(...)``)."""

    def __init__(self, value: object) -> None:
        self._value = value

    def __await__(self):  # noqa: ANN204
        async def _coro():  # noqa: ANN202
            return self._value

        return _coro().__await__()


class _FakeRoot:
    def __init__(self, system_prompt_payload=None, tools_payload=None) -> None:
        self._sp = system_prompt_payload
        self._tools = tools_payload
        self.history_called = False

    def get_history(self, limit=None, offset=0):  # noqa: ANN001, ANN201
        self.history_called = True
        return _AwaitableValue(json.dumps({"success": True, "history": []}))

    def get_system_prompt(self):  # noqa: ANN201
        return _AwaitableValue(json.dumps(self._sp))

    def get_tools(self):  # noqa: ANN201
        return _AwaitableValue(json.dumps(self._tools))


class _StaleRoot:
    """Simulates an agent process that PREDATES this feature: its RPyC
    service has no get_system_prompt / get_tools, so attribute access raises
    AttributeError exactly like an RPyC netref would."""

    history_called = False

    def get_history(self, limit=None, offset=0):  # noqa: ANN001, ANN201
        self.history_called = True
        return _AwaitableValue(json.dumps({"success": True, "history": []}))

    def __getattr__(self, name):  # noqa: ANN001, ANN204
        if name in ("get_system_prompt", "get_tools"):
            raise AttributeError(
                f"'BoundAgentService' object has no attribute '{name}'"
            )
        raise AttributeError(name)


@pytest.fixture
def service() -> AgentService:
    return AgentService(registry=MockProcessRegistry())


class TestSystemPromptMode:
    async def test_renders_invariant_system_prompt(self, service: AgentService) -> None:
        root = _FakeRoot(
            system_prompt_payload={
                "success": True,
                "system_prompt": "INVARIANT BODY",
                "provider": "claude-code",
                "model": "opus",
            }
        )
        service._roots["a"] = root  # type: ignore[attr-defined]

        out = await service.get_agent_dialog_formatted("a", verbosity="system_prompt")

        assert "INVARIANT BODY" in out
        assert "provider=claude-code" in out
        assert "model=opus" in out
        assert root.history_called is False  # must NOT fetch history

    async def test_system_prompt_failure_raises(self, service: AgentService) -> None:
        root = _FakeRoot(
            system_prompt_payload={"success": False, "error": "boom"}
        )
        service._roots["a"] = root  # type: ignore[attr-defined]
        with pytest.raises(ValueError):
            await service.get_agent_dialog_formatted("a", verbosity="system_prompt")


class TestToolsMode:
    async def test_renders_tool_list(self, service: AgentService) -> None:
        root = _FakeRoot(
            tools_payload={
                "success": True,
                "provider": "claude-code",
                "tools": [
                    {"name": "execute", "description": "Run a shell command"},
                    {"name": "read_file", "description": "Read a file"},
                ],
            }
        )
        service._roots["a"] = root  # type: ignore[attr-defined]

        out = await service.get_agent_dialog_formatted("a", verbosity="tools")

        assert "- execute: Run a shell command" in out
        assert "- read_file: Read a file" in out
        assert "count=2" in out
        assert root.history_called is False

    async def test_tools_failure_raises(self, service: AgentService) -> None:
        root = _FakeRoot(tools_payload={"success": False, "error": "nope"})
        service._roots["a"] = root  # type: ignore[attr-defined]
        with pytest.raises(ValueError):
            await service.get_agent_dialog_formatted("a", verbosity="tools")


class TestStaleAgentGracefulFallback:
    async def test_system_prompt_stale_agent_no_config_returns_hint(
        self, service: AgentService, monkeypatch
    ) -> None:
        # Stale agent AND no readable session_metadata.json → hint.
        monkeypatch.setattr(
            type(service),
            "_compute_invariant_system_prompt_locally",
            lambda self, name: None,
        )
        service._roots["nonexistent_xyz"] = _StaleRoot()  # type: ignore[attr-defined]
        out = await service.get_agent_dialog_formatted(
            "nonexistent_xyz", verbosity="system_prompt"
        )
        assert "System Prompt (unavailable)" in out
        assert "Restart the agent" in out
        # MUST NOT silently fall back to dumping conversation history.
        assert "--- User ---" not in out
        assert "--- Assistant ---" not in out

    async def test_system_prompt_stale_agent_falls_back_to_local_compute(
        self, service: AgentService, monkeypatch
    ) -> None:
        # Stale agent BUT config readable → compute the invariant prompt
        # web-side; NEVER show history.
        monkeypatch.setattr(
            type(service),
            "_compute_invariant_system_prompt_locally",
            lambda self, name: (
                "--- System Prompt (invariant, provider=claude-code, "
                "model=opus) ---\nLOCAL INVARIANT BODY"
            ),
        )
        service._roots["a"] = _StaleRoot()  # type: ignore[attr-defined]
        out = await service.get_agent_dialog_formatted("a", verbosity="system_prompt")
        assert "LOCAL INVARIANT BODY" in out
        assert "--- System Prompt (invariant" in out
        assert "unavailable" not in out
        assert "--- Summarization ---" not in out
        assert "--- User ---" not in out

    async def test_tools_stale_agent_returns_hint(self, service: AgentService) -> None:
        service._roots["a"] = _StaleRoot()  # type: ignore[attr-defined]
        out = await service.get_agent_dialog_formatted("a", verbosity="tools")
        assert "Available Tools (unavailable)" in out
        assert "Restart the agent" in out
        assert "--- User ---" not in out


class TestLocalSystemPromptComputation:
    def test_reads_metadata_and_renders(
        self, service: AgentService, tmp_path, monkeypatch
    ) -> None:
        import json as _json
        from pathlib import Path

        # Fake HOME with a session_metadata.json for agent "demo".
        agent_dir = tmp_path / Path("~/.deep_worker").name / "demo"
        agent_dir.mkdir(parents=True)
        (agent_dir / "session_metadata.json").write_text(
            _json.dumps(
                {
                    "launch_config": {
                        "working_dir": str(tmp_path / "ws"),
                        "provider": "claude-code",
                        "model": "opus",
                    }
                }
            )
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        out = service._compute_invariant_system_prompt_locally("demo")
        assert out is not None
        assert "--- System Prompt (invariant, provider=claude-code, model=opus) ---" in out
        # The real invariant prompt embeds the agent's identity name.
        assert "demo" in out

    def test_missing_metadata_returns_none(
        self, service: AgentService, tmp_path, monkeypatch
    ) -> None:
        from pathlib import Path

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert service._compute_invariant_system_prompt_locally("nope") is None


class TestHistoryModesUnaffected:
    async def test_final_still_fetches_history(self, service: AgentService) -> None:
        root = _FakeRoot()
        service._roots["a"] = root  # type: ignore[attr-defined]
        out = await service.get_agent_dialog_formatted("a", verbosity="final")
        assert root.history_called is True
        assert isinstance(out, str)
