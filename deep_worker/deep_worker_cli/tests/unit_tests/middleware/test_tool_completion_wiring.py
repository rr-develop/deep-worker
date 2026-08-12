"""Wiring tests for the ``turn_ending_tools`` parameter of ``create_cli_agent``.

See design document:
    deep_worker_cli/docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md

We avoid spinning up a real LangGraph — we monkeypatch ``create_deep_agent``
to capture kwargs, following the pattern of
``test_inter_agent_reply_integration.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import deepagents as da_mod

    captured: dict[str, Any] = {}

    def _fake_create_deep_agent(**kwargs: Any) -> Any:
        captured.update(kwargs)

        class _Fake:
            def with_config(self, _cfg: Any) -> _Fake:
                return self

        return _Fake()

    monkeypatch.setattr(da_mod, "create_deep_agent", _fake_create_deep_agent)
    return captured


def _run(tmp_path: Path, assistant_id: str, **extra: Any) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from deep_worker_cli import agent as agent_mod

    agent_mod.create_cli_agent(
        model=FakeListChatModel(responses=["ok"]),
        assistant_id=assistant_id,
        tools=[],
        auto_approve=False,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
        project_memory_dir=tmp_path,
        **extra,
    )


def _completion_middleware(captured: dict[str, Any]):
    from deep_worker_cli.middleware.send_message_completion import (
        SendMessageCompletionMiddleware,
    )

    found = [
        m
        for m in (captured.get("middleware") or [])
        if isinstance(m, SendMessageCompletionMiddleware)
    ]
    assert len(found) == 1, (
        "exactly one SendMessageCompletionMiddleware must be wired; "
        f"got {[type(m).__name__ for m in (captured.get('middleware') or [])]}"
    )
    return found[0]


class TestTurnEndingToolsWiring:
    def test_default_registers_only_the_builtin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured = _capture(monkeypatch)
        _run(tmp_path, "test_tet_default")
        mw = _completion_middleware(captured)
        assert set(mw._by_name) == {"send_message_to_agent"}

    def test_explicit_none_matches_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured = _capture(monkeypatch)
        _run(tmp_path, "test_tet_none", turn_ending_tools=None)
        mw = _completion_middleware(captured)
        assert set(mw._by_name) == {"send_message_to_agent"}

    def test_additional_specs_are_registered(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from deep_worker_cli.middleware.tool_completion import TurnEndingTool

        captured = _capture(monkeypatch)
        _run(
            tmp_path,
            "test_tet_extra",
            turn_ending_tools=[
                TurnEndingTool("submit_fragments", echo_field=None),
            ],
        )
        mw = _completion_middleware(captured)
        assert set(mw._by_name) == {"send_message_to_agent", "submit_fragments"}
        assert mw._by_name["submit_fragments"].echo_field is None

    def test_collision_with_builtin_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from deep_worker_cli.middleware.tool_completion import (
            DuplicateTurnEndingToolError,
            TurnEndingTool,
        )

        _capture(monkeypatch)
        with pytest.raises(DuplicateTurnEndingToolError) as exc:
            _run(
                tmp_path,
                "test_tet_collide",
                turn_ending_tools=[TurnEndingTool("send_message_to_agent")],
            )
        assert "send_message_to_agent" in str(exc.value)

    def test_stack_position_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Registering extra specs must not move the middleware in the stack."""
        from deep_worker_cli.middleware.tool_completion import TurnEndingTool

        captured_plain = _capture(monkeypatch)
        _run(tmp_path, "test_tet_pos_a")
        plain = [type(m).__name__ for m in captured_plain["middleware"]]

        captured_extra = _capture(monkeypatch)
        _run(
            tmp_path,
            "test_tet_pos_b",
            turn_ending_tools=[TurnEndingTool("submit_fragments", echo_field=None)],
        )
        extra = [type(m).__name__ for m in captured_extra["middleware"]]

        assert plain == extra
        assert plain.index("SendMessageCompletionMiddleware") < plain.index(
            "InterAgentReplyMiddleware"
        )


class TestPublicExports:
    def test_types_exported_from_middleware_package(self) -> None:
        from deep_worker_cli.middleware import (
            DuplicateTurnEndingToolError,
            ToolCompletionMiddleware,
            TurnEndingTool,
        )

        assert TurnEndingTool("x").tool_name == "x"
        assert issubclass(DuplicateTurnEndingToolError, ValueError)
        assert ToolCompletionMiddleware().tools == []

    def test_send_message_completion_backwards_compatible_surface(self) -> None:
        """Legacy import paths must keep working after the refactor."""
        from deep_worker_cli.middleware.send_message_completion import (
            TOOL_NAME,
            SendMessageCompletionMiddleware,
            _parse_tool_result,
        )
        from deep_worker_cli.middleware.tool_completion import ToolCompletionMiddleware

        assert TOOL_NAME == "send_message_to_agent"
        assert issubclass(SendMessageCompletionMiddleware, ToolCompletionMiddleware)
        assert _parse_tool_result('{"a": 1}') == {"a": 1}
