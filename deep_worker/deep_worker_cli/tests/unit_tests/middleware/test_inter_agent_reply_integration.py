"""Integration-ish tests for InterAgentReplyMiddleware wiring.

These tests verify that:

1. ``create_cli_agent`` registers the middleware unconditionally.
2. The middleware is ordered AFTER SendMessageCompletionMiddleware.

We avoid spinning up a real LangGraph — we monkeypatch ``create_deep_agent``
to capture kwargs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class TestCreateCliAgentWiring:

    def test_middleware_is_registered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from deep_worker_cli import agent as agent_mod
        from deep_worker_cli.middleware.inter_agent_reply import (
            InterAgentReplyMiddleware,
        )

        import deepagents as da_mod

        captured: dict[str, Any] = {}

        def _fake_create_deep_agent(**kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Fake:
                def with_config(self, _cfg: Any) -> _Fake:
                    return self

            return _Fake()

        monkeypatch.setattr(da_mod, "create_deep_agent", _fake_create_deep_agent)

        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        fake_model = FakeListChatModel(responses=["ok"])

        agent_mod.create_cli_agent(
            model=fake_model,
            assistant_id="test_inter_agent_reply",
            tools=[],
            auto_approve=False,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            project_memory_dir=tmp_path,
        )

        middleware = captured.get("middleware") or []
        types = [type(m).__name__ for m in middleware]
        assert "InterAgentReplyMiddleware" in types, (
            f"InterAgentReplyMiddleware must be in create_deep_agent's "
            f"middleware list; got {types}"
        )

    def test_middleware_registered_after_send_message_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from deep_worker_cli.middleware.inter_agent_reply import (
            InterAgentReplyMiddleware,
        )
        from deep_worker_cli.middleware.send_message_completion import (
            SendMessageCompletionMiddleware,
        )

        import deepagents as da_mod

        captured: dict[str, Any] = {}

        def _fake_create_deep_agent(**kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Fake:
                def with_config(self, _cfg: Any) -> _Fake:
                    return self

            return _Fake()

        monkeypatch.setattr(da_mod, "create_deep_agent", _fake_create_deep_agent)

        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        fake_model = FakeListChatModel(responses=["ok"])

        from deep_worker_cli import agent as agent_mod

        agent_mod.create_cli_agent(
            model=fake_model,
            assistant_id="test_inter_agent_reply2",
            tools=[],
            auto_approve=False,
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            project_memory_dir=tmp_path,
        )

        middleware = captured.get("middleware") or []
        send_completion_idx = next(
            i for i, m in enumerate(middleware)
            if isinstance(m, SendMessageCompletionMiddleware)
        )
        inter_agent_reply_idx = next(
            i for i, m in enumerate(middleware)
            if isinstance(m, InterAgentReplyMiddleware)
        )
        assert inter_agent_reply_idx > send_completion_idx, (
            f"InterAgentReplyMiddleware (index {inter_agent_reply_idx}) should be "
            f"AFTER SendMessageCompletionMiddleware (index {send_completion_idx})"
        )
