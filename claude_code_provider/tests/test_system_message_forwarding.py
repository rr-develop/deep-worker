# pyright: reportPrivateUsage=false, reportUnusedImport=false
"""Tests for SystemMessage forwarding to Claude SDK.

SystemMessage from LangGraph middleware (e.g. ProjectMemoryMiddleware cognitive
hint) MUST reach the Claude Code SDK session. Without this, agents don't receive
critical instructions about available tools like project_memory.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class TestSystemMessageForwarding:
    """SystemMessage MUST be forwarded to Claude SDK — not silently dropped."""

    def test_ainvoke_loop_handles_system_message(self):
        """Guard: ainvoke() must have explicit SystemMessage handling.

        If someone removes the SystemMessage branch from the message loop,
        this test catches it.
        """
        from claude_code_provider import chat_model

        source = Path(chat_model.__file__).read_text()

        # The ainvoke loop must explicitly handle SystemMessage
        assert "isinstance(msg, SystemMessage)" in source, (
            "ainvoke() MUST handle SystemMessage explicitly. "
            "Without this, middleware cognitive hints (e.g. project_memory) "
            "are silently dropped and never reach the agent."
        )

    def test_system_message_passed_to_client_creation(self):
        """Guard: system prompt must be passed to get_or_create_client."""
        from claude_code_provider import client_manager

        source = Path(client_manager.__file__).read_text()

        assert "system_prompt" in source, (
            "ClientManager.get_or_create_client MUST accept system_prompt "
            "parameter to forward middleware SystemMessage to Claude SDK session."
        )


class TestBuildMcpContentBackwardCompat:
    """_build_mcp_content must preserve backward compat for all content types."""

    def test_string_content(self):
        from claude_code_provider.chat_model import _build_mcp_content
        result = _build_mcp_content("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_list_with_image_preserved(self):
        from claude_code_provider.chat_model import _build_mcp_content
        content = [
            {"type": "text", "text": "desc"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
        ]
        result = _build_mcp_content(content)
        assert result[1]["type"] == "image"
        assert result[1]["data"] == "ABC"
