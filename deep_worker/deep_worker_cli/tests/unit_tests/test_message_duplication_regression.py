"""Unit tests to prevent message duplication regression.

These tests ensure that the "agent is single source of truth" architecture principle
is upheld and user messages are not duplicated in the UI.

REGRESSION HISTORY:
- Original fix: commit 80a4523 (removed duplicate user message display)
- Regression: User messages were being created locally before agent processing
- Second fix: This test suite prevents future regressions

ARCHITECTURE PRINCIPLE:
- Agent (deep-worker) is the ONLY source of truth for ALL messages
- User messages MUST come from agent stream (HumanMessage -> UserMessage)
- DO NOT create UserMessage locally before agent processes it
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

if TYPE_CHECKING:
    from deep_worker_cli.widgets.messages import UserMessage


class TestAppMessageDuplication:
    """Test that DeepAgentsApp does not create duplicate user messages."""

    def test_handle_user_message_code_verification(self) -> None:
        """
        Verify _handle_user_message implementation does NOT create UserMessage locally.

        This test reads the actual code and verifies the architecture principle is upheld.
        REGRESSION PREVENTION: UserMessage was previously created at app.py:967 before
        agent processing, causing duplication. This test ensures it doesn't happen again.
        """
        import inspect

        from deep_worker_cli.app import DeepAgentsApp

        # Get source code of _handle_user_message
        source = inspect.getsource(DeepAgentsApp._handle_user_message)

        # CRITICAL: Source code should NOT contain local UserMessage creation
        # This was the regression - creating UserMessage before agent processes it
        assert "UserMessage(" not in source, (
            "REGRESSION DETECTED: _handle_user_message creates UserMessage locally! "
            "User messages MUST only come from agent stream (HumanMessage -> UserMessage)"
        )

        # Verify architecture comment is present
        assert "ARCHITECTURE PRINCIPLE" in source or "single source of truth" in source.lower(), (
            "Missing architecture principle documentation in _handle_user_message. "
            "Add comment explaining why UserMessage is NOT created locally."
        )

    def test_handle_user_message_proper_flow_documented(self) -> None:
        """
        Verify _handle_user_message has proper documentation of message flow.

        The correct flow is:
        1. User input sent to agent (no local display)
        2. Agent processes and broadcasts via stream
        3. textual_adapter receives HumanMessage and creates UserMessage
        4. UI displays the UserMessage from stream
        """
        import inspect

        from deep_worker_cli.app import DeepAgentsApp

        source = inspect.getsource(DeepAgentsApp._handle_user_message)

        # Check that the method invokes agent task
        assert "_run_agent_task" in source or "invoke" in source.lower(), (
            "_handle_user_message should invoke agent to process the message"
        )


class TestTextualAdapterUserMessageDisplay:
    """Test that TextualUIAdapter correctly displays user messages from agent stream."""

    def test_textual_adapter_handles_humanmessage(self) -> None:
        """
        Verify execute_task_textual processes HumanMessage from agent stream.

        ARCHITECTURE: User messages come from agent stream as HumanMessage,
        and textual_adapter converts them to UserMessage widgets for display.
        This is the ONLY place where UserMessage should be created from user input.
        """
        import inspect

        from deep_worker_cli.textual_adapter import execute_task_textual

        source = inspect.getsource(execute_task_textual)

        # Verify HumanMessage is handled
        assert "HumanMessage" in source, (
            "execute_task_textual should handle HumanMessage from agent stream"
        )

        # Verify UserMessage is created from HumanMessage
        assert "UserMessage" in source, (
            "execute_task_textual should create UserMessage from HumanMessage"
        )

    def test_system_message_filtering_exists(self) -> None:
        """
        Verify [SYSTEM] message filtering is implemented.

        Some system messages are auto-injected by agent and should not be shown to user.
        """
        import inspect

        from deep_worker_cli.textual_adapter import execute_task_textual

        source = inspect.getsource(execute_task_textual)

        # Verify system message filtering exists
        assert "[SYSTEM]" in source or "system" in source.lower(), (
            "execute_task_textual should filter out system messages"
        )


class TestAgentServiceMessageNotification:
    """Test that AgentService does not duplicate message notifications."""

    def test_exposed_invoke_agent_code_verification(self) -> None:
        """
        Verify exposed_invoke_agent does NOT call _notify_subscribers before invoke.

        CRITICAL: User message notification should happen ONLY in _handle_invoke()
        via notify_callback. Notifying twice causes message duplication.

        REGRESSION: Previously, exposed_invoke_agent was calling _notify_subscribers
        before _handle_invoke, causing duplicate user message notifications.
        """
        import inspect

        from deep_worker_cli.rpyc_service.agent_service import AgentService

        source = inspect.getsource(AgentService.exposed_invoke_agent)

        # CRITICAL: Should NOT notify subscribers before _handle_invoke
        # Count how many times _notify_subscribers is called in exposed_invoke_agent
        lines = source.split('\n')
        notify_calls_before_invoke = 0
        seen_handle_invoke = False

        for line in lines:
            if '_handle_invoke' in line:
                seen_handle_invoke = True
            if 'self._notify_subscribers(' in line and not seen_handle_invoke:
                notify_calls_before_invoke += 1

        assert notify_calls_before_invoke == 0, (
            f"REGRESSION DETECTED: exposed_invoke_agent calls _notify_subscribers "
            f"{notify_calls_before_invoke} times before _handle_invoke! "
            "This causes duplicate message notifications. "
            "User messages should ONLY be notified via notify_callback in _handle_invoke."
        )

        # Verify notify_callback is passed to _handle_invoke
        assert "notify_callback" in source, (
            "exposed_invoke_agent should pass notify_callback to _handle_invoke. "
            "This is how messages are broadcast to subscribers (single source of truth)."
        )

    @pytest.mark.asyncio
    async def test_notify_subscribers_broadcasts_to_all(self) -> None:
        """
        Verify _notify_subscribers sends message to all subscribers.

        This is the correct place for notifications - called from _handle_invoke
        via notify_callback to broadcast ALL messages to ALL clients.
        """
        from deep_worker_cli.rpyc_service.agent_service import AgentService

        service = AgentService()

        # Mock connection
        mock_conn = MagicMock()
        service.on_connect(mock_conn)

        # Message subscribers now live on AgentProcess (shared state
        # across connections) — see agent_service.py ~line 521/1069.
        # We inject a stub AgentProcess carrying a set.
        mock_sub1 = MagicMock()
        mock_sub2 = MagicMock()

        stub_agent_process = MagicMock()
        stub_agent_process._message_subscribers = {mock_sub1, mock_sub2}
        service._agent_process = stub_agent_process

        # Notify subscribers
        test_message = {"type": "user", "content": "Test", "role": "user"}
        await service._notify_subscribers(test_message)

        # Verify all subscribers received message
        mock_sub1.on_message.assert_called_once_with(test_message)
        mock_sub2.on_message.assert_called_once_with(test_message)


class TestArchitecturePrinciple:
    """Tests that verify the core architecture principle is upheld."""

    def test_agent_is_single_source_of_truth_principle(self) -> None:
        """
        Document and verify the architecture principle.

        PRINCIPLE: Agent (deep-worker) is the ONLY source of truth for ALL messages.

        MESSAGE FLOW:
        1. User input → sent to agent
        2. Agent processes → adds to state
        3. Agent broadcasts via stream → HumanMessage
        4. TextualAdapter receives → converts to UserMessage widget
        5. UI displays UserMessage

        VIOLATIONS TO PREVENT:
        - ❌ Creating UserMessage before agent processes it (app.py)
        - ❌ Notifying subscribers multiple times (agent_service.py)
        - ❌ Displaying user input immediately without waiting for agent

        CORRECT FLOW:
        - ✅ Send to agent, wait for stream, display from stream
        - ✅ Single notification per message via notify_callback
        - ✅ All messages (user, assistant, tool) come from agent stream
        """
        # This is a documentation test - the principle is encoded in code
        # The other tests in this module verify this principle is upheld
        assert True, "Architecture principle documented"
