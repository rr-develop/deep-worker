"""Unit tests for slash command handling in InteractiveApp.

Tests cover both client-side commands (/help, /version, /quit) and
agent-side commands (/clear, /remember, /tokens, /threads).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from textual.widgets import Static

from deep_worker_cli.cli.interactive import InteractiveApp
from deep_worker_cli.widgets.chat_input import ChatInput
from deep_worker_cli.widgets.messages import UserMessage, AppMessage


class TestClientSideCommands:
    """Test client-side slash commands (no agent interaction)."""

    @pytest.mark.asyncio
    async def test_help_command_displays_help_text(self):
        """Test /help command displays help text."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock _mount_message to capture output
            mounted_messages = []
            original_mount = app._mount_message

            async def mock_mount(widget):
                mounted_messages.append(widget)
                await original_mount(widget)

            app._mount_message = mock_mount

            # Execute /help command
            await app._handle_command("/help")

            # Should mount 2 messages: UserMessage + AppMessage with help text
            assert len(mounted_messages) == 2
            assert isinstance(mounted_messages[0], UserMessage)
            assert mounted_messages[0]._content == "/help"

            assert isinstance(mounted_messages[1], AppMessage)
            help_text = mounted_messages[1]._content
            assert "Commands:" in help_text
            assert "/quit" in help_text
            assert "/clear" in help_text
            assert "/help" in help_text

    @pytest.mark.asyncio
    async def test_version_command_displays_version(self):
        """Test /version command displays version info."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            mounted_messages = []
            original_mount = app._mount_message

            async def mock_mount(widget):
                mounted_messages.append(widget)
                await original_mount(widget)

            app._mount_message = mock_mount

            # Execute /version command
            await app._handle_command("/version")

            # Should mount 2 messages
            assert len(mounted_messages) == 2
            assert isinstance(mounted_messages[0], UserMessage)
            assert isinstance(mounted_messages[1], AppMessage)
            assert "version" in mounted_messages[1]._content.lower()

    @pytest.mark.asyncio
    async def test_quit_command_exits_app(self):
        """Test /quit command exits the application."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock exit
            exit_called = False
            original_exit = app.exit

            def mock_exit(*args, **kwargs):
                nonlocal exit_called
                exit_called = True

            app.exit = mock_exit

            # Execute /quit
            await app._handle_command("/quit")

            assert exit_called is True

    @pytest.mark.asyncio
    async def test_quit_alias_q_works(self):
        """Test /q as alias for /quit."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            exit_called = False

            def mock_exit(*args, **kwargs):
                nonlocal exit_called
                exit_called = True

            app.exit = mock_exit

            # Execute /q
            await app._handle_command("/q")

            assert exit_called is True

    @pytest.mark.asyncio
    async def test_unknown_command_shows_error(self):
        """Test unknown command shows error and delegates to agent."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock agent command handler
            agent_commands_called = []

            async def mock_agent_command(cmd):
                agent_commands_called.append(cmd)

            app._handle_agent_command = mock_agent_command

            # Execute unknown command
            await app._handle_command("/unknown")

            # Should delegate to agent command handler
            assert len(agent_commands_called) == 1
            assert agent_commands_called[0] == "/unknown"


class TestAgentSideCommands:
    """Test agent-side slash commands (require RPyC)."""

    @pytest.mark.asyncio
    async def test_clear_command_requires_agent(self):
        """Test /clear command when not connected shows error."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Ensure not connected
            app.rpyc_session = None
            app._connected = False

            mounted_messages = []
            original_mount = app._mount_message

            async def mock_mount(widget):
                mounted_messages.append(widget)
                await original_mount(widget)

            app._mount_message = mock_mount

            # Execute /clear
            await app._handle_agent_command("/clear")

            # Should show error message
            assert len(mounted_messages) >= 1
            error_msg = None
            for msg in mounted_messages:
                if isinstance(msg, AppMessage) and "Not connected" in msg._content:
                    error_msg = msg
                    break

            assert error_msg is not None

    @pytest.mark.asyncio
    async def test_clear_command_calls_agent_via_rpyc(self):
        """Test /clear command sends RPC call to agent."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock RPyC session
            mock_session = MagicMock()
            mock_session.async_call = AsyncMock(return_value={
                "success": True,
                "response": "Started new thread: abc123",
                "action": "clear_thread",
                "data": {
                    "old_thread_id": "old123",
                    "new_thread_id": "abc123"
                }
            })

            app.rpyc_session = mock_session
            app._connected = True

            # Mock _clear_messages
            clear_called = False

            async def mock_clear():
                nonlocal clear_called
                clear_called = True

            app._clear_messages = mock_clear

            # Execute /clear
            await app._handle_agent_command("/clear")

            # Should call RPyC
            mock_session.async_call.assert_called_once()
            call_args = mock_session.async_call.call_args
            assert "handle_command" in str(call_args)

            # Should clear local messages
            assert clear_called is True

    @pytest.mark.asyncio
    async def test_tokens_command_displays_info(self):
        """Test /tokens command displays token info."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock RPyC session
            mock_session = MagicMock()
            mock_session.async_call = AsyncMock(return_value={
                "success": True,
                "response": "Current context: 1.5K tokens",
                "action": "display_info",
                "data": {"token_count": 1500}
            })

            app.rpyc_session = mock_session
            app._connected = True

            mounted_messages = []
            original_mount = app._mount_message

            async def mock_mount(widget):
                mounted_messages.append(widget)
                await original_mount(widget)

            app._mount_message = mock_mount

            # Execute /tokens
            await app._handle_agent_command("/tokens")

            # Should display info
            assert len(mounted_messages) >= 1
            info_msg = None
            for msg in mounted_messages:
                if isinstance(msg, AppMessage) and "tokens" in msg._content.lower():
                    info_msg = msg
                    break

            assert info_msg is not None

    @pytest.mark.asyncio
    async def test_remember_command_sends_to_agent(self):
        """Test /remember command sends prompt to agent."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock RPyC session
            mock_session = MagicMock()
            mock_session.async_call = AsyncMock(return_value={
                "success": True,
                "response": "",
                "action": "send_to_agent",
                "data": {"message": "REMEMBER_PROMPT_TEXT"}
            })

            app.rpyc_session = mock_session
            app._connected = True

            # Mock _handle_user_message
            user_messages_sent = []

            async def mock_user_message(msg):
                user_messages_sent.append(msg)

            app._handle_user_message = mock_user_message

            # Execute /remember
            await app._handle_agent_command("/remember")

            # Should send to agent
            assert len(user_messages_sent) == 1

    @pytest.mark.asyncio
    async def test_remember_with_context_appends_context(self):
        """Test /remember with additional context."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock RPyC session
            mock_session = MagicMock()

            async def mock_call(method, command):
                # Verify command includes context
                assert command == "/remember important stuff"
                return {
                    "success": True,
                    "response": "",
                    "action": "send_to_agent",
                    "data": {"message": "REMEMBER_PROMPT with context"}
                }

            mock_session.async_call = mock_call
            app.rpyc_session = mock_session
            app._connected = True

            user_messages_sent = []

            async def mock_user_message(msg):
                user_messages_sent.append(msg)

            app._handle_user_message = mock_user_message

            # Execute /remember with context
            await app._handle_agent_command("/remember important stuff")

            # Should send to agent
            assert len(user_messages_sent) == 1


class TestCommandModeDetection:
    """Test that ChatInput correctly triggers command mode."""

    @pytest.mark.asyncio
    async def test_slash_triggers_command_mode(self):
        """Test that / prefix triggers command mode."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock command handler
            commands_handled = []

            async def mock_handle_command(cmd):
                commands_handled.append(cmd)

            app._handle_command = mock_handle_command

            # Simulate ChatInput submission with command mode
            from deep_worker_cli.widgets.chat_input import ChatInput

            event = ChatInput.Submitted(
                value="/help",
                mode="command"
            )

            await app.on_chat_input_submitted(event)

            # Should call command handler
            assert len(commands_handled) == 1
            assert commands_handled[0] == "/help"

    @pytest.mark.asyncio
    async def test_normal_text_triggers_message_mode(self):
        """Test that normal text triggers message mode."""
        app = InteractiveApp(agent_name="test-agent")

        async with app.run_test() as pilot:
            # Mock message handler
            messages_handled = []

            async def mock_handle_message(msg):
                messages_handled.append(msg)

            app._handle_user_message = mock_handle_message

            # Simulate normal message
            from deep_worker_cli.widgets.chat_input import ChatInput

            event = ChatInput.Submitted(
                value="Hello agent",
                mode="normal"
            )

            await app.on_chat_input_submitted(event)

            # Should call message handler
            assert len(messages_handled) == 1
            assert messages_handled[0] == "Hello agent"
