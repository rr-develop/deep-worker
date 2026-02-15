"""Interactive Textual UI for connecting to running deep-worker agents."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import App
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static

# Add arpyc to path - workaround for import from source
_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))

from arpyc import RpycAsyncSession
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.widgets.chat_input import ChatInput
from deep_worker_cli.widgets.loading import LoadingWidget
from deep_worker_cli.widgets.message_store import MessageStore
from deep_worker_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    UserMessage,
)
from deep_worker_cli.widgets.status import StatusBar
from deep_worker_cli.widgets.welcome import WelcomeBanner

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Resize


class InteractiveApp(App):
    """Textual TUI app for interacting with a running deep-worker agent.

    This app connects to an already-running agent via RPyC and
    provides an interactive chat interface similar to deepagents-cli.
    """

    TITLE = "dwcli"
    CSS_PATH = "interactive.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "quit_app", "Quit", show=False),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        agent_name: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the interactive app.

        Args:
            agent_name: Name of the running agent to connect to
            **kwargs: Additional arguments passed to parent

        Raises:
            ValueError: If agent_name is empty
        """
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name must be a non-empty string")

        super().__init__(**kwargs)
        self.agent_name = agent_name

        # RPyC session for real-time messaging
        self.rpyc_session: RpycAsyncSession | None = None
        self.rpyc_port: int | None = None
        self.rpyc_callback: Any = None

        self._status_bar: StatusBar | None = None
        self._chat_input: ChatInput | None = None
        self._message_store = MessageStore()
        self._loading_widget: LoadingWidget | None = None
        self._connected = False

        # Pagination state
        self._history_offset = 0  # Current pagination offset
        self._history_total = 0   # Total messages available
        self._loading_history = False  # Prevent concurrent loads

    def compose(self) -> ComposeResult:
        """Compose the application layout.

        Yields:
            UI components for the main chat area and status bar.
        """
        # Generate a session ID for this connection
        session_id = uuid.uuid4().hex[:8]

        # Main chat area with scrollable messages
        with VerticalScroll(id="chat"):
            yield WelcomeBanner(thread_id=session_id, id="welcome-banner")
            yield Container(id="messages")
            with Container(id="bottom-app-container"):
                yield ChatInput(cwd=str(Path.cwd()), id="input-area")
            yield Static(id="chat-spacer")

        # Status bar at bottom
        yield StatusBar(cwd=str(Path.cwd()), id="status-bar")

    async def on_mount(self) -> None:
        """Initialize components and connect to agent."""
        self._status_bar = self.query_one("#status-bar", StatusBar)
        self._chat_input = self.query_one("#input-area", ChatInput)

        # Focus the input
        self._chat_input.focus_input()

        # Connect to agent
        await self._connect_to_agent()

        # Show connection status
        if self._connected:
            await self._mount_message(
                AppMessage(f"Connected to agent '{self.agent_name}'")
            )
            # Load conversation history after connecting
            await self._load_history()
        else:
            await self._mount_message(
                AppMessage(f"⚠ Failed to connect to agent '{self.agent_name}'")
            )

        # CRITICAL: After initial render, ALWAYS ensure input area is visible!
        # This handles cases where:
        # 1. No history exists (checkpoint DB missing)
        # 2. History loading failed
        # 3. Textual's default scroll position hides the input
        # We scroll to BOTTOM so the input area at bottom of content is visible.
        chat_scroll = self.query_one("#chat", VerticalScroll)
        chat_scroll.scroll_end(animate=False)

        # Setup periodic scroll monitoring for pagination
        self.set_interval(0.5, self._check_scroll_position)

    async def _agent_task(self, message: str) -> None:
        """Run agent request in background.

        CRITICAL REQUIREMENTS:
        1. User message is ALREADY displayed (by _handle_user_message)
        2. Only load NEW messages incrementally (NEVER reload entire history)
        3. Messages must appear as they are created (real-time)

        Args:
            message: User message to send
        """
        # Log to file for debugging (env var might not reach subprocess)
        log_file = "/tmp/dwcli_debug.log"
        def log_msg(msg_text):
            try:
                with open(log_file, "a") as f:
                    f.write(f"[DWCLI] {msg_text}\n")
            except:
                pass

        try:
            log_msg("_agent_task started")

            # Send message via IPC and get response
            response = await self.send_user_message(message)
            log_msg(f"got response: success={response.get('success')}")

            if response.get("success"):
                # CRITICAL: With RPyC, messages arrive in real-time via callbacks!
                # NO NEED to poll for new messages - they are pushed to us!

                # TaskManager intermediate messages (corrective prompts and responses)
                # are automatically received via _handle_realtime_message callback

                # RPyC mode: messages already received via callbacks
                log_msg("Messages received via RPyC callbacks (no polling needed)")

                self._update_status("Ready")
                log_msg(f"response message ready")
            else:
                # Handle error response
                error_msg = response.get("error", "Unknown error")
                await self._mount_message(AppMessage(f"⚠ Error: {error_msg}"))
                self._update_status(f"Error: {error_msg}")
                log_msg(f"error response: {error_msg}")

        except Exception as e:
            import traceback
            log_msg(f"exception in agent task: {e}")
            log_msg(traceback.format_exc())

            await self._mount_message(AppMessage(f"⚠ Error: {e}"))
            self._update_status(f"Error: {e}")

    async def _connect_to_agent(self) -> None:
        """Connect to the running agent via RPyC."""
        try:
            await self._connect_via_rpyc()
        except Exception as e:
            self._connected = False
            self._update_status(f"Connection failed: {e}")

    async def _connect_via_rpyc(self) -> None:
        """Connect to agent via RPyC and subscribe to real-time messages."""
        # Find RPyC port from registry
        registry = ProcessRegistry()
        agent_info = registry.lookup(self.agent_name)

        if not agent_info:
            raise RuntimeError(f"Agent '{self.agent_name}' not found in registry")

        # Get RPyC port from agent info
        self.rpyc_port = agent_info.rpyc_port

        if not self.rpyc_port:
            raise RuntimeError(f"Agent '{self.agent_name}' does not have RPyC enabled")

        # Create RPyC session
        # SECURITY: Connect only to localhost (127.0.0.1), never to external hosts
        self.rpyc_session = RpycAsyncSession("127.0.0.1", self.rpyc_port)
        await self.rpyc_session.connect()

        # Subscribe to real-time messages
        await self._subscribe_to_messages()

        self._connected = True
        self._update_status(f"Connected to {self.agent_name} (RPyC)")

    async def _subscribe_to_messages(self) -> None:
        """Subscribe to real-time message events via RPyC callbacks."""
        if not self.rpyc_session:
            return

        loop = asyncio.get_running_loop()

        # Create callback object for message events
        def handle_message(msg: dict[str, Any]) -> None:
            """Handle incoming message from agent."""
            # Schedule message handling in main event loop
            loop.call_soon_threadsafe(
                asyncio.create_task,
                self._handle_realtime_message(msg)
            )

        # Create callback class
        self.rpyc_callback = type(
            "MessageCallback",
            (),
            {"on_message": lambda self, msg: handle_message(msg)},
        )()

        # Subscribe via RPyC
        if self.rpyc_session:
            success = await self.rpyc_session.async_call(
                self.rpyc_session.root.subscribe_messages,
                self.rpyc_callback
            )
            if not success:
                raise RuntimeError("Failed to subscribe to messages")

    async def _handle_realtime_message(self, msg: dict[str, Any]) -> None:
        """Handle real-time message received via RPyC callback.

        Args:
            msg: Message dict with keys: type, role, content
        """
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            # User message - display it
            await self._mount_message(UserMessage(content))
        elif role == "assistant":
            # Assistant message - display it
            await self._mount_message(AssistantMessage(content))

        # Update total count
        self._history_total += 1

    async def send_user_message(self, message: str) -> dict[str, Any]:
        """Send a user message to the agent and get response.

        Args:
            message: User message text

        Returns:
            Response from agent with format:
                {
                    "success": bool,
                    "response": {
                        "role": "assistant",
                        "content": str
                    }
                }
        """
        # Send via RPyC
        if not self.rpyc_session:
            raise RuntimeError("Not connected to agent (RPyC)")

        response = await self.rpyc_session.async_call(
            self.rpyc_session.root.invoke_agent,
            message,
            False  # stream=False
        )
        return response

    async def disconnect_from_agent(self) -> None:
        """Disconnect from the agent gracefully."""
        # Disconnect from RPyC
        if self.rpyc_session and self.rpyc_callback:
            # Unsubscribe from messages
            try:
                await self.rpyc_session.async_call(
                    self.rpyc_session.root.unsubscribe_messages,
                    self.rpyc_callback
                )
            except Exception:
                pass

        if self.rpyc_session:
            await self.rpyc_session.disconnect()
            self.rpyc_session = None
            self.rpyc_callback = None

        self._connected = False

    async def _handle_user_message(self, message: str) -> None:
        """Handle user message submission.

        CRITICAL REQUIREMENTS:
        1. Display user message IMMEDIATELY (before agent responds)
        2. User must see their message right away for interactivity
        3. Update _history_total to track that we added a message

        Args:
            message: User message text
        """
        import os
        debug = os.environ.get("DWCLI_DEBUG") == "1"

        if debug:
            import sys
            print(f"[DWCLI_DEBUG] _handle_user_message called with: '{message}'", file=sys.stderr)

        if not message.strip():
            if debug:
                import sys
                print(f"[DWCLI_DEBUG] message is empty/whitespace, returning", file=sys.stderr)
            return

        if not self._connected:
            if debug:
                import sys
                print(f"[DWCLI_DEBUG] not connected, showing error", file=sys.stderr)

            await self._mount_message(
                AppMessage("⚠ Not connected to agent. Cannot send message.")
            )
            return

        # CRITICAL: Display user message IMMEDIATELY!
        # User must see their message right away, not after agent completes.
        await self._mount_message(UserMessage(message))

        # Increment history total - we just added user message
        self._history_total += 1

        if debug:
            import sys
            print(f"[DWCLI_DEBUG] sending message to agent: '{message}'", file=sys.stderr)

        self._update_status("Sending message...")

        # Run agent request in background worker (like deepagents does)
        # This keeps Textual's event loop responsive for rendering
        self.run_worker(self._agent_task(message))

    async def _mount_message(self, widget: Any) -> None:
        """Mount a message widget to the messages container.

        Args:
            widget: Message widget to mount
        """
        log_file = "/tmp/dwcli_debug.log"
        def log_msg(msg_text):
            try:
                with open(log_file, "a") as f:
                    f.write(f"[DWCLI] {msg_text}\n")
            except:
                pass

        log_msg(f"_mount_message: mounting {type(widget).__name__}")
        messages_container = self.query_one("#messages", Container)
        log_msg(f"_mount_message: got messages container")

        await messages_container.mount(widget)
        log_msg(f"_mount_message: widget mounted to DOM")

        self._scroll_chat_to_bottom()
        log_msg(f"_mount_message: scrolled to bottom")

        # Make sure input is always visible (like deepagents does)
        try:
            input_container = self.query_one("#bottom-app-container", Container)
            input_container.scroll_visible()
            log_msg(f"_mount_message: scrolled input into view")
        except Exception as e:
            log_msg(f"_mount_message: error scrolling input: {e}")

    def _scroll_chat_to_bottom(self) -> None:
        """Scroll chat to bottom."""
        chat = self.query_one("#chat", VerticalScroll)
        chat.scroll_end(animate=False)

    async def _load_history(self) -> None:
        """Load and display conversation history from the agent.

        This method retrieves recent messages from the agent's checkpoint state
        and displays them in the chat window WITHOUT auto-scrolling to bottom.
        The input area must remain visible after loading history.
        """
        if not self.rpyc_session:
            return

        try:
            # Request recent history via RPyC
            response = await self.rpyc_session.async_call(
                self.rpyc_session.root.get_history,
                limit=50,
                offset=0
            )

            if not response.get("success"):
                # Don't show error to user - just log it
                import sys
                print(f"[HISTORY] Failed to load history: {response.get('error')}", file=sys.stderr)
                return

            history = response.get("history", [])

            # Update pagination state
            self._history_total = response.get("total", 0)
            self._history_offset = len(history)  # Messages loaded so far

            if not history:
                # No history to display
                return

            # Track tool calls for matching with ToolMessages
            pending_tool_calls: dict[str, dict] = {}

            # Get messages container
            messages_container = self.query_one("#messages", Container)

            # Display history messages in chronological order
            # BUT don't auto-scroll - let input stay visible
            for msg_data in history:
                msg_type = msg_data.get("type")

                if msg_type == "HumanMessage":
                    content = msg_data.get("content", "")
                    # Skip system messages
                    if content.startswith("[SYSTEM]"):
                        continue
                    widget = UserMessage(content)
                    # Mount without auto-scroll
                    await messages_container.mount(widget)

                elif msg_type == "AIMessage":
                    content = msg_data.get("content", "").strip()
                    if content:
                        widget = AssistantMessage(content)
                        await messages_container.mount(widget)
                        # Render markdown content
                        await widget.write_initial_content()

                    # Track tool calls
                    tool_calls = msg_data.get("tool_calls", [])
                    for tc in tool_calls:
                        tc_id = tc.get("id")
                        if tc_id:
                            from deep_worker_cli.widgets.messages import ToolCallMessage
                            tool_widget = ToolCallMessage(
                                tc.get("name", "unknown"),
                                tc.get("args", {}),
                            )
                            await messages_container.mount(tool_widget)
                            pending_tool_calls[tc_id] = {
                                "widget": tool_widget,
                                "name": tc.get("name"),
                            }

                elif msg_type == "ToolMessage":
                    tc_id = msg_data.get("tool_call_id")
                    if tc_id and tc_id in pending_tool_calls:
                        tool_info = pending_tool_calls.pop(tc_id)
                        widget = tool_info["widget"]
                        status = msg_data.get("status", "success")
                        content = msg_data.get("content", "")
                        if status == "success":
                            widget.set_success(content)
                        else:
                            widget.set_error(content)

            # Mark any unmatched tool calls as interrupted
            for tool_info in pending_tool_calls.values():
                widget = tool_info["widget"]
                widget.set_rejected()

            # CRITICAL: Scroll to BOTTOM after loading history!
            # When we load many messages, #messages container grows and pushes
            # #bottom-app-container (input area) down. We MUST scroll to BOTTOM
            # so the user sees the most recent messages AND the input area.
            # The input area must always be visible for the user to type.
            chat_scroll = self.query_one("#chat", VerticalScroll)
            chat_scroll.scroll_end(animate=False)

        except Exception as e:
            import sys
            import traceback
            print(f"[HISTORY] Exception loading history: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    async def _load_more_history(self) -> None:
        """Load additional older messages when scrolling up.

        IMPORTANT: This preserves the user's current scroll position by
        adjusting scroll_y after inserting messages at the top.
        """
        # Prevent concurrent loads
        if self._loading_history:
            return

        # Check if there's more history to load
        if self._history_offset >= self._history_total:
            return  # Already loaded everything

        if not self.rpyc_session:
            return

        self._loading_history = True

        try:
            # Log pagination event for testing
            with open("/tmp/dwcli_pagination.log", "a") as f:
                f.write(f"PAGINATION: offset={self._history_offset}, total={self._history_total}\n")

            # Save current scroll position BEFORE loading
            chat_scroll = self.query_one("#chat", VerticalScroll)
            old_scroll_y = chat_scroll.scroll_y
            old_max_y = chat_scroll.max_scroll_y

            # Load next page of older messages via RPyC
            response = await self.rpyc_session.async_call(
                self.rpyc_session.root.get_history,
                limit=20,
                offset=self._history_offset
            )

            if not response.get("success"):
                return

            history = response.get("history", [])
            if not history:
                return

            # Update offset
            self._history_offset += len(history)

            # Get messages container
            messages_container = self.query_one("#messages", Container)

            # Insert older messages at the TOP of the container
            # We need to reverse mount order so oldest appears first
            for msg_data in reversed(history):
                msg_type = msg_data.get("type")

                if msg_type == "HumanMessage":
                    content = msg_data.get("content", "")
                    if content.startswith("[SYSTEM]"):
                        continue
                    widget = UserMessage(content)
                    # Mount at index 0 to insert at top
                    await messages_container.mount(widget, before=0)

                elif msg_type == "AIMessage":
                    content = msg_data.get("content", "").strip()
                    if content:
                        widget = AssistantMessage(content)
                        await messages_container.mount(widget, before=0)
                        await widget.write_initial_content()

            # CRITICAL: Restore scroll position after inserting messages
            # New content was added at top, so we need to adjust scroll_y
            # to maintain the user's view of the same messages
            new_max_y = chat_scroll.max_scroll_y
            height_added = new_max_y - old_max_y

            # Adjust scroll position to compensate for added content
            if height_added > 0:
                chat_scroll.scroll_y = old_scroll_y + height_added

        finally:
            self._loading_history = False

    def _check_scroll_position(self) -> None:
        """Periodically check scroll position and trigger pagination if near top.

        This method is called every 0.5 seconds via set_interval.
        It checks if user has scrolled near the top of chat and triggers
        pagination to load older messages.
        """
        try:
            chat_scroll = self.query_one("#chat", VerticalScroll)

            # Check if scrolled near the top (within 100 pixels)
            if chat_scroll.scroll_y <= 100:
                # Only trigger if not already loading
                if not self._loading_history and self._history_offset < self._history_total:
                    # Log for debugging
                    with open("/tmp/dwcli_scroll_debug.log", "a") as f:
                        f.write(f"SCROLL_CHECK: y={chat_scroll.scroll_y}, offset={self._history_offset}/{self._history_total}, triggering pagination\n")

                    # Trigger background history load
                    self.run_worker(self._load_more_history())
        except NoMatches:
            # Chat container not yet mounted
            pass
        except Exception as e:
            # Log errors but don't crash
            try:
                with open("/tmp/dwcli_scroll_debug.log", "a") as f:
                    f.write(f"ERROR in scroll check: {e}\n")
            except:
                pass

    def _update_status(self, message: str) -> None:
        """Update the status bar with a message.

        Args:
            message: Status message
        """
        if self._status_bar:
            self._status_bar.set_status_message(message)

    def on_resize(self, _event: Resize) -> None:
        """Handle terminal resize."""
        try:
            self.query_one("#chat-spacer", Static)
            self.call_after_refresh(lambda: None)
        except NoMatches:
            pass

    async def action_quit_app(self) -> None:
        """Handle quit action (Ctrl+C or Ctrl+D)."""
        await self.disconnect_from_agent()
        self.exit(0)

    async def on_chat_input_submitted(self, message: ChatInput.Submitted) -> None:
        """Handle chat input submission.

        Args:
            message: ChatInput.Submitted message
        """
        import os
        debug = os.environ.get("DWCLI_DEBUG") == "1"

        if debug:
            import sys
            print(f"[DWCLI_DEBUG] on_chat_input_submitted called with message: '{message.value}'", file=sys.stderr)

        await self._handle_user_message(message.value)
