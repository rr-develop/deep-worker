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
from textual.message import Message
from textual.widgets import Static

# ============================================================================
# CRITICAL: ARPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
# DO NOT use 'pip install arpyc' or ANY virtual environment (.venv)!
# ALWAYS import arpyc from the arpyc source directory at the repository root.
# Using installed packages causes version conflicts and compatibility issues.
# This project MUST run globally without venv to ensure correct arpyc version.
# ============================================================================
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
    ToolCallMessage,
    UserMessage,
)
from deep_worker_cli.widgets.status import StatusBar
from deep_worker_cli.widgets.welcome import WelcomeBanner

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Resize


class ChatScroll(VerticalScroll):
    """Scrollable chat container that posts ScrolledUp on upward user scroll.

    Subclasses VerticalScroll to hook into the ``watch_scroll_y`` reactive
    watcher.  This watcher fires on **all** scroll input sources:
    - Mouse wheel
    - Keyboard (Page Up, arrow keys)
    - Scrollbar drag

    This is intentionally more complete than the ``ScrollUp`` event from
    ``textual.scrollbar``, which only fires on scrollbar arrow clicks.

    The ``ScrolledUp`` message bubbles up to ``InteractiveApp``, which
    uses it to decide whether to load the next page of conversation history.
    """

    class ScrolledUp(Message):
        """Posted each time the user scrolls upward in the chat area.

        ``InteractiveApp.on_chat_scroll_scrolled_up`` handles this message
        and delegates to ``_maybe_load_more_history``.
        """

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Detect upward scroll and post ``ScrolledUp``.

        Called automatically by Textual's reactive system whenever
        ``scroll_y`` changes.

        CRITICAL: ``super().watch_scroll_y()`` MUST be called first.
        The parent implementation does three essential things:
        1. Updates the visual scrollbar widget position.
        2. Checks the anchor (sticky-scroll-to-bottom mechanism).
        3. Calls ``_refresh_scroll()`` which redraws the scroll content.

        Without the super() call, ``_refresh_scroll()`` is never invoked,
        so the display does NOT update when ``scroll_y`` changes.  This
        causes two visible bugs:
        - The input area border disappears (scroll position looks "stuck"
          at top because the display was never redrawn after scroll_end()).
        - ``widget.region.y`` values stay stale (layout not recalculated),
          making ``_maybe_load_more_history`` think the topmost visible
          widget is at index 0 and spuriously loading history → query
          duplication.

        Args:
            old_value: Previous scroll_y position (pixels from top).
            new_value: New scroll_y position (pixels from top).
        """
        # Always run Textual's default scroll-watcher first so that:
        # - the scrollbar thumb moves to the correct position,
        # - the anchor mechanism fires (_check_anchor),
        # - _refresh_scroll() redraws the content at the new scroll offset.
        super().watch_scroll_y(old_value, new_value)

        if new_value < old_value:
            # Scrolled toward the top → post message that bubbles to App
            self.post_message(self.ScrolledUp())


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

        # Tool message tracking for updating status on result
        self._current_tool_messages: dict[str, ToolCallMessage] = {}

    def compose(self) -> ComposeResult:
        """Compose the application layout.

        Yields:
            UI components for the main chat area and status bar.
        """
        # Generate a session ID for this connection
        session_id = uuid.uuid4().hex[:8]

        # Main chat area with scrollable messages.
        # ChatScroll (not plain VerticalScroll) is used so that upward scrolls
        # are detected via watch_scroll_y and trigger history pagination.
        with ChatScroll(id="chat"):
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
        # We use call_after_refresh so the scroll runs after the first layout
        # pass, ensuring max_scroll_y reflects the final widget heights.
        def _scroll_to_bottom_on_mount() -> None:
            try:
                chat_scroll = self.query_one("#chat", VerticalScroll)
                chat_scroll.scroll_end(animate=False, immediate=True)
            except Exception:
                pass

        self.call_after_refresh(_scroll_to_bottom_on_mount)

        # Pagination is now event-driven via ChatScroll.watch_scroll_y.
        # When the user scrolls up, ChatScroll posts ScrolledUp, which
        # this app handles in on_chat_scroll_scrolled_up.  No polling needed.

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

        # Create callback object for message events
        def handle_message(msg: dict[str, Any]) -> None:
            """Handle incoming message from agent."""
            # CRITICAL: Use Textual's call_from_thread() for thread-safe scheduling
            # This properly schedules work IN THE CONTEXT of Textual app.
            # DO NOT use loop.call_soon_threadsafe() - it runs outside Textual context!
            self.call_from_thread(self._handle_realtime_message, msg)

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

        This is the ONLY place where messages are displayed in dwcli!

        ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
        ============================================================
        The agent deep-worker is the ONLY source of truth for ALL messages.
        ALL messages (user + assistant + tool calls + tool results) MUST be
        displayed via this callback.

        Message types:
        - type: "user" - User message
        - type: "assistant" - Complete assistant message
        - type: "assistant_chunk" - Streaming text chunk from assistant
        - type: "tool_call" - Tool invocation
        - type: "tool_result" - Tool execution result

        User messages can originate from:
        - This dwcli instance
        - Other clients (other dwcli instances, task manager, etc.)

        Regardless of origin, ALL messages go through agent and come back
        via RPyC callbacks. This ensures dwcli shows complete conversation
        state from agent, maintaining single source of truth.

        Args:
            msg: Message dict with keys: type, role, content (and others depending on type)
        """
        msg_type = msg.get("type")
        role = msg.get("role")
        content = msg.get("content", "")

        if msg_type == "user" or (msg_type is None and role == "user"):
            # User message from agent - display it
            # (Could have originated from this dwcli, another client, or task manager)
            import uuid
            msg_id = f"user-{uuid.uuid4().hex[:8]}"
            await self._mount_message(UserMessage(content, id=msg_id))
            # Update total count
            self._history_total += 1

        elif msg_type == "assistant" or (msg_type is None and role == "assistant"):
            # Complete assistant message from agent - display it
            import uuid
            msg_id = f"asst-{uuid.uuid4().hex[:8]}"
            widget = AssistantMessage(content, id=msg_id)
            await self._mount_message(widget)
            # Render markdown content
            try:
                await widget.write_initial_content()
            except Exception as e:
                # Log error but don't crash
                import sys
                print(f"[DWCLI] Error rendering markdown: {e}", file=sys.stderr)
            # Update total count
            self._history_total += 1

        elif msg_type == "assistant_chunk":
            # Streaming text chunk from assistant
            # Find or create the current assistant message widget
            messages_container = self.query_one("#messages")

            # Get last message widget
            last_widget = None
            if messages_container.children:
                last_widget = messages_container.children[-1]

            # If last widget is AssistantMessage, append to it
            if last_widget and isinstance(last_widget, AssistantMessage):
                await last_widget.append_content(content)
            else:
                # No existing assistant message, create new one
                import uuid
                msg_id = f"asst-{uuid.uuid4().hex[:8]}"
                widget = AssistantMessage("", id=msg_id)
                await self._mount_message(widget)
                await widget.append_content(content)
                # Update total count only when creating new message
                self._history_total += 1

        elif msg_type == "tool_call":
            # Tool call - create ToolCallMessage widget
            tool_name = msg.get("name", "unknown")
            tool_args = msg.get("args", {})
            tool_id = msg.get("id", "unknown")

            import uuid
            msg_id = f"tool-call-{uuid.uuid4().hex[:8]}"
            widget = ToolCallMessage(tool_name=tool_name, args=tool_args, id=msg_id)
            await self._mount_message(widget)

            # Mark as running and track for later status update
            widget.set_running()
            self._current_tool_messages[tool_id] = widget

            # Update total count
            self._history_total += 1

        elif msg_type == "tool_result":
            # Tool result - update corresponding ToolCallMessage status
            tool_call_id = msg.get("tool_call_id", "")
            tool_status = msg.get("status", "unknown")
            tool_content = msg.get("content", "")

            # Find corresponding tool call widget and update it
            if tool_call_id in self._current_tool_messages:
                tool_widget = self._current_tool_messages[tool_call_id]
                if tool_status == "success":
                    tool_widget.set_success(tool_content)
                else:
                    tool_widget.set_error(tool_content or "Error")
                # Clean up - remove from tracking after update
                del self._current_tool_messages[tool_call_id]
            else:
                # Orphaned tool result (no matching call) - log and ignore
                import sys
                print(
                    f"[DWCLI] Received tool_result for unknown tool_call_id: {tool_call_id}",
                    file=sys.stderr,
                )

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

        # ============================================================================
        # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
        # ============================================================================
        # DO NOT display user message locally here!
        # The agent deep-worker is the ONLY source of truth for ALL messages,
        # including user messages.
        #
        # User messages can originate from:
        # - This dwcli instance
        # - Other dwcli instances connected to same agent
        # - Task manager sending commands to agent
        # - Any other client
        #
        # ALL messages MUST come through agent's RPyC callbacks to dwcli.
        # This ensures dwcli displays the complete conversation state from agent,
        # regardless of message origin.
        #
        # The message will be displayed when agent echoes it back via callback
        # in _handle_realtime_message() after exposed_invoke_agent() notifies
        # subscribers.
        # ============================================================================

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
        messages_container = self.query_one("#messages", Container)
        await messages_container.mount(widget)

        self._scroll_chat_to_bottom()

        # Make sure input is always visible (like deepagents does)
        try:
            input_container = self.query_one("#bottom-app-container", Container)
            input_container.scroll_visible()
        except Exception:
            pass

    def _scroll_chat_to_bottom(self) -> None:
        """Scroll chat to bottom.

        Uses immediate=True so the scroll is applied in the current event-loop
        tick rather than being queued behind pending messages.  Without it a
        burst of incoming messages can leave scroll_y lagging behind max_scroll_y.
        """
        chat = self.query_one("#chat", VerticalScroll)
        chat.scroll_end(animate=False, immediate=True)

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

            # DEBUG: Log history count
            import os
            debug_enabled = os.environ.get("DWCLI_DEBUG_CALLBACKS") == "1"
            if debug_enabled:
                with open("/tmp/dwcli_callback_debug.log", "a") as f:
                    f.write(f"  history count: {len(history)}\n")
                    for i, msg_data in enumerate(history):
                        msg_type = msg_data.get("type")
                        content = msg_data.get("content", "")[:50]
                        f.write(f"    [{i}] {msg_type}: {content}...\n")

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
            #
            # TIMING: write_initial_content() queues markdown updates, but the
            # resulting height changes are applied in a subsequent layout pass
            # (async).  Calling scroll_end() synchronously here would use a
            # stale max_scroll_y computed before markdown compression, landing
            # at the wrong position.
            #
            # Fix (same pattern as deepagents): schedule scroll via set_timer so
            # it runs AFTER the layout pass, when max_scroll_y is accurate.
            def _scroll_to_bottom_after_history() -> None:
                try:
                    chat = self.query_one("#chat", VerticalScroll)
                    chat.scroll_end(animate=False, immediate=True)
                except Exception:
                    pass

            self.set_timer(0.1, _scroll_to_bottom_after_history)

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

    # ------------------------------------------------------------------
    # Scroll-based pagination
    # ------------------------------------------------------------------

    def on_chat_scroll_scrolled_up(self, _event: ChatScroll.ScrolledUp) -> None:
        """Handle upward scroll in the chat area.

        Called by Textual's message routing whenever ``ChatScroll`` posts
        ``ScrolledUp`` (i.e. the user scrolled toward the top via mouse wheel,
        keyboard, or scrollbar).  Delegates to ``_maybe_load_more_history``
        which decides whether to fetch an older history page from the agent.
        """
        self._maybe_load_more_history()

    def _maybe_load_more_history(self) -> None:
        """Conditionally trigger loading of an older history page.

        This is the core pagination decision function.  It applies three
        guards in order (cheapest first) and only starts a load when all
        pass:

        1. **History not fully loaded** – checked against the cached
           ``_history_offset`` / ``_history_total`` counters.  No RPyC call
           is needed; the counters were set during ``_load_history()`` and
           are kept up to date by ``_load_more_history()``.

        2. **No concurrent load** – ``_loading_history`` flag prevents
           overlapping RPyC requests.

        3. **Topmost visible message is near the top of the DOM window** –
           finds the first message widget whose ``region.y >= 0`` (i.e. the
           topmost widget currently visible on screen).  If its index in
           ``#messages`` children is < 10 we are close enough to the start
           of the loaded window that the user would soon run out of content;
           load more now.

        The threshold of 10 provides comfortable read-ahead: the user will
        see the older messages appear before reaching the very top of what
        is currently rendered in the DOM.

        Note on ``widget.region.y``:
            In a scrollable Textual container, ``region`` gives the widget's
            position relative to the screen after layout.  Widgets scrolled
            above the viewport have ``region.y < 0``; widgets at or below
            the top of the viewport have ``region.y >= 0``.
        """
        # Guard 1: all history already loaded (cached, no RPyC call needed)
        if self._history_offset >= self._history_total:
            return

        # Guard 2: a load is already in progress
        if self._loading_history:
            return

        # Guard 3: topmost visible message index
        try:
            messages_container = self.query_one("#messages", Container)
            children = list(messages_container.children)
        except NoMatches:
            return

        if not children:
            return

        # Find the index of the topmost visible widget.
        # Widgets above the viewport: region.y < 0  (scrolled out of view)
        # Widgets in/below viewport: region.y >= 0
        topmost_visible_index: int | None = None
        for i, child in enumerate(children):
            if child.region.y >= 0:
                topmost_visible_index = i
                break

        if topmost_visible_index is None:
            # No visible messages (layout not yet computed, or all off-screen)
            return

        # Load if fewer than 10 earlier messages are available in the DOM
        # above the topmost visible one.
        if topmost_visible_index < 10:
            self.run_worker(self._load_more_history())

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

        # NOTE: Do NOT clear input here! Input text should stay visible while agent processes.
        # Input will be cleared after agent responds (in _handle_realtime_message for assistant message)

        await self._handle_user_message(message.value)
