"""Interactive Textual UI for connecting to running deep-worker agents."""

from __future__ import annotations

import asyncio
import json
import subprocess  # S404: subprocess required for user-initiated shell commands via ! prefix
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from textual.app import App
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Static

# ============================================================================
# rpyc-async is a regular PyPI dependency (>=1.0.0) providing the
# AsyncioServer-based protocol (rpyc_async.core.async_connect). The previous
# sys.path injection of a local rpyc_async checkout was removed: on a
# developer machine it shadowed the PyPI package with sources, making
# "clean env" wheel installs test the wrong code.
# ============================================================================

import rpyc_async as rpyc
from rpyc_async.core.async_connect import async_connect
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.widgets.chat_input import ChatInput
from deep_worker_cli.widgets.loading import LoadingWidget
from deep_worker_cli.widgets.message_store import MessageStore
from deep_worker_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    ErrorMessage,
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
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+c", "quit_or_interrupt", "Quit/Interrupt", show=False),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
        Binding("shift+tab", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
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

        # RPyC connection for real-time messaging
        self._rpyc_conn: rpyc.Connection | None = None
        self._rpyc_root: Any = None  # Obtained from async_connect (eager handshake)
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

        # Approval menu tracking (for interrupt handling)
        self._pending_approval_widget: Any = None  # ApprovalMenu widget if active
        self._current_approval_id: str | None = None  # Current approval ID

        # Token tracking
        self._current_token_count: int = 0  # Current token count from agent

        # Auto-approve mode
        self._auto_approve: bool = False  # Auto-approve dangerous operations

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

            # Show "Thinking..." spinner immediately (matches deepagents behavior)
            await self._set_spinner("Thinking")

            # Send message via IPC and get response
            response = await self.send_user_message(message)
            log_msg(f"got response: success={response.get('success')}")

            # Note: send_user_message returns immediately (fire-and-forget)
            # Messages arrive via RPyC callbacks in _handle_realtime_message
            # Spinner will be hidden when first message arrives

            if not response.get("success"):
                # Handle error response (e.g., connection failed)
                error_msg = response.get("error", "Unknown error")
                await self._set_spinner(None)  # Hide spinner on error
                await self._mount_message(AppMessage(f"⚠ Error: {error_msg}"))
                self._update_status(f"Error: {error_msg}")
                log_msg(f"error response: {error_msg}")

        except Exception as e:
            import traceback
            log_msg(f"exception in agent task: {e}")
            log_msg(traceback.format_exc())

            await self._set_spinner(None)  # Hide spinner on exception
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
        # ============================================================================
        # CRITICAL: NEVER ADD TIMEOUTS TO RPyC CONNECTION!
        # ============================================================================
        # deep-worker agents can work with dwcli for DAYS or WEEKS continuously.
        # RPyC connection lifetime timeouts are FORBIDDEN.
        # Agent may be busy with:
        # - Long-running TaskManager evaluation loops (minutes to hours)
        # - Complex multi-step operations
        # - Large file processing
        # - Any other legitimate long-running work
        #
        # FORBIDDEN:
        # - "sync_request_timeout": <any value>  # NEVER ADD THIS
        # - Any other timeout on connection lifetime
        #
        # If connection times out, it's a BUG, not a feature!
        # ============================================================================
        config = {
            "allow_public_attrs": True,
            # NO TIMEOUT - connection can live for days/weeks
        }
        # ============================================================================
        # Use async_connect, NOT sync rpyc.connect.
        # ============================================================================
        # We're inside a Textual asyncio event loop. Synchronous rpyc.connect()
        # blocks the loop on TCP handshake and on every later sync_request; since
        # rpyc_async's policy update it also raises RuntimeError when called from
        # a running loop.
        #
        # async_connect() is natively non-blocking, auto-enables asyncio serving
        # internally, and eager-fetches conn.root — so we don't need a separate
        # enable_asyncio_serving call afterwards.
        # ============================================================================
        self._rpyc_conn = await async_connect(
            "127.0.0.1",
            self.rpyc_port,
            config=config,
        )
        self._rpyc_root = self._rpyc_conn.root

        # Subscribe to real-time messages
        await self._subscribe_to_messages()

        self._connected = True
        self._update_status(f"Connected to {self.agent_name} (RPyC)")

    async def _subscribe_to_messages(self) -> None:
        """Subscribe to real-time message events via RPyC callbacks."""
        if not self._rpyc_root:
            return

        # ========================================================================
        # CRITICAL: Callback MUST be RPyC Service with exposed_ async method!
        # ========================================================================
        # With AsyncioServer, callbacks must be proper RPyC Services.
        # The on_message method MUST be async to match deepagents architecture
        # where all callbacks are async (await adapter._mount_message, etc.)
        #
        # This ensures the CLI follows the SAME async patterns as
        # the original deepagents implementation.
        #
        # ARCHITECTURE: Single-thread, single event loop
        # ================================================
        # RPyC AsyncioServer callbacks run in agent's event loop, but they can
        # safely call Textual's call_from_thread() which is designed for
        # cross-thread communication. Textual handles the thread-safety internally.
        # No worker thread or queue needed - direct call_from_thread is sufficient.
        # ========================================================================
        class MessageCallbackService(rpyc.Service):
            """RPyC service for receiving messages from agent."""

            async def exposed_on_message(self_inner, msg: dict[str, Any]) -> None:
                """Receive message from agent (async exposed for RPyC!).

                CRITICAL: This callback is invoked by RPyC AsyncioServer in the agent's
                event loop. Since dwcli uses async_connect() (which auto-enables
                asyncio serving) against Textual's event loop, this callback runs in
                dwcli's asyncio loop.

                We MUST NOT block here! call_from_thread() is a BLOCKING call that waits
                for Textual to process the callback. Instead, we schedule the handler
                as an async task that will run in the same event loop without blocking
                the RPyC callback return.

                This ensures RPyC AsyncioServer doesn't block waiting for Textual processing.
                """
                # Schedule async handler as task - non-blocking!
                asyncio.create_task(self._handle_realtime_message(msg))

        # Create callback service instance
        self.rpyc_callback = MessageCallbackService()

        # Subscribe via RPyC
        if self._rpyc_root:
            success = await self._rpyc_root.subscribe_messages(self.rpyc_callback)
            if not success:
                raise RuntimeError("Failed to subscribe to messages")

    async def _handle_realtime_message(self, msg: Any) -> None:
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
        - type: "error" - Error message (API errors, invocation failures, etc.)
        - type: "agent_interrupted" - Agent was interrupted

        User messages can originate from:
        - This dwcli instance
        - Other clients (other dwcli instances, task manager, etc.)

        Regardless of origin, ALL messages go through agent and come back
        via RPyC callbacks. This ensures dwcli shows complete conversation
        state from agent, maintaining single source of truth.

        Args:
            msg: Either a JSON-string (new agent contract) or a dict (legacy).
                 The agent-side ``_notify_subscribers`` in the RPyC service now
                 ``json.dumps`` the payload once and ships a plain ``str`` so
                 subscribers never touch netrefs. We accept both shapes for
                 transitional compatibility.
        """
        import json

        if isinstance(msg, (bytes, bytearray)):
            try:
                msg = msg.decode("utf-8")
            except UnicodeDecodeError:
                return
        if isinstance(msg, str):
            try:
                msg = json.loads(msg)
            except (json.JSONDecodeError, ValueError):
                # Malformed — just surface the raw text as an error line.
                msg = {"type": "error", "content": msg}
        if not isinstance(msg, dict):
            # Unknown shape (netref from an even older agent, or None).
            return

        # Narrow for the type checker: ``msg`` is now definitely a local
        # ``dict[str, Any]`` (decoded above from JSON-string/bytes or
        # received directly from a legacy agent).
        msg = cast("dict[str, Any]", msg)
        msg_type = msg.get("type")
        role = msg.get("role")
        content = msg.get("content", "")

        # Hide spinner when agent provides substantial response
        # Keep spinner visible for very short responses (e.g., single char "I")
        # to avoid flickering and show "Thinking..." while model is still generating
        should_hide_spinner = False
        if self._loading_widget is not None:
            if msg_type == "tool_call":
                # Always hide on tool_call - agent is taking action
                should_hide_spinner = True
            elif msg_type == "assistant":
                # Complete message - hide spinner
                should_hide_spinner = True
            elif msg_type == "assistant_chunk":
                # Only hide if we've accumulated substantial text (>3 chars)
                # This prevents hiding spinner too early when model emits just "I"
                # but allows short complete answers like "4" to hide the spinner
                messages_container = self.query_one("#messages")
                if messages_container.children:
                    last_widget = messages_container.children[-1]
                    if isinstance(last_widget, AssistantMessage):
                        # Check accumulated content length
                        accumulated_len = len(last_widget._content) + len(content)
                        if accumulated_len > 3:
                            should_hide_spinner = True

        if should_hide_spinner:
            await self._set_spinner(None)
            self._update_status("Ready")

        if msg_type == "user" or (msg_type is None and role == "user"):
            # User message from agent - display it
            # (Could have originated from this dwcli, another client, or task manager)

            # CRITICAL FIX: Remove completed tool calls from previous queries when a NEW
            # user message arrives. This prevents duplication bug. We check if this is
            # not the first message to avoid removing during initial load.
            if self._history_total > 0:
                # Not first message - safe to remove completed tool calls
                await self._remove_completed_tool_calls()

            import uuid
            msg_id = f"user-{uuid.uuid4().hex[:8]}"
            await self._mount_message(UserMessage(content, id=msg_id))
            # Update total count
            self._history_total += 1
            # DO NOT scroll after user message - this scrolls ChatInput off-screen
            # for long multi-line messages, making dwcli completely unusable.
            # User is already at bottom when typing, so no scroll needed.

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
            # Scroll handled by sticky scroll in _mount_message() -> _scroll_chat_to_bottom()

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

        elif msg_type == "error":
            # Error message from agent (e.g., API errors, invocation failures)
            # Hide spinner and show error
            self._update_status("Error")
            await self._set_spinner(None)

            # Show error message using ErrorMessage widget
            await self._mount_message(
                ErrorMessage(content)
            )

        elif msg_type == "agent_interrupted":
            # Agent was interrupted (by this or another client)
            self._update_status("Interrupted")
            await self._set_spinner(None)  # Hide spinner

            # Show interruption message
            await self._mount_message(
                AppMessage("⚠ Interrupted")
            )

        elif msg_type == "token_update":
            # Update token count from agent stream metadata
            total_tokens = msg.get("total_tokens", 0)
            self._current_token_count = total_tokens

            if self._status_bar:
                self._status_bar.set_tokens(total_tokens)

        elif msg_type == "file_diff":
            # Display diff widget for file operations
            from deep_worker_cli.widgets.diff import EnhancedDiff

            diff_content = msg.get("diff_content", "")
            file_path = msg.get("file_path", "File")
            tool_name = msg.get("tool_name", "file operation")

            # Create and mount diff widget
            diff_widget = EnhancedDiff(
                diff=diff_content,
                title=f"{tool_name}: {file_path}",
                max_lines=100
            )
            await self._mount_message(diff_widget)

        elif msg_type == "approval_request":
            # Agent requests approval for dangerous operation
            # This is broadcast to ALL clients, but only one client needs to respond
            await self._show_approval_menu(
                approval_id=msg["approval_id"],
                actions=msg["actions"]
            )

        elif msg_type == "approval_result":
            # Another client (or this client) made a decision
            await self._hide_approval_menu()

            # Show result message
            decision = msg.get("decision", {})
            decision_type = decision.get("type", "unknown")

            if decision_type == "approve":
                result_msg = "✓ Approved"
            elif decision_type == "reject":
                result_msg = "✗ Rejected"
            elif decision_type == "auto_approve":
                result_msg = "✓ Auto-approved all"
            else:
                result_msg = f"Decision: {decision_type}"

            await self._mount_message(AppMessage(result_msg))

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
        if not self._rpyc_root:
            raise RuntimeError("Not connected to agent (RPyC)")

        # Type narrowing: after the check above, _rpyc_root is guaranteed to be not None
        assert self._rpyc_root is not None

        # DEBUG: Log before RPyC call
        # CRITICAL FIX: With stream=True, DON'T await the response!
        # Deep-worker uses RPyC client-server architecture (unlike deepagents which calls directly).
        # When we await async_call(), the RPyC connection is BLOCKED waiting for the response,
        # and CANNOT receive async callbacks simultaneously!
        #
        # Solution: Fire-and-forget the invoke call, rely ONLY on RPyC callbacks for messages.
        # This mimics deepagents architecture where agent.astream() chunks are processed in
        # the same event loop, but adapted for client-server separation.

        import asyncio

        # CRITICAL FIX: DON'T await the invoke_agent() result when streaming!
        # The issue: awaiting the result BLOCKS waiting for the response,
        # but we want the connection to remain free for callbacks.
        #
        # Solution: Call invoke_agent() but DON'T await it. This keeps
        # the RPyC connection FREE to process reverse calls from server->client.
        #
        # The invoke_agent() call will execute on the server side, and all messages
        # will arrive via RPyC callbacks (subscribe_messages -> on_message).
        try:
            # Type narrowing: _rpyc_root is guaranteed to be not None
            assert self._rpyc_root is not None

            # Fire-and-forget: Call but DON'T await it!
            # This allows the RPyC connection to remain free for callbacks
            # Note: We create a task but don't await it
            asyncio.create_task(self._rpyc_root.invoke_agent(message, True))
        except Exception as e:
            pass  # Log errors if needed

        # Return immediately - messages will arrive via RPyC callbacks
        return {
            "success": True,
            "response": {
                "role": "assistant",
                "content": "(streaming - messages delivered via RPyC callbacks)"
            }
        }

    async def disconnect_from_agent(self) -> None:
        """Disconnect from the agent gracefully."""
        # Disconnect from RPyC
        if self._rpyc_root and self.rpyc_callback:
            # Unsubscribe from messages
            try:
                await self._rpyc_root.unsubscribe_messages(self.rpyc_callback)
            except Exception:
                pass

        if self._rpyc_conn:
            try:
                # ═══════════════════════════════════════════════════════════════
                # async close — matches async_connect's non-blocking contract.
                # sync close() here would hit rpyc_async's sync_request guard
                # inside the running Textual loop.
                #
                # FORBIDDEN: annotating this call with type ignore comments
                # (type: ignore, ty: ignore[...], pyright: ignore[...], etc.)
                # Type checks MUST see `rpyc_async` via
                # `[tool.ty.environment].extra-paths` in pyproject.toml.
                # If `ty` ever claims `Connection` has no `aclose`, the fix is
                # to restore that extra-paths entry, NOT to silence the check.
                # ═══════════════════════════════════════════════════════════════
                await self._rpyc_conn.aclose()
            except Exception:
                pass
            self._rpyc_conn = None
            self._rpyc_root = None
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

        # CRITICAL FIX: Remove completed tool calls from previous queries IMMEDIATELY
        # when user sends a new message, BEFORE it goes to the agent.
        # This ensures old tool call widgets disappear instantly when user presses Enter,
        # providing responsive UX without waiting for round-trip through agent.
        #
        # We still remove again when user message comes back via RPyC callback (line 464),
        # which handles cases where user message originates from OTHER clients.
        if self._history_total > 0:
            await self._remove_completed_tool_calls()

        # Don't update status here - spinner already shows "Thinking..."
        # Status will be updated when response arrives

        # Run agent request in background worker (like deepagents does)
        # This keeps Textual's event loop responsive for rendering
        self.run_worker(self._agent_task(message))

    async def _set_spinner(self, status: str | None) -> None:
        """Show, update, or hide the loading spinner.

        Args:
            status: The status text to display (e.g., "Thinking", "Summarizing"),
                or `None` to hide the spinner.
        """
        # DEBUG: Log spinner state changes
        if status is None:
            # Hide
            if self._loading_widget:
                await self._loading_widget.remove()
                self._loading_widget = None
            return

        messages_container = self.query_one("#messages", Container)

        if self._loading_widget is None:
            # Create new
            self._loading_widget = LoadingWidget(status)
            await messages_container.mount(self._loading_widget)
        else:
            # Update existing
            self._loading_widget.set_status(status)
            # Reposition if not at the end (e.g., after tool message was added)
            children = list(messages_container.children)
            if children and children[-1] != self._loading_widget:
                await self._loading_widget.remove()
                await messages_container.mount(self._loading_widget)
        # Don't auto-scroll - let user stay at their current position

    async def _mount_message(self, widget: Any) -> None:
        """Mount a message widget to the messages container.

        Args:
            widget: Message widget to mount

        Scroll policy (sticky-scroll-to-bottom):
        - If the user is already near the bottom (was watching live messages),
          we keep the input area + newest message visible by scrolling down.
        - If the user has scrolled up to read older messages, we DO NOT yank
          them back to the bottom; their reading position is preserved and
          the input area is allowed to remain off-screen below until the user
          chooses to scroll back down.

        This matches the user-stated requirement: the input must remain
        visible at the bottom IF it was visible to begin with (user not
        scrolled up), even as new messages stream in.
        """
        messages_container = self.query_one("#messages", Container)
        await messages_container.mount(widget)

        # Decide stickiness BEFORE scrolling, using the pre-mount distance from
        # bottom that _scroll_chat_to_bottom() also uses. We replicate the
        # threshold here so we can guard `scroll_visible()` on the same
        # condition — otherwise an unconditional scroll_visible() on the input
        # container would drag the user back down even when they intentionally
        # scrolled up to read older history.
        was_near_bottom = True
        try:
            chat = self.query_one("#chat", VerticalScroll)
            if chat.max_scroll_y > 0:
                distance_from_bottom = chat.max_scroll_y - chat.scroll_y
                was_near_bottom = distance_from_bottom < 100
        except Exception:
            pass

        self._scroll_chat_to_bottom()

        # Only keep the input area in view if the user was already near the
        # bottom. If they scrolled up to read, leave them where they are.
        if was_near_bottom:
            try:
                input_container = self.query_one("#bottom-app-container", Container)
                input_container.scroll_visible()
            except Exception:
                pass

    def _scroll_chat_to_bottom(self) -> None:
        """Scroll chat to bottom using sticky scroll pattern (like deepagents).

        Only scrolls if user is already at/near the bottom.
        This prevents dragging the user back if they've scrolled up to read,
        and prevents scrolling the ChatInput widget off-screen for long messages.

        Uses immediate=True so the scroll is applied in the current event-loop
        tick rather than being queued behind pending messages.  Without it a
        burst of incoming messages can leave scroll_y lagging behind max_scroll_y.
        """
        chat = self.query_one("#chat", VerticalScroll)

        # Nothing to scroll if content fits in viewport
        if chat.max_scroll_y <= 0:
            return

        # Sticky scroll: only scroll to bottom if user is near the bottom
        # "Near" means within 100 pixels of the bottom (about 6-7 lines)
        # This matches deepagents implementation
        distance_from_bottom = chat.max_scroll_y - chat.scroll_y
        if distance_from_bottom < 100:
            chat.scroll_end(animate=False, immediate=True)

    async def _remove_completed_tool_calls(self) -> None:
        """Remove completed tool call widgets from previous queries.

        This prevents the tool call duplication bug where old tool calls
        from previous queries remain visible when a new user query starts.

        Tool calls that are still running (in _current_tool_messages) are
        kept visible. Only completed tool calls are removed from DOM.

        Uses async remove() like deepagents does for message pruning.
        """
        try:
            messages_container = self.query_one("#messages", Container)

            # Find all ToolCallMessage widgets
            tool_widgets = list(messages_container.query(ToolCallMessage))

            # If no tool widgets exist yet, nothing to remove
            if not tool_widgets:
                return

            # Remove completed tool calls from DOM
            removed_count = 0
            for widget in tool_widgets:
                # Check if this tool call is still running
                # Tool calls are removed from _current_tool_messages when completed
                is_running = any(
                    tool_widget is widget
                    for tool_widget in self._current_tool_messages.values()
                )
                # Remove completed tool calls (like deepagents pruning)
                if not is_running:
                    await widget.remove()
                    removed_count += 1
        except NoMatches:
            # No messages container yet - skip
            pass
        except Exception as e:
            # Don't crash on unexpected errors - just skip cleanup
            import sys
            import traceback
            print(f"[DWCLI_DEBUG] _remove_completed_tool_calls error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    async def _load_history(self) -> None:
        """Load and display conversation history from the agent.

        This method retrieves recent messages from the agent's checkpoint state
        and displays them in the chat window WITHOUT auto-scrolling to bottom.
        The input area must remain visible after loading history.
        """
        if not self._rpyc_root:
            return

        try:
            # Request recent history via RPyC
            # Server returns JSON string (per RPYC_SERIALIZATION_DESIGN.md optimization)
            response_json_str = await self._rpyc_root.get_history(limit=50, offset=0)
            response = json.loads(response_json_str)

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

        if not self._rpyc_root:
            return

        self._loading_history = True

        try:
            # Log pagination event for testing
            # Save current scroll position BEFORE loading
            chat_scroll = self.query_one("#chat", VerticalScroll)
            old_scroll_y = chat_scroll.scroll_y
            old_max_y = chat_scroll.max_scroll_y

            # Load next page of older messages via RPyC
            # Server returns JSON string (per RPYC_SERIALIZATION_DESIGN.md optimization)
            response_json_str = await self._rpyc_root.get_history(limit=20, offset=self._history_offset)
            response = json.loads(response_json_str)

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

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands.

        Client-side commands are handled locally (/help, /version, /quit).
        Agent-side commands are delegated to _handle_agent_command.

        Args:
            command: The slash command (including /)
        """
        cmd = command.lower().strip()

        # Client-side commands (no agent interaction needed)
        if cmd in {"/quit", "/q"}:
            await self.action_quit_app()

        elif cmd == "/help":
            await self._mount_message(UserMessage(command))
            help_text = (
                "Commands: /quit, /clear, /remember, /tokens, /threads, /help\n\n"
                "Interactive Features:\n"
                "  Enter           Submit your message\n"
                "  Ctrl+J          Insert newline\n"
                "  Shift+Tab       Toggle auto-approve mode\n"
                "  @filename       Auto-complete files and inject content\n"
                "  /command        Slash commands (/help, /clear, /quit)\n"
                "  !command        Run bash commands directly\n\n"
                "Docs: https://docs.langchain.com/oss/python/deepagents/cli"
            )
            await self._mount_message(AppMessage(help_text))

        elif cmd == "/version":
            await self._mount_message(UserMessage(command))
            try:
                from deep_worker_cli._version import __version__
                await self._mount_message(
                    AppMessage(f"deep-worker-cli version: {__version__}")
                )
            except Exception:
                await self._mount_message(
                    AppMessage("deep-worker-cli version: unknown")
                )

        else:
            # Delegate to agent-side command handler
            await self._handle_agent_command(command)

    async def _handle_bash_command(self, command: str) -> None:
        """Handle bash command (! prefix).

        Executes command locally (not via agent) for immediate response.
        This matches deepagents behavior where bash commands run in client process.

        SECURITY NOTE: User explicitly requested shell execution via ! prefix.

        Args:
            command: The bash command to execute (without ! prefix)
        """
        import subprocess
        import asyncio

        # Display user's bash command
        await self._mount_message(UserMessage(f"!{command}"))

        # Execute locally (same as deepagents)
        # S604: shell=True is intentional - user explicitly requested via ! prefix
        try:
            result = await asyncio.to_thread(  # noqa: S604
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(Path.cwd()),
                timeout=60,
            )

            stdout = result.stdout
            stderr = result.stderr
            if not isinstance(stdout, str):
                stdout = stdout.decode() if stdout else ""
            output = stdout.strip()

            if stderr:
                if not isinstance(stderr, str):
                    stderr = stderr.decode() if stderr else ""
                output += f"\n[stderr]\n{stderr.strip()}"

            if output:
                # Display output as assistant message (uses markdown for code blocks)
                msg = AssistantMessage(f"```\n{output}\n```")
                await self._mount_message(msg)
                await msg.write_initial_content()
            else:
                await self._mount_message(
                    AppMessage("Command completed (no output)")
                )

            if result.returncode != 0:
                await self._mount_message(
                    AppMessage(f"Exit code: {result.returncode}")
                )

            # Scroll handled by sticky scroll in _mount_message() -> _scroll_chat_to_bottom()

        except subprocess.TimeoutExpired:
            await self._mount_message(
                AppMessage("Command timed out (60s limit)")
            )
        except OSError as e:
            await self._mount_message(AppMessage(f"Error: {e}"))

    async def _handle_agent_command(self, command: str) -> None:
        """Handle commands that require agent interaction.

        These commands are sent to the agent via RPyC:
        - /clear - Reset thread
        - /remember - Review conversation and update memory
        - /tokens - Display token count
        - /threads - Display current thread ID

        Args:
            command: Slash command string
        """
        if not self._rpyc_root:
            await self._mount_message(
                AppMessage("⚠ Not connected to agent. Cannot execute command.")
            )
            return

        # Display the command
        await self._mount_message(UserMessage(command))

        # Send to agent via RPyC
        try:
            # Server returns JSON string (per RPYC_SERIALIZATION_DESIGN.md optimization)
            response_json_str = await self._rpyc_root.handle_command(command)
            response = json.loads(response_json_str)

            if not response.get("success"):
                error_msg = response.get("response", "Unknown error")
                await self._mount_message(AppMessage(f"⚠ {error_msg}"))
                return

            action = response.get("action")

            if action == "clear_thread":
                # Clear local message store
                await self._clear_messages()
                self._update_status("")
                # Agent will broadcast system message to all clients

            elif action == "display_info":
                response_text = response.get("response", "")
                if response_text:
                    await self._mount_message(AppMessage(response_text))

            elif action == "send_to_agent":
                # Command wants to send message to agent (e.g., /remember)
                message = response["data"]["message"]
                await self._handle_user_message(message)

            else:
                # Default: just display response
                response_text = response.get("response", "")
                if response_text:
                    await self._mount_message(AppMessage(response_text))

        except Exception as e:
            await self._mount_message(
                AppMessage(f"⚠ Command failed: {e}")
            )

    async def _clear_messages(self) -> None:
        """Clear the messages area and message store."""
        self._message_store.clear()
        try:
            messages = self.query_one("#messages", Container)
            await messages.remove_children()
        except NoMatches:
            pass

    async def _show_approval_menu(
        self,
        approval_id: str,
        actions: list[dict[str, Any]],
    ) -> None:
        """Show approval menu for dangerous operations.

        If auto-approve is enabled, automatically approve without showing menu.

        Args:
            approval_id: Unique ID for this approval request
            actions: List of actions requiring approval
        """
        # Check auto-approve mode
        if self._auto_approve:
            # Auto-approve immediately
            await self._send_approval_decision(
                approval_id,
                {"type": "approve"}
            )

            # Show notification
            await self._mount_message(
                AppMessage("✓ Auto-approved (auto-approve enabled)")
            )
            return

        # Show approval menu
        from deep_worker_cli.widgets.approval import ApprovalMenu

        # Store approval ID
        self._current_approval_id = approval_id

        # Create approval menu
        unique_id = f"approval-menu-{approval_id}"
        menu = ApprovalMenu(actions, id=unique_id)

        # Store reference
        self._pending_approval_widget = menu

        # Mount inline in messages
        try:
            messages = self.query_one("#messages", Container)
            await messages.mount(menu)
            self.call_after_refresh(menu.scroll_visible)
            self.call_after_refresh(menu.focus)
        except Exception as e:
            self._pending_approval_widget = None
            self._current_approval_id = None
            await self._mount_message(
                AppMessage(f"⚠ Failed to show approval menu: {e}")
            )

    async def _hide_approval_menu(self) -> None:
        """Hide approval menu."""
        if self._pending_approval_widget:
            try:
                await self._pending_approval_widget.remove()
            except Exception:
                pass
            self._pending_approval_widget = None
            self._current_approval_id = None

    async def _send_approval_decision(
        self,
        approval_id: str,
        decision: dict[str, Any]
    ) -> None:
        """Send approval decision to agent via RPyC.

        Args:
            approval_id: Approval ID
            decision: Decision dict with "type" field
        """
        if not self._rpyc_root:
            return

        try:
            # Server returns JSON string (per RPYC_SERIALIZATION_DESIGN.md optimization)
            response_json_str = await self._rpyc_root.respond_to_approval(approval_id, decision)
            response = json.loads(response_json_str)

            if not response.get("success"):
                await self._mount_message(
                    AppMessage(f"⚠ {response.get('message', 'Failed to send decision')}")
                )
        except Exception as e:
            await self._mount_message(
                AppMessage(f"⚠ Failed to send approval: {e}")
            )

    async def on_approval_menu_decided(self, event: Any) -> None:
        """Handle approval menu decision.

        Args:
            event: ApprovalMenu.Decided event with decision
        """
        if not self._rpyc_root or not self._current_approval_id:
            return

        # Send decision to agent
        await self._send_approval_decision(
            self._current_approval_id,
            event.decision
        )

        # Hide menu
        await self._hide_approval_menu()

        # Refocus input
        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    def action_toggle_auto_approve(self) -> None:
        """Toggle auto-approve mode (Shift+Tab)."""
        self._auto_approve = not self._auto_approve

        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=self._auto_approve)

        # Notify user
        status = "enabled" if self._auto_approve else "disabled"
        self.notify(f"Auto-approve {status}", timeout=2)

    def action_interrupt(self) -> None:
        """Handle interrupt action (Escape key).

        Priority order:
        1. If approval menu active, reject it
        2. Otherwise, interrupt agent
        """
        # If approval menu active, reject it
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            return

        # Otherwise, interrupt agent
        self.run_worker(self._interrupt_agent())

    def action_quit_or_interrupt(self) -> None:
        """Handle Ctrl+C: Interrupt agent if running, otherwise quit.

        This matches deepagents behavior where Ctrl+C prioritizes interruption.
        Use Ctrl+D to quit directly.
        """
        # If approval menu active, reject it
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            return

        # Otherwise, interrupt agent (if running)
        # We don't track agent running state locally, so always try to interrupt
        self.run_worker(self._interrupt_agent())

    async def _interrupt_agent(self) -> None:
        """Send interrupt signal to agent via RPyC.

        This is a best-effort operation - if agent is not running,
        the error is silently ignored.
        """
        if not self._rpyc_root:
            return

        try:
            # Server returns JSON string (per RPYC_SERIALIZATION_DESIGN.md optimization)
            response_json_str = await self._rpyc_root.interrupt_agent()
            response = json.loads(response_json_str)

            if response.get("success"):
                self._update_status("Interrupted")
            # If not successful, agent likely wasn't running - ignore silently

        except Exception as e:
            # Log but don't show error (interrupt is best-effort)
            import sys
            print(f"[INTERRUPT] Failed to interrupt: {e}", file=sys.stderr)

    async def action_quit_app(self) -> None:
        """Handle quit action (Ctrl+D)."""
        await self.disconnect_from_agent()
        self.exit(0)

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle chat input submission.

        Dispatches to appropriate handler based on mode:
        - "command" -> slash commands (/help, /quit, etc.)
        - "bash" -> bash commands (! prefix)
        - "normal" -> user message to agent

        Args:
            event: ChatInput.Submitted event with value and mode
        """
        import os
        debug = os.environ.get("DWCLI_DEBUG") == "1"

        if debug:
            import sys
            print(f"[DWCLI_DEBUG] on_chat_input_submitted: value='{event.value}', mode='{event.mode}'", file=sys.stderr)

        # Dispatch based on mode
        if event.mode == "command":
            await self._handle_command(event.value)
        elif event.mode == "bash":
            await self._handle_bash_command(event.value.removeprefix("!"))
        else:
            # Normal message
            # NOTE: Do NOT clear input here! Input text should stay visible while agent processes.
            # Input will be cleared after agent responds (in _handle_realtime_message for assistant message)
            await self._handle_user_message(event.value)
