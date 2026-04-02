# Design Document: dwcli Missing Features Implementation

**Date:** 2026-03-14
**Status:** Draft for Review

---

## Executive Summary

This document describes the design for implementing missing TUI features in `deep-worker-cli` (dwcli) to achieve feature parity with the original `deepagents-cli` interactive mode, while maintaining the client-server architecture via RPyC.

**Key Principle:** Agent remains the single source of truth for all messages passing through the agent graph. Multiple clients can connect to the same agent, and all clients must display all messages (including user messages from other clients) identically.

---

## 1. Missing Features Analysis

### 1.1 Critical Missing Features

| Feature | deepagents | dwcli | Impact |
|---------|-----------|-------|--------|
| **Slash commands** | ✅ 7 commands | ❌ None | HIGH - Core UX functionality |
| **Interrupt agent** | ✅ Esc/Ctrl+C | ❌ None | CRITICAL - User control |
| **Approval menu (HITL)** | ✅ Full support | ❌ None | HIGH - Dangerous operations |
| **Bash commands** | ✅ ! prefix | ❌ None | MEDIUM - Power user feature |
| **Auto-approve** | ✅ Toggle + status | ❌ None | MEDIUM - UX convenience |
| **Token tracking** | ✅ Live display | ❌ Not updated | LOW - Informational |
| **DiffWidget** | ✅ File diffs | ❌ Not displayed | MEDIUM - Code review |

### 1.2 Architecture Context

**deepagents architecture:**
```
┌─────────────────────────────────┐
│  DeepAgentsApp (Textual)        │
│  ├─ TextualUIAdapter            │
│  ├─ Agent (LangGraph) [LOCAL]   │
│  └─ Direct async streaming      │
└─────────────────────────────────┘
```

**dwcli architecture:**
```
┌──────────────────┐         RPyC          ┌──────────────────┐
│ InteractiveApp   │<--------------------- │ AgentProcess     │
│ (Textual Client) │   Callbacks/Stream    │ (Server)         │
│                  │ ---------------------> │ ├─ Agent (Graph) │
│                  │   RPC Calls           │ └─ AgentService  │
└──────────────────┘                       └──────────────────┘
        ▲                                           ▲
        │                                           │
        └─── Multiple clients can connect ─────────┘
             (dwcli, task-manager, other TUIs)
```

**Key constraint:** All features must work through RPyC with multiple concurrent clients.

---

## 2. Implementation Strategy

### 2.1 Feature Categories

Features fall into three categories based on where they execute:

#### Category A: Client-Side Features (No Agent State Change)
Execute locally in dwcli, no RPyC needed.

- Slash commands: `/help`, `/version`
- UI controls: Scroll, focus, display toggles

#### Category B: Agent-Side Features (Modify Agent State)
Execute on agent, require RPyC calls, affect all clients.

- Slash commands: `/clear`, `/remember`, `/tokens`, `/threads`
- Bash commands (via tool execution)
- Agent interruption

#### Category C: Hybrid Features (Local + Remote)
UI rendered locally, but interact with agent state via RPyC.

- Approval menu (HITL) - UI local, decision sent to agent
- Auto-approve toggle - state synced via RPyC
- Token tracking - received from agent stream metadata

---

## 3. Detailed Feature Designs

### 3.1 Slash Commands

#### 3.1.1 Client-Side Commands

**Commands:** `/help`, `/version`, `/quit`

**Implementation Location:** `deep_worker_cli/cli/interactive.py`

**Design:**
```python
class InteractiveApp(App):
    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        value = event.value
        mode = event.mode

        if mode == "command":
            await self._handle_command(value)
        else:
            await self._handle_user_message(value)

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        cmd = command.lower().strip()

        if cmd in {"/quit", "/q"}:
            self.exit()

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
            # Delegate to agent-side commands
            await self._handle_agent_command(command)
```

**Files Modified:**
- `deep_worker_cli/cli/interactive.py` - Add command handling
- `deep_worker_cli/widgets/chat_input.py` - Detect command mode (already supports this)

**Key Bindings:**
- Existing ChatInput already detects `/` prefix and sets mode="command"
- No additional bindings needed

#### 3.1.2 Agent-Side Commands

**Commands:** `/clear`, `/remember`, `/tokens`, `/threads`

**RPyC Service Extension:**

**File:** `deep_worker_cli/rpyc_service/agent_service.py`

```python
class AgentService(rpyc.SlaveService):
    # ... existing methods ...

    def exposed_handle_command(self, command: str) -> dict[str, Any]:
        """Handle agent-side slash commands.

        Args:
            command: Full command string (e.g., "/clear", "/remember", etc.)

        Returns:
            {
                "success": bool,
                "response": str,  # Response message to display
                "action": str,    # Action taken: "clear_thread", "display_info", etc.
                "data": dict      # Optional additional data
            }
        """
        cmd = command.lower().strip()

        if cmd == "/clear":
            # Reset thread to new ID
            old_thread_id = self.agent_process.current_thread_id
            new_thread_id = uuid.uuid4().hex[:8]
            self.agent_process.current_thread_id = new_thread_id

            # Notify all subscribers about thread reset
            self._broadcast_message({
                "type": "system",
                "role": "system",
                "content": f"Started new thread: {new_thread_id}",
                "action": "clear_thread",
                "old_thread_id": old_thread_id,
                "new_thread_id": new_thread_id,
            })

            return {
                "success": True,
                "response": f"Started new thread: {new_thread_id}",
                "action": "clear_thread",
                "data": {
                    "old_thread_id": old_thread_id,
                    "new_thread_id": new_thread_id,
                }
            }

        elif cmd == "/threads":
            thread_id = self.agent_process.current_thread_id
            return {
                "success": True,
                "response": f"Current thread: {thread_id}",
                "action": "display_info",
                "data": {"thread_id": thread_id}
            }

        elif cmd == "/tokens":
            # Get token count from last response metadata
            token_count = self.agent_process.last_token_count or 0
            if token_count >= 1000:
                formatted = f"{token_count / 1000:.1f}K"
            else:
                formatted = str(token_count)

            return {
                "success": True,
                "response": f"Current context: {formatted} tokens",
                "action": "display_info",
                "data": {"token_count": token_count}
            }

        elif cmd == "/remember" or cmd.startswith("/remember "):
            # Extract context
            additional_context = ""
            if cmd.startswith("/remember "):
                additional_context = command.strip()[len("/remember "):]

            # Build remember prompt (same as deepagents)
            remember_prompt = REMEMBER_PROMPT
            if additional_context:
                remember_prompt += f"\n\n**Additional context:** {additional_context}"

            # Send as user message to agent (will be processed normally)
            return {
                "success": True,
                "response": "",  # No immediate response, agent will process
                "action": "send_to_agent",
                "data": {"message": remember_prompt}
            }

        else:
            return {
                "success": False,
                "response": f"Unknown command: {cmd}",
                "action": "error",
            }

    def _broadcast_message(self, message: dict[str, Any]) -> None:
        """Broadcast message to all subscribed clients."""
        for callback in self._subscribers:
            try:
                callback.on_message(message)
            except Exception:
                pass  # Ignore callback errors
```

**Client-Side Handler:**

**File:** `deep_worker_cli/cli/interactive.py`

```python
class InteractiveApp(App):
    async def _handle_agent_command(self, command: str) -> None:
        """Handle commands that require agent interaction.

        Args:
            command: Slash command string
        """
        if not self.rpyc_session:
            await self._mount_message(
                AppMessage("⚠ Not connected to agent. Cannot execute command.")
            )
            return

        # Display the command
        await self._mount_message(UserMessage(command))

        # Send to agent via RPyC
        response = await self.rpyc_session.async_call(
            self.rpyc_session.root.handle_command,
            command
        )

        if not response.get("success"):
            error_msg = response.get("response", "Unknown error")
            await self._mount_message(AppMessage(f"⚠ {error_msg}"))
            return

        action = response.get("action")

        if action == "clear_thread":
            # Clear local message store
            await self._clear_messages()
            if self._token_tracker:
                self._token_tracker.reset()
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

    async def _clear_messages(self) -> None:
        """Clear the messages area and message store."""
        self._message_store.clear()
        try:
            messages = self.query_one("#messages", Container)
            await messages.remove_children()
        except NoMatches:
            pass
```

**Files Modified:**
- `deep_worker_cli/rpyc_service/agent_service.py` - Add `exposed_handle_command()`
- `deep_worker_cli/process/agent_process.py` - Track `current_thread_id`, `last_token_count`
- `deep_worker_cli/cli/interactive.py` - Add `_handle_agent_command()`, `_clear_messages()`

**Multi-Client Behavior:**
- `/clear` broadcasts to ALL clients → all clear their UI
- `/threads`, `/tokens` display info only to requesting client
- `/remember` sends prompt to agent → all clients see the conversation

---

### 3.2 Bash Commands (! prefix)

**Implementation:** Execute via agent's bash tool (if available) OR locally in client.

**Design Decision:** Execute locally in dwcli (simpler, matches deepagents UX).

**Why local execution:**
- Bash commands don't modify agent state
- Faster (no RPyC round-trip)
- User expects immediate local execution
- Consistent with deepagents behavior

**Implementation Location:** `deep_worker_cli/cli/interactive.py`

```python
class InteractiveApp(App):
    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        value = event.value
        mode = event.mode

        if mode == "bash":
            await self._handle_bash_command(value.removeprefix("!"))
        elif mode == "command":
            await self._handle_command(value)
        else:
            await self._handle_user_message(value)

    async def _handle_bash_command(self, command: str) -> None:
        """Handle bash command (! prefix).

        Args:
            command: Bash command to execute (without ! prefix)
        """
        import subprocess

        # Display user's bash command
        await self._mount_message(UserMessage(f"!{command}"))

        # Execute locally (same as deepagents)
        try:
            result = await asyncio.to_thread(
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

            # Scroll to show output
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)

        except subprocess.TimeoutExpired:
            await self._mount_message(
                AppMessage("Command timed out (60s limit)")
            )
        except OSError as e:
            await self._mount_message(AppMessage(f"Error: {e}"))
```

**Files Modified:**
- `deep_worker_cli/cli/interactive.py` - Add `_handle_bash_command()`
- `deep_worker_cli/widgets/chat_input.py` - Already detects `!` prefix (verify)

**Security Note:** Matches deepagents security model - user explicitly requested shell execution.

---

### 3.3 Approval Menu (HITL - Human-In-The-Loop)

**Challenge:** Agent runs in separate process, approval UI must be in client.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│ AgentProcess (Server)                                           │
│                                                                 │
│ Agent executes → Encounters approval interrupt → RPyC callback │
│   └─> Sends approval request to ALL clients                    │
└─────────────────────────────────────────────────────────────────┘
                          ↓ RPyC Callback
┌─────────────────────────────────────────────────────────────────┐
│ InteractiveApp (Client)                                         │
│                                                                 │
│ Receives approval request → Shows ApprovalMenu                 │
│ User decides (Approve/Reject/Auto) → Sends decision via RPyC   │
└─────────────────────────────────────────────────────────────────┘
                          ↓ RPyC Call
┌─────────────────────────────────────────────────────────────────┐
│ AgentProcess (Server)                                           │
│                                                                 │
│ Receives decision → Resumes agent with decision                │
│   └─> Broadcasts result to ALL clients                         │
└─────────────────────────────────────────────────────────────────┘
```

**Multi-Client Behavior:**
- **First client to respond wins** - decision is applied
- All clients see the approval request
- All clients see the decision result (approved/rejected)

**RPyC Service Extension:**

**File:** `deep_worker_cli/rpyc_service/agent_service.py`

```python
class AgentService(rpyc.SlaveService):
    def __init__(self, agent_process):
        super().__init__()
        self.agent_process = agent_process
        self._subscribers = []
        self._approval_futures = {}  # approval_id -> asyncio.Future

    def exposed_respond_to_approval(
        self,
        approval_id: str,
        decision: dict[str, Any]
    ) -> dict[str, Any]:
        """Receive approval decision from client.

        Args:
            approval_id: Unique ID for this approval request
            decision: {
                "type": "approve" | "reject" | "auto_approve",
                "edits": {...}  # Optional edits for tools
            }

        Returns:
            {"success": bool, "message": str}
        """
        if approval_id not in self._approval_futures:
            return {
                "success": False,
                "message": "Approval request not found or already resolved"
            }

        future = self._approval_futures[approval_id]

        if future.done():
            return {
                "success": False,
                "message": "Approval already resolved by another client"
            }

        # Set the future result (resumes agent)
        future.set_result(decision)

        # Broadcast decision to all clients
        self._broadcast_message({
            "type": "approval_result",
            "approval_id": approval_id,
            "decision": decision,
        })

        # Clean up
        del self._approval_futures[approval_id]

        return {"success": True, "message": "Decision received"}
```

**Agent Process Integration:**

**File:** `deep_worker_cli/process/agent_process.py`

```python
class AgentProcess:
    async def _execute_agent_request(self, prompt: str, config: dict) -> None:
        """Execute agent with HITL support."""

        # Create custom config with approval callback
        approval_callback = self._request_approval_via_rpyc

        stream_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "approval_callback": approval_callback,
            }
        }

        # Stream agent responses
        async for chunk in self.agent.astream(..., config=stream_config):
            # Check for approval interrupts
            if isinstance(chunk, dict) and chunk.get("type") == "interrupt":
                interrupt_data = chunk.get("data", {})
                if interrupt_data.get("reason") == "approval_required":
                    # Approval is handled via callback, continue streaming
                    continue

            # Normal streaming...
            await self._process_chunk(chunk)

    async def _request_approval_via_rpyc(
        self,
        action_requests: list[dict[str, Any]]
    ) -> asyncio.Future:
        """Request approval from clients via RPyC.

        This is called by the agent when it encounters an approval interrupt.

        Args:
            action_requests: List of actions requiring approval

        Returns:
            Future that resolves to approval decision
        """
        approval_id = uuid.uuid4().hex[:8]

        # Create future that will be resolved when client responds
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Store in service for client callback
        self.rpyc_service._approval_futures[approval_id] = future

        # Broadcast approval request to ALL clients
        approval_message = {
            "type": "approval_request",
            "approval_id": approval_id,
            "actions": action_requests,
            "assistant_id": self.assistant_id,
        }

        self.rpyc_service._broadcast_message(approval_message)

        # Return future (agent will await this)
        return future
```

**Client-Side Implementation:**

**File:** `deep_worker_cli/cli/interactive.py`

```python
class InteractiveApp(App):
    BINDINGS: ClassVar[list[BindingType]] = [
        # ... existing bindings ...

        # Approval menu keys
        Binding("up", "approval_up", "Up", show=False),
        Binding("k", "approval_up", "Up", show=False),
        Binding("down", "approval_down", "Down", show=False),
        Binding("j", "approval_down", "Down", show=False),
        Binding("enter", "approval_select", "Select", show=False),
        Binding("y", "approval_yes", "Yes", show=False),
        Binding("1", "approval_yes", "Yes", show=False),
        Binding("n", "approval_no", "No", show=False),
        Binding("2", "approval_no", "No", show=False),
        Binding("a", "approval_auto", "Auto", show=False),
        Binding("3", "approval_auto", "Auto", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... existing init ...
        self._pending_approval_widget: ApprovalMenu | None = None
        self._current_approval_id: str | None = None

    async def _handle_realtime_message(self, msg: dict[str, Any]) -> None:
        """Handle real-time message from agent."""
        msg_type = msg.get("type")

        # ... existing message handling ...

        if msg_type == "approval_request":
            # Show approval menu
            await self._show_approval_menu(
                approval_id=msg["approval_id"],
                actions=msg["actions"],
                assistant_id=msg.get("assistant_id"),
            )

        elif msg_type == "approval_result":
            # Another client made decision, hide our menu
            await self._hide_approval_menu()

            # Show result message
            decision_type = msg["decision"]["type"]
            if decision_type == "approve":
                result_msg = "✓ Approved"
            elif decision_type == "reject":
                result_msg = "✗ Rejected"
            else:
                result_msg = "✓ Auto-approved all"

            await self._mount_message(AppMessage(result_msg))

    async def _show_approval_menu(
        self,
        approval_id: str,
        actions: list[dict[str, Any]],
        assistant_id: str | None,
    ) -> None:
        """Show approval menu for action requests.

        Args:
            approval_id: Unique ID for this approval
            actions: List of actions to approve
            assistant_id: Agent ID
        """
        # If already showing approval, queue this one
        if self._pending_approval_widget is not None:
            # Wait for current approval to complete
            while self._pending_approval_widget is not None:
                await asyncio.sleep(0.1)

        # Create approval menu
        from deep_worker_cli.widgets.approval import ApprovalMenu

        unique_id = f"approval-menu-{uuid.uuid4().hex[:8]}"
        menu = ApprovalMenu(actions, assistant_id, id=unique_id)

        # Store reference
        self._pending_approval_widget = menu
        self._current_approval_id = approval_id

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
            await self._pending_approval_widget.remove()
            self._pending_approval_widget = None
            self._current_approval_id = None

    async def on_approval_menu_decided(self, event: ApprovalMenu.Decided) -> None:
        """Handle approval menu decision."""
        if not self.rpyc_session or not self._current_approval_id:
            return

        # Send decision to agent
        response = await self.rpyc_session.async_call(
            self.rpyc_session.root.respond_to_approval,
            self._current_approval_id,
            event.decision
        )

        if not response.get("success"):
            await self._mount_message(
                AppMessage(f"⚠ {response.get('message', 'Failed to send decision')}")
            )

        # Hide menu
        await self._hide_approval_menu()

        # Refocus input
        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    # Approval menu action handlers (delegate to widget if present)
    def action_approval_up(self) -> None:
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_up()

    def action_approval_down(self) -> None:
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_down()

    def action_approval_select(self) -> None:
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_select()

    def action_approval_yes(self) -> None:
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_approve()

    def action_approval_no(self) -> None:
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    def action_approval_auto(self) -> None:
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_auto()

    def _is_input_focused(self) -> bool:
        """Check if chat input has focus."""
        if not self._chat_input:
            return False
        focused = self.focused
        if focused is None:
            return False
        return focused.id == "chat-input" or focused in self._chat_input.walk_children()
```

**Files Modified:**
- `deep_worker_cli/rpyc_service/agent_service.py` - Add approval request/response handling
- `deep_worker_cli/process/agent_process.py` - Integrate approval callback
- `deep_worker_cli/cli/interactive.py` - Add approval menu display and handling
- `deep_worker_cli/widgets/approval.py` - **VERIFY ALREADY EXISTS** (it should)

**Multi-Client Edge Cases:**
- If two clients respond simultaneously, first response wins (Future already resolved)
- All clients see the result via broadcast
- If requesting client disconnects, other clients can still respond

---

### 3.4 Auto-Approve Toggle

**Implementation:** Client-side state + status bar indicator.

**Design:**

```python
class InteractiveApp(App):
    BINDINGS: ClassVar[list[BindingType]] = [
        # ... existing ...
        Binding("ctrl+t", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
        Binding("shift+tab", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... existing ...
        self._auto_approve = False

    def action_toggle_auto_approve(self) -> None:
        """Toggle auto-approve mode."""
        self._auto_approve = not self._auto_approve

        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=self._auto_approve)

        # Notify user
        status = "enabled" if self._auto_approve else "disabled"
        self.notify(f"Auto-approve {status}", timeout=2)

    async def _show_approval_menu(
        self,
        approval_id: str,
        actions: list[dict[str, Any]],
        assistant_id: str | None,
    ) -> None:
        """Show approval menu (or auto-approve if enabled)."""

        # Check auto-approve
        if self._auto_approve:
            # Auto-approve all actions
            await self._send_approval_decision(
                approval_id,
                {"type": "approve"}
            )

            # Show notification
            await self._mount_message(
                AppMessage("✓ Auto-approved (auto-approve enabled)")
            )
            return

        # Show menu normally
        # ... existing menu code ...

    async def _send_approval_decision(
        self,
        approval_id: str,
        decision: dict[str, Any]
    ) -> None:
        """Send approval decision to agent."""
        if not self.rpyc_session:
            return

        response = await self.rpyc_session.async_call(
            self.rpyc_session.root.respond_to_approval,
            approval_id,
            decision
        )

        if not response.get("success"):
            await self._mount_message(
                AppMessage(f"⚠ {response.get('message', 'Failed to send decision')}")
            )
```

**Files Modified:**
- `deep_worker_cli/cli/interactive.py` - Add auto-approve toggle
- `deep_worker_cli/widgets/status.py` - **VERIFY** `set_auto_approve()` method exists

**Multi-Client Behavior:**
- Auto-approve is **per-client** (not synced across clients)
- Each client can have different auto-approve setting
- First response (auto or manual) wins

---

### 3.5 Agent Interruption (Escape/Ctrl+C)

**Challenge:** Agent runs in separate process, must send interrupt via RPyC.

**Design:**

**RPyC Service Extension:**

**File:** `deep_worker_cli/rpyc_service/agent_service.py`

```python
class AgentService(rpyc.SlaveService):
    def exposed_interrupt_agent(self) -> dict[str, Any]:
        """Interrupt currently running agent execution.

        Returns:
            {"success": bool, "message": str}
        """
        if not self.agent_process.is_running:
            return {
                "success": False,
                "message": "No agent execution in progress"
            }

        # Cancel the agent worker task
        if self.agent_process.current_worker_task:
            self.agent_process.current_worker_task.cancel()

            # Broadcast interrupt to all clients
            self._broadcast_message({
                "type": "agent_interrupted",
                "role": "system",
                "content": "Agent interrupted by user",
            })

            return {
                "success": True,
                "message": "Agent interrupted"
            }

        return {
            "success": False,
            "message": "No active agent task to interrupt"
        }
```

**Agent Process Tracking:**

**File:** `deep_worker_cli/process/agent_process.py`

```python
class AgentProcess:
    def __init__(self, ...):
        # ... existing ...
        self.is_running = False
        self.current_worker_task = None

    async def _execute_agent_request(self, prompt: str, config: dict) -> None:
        """Execute agent with cancellation support."""
        self.is_running = True

        # Create task for this execution
        self.current_worker_task = asyncio.current_task()

        try:
            # Stream agent responses
            async for chunk in self.agent.astream(...):
                # Process chunk
                await self._process_chunk(chunk)

        except asyncio.CancelledError:
            # Task was cancelled (interrupted)
            # Clean up and re-raise
            self._broadcast_message({
                "type": "agent_interrupted",
                "role": "system",
                "content": "Interrupted",
            })
            raise

        finally:
            self.is_running = False
            self.current_worker_task = None
```

**Client-Side Interrupt:**

**File:** `deep_worker_cli/cli/interactive.py`

```python
class InteractiveApp(App):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+c", "quit_or_interrupt", "Quit/Interrupt", show=False),
        # ... existing ...
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... existing ...
        self._quit_pending = False

    def action_interrupt(self) -> None:
        """Interrupt agent (Escape key)."""
        # If approval menu active, reject it
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            return

        # Otherwise, interrupt agent
        self.run_worker(self._interrupt_agent())

    def action_quit_or_interrupt(self) -> None:
        """Ctrl+C: Interrupt agent, or quit on double press."""
        # If approval menu active, reject it
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            self._quit_pending = False
            return

        # If agent running, interrupt it
        # (We don't track agent state locally, so always try to interrupt)
        self.run_worker(self._interrupt_agent())
        self._quit_pending = False
        return

        # Double Ctrl+C to quit
        # (Removed - interrupt takes priority, use Ctrl+D to quit)

    async def _interrupt_agent(self) -> None:
        """Send interrupt signal to agent."""
        if not self.rpyc_session:
            return

        try:
            response = await self.rpyc_session.async_call(
                self.rpyc_session.root.interrupt_agent
            )

            if response.get("success"):
                self._update_status("Interrupted")
            else:
                # Agent not running, ignore silently
                pass

        except Exception as e:
            # Log but don't show error (interrupt is best-effort)
            import sys
            print(f"[INTERRUPT] Failed to interrupt: {e}", file=sys.stderr)

    async def _handle_realtime_message(self, msg: dict[str, Any]) -> None:
        """Handle messages from agent."""
        msg_type = msg.get("type")

        # ... existing handling ...

        if msg_type == "agent_interrupted":
            # Agent was interrupted
            self._update_status("Interrupted")
            await self._set_spinner(None)  # Hide spinner

            # Show interruption message
            await self._mount_message(
                AppMessage("⚠ Interrupted")
            )
```

**Files Modified:**
- `deep_worker_cli/rpyc_service/agent_service.py` - Add `exposed_interrupt_agent()`
- `deep_worker_cli/process/agent_process.py` - Track running state, handle cancellation
- `deep_worker_cli/cli/interactive.py` - Add interrupt actions and handlers

**Multi-Client Behavior:**
- Any client can interrupt
- All clients receive "agent_interrupted" notification
- Agent stops execution immediately

---

### 3.6 Token Tracking

**Implementation:** Stream metadata from agent, display in status bar.

**Design:**

**Agent Process - Extract Token Count:**

**File:** `deep_worker_cli/process/stream_processor.py`

```python
async def process_message_chunk(
    message: AIMessage | ToolMessage | HumanMessage,
    metadata: dict[str, Any],
    notify_callback: Callable[[dict[str, Any]], None],
    tool_call_buffer: Optional[dict[str, Any]] = None,
) -> None:
    """Process message chunk and extract token usage."""

    # ... existing message processing ...

    # Extract token usage from AIMessage
    if isinstance(message, AIMessage):
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            total_tokens = (
                usage_metadata.get("input_tokens", 0) +
                usage_metadata.get("output_tokens", 0)
            )

            if total_tokens > 0:
                # Send token update
                notify_callback({
                    "type": "token_update",
                    "total_tokens": total_tokens,
                    "input_tokens": usage_metadata.get("input_tokens", 0),
                    "output_tokens": usage_metadata.get("output_tokens", 0),
                })
```

**Client-Side Token Display:**

**File:** `deep_worker_cli/cli/interactive.py`

```python
class InteractiveApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... existing ...
        self._current_token_count = 0

    async def _handle_realtime_message(self, msg: dict[str, Any]) -> None:
        """Handle messages from agent."""
        msg_type = msg.get("type")

        # ... existing handling ...

        if msg_type == "token_update":
            # Update token count
            total_tokens = msg.get("total_tokens", 0)
            self._current_token_count = total_tokens

            if self._status_bar:
                self._status_bar.set_tokens(total_tokens)
```

**Files Modified:**
- `deep_worker_cli/process/stream_processor.py` - Extract and send token metadata
- `deep_worker_cli/cli/interactive.py` - Receive and display tokens
- `deep_worker_cli/widgets/status.py` - **VERIFY** `set_tokens()` method exists

---

### 3.7 DiffWidget Display

**Implementation:** Render file diffs when agent performs file operations.

**Design:**

**Agent Process - Detect File Operations:**

**File:** `deep_worker_cli/process/stream_processor.py`

```python
async def process_message_chunk(
    message: AIMessage | ToolMessage | HumanMessage,
    metadata: dict[str, Any],
    notify_callback: Callable[[dict[str, Any]], None],
    tool_call_buffer: Optional[dict[str, Any]] = None,
) -> None:
    """Process message chunk and extract file diffs."""

    # ... existing processing ...

    # Check for file operation tool results
    if isinstance(message, ToolMessage):
        tool_name = metadata.get("tool_name", "")

        if tool_name in {"edit_file", "write_file"}:
            # Extract file path and diff from tool result
            content = message.content

            # Parse tool result for diff
            # (This depends on how deepagents formats tool results)
            if isinstance(content, str):
                # Check if content contains diff markers
                if "---" in content and "+++" in content:
                    # Send diff notification
                    notify_callback({
                        "type": "file_diff",
                        "tool_name": tool_name,
                        "tool_call_id": message.tool_call_id,
                        "diff_content": content,
                    })
```

**Client-Side Diff Display:**

**File:** `deep_worker_cli/cli/interactive.py`

```python
class InteractiveApp(App):
    async def _handle_realtime_message(self, msg: dict[str, Any]) -> None:
        """Handle messages from agent."""
        msg_type = msg.get("type")

        # ... existing handling ...

        if msg_type == "file_diff":
            # Display diff widget
            from deep_worker_cli.widgets.diff import DiffWidget

            diff_content = msg.get("diff_content", "")
            diff_widget = DiffWidget(diff_content)
            await self._mount_message(diff_widget)
```

**Files Modified:**
- `deep_worker_cli/process/stream_processor.py` - Detect file operations and extract diffs
- `deep_worker_cli/cli/interactive.py` - Display DiffWidget
- `deep_worker_cli/widgets/diff.py` - **VERIFY ALREADY EXISTS**

**Note:** This feature requires understanding how deepagents formats file operation results. May need adjustment based on actual tool output format.

---

## 4. Implementation Priorities

### Phase 1: Core Functionality (CRITICAL)
1. **Slash commands** (client-side + agent-side)
2. **Agent interruption** (Escape/Ctrl+C)
3. **Approval menu (HITL)** (full RPyC integration)

**Rationale:** These are essential for user control and safety.

**Estimated Effort:** 2-3 days

### Phase 2: UX Enhancements (HIGH)
4. **Bash commands** (! prefix)
5. **Auto-approve toggle**
6. **Token tracking**

**Rationale:** Improve usability and feature parity with deepagents.

**Estimated Effort:** 1-2 days

### Phase 3: Advanced Features (MEDIUM)
7. **DiffWidget display**

**Rationale:** Nice-to-have for code review workflows.

**Estimated Effort:** 1 day

---

## 5. Testing Strategy

### 5.1 Unit Tests

**New Test Files:**
- `tests/unit_tests/cli/test_slash_commands.py`
- `tests/unit_tests/cli/test_bash_commands.py`
- `tests/unit_tests/cli/test_approval_menu.py`
- `tests/unit_tests/cli/test_agent_interruption.py`
- `tests/unit_tests/rpyc_service/test_command_handling.py`
- `tests/unit_tests/rpyc_service/test_approval_flow.py`

### 5.2 Integration Tests

**Scenarios:**
1. **Slash commands** - Execute each command, verify agent state changes
2. **Bash commands** - Execute safe commands, verify output display
3. **Approval flow** - Request approval, verify menu display, send decision
4. **Multi-client approval** - Two clients, first to respond wins
5. **Agent interruption** - Start agent task, interrupt, verify cancellation
6. **Multi-client interrupt** - One client interrupts, all clients notified

### 5.3 Multi-Client Test Framework

**File:** `tests/integration_tests/test_multi_client.py`

```python
@pytest.mark.asyncio
async def test_approval_first_response_wins():
    """Test that first approval response wins with multiple clients."""
    # Start agent
    agent = start_agent("test-agent")

    # Connect two clients
    client1 = connect_client("test-agent")
    client2 = connect_client("test-agent")

    # Trigger approval (e.g., dangerous file operation)
    client1.send_message("Delete all files")

    # Both clients should receive approval request
    approval1 = await client1.wait_for_approval()
    approval2 = await client2.wait_for_approval()

    assert approval1.approval_id == approval2.approval_id

    # Client 1 approves first
    response1 = await client1.approve()
    assert response1["success"] is True

    # Client 2 tries to approve (should fail - already resolved)
    response2 = await client2.approve()
    assert response2["success"] is False
    assert "already resolved" in response2["message"].lower()

    # Both clients should see approval result
    result1 = await client1.wait_for_message(type="approval_result")
    result2 = await client2.wait_for_message(type="approval_result")

    assert result1 == result2
```

---

## 6. Files to Modify/Create

### 6.1 Core Files to Modify

| File | Changes | Lines Est. |
|------|---------|------------|
| `deep_worker_cli/cli/interactive.py` | Add command/bash/interrupt handlers, approval menu | +500 |
| `deep_worker_cli/rpyc_service/agent_service.py` | Add command/approval/interrupt RPC methods | +300 |
| `deep_worker_cli/process/agent_process.py` | Track state, handle interrupts, approval callback | +200 |
| `deep_worker_cli/process/stream_processor.py` | Extract tokens, diffs | +100 |
| `deep_worker_cli/widgets/chat_input.py` | Verify command/bash mode detection | ~50 |
| `deep_worker_cli/widgets/status.py` | Verify auto-approve/token methods exist | ~50 |

**Total Estimated Addition:** ~1200 lines

### 6.2 Widget Files to Verify

These should already exist (copied from deepagents):
- `deep_worker_cli/widgets/approval.py` - ApprovalMenu widget
- `deep_worker_cli/widgets/diff.py` - DiffWidget
- `deep_worker_cli/widgets/status.py` - StatusBar with auto-approve indicator

**Action:** Verify these have necessary methods, update if needed.

### 6.3 Test Files to Create

- `tests/unit_tests/cli/test_slash_commands.py` (~200 lines)
- `tests/unit_tests/cli/test_bash_commands.py` (~150 lines)
- `tests/unit_tests/cli/test_approval_menu.py` (~300 lines)
- `tests/unit_tests/cli/test_agent_interruption.py` (~200 lines)
- `tests/unit_tests/rpyc_service/test_command_handling.py` (~250 lines)
- `tests/integration_tests/test_multi_client.py` (~500 lines)

**Total Test Lines:** ~1600 lines

---

## 7. Architecture Diagrams

### 7.1 Approval Flow (Multi-Client)

```
┌─────────────┐                                  ┌─────────────┐
│  Client A   │                                  │  Client B   │
│ (dwcli TUI) │                                  │ (dwcli TUI) │
└──────┬──────┘                                  └──────┬──────┘
       │                                                │
       │  ① User sends message                         │
       │  "Delete all files"                           │
       │                                                │
       └────────────────┐                              │
                        ▼                              │
              ┌──────────────────┐                     │
              │  AgentProcess    │                     │
              │  (Server)        │                     │
              │                  │                     │
              │  ② Agent detects │                     │
              │     dangerous op │                     │
              │                  │                     │
              │  ③ Broadcasts    │                     │
              │     approval req │                     │
              └────────┬─────────┘                     │
                       │                               │
       ┌───────────────┴───────────────┐              │
       │  RPyC Callback (broadcast)    │              │
       ▼                               ▼              │
┌──────────────┐              ┌──────────────┐        │
│  Client A    │              │  Client B    │        │
│  ④ Shows     │              │  ④ Shows     │        │
│     Approval │              │     Approval │        │
│     Menu     │              │     Menu     │        │
└──────┬───────┘              └──────┬───────┘        │
       │                              │                │
       │ ⑤ User clicks                │ ⑤ User waits  │
       │   "Approve"                  │                │
       │                              │                │
       │  ⑥ RPC Call                  │                │
       │  respond_to_approval()       │                │
       └────────────────┐             │                │
                        ▼             │                │
              ┌──────────────────┐    │                │
              │  AgentProcess    │    │                │
              │                  │    │                │
              │  ⑦ Sets Future   │    │                │
              │     result       │    │                │
              │                  │    │                │
              │  ⑧ Broadcasts    │    │                │
              │     result       │    │                │
              └────────┬─────────┘    │                │
                       │              │                │
       ┌───────────────┴──────────────┴────┐          │
       │  RPyC Callback (broadcast)        │          │
       ▼                                   ▼          │
┌──────────────┐                    ┌──────────────┐  │
│  Client A    │                    │  Client B    │  │
│  ⑨ Hides     │                    │  ⑨ Hides     │  │
│     menu     │                    │     menu     │  │
│  Shows       │                    │  Shows       │  │
│  "✓Approved" │                    │  "✓Approved" │  │
└──────────────┘                    └──────────────┘  │
```

### 7.2 Slash Command Flow

```
Client-Side Commands             Agent-Side Commands
(/help, /version, /quit)         (/clear, /remember, /tokens, /threads)

┌─────────────────┐              ┌─────────────────────────────────────┐
│  InteractiveApp │              │  InteractiveApp                     │
│                 │              │                                     │
│  User types:    │              │  User types: /clear                │
│  /help          │              │                                     │
│                 │              │  ① Send via RPC                     │
│  ① Detect mode  │              │     handle_command()               │
│     "command"   │              └────────────┬────────────────────────┘
│                 │                           │
│  ② Handle local │                           ▼
│     _handle_    │              ┌─────────────────────────────────────┐
│     command()   │              │  AgentService                       │
│                 │              │                                     │
│  ③ Display      │              │  exposed_handle_command("/clear")  │
│     help text   │              │                                     │
│                 │              │  ② Reset thread_id                 │
│  No RPC call    │              │                                     │
└─────────────────┘              │  ③ Broadcast to ALL clients:       │
                                 │     {"type": "system",              │
                                 │      "action": "clear_thread"}     │
                                 └────────────┬────────────────────────┘
                                              │
                          ┌───────────────────┴────────────────┐
                          │  RPyC Callback (all clients)       │
                          ▼                                    ▼
                   ┌──────────────┐                     ┌──────────────┐
                   │  Client A    │                     │  Client B    │
                   │  ④ Clear UI  │                     │  ④ Clear UI  │
                   │  ⑤ Show msg  │                     │  ⑤ Show msg  │
                   └──────────────┘                     └──────────────┘
```

---

## 8. Risk Assessment

### 8.1 High-Risk Areas

| Risk | Mitigation |
|------|------------|
| **Race conditions in approval** | Use Future-based synchronization, first response wins |
| **Agent interrupt leaves stale state** | Proper cleanup in `finally` blocks, broadcast to clients |
| **RPyC connection loss during approval** | Timeout on approval requests, fallback to reject |
| **Command injection in bash commands** | User explicitly requested via `!` prefix, warn in docs |

### 8.2 Multi-Client Edge Cases

| Scenario | Behavior |
|----------|----------|
| Client A requests approval, Client B responds | ✓ Accepted - any client can respond |
| Client A requests approval, disconnects before response | ✓ Other clients can still respond |
| Two clients respond simultaneously | ✓ First response wins (Future already resolved) |
| Client A interrupts, Client B still streaming | ✓ All clients notified, streaming stops |
| Client A sends `/clear`, Client B sees partial history | ✓ All clients cleared via broadcast |

---

## 9. Documentation Updates

### 9.1 User-Facing Documentation

**File:** `deep_worker_cli/README.md` (or new `docs/INTERACTIVE_MODE.md`)

Topics to document:
- All slash commands and their behavior
- Bash command execution (! prefix)
- Approval menu interaction
- Auto-approve mode
- Agent interruption
- Multi-client behavior (important!)

### 9.2 Architecture Documentation

**File:** `deep_worker_cli/ARCHITECTURE.md`

Topics to document:
- Client-server separation
- RPyC message types
- Approval flow with multiple clients
- Interrupt handling
- Command routing (client-side vs agent-side)

---

## 10. Success Criteria

Implementation is complete when:

1. ✅ All slash commands work (7 commands: /help, /version, /quit, /clear, /remember, /tokens, /threads)
2. ✅ Bash commands execute and display output
3. ✅ Approval menu appears for dangerous operations
4. ✅ Auto-approve toggle works with status bar indicator
5. ✅ Agent interruption works (Escape/Ctrl+C)
6. ✅ Token count displays in status bar
7. ✅ File diffs display for edit/write operations
8. ✅ Multi-client tests pass (approval, interrupt, commands)
9. ✅ All new features have unit tests (>90% coverage)
10. ✅ Integration tests pass with 2+ concurrent clients

---

## 11. Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Core** | 3 days | Slash commands, interruption, approval menu |
| **Phase 2: UX** | 2 days | Bash commands, auto-approve, tokens |
| **Phase 3: Advanced** | 1 day | DiffWidget display |
| **Testing** | 2 days | Unit tests, integration tests, multi-client tests |
| **Documentation** | 1 day | User docs, architecture docs |
| **Total** | **9 days** | Full feature parity with deepagents |

---

## 12. Open Questions

1. **HITL Middleware Integration:** Does the agent already use `HITLMiddleware`? If not, needs to be added to agent graph.

2. **Tool Output Format:** What format does deepagents use for file operation results? Need to verify diff extraction logic.

3. **Approval Auto-Close:** Should approval menu auto-close if another client responds? Current design: yes (remove menu, show result message).

4. **Token Count Source:** Does agent stream include `usage_metadata` on AIMessage? If not, need alternative source (checkpoint metadata?).

5. **Shell Allow-List:** Should bash commands check `settings.shell_allow_list` or execute all (since user explicitly typed `!`)? Current design: execute all (user intent is clear).

---

## Appendix A: deepagents Feature Checklist

Comparing deepagents/app.py features to dwcli:

| Feature | deepagents | dwcli (current) | dwcli (after implementation) |
|---------|-----------|-----------------|------------------------------|
| WelcomeBanner | ✅ | ✅ | ✅ |
| ChatInput | ✅ | ✅ | ✅ |
| StatusBar | ✅ | ✅ | ✅ |
| LoadingWidget (spinner) | ✅ | ✅ | ✅ |
| Message widgets | ✅ | ✅ | ✅ |
| MessageStore | ✅ | ✅ | ✅ |
| ApprovalMenu | ✅ | ❌ | ✅ |
| DiffWidget | ✅ | ❌ | ✅ |
| Slash commands | ✅ (7) | ❌ | ✅ (7) |
| Bash commands (!) | ✅ | ❌ | ✅ |
| Auto-approve | ✅ | ❌ | ✅ |
| Agent interrupt | ✅ | ❌ | ✅ |
| Token tracking | ✅ | ❌ | ✅ |
| Thread management | ✅ | ❌ | ✅ |
| File @ mentions | ✅ | ✅ (verify) | ✅ |
| Autocomplete | ✅ | ✅ (verify) | ✅ |
| iTerm2 cursor guide fix | ✅ | ❌ | Not needed (client-only concern) |

---

## Appendix B: Key Code Locations

### deepagents (reference implementation)

```
deepagents_cli/
├── app.py                    # Main DeepAgentsApp (1468 lines)
├── textual_adapter.py        # TextualUIAdapter (streaming logic)
├── widgets/
│   ├── approval.py           # ApprovalMenu widget
│   ├── chat_input.py         # ChatInput with command detection
│   ├── diff.py               # DiffWidget
│   ├── messages.py           # All message widgets
│   ├── status.py             # StatusBar with auto-approve indicator
│   └── ...
```

### dwcli (to be modified)

```
deep_worker/deep_worker_cli/deep_worker_cli/
├── cli/
│   └── interactive.py        # InteractiveApp (1119 lines → ~1600 lines)
├── rpyc_service/
│   └── agent_service.py      # AgentService (add ~300 lines)
├── process/
│   ├── agent_process.py      # AgentProcess (add ~200 lines)
│   └── stream_processor.py   # Stream processing (add ~100 lines)
├── widgets/
│   ├── approval.py           # ✓ Already exists (verify)
│   ├── chat_input.py         # ✓ Already exists (verify command mode)
│   ├── diff.py               # ✓ Already exists (verify)
│   ├── status.py             # ✓ Already exists (verify methods)
│   └── ...
```

---

**END OF DESIGN DOCUMENT**
