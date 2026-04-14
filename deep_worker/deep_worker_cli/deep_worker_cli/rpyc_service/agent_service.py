"""RPyC service for deep-worker agent.

This module provides AgentService - an RPyC service that exposes agent
functionality via RPC calls and supports real-time message streaming via callbacks.

ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
============================================================
The agent deep-worker is the ONLY source of truth for ALL messages, including
user messages. ALL messages MUST be broadcast to ALL subscribers via callbacks.

User messages can originate from:
- dwcli instances
- Task manager
- Other clients

Regardless of origin, ALL messages go through agent and are broadcast to ALL
subscribers. This ensures every client sees the complete conversation state.

CRITICAL: Always notify subscribers about BOTH user and assistant messages!

SECURITY WARNING:
================
This service is designed for LOCAL TRUSTED ENVIRONMENT ONLY!

- NO authentication or authorization
- Allows pickle serialization (unsafe if exposed externally)
- Allows public attribute access
- Suitable ONLY for localhost (127.0.0.1) binding

DO NOT expose this service to external networks!
See SECURITY.md for full security requirements.
"""

from __future__ import annotations

# GUARANTEED LOG AT MODULE IMPORT - MUST APPEAR IF MODULE IS LOADED
import time as _log_time
with open("/tmp/dwcli_agent_service_IMPORT.log", "a") as _f:
    _f.write(f"\n[{_log_time.time():.3f}] agent_service.py MODULE IMPORTED\n")

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import rpyc

if TYPE_CHECKING:
    from deep_worker_cli.process.agent_process import AgentProcess

logger = logging.getLogger(__name__)

# Prompt for /remember command - triggers agent to review conversation and update
# memory/skills (copied from deepagents-cli for compatibility)
REMEMBER_PROMPT = """Review our conversation and capture valuable knowledge. Focus especially on **best practices** we discussed or discovered—these are the most important things to preserve.

## Step 1: Identify Best Practices and Key Learnings

Scan the conversation for:

### Best Practices (highest priority)
- **Patterns that worked well** - approaches, techniques, or solutions we found effective
- **Anti-patterns to avoid** - mistakes, gotchas, or approaches that caused problems
- **Quality standards** - criteria we established for good code, documentation, or processes
- **Decision rationale** - why we chose one approach over another

### Other Valuable Knowledge
- Coding conventions and style preferences
- Project architecture decisions
- Workflows and processes we developed
- Tools, libraries, or techniques worth remembering
- Feedback I gave about your behavior or outputs

## Step 2: Decide Where to Store Each Learning

For each best practice or learning, choose the right destination:

### → Memory (AGENTS.md) for preferences and guidelines
Use memory when the knowledge is:
- A preference or guideline (not a multi-step process)
- Something to always keep in mind
- A simple rule or pattern

**Global** (`~/.deepagents/agent/AGENTS.md`): Universal preferences across all projects
**Project** (`.deepagents/AGENTS.md`): Project-specific conventions and decisions

### → Skill for reusable workflows and methodologies
**Create a skill when** we developed:
- A multi-step process worth reusing
- A methodology for a specific type of task
- A workflow with best practices baked in
- A procedure that should be followed consistently

Skills are more powerful than memory entries because they can encode **how** to do something well, not just **what** to remember.

## Step 3: Create Skills for Significant Best Practices

If we established best practices around a workflow or process, capture them in a skill.

**Example:** If we discussed best practices for code review, create a `code-review` skill that encodes those practices into a reusable workflow.

### Skill Location
`~/.deepagents/agent/skills/<skill-name>/SKILL.md`

### Skill Structure
```
skill-name/
├── SKILL.md          (required - main instructions with best practices)
├── scripts/          (optional - executable code)
├── references/       (optional - detailed documentation)
└── assets/           (optional - templates, examples)
```

### SKILL.md Format
```markdown
---
name: skill-name
description: "What this skill does AND when to use it. Include triggers like 'when the user asks to X' or 'when working with Y'. This description determines when the skill activates."
---

# Skill Name

## Overview
Brief explanation of what this skill accomplishes.

## Best Practices
Capture the key best practices upfront:
- Best practice 1: explanation
- Best practice 2: explanation

## Process
Step-by-step instructions (imperative form):
1. First, do X
2. Then, do Y
3. Finally, do Z

## Common Pitfalls
- Pitfall to avoid and why
- Another anti-pattern we discovered
```

### Key Principles
1. **Encode best practices prominently** - Put them near the top so they guide the entire workflow
2. **Concise is key** - Only include non-obvious knowledge. Every paragraph should justify its token cost.
3. **Clear triggers** - The description determines when the skill activates. Be specific.
4. **Imperative form** - Write as commands: "Create a file" not "You should create a file"
5. **Include anti-patterns** - What NOT to do is often as valuable as what to do

## Step 4: Update Memory for Simpler Learnings

For preferences, guidelines, and simple rules that don't warrant a full skill:

```markdown
## Best Practices
- When doing X, always Y because Z
- Avoid A because it leads to B
```

Use `edit_file` to update existing files or `write_file` to create new ones.

## Step 5: Summarize Changes

List what you captured and where you stored it:
- Skills created (with key best practices encoded)
- Memory entries added (with location)
"""  # noqa: E501


class AgentService(rpyc.Service):
    """
    RPyC service for deep-worker agent.

    Exposes:
    - invoke_agent: Execute agent with message
    - get_history: Retrieve conversation history
    - get_status: Get agent status
    - subscribe_messages: Subscribe to real-time messages
    - unsubscribe_messages: Unsubscribe from messages
    """

    def on_connect(self, conn: Any) -> None:
        """Called when client connects.

        CRITICAL: This is a SYNC method called by RPyC!
        NEVER create asyncio.Lock() or other async primitives here.
        Each connection gets its own service instance, but message subscribers
        are shared via AgentProcess to ensure ALL connections receive messages.
        """
        # Initialize instance attributes (each connection = new instance)
        self._agent_process: AgentProcess | None = None
        self._event_loop: Any = None  # asyncio.AbstractEventLoop
        # NO _message_subscribers here! They're stored in AgentProcess (shared state)

        logger.info(f"RPyC client connected: {conn}")

    def on_disconnect(self, conn: Any) -> None:
        """Called when client disconnects.

        CRITICAL: This MUST be sync (not async) because RPyC calls it synchronously!
        If this is async, it will never be awaited and leave locks in bad state.
        """
        logger.info(f"RPyC client disconnected: {conn}")

        # NOTE: We don't clear _message_subscribers here anymore because they're
        # stored in AgentProcess (shared across all connections). Each subscriber
        # is responsible for unsubscribing via exposed_unsubscribe_messages().

    def set_agent_process(self, agent_process: AgentProcess) -> None:
        """Set reference to agent process."""
        self._agent_process = agent_process

    def set_event_loop(self, event_loop: Any) -> None:
        """Set reference to event loop."""
        self._event_loop = event_loop

    # ========================================================================
    # Override base RPyC Service methods as async for AsyncioServer
    # ========================================================================

    async def exposed_get_service_name(self) -> str:
        """Get service name (async override for AsyncioServer)."""
        return "AGENT_SERVICE"

    async def exposed_get_service_aliases(self) -> str:
        """Get service aliases (async override for AsyncioServer).

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.
        """
        return json.dumps([])

    # ========================================================================
    # AgentService exposed methods
    # ========================================================================

    async def exposed_get_status(self) -> str:
        """Get agent status (async for AsyncioServer).

        Returns minimal status response for high-frequency polling.
        Only includes status + optional error message.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response. This reduces RPyC messages from 24 to 2
        per get_status() call (see docs/RPYC_SERIALIZATION_DESIGN.md).

        This optimization reduces response size from 10-100 KB to < 200 bytes,
        eliminating unnecessary serialization overhead on every status check.
        """
        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        try:
            if not self._agent_process.session_manager:
                result = {"success": False, "error": "Session manager not initialized"}
                return json.dumps(result)

            from deep_worker_cli.basesession.session_adapter import map_agent_status

            # Get status directly from session manager (no full SessionInfo needed)
            status = map_agent_status(self._agent_process.session_manager.status)
            error_message = self._agent_process.session_manager.error_message

            result: dict[str, Any] = {
                "success": True,
                "status": status.value,  # SessionStatus enum value
                "error": error_message,  # Only if status == "error"
            }

            # Include task_manager fields only if TaskManager is configured
            if self._agent_process.task_manager is not None:
                result["task_manager_active"] = (
                    self._agent_process.task_manager.state != "inactive"
                )
                result["task_manager_disabled"] = self._agent_process.task_manager.disabled

            return json.dumps(result)
        except Exception as e:
            logger.exception("Error getting status: %s", e)
            result = {"success": False, "error": str(e)}
            return json.dumps(result)

    async def exposed_invoke_agent(
        self, message: str, stream: bool = False
    ) -> str:
        """
        Invoke agent with message (async for AsyncioServer!).

        MIGRATION NOTE: With AsyncioServer, we're already in the event loop!
        No need for run_coroutine_threadsafe - just await directly.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.

        Args:
            message: User message
            stream: Whether to stream response

        Returns:
            JSON string with success status and agent response
        """
        # DEBUG: Log entry
        import time as time_mod
        with open("/tmp/dwcli_rpyc_service.log", "a") as f:
            f.write(f"\n[{time_mod.time():.3f}] exposed_invoke_agent CALLED (ASYNC!)\n")
            f.write(f"  stream={stream}\n")
            f.write(f"  message={message[:80] if message else 'None'}...\n")

        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        try:
            # ====================================================================
            # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
            # ====================================================================
            # User message notification happens INSIDE _handle_invoke() via
            # notify_callback. This ensures messages are broadcast ONCE when
            # agent processes them, preventing duplication.
            #
            # DO NOT notify subscribers here! User message will be broadcast
            # by _handle_invoke() after agent adds it to state.
            # ====================================================================

            # ✨ ASYNC MAGIC: With AsyncioServer, we're already in the event loop!
            # Just await directly - no threading needed!
            # Pass _notify_subscribers as callback so the TaskManager evaluation
            # loop can broadcast ALL messages (user, assistant, tool_call, tool_result)
            # to all connected dwcli subscribers in real time.
            # Approval callback for HITL interrupts in non-streaming mode.
            # Bridges _handle_invoke's interrupt handling with the approval system.
            async def _approval_callback(interrupts: tuple) -> dict:
                """Handle interrupts by broadcasting and waiting for decisions."""
                hitl_response = {}
                for intr in interrupts:
                    actions = intr.value.get("action_requests", [])
                    decision = await self.exposed_request_approval(
                        action_requests=actions,
                        assistant_id=self._agent_process.name if self._agent_process else "unknown",
                    )
                    # Convert single decision to per-action decisions list
                    if decision.get("type") == "approve":
                        hitl_response[intr.id] = {
                            "decisions": [{"type": "approve"} for _ in actions]
                        }
                    elif decision.get("type") == "reject":
                        hitl_response[intr.id] = {
                            "decisions": [{"type": "reject", "message": decision.get("message", "Rejected by user")} for _ in actions]
                        }
                    elif decision.get("type") == "auto_approve_all":
                        hitl_response[intr.id] = {
                            "decisions": [{"type": "approve"} for _ in actions]
                        }
                    else:
                        # Default: approve
                        hitl_response[intr.id] = {
                            "decisions": [{"type": "approve"} for _ in actions]
                        }
                return hitl_response

            response = await self._agent_process._handle_invoke(
                {"command": "invoke", "message": message, "stream": stream},
                connection=None,
                notify_callback=self._notify_subscribers,
                approval_callback=_approval_callback,
            )

            # DEBUG: Log after getting response
            with open("/tmp/dwcli_rpyc_service.log", "a") as f:
                f.write(f"[{time_mod.time():.3f}] Got response (async await)\n")
                f.write(f"  success={response.get('success')}\n")

            return json.dumps(response)

        except asyncio.TimeoutError:
            logger.error("Agent invocation timed out")
            result = {"success": False, "error": "Agent invocation timed out"}
            return json.dumps(result)
        except Exception as e:
            logger.exception("Error invoking agent: %s", e)
            result = {"success": False, "error": str(e)}
            return json.dumps(result)

    async def exposed_get_history(
        self, limit: int = 50, offset: int = 0
    ) -> str:
        """Get conversation history (async for AsyncioServer!).

        MIGRATION NOTE: With AsyncioServer, we're already in the event loop!
        No need for run_coroutine_threadsafe - just await directly.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.
        """
        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        try:
            # ✨ ASYNC MAGIC: With AsyncioServer, we're already in the event loop!
            # Just await directly - no threading needed!
            response = await self._agent_process._handle_get_history(
                {"command": "get_history", "limit": limit, "offset": offset},
                connection=None,
            )
            return json.dumps(response)
        except Exception as e:
            logger.exception("Error getting history: %s", e)
            result = {"success": False, "error": str(e)}
            return json.dumps(result)

    async def exposed_get_last_response(self) -> str:
        """Get the last assistant response (async for AsyncioServer!).

        Returns only the last AIMessage with non-empty text content.
        More efficient than get_history when only the latest response is needed.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.
        """
        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        try:
            response = await self._agent_process._handle_get_last_response()
            return json.dumps(response)
        except Exception as e:
            logger.exception("Error getting last response: %s", e)
            result = {"success": False, "error": str(e)}
            return json.dumps(result)

    async def exposed_set_task_manager_enabled(self, enabled: bool) -> str:
        """Enable or disable TaskManager at runtime.

        Returns JSON string: {"success": true, "enabled": bool}
        """
        if not self._agent_process or not self._agent_process.task_manager:
            return json.dumps({"success": False, "error": "TaskManager not configured"})

        if enabled:
            self._agent_process.task_manager.enable()
        else:
            self._agent_process.task_manager.disable()

        return json.dumps({"success": True, "enabled": enabled})

    async def exposed_reset_task_manager(self) -> str:
        """Reset TaskManager to initial state.

        Clears evaluation history, reinitializes manager agent.
        Returns JSON string: {"success": true}
        """
        if not self._agent_process or not self._agent_process.task_manager:
            return json.dumps({"success": False, "error": "TaskManager not configured"})

        try:
            await self._agent_process.task_manager.reset()
            return json.dumps({"success": True, "message": "TaskManager reset to initial state"})
        except Exception as e:
            logger.exception("Error resetting TaskManager: %s", e)
            return json.dumps({"success": False, "error": str(e)})

    async def exposed_subscribe_messages(self, callback: Any) -> bool:
        """
        Subscribe to real-time messages.

        Args:
            callback: Callback object with on_message(msg: dict) method

        Returns:
            True if subscription successful
        """
        try:
            # Add to SHARED subscribers set in AgentProcess
            # This ensures ALL connections see messages from ANY connection
            if self._agent_process:
                self._agent_process._message_subscribers.add(callback)
                logger.info(f"Client subscribed to messages: {callback} (total subscribers: {len(self._agent_process._message_subscribers)})")
            else:
                logger.error("Cannot subscribe: agent_process not set")
                return False
            return True
        except Exception as e:
            logger.exception("Error subscribing: %s", e)
            return False

    async def exposed_unsubscribe_messages(self, callback: Any) -> bool:
        """
        Unsubscribe from messages.

        Args:
            callback: Callback object to remove

        Returns:
            True if unsubscription successful
        """
        try:
            # Remove from SHARED subscribers set in AgentProcess
            if self._agent_process:
                self._agent_process._message_subscribers.discard(callback)
                logger.info(f"Client unsubscribed from messages: {callback} (remaining: {len(self._agent_process._message_subscribers)})")
            else:
                logger.error("Cannot unsubscribe: agent_process not set")
                return False
            return True
        except Exception as e:
            logger.exception("Error unsubscribing: %s", e)
            return False

    async def exposed_get_metadata(self) -> str:
        """
        Get session metadata.

        Returns metadata including working_dir, pid, started_at, rpyc_port, model.
        Needed for DwSession compatibility.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.

        Returns:
            JSON string with metadata
        """
        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        try:
            import os

            # Get working_dir from config
            working_dir = (
                self._agent_process.config.working_dir
                if self._agent_process.config
                else None
            )

            metadata = {
                "directory": str(working_dir) if working_dir else None,
                "pid": os.getpid(),  # Current process PID
                "started": (
                    self._agent_process.started_at.isoformat()
                    if self._agent_process.started_at
                    else None
                ),
                "rpyc_port": self._agent_process.rpyc_port,
                "agent_type": "deep-worker",
                "model": (
                    self._agent_process.config.resolved_model
                    if self._agent_process.config
                    else None
                ),
                "provider": (
                    self._agent_process.config.provider
                    if self._agent_process.config
                    else None
                ),
            }

            result = {"success": True, "metadata": metadata}
            return json.dumps(result)

        except Exception as e:
            logger.exception("Error getting metadata: %s", e)
            result = {"success": False, "error": str(e)}
            return json.dumps(result)

    async def exposed_handle_command(self, command: str) -> str:
        """
        Handle agent-side slash commands.

        Commands processed:
        - /clear - Reset thread to new ID, broadcast to all clients
        - /remember [context] - Send remember prompt to agent
        - /tokens - Display current token count
        - /threads - Display current thread ID

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.

        Args:
            command: Full command string (e.g., "/clear", "/remember foo", etc.)

        Returns:
            JSON string with:
            {
                "success": bool,
                "response": str,  # Response message to display
                "action": str,    # Action taken: "clear_thread", "display_info", "send_to_agent"
                "data": dict      # Optional additional data
            }
        """
        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        cmd = command.lower().strip()

        try:
            if cmd == "/clear":
                # Reset thread to new ID
                import uuid

                old_thread_id = getattr(self._agent_process, 'current_thread_id', None)
                new_thread_id = uuid.uuid4().hex[:8]
                self._agent_process.current_thread_id = new_thread_id

                # Broadcast to ALL clients about thread reset
                # This ensures all connected clients clear their UI
                await self._notify_subscribers({
                    "type": "system",
                    "role": "system",
                    "content": f"Started new thread: {new_thread_id}",
                    "action": "clear_thread",
                    "old_thread_id": old_thread_id,
                    "new_thread_id": new_thread_id,
                })

                result = {
                    "success": True,
                    "response": f"Started new thread: {new_thread_id}",
                    "action": "clear_thread",
                    "data": {
                        "old_thread_id": old_thread_id,
                        "new_thread_id": new_thread_id,
                    }
                }
                return json.dumps(result)

            elif cmd == "/threads":
                # Display current thread ID
                thread_id = getattr(self._agent_process, 'current_thread_id', 'unknown')
                result = {
                    "success": True,
                    "response": f"Current thread: {thread_id}",
                    "action": "display_info",
                    "data": {"thread_id": thread_id}
                }
                return json.dumps(result)

            elif cmd == "/tokens":
                # Get token count from last response metadata
                token_count = getattr(self._agent_process, 'last_token_count', 0) or 0
                if token_count >= 1000:
                    formatted = f"{token_count / 1000:.1f}K"
                else:
                    formatted = str(token_count)

                result = {
                    "success": True,
                    "response": f"Current context: {formatted} tokens",
                    "action": "display_info",
                    "data": {"token_count": token_count}
                }
                return json.dumps(result)

            elif cmd == "/remember" or cmd.startswith("/remember "):
                # Extract additional context after /remember
                additional_context = ""
                if cmd.startswith("/remember "):
                    # Get original case from command (not lowercased cmd)
                    additional_context = command.strip()[len("/remember "):].strip()

                # Build remember prompt
                remember_prompt = REMEMBER_PROMPT
                if additional_context:
                    remember_prompt += f"\n\n**Additional context from user:** {additional_context}"

                # Return action to send this as user message to agent
                result = {
                    "success": True,
                    "response": "",  # No immediate response, agent will process
                    "action": "send_to_agent",
                    "data": {"message": remember_prompt}
                }
                return json.dumps(result)

            else:
                # Unknown command
                result = {
                    "success": False,
                    "response": f"Unknown command: {cmd}",
                    "action": "error",
                }
                return json.dumps(result)

        except Exception as e:
            logger.exception("Error handling command: %s", e)
            result = {
                "success": False,
                "error": str(e),
                "action": "error"
            }
            return json.dumps(result)

    async def exposed_respond_to_approval(
        self,
        approval_id: str,
        decision: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Receive approval decision from client (HITL support).

        This method resolves pending approval futures created by
        exposed_request_approval(), enabling blocking approval workflow.

        Args:
            approval_id: Unique ID for the approval request
            decision: Decision dict with "type" field ("approve", "reject", etc.)

        Returns:
            {"success": bool, "message": str}
        """
        if not self._agent_process:
            return {"success": False, "message": "Agent not initialized"}

        # Check if there's a pending approval future for this ID
        if hasattr(self, '_pending_approvals') and approval_id in self._pending_approvals:
            future = self._pending_approvals[approval_id]

            if future.done():
                return {
                    "success": False,
                    "message": "Approval already resolved by another client"
                }

            # Resolve the future with the decision
            future.set_result(decision)

            # Broadcast result to all clients
            await self._notify_subscribers({
                "type": "approval_result",
                "approval_id": approval_id,
                "decision": decision,
            })

            return {"success": True, "message": "Decision received and applied"}

        # Fallback: Store approval decisions for simplified HITL implementation
        # (for backward compatibility with old code paths)
        # Note: _approval_decisions is now a properly typed attribute in AgentProcess
        approval_decisions = self._agent_process._approval_decisions

        # Check if already resolved
        if approval_id in approval_decisions:
            return {
                "success": False,
                "message": "Approval already resolved by another client"
            }

        # Store decision
        approval_decisions[approval_id] = decision

        # Broadcast result to all clients
        await self._notify_subscribers({
            "type": "approval_result",
            "approval_id": approval_id,
            "decision": decision,
        })

        return {"success": True, "message": "Decision received"}

    async def exposed_interrupt_agent(self) -> str:
        """
        Interrupt currently running agent execution.

        Cancels the current agent worker task if running, broadcasting
        an "agent_interrupted" message to all connected clients.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.

        Returns:
            JSON string with:
            {
                "success": bool,
                "message": str  # Status message
            }
        """
        if not self._agent_process:
            result = {"success": False, "message": "Agent not initialized"}
            return json.dumps(result)

        # Check if agent is currently executing
        is_running = getattr(self._agent_process, 'is_running', False)
        current_task = getattr(self._agent_process, 'current_worker_task', None)

        if not is_running or current_task is None:
            result = {
                "success": False,
                "message": "No agent execution in progress"
            }
            return json.dumps(result)

        # Cancel the worker task
        try:
            current_task.cancel()

            # Broadcast interrupt to ALL clients
            # This ensures all connected clients see the interruption
            await self._notify_subscribers({
                "type": "agent_interrupted",
                "role": "system",
                "content": "Agent interrupted by user",
            })

            result = {
                "success": True,
                "message": "Agent interrupted"
            }
            return json.dumps(result)

        except Exception as e:
            logger.exception("Error interrupting agent: %s", e)
            result = {
                "success": False,
                "message": f"Error interrupting: {e}"
            }
            return json.dumps(result)

    async def exposed_shutdown(self) -> str:
        """
        Request graceful shutdown of agent process.

        Returns immediately with success status. Actual shutdown happens asynchronously.
        The RPyC server will be closed after a short delay to allow the response to be sent.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response.

        Returns:
            JSON string with success status
        """
        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        try:
            # Signal agent to shutdown gracefully
            self._agent_process.running = False

            # Close RPyC server after response is sent
            async def delayed_close_async() -> None:
                """Async wrapper for closing RPyC server."""
                import time

                time.sleep(0.5)  # Give time for response to be sent
                if self._agent_process and self._agent_process.rpyc_server:
                    try:
                        await self._agent_process.rpyc_server.close()
                    except Exception:
                        pass

            # Schedule the async close in the event loop
            asyncio.create_task(delayed_close_async())

            result = {"success": True, "message": "Shutdown initiated"}
            return json.dumps(result)

        except Exception as e:
            logger.exception("Error during shutdown: %s", e)
            result = {"success": False, "error": str(e)}
            return json.dumps(result)

    # ========================================================================
    # NEW METHODS - Renamed to match DeepAgents naming conventions
    # ========================================================================

    async def exposed_run_agent_task(self, message: str) -> None:
        """Execute agent with user message (matching DeepAgents name).

        This method replaces exposed_invoke_agent() to match the naming
        convention from the original DeepAgents codebase.

        Args:
            message: User message to send to agent

        Note:
            This method returns immediately (fire-and-forget). Results are
            streamed to subscribers via real-time callbacks.
        """
        # Delegate to existing invoke implementation
        await self.exposed_invoke_agent(message, stream=True)

    async def exposed_load_thread_history(
        self, limit: int = 50, offset: int = 0
    ) -> str:
        """Load conversation history (matching DeepAgents name).

        This method replaces exposed_get_history() to match the naming
        convention from the original DeepAgents codebase.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref overhead.
        Client must json.loads() the response (see docs/RPYC_SERIALIZATION_DESIGN.md).

        Args:
            limit: Maximum number of messages to retrieve (default: 50)
            offset: Number of recent messages to skip for pagination (default: 0)

        Returns:
            JSON string with format:
            {
                "success": bool,
                "history": [list of message dicts],
                "total": total message count,
                "has_more": bool indicating if more messages exist
            }
        """
        # Delegate to existing get_history implementation
        return await self.exposed_get_history(limit, offset)

    async def exposed_set_spinner(self, status: str | None) -> None:
        """Set spinner status on all connected clients (matching DeepAgents name).

        This method enables explicit UI control via rpyc_async callbacks,
        matching the TextualUIAdapter.set_spinner() pattern from DeepAgents.

        Args:
            status: Spinner status text (e.g., "Thinking...") or None to hide

        Example:
            await service.exposed_set_spinner("Thinking...")  # Show
            await service.exposed_set_spinner(None)  # Hide
        """
        await self._notify_subscribers({
            "type": "spinner",
            "status": status
        })

    async def exposed_request_approval(
        self,
        action_requests: list[dict],
        assistant_id: str
    ) -> dict[str, Any]:
        """Request user approval and WAIT for decision (matching DeepAgents name).

        This method broadcasts an approval request to all connected clients
        and blocks until a decision is received, matching the
        TextualUIAdapter.request_approval() pattern from DeepAgents.

        Args:
            action_requests: List of actions requiring approval
            assistant_id: ID of the assistant requesting approval

        Returns:
            Decision dict from user (e.g., {"type": "approve", "action": "all"})

        Note:
            This is a BLOCKING call - the agent will wait for user decision.
        """
        import uuid

        approval_id = str(uuid.uuid4())

        # Create future to wait for decision
        future: asyncio.Future = asyncio.Future()

        # Initialize _pending_approvals if not exists
        if not hasattr(self, '_pending_approvals'):
            self._pending_approvals: dict[str, asyncio.Future] = {}

        self._pending_approvals[approval_id] = future

        # Broadcast approval request to all clients
        await self._notify_subscribers({
            "type": "approval_request",
            "approval_id": approval_id,
            "actions": action_requests,
            "assistant_id": assistant_id
        })

        # Wait for decision from client
        try:
            decision = await future
            return decision
        finally:
            # Clean up pending approval
            if approval_id in self._pending_approvals:
                del self._pending_approvals[approval_id]

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    async def _notify_subscribers(self, message: dict[str, Any]) -> None:
        """
        Notify all subscribers about a new message (async for AsyncioServer!).

        MIGRATION NOTE: With AsyncioServer and async callbacks, this is now
        truly async and non-blocking! Much faster than old ThreadedServer approach.

        Combined with text chunk batching (stream_processor.py), this reduces
        overhead from ~hundreds of RPyC calls to ~10-20 calls per response.

        Args:
            message: Message dict to send to subscribers
        """
        # DEBUG: Log notification with timing
        import time as time_mod
        import os
        debug_enabled = os.environ.get("DWCLI_DEBUG_CALLBACKS") == "1"

        # ALWAYS log timing for investigating latency
        start_time = time_mod.time()
        with open("/tmp/dwcli_rpyc_timing.log", "a") as f:
            f.write(f"\n[{start_time:.3f}] _notify_subscribers CALLED\n")
            f.write(f"  msg_type={message.get('type')}\n")
            f.write(f"  content_len={len(str(message.get('content', '')))}\n")

        if debug_enabled:
            import traceback
            content = message.get("content", "")[:50]
            with open("/tmp/dwcli_callback_debug.log", "a") as f:
                f.write(f"\n[AGENT_DEBUG] _notify_subscribers called:\n")
                f.write(f"  message: role={message.get('role')}, content={content}...\n")
                f.write(f"  Stack trace:\n")
                for line in traceback.format_stack()[:-1]:
                    f.write(f"    {line.strip()}\n")

        # Get subscribers from SHARED set in AgentProcess
        # This ensures ALL connections (TUI, headless, web UI) receive messages
        if not self._agent_process:
            logger.error("Cannot notify subscribers: agent_process not set")
            return

        subscribers = list(self._agent_process._message_subscribers)

        if debug_enabled:
            with open("/tmp/dwcli_callback_debug.log", "a") as f:
                f.write(f"  subscribers count: {len(subscribers)}\n")

        # ========================================================================
        # CRITICAL: RPyC Fire-and-Forget Callbacks Using rpyc.async_()
        # ========================================================================
        # PROBLEM: When dwcli calls invoke_agent() via RPyC async_call, it waits for
        # the response. If invoke_agent() tries to call subscriber.on_message()
        # synchronously (reentrant call: client->server->client), this creates a
        # DEADLOCK because:
        #   - Client is waiting for server response (async_call blocks)
        #   - Server tries to call client callback synchronously
        #   - Client cannot process callback while waiting for response
        #   - Result: "EOFError: stream has been closed"
        #
        # threading.Thread() does NOT fix this - RPyC netref objects are NOT thread-safe
        # and calling them from background threads still causes "stream has been closed"!
        #
        # SOLUTION: Use rpyc.async_() to wrap the callback and invoke WITHOUT waiting
        # for the result. This is RPyC's built-in fire-and-forget mechanism:
        #   - rpyc.async_(fn) returns an async-wrapped proxy
        #   - Calling it returns AsyncResult immediately
        #   - We DON'T wait for the AsyncResult - true fire-and-forget
        #   - Client's BgServingThread handles the callback asynchronously
        #
        # This is the CORRECT way to do server-to-client callbacks in RPyC!
        # Reference: RPyC docs on async_() and AsyncResult
        # ========================================================================

        for subscriber in subscribers:
            try:
                rpyc_start = time_mod.time()

                # ✨ ASYNC CALLBACK: Match deepagents architecture!
                # In deepagents, all callbacks (mount_message, etc.) are called with await.
                # We preserve this async pattern across the RPyC boundary.
                # With AsyncioServer, await works correctly on proxy objects!
                await subscriber.on_message(message)

                rpyc_end = time_mod.time()
                rpyc_duration_ms = (rpyc_end - rpyc_start) * 1000
                with open("/tmp/dwcli_rpyc_timing.log", "a") as f:
                    f.write(f"  RPyC ASYNC callback took {rpyc_duration_ms:.1f}ms\n")
            except Exception as e:
                # Use repr() to avoid triggering RPyC netref attribute access
                logger.error(f"Error dispatching ASYNC callback to subscriber (id={id(subscriber)}): {e}")
