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


def evict_subscriber_by_identity(subscriber_set: set[Any], target: Any) -> None:
    """Remove ``target`` from ``subscriber_set`` WITHOUT hashing/comparing it.

    Subscribers are RPyC **netrefs**. Once the peer's channel dies, ANY
    operation that crosses the wire raises ``EOFError('stream has been
    closed')`` — and that includes ``netref.__hash__`` and ``netref.__eq__``
    (both do a synchronous ``syncreq`` over the channel). So the obvious
    ``subscriber_set.discard(dead_netref)`` BLOWS UP: ``set.discard`` needs
    ``hash(dead_netref)`` → wire round-trip → ``EOFError`` again. In
    ``_notify_subscribers`` that secondary EOFError used to escape into
    ``_handle_invoke`` and force the agent to ERROR over a dead UI socket
    (a related internal incident).

    The ONLY safe way to evict a dead netref is by **object identity**: we
    iterate the set (iteration never hashes elements) and rebuild it,
    dropping the one element that ``is`` the target. ``is`` never crosses the
    wire. Do NOT replace this with ``discard`` / ``remove`` / ``in`` — those
    all call ``__hash__``/``__eq__`` and will re-raise on a dead channel.

    Re-adding survivors DOES hash them, and a second simultaneously-dead
    subscriber would raise there too — so we add one at a time and skip any
    that explode (they're dead and will be evicted on their own failed push).
    The whole helper is best-effort and never raises.
    """
    survivors = [s for s in subscriber_set if s is not target]
    subscriber_set.clear()
    for s in survivors:
        try:
            subscriber_set.add(s)
        except Exception:  # noqa: BLE001 — another dead netref; drop it too.
            pass

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

            # ════════════════════════════════════════════════════════════
            # DECOUPLE THE INVOKE FROM THIS RPyC DISPATCH TASK
            # ════════════════════════════════════════════════════════════
            # exposed_invoke_agent runs INSIDE the dispatch task of the RPyC
            # connection that delivered it. rpyc_async's inbound-quarantine
            # (_drain_inbound_dispatch) cancels every dispatch task on an
            # overloaded connection — and a flood can come from a DIFFERENT
            # client (the web UI), not the one that sent this invoke. If we
            # awaited _handle_invoke inline, that cancellation would kill the
            # running agent turn and it would be misread as a user interrupt
            # → spurious INTERRUPTED (a related internal incident).
            #
            # So run _handle_invoke in a DEDICATED task and await it via
            # asyncio.shield. If THIS dispatch task is cancelled (quarantine,
            # client disconnect), the shield re-raises CancelledError here but
            # the invoke task keeps running and finishes the turn; the agent's
            # status is driven by _handle_invoke via session_manager, which
            # every live channel + the web UI read. A genuine user interrupt
            # goes through exposed_interrupt_agent → current_worker_task.cancel
            # (the dedicated task), which IS surfaced through the shield as the
            # clean "interrupted" result below.
            invoke_task = asyncio.ensure_future(
                self._agent_process._handle_invoke(
                    {"command": "invoke", "message": message, "stream": stream},
                    connection=None,
                    notify_callback=self._notify_subscribers,
                    approval_callback=_approval_callback,
                )
            )
            try:
                response = await asyncio.shield(invoke_task)
            except asyncio.CancelledError:
                # Distinguish the two cancellation sources:
                #  • invoke_task itself was cancelled  → real user interrupt
                #    (exposed_interrupt_agent). Re-raise so the outer handler
                #    returns the clean "interrupted" result.
                #  • only THIS awaiter was cancelled (dispatch-task drain /
                #    quarantine) → the invoke is STILL running. We must NOT
                #    cancel it and NOT touch agent status. The agent keeps
                #    working and settles its own status via session_manager.
                if invoke_task.cancelled():
                    raise  # real interrupt → outer CancelledError handler
                with open("/tmp/dwcli_rpyc_service.log", "a") as f:
                    f.write(
                        f"[{time_mod.time():.3f}] dispatch task cancelled but "
                        f"invoke detached and still running (channel drain/"
                        f"quarantine) — agent keeps working\n"
                    )
                # Pin the detached task on the agent process so it is not
                # garbage-collected before it finishes (it is no longer
                # awaited by anything).
                self._agent_process._detached_invoke_tasks.add(invoke_task)
                invoke_task.add_done_callback(
                    self._agent_process._detached_invoke_tasks.discard
                )
                # Honor the cancellation of THIS dispatch task by re-raising.
                # The outer CancelledError handler checks invoke_task.cancelled()
                # to tell this (channel drain — NOT a user interrupt) apart from
                # a real interrupt, and unwinds without reporting an interrupt.
                raise

            # DEBUG: Log after getting response
            with open("/tmp/dwcli_rpyc_service.log", "a") as f:
                f.write(f"[{time_mod.time():.3f}] Got response (async await)\n")
                f.write(f"  success={response.get('success')}\n")

            return json.dumps(response)

        except asyncio.CancelledError:
            # Two ways we can land here (see the shield block above):
            #
            # 1. CHANNEL DRAIN (NOT a user interrupt): only this dispatch
            #    task was cancelled (rpyc inbound-quarantine /
            #    _drain_inbound_dispatch / client disconnect) while the
            #    invoke task keeps running. ``invoke_task`` is alive (not
            #    cancelled). We must NOT report an interrupt and must NOT
            #    touch agent status — the agent keeps working. Just unwind
            #    this dispatch task by re-raising; the response goes nowhere
            #    (the channel is gone) and the detached invoke settles its
            #    own status via session_manager.
            #
            # 2. USER INTERRUPT: ``invoke_task`` itself was cancelled
            #    (exposed_interrupt_agent → current_worker_task.cancel).
            #    ``_handle_invoke`` already transitioned to INTERRUPTED and
            #    broadcast ``agent_interrupted``. Return a clean structured
            #    *interrupted* result instead of letting CancelledError cross
            #    the RPyC boundary as a remote exception (which the web side
            #    would mis-render as an error — user report 2026-05-20).
            _invoke_task = locals().get("invoke_task")
            if _invoke_task is not None and not _invoke_task.cancelled():
                # Channel drain — agent keeps working, no interrupt reported.
                raise
            with open("/tmp/dwcli_rpyc_service.log", "a") as f:
                f.write(f"[{time_mod.time():.3f}] invoke interrupted (clean)\n")
            result = {
                "success": False,
                "interrupted": True,
                "message": "Agent invocation interrupted by user",
            }
            return json.dumps(result)
        except asyncio.TimeoutError:
            logger.error("Agent invocation timed out")
            result = {"success": False, "error": "Agent invocation timed out"}
            return json.dumps(result)
        except Exception as e:
            logger.exception("Error invoking agent: %s", e)
            result = {"success": False, "error": str(e)}
            return json.dumps(result)

    async def exposed_enqueue_message(self, message: str, reply_back: bool = True) -> str:
        """Fire-and-forget delivery: enqueue ``message`` as an incoming
        user-role message and return immediately, without waiting for the
        agent's LLM turn to complete.

        Used by the inter-agent ``send_message_to_agent`` tool so that the
        sending agent's turn can end promptly (its SendMessageCompletionMiddleware
        injects the echo AIMessage and jumps to the graph end) instead of
        blocking until the recipient finishes processing.

        Returns a JSON string ``{"success": true, "delivered_content": "<message>"}``
        on successful enqueue, or ``{"success": false, "error": "..."}`` on
        failure. The background turn's result is NOT awaited here.
        """
        if not self._agent_process:
            return json.dumps({"success": False, "error": "Agent not initialized"})

        # Pin the non-None reference locally. Narrowing of `self._agent_process`
        # does NOT survive into the nested async function below — ty re-reads
        # the attribute and sees `AgentProcess | None`.
        agent_process = self._agent_process

        try:
            # Fire off the agent turn in the background. Exceptions inside
            # the task are logged but don't affect the enqueue response.
            async def _bg_turn() -> None:
                try:
                    await agent_process._handle_invoke(
                        {
                            "command": "invoke",
                            "message": message,
                            "stream": True,
                            "reply_back": reply_back,
                            "inter_agent_delivery": True,
                        },
                        connection=None,
                        notify_callback=self._notify_subscribers,
                        approval_callback=None,
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "Background turn failed for enqueued message"
                    )

            asyncio.create_task(_bg_turn())

            return json.dumps({
                "success": True,
                "delivered_content": message,
            })
        except Exception as e:
            logger.exception("Error enqueueing message: %s", e)
            return json.dumps({"success": False, "error": str(e)})

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

    async def exposed_get_system_prompt(self) -> str:
        """Get the invariant (startup) system prompt (async for AsyncioServer).

        Backs Full Dialog verbosity mode ``system_prompt`` (§9 of
        the Full Dialog design document on final response pairs).

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref
        overhead. Client must json.loads() the response.
        """
        if not self._agent_process:
            return json.dumps({"success": False, "error": "Agent not initialized"})
        try:
            # NOTE: dispatched via getattr to match the established loose-access
            # style used throughout this file (see ``getattr(self._agent_process,
            # ...)`` at lines 817/880/908/...). The ty 0.0.32 checker fails to
            # surface freshly-added AgentProcess methods cross-module (verified:
            # runtime + single-file ``ty check agent_process.py`` both see
            # ``_handle_get_system_prompt``); getattr keeps the dispatch
            # type-clean without an ignore.
            handler = getattr(self._agent_process, "_handle_get_system_prompt")
            response = await handler()
            return json.dumps(response)
        except Exception as e:
            logger.exception("Error getting system prompt: %s", e)
            return json.dumps({"success": False, "error": str(e)})

    async def exposed_get_tools(self) -> str:
        """Get the agent's available tool list (async for AsyncioServer).

        Backs Full Dialog verbosity mode ``tools`` (§9). The list may depend
        on the active provider.

        IMPORTANT: Returns JSON-serialized string to eliminate RPyC netref
        overhead. Client must json.loads() the response.
        """
        if not self._agent_process:
            return json.dumps({"success": False, "error": "Agent not initialized"})
        try:
            # See the getattr rationale in exposed_get_system_prompt above
            # (ty 0.0.32 cross-module new-member resolution bug).
            handler = getattr(self._agent_process, "_handle_get_tools")
            response = await handler()
            return json.dumps(response)
        except Exception as e:
            logger.exception("Error getting tools: %s", e)
            return json.dumps({"success": False, "error": str(e)})

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

        Immediately after registration we push a one-shot
        ``status_change`` snapshot of the agent's current state to
        this subscriber. That is what turns the event stream into a
        self-contained feed: a UI client that reconnects mid-session
        does NOT have to poll to learn the current status / TM flags
        — it receives them as the first message.

        Args:
            callback: Callback object with on_message(msg: str) method

        Returns:
            True if subscription successful
        """
        try:
            if not self._agent_process:
                logger.error("Cannot subscribe: agent_process not set")
                return False

            # Add to SHARED subscribers set in AgentProcess
            # This ensures ALL connections see messages from ANY connection
            self._agent_process._message_subscribers.add(callback)
            logger.info(
                "Client subscribed to messages (total subscribers: %d)",
                len(self._agent_process._message_subscribers),
            )

            # Push the current-state snapshot, BUT from OUTSIDE the
            # current RPC dispatch. We are still inside the
            # ``subscribe_messages`` RPC handler — the client has not
            # yet received our ``True`` reply. ``loop.call_later(0,
            # ...)`` schedules delivery on the next loop tick so this
            # handler returns first; by then the subscribe RPC has
            # flushed its reply and ``await on_message(...)`` is not a
            # re-entrant sync_request from inside the same dispatch.
            #
            # A plain ``await on_message(snapshot_str)`` on an
            # async-flagged rpyc_async netref routes through
            # ``HANDLE_ASYNC_CALL`` on its own, so the peer's
            # ``_dispatch_request_async`` awaits ``exposed_on_message``
            # correctly. Do NOT wrap with ``rpyc.async_`` — that forces
            # ``HANDLE_CALL`` (sync) and the peer's async handler is
            # left as an un-awaited coroutine ("Boxing coroutine object
            # ... Did you forget to await?"), silently dropping every
            # status_change on the floor.
            snapshot = self._agent_process._current_status_snapshot()
            snapshot_str = json.dumps(snapshot, default=str)

            async def _deliver_snapshot() -> None:
                try:
                    # Use asyncreq(HANDLE_CALLATTR) to avoid sync_request
                    # guard (same pattern as _dispatch_status_event in
                    # agent_process.py, issue #48).
                    from rpyc.core.netref import asyncreq as _asyncreq  # noqa: PLC0415
                    ar = _asyncreq(
                        callback,
                        8,  # HANDLE_CALLATTR
                        "on_message",
                        (snapshot_str,),
                        (),  # kwargs
                    )
                    await ar
                except Exception as snap_exc:  # noqa: BLE001
                    logger.warning(
                        "initial status snapshot push failed: %s", snap_exc
                    )

            loop = asyncio.get_running_loop()
            loop.call_later(
                0,
                lambda: asyncio.ensure_future(_deliver_snapshot()),
            )

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
                # Evict by identity (no __hash__ on the netref). Graceful
                # unsubscribe normally happens on a live channel, but a client
                # racing disconnect+unsubscribe could present a dead netref,
                # and ``discard`` would then re-raise EOFError (a related
                # internal incident). Identity eviction is always safe.
                evict_subscriber_by_identity(
                    self._agent_process._message_subscribers, callback
                )
                logger.info(
                    "Client unsubscribed from messages (remaining: %d)",
                    len(self._agent_process._message_subscribers),
                )
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

            config = self._agent_process.config
            model = None
            provider = None
            if config:
                provider = config.provider
                dynamic_model = getattr(config, "dynamic_display_model", None)
                model = config.model or dynamic_model or config.resolved_model

            status_metadata = None
            session_manager = getattr(self._agent_process, "session_manager", None)
            if session_manager is not None:
                status_metadata = getattr(session_manager, "status_metadata", None)

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
                "model": model,
                "provider": provider,
                "status_metadata": status_metadata,
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

    async def exposed_shutdown(
        self,
        grace_seconds: float = 5.0,
        force: bool = False,
    ) -> str:
        """Request shutdown. Returns immediately; real shutdown runs async.

        The actual exit pipeline is event-driven:

        * ``force=True`` — immediately schedule ``os._exit(0)`` via
          ``loop.call_later(0.1, ...)``. The 100 ms delay lets the RPyC
          response flush before the process vanishes.
        * otherwise — flip ``agent_process.running = False`` (cooperative
          stop), try to close the RPyC server, and arm a SIGKILL fallback
          at ``grace_seconds``. The fallback uses ``os._exit(0)`` so we
          stay inside this process's event loop — no external signals,
          no new threads.

        No ``time.sleep``, no ``asyncio.sleep`` loop. Returns JSON string.
        """
        if not self._agent_process:
            result = {"success": False, "error": "Agent not initialized"}
            return json.dumps(result)

        try:
            import os as _os

            loop = asyncio.get_running_loop()
            grace = max(0.0, float(grace_seconds))

            if force:
                # Panic exit — 0.1 s is just enough for the RPyC frame to
                # flush to the wire before we vanish.
                loop.call_later(0.1, lambda: _os._exit(0))
                logger.warning(
                    "exposed_shutdown(force=True): process will exit in 100ms"
                )
                result = {
                    "success": True,
                    "message": "Shutdown initiated (forced, immediate)",
                    "grace_seconds": 0.0,
                    "force": True,
                }
                return json.dumps(result)

            # Graceful: flip the cooperative flag, close RPyC, arm SIGKILL.
            self._agent_process.running = False

            async def graceful_close() -> None:
                """Close the RPyC server after a short flush window."""
                # 50 ms is enough for the JSON response to leave our socket.
                await asyncio.sleep(0.05)
                srv = getattr(self._agent_process, "rpyc_server", None)
                if srv is not None:
                    try:
                        await srv.close()
                    except Exception:
                        logger.debug(
                            "exposed_shutdown: rpyc_server.close() raised",
                            exc_info=True,
                        )

            asyncio.create_task(graceful_close())

            # Hard-kill fallback. If cooperative shutdown worked the
            # process is already gone; otherwise this guarantees exit
            # after grace_seconds without external signals.
            loop.call_later(grace, lambda: _os._exit(0))

            result = {
                "success": True,
                "message": "Shutdown initiated",
                "grace_seconds": grace,
                "force": False,
            }
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

        # ════════════════════════════════════════════════════════════════
        # CRITICAL: Dispatch via fire_and_forget_async with timeout
        # ════════════════════════════════════════════════════════════════
        # The previous implementation awaited ``subscriber.on_message``
        # inline. When the web peer's ``MessageCallbackService.on_message``
        # hung (e.g. its broadcast queue was full or its WS client was
        # dead), the agent's whole ``_handle_invoke`` flow blocked on the
        # very first pre-flight broadcast and no user message ever
        # reached ``ainvoke``. Production incident: a hung web peer
        # (>60 min hang, 3 parked ``_dispatch_request_async`` tasks).
        #
        # The fix: hand each ``on_message`` call to
        # ``fire_and_forget_async`` which:
        #   * dispatches the netref call in a fresh Task (returns
        #     immediately — the agent continues to ``ainvoke``),
        #   * wraps the awaitable in ``asyncio.wait_for(timeout=N)`` so
        #     a hung peer cannot pin an ``AsyncResult`` slot on
        #     ``Connection._request_callbacks`` forever,
        #   * pins the Task in ``rpyc.utils.helpers._INFLIGHT`` (with
        #     auto-discard via ``add_done_callback``) so a GC pass can't
        #     drop a half-finished outbound RPC and leak the chain
        #     (incidents 2026-05-08, 2026-05-11).
        #
        # Why NOT ``rpyc.async_(...)``: that forces ``HANDLE_CALL`` (sync
        # handler dispatch on the peer). Web's ``exposed_on_message`` is
        # declared ``async def`` — a sync handler returns the un-awaited
        # coroutine ("Boxing coroutine object … Did you forget to
        # await?") and the payload silently drops. The async-flagged
        # netref already picks ``HANDLE_ASYNC_CALL`` on its own.
        #
        # Why serialise once on the agent side: a netref payload forces
        # the subscriber's async loop to walk it via ``sync_request``,
        # which RPyC blocks with the "sync_request() from asyncio loop"
        # guard and leaks the remote ref. See
        # the design document on agent-service memory fixes (F2).
        # ════════════════════════════════════════════════════════════════

        try:
            payload_str = json.dumps(message, default=str)
        except (TypeError, ValueError) as exc:
            # Defensive: never drop a broadcast silently.
            logger.warning(
                "Could not JSON-serialise broadcast message (%s); falling back to str()",
                exc,
            )
            payload_str = json.dumps({"type": "raw", "content": str(message)})

        # Timeout for one outbound on_message round-trip. Overridable
        # via env so tests can drive a tight one without sleeping for
        # the production default. The production default is intentionally
        # generous: a healthy WS broadcast on localhost is sub-ms, but a
        # network-attached subscriber might legitimately need a few
        # seconds when the peer's event loop is busy.
        try:
            notify_timeout = float(os.environ.get("DWCLI_NOTIFY_TIMEOUT_SEC", "10"))
        except ValueError:
            notify_timeout = 10.0

        from rpyc.utils.helpers import fire_and_forget_async  # noqa: PLC0415

        # The shared subscriber set we evict dead peers from. Eviction
        # matters here for the same reason as in
        # ``AgentProcess._dispatch_status_event``: ``on_disconnect`` does
        # NOT clean up (it relies on a graceful unsubscribe that never
        # arrives when a web client dies/restarts ungracefully). Without
        # eviction the dead netref lingers forever and every broadcast
        # wastes a 10 s timeout against the corpse (a related internal
        # incident). A live web re-subscribes on its next connect_to_agent.
        subscriber_set = self._agent_process._message_subscribers

        for subscriber in subscribers:
            subscriber_id = id(subscriber)

            async def _on_error(
                exc: BaseException,
                _sid: int = subscriber_id,
                _sub: Any = subscriber,
            ) -> None:
                # NOTE: must not touch the netref here — that would
                # re-enter the very same dead connection. ``_sid`` is
                # captured by value (default arg) so we don't pin the
                # subscriber for the lifetime of this closure.
                logger.warning(
                    "subscriber on_message failed (id=%s): %r — evicting",
                    _sid,
                    exc,
                )
                # Evict by IDENTITY — ``set.discard(netref)`` would call
                # ``netref.__hash__`` → wire round-trip → EOFError on a dead
                # channel (a related internal incident). See evict_subscriber_by_identity.
                evict_subscriber_by_identity(subscriber_set, _sub)

            try:
                # Use asyncreq(HANDLE_CALLATTR=8) to avoid the
                # sync_request guard (same pattern as agent_process.py
                # _dispatch_status_event, issue #48).
                from rpyc.core.netref import asyncreq as _asyncreq  # noqa: PLC0415
                awaitable = _asyncreq(
                    subscriber,
                    8,  # HANDLE_CALLATTR
                    "on_message",
                    (payload_str,),
                    (),  # kwargs
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "could not dispatch on_message to subscriber (id=%s): %r — evicting",
                    subscriber_id,
                    exc,
                )
                # Evict by IDENTITY — ``set.discard(netref)`` would re-raise
                # EOFError via ``netref.__hash__`` on the dead channel, and
                # that secondary error used to escape into _handle_invoke and
                # force the agent to ERROR (a related internal incident).
                evict_subscriber_by_identity(subscriber_set, subscriber)
                continue

            fire_and_forget_async(
                awaitable,
                timeout=notify_timeout,
                error_callback=_on_error,
                name=f"notify-subscriber-{subscriber_id}",
            )
