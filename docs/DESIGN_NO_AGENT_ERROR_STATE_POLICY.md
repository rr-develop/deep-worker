# Design: No-Agent-Error-State Policy

**Status**: Draft → review (v2 — incorporated reviewer feedback)
**Date**: 2026-06-11
**Related**: `docs/ARCHITECTURE.md`, the CLI's `process/agent_process.py` and `types.py` (`AgentStatus.ERROR`), `claude_code_sdk_provider/chat_model.py`, `codex_provider/exec_runner.py`

## 1. User requirement (verbatim)

> Agents must ABSOLUTELY NEVER end up in the ERROR state if you can tell the agent "continue" and it will keep working.
>
> Any error that put it in that state could have been caught and sent to the agent as the result of a tool call (or whatever it was doing that led it into that state).
>
> Tools themselves must not throw exceptions; they must return error codes with clear explanations (and possibly write them to the log if something abnormal happens). But if some careless agent wrote a tool that raises exceptions, that must not crash the agent — NEVER — it must be returned to the agent as a tool-call error and also go into the log if it really is an error.

## 2. Policy (canonical text)

This text is the source of truth. The same wording (without profanity) is placed in `docs/POLICIES.md` and as comments in the relevant code paths.

> **No-Agent-Error-State Policy.**
>
> 1. An agent process must never transition to `AgentStatus.ERROR` for any failure that can be reported back to the model as a tool-call or system message. The default response to a failed tool call, a failed middleware, an unhandled provider exception, an empty-response from the model, an exceeded iteration cap, or any other unhandled internal failure inside the agent invocation pipeline is:
>    * convert it to a textual tool / system message,
>    * append it to the conversation,
>    * transition the agent to `IDLE` (so the user can simply type "continue"),
>    * and persist a structured record to the error log.
>
>    The `ERROR` status is reserved for truly fatal, non-recoverable conditions — usage / billing limit, corrupted checkpoint, unrecoverable provider auth failure, agent not initialised. The set of fatal exceptions is enumerated in code; everything else is recoverable by default.
>
> 2. Tools must not raise exceptions to signal expected failure modes. They must return a structured result whose payload contains an `error` field plus a human-readable explanation. If the failure represents an unexpected / internal condition (bug, OS-level error, invariant violation) the tool may *additionally* log it — but it must still return, not raise.
>
> 3. If a tool nevertheless raises (because someone wrote a sloppy tool), the agent runtime is responsible for catching that exception at the tool-dispatch boundary, converting it into a structured `ToolMessage` with `status="error"` visible to the model, logging it at `ERROR` level, and continuing. This catch is mandatory; bypassing it is a policy violation.
>
> 4. Diagnostic `print(...)` calls inside async hot paths (middleware, tool dispatch, chat-model invoke, the invoke catch-all itself) are forbidden, because `stderr` blocking under load (`BlockingIOError`) has caused the very failure mode this policy outlaws. Use the standard `logging` module with non-blocking handlers.
>
> Anyone breaking these rules writes bad code. Reviewers must reject such diffs.

## 3. Current behaviour (root cause)

* `agent_process.AgentProcess._handle_invoke` has a top-level `except Exception` that calls `session_manager.set_error(...)` and returns `success=False`. Any unhandled exception below — including a `print(...)` inside `claude_code_sdk_provider/chat_model.py:721` raising `BlockingIOError` — flips the agent to `ERROR`. The catch-all itself uses `print(..., file=sys.stderr)` at lines 2812–2813 and inside the inner `except` at 2827 / 2853, so the *fix* path is itself fragile.
* `chat_model.py` contains 37 `print(..., file=sys.stderr)` calls. Any of them can raise `BlockingIOError` under load.
* `_handle_invoke` also calls `set_error(...)` *without* an exception in three places:
  * line 2463 — `"No messages returned from agent"` (model returned an empty response).
  * line 2660 — `"Agent exceeded maximum iterations"`.
  * line 2770 — `"Agent not initialized"`.
  The first two are recoverable (the user can re-prompt); the third is genuinely fatal.
* Tools are typically `async def tool(...)` functions whose exceptions propagate through the dispatch layer (LangGraph `ToolNode`). Neither `ToolNode` nor our middleware wraps every tool call in a `try/except` that synthesises a `ToolMessage` with `status="error"`.
* Per-middleware code (`filesystem`, `subagents`, `memory`, `skills`, `local_context`, `project_memory`, `history_memory`, `message_timestamps`, `prompt_caching`, `todo`) re-raises whatever the inner handler raises. The catch at the very top loses the structure (no message id, no tool-call id) and the user only sees `INVOKE_ERROR error=...` in the log.

## 4. Proposed change

### 4.1 Fatal-exception registry — narrow `ERROR`

Add to a single new module `errors.py` in the CLI:

```python
"""Centralised exception taxonomy for the No-Agent-Error-State Policy.

See docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md.
"""


class CheckpointCorruptedError(RuntimeError):
    """Raised by storage when the on-disk thread state cannot be loaded."""


class ProviderAuthFatalError(RuntimeError):
    """Raised by a provider when credentials are revoked / unrecoverable."""


class AgentNotInitialisedError(RuntimeError):
    """Raised when the agent's runtime graph is missing — cannot proceed."""


# Match by class name so the table stays provider-agnostic (codex agents
# never import the claude-code-sdk module and vice versa).
FATAL_EXCEPTION_TYPES: frozenset[str] = frozenset({
    "ClaudeUsageLimitError",
    "CodexUsageLimitError",
    "CheckpointCorruptedError",
    "ProviderAuthFatalError",
    "AgentNotInitialisedError",
})
```

`AgentStatus.ERROR` is set only for these. Everything else is recoverable.

### 4.2 New top-level catch in `_handle_invoke`

Replace the existing `except Exception as e` block (`process/agent_process.py` ≈ lines 2811–2898) with:

```python
except Exception as e:
    exc_name = type(e).__name__
    is_fatal = exc_name in FATAL_EXCEPTION_TYPES

    # Always log with traceback (logger, not print — see §4.6).
    _logger.exception("[INVOKE] Unhandled exception in agent invocation")

    # Always tell the TUI / web what happened.
    if notify_callback is not None:
        with contextlib.suppress(Exception):
            await notify_callback({
                "type": "error" if is_fatal else "system",
                "role": "system",
                "content": _format_error_for_user(e, is_fatal),
            })

    if is_fatal:
        # Preserves the existing fatal path: usage-limit handlers, session
        # marker, failover dispatch.  Implementation MUST keep calling
        # ``self._schedule_failover(...)`` for ClaudeUsageLimitError /
        # CodexUsageLimitError exactly as today (lines 2841 and 2866).
        return await self._handle_fatal(
            exc=e,
            input_message_id=input_message_id,
            last_error_messages=last_error_messages,
            notify_callback=notify_callback,
        )

    # Recoverable: persist a SystemMessage in the thread, leave status IDLE,
    # so the next user message ("continue") just resumes the conversation.
    await self._append_recoverable_error_to_history(
        exception=e,
        input_message_id=input_message_id,
    )
    if self.session_manager:
        await self.session_manager.set_status(AgentStatus.IDLE)
    return {
        "success": False,
        "recovered": True,
        "error": str(e),
    }
```

`_append_recoverable_error_to_history` writes a `SystemMessage` that explains in plain text what happened and that the model should react to it on the next turn. It uses the existing checkpointer to persist. Files are created with `0o600` mode (no `0o777` / `0o666`).

### 4.3 Non-exception `set_error` sites

The three callers at lines 2463, 2660, 2770 are reclassified:

| Site | Current | New behaviour |
|------|---------|---------------|
| 2463 — `"No messages returned from agent"` | `set_error` → `ERROR` | append `SystemMessage("The model returned an empty response; reply to continue.")` → `set_status(IDLE)` |
| 2660 — `"Agent exceeded maximum iterations"` | `set_error` → `ERROR` | append `SystemMessage("The agent hit its iteration cap. Send 'continue' to resume.")` → `set_status(IDLE)` |
| 2770 — `"Agent not initialized"` | `set_error` → `ERROR` | **stays fatal** — raise `AgentNotInitialisedError`, which goes through the new fatal path |

This is implemented as a small helper `_recover_or_fatal(reason, *, fatal: bool, history_msg: str | None)` so the three sites stay short.

### 4.4 Tool-dispatch safety net — `SafeToolNode`

Following the reviewer's recommendation (Option A): wrap LangGraph's `ToolNode` rather than inserting an extra middleware.

```python
# the CLI's runtime/safe_tool_node.py

class SafeToolNode:
    """Wraps ``langgraph.prebuilt.ToolNode`` so every tool exception is
    converted to a ``ToolMessage(status="error")`` instead of propagating
    up to the agent invoke catch-all.

    This is the policy boundary required by §3 of
    docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md.
    """

    def __init__(self, tools):
        self._inner = ToolNode(tools)

    async def ainvoke(self, input, config=None, **kwargs):
        try:
            return await self._inner.ainvoke(input, config=config, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.error(
                "Tool dispatch raised %s: %s",
                type(exc).__name__, exc, exc_info=True,
            )
            # ToolNode returns a list of ToolMessage instances; mirror that.
            call_id = _extract_first_tool_call_id(input) or "unknown"
            return [ToolMessage(
                tool_call_id=call_id,
                name="<dispatch>",
                status="error",
                content=(
                    f"Tool dispatch raised an unhandled exception "
                    f"({type(exc).__name__}): {exc}. The agent runtime "
                    f"converted it into this error message so you can continue."
                ),
            )]

    # Forward graph-protocol attributes (``tools_by_name``, ``__call__``,
    # streaming hooks) to the inner node so LangGraph treats us identically.
    def __getattr__(self, name):
        return getattr(self._inner, name)
```

Every site that today instantiates `ToolNode(tools)` instantiates `SafeToolNode(tools)` instead. The build sites are:
* `deepagents/factory` (the create-agent factory).
* the CLI's custom agent assembly (only if it bypasses the factory).

The middleware chain is **not** modified — the safety net is one layer, at the dispatcher.

### 4.5 Middleware safety net — factory-level wrap

Factory wrap of `awrap_model_call` middleware: at the create-agent factory we wrap the composed middleware in one outer `try/except` that converts non-`CancelledError` exceptions into a `SystemMessage` injected into the model response. This way we avoid editing every individual middleware and any future middleware is automatically covered.

### 4.6 Forbid `print()` in async hot paths

* Replace `print(..., file=sys.stderr)` in:
  * `claude_code_sdk_provider/chat_model.py` — all 37 sites.
  * `claude_code_sdk_provider/proxy_mcp_server.py` — the `[MCP_CAPTURE handler] ...` debug prints that should have been removed when the related fix landed.
  * the CLI's `process/agent_process.py` — including the prints **inside** the catch-all itself (lines 2812–2813, 2827, 2833, 2849, 2858, 2883).
  * `deepagents/middleware/*.py` — any `[DEBUG ...] ` prints.
* All replaced sites use `logging.getLogger(__name__)`. Our existing logging stack uses `RotatingFileHandler` + `QueueHandler` so log writes are non-blocking from the caller's point of view (the queue drains on a dedicated worker — that worker existed before this change and is not counted as a "new thread").
* Add a banner comment at each replacement site:
  > "NEVER use print() in this code path — stderr can block under load and `BlockingIOError` propagates up to the agent invoke catch-all, which used to flip the agent to `ERROR`. See `docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md`."

Pre-commit guard: a small grep-based hook in `scripts/check_no_print_in_hot_paths.py` flags new `print(` in the listed files. Allowed exceptions must carry `# noqa: T201 reason=...`. The hook runs from `.pre-commit-config.yaml`; bypassing pre-commit is already forbidden by the user's policy.

### 4.7 Logging the recoverable error

`_log_error_context(...)` is extended with `recoverable: bool`. A single structured JSON line is emitted per occurrence so the web UI can render per-agent error counters. Fields: `agent_name`, `recoverable`, `exc_type`, `exc_repr`, `input_message_id`, `traceback`, `last_two_messages_summarised`.

## 5. Non-goals

* This design does not change `INTERRUPTED`, `CANCELLED`, `STOPPED`. It only narrows `ERROR`.
* No watchdog / auto-resume. The user still has to send "continue" — that is exactly the user's stated expectation.
* No polling, no new threads (the logging queue worker already exists).
* No `0o777` / `0o666` permissions on any file or directory introduced by this change; new files default to `0o600`, new directories to `0o700`.

## 6. Migration / compatibility

* Web UI: `ERROR` is shown red today. Once an agent recovers, it returns to IDLE and the badge disappears. No schema migration is required — `AgentStatus.IDLE` already exists. The new `"type": "system"` notify payload is already a valid value on the web side; if it isn't, we extend the web mapping only (no contract change in the runtime).
* For existing agents that are stuck in `ERROR`, the first user message triggers a normal turn — no special migration.

## 7. Testing

* **Unit**: `tests/test_no_error_state_policy.py`
  1. Tool raises `RuntimeError` → `SafeToolNode` returns `ToolMessage(status="error")`, agent stays `IDLE`.
  2. Middleware raises → factory wrap injects a `SystemMessage`, agent stays `IDLE`.
  3. `ClaudeUsageLimitError` raised → agent goes to `ERROR`, `_schedule_failover("claude-code-sdk", ...)` called exactly once.
  4. `CodexUsageLimitError` raised → agent goes to `ERROR`, `_schedule_failover("codex", ...)` called exactly once.
  5. Fake `BlockingIOError` from a stub print-like call inside chat-model → agent stays `IDLE`.
  6. Empty-message path (line 2463) → agent stays `IDLE`, `SystemMessage` appended.
  7. Max-iterations path (line 2660) → agent stays `IDLE`, `SystemMessage` appended.
  8. `AgentNotInitialisedError` (line 2770) → agent goes to `ERROR` (fatal).
* **Integration (no mocks)**: spin up a throw-away agent (`test_policy_$rand`) via `deep-worker`, monkey-patch one tool to raise, send a user message, assert the agent ends `IDLE`, the next user message ("continue") proceeds normally, and the JSONL log contains exactly one `recoverable=true` line for that turn.

## 8. Open questions for the user

None I need answered before implementing. The reviewer's required changes are all incorporated; the policy and scope are explicit in the user's message.

## 9. Out of scope

* Rate-limit / back-off tuning of providers.
* Web UI polish for the new system-message badge (a follow-up).
