# DESIGN: restart_self — Synchronous Self-Restart Tool

Status: implemented (2026-06-12, revision 2 — hard-exit added 2026-06-12)

## Revision 2 — sever-current-turn

The initial implementation (2026-06-12 morning) had a race:
``subprocess.Popen`` spawned the replacement process but the
**original** process kept running. Its in-flight LangGraph turn
continued — the LLM stream finalised, the assistant message committed
to the checkpointer, and the auto-continuation from the new process
appeared as a *separate* idle-state turn instead of the same logical
turn. The user reported this 2026-06-12 ("I don't see that you
restarted … after that you're in idle, which is NOT RIGHT").

**Fix:** after Popen succeeds, the tool calls ``os._exit(0)``
**immediately**. This:

- Kills the old process before LangGraph can commit the turn.
- The checkpointer keeps the last *persisted* state (the state right
  before the tool call). The pending tool call itself is NOT yet
  finalised in the message list.
- The replacement process resumes from that state, consumes the
  restart-notice, and its auto-continuation IS the same logical turn —
  not a separate idle-state turn.

The tool's structured ``{"success": True, ...}`` return is now
**unreachable** in production (kept only for the type contract and for
unit tests that patch ``os._exit``). Failure paths (notice write,
``command_unavailable``, ``spawn_failed``) intentionally do NOT call
``os._exit`` — the agent must stay alive to report the error.

## Goal

A tool an agent calls on itself to restart its own process — picking up
fresh on-disk code — WITHOUT losing thread state. The conversation
history and any in-progress task are preserved via the on-disk
checkpointer.

## Semantics — IMPORTANT

The tool is **SYNCHRONOUS**, like every other regular tool:

1. **Calling `restart_self` triggers the restart at call time.** It is
   NOT deferred to "after the current turn completes", it is NOT
   scheduled, it does NOT require the agent to first become idle.

2. **Calling `restart_self` does NOT put the agent into idle.** Control
   flow is identical to any other tool call. The new process picks up
   the task automatically (see auto-continue below).

3. **The new process receives a SYSTEM notice in its next invoke.** The
   notice is informational: it records the timestamp of the restart and
   the cause (`"self"` for restart_self). It does NOT ask the agent to
   acknowledge anything. The agent is expected to continue working.

These three properties were the explicit requirements from the project
owner (2026-06-12). They override any earlier draft of this document
that suggested deferred/scheduled semantics.

## High-level flow

```
[OLD PROCESS]
  Agent calls restart_self() tool
      │
      ├─ 1. Write ~/.deep_worker/<name>/.restart_notice.json
      │     contents: {"ts": "<iso>", "cause": "self", "message": "..."}
      │
      ├─ 2. subprocess.Popen(["deep-worker", "--restart", name],
      │                       start_new_session=True)
      │     — fire-and-forget. The Popen returns immediately.
      │
      └─ 3. Return {"success": True, "restart_notice": "<path>"}

  Almost immediately afterwards:
      deep-worker --restart driver sends SIGTERM to the old process.
      Old process shuts down (RPyC server stops, registry unregister).

[NEW PROCESS]  (started by `deep-worker --restart`)
  AgentProcess.start()
      │
      ├─ Read & remove .restart_notice.json
      │     → store dict on self._pending_restart_notice
      │
      ├─ Register in registry, expose RPyC port
      │
      └─ asyncio.create_task(self._auto_continue_after_restart())
            │
            └─ Calls _handle_invoke() with a placeholder user message.
               _build_invoke_messages prepends a SystemMessage with the
               notice text in front of the placeholder HumanMessage.
               The model receives:
                   [SystemMessage(restart notice), HumanMessage(continue)]
               and produces the next assistant turn — no manual user
               input required.

  The agent is now running on fresh code, with the model aware of the
  restart, and continues the task in progress.
```

## Files

### Tool implementation
`deep_worker_cli/tools.py` — `make_restart_self_tool(sender_name)`

The tool:
- Writes `.restart_notice.json` synchronously into the agent's data
  directory (`~/.deep_worker/<name>/`).
- Calls `shutil.which("deep-worker")` to locate the driver.
- `subprocess.Popen(...)` with `start_new_session=True` so the new
  process survives SIGTERM of the old one.
- Returns a regular tool-result dict.

If `deep-worker` is not on PATH the tool returns
`{"success": false, "error": "command_unavailable"}` and the notice
file is still on disk (harmless — it will be consumed if the agent is
restarted later by any means).

### Notice loader
`deep_worker_cli/process/agent_process.py` —
`AgentProcess._load_and_consume_restart_notice()` and
`AgentProcess._format_restart_notice_message()`.

The loader is single-shot: the file is removed whether the JSON parsed
or not, so a corrupt notice cannot loop the agent.

### Auto-continue
`deep_worker_cli/process/agent_process.py` —
`AgentProcess._auto_continue_after_restart()`.

Scheduled at the end of `start()`, before the main `shutdown_event`
wait, IF a restart-notice was consumed. It calls `_handle_invoke`
itself so the model produces its next turn without the user typing
anything.

### Message builder
`deep_worker_cli/process/agent_process.py` —
`_build_invoke_messages(content)` inside `_handle_invoke`.

Prepends the formatted SystemMessage to the HumanMessage when a notice
is pending, then clears the notice (single-shot).

## Failure modes

| Symptom | Cause | Handling |
| --- | --- | --- |
| `deep-worker` not on PATH | broken install | Tool returns `command_unavailable`. Notice file remains; if the user restarts the agent some other way, the notice is delivered. |
| Notice JSON corrupt | disk fault | Loader returns `None` and unlinks the file. Agent starts normally, no auto-continue. |
| `_handle_invoke` raises inside auto-continue | model API outage | Logged at WARN; the agent stays alive, ready to handle the next RPyC invocation. |
| Old process is killed by SIGTERM before `restart_self` returns | racing the driver | Harmless: the tool result is never observed by the model, but the new process is already up. Caller saw the call; we don't depend on the return value being observed. |

## Backward compatibility

The previous (deferred) design wrote `~/.deep_worker/<name>/.restart_self`
as a flag-file and relied on the running process's main loop to spawn
the restart subprocess. That mechanism is **removed**. As a courtesy:

- `AgentProcess.start()` still unlinks a stale legacy `.restart_self`
  flag if one is present on disk (from a process that hadn't been
  restarted with the new code yet). The flag is no longer interpreted.
- No more `_restart_pending` attribute on `AgentProcess`.
- No more "Restart scheduled" / "after the current turn completes"
  language anywhere in code or tool output. The user explicitly
  forbade that vocabulary (2026-06-12): it was never part of the
  requirements and was incorrectly extrapolated by an earlier author.

## Tests

- `tests/integration_tools_schema/test_restart_self_sync_mechanism.py`
  — locks the contract: synchronous Popen, notice file written, no
  flag file, no forbidden vocabulary in the tool's message.
- `tests/unit_tests/process/test_restart_notice_delivery.py` —
  load/consume/format mechanics.
- `tests/integration_ai/test_restart_self_e2e.py` — full e2e:
  spin up a real agent, give it a test prompt (kept separate from
  the system prompt!), trigger restart_self via the model, verify
  the restarted process delivered the system notice AND the model
  replied with the agreed-upon marker.
