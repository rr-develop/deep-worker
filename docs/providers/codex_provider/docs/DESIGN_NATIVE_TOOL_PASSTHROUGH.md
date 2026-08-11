# Design: Native Codex tools recorded into LangGraph as real tool calls

**Status:** ✅ IMPLEMENTED (rev 2) — landed 2026-05-18.
See `codex_provider/{event_pump,native_tools}.py` and the matching
test files. Pyright strict 0 errors, 101 unit tests pass, live agent
no longer crashes with `status="error"` on native tool events.
**Date:** 2026-05-18
**Supersedes:** the `_raise_if_blocked_native_item` defense-in-depth
behaviour added in commit `ca13bfb` (2026-05-17). That defence raises
a `RuntimeError` whenever Codex emits a native tool event — which
aborts the turn, returns `status="error"` to the user, and throws
away a perfectly valid Codex action. There is no policy under which
raising is correct: either we drive a deep-worker base tool, or Codex
drives a native tool. Both must be representable in LangGraph history.

**Revision history:**
- rev 1 (2026-05-18 earlier): proposed recording a single combined
  pair at `item.completed`. Caveat noted: the in-progress window was
  invisible to the LangGraph UI.
- rev 2 (2026-05-18 — this document): **the tool call is open in
  LangGraph for the full duration of Codex's native execution**, so
  the agent's running status reflects reality. Achieved via an
  asyncio.Future parked inside an async tool body, woken event-driven
  when Codex emits `item.completed`. No polling anywhere.

## 1. Problem statement

Codex CLI 0.130 ships seven native tools that no `--disable <feature>`
or top-level config key can strip from the API request:

```
apply_patch, update_plan, view_image, request_user_input,
list_mcp_resources, list_mcp_resource_templates, read_mcp_resource
```

(See `DESIGN_CODEX_PROVIDER.md` Implementation note 6 for the full
mapping and the live verification.) The model is free to pick any of
them mid-turn, and the matching `item.started`/`item.completed`
events arrive on the `codex exec --experimental-json` stdout stream.

Today `CodexChatModel._raise_if_blocked_native_item` raises a
`RuntimeError` on every such event. That:

- aborts the LangGraph turn,
- propagates through `AgentProcess._handle_invoke`,
- writes `status="error"` and `error_message="Agent invocation failed:
  Blocked native Codex tool event..."` into
  the agent's session metadata file (`session_metadata.json`) under its
  data directory,
- returns `{"success": false, ...}` to the client,
- leaves `state.current_exec` pointing at a now-dead subprocess.

The agent process itself stays alive, but the turn is dropped.

## 2. What we observe from Codex on the wire

Live capture against codex-cli 0.130.0 for the prompt "run `ls /tmp`":

```jsonl
{"type":"thread.started","thread_id":"..."}
{"type":"turn.started"}
{"type":"item.started","item":{"id":"item_0","type":"command_execution",
  "command":"/bin/bash -lc 'ls /tmp'","aggregated_output":"",
  "exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_0","type":"command_execution",
  "command":"/bin/bash -lc 'ls /tmp'","aggregated_output":"<stdout>",
  "exit_code":0,"status":"completed"}}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message",
  "text":"`ls /tmp` completed..."}}
{"type":"turn.completed","usage":{...}}
```

Two events flank Codex's actual work:

- `item.started` — emitted **the moment Codex decides to run the
  tool**, before any output exists. Carries the full arguments
  (`command` for `command_execution`, the patch for `apply_patch`,
  etc.). `status="in_progress"`, `aggregated_output=""`, `exit_code=null`.
- `item.completed` — emitted **the moment Codex finishes the tool**.
  Carries the full result (`aggregated_output`, `exit_code`, terminal
  `status`).

Between these two events Codex is actively running the tool. The
window can be tens of milliseconds (`ls`) or several minutes (a
large `find`). The design uses both events.

## 3. The contract in one paragraph

When Codex emits `item.started` for a tool call that is **not** a
deep-worker proxy MCP call, the codex_provider returns an
`AIMessage(tool_calls=[<real Codex tool record>])` from `ainvoke` and
makes LangGraph route into ToolNode. The matching async tool body
parks on an `asyncio.Future` that lives in shared state. A
**background event-pump task**, started at `codex exec` spawn and
running in the same event loop, drains `events_iter` continuously and
sets that Future the moment Codex emits `item.completed`. The tool
body returns the result string, ToolNode wraps it into a
`ToolMessage`, LangGraph records the pair, and the next `ainvoke`
sees a closed tool call — it makes no API call to Codex (Codex was
never waiting for us, it has already moved on by itself) and instead
keeps reading subsequent events from the same pump until Codex emits
the next tool call or the final `agent_message`. No polling. No
exceptions. The tool_call name, args, and result are exactly what
Codex sent — LangGraph history shows a real Codex native tool call,
not a synthetic surrogate.

## 4. LangGraph history shape — what users see

For one native `command_execution` of `ls /tmp`:

```python
AIMessage(content="", tool_calls=[
    {
        "id":   "item_0",                                # Codex's own id
        "name": "command_execution",                     # Codex's tool name
        "args": {"command": "/bin/bash -lc 'ls /tmp'"},  # Codex's args
        "type": "tool_call",
    },
])
ToolMessage(
    content     = "<aggregated_output>",
    tool_call_id= "item_0",
    name        = "command_execution",
    status      = "success",                             # or "error" if exit_code!=0
    additional_kwargs={
        # Optional: surface Codex-side metadata so dwcli / history
        # views can render it. Same shape Codex itself uses.
        "codex_native": True,
        "exit_code":    0,
        "codex_status": "completed",
    },
)
```

Nothing in the visible payload identifies this as anything other than
a Codex native tool call. `name="command_execution"` (or
`"apply_patch"`, `"web_search"`, etc.) is exactly what Codex
advertised in its API tool schema. `tool_call_id="item_0"` is exactly
the id Codex stamped on the event. The fact that we routed it
through a LangGraph BaseTool implementation is an internal detail of
the codex_provider — not visible in history, not visible in dwcli
output, not visible in the agent's TUI.

The `additional_kwargs.codex_native=True` flag is the only marker,
and it lives in `ToolMessage.additional_kwargs` precisely because
that field is **for** provider-side metadata that should travel with
the message but does not change its meaning. Outbound (§6) uses this
flag to short-circuit Codex re-injection. Removing it from the
history view (if dwcli renders `additional_kwargs`) is purely a
display choice; nothing depends on it being shown.

## 5. Architecture overview

```
                ┌──────────────────────────────────────────┐
                │ codex exec --experimental-json (subproc) │
                └──────────────────────┬───────────────────┘
                                       │ JSONL stdout
                                       ▼
        ┌──────────────────────────────────────────────────────┐
        │ ExecHandle._iter_events  (async generator, today)    │
        └──────────────────────┬───────────────────────────────┘
                               │
                               ▼  drained event-by-event
        ┌──────────────────────────────────────────────────────┐
        │ EventPumpTask  (NEW — one asyncio.Task per ExecHandle│
        │  spawned at ExecHandle creation time)                │
        │                                                      │
        │   for ev in events_iter:                             │
        │       match ev:                                      │
        │         ThreadStartedEvent → record session uuid     │
        │         ItemStartedEvent(McpToolCallItem on proxy):  │
        │             enqueue ev into ainvoke_queue            │
        │         ItemStartedEvent(GenericItem, native):       │
        │             native_futures[item.id] = Future()       │
        │             enqueue ev into ainvoke_queue            │
        │         ItemCompletedEvent(McpToolCallItem on proxy):│
        │             ignore (already resolved by us)          │
        │         ItemCompletedEvent(GenericItem, native):     │
        │             native_futures[item.id].set_result(...)  │
        │             # ainvoke does NOT need this event       │
        │         ItemCompletedEvent(AgentMessageItem):        │
        │             enqueue into ainvoke_queue               │
        │         TurnCompletedEvent / TurnFailedEvent:        │
        │             enqueue into ainvoke_queue, then stop    │
        └──────────────────────┬───────────────────────────────┘
                               │
              ┌────────────────┴───────────────┐
              ▼                                ▼
   ┌─────────────────────────┐   ┌──────────────────────────────┐
   │ ainvoke_queue           │   │ native_futures               │
   │ (asyncio.Queue[Event])  │   │ (dict[item_id → Future[str]])│
   │                         │   │                              │
   │ read by:                │   │ awaited by:                  │
   │   CodexChatModel.ainvoke│   │   codex_provider tool bodies │
   │   inside its event loop │   │   inside ToolNode execution  │
   └─────────────────────────┘   └──────────────────────────────┘
```

Three pieces, all event-driven, no polling anywhere:

- **`EventPumpTask`** owns the single reader of `events_iter`. It
  routes everything. Two destinations: an `asyncio.Queue` consumed by
  `ainvoke`, and a dict of `asyncio.Future`s awaited by the tool
  bodies. Each event lands in exactly one destination (proxy MCP
  resolution stays in `ainvoke`'s direct concern; native completion
  stays in the Future map). The pump task is created at
  `CodexExecRunner.spawn` time and torn down by `ExecHandle.aclose`.
- **`ainvoke_queue`** is the bounded asyncio.Queue that `ainvoke`
  reads from instead of reading `events_iter` directly. Bounded queue
  means natural back-pressure on the rare path where Codex emits
  faster than ainvoke consumes; default size 64 is generous.
- **`native_futures`** is a small dict on `ExecHandle` (so its
  lifetime is tied to the subprocess, not the global state).
  Populated by the pump on `item.started`, resolved by the pump on
  `item.completed`, looked up by the tool body. The tool body is
  the **only** consumer of any given Future; resolution wakes it
  exactly once.

## 6. Inbound flow — opening the tool call at `item.started`

`CodexChatModel.ainvoke` step 10 changes from "loop directly on
`events_iter`" to "loop on `ainvoke_queue`". For native tools the
behaviour is:

```python
async for event in ainvoke_queue:
    if isinstance(event, ItemStartedEvent):
        it = event.item

        if isinstance(it, McpToolCallItem) and \
           it.server == self._proxy_mcp.server_name and \
           it.status == "in_progress":
            # Proxy MCP — unchanged from today.
            pending_tool_calls = await self._proxy_mcp.wait_for_pending_calls(
                tool_name=it.tool, arguments=it.arguments, timeout=30.0,
            )
            if pending_tool_calls:
                break

        elif isinstance(it, GenericItem) and \
             it.type in NATIVE_TOOL_ITEM_TYPES:
            # Native tool starting. Codex is now running it. The pump
            # has already created native_futures[it.id] for us.
            pending_tool_calls.append({
                "id":   it.id,
                "name": it.type,
                "args": _native_args(it),
                "type": "tool_call",
            })
            break   # return to LangGraph immediately; the tool body
                    # will await native_futures[it.id]

    elif isinstance(event, ItemCompletedEvent):
        it = event.item
        if isinstance(it, AgentMessageItem):
            final_text = it.text
        # McpToolCallItem completion is silently dropped (we already
        # have the ToolMessage). Native completion never reaches this
        # loop — the pump consumed it into native_futures.

    elif isinstance(event, TurnCompletedEvent):
        turn_done = True
        break
    elif isinstance(event, TurnFailedEvent):
        turn_done = True
        raise RuntimeError(f"Codex turn failed: {event.error_message}")
```

Key invariant: **we break on `item.started` for natives, not on
`item.completed`**. That's what makes the tool call appear in
LangGraph history while Codex is still running it. The agent's
`status` shows the standard "tool running" state for the entire
in-progress window, identically to MCP tools.

### 6.1 What event types are "native"

```python
NATIVE_TOOL_ITEM_TYPES = frozenset({
    "command_execution",
    "file_change",          # apply_patch emits this
    "web_search",
    "tool_search",
    "custom_tool_call",
    "function_call",        # generic envelope (rare on exp-json,
                            # common in session_meta logs)
    "function_call_output", # paired with function_call
})
```

This list is informational — adding a new type is a single-line
change. Anything not in the list and not a known `GenericItem`
subtype (reasoning, etc.) is logged and ignored as it is today
(no `raise`).

### 6.2 Extracting tool name and args

```python
def _native_tool_name(it: GenericItem) -> str:
    """For most types, the type IS the tool name."""
    return it.type

def _native_args(it: GenericItem) -> dict[str, Any]:
    """Project the payload to model-visible arguments only.
    Excludes output fields that are populated later (aggregated_output,
    exit_code) — those belong in the ToolMessage, not the tool_call.
    """
    p = it.payload
    if it.type == "command_execution":
        return {"command": p.get("command", "")}
    if it.type == "file_change":
        return {k: v for k, v in p.items() if k != "aggregated_output"}
    if it.type == "web_search":
        return {"query": p.get("query", "")}
    return {k: v for k, v in p.items()
            if k not in ("aggregated_output", "exit_code", "status",
                         "result", "error")}
```

## 7. The async tool bodies — event-driven wait, no polling

One BaseTool per native item type, all `async def`. Body skeleton:

```python
@tool
async def command_execution(command: str) -> str:
    """Codex native shell. Implementation is event-driven: parks on
    the Future created by EventPumpTask at item.started and woken at
    item.completed.
    """
    # Pull the tool_call_id from the ambient LangGraph context. Both
    # langchain_core.runnables.config.get_runnable_config() and the
    # current ToolCall.id are available.
    cfg = get_runnable_config()
    tool_call_id = cfg["metadata"]["tool_call_id"]   # set by ToolNode

    handle = _current_handle_for_thread(cfg["configurable"]["thread_id"])
    future = handle.native_futures.get(tool_call_id)
    if future is None:
        # Defensive — should never happen, the pump always pre-creates
        # the Future on item.started before ainvoke returns the
        # AIMessage that triggers ToolNode.
        raise RuntimeError(
            f"No pending native call registered for id={tool_call_id!r}"
        )

    # Pure await — no polling, no sleep, no timeout loop.
    result = await future
    # Once consumed, drop from the map so the dict does not grow
    # unbounded across a long session.
    handle.native_futures.pop(tool_call_id, None)
    return result
```

`await future` blocks the ToolNode coroutine. LangGraph's executor
schedules other coroutines (event pump, web requests, etc.) in the
same loop. The Future resolves when the pump processes
`item.completed`. No timer, no `asyncio.sleep`, no `while True`.

The same skeleton serves every native type: `apply_patch`,
`web_search`, `tool_search`, `file_change` (apply_patch result),
`update_plan`, `view_image`. The tool signature mirrors the args
projection from §6.2 so the LLM-facing JSON schema matches what
Codex actually advertised.

### 7.1 How the tool body finds its ExecHandle

The tool body needs to reach `handle.native_futures`. Two viable
plumbing options:

1. **`thread_id` → ExecHandle lookup in `GlobalClientState`.**
   `ClientState` already has a per-thread record (today carries
   `codex_session_uuid`, `sent_count`, `current_exec`). The tool body
   reads `thread_id` from LangGraph runnable config, then
   `client_state.get(thread_id).current_exec`. Zero new globals.
2. **contextvar set by `ainvoke` before returning the AIMessage.**
   Cleaner separation but contextvars cross asyncio.Task boundaries
   unreliably; we avoid them.

**Decision:** option 1. `current_exec` already exists, holding the
ExecHandle is its purpose, the tool body reads it via the standard
`thread_id` path.

### 7.2 The tool body's tool_call_id source

LangGraph's standard ToolNode injects the tool_call_id into the
tool's runnable config (and also as an annotated argument when the
tool signature includes `tool_call_id: Annotated[str, InjectedToolCallId]`).
The body uses whichever LangChain provides. The skeleton above shows
the runnable-config path; the InjectedToolCallId path is equivalent
and may be preferable for explicitness. Pick the form that current
LangChain version supports without warnings.

## 8. The EventPumpTask — the only reader of events_iter

Created inside `CodexExecRunner.spawn` immediately after the
subprocess starts:

```python
async def spawn(self, args: ExecArgs) -> ExecHandle:
    # ... existing subprocess + stdin handling ...
    events = self._iter_events(process, args.idle_timeout_seconds)
    handle = ExecHandle(
        process=process,
        events=events,                 # kept private to the pump
        override_file_to_cleanup=override_file,
    )
    handle.ainvoke_queue   = asyncio.Queue(maxsize=64)
    handle.native_futures  = {}        # dict[str, asyncio.Future[str]]
    handle.pump_task = asyncio.create_task(
        _event_pump(handle, proxy_mcp_server_name=...),
        name=f"codex-event-pump-{process.pid}",
    )
    return handle
```

The pump body:

```python
async def _event_pump(
    handle: ExecHandle,
    proxy_mcp_server_name: str,
) -> None:
    try:
        async for ev in handle.events:
            await _route_event(handle, ev, proxy_mcp_server_name)
    finally:
        # Unblock any stuck native tool bodies — Codex is gone.
        for fut in handle.native_futures.values():
            if not fut.done():
                fut.set_exception(
                    RuntimeError("codex exec exited before "
                                 "item.completed for this native call")
                )
        # Signal end-of-stream to ainvoke.
        with contextlib.suppress(asyncio.QueueFull):
            handle.ainvoke_queue.put_nowait(_END_OF_STREAM)


async def _route_event(handle, ev, proxy_server_name):
    if isinstance(ev, ItemStartedEvent) and \
       isinstance(ev.item, GenericItem) and \
       ev.item.type in NATIVE_TOOL_ITEM_TYPES:
        # Pre-create the Future BEFORE the AIMessage with this
        # tool_call_id can be observed by ToolNode. Guarantees the
        # tool body never races and finds an empty slot.
        loop = asyncio.get_running_loop()
        handle.native_futures[ev.item.id] = loop.create_future()
        await handle.ainvoke_queue.put(ev)
        return

    if isinstance(ev, ItemCompletedEvent) and \
       isinstance(ev.item, GenericItem) and \
       ev.item.type in NATIVE_TOOL_ITEM_TYPES:
        # Resolve the parked Future. ainvoke does NOT need this event.
        fut = handle.native_futures.get(ev.item.id)
        if fut is not None and not fut.done():
            fut.set_result(_native_result_text(ev.item))
        return

    if isinstance(ev, ItemCompletedEvent) and \
       isinstance(ev.item, McpToolCallItem) and \
       ev.item.server == proxy_server_name:
        # Already resolved by us via _proxy_mcp; nothing to do here.
        # ainvoke does not need this event either.
        return

    # Everything else (ThreadStarted, TurnStarted, mcp ItemStarted,
    # AgentMessage ItemCompleted, TurnCompleted, TurnFailed, any
    # GenericItem we don't classify as native) goes to ainvoke.
    await handle.ainvoke_queue.put(ev)
```

### 8.1 Encoding the result string

```python
def _native_result_text(it: GenericItem) -> str:
    p = it.payload
    if it.type == "command_execution":
        rc  = p.get("exit_code")
        st  = p.get("status", "")
        out = p.get("aggregated_output", "")
        # Identical shape Codex itself shows in session logs.
        return out if rc == 0 else \
               f"exit_code={rc} status={st!r}\n{out}"
    if it.type == "file_change":
        return p.get("aggregated_summary") or json.dumps(p)
    return json.dumps(p, ensure_ascii=False)
```

For `ToolMessage.status`: `"success"` when `exit_code == 0`,
`"error"` otherwise. The skeleton in the tool body returns the
plain string; the ToolMessage shape (including status field) is
populated either by ToolNode standard behaviour (when the tool
raises) or by the tool itself returning a tuple / a
`Command(update=...)` — pick whichever cleanly carries
status="error" through. If LangChain does not give us a clean way,
prepend an exit_code marker to the content and leave
status="success" — that loses one field but the content carries it.

## 9. Outbound flow — letting Codex keep running

`CodexChatModel.ainvoke` step 7 (the `for msg in new_messages` loop)
already does the right thing for proxy MCP `ToolMessage`s:
`_proxy_mcp.resolve_call(msg.tool_call_id, ...)` unblocks the parked
HTTP handler, which lets Codex receive the result and continue
inside the same `codex exec` process.

Native `ToolMessage`s need the **opposite**: Codex did not pause for
this tool (it ran it itself in-process). It has already moved on,
emitting the next event. We must NOT inject anything back into
Codex.

The discriminator is `ToolMessage.additional_kwargs["codex_native"]`
set in §4 (carried through LangGraph state untouched):

```python
for msg in new_messages:
    if isinstance(msg, ToolMessage):
        is_native = bool(
            msg.additional_kwargs.get("codex_native", False)
        )
        if is_native:
            # Codex never blocked on this. The event pump already
            # consumed item.completed and woke the tool body. Nothing
            # to push into Codex.
            saw_native_resume = True
        else:
            # Proxy MCP: Codex IS blocked on _proxy_mcp's HTTP handler
            # awaiting this result; unblock it.
            self._proxy_mcp.resolve_call(
                msg.tool_call_id,
                {"content": _build_mcp_content(msg.content)},
            )
            tool_resolved = True
    # ... HumanMessage / AIMessage / SystemMessage unchanged ...
```

Mid-turn reuse triggers on either signal:

```python
if (tool_resolved or saw_native_resume) and \
   state.current_exec is not None and not user_text_parts:
    handle = state.current_exec
    # Keep reading the SAME ainvoke_queue. Background pump kept
    # filling it while ToolNode was running our async tool body.
else:
    # New user message OR no resume needed — close stale exec,
    # spawn fresh. Unchanged from today.
    ...
```

Concretely: the queue most likely already has the next event sitting
in it — Codex did not wait for us. The first iteration of step-10's
`async for event in ainvoke_queue` returns immediately, either with
the next `ItemStartedEvent` (more tool calls) or with the final
`AgentMessageItem` + `TurnCompletedEvent`.

### 9.1 The fallthrough when no message resumes the turn

If LangGraph's routing somehow delivers a `HumanMessage` mid-turn
without first delivering the ToolMessages (e.g. user pre-empts), the
existing `else` branch already closes the stale ExecHandle and spawns
a fresh `codex exec`. The pump task is cancelled by
`ExecHandle.aclose` (see §11), so a half-running Codex turn cannot
leak.

## 10. Tool registration

The async tools live in `codex_provider/native_tools.py` as
LangChain BaseTool objects. They are attached to the agent's bound
tool set **wherever a CodexChatModel-backed agent is created with
`provider == "codex"`** — there are two such sites today:

1. **Main agent** (`deep_worker_cli/process/agent_process.py:475-477`):
   appends `NATIVE_TOOLS` to `web_tools` before passing to
   `create_cli_agent`.
2. **TaskManager (`-tm`) manager agent**
   (`deep_worker_cli/task_manager/task_manager.py:_initialize_manager`):
   the manager runs against the same Codex backend as the subordinate
   (`create_model(provider=self.config.provider, ...)`), so it needs
   the same passthrough surface. Without it, a native tool emitted
   during a manager turn (e.g. `update_plan`) would arrive at
   LangGraph's ToolNode as `tool_calls[{"name": "update_plan", ...}]`
   and `ToolNode` would raise `Tool 'update_plan' not found`,
   crashing the manager turn and triggering its `disabled=True`
   degradation path. The manager extends `manager_tools` with
   `NATIVE_TOOLS` symmetrically to the main-agent site.

Both sites guard with `if config.provider == "codex"` so non-codex
providers do not see Codex-specific BaseTool objects in their bound
set.

The native tools are NOT pushed into the ProxyMCPServer's
`tools_metadata`, because Codex must not be told about them (Codex
already has its own native equivalents and is the authoritative
caller).

Tool names exactly match the Codex `item.type` values they
correspond to (`command_execution`, `apply_patch`, `web_search`,
`tool_search`, `file_change`, `custom_tool_call`, `function_call`).
This is what makes LangGraph history look like real Codex calls in
§4 — the same name flows through `tool_calls[].name` and
`ToolMessage.name`.

### 10.1 Known name collision with `web_search_delegation_tool`

The Codex native passthrough `web_search` has the same name as
deep-worker's universal `web_search_delegation_tool` (defined in
`deep_worker_cli/tools.py`; instance name is `"web_search"`). Both
end up in the bound tool set of every Codex-provider agent and every
TaskManager agent that runs against a Codex subordinate. LangGraph
resolves `tool_call.name="web_search"` by **last registration**, so
the native passthrough wins.

This is acceptable today because:

- Codex CLI's own native `web_search` is already disabled in
  `CodexExecRunner._build_argv` via the top-level
  `web_search="disabled"` config override (Codex 0.130 moved
  `web_search` out of feature flags into a top-level string). So in
  practice the model never sees a native `web_search` tool in its API
  request and the passthrough never gets called.
- The native passthrough is a defense-in-depth: if a future Codex
  release re-exposes native web_search, the passthrough will record
  the call into LangGraph history, exactly as for any other native
  tool.
- `web_search_delegation_tool` remains the model's actual web-search
  surface because Codex CLI also accepts our bound tool set via the
  proxy MCP server (under `mcp__deepworker_<pid>__web_search`), which
  Codex *does* expose to the model in the API request.

Future cleanup option: rename one of them to remove the collision.
Out of scope for this design.

Function-level docstrings on these tools are deliberately terse and
neutral (e.g. `command_execution` → "Codex native shell command
executor"). They will never reach the model (Codex never sees them);
they exist only to populate the JSON schema LangGraph uses
internally and to give human readers of the source a one-line idea
of what each represents.

## 11. Lifecycle

`ExecHandle.aclose` gains pump-task cleanup:

```python
async def aclose(self) -> None:
    # Stop the pump first so it can no longer touch native_futures.
    if self.pump_task is not None and not self.pump_task.done():
        self.pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.pump_task
        self.pump_task = None

    # Cancel any still-parked native tool body.
    for fut in self.native_futures.values():
        if not fut.done():
            fut.cancel()
    self.native_futures.clear()

    # ... existing: close events generator, terminate subprocess,
    # remove AGENTS.override.md ...
```

`aclose` is called from two places today: `ainvoke`'s `finally` when
`turn_done=True`, and the mid-turn-spawn-fresh branch. Both stay
correct: closing the handle closes the pump.

Cancellation propagation: `pump_task.cancel()` raises CancelledError
inside the pump's `async for ev in handle.events` — the pump's
`finally` block (§8) clears native_futures with a
`RuntimeError("codex exec exited...")`. Any tool body still awaiting
a Future re-raises that error into ToolNode, which produces a
ToolMessage with status="error" — clean shutdown, no orphans.

## 12. What goes away

- `_raise_if_blocked_native_item` — deleted entirely.
- `BLOCKED_NATIVE_ITEM_TYPES` — renamed to `NATIVE_TOOL_ITEM_TYPES`,
  semantics inverted ("recordable native types" not "rejected types").
- Both `_raise_if_blocked_native_item` call sites in
  `CodexChatModel.ainvoke` — replaced by the §6 record-on-started
  logic.
- `TOOL_CONTRACT`'s warning that `apply_patch` "will fail
  the turn" — dropped. apply_patch no longer fails any turn; whether
  the model should prefer deep-worker `write_file`/`edit_file` is now
  a soft preference in the prompt, not a runtime invariant.

The integration test
`tests/integration_ai/test_codex_native_command_execution_crash.py`
is updated: status="error" is no longer the expected outcome. New
assertion: `messages` history contains
`AIMessage(tool_calls=[{name: "command_execution", id: "item_0",
args: {"command": ...}}])` followed by `ToolMessage(name:
"command_execution", tool_call_id: "item_0", content: <real ls
output>)`, and `session_metadata.json.status` is `"waiting_input"`
or `"idle"`.

## 13. Tests (TDD plan)

### 13.1 Unit (`codex_provider/tests/`)

- `test_event_pump.py::test_routes_native_started_to_queue_and_creates_future`
  — scripted events, single Future created on `item.started`,
  event also delivered to ainvoke_queue.
- `test_event_pump.py::test_routes_native_completed_to_future_not_queue`
  — `item.completed` resolves the future, does NOT enqueue.
- `test_event_pump.py::test_proxy_completed_swallowed`
  — McpToolCallItem `item.completed` neither enqueued nor
  future-resolved.
- `test_event_pump.py::test_pump_cancellation_releases_futures`
  — cancel pump → all native_futures end up with a RuntimeError.

- `test_native_tools.py::test_command_execution_returns_on_future_resolution`
  — manually create a Future, set_result while the tool body awaits;
  body returns that string. Verifies event-driven, no polling.
- `test_native_tools.py::test_tool_body_resolves_with_codex_exit_failure`
  — exit_code=2 path; result string includes the marker.

- `test_chat_model.py::TestNativePassthrough::
  test_ainvoke_returns_native_tool_call_on_item_started`
  — feed events: thread.started → item.started(command_execution) →
  (nothing else yet). Assert ainvoke returns AIMessage with one
  tool_call whose name is "command_execution", id matches item.id,
  args contains command.
- `test_chat_model.py::TestNativePassthrough::
  test_ainvoke_mixed_native_and_proxy_in_same_turn`
  — turn yields a native start, then a proxy start; the AIMessage
  carries both tool_calls.
- `test_chat_model.py::TestNativePassthrough::
  test_next_ainvoke_after_native_toolmessage_does_not_resolve_call`
  — feed AIMessage(tool_calls=[native]) → ToolMessage(codex_native=True)
  → assert _proxy_mcp.resolve_call NOT called, saw_native_resume
  path taken, ainvoke continues reading events_queue.
- `test_chat_model.py::TestNativePassthrough::
  test_no_runtime_error_for_native_items`
  — feed full native turn; assert no exception raised.

### 13.2 Integration (live, gated by `CODEX_INTEGRATION_TESTS=1`)

- `tests/integration_ai/test_codex_native_passthrough.py` — spawn real
  codex agent the way the user does
  (`deep-worker test_xxx -sp -tm --provider codex --dev`), ask
  "How many files are in your working directory?". Assertions:
  - agent reaches `status="waiting_input"` (NOT `"error"`),
  - final assistant message contains a plausible count,
  - `dwcli <agent> --history` (via the offline API) shows
    `AIMessage(tool_calls=[{"name": "command_execution", ...}])`
    paired with the matching ToolMessage,
  - `tool_call_id` matches Codex's `item_N` form,
  - no `Blocked native Codex tool event` substring anywhere in
    the agent's error log under its data directory.
- Update `test_codex_native_command_execution_crash.py` to match
  §12 (new expectation: status="waiting_input", real tool_call pair
  in history). Repurpose the test from "must crash" to "must NOT
  crash and must record the native call honestly".

## 14. File layout after the change

```
codex_provider/
├── chat_model.py                 # native record path in ainvoke
├── client_state.py               # unchanged
├── exec_runner.py                # ExecHandle gains pump_task /
│                                 # ainvoke_queue / native_futures;
│                                 # spawn launches pump task;
│                                 # aclose tears it down
├── event_pump.py                 # NEW: _event_pump coroutine and
│                                 # _route_event
├── native_tools.py               # NEW: async BaseTool objects, one
│                                 # per NATIVE_TOOL_ITEM_TYPE, plus
│                                 # _native_args/_native_result_text
│                                 # helpers
├── proxy_mcp_server.py           # unchanged
└── docs/
    ├── DESIGN_CODEX_PROVIDER.md
    └── DESIGN_NATIVE_TOOL_PASSTHROUGH.md     # this document

deep_worker/deep_worker_cli/deep_worker_cli/
├── process/agent_process.py      # appends NATIVE_TOOLS to web_tools
│                                 # when config.provider == "codex"
│                                 # (main subordinate agent)
└── task_manager/task_manager.py  # appends NATIVE_TOOLS to manager_tools
                                  # when config.provider == "codex"
                                  # (-tm manager agent — must be in sync
                                  # with main agent or manager turn dies
                                  # on any native tool call)
```

## 15. Concurrency invariants (event-driven, no polling)

For reviewer convenience, the entire concurrency surface stated
formally:

1. **`events_iter` is read exactly once, by EventPumpTask.** No
   other code path reads it. ainvoke reads `handle.ainvoke_queue`.
2. **`native_futures[id]` is created exactly once, by the pump on
   `item.started`.** It is resolved exactly once, by the pump on
   `item.completed`, OR cancelled exactly once by `aclose`. It is
   `await`ed exactly once, by the matching tool body.
3. **No `asyncio.sleep` in the critical path.** No `while True` with
   timeout-based wait. No polling of any kind. The only `await`s are
   on the asyncio.Queue (built-in event-driven), on the
   asyncio.Future (built-in event-driven), and on `events_iter`
   readline (built-in event-driven via epoll).
4. **Bounded queue (`maxsize=64`).** Prevents unbounded memory growth
   if ainvoke is slow; back-pressures the pump naturally via
   `await queue.put(ev)`.
5. **Pump cancellation is the single shutdown signal.**
   `aclose → pump_task.cancel() → pump finally → futures get
   exception → tool bodies wake with exception → ToolNode emits
   ToolMessage(status="error") → LangGraph state stays consistent.**

## 16. Out of scope

- HITL gating on native tools. Codex with
  `--dangerously-bypass-approvals-and-sandbox` runs everything
  unconditionally; if the user wants approval before a native edit,
  the lever is the bypass flag, not this design.
- Conversion between native names and deep-worker names (e.g. would
  history be cleaner if we renamed `command_execution` → `execute`?
  No — they are semantically different tools, and inconsistent
  naming across sessions would confuse downstream consumers more
  than it helps. Keep Codex's native names verbatim.).
- Replaying a half-recorded turn after a `codex exec` crash. The
  Future is cancelled, ToolMessage is recorded with status="error",
  the turn completes cleanly with that as the final state — no
  replay logic.
