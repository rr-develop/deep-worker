# Design: web resubscribe on agent restart + delivery-only inter-agent timeout

Two independent bugs surfaced from an inter-agent code-review session
(2026-05-24). Both are fixed here. Bug 1 lives in the web app; Bug 2 lives
in `claude_code_provider` + the `deep_worker_cli` middleware.

---

## Bug 1 — "agent visible in web UI but its events never arrive"

### Observed

The agent was alive and processing, but the web UI showed it as inactive: no
status/message events. The user restarted it repeatedly because of this. The
agent's own log showed it restarted ≥8 times (each `[AGENT_INIT]` on a *new*
RPyC port), and the web peer kept hitting
`initial status snapshot push failed: CancelledError`.

### Root cause (verified in code)

The UI list and the event subscription are decoupled:

* `AgentService.list_agents()` (`agent_service.py:315`) returns `self._agents`,
  populated purely from the **registry** (`_discover_agents`, :260-282). An
  agent is *visible* whenever it is in the registry — independent of whether we
  hold a live RPyC subscription.
* The subscription is established lazily in `connect_to_agent()` →
  `_subscribe_to_agent()`, only when some route touches the agent.
* `connect_to_agent()` has a fast-path `if await self.is_connected(name): return True`
  (`:431`). `is_connected()` (`:357-370`) treats a connection as live unless
  rpyc's local flag `conn._closed is True`. That flag is only set on a
  **graceful** teardown. When an agent process is killed and respawned
  (ungraceful, new PID, new port), the old socket's `_closed` can stay `False`,
  so `is_connected()` lies "connected" and `connect_to_agent()` returns early —
  **never reconnecting to the new port, never resubscribing.**
* There is **no persistent registry watcher**. The inotify in
  `lifecycle._wait_registered()` is one-shot (waits for an agent to *appear*
  during a web-initiated spawn), so a restart of an already-known agent is not
  noticed event-driven.

Net: after an ungraceful restart the web holds a dead connection to the old
process; the agent is still listed (registry), so it looks alive but is silent.

### Invariant we are enforcing

**If an agent is visible in the UI, the web is subscribed to its *current*
process.** "Current" is keyed by process identity `(pid, rpyc_port)`, not by
rpyc's `_closed` flag.

### Fix (event-driven, no polling, no extra threads)

1. **Identity-aware connection tracking.** Record the `(pid, rpyc_port)` the
   live subscription was established against (a new `self._conn_identity[name]`
   dict). `is_connected(name)` returns `True` only if we hold a connection AND
   the registry's current `(pid, rpyc_port)` for `name` still matches that
   identity (and `_closed` is not True). On mismatch → the old process is gone:
   evict (`_connections`/`_roots`/`_callback_services`/`_conn_identity`), best-
   effort `aclose()` the stale conn, return `False`. The next
   `connect_to_agent()` then does a clean connect+subscribe to the new port.

2. **Persistent registry watcher.** A long-lived inotify reader on the registry
   directory (`/tmp`, same mechanism already used in `lifecycle.py:894-966`),
   wired via `loop.add_reader` (no `asyncio.sleep`, no thread). On every
   registry change: re-discover into `self._agents`; for every agent whose
   `(pid, rpyc_port)` changed vs. what we last subscribed against, evict the
   stale connection and reconnect+resubscribe; for agents that vanished, evict.
   This makes "visible ⇒ subscribed" hold continuously, not just on the next
   user action.

3. **Subscription as a consequence of discovery.** After (re)discovery seeds an
   agent into `self._agents`, ensure-subscribe it (reuse `connect_to_agent`,
   which is idempotent once identity tracking is correct). So becoming visible
   drives subscription rather than a separate lazy path.

The agent-side `initial status snapshot push failed: CancelledError`
(`agent_service.py:682`) is a *symptom* of delivering to an already-dead web
netref; it disappears once the web stops holding dead subscriptions. No
agent-side change required, but re-verify after the web fix.

### TDD (unit, MockProcessRegistry — no real agents)

* `is_connected` returns `False` when the registry `(pid,port)` for the agent
  differs from the identity the held connection was opened against (restart),
  and evicts the stale connection.
* `is_connected` still returns `True` when identity matches and conn not closed.
* `connect_to_agent` after a simulated restart opens a NEW connection to the new
  port and resubscribes (old callback evicted, new `subscribe_messages` called).
* registry-change handler: agent with changed `(pid,port)` → evict + reconnect;
  vanished agent → evict; unchanged agent → no churn (no duplicate subscribe).
* invariant: an agent present in `list_agents()` ends up with a live
  subscription after the discovery/watch path runs.

---

## Bug 2 — inter-agent send "times out" although delivered

### Observed

Every `send_message_to_agent` from the sending agent (claude-code) was **delivered**
(the recipient replied with real reviews), yet Claude reported "The send timed out".
Measured gap between Claude emitting the tool call and our code returning the
tool result: **121–268 s every time** (≥120 s). The MCP handler default
timeout is 120 s (`proxy_mcp_server.py:71,131`).

### Root cause (verified in code + timestamps)

proxy-MCP architecture: the MCP tool handler does NOT execute the tool — it
creates an `asyncio.Future`, registers it in `pending_calls`, and blocks on
`await asyncio.wait_for(future, timeout=120)` (`proxy_mcp_server.py:120-144`).
The tool runs via LangGraph ToolNode; the Future is resolved later, in the
**send-loop of the *next* `ChatModel.ainvoke`**, when it sees the resulting
`ToolMessage` and calls `proxy_mcp.resolve_call(tool_call_id, ...)`
(`chat_model.py:695-706`).

For a normal tool that next `ainvoke` is part of the **same graph run**
(model→tools→model loop) — resolve fires within milliseconds. But
`SendMessageCompletionMiddleware.before_model` returns `{"jump_to": "end"}` on a
successful `send_message_to_agent` `ToolMessage`
(`send_message_completion.py:139-172`). `before_model` runs **before** the model
node, so the graph terminates *before* the next `ainvoke` — the resolve is
skipped. It only fires on the **next external impulse** (a new human/agent
message), 120-270 s later. Claude CLI's internal wait for the MCP tool_result
expires first → "send timed out".

Crucially: the tool itself is fire-and-forget. `send_message_to_agent`
(`tools.py:539-577`) does `connect()` + `enqueue_message` and returns
`{"success": true, "delivered_content": ...}` immediately; the server's
`exposed_enqueue_message` schedules the recipient's turn with
`asyncio.create_task` and returns at once (`agent_service.py:541-546`). So the
**delivery result is fully known the instant ToolNode produces the
ToolMessage** — there is no reason to wait for the recipient to finish.

### Principle

Delivery time and recipient-reply time are independent. The MCP Future for an
inter-agent send must resolve **at delivery confirmation** (enqueue success),
which is near-instant. The 120 s MCP timeout then only bounds *delivery* — i.e.
it detects a genuinely dead/unreachable channel — never the recipient's work
(which legitimately takes minutes). Event-driven; no polling, no extra threads.

### Layering constraint

`SendMessageCompletionMiddleware` is provider-agnostic (applied to codex /
deepseek / claude-code alike, `agent.py:672`); only claude-code has a
proxy-MCP. So the resolve must NOT be hard-wired into the shared middleware
reaching into claude-code internals.

### Fix — resolve at delivery, inside the provider, decoupled from the round-trip

The send-loop already resolves a `ToolMessage` the moment it appears in
`messages[sent_count:]` (`chat_model.py:695-706`). The only failure is that for
a turn-ending tool the `ToolMessage` never reaches a subsequent `ainvoke`. We
close that gap **without** changing tool semantics or layering:

* **Provider exposes a public, idempotent resolve** keyed by `thread_id`:
  `ClaudeCodeChatModel.resolve_pending_tool(thread_id, tool_call_id, content)`,
  delegating to that thread's `proxy_mcp.resolve_call(...)`. `resolve_call` is
  already idempotent (`proxy_mcp_server.py:278-281`: guarded by
  `call_id in pending_calls` and `not future.done()`), and the handler's
  `finally` pops the key (`:144`) — so an early resolve here plus any later
  send-loop resolve cannot double-fire.

* **A claude-code-only delivery-resolve middleware** performs the resolve at
  delivery, using the `awrap_tool_call` seam.

  Chosen seam (verified against `langchain.agents.middleware.types`):
  `AgentMiddleware.awrap_tool_call(self, request: ToolCallRequest, handler)`.
  The handler executes the tool and returns the `ToolMessage`; this runs
  **inside the tools node, before control returns to the model node** — i.e.
  before `SendMessageCompletionMiddleware.before_model` jumps to end. So
  resolving here happens on time, in the same graph run.

  `ToolCallRequest` exposes `tool_call` (has `name` + `id`), `state`, and
  `runtime` (LangGraph runtime → `thread_id` via its config). It does NOT
  expose the model, so the middleware resolves via the provider's own singleton
  `claude_code_provider.get_global_client_manager()` — which is legitimate
  *because this middleware itself lives in / is gated to claude-code*.

  Flow inside `awrap_tool_call`:
  1. `result = await handler(request)` — run the tool, get the `ToolMessage`.
  2. If `request.tool_call["name"] == "send_message_to_agent"` and the result
     parses as `{"success": true, ...}`: resolve the proxy Future for this
     thread immediately — `proxy.resolve_call(request.tool_call["id"], {...})`.
  3. Return `result` unchanged so the graph proceeds (echo + jump as before).

  Failure (`success: false`) → do NOT resolve early; let the normal path
  surface the error to the model.

  Registration: `create_cli_agent` (`agent.py:518`) already receives `model`.
  Add this middleware to the stack ONLY when
  `isinstance(model, ClaudeCodeChatModel)`. The shared
  `SendMessageCompletionMiddleware` is left untouched and provider-agnostic;
  all proxy-MCP knowledge stays in the claude-code layer.

  `ClaudeCodeChatModel` also gains a small public, idempotent
  `resolve_pending_tool(thread_id, tool_call_id, content)` so the resolve is a
  supported API rather than reaching into privates; the middleware may call it
  (preferred) or go through the client manager directly. Both are equivalent
  given `resolve_call`'s idempotency guard.

* **Delivery-only timeout semantics.** With the Future resolved at delivery, the
  120 s handler timeout (`proxy_mcp_server.py:131`) now fires ONLY when delivery
  itself didn't confirm in time — i.e. the target channel is dead. Keep it as
  the dead-channel detector; it no longer races the recipient's work.

### Correctness notes

* After resolve the Claude CLI receives the tool_result and finishes its turn
  (emits `ResultMessage`); since the graph jumped to end, nothing awaits it —
  identical to today's behavior, just on time.
* No double resolve (idempotent guard + `pending_calls.pop`).
* Tool semantics unchanged: still fire-and-forget; we only stop *deferring* the
  resolve of an already-known result.

### TDD

* proxy: `resolve_call` is idempotent (second call / after `done()` is a no-op) —
  lock in the guarantee the fix relies on.
* provider: `resolve_pending_tool(thread_id, id, content)` resolves the matching
  pending Future; unknown thread/id is a safe no-op.
* completion path: a successful `send_message_to_agent` ToolMessage resolves the
  proxy Future **before** the turn-ending jump (Future done immediately, not on
  the next ainvoke); failure ToolMessage does NOT resolve (model must still see
  the error).
* non-claude-code provider (no `resolve_pending_tool`): the completion path is a
  no-op and does not raise.
* timeout semantics: when delivery confirms, the handler returns the delivered
  result well under 120 s; only a never-confirmed delivery hits the timeout
  (dead channel).
