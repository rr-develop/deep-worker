## DESIGN: Tool-name priority resolution + closing dangling tool-calls on restart

**Status:** design (no code changed). Branch: `dev`.
**Author context:** investigation of an agent stuck in `error`.

---

## 1. The two bugs (root causes, verified against code + a live agent log)

### Bug A — tool-name collision between base/proxy tools and Codex native passthrough

For a Codex-provider agent (and the TaskManager of a Codex subordinate) the
bound tool set contains BOTH:

- base / proxy-MCP deep-worker tools (`read_file`, `execute`, `web_search` via
  `web_search_delegation_tool`, …), AND
- the Codex native passthrough `BaseTool`s from
  `codex_provider/native_tools.py` (`command_execution`, `apply_patch`,
  `web_search`, `file_change`, `tool_search`, `custom_tool_call`,
  `function_call`, `function_call_output`).

They are appended in `deep_worker_cli/process/agent_process.py:785-787`
(`web_tools.extend(NATIVE_TOOLS)`) and symmetrically in
`deep_worker_cli/task_manager/task_manager.py:280-282`.

`langgraph.prebuilt.ToolNode.__init__` builds `self._tools_by_name[tool.name] =
tool` in iteration order, so **last registration wins**. `web_search` exists in
both lists → the native passthrough silently overrides the proxy tool. This is
already documented as a known collision in
`codex_provider/docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md §10.1`.

The collision is non-deterministic w.r.t. intent: which implementation answers
a `tool_call.name="web_search"` depends purely on list order, not on which
mechanism is actually able to service the call.

### Bug B — dangling tool-calls from a prior session are re-invoked on restart (CRITICAL)

An agent is stuck in `error`. The fatal exception (real traceback, not
quoted log text) is:

```
RuntimeError: No pending native call registered for id='apply_patch_136_136716975835008'.
  ... codex_provider/native_tools.py:135 apply_patch
  ... codex_provider/native_tools.py:70 _native_future_for -> raise
```

The id `apply_patch_136_136716975835008` is NOT a fresh Codex `item.id` (Codex
emits `item_N`). It has the proxy-MCP shape `f"{tool_name}_{counter}_{id(args)}"`
(`codex_provider/proxy_mcp_server.py:199`, identical in
`claude_code_provider/proxy_mcp_server.py`). It is a tool-call id **persisted in
the LangGraph checkpoint** from an earlier turn whose `ToolMessage` was never
produced (the turn was interrupted — e.g. a Claude CLI subprocess hung and was
killed, per the persisted `error_message`).

On the next invocation after restart the agent is called with a NEW
`HumanMessage` (`agent_process.py:1800-1801`,
`{"messages": [{"role":"user", ...}]}`), so LangGraph appends it to the
checkpointed history and re-enters the graph at `START`. The checkpointed
history now contains an `AIMessage` with `tool_calls=[…]` that has **no matching
`ToolMessage`**. LangGraph routes the unresolved tool-call into `ToolNode`,
which dispatches by name into the bound tool body. The body looks up its future
and finds nothing → raises → `_handle_invoke` calls `set_error` → `status=error`,
forever (the poison message stays in history, so every subsequent turn re-hits
it; the log shows the same ids 22+ times).

**This is the dangerous behaviour the user flagged:** an interrupted tool call
from a previous session is *re-executed* on restart. Re-executing a tool has
uncontrolled side effects and must never happen implicitly.

---

## 2. Goals & constraints

- Bug A: make tool-name resolution **deterministic** with a single rule —
  **base/proxy tools take priority; native passthrough is only a fallback for
  names not already provided**. No two-list manual sync.
- Bug B: **never re-invoke** a tool call carried over from a prior session.
  Instead, **close** each dangling tool-call with a synthetic `ToolMessage`
  that tells the agent the call was issued last session but never confirmed
  completed, and let the agent decide what to do next.
- Event-driven only; no polling; no new threads; no blocking. (Both fixes are
  pure in-process synchronous list/state transforms — they add neither.)
- No architecture/contract changes: the agent is still the single source of
  truth; tool calls still flow through `ToolNode`; the checkpoint schema is
  unchanged (we only append normal `ToolMessage`s).
- Provider-agnostic where it can be: Bug B affects ANY provider (the same
  `{name}_{counter}_{id(args)}` ids exist in claude-code too), so its fix lives
  in the shared `deep_worker_cli` layer, not inside a provider.

---

## 3. Fix A — priority de-duplication of the bound tool set

### Where
`deep_worker_cli/agent.py::create_cli_agent`, at the single point where the
final `tools` sequence is assembled before `create_deep_agent(tools=tools, …)`
(`agent.py:865-875`). This is the one funnel through which BOTH the main agent
and the TaskManager pass (the TaskManager calls `create_cli_agent`), so fixing
it here covers both `extend(NATIVE_TOOLS)` sites.

### What
Add a small, pure helper `_dedupe_tools_by_name_priority(tools)` that walks the
sequence once and keeps the **first** occurrence of each tool name, dropping any
later duplicate. Because base/proxy tools are assembled first and
`NATIVE_TOOLS` is appended after, "keep first" == "base/proxy wins, native is
fallback for new names only" — exactly the user's rule. The helper is
order-preserving and provider-agnostic (it only looks at `.name`).

```
seen: set[str] = set()
out = []
for t in tools:
    name = getattr(t, "name", None)
    if name is not None and name in seen:
        continue            # a higher-priority tool already owns this name
    if name is not None:
        seen.add(name)
    out.append(t)
return out
```

Apply it right before `create_deep_agent`:
`tools = _dedupe_tools_by_name_priority(tools)`.

### Why not reorder / why not edit the two extend sites
- "Keep first" is one deterministic rule applied at one funnel; it cannot drift
  out of sync the way two curated allow/deny lists would (the user's explicit
  concern).
- It does not require knowing *which* names overlap; new future overlaps are
  handled automatically.
- It does not touch `NATIVE_TOOL_ITEM_TYPES` or the event-pump/ainvoke routing:
  Codex still emits native `item.started`, the pump still parks the future; only
  the *LangGraph-visible BaseTool* that services a colliding name is now the
  base/proxy one. For genuinely-native-only names (`apply_patch`,
  `command_execution`, …) nothing changes — they are kept (first and only
  occurrence).

### Note on the Codex-native path for non-colliding names
Native passthrough tools whose names do NOT collide (e.g. `apply_patch`) remain
bound and keep working exactly as today for *live* Codex turns (the pump parks
`native_futures[item.id]`, the body awaits it). Fix A only removes the
*duplicate* registrations (currently just `web_search`). Bug B (below) is what
prevents the *stale* `apply_patch` call from crashing.

---

## 4. Fix B — close dangling tool-calls before the model, never re-invoke

### Seam (verified)
`langchain.agents.create_agent` wires `START → entry_node` where `entry_node`
is the **first `before_model` middleware** (when any exists), and tools loop
back to that same node — never directly to `ToolNode` on a fresh super-step.
The agent is always invoked with a NEW `HumanMessage`
(`agent_process.py:1800`), i.e. a fresh super-step from `START`, so a
`before_model` middleware runs **before** the model and **before** any tool
execution. This is the same hook `SendMessageCompletionMiddleware` already uses
(`deep_worker_cli/middleware/send_message_completion.py`).

### New middleware `DanglingToolCallMiddleware` (provider-agnostic)
File: `deep_worker_cli/middleware/dangling_tool_calls.py`.

`before_model(state)`:
1. Read `messages = state["messages"]` (defensive `.get`, like the existing
   middleware).
2. Compute the set of `tool_call_id`s that already have a `ToolMessage`
   (`{m.tool_call_id for m in messages if isinstance(m, ToolMessage)}`).
3. Walk messages; for each `AIMessage` with `tool_calls`, collect any
   `tool_call["id"]` that is NOT in the resolved set → these are dangling.
4. If there are none → return `None` (normal flow; zero overhead on the common
   path).
5. If there are dangling ids → return `{"messages": [ToolMessage(...) for each
   dangling id]}` — one synthetic `ToolMessage` per dangling call:
   - `tool_call_id` = the dangling id,
   - `name` = the dangling call's tool name,
   - `content` = a clear, honest **warning** notice (user decision Q1), e.g.:
     `"[deep-worker] WARNING: this tool call ('<name>') was issued in a
     PREVIOUS session that was interrupted (the agent process was likely
     killed) WHILE the tool was being invoked. Its outcome is UNKNOWN — it may
     have completed, partially run, or not run at all — and it was NOT
     re-executed on restart (re-running tools automatically is unsafe). A NEW
     session has now started. Decide for yourself whether to inspect any
     resulting state and/or call the tool again."`
   - **Status / "warning" semantics (Q1).** The user asked for a *warning*
     status: we do not know whether the tool ran, so the closer must NOT be
     presented as a real successful result, but it is also not a normal tool
     error. `langchain_core.messages.ToolMessage.status` is typed
     `Literal["success", "error"]` — there is NO `"warning"` literal. We
     therefore encode "warning" as:
       - `status="error"` (so no consumer mistakes it for a genuine successful
         tool result), PLUS
       - `additional_kwargs={"dangling_tool_call": True,
         "warning_kind": "outcome_unknown"}` — a machine-readable marker
         distinguishing this synthetic "previous-session, outcome-unknown"
         closer from a real tool execution error. The extra
         `warning_kind` key (review suggestion) makes the
         warning *type* explicit and extensible for future warning kinds
         (e.g. `"tool_disabled"`, `"rate_limited"`) without a schema change,
         PLUS
       - the explicit warning wording above so the model reads it as
         "unknown outcome from a prior session", not as "the tool failed now".
     If a future langchain version adds a real `"warning"` status we switch to
     it; until then this is the faithful representation of the user's intent.

   We do NOT set `jump_to`; after injecting the closers we let the model run
   normally so it can react to the new `HumanMessage` with full, valid history.

### Why this is correct and safe
- **No re-invocation.** By appending the `ToolMessage` closers in
  `before_model`, every previously-dangling `AIMessage.tool_call` now has a
  matching `ToolMessage`, so LangGraph's routing has nothing unresolved to send
  into `ToolNode`. The native/proxy tool body is never entered for a stale id →
  the `apply_patch_136_…` crash cannot occur.
- **Valid history for the model API.** Anthropic/OpenAI/Codex all require every
  assistant tool-use to be followed by a tool-result. Synthesizing the closers
  also makes the replayed history API-valid (otherwise the next model call
  could itself error).
- **Agent decides.** The honest warning wording (`status="error"` +
  `dangling_tool_call` marker) means the model is informed that the
  call came from a prior, interrupted session with unknown outcome — not
  deceived with a fake result. It chooses whether to inspect state / redo the
  work.
- **Idempotent.** Once closers are appended they become real `ToolMessage`s in
  the checkpoint; on the next turn step 2 sees them as resolved, so step 4
  returns `None`. No repeated injection.
- **Provider-agnostic & event-driven.** Pure synchronous state inspection; no
  I/O, no polling, no thread, no provider internals.

### Ordering in the middleware stack
`DanglingToolCallMiddleware` must be the FIRST `before_model` middleware so it is
the graph `entry_node` and closes dangling calls before any other `before_model`
logic (e.g. `SendMessageCompletionMiddleware`, memory, skills) inspects the tail.
It will be inserted at the front of `agent_middleware` in `create_cli_agent`
(before `SendMessageCompletionMiddleware.append`).

### Interaction with the live mid-turn path (NOT affected)
During a *live* turn (same process), a real tool-call `AIMessage` is immediately
followed by `ToolNode` producing its `ToolMessage` within the same super-step;
`before_model` only runs at the start of a super-step, when the tail is already
resolved (or is a fresh HumanMessage). So the middleware only ever fires for
*genuinely* carried-over, unresolved calls from a previous process/session — not
for in-flight calls.

---

## 5. TDD plan

Unit tests (no mocks of the units under test):

Fix A — `_dedupe_tools_by_name_priority`:
- duplicate name → first kept, later dropped; order preserved.
- no duplicates → list unchanged.
- a real collision case: `[web_search_proxy, …, web_search_native]` → the proxy
  (first) survives, native dropped; non-colliding native tools (`apply_patch`)
  retained.
- tools without `.name` are passed through untouched.

Fix B — `DanglingToolCallMiddleware.before_model`:
- AIMessage with a tool_call that HAS a matching ToolMessage → returns `None`.
- AIMessage with a tool_call that has NO matching ToolMessage → returns one
  ToolMessage closer with the right `tool_call_id`, `name`, `status="error"`.
- multiple dangling calls (incl. multiple in one AIMessage) → one closer each.
- empty / no-AIMessage / fresh-HumanMessage-only state → `None`.
- idempotency: feeding the output state back → `None` (no re-injection).
- a tail that is a fresh HumanMessage after a *resolved* prior turn → `None`.

Integration test (NO mocks) — see §6.

---

## 6. Integration test (no mocks)

A real end-to-end regression test that reproduces Bug B and proves the fix,
without mocking the graph or the provider:

- Build a REAL agent via `create_cli_agent` with a real in-memory checkpointer
  and a trivial real tool, OR drive the public middleware stack against a real
  `create_deep_agent` graph.
- Seed the checkpoint with a history whose tail is an `AIMessage` carrying a
  tool_call id of the poison shape (`apply_patch_…_…`) and NO matching
  ToolMessage (simulating an interrupted prior session).
- Invoke the agent with a new HumanMessage.
- Assert: the run does NOT raise / does NOT end in error, the dangling call was
  closed by a synthetic ToolMessage (visible in `result["messages"]`), and the
  poison tool body was never executed.
- Place under `deep_worker/deep_worker_cli/tests/integration_ai/` following the
  existing live-test conventions; gate any provider-specific live variant behind
  the existing env flag, but the core graph-level test must run hermetically
  (real graph, real checkpointer, real middleware — just no network/CLI).

---

## 7. Files touched (planned)

- `deep_worker_cli/agent.py` — add `_dedupe_tools_by_name_priority`, apply
  before `create_deep_agent`; prepend `DanglingToolCallMiddleware` to the
  middleware stack.
- NEW `deep_worker_cli/middleware/dangling_tool_calls.py` —
  `DanglingToolCallMiddleware`.
- NEW `deep_worker_cli/tests/unit_tests/...` — unit tests for both units.
- NEW `deep_worker_cli/tests/integration_ai/...` — no-mock regression test.
- docs: this file; cross-reference from
  `codex_provider/docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md §10.1` (the documented
  collision is now resolved by Fix A).

No changes to provider internals, checkpoint schema, RPyC contracts, or graph
topology.

---

## 8. Open questions for the user — RESOLVED (2026-05-28)

1. **Closer `ToolMessage` status → WARNING semantics.** The user wants a
   *warning*: we do not know whether the tool ran (the prior session was likely
   killed mid-call), so the agent must understand it was a call from a PREVIOUS
   session whose outcome is unknown, and decide for itself (possibly inspect
   results and/or re-call). Since `ToolMessage.status` only allows
   `"success"`/`"error"`, we encode warning as `status="error"` +
   `additional_kwargs={"dangling_tool_call": True}` + explicit
   warning wording (see §4.5). Not presented as a success, not a real error.
2. **Fix A priority via assembly order → CONFIRMED.** Keep the FIRST tool by
   name; base/proxy are assembled first so they win, native is fallback. No
   explicit per-tool priority tag.
3. **Collisions → BASE tool wins.** On any name collision the base tool wins
   (current simple "keep first" rule). Explicit priorities / changing this may
   come later, but for now base wins. Documented as a known caveat.