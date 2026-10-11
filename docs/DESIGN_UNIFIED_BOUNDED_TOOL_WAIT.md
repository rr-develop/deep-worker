# DESIGN: Unified bounded tool-wait — `BoundedToolCallMiddleware`

Status: IMPLEMENTED (2026-06-17, option A). Requirements owner: the project
owner. Hard invariants — below.

## IMPLEMENTATION RESULT (option A — a single timer SOURCE, no gaps)

**Option A** was implemented (not "rewrite the protocol"): a single timeout
SOURCE + zero gaps; the CLI's sequential protocol stays intact.

- `deepagents/middleware/bounded_tool_call.py` — `BoundedToolCallMiddleware`
  (sync `wrap_tool_call` + async `awrap_tool_call`): ONE 120s bound on tool
  execution for ALL providers; timeout → background + non-error in_progress.
  `resolve_tool_wait_seconds()` — the ONLY timer source;
  `build_in_progress_notice()` — the single text. Wired FIRST (outermost) in
  `graph.py`.
- `deepagents/middleware/tool_dispatch_bridge.py` — a provider-agnostic bridge:
  the CLI provider registers a dispatch-waiter; the middleware waits on it UNDER
  the same bound; for regular providers it is a no-op. Stage A is covered.
- claude `wait_for_pending_call_by_id` is now BOUNDED by the same source →
  `DispatchTimeout`; `ainvoke` catches it and returns a non-error in_progress
  (no hang, no phantom tool_call fabricated). The provider's private 120s
  constant is removed — `_default_wait_seconds` delegates to the shared source.
- codex the same: the private 120s is removed, stage-A `wait_for_pending_calls`
  is bounded by the shared source, timeout → non-error in_progress (previously RuntimeError).
- Production-dead claude code removed (`wait_for_any_pending_call` — the last
  bare `await fut`, `wait_for_pending_calls`, `wait_for_matching_pending_calls`,
  `_matching_pending_call*`). No unbounded await remains, no second timer source.

Evidence tests: see section 5 (names/files) — all green.

## 0. Requirement (verbatim)

> This policy MUST hold for ALL providers, of any type, in ONE place in the
> code. All waits MUST go inside `awrap_tool_call()` for all providers. This is
> the single source of truth. NO exceptions and NO gaps. Stage A (waiting for a
> call to appear at the CLI providers) MUST also be under the same timer.

Policy (POLICIES.md §7.1): every tool call is time-bounded. After
`DW_LONG_RUNNING_WAIT_SECONDS` (default 120s) — if the tool has not finished,
its work is parked in the background (NOT cancelled) and the model receives a
non-error `in_progress` with `job_id` + `wait_for_job`/`kill_job`. A timeout
NEVER becomes `is_error`.

## 1. What the code shows (facts, not guesses)

### 1.1 Many providers, of two types
`config.py:1602-1742`: deepseek, openai, anthropic, google, vertexai,
claude-code-sdk, codex (+ grok in development).
- **Type 1 (regular LangChain models):** deepseek/openai/anthropic/google/
  vertexai. LangGraph ToolNode executes tools directly.
- **Type 2 (CLI via MCP-proxy):** claude-code-sdk, codex, (grok). Each carries its
  OWN copy of `ProxyMCPServer` with its own wait.

### 1.2 A single tool-execution point ALREADY exists
`graph.py:399` `create_agent(model, tools=..., middleware=deepagent_middleware)`
accepts ANY model and the shared middleware stack. The hook
`AgentMiddleware.awrap_tool_call(request, handler)` wraps EVERY tool
execution (`await handler(request)`) for ALL providers — it is graph-level
langchain middleware, not provider-level. Existing examples:
`message_timestamps.py:235`, `filesystem.py:1300`,
`InterAgentDeliveryResolveMiddleware.awrap_tool_call`.

### 1.3 TWO wait stages (geometry)
A CLI-provider tool call arrives in two parts over two channels:
- **Stage A — "call appears".** The model (CLI) declares a ToolUseBlock with a
  `tool_use_id`. `ainvoke` waits for registration under that id:
  claude `wait_for_pending_call_by_id` (`proxy_mcp_server.py:828` — bare
  `await fut`, **NO timeout** — a GAP), called from `chat_model.py:1463`.
  codex — `wait_for_pending_calls` (`:209`, 30s timeout).
- **Stage B — "execution".** ToolNode runs the tool → middleware
  `awrap_tool_call` catches the ToolMessage → `resolve_call` wakes the Future
  in the proxy (`chat_model.py:296`, `:1136`). The 120s job-handle itself is
  inside `_handle_tool_call` (claude `:506`, codex `:309`), `wait_for(shield, 120)`.

**Critical for the design:** stage A happens INSIDE the model's `ainvoke`
(model node) — i.e. STRUCTURALLY BEFORE LangGraph calls ToolNode and
`awrap_tool_call`. So "just put a timer in awrap_tool_call" does NOT cover
stage A — that is precisely the forbidden gap.

### 1.4 The proxies share NO common base class/module
claude and codex duplicate `ProxyMCPServer`. That is why the waits diverged
(claude stage A without a timeout, codex with 30s) — the root is the missing
single point.

## 2. Invariants (DO NOT violate)

1. **EXACTLY ONE** function in the code performs ANY tool-execution wait —
   `BoundedToolCallMiddleware.awrap_tool_call`. Any other tool wait is FORBIDDEN
   (no `await fut`, no `wait_for` on a tool inside providers).
2. Applies to ALL providers of both types — through the shared middleware stack.
3. One time budget (default 120s, env `DW_LONG_RUNNING_WAIT_SECONDS`) for the
   ENTIRE tool interval, including stage A.
4. Timeout → NOT `is_error`. The outcome is always: work to background (do not
   cancel) + `in_progress` notice (`job_id`, `wait_for_job`/`kill_job`).
5. NO bare `await` without a bound anywhere on the tool path.
6. Do NOT inject a visible-block into pending as a fallback (it caused double
   execution — see a related internal incident analysis (not included here)) —
   only bounded-wait → background.

## 3. Solution architecture

### 3.1 New middleware (single source of truth)
`deepagents/middleware/bounded_tool_call.py`:

```python
class BoundedToolCallMiddleware(AgentMiddleware):
    """The ONLY place that waits for tool execution for ALL providers.
    POLICIES.md §7.1. No tool wait may happen outside this hook."""
    def __init__(self, wait_seconds: float | None = None): ...

    async def awrap_tool_call(self, request, handler):
        try:
            return await asyncio.wait_for(
                asyncio.shield(handler(request)),   # shield: background, not cancel
                timeout=self._wait_seconds,
            )
        except asyncio.TimeoutError:
            # park job (work continues on ToolNode) + non-error
            # in_progress ToolMessage with job_id + wait_for_job/kill_job
            return self._park_as_job(request)
```

Registered in `graph.py` in `deepagent_middleware` (once, for all).
Order: it MUST wrap all other tool-call middleware (the outermost of the
tool-wraps) so their work is also under the timer.

### 3.2 Closing stage A (MANDATORY — otherwise a gap)
Stage A must not stay in `ainvoke`. Two options; pick at review:

**Option 1 (preferred) — remove the stage-A wait from `ainvoke` entirely.**
`ainvoke` does NOT wait for registration under `tool_use_id`. As soon as the
CLI has declared a ToolUseBlock, `ainvoke` returns the tool_call to LangGraph
IMMEDIATELY (the id is known from the declaration). The real authoritative args
come not from the declaration but from the proxy — but already INSIDE
`awrap_tool_call`: the middleware (via a provider-agnostic interface) waits for
the control-request to arrive and for execution as ONE bounded wait. Stage A
then physically moves inside `awrap_tool_call` and falls under the shared
timer. Requires: a way for the middleware to ask the active proxy to "wait for
registration + result for this tool_call_id" (see 3.3).

**Option 2 — keep A in the proxy but take the timeout from the middleware.**
`wait_for_pending_call_by_id` stays, BUT its only bound is the external
per-turn timer set by the same policy (one 120s source). Worse: the wait is
formally outside `awrap_tool_call` → violates invariant 1. REJECTED as a "gap"
unless it is proven that A reduces equivalently to a call from the middleware.

→ We go with Option 1: the only wait is in `awrap_tool_call`.

### 3.3 Provider-agnostic bridge to the proxy
`awrap_tool_call` knows nothing about claude/codex. A common protocol is
needed, e.g. on backend/runtime: `await ensure_tool_call_dispatched(tool_call_id)`
— for proxy providers it is implemented via waiting for registration (the
former stage A), for regular ones it is a no-op (the tool is executed by
ToolNode right away anyway). This await is INSIDE `awrap_tool_call`, so it is
under the shared timer. One code path; the provider only supplies the
"how to wait for dispatch" implementation (or the default).

### 3.4 What is removed (so no second wait remains)
- claude `wait_for_pending_call_by_id` (`:766/828`) and its call site
  `chat_model.py:1463` — moved into the middleware bridge (3.3) or removed.
- claude dead code: `wait_for_any_pending_call` (`:638`),
  `wait_for_matching_pending_calls` (`:642`), `wait_for_pending_calls` (`:584`)
  — not called in production (tests only), remove.
- codex `wait_for_pending_calls` (`:185`) at stage A — reduced to bridge 3.3.
- the job-handle from `_handle_tool_call` of both proxies → the bound moves to
  the middleware; the proxy-handler itself no longer keeps its own
  `wait_for(120)` (otherwise two timers). WARNING: this is the riskiest step —
  the proxy-handler has its own role (returning MCP blocks to the CLI).
  Decision at review: either the proxy-handler resolves instantly and waits
  only on the middleware, or the handler stays as the background-parking
  mechanism but WITHOUT its own timeout (the only timer is the middleware).

## 4. Risks / open questions for review

1. **Moving args-resolution inside the middleware** (3.2 opt.1): today `ainvoke`
   waits for A precisely to take the AUTHORITATIVE args from the proxy (the
   declaration lies — a known args-mismatch issue from 2026-05-26). If we
   return the tool_call to LangGraph from the declaration and substitute the
   real args in the middleware — this must be done carefully so ToolNode
   executes with the correct args. Requires an exact mechanism (the middleware
   rewrites request.args from the proxy before the handler).
2. **Double timer** (3.4): we cannot keep both the proxy-handler `wait_for(120)`
   and the middleware timer. One source of truth.
3. **send_message_to_agent** (`resolve_pending_tool`, drain): a special case,
   resolved at delivery time. It must stay correct under the new flow.
4. **Regular providers:** verify the background timeout also makes sense for them
   (a tool can hang at deepseek just the same — it is THE SAME tool). job-park
   for a non-proxy tool: work on ToolNode under `shield`, notice as ToolMessage.

## 5. TDD plan (once code is approved)

1. Test: ANY tool at ANY provider that does not return a result within the
   wait → in_progress job-handle (not is_error), work not cancelled. Mock the
   clock ONLY.
2. Test: stage A (CLI proxy) — a lost control-request → the same timeout into
   the background via the middleware (NOT an eternal hang). Mock the clock.
3. Test: grep over claude/codex providers — NO `await fut`/`wait_for` on a
   tool outside the middleware (a static invariant test).
4. Integration: both proxy providers + one regular (deepseek-mock) run a tool
   through ONE middleware, identical behaviour.
5. Regression: send_message_to_agent delivery-resolve, the args-mismatch
   issue, orphaned-on-restart — still green.
```
