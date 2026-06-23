# Design: restoring the ProxyMCPServer proxy architecture

## Update 2026-05-16

The proxy architecture remains the same, but the sleep-based grace wait is
no longer used. The implementation switched to event-driven waiting via
`asyncio.Future` (see `_pending_change_waiters` in `ProxyMCPServer`),
which is woken from `_wake_matching_pending_waiters` right after
`pending_calls[call_id]` is created. No `await asyncio.sleep(0)`, no
polling. See the Update 2026-05-17 below — the batch-semantics wording of
that commit turned out to be wrong (it deadlocked with multiple parallel
ToolUseBlocks) and was fixed.

## Update 2026-05-17 — Fix: `mcp__deepworker__` prefix + sequential MCP dispatch

### Symptom
A development agent (claude-code provider) hung after several user
messages. In the agent's error log (`<agent>.err.log`), the `ainvoke`
loop showed that for each proxy `ToolUseBlock` the
`Built-in tool — continuing receive_response` branch fired, after which
`receive_response()` either hung or returned only ONE `tool_call` of
several to LangGraph. Unresolved MCP handler tasks stayed blocked on an
`asyncio.Future`, the Claude CLI waited for tool_result for the same
`tool_use_id`s, and the session degraded — on the next message
`receive_response()` blocked forever, the RPyC inbound queue filled up,
and the backpressure quarantine from commit `b813e4d` kicked in.

### Root cause

Two intertwined code errors introduced in commits `87c80b8`
("Remove Claude Code provider polling") and `4c7ff36` ("Handle batched
Claude proxy tool calls"):

1. **The `mcp__deepworker__` prefix.** The check
   `block.name in proxy_tool_names` compared the **prefixed** name
   `mcp__deepworker__<tool>` against a set of **bare** names from
   `ToolMetadata`. The Claude Agent SDK always exposes MCP tools with
   the prefix `mcp__<server_name>__<tool_name>` (our server name is
   `"deepworker"`, see `proxy_mcp_server.py: create_sdk_mcp_server(name="deepworker", ...)`).
   Therefore the membership check ALWAYS failed for production payloads,
   and every proxy tool was classified as a built-in Claude tool.
   Existing unit tests did not catch this because they used bare names in
   mock `ToolUseBlock`s.

2. **Trying to wait for ALL pending_calls to register (deadlock).**
   Commit `4c7ff36` introduced `wait_for_matching_pending_calls(expected_calls=[…all blocks])`,
   which waited until **all** MCP handler tasks had registered their
   `pending_call`. This was based on the incorrect assumption that all
   handlers start in parallel. The real Claude CLI behavior: MCP
   `control_request`s are sent **sequentially** — the next request goes
   out only after the `tool_result` of the previous one is received.
   Therefore only ONE handler exists in `pending_calls` at a time;
   waiting for the rest is a 30s deadlock → RuntimeError. This was the
   root symptom of the `test_followup_turn_after_mcp_tools_still_works` test.

### The fix

#### `claude_code_provider/chat_model.py`

- Added the module-level constant `_PROXY_MCP_TOOL_PREFIX = "mcp__deepworker__"`
  and the helper `_strip_proxy_mcp_prefix(name)`. The prefix is stripped
  BEFORE comparison with `proxy_tool_names`.
- The `receive_response()` loop logic was simplified and brought to the
  correct event-driven semantics: on the first proxy `ToolUseBlock` we
  exit the loop and call the event-driven wait
  `proxy_mcp.wait_for_any_pending_call(timeout=30.0)`. This call parks
  on a single asyncio Future that is woken from
  `ProxyMCPServer._wake_matching_pending_waiters` immediately after
  `self.pending_calls[call_id] = …`. No polling, no `asyncio.sleep`.
- After being woken, we take `proxy_mcp.get_pending_calls()` and return
  EVERYTHING that is registered (usually one) to LangGraph. The next
  pending_call will be registered on the next `ainvoke` iteration —
  after LangGraph executes the tool, we call `resolve_call`, Claude
  receives the result and sends the next `control_request`.

#### `claude_code_provider/proxy_mcp_server.py`

- Added the method
  `wait_for_any_pending_call(timeout=30.0) -> None`.
  It returns immediately if a pending_call already exists; otherwise it
  parks on an `asyncio.Future` from `_pending_change_waiters` that is
  woken by `_wake_matching_pending_waiters` when the first pending_call
  is registered. The 30s timeout limits ONLY the registration race
  between the `ToolUseBlock` (which arrives via the message stream) and
  the corresponding `control_request` (which arrives via the SDK control
  channel) — the typical delay is milliseconds. The timeout does not
  affect tool execution itself: after registration the handler blocks on
  its own `self.timeout` from `ProxyMCPServer.__init__` (default 120s).
- The existing `wait_for_matching_pending_calls` is kept (it is used
  only by tests as a batch-wait utility), but production code no longer
  goes through it.

### Test coverage

Unit (`claude_code_provider/tests/test_proxy_mcp_server.py`,
class `TestProxyMCPServerPendingCalls`):
- `test_wait_for_any_pending_call_returns_existing` — returns
  immediately if a pending call already exists.
- `test_wait_for_any_pending_call_is_event_driven_no_polling` —
  parks on a Future and is woken by the handler.
- `test_wait_for_any_pending_call_times_out_if_no_handler` — times out
  when there is no handler at all.
- `test_wait_for_any_pending_call_returns_on_first_of_many` —
  returns on the FIRST registered handler, does not wait for the rest
  (matches the Claude CLI sequential dispatch).

Unit (`claude_code_provider/tests/test_event_based_tool_detection.py`,
class `TestMcpDeepworkerPrefixedToolDetection`):
- `test_single_prefixed_mcp_tool_is_detected` — a single
  `mcp__deepworker__<name>` is recognized as a proxy tool.
- `test_multiple_prefixed_mcp_tools_in_one_message_returns_first` —
  with N prefixed `ToolUseBlock`s in one `AssistantMessage`, only the
  one currently registered is returned (no deadlock).
- `test_prefixed_unknown_tool_is_built_in` — `mcp__*` names that are
  not in `ToolMetadata` are still considered built-in.

Honest integration (no mocks, through `agent_fixture`, a real
claude-code agent; integration test `test_claude_code_mcp_prefix.py`):
- `test_multiple_mcp_tools_in_one_turn_do_not_hang` — one turn,
  the agent uses `ls /tmp` + `execute echo` + `read_file fixture`,
  and must reply with content containing the fixture TOKEN.
- `test_followup_turn_after_mcp_tools_still_works` — an exact
  reproduction of the dev-agent freeze: turn 1 with batched MCP tools
  (`ls + execute`), turn 2 — arithmetic; the first turn used to hang
  on `wait_for_matching_pending_calls`, now both turns pass reliably.

All requests in the honest integration tests are safe: read-only
operations only (`ls /tmp`, `echo`, reading a temporary file created by
the test itself and removed afterwards), no network, no writes outside
the test fixture directory.

## Problem

Commits `a8cbdcd` and `01872d7` (Apr 9) replaced the proxy architecture with
direct execution. Tools are now executed INSIDE the MCP handler → LangGraph does
not see tool calls in `result.messages` → dwcli cannot display tool calls → the
agent_process broadcast only the final answer.

## Goal

Restore the proxy: tool calls go through LangGraph, are visible in
`result.messages`, and are broadcast to dwcli. Event-driven, no polling, no extra
threads, everything in one event loop.

## How the Claude SDK works (facts from the research)

When Claude calls an MCP tool:

1. **The Claude CLI sends `AssistantMessage(ToolUseBlock)` IMMEDIATELY** — BEFORE
   tool execution starts. This is a regular message; it goes through the message
   stream (`_message_send` queue) and is available in `receive_response()`.

2. **The Claude CLI sends `control_request(subtype="mcp_message")`** — a request
   to execute the MCP tool. The SDK calls `_spawn_control_request_handler()` →
   `asyncio.create_task()` → the handler runs in a **separate asyncio task**.

3. **The handler returns a result** → the SDK sends `control_response` back to
   the CLI.

4. **The Claude CLI receives the tool result**, continues generation, and sends
   further messages and a `ResultMessage`.

**Key point**: step 2 (handler) runs in a separate task from step 1 (message
stream). `receive_response()` CAN receive `AssistantMessage(ToolUseBlock)` and
continue processing WITHOUT waiting for the tool handler to finish.

## Solution architecture

### Principle: receive_response exits on ToolUseBlock

```
ainvoke() calls receive_response()
    ↓
AssistantMessage(TextBlock) → accumulate text
AssistantMessage(ToolUseBlock) → record tool call
    ↓
Got ToolUseBlock → EXIT receive_response (break/return)
    ↓ (do NOT wait for ResultMessage)
ainvoke() checks: ToolUseBlock present → collect pending_calls
    ↓
Return AIMessage(tool_calls=[...]) to LangGraph
    ↓
LangGraph executes the tool via ToolNode → ToolMessage in state
    ↓
LangGraph calls ainvoke() again with ToolMessage
    ↓
ainvoke() calls resolve_call(tool_call_id, result)
    → Future is unlocked
    → MCP handler returns the result
    → control_response goes to the Claude CLI
    ↓
ainvoke() does NOT call client.query() — the result already went through MCP
ainvoke() calls receive_response() again
    ↓
Claude CLI continues generation → new AssistantMessage → ResultMessage
    ↓
If there are more tool calls → repeat the loop
If it is the final answer → return AIMessage(content=text)
```

### Message flow (event-driven, no polling)

```
                    ainvoke()                LangGraph               MCP Handler
                       │                        │                       │
client.query(msg) ─────┤                        │                       │
                       │                        │                       │
receive_response()  ───┤                        │                       │
  │                    │                        │                       │
  ├─ TextBlock ────────┤ (accumulate text)      │                       │
  ├─ ToolUseBlock ─────┤ (record tool call)     │                       │
  │  BREAK! ───────────┤                        │         ┌─────────────┤
  │                    │                        │         │ handler()   │
  │                    │                        │         │ await future│
  │                    │                        │         │ (blocked)   │
  │  has_pending=True ─┤                        │         │             │
  │                    │                        │         │             │
  │  return AIMessage( │                        │         │             │
  │    tool_calls)  ───┼────────────────────────┤         │             │
  │                    │                  ToolNode:       │             │
  │                    │                  execute(tool)   │             │
  │                    │                        │         │             │
  │                    │              ToolMessage│         │             │
  │                    │◄───────────────────────┤         │             │
  │                    │                        │         │             │
  │ resolve_call() ────┼────────────────────────┼─────────┤             │
  │                    │                        │         │ future done │
  │                    │                        │         │ return res  │
  │                    │                        │         └─────────────┤
  │                    │                        │                       │
  │                    │                        │    control_response───┤──→ Claude CLI
  │                    │                        │                       │
receive_response()  ───┤                        │                       │
  │                    │                        │                       │
  ├─ TextBlock ────────┤ (accumulate text)      │                       │
  ├─ ResultMessage ────┤ BREAK                  │                       │
  │                    │                        │                       │
  │  return AIMessage( │                        │                       │
  │    content=text)───┼────────────────────────┤                       │
                       │                  (graph continues)             │
```

### Change 1: ProxyMCPServer — restore the Future-based proxy

**File:** `claude_code_provider/proxy_mcp_server.py`

Remove `direct_handler`. All tool handlers use the Future-based proxy:

```python
def _create_tool_handler(self, tool_name: str) -> Any:
    """Create async handler that blocks on Future.
    
    ⚠️ ARCHITECTURAL RULE: the handler MUST block on a Future!
    This ensures tool calls pass through the LangGraph ToolNode,
    which is required for:
    1. Visibility of tool calls in result.messages (for broadcast to dwcli)
    2. Correct checkpointer behavior (tool history persistence)
    3. Compatibility with interrupt_on (HITL approval)
    
    DO NOT replace with direct execution! Running tools directly in the
    handler makes them invisible to LangGraph and dwcli.
    """
    async def handler(args: dict[str, Any]) -> MCPResult:
        self._call_counter += 1
        call_id = f"{tool_name}_{self._call_counter}_{id(args)}"
        
        future: asyncio.Future[MCPResult] = asyncio.Future()
        self.pending_calls[call_id] = {
            "future": future,
            "name": tool_name,
            "args": args,
        }
        
        # Block on the Future — resolve_call() unlocks it
        # when LangGraph executes the tool and returns the result
        result = await future
        return result
    
    return handler
```

Remove `tool_executors`, `_make_runtime_executor`, `_make_tool_call_executor`,
`__agent_name__` injection — all of this was for the direct execution that is
being removed.

Keep `working_dir` and `agent_name` — they are needed for other purposes
(agent_name for subagent thread_id).

### Change 2: ainvoke() — exit receive_response on ToolUseBlock

**File:** `claude_code_provider/chat_model.py`

```python
async for event in client.receive_response():
    if isinstance(event, AssistantMessage):
        for block in event.content:
            if isinstance(block, TextBlock):
                response_text += block.text
            elif isinstance(block, ToolUseBlock):
                tool_use_blocks.append(block)
        
        # If there is a ToolUseBlock — EXIT IMMEDIATELY
        # Do NOT wait for ResultMessage — the Claude CLI is waiting for the tool result
        if tool_use_blocks:
            break
    
    elif isinstance(event, ResultMessage):
        break
```

### Change 3: ainvoke() — resolve_call for ToolMessage

**File:** `claude_code_provider/chat_model.py`

When a `ToolMessage` is received from LangGraph — call `resolve_call()` instead
of `client.query()`:

```python
elif isinstance(msg, ToolMessage):
    # Resolve pending tool call — unlocks the MCP handler Future
    tool_call_id = msg.tool_call_id
    mcp_result: MCPResult = {
        "content": [{"type": "text", "text": str(msg.content)}],
    }
    proxy_mcp.resolve_call(tool_call_id, mcp_result)
    tool_resolved = True
```

### Change 4: ainvoke() — build pending_calls from ToolUseBlock

Instead of checking `proxy_mcp.has_pending_calls()` (which depends on the race
condition between the spawned task and receive_response), build tool_calls
directly from `tool_use_blocks`:

```python
if tool_use_blocks:
    tool_calls = [
        {
            "id": block.id,
            "name": f"mcp__deepworker__{block.name}" if not block.name.startswith("mcp__") else block.name,
            "args": block.input,
        }
        for block in tool_use_blocks
    ]
    return AIMessage(content=response_text, tool_calls=tool_calls)
```

BUT: `resolve_call` needs the `call_id` from `pending_calls`, not from
`ToolUseBlock.id`. A mapping is needed: `ToolUseBlock.id` → `pending_call.id`.
Or use `ToolUseBlock.id` directly as the pending_calls key (since the MCP
handler in the spawned task receives the same id).

**Recommended approach**: the MCP handler uses the `tool_use_id` from the
control_request as `call_id`. Then `ToolUseBlock.id` == `call_id` and no
mapping is needed.

Need to verify: does the Claude SDK use the same ID in `ToolUseBlock` and in
`control_request` → `_handle_sdk_mcp_request`. If so, `block.id` can be used as
the key for `resolve_call`.

> ⚠️ VERIFIED (2026-06-16) — there is a hidden trap here too:
> `ToolUseBlock.id` (`content[].id`, `toolu_*`) for a single call is indeed
> equal to the `_meta.claudecode/toolUseId` of the same call on the
> control channel — so taking `call_id` from the control_request is CORRECT.
> BUT: (1) do NOT confuse `ToolUseBlock.id` (`toolu_*`, the call id) with the
> message-block id (`msg_*`) — these are DIFFERENT namespaces and are never
> equal (0/46683); (2) one assistant block carries MANY calls (up to 52), each
> with its own `toolu_*`, so taking "the block.id of the first block" as a
> single key is NOT allowed; (3) a built-in Claude tool and our proxy tool
> can match on base-name (`execute` vs `mcp__deepworker__execute`)
> → we could end up waiting forever for the `toolu_*` of someone else's call
> (deadlock observed on 2026-06-15). The ONLY source of identity is the
> control-channel `tool_use_id` of our proxy tool. See
> `FACTS_TOOL_USE_ID_VS_BLOCK_ID.md`.

**Current approach** (Update 2026-05-17): wait for AT LEAST ONE pending_call
to appear (it is created in the spawned task from the `control_request`),
then take `proxy_mcp.get_pending_calls()` and return it to LangGraph.
There is a race between receiving the `ToolUseBlock` in the message stream and
the pending_call appearing in the spawned task, so `ainvoke()` does not
yield/sleep — it waits for an event-driven signal from
`ProxyMCPServer.wait_for_any_pending_call(...)`. We do NOT try to
accumulate all ToolUseBlocks from one AssistantMessage and wait for all
handlers to register — the Claude CLI sends MCP `control_request`s strictly
**sequentially** (the next one only after receiving the `tool_result` of the
previous one), so waiting for the rest is guaranteed to deadlock.

```python
# NOTE (2026-05-17): strip the ``mcp__deepworker__`` prefix that
# Claude Agent SDK adds to MCP tool names. See
# ``_strip_proxy_mcp_prefix`` in chat_model.py.
proxy_blocks = [
    b for b in tool_use_blocks
    if _strip_proxy_mcp_prefix(b.name) in proxy_tool_names
]
if proxy_blocks:
    # Wait event-driven for the FIRST handler to register. We do NOT
    # wait for all of them — Claude CLI dispatches MCP control_requests
    # sequentially, so only one handler runs at a time. The rest
    # surface on subsequent ainvoke iterations after resolve_call.
    await proxy_mcp.wait_for_any_pending_call(timeout=30.0)
    pending = proxy_mcp.get_pending_calls()
    return AIMessage(content=response_text, tool_calls=pending)
```

### Change 5: Remove the direct execution infrastructure

**Files:** `chat_model.py`, `proxy_mcp_server.py`, `client_manager.py`

Remove:
- the `_tool_executors` dict
- `_make_runtime_executor()`
- `_make_tool_call_executor()`
- `_has_injected_tool_call_id()`
- the `tool_executors` parameter in `ProxyMCPServer.__init__`
- the `tool_executors` parameter in `ClientManager.get_or_create_client`
- `direct_handler` in `ProxyMCPServer._create_tool_handler`
- the `__agent_name__` injection in `direct_handler`

Tools are invoked by the LangGraph ToolNode, which uses `BaseTool.ainvoke()`.
It does not need wrappers — the ToolNode injects `ToolRuntime` itself.

### Change 6: bind_tools — metadata only, no executors

**File:** `chat_model.py`

```python
def bind_tools(self, tools, **kwargs):
    tools_metadata = []
    for tool_item in tools:
        if isinstance(tool_item, BaseTool):
            # Extract metadata only — tools execute via LangGraph ToolNode
            metadata = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": schema,
            }
            tools_metadata.append(metadata)
    
    new_model = ClaudeCodeChatModel(base_options=self._base_options)
    new_model._tools_metadata = tools_metadata
    return new_model
```

### Change 7: Mapping tool_call_id between LangGraph and MCP

The LangGraph ToolNode uses `tool_call.id` from `AIMessage.tool_calls`. This ID
must match the `call_id` in the ProxyMCPServer pending_calls so that
`resolve_call(tool_call_id, result)` works.

**Problem**: the `pending_calls` key is built in the handler as
`f"{tool_name}_{counter}_{id(args)}"`, while `tool_call.id` in the AIMessage is
taken from ToolUseBlock.id (Claude format: `toolu_xxxx`).

**Solution**: use `ToolUseBlock.id` as the pending_calls key:

> ⚠️ See the clarification above and `FACTS_TOOL_USE_ID_VS_BLOCK_ID.md`:
> `ToolUseBlock.id` here = the `toolu_*` of a SINGLE call (= `tool_use_id`),
> not `msg_*`. The source of this id in the handler is `_meta.claudecode/
> toolUseId` (via the SDK ContextVar). One block = many calls; the key is
> the per-call `tool_use_id`, not the block id.

```python
# In the handler (when receiving the control_request):
# call_id must match ToolUseBlock.id
# But the handler does not know ToolUseBlock.id...
```

This is a problem. The handler receives args through the MCP protocol but does
not receive `tool_use_id` directly. Need to verify whether the Claude SDK
passes `tool_use_id` to the MCP handler.

**Alternative solution**: in `ainvoke()`, when building `AIMessage(tool_calls)`
from pending_calls — use `pending_call["id"]` as `tool_call.id`. Then LangGraph
will return `ToolMessage(tool_call_id=pending_call["id"])`, and `resolve_call`
will find it by that id.

```python
# In ainvoke:
tool_calls = [
    {
        "id": call["id"],  # pending_call["id"] = "{tool_name}_{counter}_{...}"
        "name": call["name"],
        "args": call["args"],
    }
    for call in proxy_mcp.get_pending_calls()
]
return AIMessage(content=response_text, tool_calls=tool_calls)

# Later, when the ToolMessage is received:
proxy_mcp.resolve_call(msg.tool_call_id, mcp_result)
# tool_call_id == pending_call["id"] → match
```

This is a working approach — this is exactly how the original implementation in
`bcc7583` worked.

### Change 8: Update tests

- Remove tests for direct_handler, _make_runtime_executor, etc.
- Restore/update tests for the Future-based proxy
- Update the `test_fallback_to_future_without_executors` test — this is now the
  main path
- Add a test: ToolUseBlock → pending_calls → resolve_call → result

### Implementation order

1. Restore the Future-based handler in ProxyMCPServer (remove direct execution)
2. Update ainvoke: exit on ToolUseBlock, resolve_call for ToolMessage
3. Remove the tool_executors infrastructure from bind_tools, ClientManager
4. Update tests
5. E2E test: tool calls are visible in result.messages
6. Integration test: dwcli shows tool calls

### What is kept

- `mandatory thread_id` in ainvoke (from the previous design)
- `agent_name` in ProxyMCPServer (for subagent thread_id)
- `working_dir` in ProxyMCPServer
- Subagent thread isolation (thread_id in subagents.py)
- TaskManager on the same provider (task_manager.py)

## Risks

1. **Timing between ToolUseBlock and pending_calls**: the ToolUseBlock arrives
   via the message stream, while the pending_call is created via the
   control_request in a spawned task. There can be a race condition if the
   ToolUseBlock arrives before the control_request. This is solved with
   event-driven waiting via `wait_for_matching_pending_calls(...)`, not
   `await asyncio.sleep(0)`.

2. **Multiple tool calls in one turn**: Claude can call several tools at once.
   All ToolUseBlocks arrive in one AssistantMessage. All pending_calls must be
   resolved before receive_response is called again.

3. **Tool call ID mapping**: we must ensure pending_call.id is consistently
   used as tool_call.id throughout the whole LangGraph cycle.
