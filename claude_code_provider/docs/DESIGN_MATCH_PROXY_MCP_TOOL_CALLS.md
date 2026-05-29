# Design: Match Proxy MCP ToolUseBlock to Pending Call

Status: SUPERSEDED / REVERTED (2026-05-26)

> ⚠️ This design was implemented (commit `28f56d3`) and then reverted, because
> its core premise is false. It assumed the visible message-stream
> `ToolUseBlock.input` is byte-for-byte equal to the arguments delivered to the
> MCP control-channel handler. Live evidence (2026-05-26) showed
> they DIVERGE for the same logical call:
>
> ```text
> visible  grep(pattern='unsubscribe_messages')   control  grep(pattern='unsubscribe')
> visible  read_file(..., limit=30)               control  read_file(..., limit=40)
> ```
>
> Claude streams a partial/earlier argument JSON in the assistant message while
> the finalized args reach the MCP handler over the control channel. Requiring
> exact `(tool_name, args)` equality therefore made `ainvoke` wait forever for a
> pending call that never registers with those exact args → 30s timeout → agent
> `error` on every restart/turn that hit a divergent call.
>
> Correct behaviour (restored): the visible `ToolUseBlock` is only a SIGNAL that
> a tool is coming. The authoritative call is whatever the proxy registers in
> `pending_calls` (keyed by the handler-assigned `call_id`, which is what
> `resolve_call` later matches). `ainvoke` waits for ANY handler registration
> (`wait_for_any_pending_call`) and then surfaces `get_pending_calls()`.
>
> The `ProxyMCPServer.wait_for_pending_calls(tool_name=, arguments=)` matching
> primitive remains in the codebase (with its unit tests) but is NOT used by
> `chat_model.ainvoke` for the visible-block → pending-call bridge.

---

## Original proposal (kept for history)

Status: Proposed

## Problem

`ClaudeCodeChatModel.ainvoke()` currently treats the first visible proxy
`ToolUseBlock` as a signal to wait for *any* pending MCP call:

```python
await proxy_mcp.wait_for_any_pending_call(timeout=30.0)
pending = proxy_mcp.get_pending_calls()
```

This is not strong enough. The live log shows the visible
message-stream `ToolUseBlock` and the returned LangGraph `tool_call` can
diverge, for example:

```text
visible=mcp__deepworker__execute returned=['grep']
visible=mcp__deepworker__grep    returned=['read_file']
visible=mcp__deepworker__read_file returned=['grep']
```

After the streams drift, some turns reach:

```text
Received ToolUseBlock: grep
wait_for_any_pending_call timed out after 30s
```

The 30s timeout is not waiting for tool execution. `ProxyMCPServer` registers
`pending_calls[call_id]` before it waits for LangGraph to execute the tool.
The timeout means no handler registration was observed for the relevant turn.
The misleading part is that the code says "matching" in the error while it
actually waits for any pending call.

## Goals

1. Never surface a pending MCP call to LangGraph unless it matches the visible
   proxy `ToolUseBlock` that caused `ainvoke()` to break from
   `receive_response()`.
2. Preserve the sequential-control-request behavior: do not wait for all
   visible `ToolUseBlock`s in an assistant message, because Claude CLI sends
   MCP control requests one at a time and waits for the previous tool result.
3. Keep the implementation event-driven. No polling and no sleeps.
4. Keep tool execution visible through LangGraph `ToolNode`; do not execute
   tools directly inside the MCP handler.
5. Improve timeout diagnostics so a future failure says which visible block was
   expected and which pending calls, if any, were present.

## Non-goals

- Do not change Claude SDK transport architecture.
- Do not add threads.
- Do not re-enable direct MCP handler execution.
- Do not wait for every `ToolUseBlock` in one assistant message.

## Design

### 1. Capture the expected first proxy call

When `ainvoke()` sees proxy `ToolUseBlock`s, it should keep only the first
proxy block as the call that must be matched on this iteration:

```python
first_proxy_block = proxy_blocks[0]
expected_tool_name = _strip_proxy_mcp_prefix(first_proxy_block.name)
expected_args = dict(first_proxy_block.input or {})
```

Only the first block is used because Claude CLI dispatches control requests
sequentially. The next visible/proxy block belongs to a later iteration after
LangGraph resolves the first call.

### 2. Wait for the matching pending call, not any pending call

Replace `wait_for_any_pending_call()` in `ainvoke()` with the existing
matching wait primitive:

```python
pending_tool_calls = await proxy_mcp.wait_for_pending_calls(
    tool_name=expected_tool_name,
    arguments=expected_args,
    timeout=30.0,
)
```

`ProxyMCPServer.wait_for_pending_calls()` is already event-driven:

- It returns immediately if a matching pending call already exists.
- Otherwise it parks on a one-shot `asyncio.Future`.
- The future is woken from `_create_tool_handler()` immediately after
  `pending_calls[call_id]` is registered.

This waits for handler registration, not tool completion.

### 3. Return only the matched pending call(s)

When `pending_tool_calls` is populated by the matching wait, `ainvoke()` should
return those calls directly. It should not fall back to `get_pending_calls()`
for arbitrary existing calls in the same path.

The old fallback remains valid only for code paths where no visible proxy block
was captured but pending calls already exist. That is legacy defensive behavior;
the visible-block path must be strict.

### 4. Timeout behavior

If the matching wait times out:

1. Log the expected visible tool name and args.
2. Log the current pending calls snapshot.
3. Cleanup the Claude client for the thread, as today.
4. Raise a `RuntimeError` that says the matching pending MCP handler did not
   register, with the expected name and args included.

This keeps the current recovery policy but makes the failure honest and
actionable.

## TDD Plan

Add/adjust tests in `claude_code_provider/tests/test_chat_model.py`:

1. A proxy `ToolUseBlock` for `test_calculator` must call
   `wait_for_pending_calls(tool_name="test_calculator", arguments={...})`, not
   `wait_for_any_pending_call()`.
2. If a stale nonmatching pending call exists (for example `test_reader`) while
   the visible block is `test_calculator`, `ainvoke()` must not return the stale
   call. It must wait for and return the matching calculator call.
3. Multiple visible proxy blocks in one assistant message must still return only
   the first matching pending call and must not wait for all blocks.
4. A timeout waiting for the matching call should cleanup the client and raise a
   message containing the expected tool name.

Add/adjust `ProxyMCPServer` tests only if needed; its matching wait is already
covered.

## Practical Integration Test

Add a test under `claude_code_provider/tests/integration/` or the existing test
suite that uses a real `ProxyMCPServer` instance without mocking it:

- Create a real proxy with two tools, e.g. `execute` and `grep`.
- Start a stale/nonmatching handler task for `grep`, which registers a real
  pending call.
- Simulate the visible first block as `execute` by waiting through
  `wait_for_pending_calls(tool_name="execute", arguments={...})` while the grep
  pending call exists.
- Start the real `execute` handler and assert the waiter returns the execute
  pending call, not grep.
- Resolve both futures and await tasks to avoid leaks.

This verifies the core event-driven matching behavior without mocks and without
starting a live Claude CLI.

## Risks

- If Claude CLI genuinely emits visible ToolUseBlocks in a different order from
  MCP control requests, strict matching will surface timeouts earlier instead of
  silently executing the wrong tool. That is acceptable: wrong tool execution is
  worse than an explicit recoverable error.
- The existing live logs show stream/control mismatch. This design will prevent
  corrupt LangGraph tool calls, but a remaining CLI/SDK ordering bug may still
  need a separate fix if matching timeouts continue.
