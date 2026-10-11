# Already-Fixed-Issue: #66 — Proxy MCP pending-call tracking timeout

## Logged behavior (no longer a bug)

The log line `[WARN ainvoke] wait_for_any_pending_call timed out after 5.0s ...`
is the **expected fallback** of the fix, not a bug. When the proxy MCP handler does
not register a pending call within 5 seconds, the code gracefully degrades by
returning the visible ToolUseBlock as a normal tool_call, rather than raising
`RuntimeError` (the old behavior).

## Verification

- `wait_for_any_pending_call(timeout=5.0)` with fallback → visible block as tool_call
  exists at `claude_code_sdk_provider/chat_model.py:1064-1092`.
- Regression test `test_double_block_missing_control_request_degrades_to_tool_call`
  in `tests/test_stall_fallback_on_missing_control_request.py` verifies this behavior.
- Full suite: 241 passed, 1 skipped.
