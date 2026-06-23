# A related issue — Proxy MCP pending-call timeout fallback

**Kind:** agent_tool_complaint | **Severity:** MEDIUM

## Log sample
```
[WARN ainvoke] wait_for_any_pending_call timed out after 5.0s for
thread_id=<agent>; visible proxy block tool='grep' args={...}, but no
mcp handler registered a pending call; returning the visible block as
a normal tool_call fallback.
```

## Verification

**Already fixed** by commit `579bfe0` (Jun 1, 2026, already applied):

> `fix(proxy): reduce wait_for_any_pending_call timeout from 30s to 5s`

**What changed:**
- `chat_model.py` — both call sites now use 5s timeout (was 30s):
  - Line 925-928: `asyncio.wait_for(proxy_mcp.wait_for_any_pending_call(), timeout=5.0)`
  - Line 1064: `proxy_mcp.wait_for_any_pending_call(timeout=5.0)`
- `proxy_mcp_server.py:195` — default timeout unchanged at 30.0 (only used for other callers)

**Why it's fixed:**
1. The 5s timeout is a 50× safety margin over the typical millisecond registration race.
2. The fallback (visible block → tool_call) fires after 5s instead of 30s.
3. All 241 tests pass (`241 passed, 1 skipped`).
4. `test_stall_fallback_on_missing_control_request.py` (2 tests) specifically validates this fallback path.

The WARN-level log entry is the **expected** behaviour after commit `579bfe0` — it fires when the control_request never arrives (e.g. mixed bare/prefixed tool names), and the fallback correctly returns the visible block as a tool_call.
