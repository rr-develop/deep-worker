# Design: Reduce MCP Proxy Pending-Call Timeout from 30s to 5s

## Problem

`wait_for_any_pending_call(timeout=30.0)` in `chat_model.py:927` waits **30 seconds** when a control_request never arrives (undispatched due to mixed bare/prefixed tool names). The fallback to visible block works correctly, but the agent hangs for 30s before the fallback fires.

Observed: 3 occurrences for a live agent, each blocking the agent for 30s.

## Root Cause

The Claude Code CLI emits a `ToolUseBlock` on the message stream but may fail to dispatch the corresponding MCP `control_request` on the control channel (e.g. when a session mixes bare and `mcp__deepworker__` prefixed tool names). The proxy never registers `pending_calls[call_id]`, the `Future` never resolves, and after 30s the timeout fires.

The fallback (lines 928-955) correctly uses the visible block's `id/name/args` as the tool call — the only problem is the 30s delay.

## Design

**Single change:** Reduce `timeout=30.0` → `timeout=5.0` at `chat_model.py:927`.

**Why 5s:**
- The registration race between ToolUseBlock (message stream) and control_request (control channel) is documented as "typically milliseconds" (code comment, line 900-903).
- 5s provides a 50× safety margin over the typical case.
- 5s keeps the agent responsive on failure paths instead of hanging for 30s.
- Even under exceptional load, a 5-second window is far more than enough for a control_request to arrive.

**Does NOT change:**
- Architecture (same flow, same fallback, same event-driven Future)
- Contracts (same method signatures, same return values)
- Policies (no polling, no threads, no 777/666)
- Fallback behavior (visible block → tool_call exactly as before)

### Tests to update

Tests that assert `timeout=30.0`:
- `test_event_based_tool_detection.py:639` — `assert_awaited_once_with(timeout=30.0)`
- `test_event_based_tool_detection.py:739` — `assert_awaited_once_with(timeout=30.0)`
- `test_chat_model.py:320` — `assert_awaited_once_with(timeout=30.0)`
- `test_chat_model.py:413` — `assert_awaited_once_with(timeout=30.0)`
- `test_chat_model.py:500` — `assert_awaited_once_with(timeout=30.0)`

All change to `timeout=5.0`.
