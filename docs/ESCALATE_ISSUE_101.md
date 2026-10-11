# Escalate a related issue — RuntimeError: Attempted to exit cancel scope in a different task

## Root Cause

The error originates from the **`mcp` library** (v1.17.0, pip package), not from our code.

**File:** the `mcp` library's site-packages `server/streamable_http_manager.py` (line 107)

```python
async with anyio.create_task_group() as tg:
    ...
    yield  # Let the application run
```

The `run()` method is decorated with `@contextlib.asynccontextmanager` and uses `anyio.create_task_group()` inside the async generator. When the generator is closed during cleanup (e.g. from a different asyncio task calling `aclose()`), the task group's `__aexit__` tries to exit a cancel scope that was entered in a **different task**, causing:

```
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

## Known Issue

This is a known bug in the MCP Python SDK:

- https://github.com/modelcontextprotocol/python-sdk/issues/521 (sse_client)
- https://github.com/modelcontextprotocol/python-sdk/issues/915 (streamable_http_client, P1, fix proposed in PR #2711)

## Why Cannot Be Fixed in Our Code

1. The bug is in the `mcp` pip package, NOT in any code under `/path/to/fixes`.
2. Fixing requires modifying `anyio.create_task_group()` usage inside async generators in the `mcp` library.
3. Per policy: requiring changes to external dependencies → Escalate.
4. No local fork of `mcp` exists under the host's repo directories to patch.

## Verification

- **grep for `CancelScope`, `anyio.`, `create_task_group` in our codebase** → no matches.
- **All relevant tests pass** (137 passed, 5 skipped in codex_provider; 241 passed, 1 skipped in claude_code_sdk_provider).
- Error traceback confirms it's purely in the `mcp` library's generator cleanup path.
