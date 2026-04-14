# RPyC Netref Elimination Design Document

**Date:** 2026-04-05
**Status:** Design Proposal

---

## Executive Summary

**Problem:** RPyC exposed methods returning `dict`/`list` create netref proxies that generate **12x overhead** (24 messages instead of 2) per `get_status()` call. With 5-second polling from the web client, this causes high CPU usage and unnecessary network traffic.

**Solution:** Modify all exposed methods to return **JSON-serialized strings** instead of dict/list, eliminating netref proxy overhead.

**Expected Impact:** Reduce RPyC messages from **24 → 2** per get_status() call (**92% reduction**).

---

## 1. Problem Analysis

### 1.1 Current Behavior

From integration test `test_complete_rpyc_trace.py`:

```
📊 Total RPyC protocol messages: 24
   - SEND (client → agent): 12
   - RECV (agent → client): 12
   - Duration: 0.010s
   - Overhead: 12.0x (expected 2, got 24)

⚠️  Detected 4 dict.get() operations
   This indicates netref proxy overhead - each field access is a separate RPC call
```

### 1.2 Message Breakdown

**Ideal RPC (2 messages):**
1. SEND: Call get_status()
2. RECV: Response dict

**Actual RPyC with netref (24 messages):**

Messages 1-8: Method call and initial response
```
1-2.  [SEND] Call get_status() → [RECV] method reference
3-4.  [SEND] Call method → [SEND] cleanup
5-6.  [RECV] dict netref → [RECV] dict netref (duplicate?)
7-8.  [SEND] Access __format__ → [RECV] formatted string
```

Messages 9-12: dict.get('success')
```
9.    [SEND] dict.get('success')
10.   [RECV] builtin_function_or_method reference
11.   [SEND] call builtin with args
12.   [RECV] True
```

Messages 13-16: dict.get('session_info')
```
13.   [SEND] dict.get('session_info')
14.   [RECV] builtin_function_or_method reference
15.   [SEND] call builtin with args
16.   [RECV] None
```

Messages 17-20: dict.get('status')
```
17.   [SEND] dict.get('status')
18.   [RECV] builtin_function_or_method reference
19.   [SEND] call builtin with args
20.   [RECV] 'ready'
```

Messages 21-24: dict.get('error')
```
21.   [SEND] dict.get('error')
22.   [RECV] builtin_function_or_method reference
23.   [SEND] call builtin with args
24.   [RECV] None
```

### 1.3 Root Cause

RPyC returns dict as **netref proxy** object. Each field access becomes a separate RPC call:

```python
# Current client code (agent_client.py:156)
response_netref = await self._root.get_status()
response = response_netref  # ❌ This is a netref proxy!

# Later when accessing fields:
success = response.get('success')      # RPC call #1 (4 messages)
session_info = response.get('session_info')  # RPC call #2 (4 messages)
status = response.get('status')        # RPC call #3 (4 messages)
error = response.get('error')          # RPC call #4 (4 messages)
```

---

## 2. Proposed Solution

### 2.1 Design Principle

**All exposed RPyC methods MUST return JSON-serialized strings, not dict/list.**

This forces RPyC to transmit the serialized data as a plain string (no netref proxy), eliminating the overhead.

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SERVER SIDE                              │
│  (deep_worker_cli/rpyc_service/agent_service.py)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  OLD:                                                        │
│    async def exposed_get_status(self) -> dict[str, Any]:   │
│        return {                                              │
│            "success": True,                                  │
│            "status": "ready",                                │
│            "error": None                                     │
│        }                                                     │
│                                                              │
│  NEW:                                                        │
│    async def exposed_get_status(self) -> str:              │
│        result = {                                            │
│            "success": True,                                  │
│            "status": "ready",                                │
│            "error": None                                     │
│        }                                                     │
│        return json.dumps(result)  # ✅ Return JSON string   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ RPyC transmits as plain string
                            │ (NO netref proxy!)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT SIDE                              │
│  (deep_worker_cli/client/agent_client.py)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  OLD:                                                        │
│    response_netref = await self._root.get_status()         │
│    response = response_netref  # ❌ netref proxy            │
│    success = response.get('success')  # RPC call!           │
│                                                              │
│  NEW:                                                        │
│    response_json_str = await self._root.get_status()       │
│    response = json.loads(response_json_str)  # ✅ Local dict│
│    success = response.get('success')  # Local access, no RPC│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Plan

### 3.1 Server-Side Changes

**File:** `deep_worker_cli/rpyc_service/agent_service.py`

#### Methods Requiring Modification

| Method | Current Return Type | New Return Type | Location |
|--------|-------------------|----------------|----------|
| `exposed_get_status()` | `dict[str, Any]` | `str` | Line 232 |
| `exposed_get_service_aliases()` | `list[str]` | `str` | Line 224 |
| `exposed_get_metadata()` | `dict[str, Any]` | `str` | Line 414 |
| `exposed_get_history()` | `dict[str, Any]` | `str` | Line 354 |
| `exposed_handle_command()` | `dict[str, Any]` | `str` | Line 460 |
| `exposed_interrupt_agent()` | `dict[str, Any]` | `str` | Line 644 |
| `exposed_shutdown()` | `dict[str, Any]` | `str` | Line 694 |

#### Example Transformation

**Before:**
```python
async def exposed_get_status(self) -> dict[str, Any]:
    """Get current agent status via RPyC.

    Returns:
        Dict with status information
    """
    try:
        if not self.agent or not self.executor:
            return {
                "success": True,
                "status": "ready",
                "error": None
            }

        # ... logic ...

        return {
            "success": True,
            "status": status,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error in exposed_get_status: {e}")
        return {
            "success": False,
            "error": str(e),
            "status": "error"
        }
```

**After:**
```python
import json  # Add to imports

async def exposed_get_status(self) -> str:
    """Get current agent status via RPyC.

    IMPORTANT: Returns JSON-serialized string to avoid RPyC netref overhead.
    Client must json.loads() the response.

    Returns:
        JSON string containing status information
    """
    try:
        if not self.agent or not self.executor:
            result = {
                "success": True,
                "status": "ready",
                "error": None
            }
            return json.dumps(result)

        # ... logic ...

        result = {
            "success": True,
            "status": status,
            "error": None
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in exposed_get_status: {e}")
        result = {
            "success": False,
            "error": str(e),
            "status": "error"
        }
        return json.dumps(result)
```

### 3.2 Client-Side Changes

**File:** `deep_worker_cli/client/agent_client.py`

#### Locations Requiring Updates

| Method | Lines | Change Required |
|--------|-------|-----------------|
| `_send_command()` - status | 152-164 | Add `json.loads()` |
| `_send_command()` - invoke | 166-174 | Add `json.loads()` |
| `_send_command()` - get_history | 176-184 | Add `json.loads()` |

#### Example Transformation

**Before:**
```python
if command_type == "status":
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[CLIENT] About to call await root.get_status()...")
    response_netref = await self._root.get_status()
    # CRITICAL: Convert netref to local dict to prevent memory leaks!
    # Without obtain(), response is netref proxy and each .get() creates leaking netrefs
    # TEMP TEST: Disable obtain() to check if it causes high CPU
    # response = obtain(response_netref)
    response = response_netref  # TEST WITHOUT obtain()
    logger.info(f"[CLIENT] await root.get_status() returned: {response}")
    # Server already returns {"success": True, "session_info": {...}}, no need to wrap again
    return response
```

**After:**
```python
import json  # Add to imports

if command_type == "status":
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[CLIENT] About to call await root.get_status()...")

    # Server returns JSON string (not dict) to avoid netref overhead
    response_json_str = await self._root.get_status()

    # Deserialize JSON string to local dict (no RPC overhead for field access)
    response = json.loads(response_json_str)

    logger.info(f"[CLIENT] await root.get_status() returned: {response}")
    return response
```

### 3.3 Consumer Code Changes

#### 3.3.1 Web client

**File:** the web client's session service module (external consumer, not included here)

No changes required - the web client calls `AgentClient.get_status()` which already returns deserialized dict.

**Verification Points:**
- Line 457: `status = await self.get_session_status_async(session_name)` ✅ No change needed
- Line 652: `status_result = await session.get_status_async()` ✅ No change needed

#### 3.3.2 LangChain bridge consumer

**File:** the external LangChain bridge module (external consumer, not included here)

**Change Required:** Lines 343, 371

**Before:**
```python
messages = self.session.get_history()  # Returns dict
```

**After:**
```python
# No change needed if session.get_history() is AgentClient method
# (AgentClient already deserializes the JSON response)
messages = self.session.get_history()
```

**Note:** If `get_history()` is called directly on RPyC connection (not through AgentClient), add `json.loads()`.

#### 3.3.3 Tests

All test files calling exposed methods must be updated:

**Files to Update:**
- `tests/test_complete_rpyc_trace.py` - Line 163
- `tests/test_single_rpyc_trace.py`
- `tests/test_single_get_status_rpyc_messages.py`
- `tests/test_rpyc_message_count.py`
- `tests/test_obtain_netref_fix.py`
- `tests/integration_ai/test_status_minimal_response.py`
- `tests/integration_ai/test_dwcli_headless_status.py`
- `tests/integration_ai/test_claude_code_integration.py`
- `tests/integration_ai/agent_fixture.py`
- `tests/integration/test_agent_service_asyncio_e2e.py`
- `tests/integration/test_asyncio_server_e2e.py`
- `tests/integration/test_full_migration_e2e.py`
- `tests/rpyc_service/test_agent_service_metadata.py`

**Pattern:**
```python
# OLD
status = await client.get_status()

# NEW (if calling RPyC directly)
status_json = await rpyc_conn.root.get_status()
status = json.loads(status_json)

# OR (if using AgentClient - no change needed)
status = await client.get_status()  # AgentClient handles deserialization
```

---

## 4. Migration Strategy

### 4.1 Backward Compatibility

**Option 1: Breaking Change (Recommended)**
- Update all server methods and all clients simultaneously
- Fast, clean implementation
- Requires coordinated deployment

**Option 2: Gradual Migration**
- Add new `exposed_get_status_json()` methods
- Keep old methods for compatibility
- Deprecate old methods over time
- More code maintenance burden

**Recommendation:** Use **Option 1** since this is internal infrastructure with controlled consumers.

### 4.2 Implementation Sequence

1. **Phase 1: Server-Side** (deep_worker_cli)
   - Update all 7 exposed methods in `agent_service.py`
   - Add `import json` at module level
   - Wrap all return values with `json.dumps()`
   - Update type hints: `dict[str, Any]` → `str`

2. **Phase 2: Client-Side** (deep_worker_cli)
   - Update `agent_client.py` to deserialize JSON responses
   - Add `json.loads()` in `_send_command()` for all command types

3. **Phase 3: Tests** (deep_worker_cli)
   - Update all test files that call exposed methods directly
   - Verify message count drops from 24 → 2

4. **Phase 4: External Consumers** (if any)
   - Update the external LangChain bridge consumer (if needed)
   - Update the web client (likely no changes needed)
   - Verify with integration tests

5. **Phase 5: Validation**
   - Run complete test suite
   - Monitor CPU usage in production
   - Verify 5-second polling overhead is eliminated

### 4.3 Testing Strategy

#### Unit Tests
- Test each exposed method returns valid JSON string
- Test JSON parsing on client side
- Test error handling with malformed JSON

#### Integration Tests
- **CRITICAL:** Update `test_complete_rpyc_trace.py` to verify message count = 2
- Test end-to-end flow: web client → AgentClient → RPyC → Agent
- Test backward compatibility (if gradual migration)

#### Performance Tests
- Measure RPyC message count before/after
- Measure CPU usage during 5-second polling
- Measure latency improvement

---

## 5. Expected Performance Impact

### 5.1 Message Count Reduction

**Before:**
```
get_status() call:
  Total messages: 24 (12 SEND + 12 RECV)
  Overhead: 12x
```

**After:**
```
get_status() call:
  Total messages: 2 (1 SEND + 1 RECV)
  Overhead: 1x (ideal)
```

**Reduction:** 24 → 2 messages = **92% reduction**

### 5.2 Network Traffic

With 5-second polling from the web client:

**Before:**
- 24 messages × 12 messages/minute = 288 messages/minute
- Each message ~100-200 bytes → ~50KB/minute

**After:**
- 2 messages × 12 messages/minute = 24 messages/minute
- ~4KB/minute

**Reduction:** ~92% network traffic reduction

### 5.3 CPU Usage

Current CPU spikes during polling are caused by:
1. 24 messages × serialization/deserialization overhead
2. 4 separate RPC calls for dict field access
3. Netref proxy object creation/cleanup

Expected CPU reduction: **50-70%** during polling intervals

---

## 6. Risk Analysis

### 6.1 Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing consumers | High | Medium | Comprehensive grep search, coordinated deployment |
| JSON serialization errors | Medium | Low | Comprehensive error handling, schema validation |
| Performance regression (serialization overhead) | Low | Very Low | JSON serialization is fast, benefits far outweigh costs |
| Backward incompatibility | High | High (if not coordinated) | Update all consumers simultaneously |

### 6.2 Rollback Plan

If issues arise:
1. Revert server-side changes (`git revert`)
2. Revert client-side changes
3. Redeploy previous version
4. Total rollback time: < 5 minutes

---

## 7. Alternative Solutions Considered

### 7.1 Alternative 1: Use `obtain()` to copy netref locally

**Approach:**
```python
from rpyc.utils.classic import obtain
response = obtain(await self._root.get_status())
```

**Pros:**
- Minimal code changes
- Copies netref to local dict

**Cons:**
- Still creates initial netref (overhead)
- `obtain()` performs recursive copy (CPU intensive for large objects)
- Doesn't eliminate initial RPC overhead

**Verdict:** ❌ Rejected (tested in previous iterations, high CPU usage)

### 7.2 Alternative 2: Custom RPyC protocol

**Approach:** Modify RPyC protocol to not create netrefs for specific types

**Pros:**
- Transparent to application code

**Cons:**
- Requires forking RPyC library
- Complex implementation
- Maintenance burden

**Verdict:** ❌ Rejected (over-engineered)

### 7.3 Alternative 3: gRPC/Protocol Buffers

**Approach:** Replace RPyC with gRPC

**Pros:**
- Industry standard
- Efficient binary serialization
- No netref issues

**Cons:**
- Complete rewrite of IPC layer
- Months of development
- Not worth it for this use case

**Verdict:** ❌ Rejected (too much work for marginal benefit)

---

## 8. Implementation Checklist

### Server-Side (agent_service.py)
- [ ] Add `import json` to imports
- [ ] Update `exposed_get_status()` → return `json.dumps(result)`
- [ ] Update `exposed_get_service_aliases()` → return `json.dumps(result)`
- [ ] Update `exposed_get_metadata()` → return `json.dumps(result)`
- [ ] Update `exposed_get_history()` → return `json.dumps(result)`
- [ ] Update `exposed_handle_command()` → return `json.dumps(result)`
- [ ] Update `exposed_interrupt_agent()` → return `json.dumps(result)`
- [ ] Update `exposed_shutdown()` → return `json.dumps(result)`

### Client-Side (agent_client.py)
- [ ] Add `import json` to imports
- [ ] Update `_send_command()` status branch → add `json.loads()`
- [ ] Update `_send_command()` invoke branch → add `json.loads()`
- [ ] Update `_send_command()` get_history branch → add `json.loads()`

### Tests
- [ ] Update `test_complete_rpyc_trace.py` - verify 24 → 2 messages
- [ ] Update `test_single_rpyc_trace.py`
- [ ] Update `test_single_get_status_rpyc_messages.py`
- [ ] Update `test_rpyc_message_count.py`
- [ ] Update `test_obtain_netref_fix.py`
- [ ] Update `tests/integration_ai/test_status_minimal_response.py`
- [ ] Update `tests/integration_ai/test_dwcli_headless_status.py`
- [ ] Update `tests/integration_ai/test_claude_code_integration.py`
- [ ] Update `tests/integration_ai/agent_fixture.py`
- [ ] Update `tests/integration/test_agent_service_asyncio_e2e.py`
- [ ] Update `tests/integration/test_asyncio_server_e2e.py`
- [ ] Update `tests/integration/test_full_migration_e2e.py`
- [ ] Update `tests/rpyc_service/test_agent_service_metadata.py`

### External Consumers
- [ ] Check the external LangChain bridge consumer - update if needed
- [ ] Check the web client - verify no changes needed
- [ ] Run integration tests

### Validation
- [ ] Run full test suite
- [ ] Verify RPyC message count = 2 per get_status()
- [ ] Monitor CPU usage during web client polling
- [ ] Verify error handling works correctly

---

## 9. Success Criteria

1. **RPyC Message Count:** 24 → 2 messages per get_status() call
2. **All Tests Pass:** No regressions in test suite
3. **CPU Usage:** 50-70% reduction during 5-second polling
4. **Network Traffic:** 90%+ reduction in RPyC message volume
5. **No Breaking Changes:** the web client and the external LangChain bridge consumer continue working

---

## 10. Timeline Estimate

| Phase | Estimated Time |
|-------|---------------|
| Server-side changes | 30 minutes |
| Client-side changes | 20 minutes |
| Test updates | 1 hour |
| External consumer verification | 30 minutes |
| Integration testing | 1 hour |
| Production validation | 1 hour |
| **Total** | **~4 hours** |

---

## 11. Appendix: Code Locations

### Server-Side Methods

```
deep_worker_cli/rpyc_service/agent_service.py

Line 224: async def exposed_get_service_aliases(self) -> list[str]
Line 232: async def exposed_get_status(self) -> dict[str, Any]
Line 354: async def exposed_get_history(self, limit: int = 50, offset: int = 0) -> dict[str, Any]
Line 414: async def exposed_get_metadata(self) -> dict[str, Any]
Line 460: async def exposed_handle_command(self, command: str) -> dict[str, Any]
Line 644: async def exposed_interrupt_agent(self) -> dict[str, Any]
Line 694: async def exposed_shutdown(self) -> dict[str, Any]
```

### Client-Side Methods

```
deep_worker_cli/client/agent_client.py

Line 152-164: _send_command() - status branch
Line 166-174: _send_command() - invoke branch
Line 176-184: _send_command() - get_history branch
```

### External Consumers

```
Web client session service (external consumer, not included here)
Line 457: status = await self.get_session_status_async(session_name)
Line 652: status_result = await session.get_status_async()

External LangChain bridge module (external consumer, not included here)
Line 343: messages = self.session.get_history()
Line 371: for msg in self.session.get_history():
```

---

## 12. References

- **Root Cause Analysis:** `/tmp/complete_rpyc_trace.txt`
- **Integration Test:** `tests/test_complete_rpyc_trace.py`
- **RPyC Documentation:** https://rpyc.readthedocs.io/en/latest/
- **Netref Explanation:** https://rpyc.readthedocs.io/en/latest/tutorial/tut3.html#netrefs

---

**END OF DESIGN DOCUMENT**
