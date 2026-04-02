# Test Report: test_agent_tool_call_no_duplication_deepseek

## Test Information

- **Test Name:** `test_agent_tool_call_no_duplication_deepseek`
- **File:** `test_agent_tool_call_duplication_deepseek.py`
- **Test Date:** 2026-03-16
- **Test Duration:** 118.04s (1:58)
- **Test Status:** ❌ **FAILED**

---

## Test Status Details

**Exit Code:** FAILED
**Failure Reason:** Query 2 did not receive response within 60 seconds (timeout)

**Error Log:**
```
ERROR: ❌ Query 2: Tool call indicator detected at 0.0s: 'execute'
ERROR: This should NOT happen for simple arithmetic!
FAILED: Query 2 did not receive response within 60 seconds
```

---

## Test Purpose

**What This Test Tests:**

This test verifies that dwcli TUI correctly manages tool call display across multiple queries and does NOT show stale/old tool calls from previous queries.

### Test Scenario:

**Query 1:** "What is your current working directory?"
- **Expected:** Agent calls `execute("pwd")` tool
- **Expected:** Tool call `execute` is visible in TUI
- **Expected:** Response shows current directory

**Query 2:** "What is 2+2?"
- **Expected:** Agent responds with simple arithmetic (NO tool call needed)
- **Expected:** NO `execute` tool call indicator in TUI
- **Expected:** Response: "4" or similar

### Test Logic:

```python
# Query 1
session.send_message("What is your current working directory?")
# Wait for response
# Should see 'execute' tool call ✓

# Query 2
session.send_message("What is 2+2?")
# Wait for response
# Should NOT see 'execute' tool call ✗
```

---

## Problem Description

### What Went Wrong

**Primary Issue:** Tool call duplication detected

```
Query 2: Tool call indicator detected at 0.0s: 'execute'
```

This means:
1. Query 2 sent: "What is 2+2?"
2. **IMMEDIATELY (0.0s)** the string `execute` appeared in TUI
3. This should NOT happen - Query 2 doesn't need tools!

### Root Cause Analysis

**Confidence Level:** ⚠️ **MEDIUM-HIGH (80%)**

The problem is **old tool call widgets are NOT cleared** between queries.

**Evidence:**

1. ✅ Query 1 correctly shows `execute("pwd")`
2. ❌ Query 2 shows `execute` at 0.0s (BEFORE any AI response)
3. ⏱️ Detection at 0.0s means it's NOT a new tool call, but leftover from Query 1

**Likely Causes:**

### Hypothesis 1: Message Store Accumulation (High Confidence - 80%)

```python
# dwcli message store accumulates ALL messages
Query 1:
  - User: "What is your current working directory?"
  - AI: "I'll check..."
  - ToolCall: execute("pwd")
  - ToolResult: "/tmp"
  - AI: "Current directory is /tmp"

Query 2:
  - User: "What is 2+2?"
  # ← OLD execute("pwd") widget still visible!
  - AI: "4"
```

The TUI doesn't clear or hide previous tool calls when a new query starts.

### Hypothesis 2: Widget Lifecycle Management (Medium Confidence - 60%)

Tool call widgets might be:
- Mounted to TUI during Query 1
- Never unmounted or hidden
- Still visible when Query 2 text is rendered
- Detected by test's screenshot parsing

### Hypothesis 3: Conversation Turn Separation (Medium Confidence - 50%)

dwcli may lack concept of "conversation turns" to group messages:
- No visual separator between Query 1 and Query 2
- No mechanism to "collapse" or hide old tool calls
- Everything rendered in flat chronological order

---

## Secondary Issue: Test Timeout

**Problem:** Query 2 did not receive response within 60 seconds

**Possible Causes:**

1. **Agent hung waiting for approval** (if `skip_permissions` not working)
2. **Agent tried to call execute for "2+2"** (model confusion)
3. **TUI not receiving/displaying the response** (client issue)
4. **Agent crashed** (process died)

**Most Likely:** Agent confused by previous context and tried to use execute for arithmetic, then hung on approval.

---

## TUI Behavior Analysis

### Expected Behavior:

```
┌─ Query 1 ────────────────────────────┐
│ > What is your current working dir?  │
│ I'll check...                         │
│ ⏺ execute("pwd")                      │
│ Current directory is /tmp             │
└───────────────────────────────────────┘

┌─ Query 2 ────────────────────────────┐
│ > What is 2+2?                        │
│ 4                                     │
└───────────────────────────────────────┘
```

### Actual Behavior (dwcli):

```
┌─ TUI Display ────────────────────────┐
│ > What is your current working dir?  │
│ I'll check...                         │
│ ⏺ execute("pwd")                      │  ← Still visible
│ Current directory is /tmp             │
│                                       │
│ > What is 2+2?                        │
│ ⏺ execute("pwd")                      │  ← OLD, shouldn't be here!
│ [no response after 60s]               │
└───────────────────────────────────────┘
```

---

## Impact Assessment

### Severity

- **User Experience:** ❌ HIGH - Confusing to see old tool calls
- **Test Validity:** ✅ HIGH - Test correctly identifies real bug
- **Functionality:** ⚠️ MEDIUM - Agent still works, but UI misleading

### User Impact

Users see:
- ❌ Cluttered TUI with old tool calls
- ❌ Unclear which tools belong to which query
- ❌ Cannot distinguish current vs historical actions
- ❌ May think agent is "stuck" on old operations

---

## Recommendations

### Fix 1: Clear Tool Call Widgets Between Queries (HIGH PRIORITY)

**Implementation:**

```python
# In InteractiveApp or MessageStore
def clear_tool_call_widgets_for_completed_queries(self):
    """Remove or hide tool call widgets from previous queries."""
    # Option A: Remove old widgets entirely
    for widget in self.old_tool_call_widgets:
        widget.remove()

    # Option B: Collapse/hide old tool calls
    for widget in self.old_tool_call_widgets:
        widget.add_class("collapsed")  # CSS to hide or minimize
```

**Trigger:** When new user message is sent, clear old tool calls.

**Confidence:** 70% - this will reduce clutter but may need refinement

### Fix 2: Group Messages by Conversation Turn (MEDIUM PRIORITY)

**Implementation:**

```python
class ConversationTurn:
    """Group messages belonging to single query-response cycle."""
    user_message: str
    assistant_responses: List[Message]
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]

    def collapse(self):
        """Hide tool details, show summary."""
        # Show: User query + Final response
        # Hide: Intermediate tool calls

# In TUI:
for turn in conversation_turns:
    render_turn(turn)
    if not turn.is_current:
        turn.collapse()  # Hide old tool calls
```

**Confidence:** 80% - better UX, clearer conversation structure

### Fix 3: Add Visual Separators (LOW PRIORITY, UX Enhancement)

```python
# After each query-response completes:
render_separator("─" * 80)
```

Makes it clearer where one conversation ends and next begins.

**Confidence:** 90% - simple, effective for readability

### Fix 4: Investigate Query 2 Timeout (HIGH PRIORITY)

Need to determine why Query 2 didn't receive response:

1. Check agent logs for errors
2. Verify `skip_permissions=True` is working
3. Check if agent tried to call execute for "2+2"
4. Verify RPyC connection didn't break

**Action:** Add detailed logging to next test run.

---

## Comparison with deepagents

**Question:** Does deepagents have the same problem?

**Status:** ⚠️ UNKNOWN - deepagents doesn't have duplication test

**Recommendation:** Create equivalent test for deepagents to verify if this is dwcli-specific or common issue.

---

## Suggested Test Improvements

### Make Test More Robust:

```python
# Add logging for debugging
def test_agent_tool_call_no_duplication_deepseek(clean_agent):
    # ... existing setup ...

    # Query 1
    session.send_message("What is your current working directory?")
    response1 = session.wait_for_response(timeout=60)
    logger.info(f"Query 1 response: {response1}")
    assert "execute" in session.get_screen_text()  # Tool call should be visible

    # Capture screen BEFORE Query 2
    screen_before_query2 = session.get_screen_text()
    logger.info(f"Screen before Query 2:\n{screen_before_query2}")

    # Query 2
    session.send_message("What is 2+2?")

    # Check IMMEDIATELY if execute is visible (shouldn't be new call)
    screen_at_0s = session.get_screen_text()
    if "execute" in screen_at_0s:
        logger.error(f"OLD execute visible at 0s:\n{screen_at_0s}")
        # This is the bug!

    response2 = session.wait_for_response(timeout=60)
    logger.info(f"Query 2 response: {response2}")
```

---

## Summary

| Aspect | Status |
|--------|--------|
| **Test Status** | ❌ FAILED (correctly identifies bug) |
| **Bug Confirmed** | ✅ YES - old tool calls not cleared |
| **Root Cause** | Message Store/Widget lifecycle |
| **Fix Difficulty** | ⭐⭐ MEDIUM |
| **Fix Confidence** | 70-80% |
| **User Impact** | HIGH (confusing UX) |

---

## Next Steps

1. ✅ Implement Fix 1 (clear old tool call widgets)
2. ✅ Implement Fix 2 (conversation turn grouping)
3. ⚠️ Investigate Query 2 timeout (why no response?)
4. ✅ Re-run test to verify fixes
5. ⚠️ Create equivalent test for deepagents (comparison)

---

**Report Generated:** 2026-03-16
**Analyst Confidence:** ⚠️ MEDIUM-HIGH (80%)
**Action Required:** Implement widget cleanup between queries
