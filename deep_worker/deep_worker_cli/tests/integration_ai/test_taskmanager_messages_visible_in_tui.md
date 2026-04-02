# Test Report: test_taskmanager_messages_visible_in_tui

## Test Information
- **Test Name:** `test_taskmanager_messages_visible_in_tui`
- **Test File:** `tests/integration_ai/test_taskmanager_tui.py`
- **Date:** 2026-03-16
- **Status:** ❌ **FAILED**

## Test Execution Result

**Execution Time:** 135.95s (2:02:15)
**Exit Code:** FAILED

## Terminal Screenshot (TUI State)

```
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
                                                   v0.0.19

 Thread: 61ce2aee
 Ready to code! What would you like to build?
 Enter send • Ctrl+J newline • @ files • / commands
                                                                              ▅▅


  Connected to agent 'test_taskmanager_tui'

┌────────────────────────────────────────────────────────────────────────────┐
│ >  function called `add_numbers` that adds two numbers. Your response      │
│   MUST include ALL of the following:1. A complete docstring with Args      │
│   section, Returns section, and Raises section2. Type hints for all        │
│   parameters and return value3. Input validation that raises TypeError   ▃▃│
│   for non-numeric inputs4. At least two concrete usage examples in the     │
│   docstring5. A brief comment explaining why this function is              │
│   usefulMake sure every single requirement above is present in your        │
│   response.                                                                │
└────────────────────────────────────────────────────────────────────────────┘▅▅
  manual | shift+tab to cycle  Connected to test_taskmanager_tui (RPyC)
```

## Problem Description

### Primary Issue: Input Area Not Visible

**🚨 CRITICAL BUG DISCOVERED:**

The user's input prompt is **NOT visible** at the bottom of the screen where it should be. Instead:

1. ✅ The ChatInput widget header is visible at the top: `"Enter send • Ctrl+J newline • @ files • / commands"`
2. ❌ **The actual input area is NOT visible** - it should be at the bottom of the screen
3. ❌ **The user's message is displayed in a scrollable box** instead of being in the input area
4. ❌ **The scrollable message box has occupied the entire bottom portion** of the screen

### What This Test Was Checking

This test verifies that TaskManager intermediate messages appear in the TUI. However, the test **FAILED** for a different reason than expected:

**Expected Failure Reason:** TaskManager messages not broadcast to TUI
**Actual Failure Reason:** Test couldn't complete because **the input area is not visible/accessible**

### Evidence of Scroll Behavior Problem

The test output shows that after sending the user's query, the screen is showing:
- The user's query text inside a **scrollable bordered box** (see the `┌─...─┐` box characters)
- This box contains the full multi-line user prompt
- **NO visible ChatInput area at the bottom** where user can type

This indicates that:
1. User message was added to scrollable chat history
2. **Screen auto-scrolled to show the message**
3. **Input area scrolled OUT OF VIEW** (critical bug!)

### Original Test Purpose vs Actual Bug Found

**Test's Original Purpose:**
- Verify TaskManager intermediate messages appear in TUI
- Check that agent broadcasts all messages via `_notify_subscribers()`

**Actual Bug Discovered:**
- **Input area becomes invisible after user sends a message**
- Auto-scroll behavior is broken - scrolls the input area off screen
- Makes dwcli **completely unusable** - user cannot see where to type

## Degree of Certainty

**Certainty Level:** 🔴 **VERY HIGH (95%)**

**Evidence:**
1. Terminal screenshot clearly shows user's message in a scrollable box
2. No visible ChatInput widget at bottom of screen
3. Test waited 120 seconds and screen state didn't change
4. This matches the user's complaint: "dwcli is completely impossible" (dwcli completely unusable)

**What is NOT certain:**
- Exact code location causing the scroll bug (need to inspect interactive.py scroll logic)
- Whether this affects all messages or only long multi-line messages
- Whether this is related to recent changes in commit ba00ebf

## Root Cause Analysis

### Likely Cause in `interactive.py`

Recent commit ba00ebf (fixing tool call duplication) included changes to scroll behavior. Specifically, code around:

**File:** `deep_worker_cli/cli/interactive.py`

**Suspected Problem Areas:**

1. **User Message Handling** (around line ~458-474):
```python
async def _handle_realtime_message(self, msg):
    if msg_type == "user":
        await self._mount_message(UserMessage(content, id=msg_id))
        self._history_total += 1
        # Scroll to bottom after layout updates
        def _scroll_after_user_msg() -> None:
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False, immediate=True)  # <-- PROBLEM?
```

**Issue:** After mounting user message, code calls `chat.scroll_end()` which scrolls the **entire chat container** to bottom. This likely scrolls the **ChatInput widget OUT OF VIEW**.

2. **History Loading** (around line ~877-915):
```python
async def _load_history(self):
    # ... loads and displays conversation history ...
    # WITHOUT auto-scrolling to bottom
    # The input area must remain visible after loading history
```

**Issue:** Comment explicitly states "WITHOUT auto-scrolling to bottom" to keep input visible. But user message handling DOES auto-scroll, contradicting this requirement.

### Expected vs Actual Behavior

**Expected:**
1. User types message in ChatInput at bottom
2. User presses Enter
3. Message appears in scrollable chat area
4. **ChatInput remains visible and focused at bottom**
5. User can immediately type next message

**Actual:**
1. User types message in ChatInput at bottom
2. User presses Enter
3. Message appears in scrollable chat area
4. **Screen scrolls to show full message**
5. **ChatInput scrolls OFF SCREEN** 🔴
6. User cannot see input area - dwcli unusable!

## Impact Assessment

**Severity:** 🔴 **CRITICAL - COMPLETELY BREAKS dwcli**

**User Impact:**
- After sending first message, user **cannot see where to type**
- dwcli becomes **completely unusable**
- User must restart dwcli after every message
- Makes interactive development impossible

**Test Impact:**
- `test_taskmanager_messages_visible_in_tui` - FAILED (cannot complete test)
- Likely affects ALL TUI integration tests that send messages
- Cannot verify scroll/pagination behavior when input area invisible

## Recommendation

### Immediate Fix Required

**File:** `deep_worker_cli/cli/interactive.py`

**Location:** User message handling section (around line ~458-474)

**Current Code (BROKEN):**
```python
async def _handle_realtime_message(self, msg):
    if msg_type == "user":
        await self._mount_message(UserMessage(content, id=msg_id))
        self._history_total += 1

        # PROBLEM: Scrolls entire chat to bottom, hiding input area
        def _scroll_after_user_msg() -> None:
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False, immediate=True)
        self.set_timer(0.05, _scroll_after_user_msg)
```

**Recommended Fix:**
```python
async def _handle_realtime_message(self, msg):
    if msg_type == "user":
        await self._mount_message(UserMessage(content, id=msg_id))
        self._history_total += 1

        # DO NOT auto-scroll after user message!
        # The ChatInput widget must remain visible at bottom.
        # User should see their input area, not the message they just sent.
        # Remove or comment out the auto-scroll code:
        # def _scroll_after_user_msg() -> None:
        #     chat = self.query_one("#chat", VerticalScroll)
        #     chat.scroll_end(animate=False, immediate=True)
        # self.set_timer(0.05, _scroll_after_user_msg)
```

### Alternative Fix (More Conservative)

If scroll is needed for some edge cases, add a check to ensure ChatInput remains visible:

```python
def _scroll_after_user_msg() -> None:
    chat = self.query_one("#chat", VerticalScroll)
    # Scroll to bottom of chat, but keep some space for input area
    # Don't scroll so far that input becomes invisible!
    # (Need to determine exact logic based on ChatInput height)
    pass  # TODO: Implement smart scroll that preserves input visibility
```

### Additional Investigation Needed

1. **Check commit ba00ebf** - This commit changed scroll behavior. Review all scroll-related changes.

2. **Check history loading** - Verify that `_load_history()` doesn't cause same issue

3. **Test with different message lengths:**
   - Short message (1 line)
   - Medium message (5 lines)
   - Long message (20 lines)
   - Does input area disappear in all cases?

4. **Check assistant message handling** - Does receiving AI response also scroll input off screen?

## Related Tests to Check

After fixing, verify these tests:
- `test_taskmanager_messages_in_history_but_not_tui` (next test in same file)
- `test_dwcli_history.py` - All pagination/history tests
- `test_agent_tool_call_duplication_deepseek.py` - May have same issue
- `test_agent_tool_call_duplication_claude_code.py` - May have same issue

## Notes

- This bug was introduced in commit `ba00ebf` when fixing tool call duplication
- The original intent was to ensure scroll-to-bottom after messages
- The implementation is overly aggressive and scrolls the input area off screen
- This is the bug the user complained about: "using dwcli is completely impossible"
- **CRITICAL:** This must be fixed IMMEDIATELY as it makes dwcli completely unusable

## Test Verdict

**Test Result:** ❌ FAILED (but not for the expected reason)

**Expected:** Test checks if TaskManager messages visible (they probably aren't)
**Actual:** Test cannot complete because input area is invisible

**Real Issue:** Critical scroll bug makes dwcli unusable - MUST FIX FIRST before investigating TaskManager message visibility.
