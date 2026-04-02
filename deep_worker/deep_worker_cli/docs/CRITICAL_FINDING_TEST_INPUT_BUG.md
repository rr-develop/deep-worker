# CRITICAL FINDING: Test Input Mechanism is Broken

**Date**: 2026-03-14 01:00
**Status**: ROOT CAUSE IDENTIFIED
**Priority**: CRITICAL - Blocks all integration testing

## Executive Summary

Tool call display tests fail NOT because of streaming bugs, but because **the test never sends the Enter key to the TUI!**

The text appears in the input box but is never submitted to the agent.

---

## Evidence

### 1. Test Screenshot Analysis

After calling `session.send_input(query, press_enter=True)`, screenshot shows:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ > Please call the bash tool to execute 'sleep 10' command, then say hello  │
│   to me. Make sure you use the bash too                                    │▃▃
│   l first.                                                                 │
```

**Problem**: Text is in the input dialog, but NO response appears even after 12+ seconds.

### 2. No Logging from Message Handling Code

Created comprehensive logging in:
- `/tmp/dwcli_handle_user_message.log` - NOT CREATED
- `/tmp/dwcli_send_message.log` - NOT CREATED
- `/tmp/dwcli_rpyc_service.log` - NOT CREATED
- `/tmp/dwcli_invoke_debug.log` - NOT CREATED
- `/tmp/dwcli_astream_debug.log` - NOT CREATED

**Conclusion**: NONE of the message handling code executed. The Enter key never reached the TUI.

### 3. Logs That DO Exist

- `/tmp/dwcli_interactive_IMPORT.log` - Interactive app module imported ✓
- Agent started successfully ✓
- TUI connected via RPyC ✓
- Status bar shows "Connected to test_slow_tool_display (RPyC)" ✓

**Conclusion**: Everything BEFORE input submission works correctly.

---

## Root Cause

**File**: `tests/integration_ai/terminal_emulator.py:292-327`

```python
def send_input(self, text: str, press_enter: bool = False) -> None:
    """Send input to dwcli terminal."""
    if press_enter:
        text = text + "\n"  # ← Adds newline

    encoded = text.encode("utf-8")
    bytes_written = os.write(self.master_fd, encoded)  # ← Writes to PTY

    time.sleep(0.1)  # ← Brief wait
    output = self._read_available_output()
```

**Problem**: The `\n` character is written to the PTY, but the TUI (Textual framework) is not recognizing it as an Enter key press.

**Likely causes**:
1. PTY input buffering not flushing
2. Textual key bindings expecting specific Enter key code (not just `\n`)
3. Input widget not focused
4. Race condition between write and TUI ready state

---

## Why This Was Missed

Previous investigations assumed test input was working and focused on:
1. ✅ Stream processing (fixed buffer_key logic)
2. ✅ Threading model (already async)
3. ❌ Never verified test input actually reaches the agent!

**Lesson**: Always verify test infrastructure FIRST before debugging application code.

---

## Impact

**ALL integration tests using `send_input()` may be affected!**

This includes:
- `test_slow_tool_display.py` - FAILS (input not sent)
- `test_deepagents_slow_tool_display.py` - Uses different input method (works)
- Any other test using `DwcliTerminalSession.send_input()`

---

## Next Steps

### Immediate Fix Options

**Option 1: Use explicit key code**
```python
# Instead of "\n", send actual Enter key code
import termios
import fcntl
self.send_key("enter")  # Already exists but may not work
```

**Option 2: Send Control+J (line feed)**
```python
# Ctrl+J is explicit newline in many terminals
os.write(self.master_fd, b"\x0a")
```

**Option 3: Match deepagents test approach**
```python
# deepagents test uses:
session.send_input(query, press_enter=False)
time.sleep(0.3)
session.send_key("enter")
```

### Verification Steps

1. Add debug logging to ChatInput widget to see if key events arrive
2. Test with manual `dwcli` invocation to verify Enter works
3. Compare PTY codes sent by working vs failing tests
4. Check Textual event loop for dropped events

---

## Files to Investigate

**Test infrastructure**:
- `tests/integration_ai/terminal_emulator.py:292-327` - send_input() method
- `tests/integration_ai/terminal_emulator.py:328-365` - send_key() method

**TUI input handling**:
- `deep_worker_cli/widgets/chat_input.py` - ChatInput widget key bindings
- `deep_worker_cli/cli/interactive.py:1043-1060` - on_chat_input_submitted handler

**Textual framework**:
- Check if Textual has specific requirements for PTY input
- Review Textual Input widget documentation for key binding

---

## Conclusion

**The stream processing fixes were CORRECT but UNTESTED** because the test input mechanism is broken.

**Priority**: Fix `send_input()` to properly submit text to TUI, THEN rerun tests to verify stream processing works.

**Expected outcome after fix**: Test should pass, showing tool calls appear within 1 second.
