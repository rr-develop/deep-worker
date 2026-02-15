# Investigation: test_dwcli_query_visibility Sandbox Issue

**Date**: 2026-02-11
**Status**: Root Cause Identified
**Issue**: dwcli does not process input from PTY-emulated terminal session

---

## Summary

The test infrastructure is **NOT sandboxed** and runs correctly on the host. However, there is a fundamental issue with how `dwcli` handles input from emulated PTY (pseudo-terminal) sessions.

---

## What We Discovered

### ✅ What Works

1. **Agent Process Management**
   - `subprocess.Popen(["deep-worker", ...])` successfully spawns agent
   - Agent daemonizes and registers with ProcessRegistry
   - Registration happens within 0.0-0.1 seconds
   - Agent process runs properly in background

2. **dwcli Connection**
   - `DwcliTerminalSession` successfully starts dwcli
   - dwcli connects to agent (shows "Connected to test_dwcli_query_visibility")
   - UI renders correctly (shows DEEP AGENTS header and prompt)
   - Terminal emulation with pyte works

3. **Test Infrastructure**
   - No sandboxing occurs
   - Agent runs directly on host
   - Registry files in `/tmp` work correctly
   - Socket IPC communication functions properly

### ❌ What Fails

**dwcli does NOT process keyboard input from PTY emulation**

#### Evidence

```python
# After sending query via PTY:
session.send_input("What is 2+2?", press_enter=False)
time.sleep(0.3)
session.send_key("enter")

# Screen remains unchanged:
# "Ready to code! What would you like to build?"
# Input box still shows ">" with NO text
```

The query is NOT visible in the input field after being sent. The screen doesn't change at all.

#### Comparison

- **deepagents**: Query appears in input field immediately, gets processed, AI responds
- **dwcli**: Query does NOT appear in input field, remains "Ready to code" screen

---

## Root Cause Analysis

### The Problem

`dwcli` UI implementation has an issue with PTY input handling:

1. **Input Not Captured**: Text sent to PTY is not captured by dwcli's input handler
2. **No Visual Feedback**: Query text doesn't appear in the input box
3. **No Processing**: Since input isn't registered, agent never receives request
4. **UI Frozen**: Screen shows welcome message indefinitely

### Why deepagents Works

`deepagents` properly:
- Reads from PTY raw input
- Displays typed characters in real-time
- Sends input to agent when Enter is pressed
- Processes responses and displays them

### Why dwcli Fails

`dwcli` implementation appears to:
- NOT read from PTY correctly
- NOT display typed characters
- NOT send input to agent
- Have different input handling than deepagents

---

## Evidence

### Test Output Timeline

```
INFO: Sending query: 'What is 2+2?'
INFO: ✓ Query sent to dwcli

[0.5 seconds later]
INFO: SCREEN RIGHT AFTER QUERY
      (Shows "Ready to code! What would you like to build?")
      (Input field is empty, shows only ">")

INFO: ✓ AI response received after 0.0 seconds
      (triggered by presence of ">", which always exists!)

INFO: Waiting for response... (but none comes)
INFO: Neither query nor response are present
```

### Manual Testing

```bash
$ timeout 5 dwcli test_debug
# Shows: "Ready to code! What would you like to build?"
# Input: (we type query)
# Result: Nothing appears in input field
# No request sent to agent
```

---

## Why This Is NOT A Sandbox Issue

This is a **dwcli implementation issue**, not a sandbox issue, because:

1. ✅ Agent runs directly on host (not sandboxed)
2. ✅ Registry files in `/tmp` are accessible
3. ✅ Socket IPC works fine
4. ✅ dwcli can start and connect
5. ✅ deepagents works fine with same test infrastructure
6. ❌ Only dwcli input handling is broken

---

## Next Steps

### To Fix dwcli Input Handling

1. **Examine dwcli input handler code**
   - Look at how it reads from stdin/PTY
   - Compare with deepagents implementation
   - Check for raw mode setup issues

2. **Verify PTY interaction**
   - Ensure dwcli enables raw mode on terminal
   - Check for proper signal handling
   - Verify input buffer management

3. **Add debug logging**
   - Log when input is received
   - Log when characters appear in input box
   - Track the input-to-agent flow

4. **Test with different input methods**
   - Direct keyboard input (manual)
   - stdin redirection
   - TTY device
   - PTY emulation

---

## Files Involved

- `tests/integration_ai/test_terminal_emulator.py:test_dwcli_query_visibility`
- `tests/integration_ai/terminal_emulator.py:DwcliTerminalSession`
- `dwcli` binary (Python script)

---

## Conclusion

The test infrastructure and agent management are **working correctly**. The issue is specifically with `dwcli`'s inability to handle input from a PTY-emulated terminal session.

This is an **application-level issue with dwcli**, not an infrastructure/sandbox issue.

---

**Status**: Ready for dwcli investigation and fix
