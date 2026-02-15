# TaskManager TUI Fix - Intermediate Messages Display

## Problem

When TaskManager was enabled (`deep-worker agent_name -tm`), the interactive TUI (dwcli) would only show the final agent response, hiding all intermediate messages from TaskManager evaluation loop.

**Example:**
- User sends: "Complete this test task"
- TaskManager loop generates:
  1. User message: "Complete this test task"
  2. Assistant: "I finished 30%"
  3. TaskManager: "Please continue to 50%..."
  4. Assistant: "I finished 50%"
  5. TaskManager: "Please continue to 70%..."
  6. Assistant: "I finished 70%"
  7. TaskManager: "Please finish completely..."
  8. Assistant: "I finished 100%"

**TUI showed only:** "I finished 100%"
**TUI should show:** ALL 8 messages

## Root Cause

The interactive TUI (`InteractiveApp`) sent user message via IPC and received only the final response. All intermediate messages existed in the agent's conversation history (LangGraph checkpoint) but were never displayed in the TUI.

The `_agent_task()` method:
1. Sent user message
2. Received final response
3. Displayed ONLY the final response
4. Never reloaded conversation history

## Solution

Modified `deep_worker_cli/cli/interactive.py`:

### Change 1: Implement Incremental Message Loading

Added new method `_load_new_messages()` that loads ONLY new messages created since last check:

```python
async def _load_new_messages(self) -> None:
    """Load and display ONLY NEW messages created since last check.

    CRITICAL: This method implements incremental message loading to maintain
    interactivity. It NEVER reloads entire history - only fetches messages
    created after the last loaded message.
    """
    # Request ONLY messages created since last check
    # offset=_history_total means "skip messages we already have"
    response = await self.client.get_history(
        limit=50,
        offset=self._history_total  # Start from where we left off
    )

    new_messages = response.get("history", [])
    # Display each new message incrementally
    for msg_data in new_messages:
        # Create and mount appropriate widget
        ...

    # Update total count
    self._history_total = response.get("total", self._history_total)
```

### Change 2: Load Only New Messages After Agent Response

In `_agent_task()` method (line ~157):

**Before:**
```python
if response.get("success"):
    # Extract and display ONLY final response
    content = response.get("response", {}).get("content", "")
    msg = AssistantMessage(content)
    await self._mount_message(msg)
    await msg.write_initial_content()
```

**After:**
```python
if response.get("success"):
    # Load ONLY new messages created during agent processing
    # DO NOT reload entire history - this destroys interactivity!
    await self._load_new_messages()
    self._update_status("Ready")
```

### Change 3: Display User Message Immediately

In `_handle_user_message()` method (line ~236):

**Before:**
```python
# Don't display user message yet
self._update_status("Sending message...")
self.run_worker(self._agent_task(message))
```

**After:**
```python
# CRITICAL: Display user message IMMEDIATELY!
# User must see their message right away, not after agent completes.
await self._mount_message(UserMessage(message))

# Increment history total - we just added user message
self._history_total += 1

self._update_status("Sending message...")
self.run_worker(self._agent_task(message))
```

## How It Works Now

1. User types message in TUI
2. **TUI displays user message IMMEDIATELY** (before sending to agent)
3. TUI increments `_history_total` counter to track new message
4. Message sent to agent via IPC
5. Agent executes TaskManager evaluation loop:
   - Each iteration adds messages to LangGraph checkpoint
   - User prompts from TaskManager
   - Assistant responses
6. Agent returns final response to TUI
7. **TUI loads ONLY NEW messages created during step 5**
   - Uses `offset=_history_total` to skip already-displayed messages
   - Loads and displays each new message incrementally
   - Updates `_history_total` to new count
8. ALL messages now visible in TUI in real-time:
   - Original user message (displayed immediately in step 2)
   - All TaskManager prompts (loaded incrementally in step 7)
   - All assistant responses (loaded incrementally in step 7)

## Testing

### Manual Test
```bash
# Start agent with TaskManager
deep-worker test_tm -tm

# Connect with TUI
dwcli test_tm

# Send test message:
# "Test. If you are asked to finish the task fully or partially,
#  then you must first answer that you have completed the task 30%
#  (and wait for the next request), then 50%, then 70%,
#  then completely."

# Expected: TUI shows ALL intermediate messages (30%, 50%, 70%, 100%)
# Previously: TUI showed ONLY final message (100%)
```

### Verify via History Command
```bash
dwcli --history test_tm
# Should show all messages including TaskManager prompts
```

## Files Changed

- `deep_worker_cli/cli/interactive.py` - Modified `_agent_task()` and `_handle_user_message()`

## Related Issues

- TaskManager implementation: `deep_worker_cli/task_manager/`
- Agent process evaluation loop: `deep_worker_cli/process/agent_process.py`

## Future Improvements

Current implementation uses incremental history loading (polling after agent completes).
For even better real-time UX, consider implementing true streaming:

- Use streaming mode (`stream=True`) in `invoke_agent()`
- Send each TaskManager message as a separate chunk during execution
- Update TUI as messages are created (no waiting for completion)
- Display progress indicators for in-flight agent responses

This would provide instant feedback for long-running tasks with many TaskManager iterations,
showing messages as they're created rather than loading them after completion.
