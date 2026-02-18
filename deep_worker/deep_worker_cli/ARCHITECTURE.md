# Deep-Worker Architecture Principles

## 🏛️ CORE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH

### Overview

The agent deep-worker is the **ONLY** source of truth for **ALL** messages in the system, including user messages.

### Why This Matters

User messages can originate from multiple sources:
- dwcli instances (interactive terminal UI)
- Task manager sending commands
- Other clients connected to the same agent
- API calls
- Any other integration

**ALL** of these messages must be visible in **ALL** connected clients.

### Implementation Rule

```
┌─────────────────────────────────────────────────────────┐
│  EVERY message MUST flow through agent and be broadcast │
│  to ALL subscribers via RPyC callbacks.                 │
│                                                         │
│  NO client should display messages locally!             │
└─────────────────────────────────────────────────────────┘
```

### Message Flow

```
┌──────────────┐
│ Message      │ (from ANY source: dwcli, task manager, other client)
│ Origin       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Agent (deep-worker)                                      │
│                                                          │
│  1. Receives message                                     │
│  2. Broadcasts to ALL subscribers via                    │
│     _notify_subscribers()                                │
│  3. Processes message                                    │
│  4. Broadcasts assistant response                        │
└──────┬───────────────────────────────────────────────────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
   dwcli #1      dwcli #2    task manager    other client
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
            ALL clients display same messages
```

### Code Locations

#### Agent Service (`agent_service.py`)

```python
def exposed_invoke_agent(self, message: str, stream: bool = False):
    # CRITICAL: Notify ALL subscribers about user message
    self._notify_subscribers(
        {"type": "user", "content": message, "role": "user"}
    )

    # Process message...
    response = agent.process(message)

    # CRITICAL: Notify ALL subscribers about assistant response
    self._notify_subscribers(
        {"type": "assistant", "content": response, "role": "assistant"}
    )
```

#### dwcli Interactive UI (`interactive.py`)

```python
async def _handle_realtime_message(self, msg: dict[str, Any]) -> None:
    """
    This is the ONLY place where messages are displayed in dwcli!

    ALL messages (user + assistant) come via RPyC callbacks from agent.
    """
    role = msg.get("role")
    content = msg.get("content", "")

    if role == "user":
        await self._mount_message(UserMessage(content))
    elif role == "assistant":
        await self._mount_message(AssistantMessage(content))
```

**WRONG CODE (violates architecture):**

```python
async def send_message_to_agent(self, message: str):
    # ❌ WRONG: DO NOT display user message locally!
    await self._mount_message(UserMessage(message))

    # Send to agent
    await self.agent.invoke(message)
```

**CORRECT CODE:**

```python
async def send_message_to_agent(self, message: str):
    # ✅ CORRECT: Just send to agent
    # Message will be displayed when agent broadcasts it back
    await self.agent.invoke(message)
```

### Anti-Pattern: Local Display

**NEVER** display user messages locally in dwcli before sending to agent.

**Why?**
1. Violates single source of truth principle
2. Creates duplicate messages
3. Hides messages from other sources (task manager, other clients)
4. Causes inconsistent state across clients

### Testing Guidelines

When writing tests:

1. ✅ **DO** verify messages appear via callbacks
2. ✅ **DO** test with multiple clients
3. ✅ **DO** test messages from task manager
4. ❌ **DON'T** rely on local message display
5. ❌ **DON'T** mock RPyC callbacks (use real agent)

### Related Documents

- `tests/integration_ai/MESSAGE_DUPLICATION_ANALYSIS.md` - Duplication fix details
- `tests/integration_ai/DUPLICATION_ROOT_CAUSE_ANALYSIS.md` - Historical analysis
- `deep_worker_cli/rpyc_service/agent_service.py` - Agent service implementation
- `deep_worker_cli/cli/interactive.py` - dwcli UI implementation

### Preventing Regressions

To prevent future violations of this architecture:

1. **Code Review**: Check for local message display before agent invocation
2. **Comments**: All message-related code has architecture principle comments
3. **Documentation**: This file and inline comments explain the principle
4. **Tests**: Integration tests verify messages come via callbacks

---

## History Pagination

### Requirements

- On startup: load the **50 most recent** messages via `get_history(limit=50, offset=0)`.
- Older messages are loaded **only on demand** when the user scrolls up toward the top
  of the loaded window.
- Live messages are delivered exclusively via RPyC push callbacks — never by reloading
  history.

### Implementation: ChatScroll + watch_scroll_y

Pagination is **event-driven**, not polling-based.  The chat area uses a custom
`ChatScroll` widget (subclass of `VerticalScroll`) instead of plain `VerticalScroll`.

```
User scrolls up (mouse wheel / keyboard / scrollbar drag)
       │
       ▼
ChatScroll.watch_scroll_y(old, new)   ← fires on ALL scroll input sources
       │  new < old → scrolled up
       ▼
ChatScroll posts ScrolledUp message   ← bubbles up to InteractiveApp
       │
       ▼
InteractiveApp.on_chat_scroll_scrolled_up()
       │
       ▼
InteractiveApp._maybe_load_more_history()
       │
       ├─ Guard 1: _history_offset >= _history_total  → return (nothing left)
       ├─ Guard 2: _loading_history == True            → return (already loading)
       ├─ Guard 3: topmost visible widget index >= 10  → return (enough buffer)
       │
       ▼
InteractiveApp.run_worker(_load_more_history())
       │
       ▼
RPyC call: get_history(limit=20, offset=_history_offset)
```

### Why not ScrollUp event?

`textual.scrollbar.ScrollUp` fires **only on scrollbar arrow clicks**, not on mouse
wheel or keyboard.  `watch_scroll_y` fires on every scroll source.

### Why not polling (set_interval)?

The previous approach (`set_interval(0.5, _check_scroll_position)`) was disabled
because it triggered too eagerly: newly added messages temporarily moved `scroll_y`
near zero, causing fresh messages to be re-fetched as "old history" (duplication).

Event-driven pagination avoids this entirely: `watch_scroll_y` fires only when the
user actively scrolls, and the topmost-visible-index guard ensures a load is only
requested when the user is genuinely near the top of the loaded window.

### Topmost Visible Index

`widget.region.y` gives the widget's screen Y coordinate after layout:

- `region.y < 0`  → widget is scrolled above the visible viewport
- `region.y >= 0` → widget is at or below the top of the viewport

The first widget in `#messages` children with `region.y >= 0` is the topmost
visible message.  If its index is `< 10`, there are fewer than 10 already-loaded
messages buffered above it, and the next history page is fetched.

### Cached State (no extra RPyC calls)

`_history_offset` and `_history_total` are set by `_load_history()` on startup and
updated by `_load_more_history()` on each pagination.  The guard in
`_maybe_load_more_history()` reads these local counters — no network round-trip.

### Code Locations

| File | Symbol | Purpose |
|------|--------|---------|
| `cli/interactive.py` | `ChatScroll` | Detects upward scroll, posts `ScrolledUp` |
| `cli/interactive.py` | `ChatScroll.watch_scroll_y` | Direction detection |
| `cli/interactive.py` | `InteractiveApp.on_chat_scroll_scrolled_up` | Event handler |
| `cli/interactive.py` | `InteractiveApp._maybe_load_more_history` | Pagination gate |
| `cli/interactive.py` | `InteractiveApp._load_more_history` | Actual RPyC fetch |
| `tests/unit_tests/cli/test_chat_scroll_pagination.py` | — | Unit tests (TDD) |

---

## Summary

```
╔═══════════════════════════════════════════════════════════╗
║  REMEMBER: Agent is the ONLY source of truth!             ║
║                                                           ║
║  ALL messages → Agent → Broadcast to ALL clients          ║
║                                                           ║
║  NO local display of messages in clients!                 ║
╠═══════════════════════════════════════════════════════════╣
║  PAGINATION: event-driven via ChatScroll.watch_scroll_y   ║
║                                                           ║
║  Scroll up → ScrolledUp → _maybe_load_more_history        ║
║  (3 guards: cache / concurrent / topmost-index < 10)      ║
╚═══════════════════════════════════════════════════════════╝
```
