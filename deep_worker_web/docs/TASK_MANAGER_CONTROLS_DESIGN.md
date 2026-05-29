# TaskManager Controls Design: Disable & Reset

## Overview

Add two runtime controls for TaskManager in deep_worker_web UI:
1. **Temporary Disable/Enable** — toggle TM on/off without restarting agent
2. **Full Reset** — clear TM history and return to initial state

---

## Current Architecture

### TaskManager State Machine

```
                    init_task()
    'inactive' ──────────────────► 'waiting'
        ▲                              │
        │                              │ on_subordinate_response()
        │                              ▼
        │                        'evaluating'
        │                         │    │
        │    status='done'        │    │ status='in_progress'
        │    status='unable'      │    │
        └─────────────────────────┘    └──► 'waiting' (corrective prompt sent)
```

### Key Attributes (task_manager.py)

| Attribute | Type | Purpose |
|---|---|---|
| `_running` | `bool` | Lifecycle flag (True after `start()`, False after `stop()`) |
| `state` | `'inactive'\|'waiting'\|'evaluating'` | Current task evaluation state |
| `manager_agent` | `Pregel` | LangGraph agent for evaluation |
| `evaluation_count` | `int` | Number of evaluations in current task |
| `is_first_evaluation` | `bool` | Whether next eval is first (uses full prompt) |

### Manager Agent Identity

- **Thread ID**: `{agent_name}:tm` (e.g. `test_dw:tm`)
- **Checkpointer**: Same instance as subordinate agent (InMemorySaver or AsyncSqliteSaver)
- **Database**: `~/.deepagents/sessions.db` (shared with subordinate)
- **System prompt**: `TASK_MANAGER_SYSTEM_PROMPT` from `prompts.py`

### Message Accumulation

Each evaluation adds messages to the `{agent_name}:tm` thread:
- `HumanMessage`: evaluation prompt (initial_request + subordinate response)
- `AIMessage`: TM verdict (STATUS/REASONING/MESSAGE)

Messages accumulate across tasks indefinitely. No clearing mechanism exists.

### How _handle_invoke Uses TaskManager

```python
# Streaming mode (lines 984-1157):
if self.task_manager:
    await self.task_manager.init_task(initial_request=message, ...)

max_iterations = 10 if self.task_manager else 1
while iteration < max_iterations:
    # run agent...
    if self.task_manager:
        corrective = await self.task_manager.on_subordinate_response(response, ...)
        if corrective:
            current_message = corrective  # loop continues
        else:
            break  # done/unable
```

Key observation: `_handle_invoke` checks `self.task_manager` existence. If it's `None`, no TM logic runs. If it exists but `_running=False` or `state='inactive'`, `on_subordinate_response()` returns `None` immediately (line 201).

---

## Feature 1: Temporary Disable/Enable

### Concept

A toggle that makes the TaskManager **stop evaluating** without destroying it. The agent continues processing messages normally, but TM doesn't intervene after each response.

### Approach

Add a `disabled` flag to TaskManager. When `disabled=True`, `on_subordinate_response()` returns `None` immediately (same as `state='inactive'`). The TM instance, history, and agent remain intact.

### Changes by Layer

#### TaskManager (task_manager.py)

New attribute:
```python
self.disabled: bool = False
```

New methods:
```python
def disable(self) -> None:
    """Temporarily disable TaskManager evaluation."""
    self.disabled = True

def enable(self) -> None:
    """Re-enable TaskManager evaluation."""
    self.disabled = False
```

Modify `on_subordinate_response()` — add check at the top (before state check):
```python
if self.disabled:
    return None
```

Modify `init_task()` — skip if disabled:
```python
if self.disabled:
    return
```

#### RPyC Service (rpyc_service/agent_service.py)

New exposed method:
```python
async def exposed_set_task_manager_enabled(self, enabled: bool) -> str:
    """Enable or disable TaskManager at runtime.
    
    Returns JSON string: {"success": true, "enabled": bool}
    """
    if not self._agent_process or not self._agent_process.task_manager:
        return json.dumps({"success": False, "error": "TaskManager not configured"})
    
    if enabled:
        self._agent_process.task_manager.enable()
    else:
        self._agent_process.task_manager.disable()
    
    return json.dumps({"success": True, "enabled": enabled})
```

Update `exposed_get_status()` — include `task_manager_disabled`:
```python
if self._agent_process.task_manager is not None:
    result["task_manager_active"] = (
        self._agent_process.task_manager.state != "inactive"
    )
    result["task_manager_disabled"] = self._agent_process.task_manager.disabled
```

#### Web Service (deep_worker_web/app/services/agent_service.py)

New method:
```python
async def set_task_manager_enabled(
    self, agent_name: str, enabled: bool
) -> dict[str, Any]:
    """Enable or disable TaskManager for an agent."""
    root = self._roots[agent_name]
    # fire_and_forget_async root.set_task_manager_enabled(enabled)
```

Update `_update_agent_status` broadcast to include `task_manager_disabled`:
```python
if "task_manager_disabled" in response:
    broadcast_msg["task_manager_disabled"] = response["task_manager_disabled"]
```

#### REST API (routers/agents.py)

New endpoint:
```
POST /api/agents/{agent_name}/task-manager/toggle
Body: {"enabled": true|false}
Response: {"success": true, "enabled": true|false}
```

#### Frontend (main.js, index.html, style.css)

**HTML**: Toggle checkbox next to Stop TaskManager button:
```html
<label class="tm-toggle">
    <input type="checkbox" id="tm-enabled-checkbox" 
           checked onchange="toggleTaskManager(this.checked)">
    TaskManager
</label>
```

**JS**:
```javascript
function toggleTaskManager(enabled) {
    fetch(`/api/agents/${selectedAgent}/task-manager/toggle`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled}),
    });
}
```

Update `handleAgentStatus()` — sync checkbox state from WebSocket:
```javascript
if (data.task_manager_disabled !== undefined) {
    document.getElementById('tm-enabled-checkbox').checked = !data.task_manager_disabled;
}
```

**Visibility**: Toggle only visible when agent has TaskManager configured (detected from first `agent_status` broadcast containing `task_manager_active` or `task_manager_disabled`).

---

## Feature 2: Full Reset

### Concept

Completely reset TaskManager to initial state:
1. Clear all evaluation history from checkpoint
2. Reset internal state (`evaluation_count`, `is_first_evaluation`, `state`)
3. Manager agent starts fresh — next evaluation uses full initial prompt
4. System prompt remains (it's set at agent creation, not stored in history)

### Approach

The manager agent's history is stored in the checkpointer under thread `{agent_name}:tm`. To reset:
1. Delete all checkpoints for that thread ID
2. Reinitialize the manager agent instance
3. Reset all task state

### Changes by Layer

#### TaskManager (task_manager.py)

New method:
```python
async def reset(self) -> None:
    """Reset TaskManager to initial state.
    
    Clears evaluation history, reinitializes manager agent.
    After reset, TM behaves as if just started.
    """
    logger.info(f"[TaskManager] Resetting...")
    
    # 1. Force state to inactive
    self.state = 'inactive'
    self.initial_request = None
    self.project_path = None
    self.session_file_path = None
    self.is_first_evaluation = True
    self.evaluation_count = 0
    
    # 2. Clear checkpoint history for TM thread
    if self.checkpointer is not None:
        thread_id = f"{self.agent_name}:tm"
        await self._clear_thread_history(thread_id)
    
    # 3. Reinitialize manager agent (fresh instance)
    if self.manager_backend and hasattr(self.manager_backend, 'cleanup'):
        with contextlib.suppress(Exception):
            await self.manager_backend.cleanup()
    
    self.manager_agent = None
    self.manager_backend = None
    await self._initialize_manager()
    
    logger.info(f"[TaskManager] Reset complete")
```

New private method for clearing history:
```python
async def _clear_thread_history(self, thread_id: str) -> None:
    """Clear all checkpoint data for a specific thread."""
    if hasattr(self.checkpointer, 'conn'):
        # AsyncSqliteSaver — clear via SQL
        await self.checkpointer.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        await self.checkpointer.conn.execute(
            "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
        )
        await self.checkpointer.conn.commit()
    # InMemorySaver — reinitializing manager_agent is sufficient
    # (old thread data becomes unreachable with new agent instance)
```

Note on system prompt: The system prompt is passed to `create_cli_agent()` during `_initialize_manager()`. It's not stored in checkpoint history — it's baked into the agent graph. So reinitializing the agent restores the system prompt automatically.

#### RPyC Service (rpyc_service/agent_service.py)

New exposed method:
```python
async def exposed_reset_task_manager(self) -> str:
    """Reset TaskManager to initial state.
    
    Clears evaluation history, reinitializes manager agent.
    Returns JSON string: {"success": true}
    """
    if not self._agent_process or not self._agent_process.task_manager:
        return json.dumps({"success": False, "error": "TaskManager not configured"})
    
    try:
        await self._agent_process.task_manager.reset()
        return json.dumps({"success": True, "message": "TaskManager reset to initial state"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
```

#### Web Service (deep_worker_web/app/services/agent_service.py)

New method:
```python
async def reset_task_manager(self, agent_name: str) -> dict[str, Any]:
    """Reset TaskManager for an agent."""
    root = self._roots[agent_name]
    # fire_and_forget_async root.reset_task_manager()
```

#### REST API (routers/agents.py)

New endpoint:
```
POST /api/agents/{agent_name}/task-manager/reset
Body: (none)
Response: {"success": true, "message": "TaskManager reset to initial state"}
```

#### Frontend (main.js, index.html)

**HTML**: Reset button next to toggle:
```html
<button id="reset-tm-btn" class="btn btn-secondary btn-sm"
        onclick="resetTaskManager()" title="Reset TaskManager history">
    🔄 Reset TM
</button>
```

**JS**:
```javascript
async function resetTaskManager() {
    if (!confirm('Reset TaskManager? This clears all evaluation history.')) return;
    
    const response = await fetch(
        `/api/agents/${selectedAgent}/task-manager/reset`,
        {method: 'POST'}
    );
    if (response.ok) {
        showNotification('TaskManager reset', 'success');
    }
}
```

**Visibility**: Only visible when agent has TaskManager configured.

---

## UI Layout

```
┌─────────────────────────────────────────────────────────┐
│ agent-header                                             │
│  Agent Name              [Full Dialog] [Stop TM]         │
│                                                          │
│  ☑ TaskManager   [🔄 Reset TM]                          │
│                                                          │
│  Provider: deepseek  Model: deepseek-v4-flash                │
├─────────────────────────────────────────────────────────┤
│ Last Response / Approval Panel / etc.                    │
└─────────────────────────────────────────────────────────┘
```

- **Checkbox "TaskManager"**: Checked = enabled (default), Unchecked = disabled
- **"Reset TM" button**: Resets history, always available when TM configured
- **"Stop TM" button**: Stops current task evaluation (existing functionality)
- All three controls hidden when agent has no TaskManager

---

## WebSocket Protocol Updates

### Status Broadcast (extended)

```json
{
    "type": "agent_status",
    "agent_name": "test_dw",
    "status": "working",
    "task_manager_active": true,
    "task_manager_disabled": false
}
```

- `task_manager_active`: Present only if TM configured. True when evaluating.
- `task_manager_disabled`: Present only if TM configured. True when temporarily disabled.

### TM Reset Notification

Not strictly needed — status broadcast will reflect the change on next poll (5s). If faster feedback is desired, the reset endpoint can return success synchronously.

---

## Files to Change

### deep_worker_cli (agent side)

| File | Change |
|---|---|
| `task_manager/task_manager.py` | Add `disabled` flag, `disable()`, `enable()`, `reset()`, `_clear_thread_history()` |
| `rpyc_service/agent_service.py` | Add `exposed_set_task_manager_enabled()`, `exposed_reset_task_manager()`, update `exposed_get_status()` |

### deep_worker_web (web side)

| File | Change |
|---|---|
| `app/services/agent_service.py` | Add `set_task_manager_enabled()`, `reset_task_manager()`, update broadcast |
| `app/routers/agents.py` | Add `POST /{name}/task-manager/toggle`, `POST /{name}/task-manager/reset` |
| `static/index.html` | Add checkbox + reset button |
| `static/js/main.js` | Add `toggleTaskManager()`, `resetTaskManager()`, sync checkbox from WS |
| `static/css/style.css` | Styles for TM controls |

### Tests

| File | Tests |
|---|---|
| `tests/unit_tests/task_manager/test_disable.py` | `disable()`, `enable()`, `on_subordinate_response` returns None when disabled |
| `tests/unit_tests/task_manager/test_reset.py` | `reset()` clears state, history, reinitializes agent |
| `tests/rpyc_service/test_agent_service_tm_controls.py` | `exposed_set_task_manager_enabled()`, `exposed_reset_task_manager()` |
| `tests/unit/test_tm_controls.py` (web) | REST endpoints, WebSocket broadcast |

---

## Implementation Order

1. **TaskManager core**: `disabled` flag + `disable()`/`enable()` + `reset()` + `_clear_thread_history()`
2. **RPyC methods**: `exposed_set_task_manager_enabled()` + `exposed_reset_task_manager()` + update `exposed_get_status()`
3. **Web service**: `set_task_manager_enabled()` + `reset_task_manager()` + broadcast updates
4. **REST endpoints**: toggle + reset
5. **Frontend**: checkbox + button + WS sync
6. **Tests**: TDD at each layer

---

## Edge Cases

| Case | Behavior |
|---|---|
| Disable TM while it's evaluating | `on_subordinate_response()` checks `disabled` at entry, returns `None`. Current eval may complete but next won't start. |
| Reset TM while it's evaluating | `reset()` forces `state='inactive'`, clears history, reinitializes agent. Current eval loop in `_handle_invoke` will see `corrective_prompt=None` and break. |
| Enable TM after disable | Next `invoke_agent` call will use TM normally. No special action needed. |
| Reset TM for agent without TM | RPyC method returns `{"success": false, "error": "TaskManager not configured"}`. |
| Multiple rapid resets | Each reset reinitializes the agent. Safe but wasteful. UI should debounce. |
| Agent restart after disable | `disabled` flag is not persisted. Agent restarts with TM enabled (default). |
