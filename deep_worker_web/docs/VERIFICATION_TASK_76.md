# Verification: a related task — already implemented

## Feature requested

Pull `get_status()` on every rpyc reconnect to heal lost `status_change` pushes (§7.1 of a related internal incident analysis (not included here)).

## Evidence — the code already does this

| Requirement | Current code | Evidence |
|---|---|---|
| `get_status()` called on every reconnect | ✅ `connect_to_agent()` line 553 calls `_update_agent_status()`, which fires `root.get_status()` via `fire_and_forget_async(timeout=5.0)` | `agent_service.py:553`, `agent_service.py:784-872` |
| Called immediately after `_subscribe_to_agent()` | ✅ Lines 547 (subscribe), 550 (metadata), 553 (status pull) | `agent_service.py:547-553` |
| Reconnect path goes through `connect_to_agent()` | ✅ `_reconcile_from_registry()` calls `connect_to_agent(name)` for known agents when `is_connected()` returns False | `agent_service.py:1900-1903` |
| One pull per reconnect, never more | ✅ The call is inside `connect_to_agent()` body, not in a timer/scheduler | `agent_service.py:458-553` |
| No retries on failure | ✅ `fire_and_forget_async` has `error_callback` that logs and returns | `agent_service.py:857-863` |
| Polling invariant holds | ✅ No `asyncio.sleep(N)` in `deep_worker_web/app/` | `grep` returns empty |
| Existing first-discovery unchanged | ✅ `connect_to_agent()` calls the same `_update_agent_status()` for both first-connect and reconnect | shared single path |

## File locations

- `deep_worker_web/app/services/agent_service.py`
  - `connect_to_agent()` at line 458 — calls subscribe + status pull
  - `_update_agent_status()` at line 784 — fires `root.get_status()` via `fire_and_forget_async(timeout=5.0)`
  - `_reconcile_from_registry()` at line 1827 — reconnect path for known agents

Already-Done-Task: a related task
