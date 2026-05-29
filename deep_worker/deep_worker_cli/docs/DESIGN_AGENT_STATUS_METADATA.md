# Design: Agent Status Metadata

Status: Draft

## Problem

`AgentStatus.IDLE` currently means only that the agent is not actively running. That is too coarse: the agent may be idle because it produced a final answer, or because it ended the turn by calling a tool that hands control to another actor, such as `send_message_to_agent`.

The web UI must keep showing plain `idle` for ordinary final answers, but show `idle(waiting_agent)` when the agent became idle after a successful `send_message_to_agent` call. Future tools should be able to attach their own idle reason without adding new top-level enum statuses.

## Constraints

- Do not change the base status enum values. `idle` remains `idle` on the wire and in persisted metadata.
- Do not encode `idle(waiting_agent)` as the actual status value; that would break enum conversion and lifecycle logic.
- No polling. Metadata is updated on the same event-driven status transitions already emitted by `SessionManager`.
- No new threads. Existing async persistence remains unchanged.
- Backward compatible: old `session_metadata.json` files without metadata load normally.

## Contract

Add optional status metadata alongside the existing base status:

```json
{
  "status": "idle",
  "status_metadata": {
    "substatus": "waiting_agent",
    "source_tool": "send_message_to_agent",
    "reason": "tool_call"
  }
}
```

Initial supported fields:

- `substatus`: short machine-readable status qualifier. For `send_message_to_agent`, this is `waiting_agent`.
- `source_tool`: tool name that caused the transition, if any.
- `reason`: generic transition category, e.g. `tool_call`.

Unknown fields are allowed and should be preserved in the dict so future tools can add data without schema churn.

Ordinary final answers set status to `idle` with no metadata (`status_metadata == null` / absent). The web UI displays this exactly as today: `idle`.

## CLI Changes

### `SessionManager`

Add `status_metadata: dict[str, Any] | None`.

- Persist it in `session_metadata.json` only when not `None`.
- Load it if present and shaped as a dict; otherwise treat it as `None`.
- Reset it to `None` on fresh launch/restart.
- Include it in `status_change` notifier payloads.
- Extend `set_status(status, *, metadata=None)` so callers can set metadata atomically with the status transition.
- `set_error()` and `clear_error()` clear metadata unless explicitly extended later.

This keeps metadata tied to the current status and prevents stale idle metadata from surviving a later running/error transition.

### `AgentProcess`

`_current_status_snapshot()` includes `status_metadata` when present.

Successful normal completion paths call:

```python
await session_manager.set_status(AgentStatus.IDLE)
```

Successful `send_message_to_agent` completion paths call:

```python
await session_manager.set_status(
    AgentStatus.IDLE,
    metadata={
        "substatus": "waiting_agent",
        "source_tool": "send_message_to_agent",
        "reason": "tool_call",
    },
)
```

Detection reuses the existing `AgentProcess._is_outgoing_inter_agent_echo()` helper. That is already the local contract for `SendMessageCompletionMiddleware`: a successful `send_message_to_agent` tool call ends the turn by injecting an outgoing echo of the form `<sender> -> <target> :> <message>`.

## RPyC Metadata

`exposed_get_metadata()` includes `status` and `status_metadata`. This lets reconnecting clients and status commands observe the same metadata even if they missed the live `status_change` event.

## Web Changes

`AgentInfo` gets optional `status_metadata: dict[str, Any] | None`.

`AgentService._dispatch_subscriber_payload()`:

- copies `status_metadata` from `status_change` payload into the in-memory `AgentInfo`;
- includes `status_metadata` in the frontend `agent_status` broadcast;
- clears cached metadata when payload omits it for a non-error/non-running settled status, so a later ordinary `idle` does not keep showing `waiting_agent`.

The frontend keeps `agent.status` as the base status. It computes a display label:

```js
idle(waiting_agent)
```

only when `agent.status === 'idle'` and `agent.status_metadata.substatus` is present. CSS class selection remains based on the base status, so `idle(waiting_agent)` uses the same ready styling as `idle` unless explicitly changed later.

## Tests

Unit tests:

1. `SessionManager` persists, loads, notifies, and clears `status_metadata` correctly.
2. `AgentProcess` status snapshots include metadata.
3. Non-streaming normal final answer sets plain idle with no metadata.
4. Non-streaming `send_message_to_agent` echo sets `idle` with `waiting_agent` metadata.
5. Web status-change handling forwards and caches `status_metadata`.
6. Frontend label helper renders `idle(waiting_agent)` while preserving plain `idle` when no substatus exists.
7. Web cache explicitly clears stale `status_metadata` when a later ordinary `idle` payload omits metadata.

Practical integration test without mocks:

- Use a real `SessionManager` with a temporary data dir and notifier callback.
- Perform real status transitions: `running` -> `idle(waiting_agent)` -> `running` -> `idle`.
- Assert the persisted `session_metadata.json` and emitted payloads match the contract and stale metadata is cleared.

## Non-Goals

- Do not add separate enum states such as `waiting_agent`.
- Do not change the semantics of `waiting_input`.
- Do not make the web backend infer substatus from message text; the agent process is the source of truth.
- Do not add periodic status refreshes or polling.
