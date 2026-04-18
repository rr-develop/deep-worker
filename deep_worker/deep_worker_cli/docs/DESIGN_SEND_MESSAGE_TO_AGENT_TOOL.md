# Design: `send_message_to_agent` — Inter-Agent Messaging Tool

**Status:** ✅ Approved — ready for implementation
**Date:** 2026-04-18 (approved 2026-04-18)
**Related:**
- `deep_worker_cli/ARCHITECTURE.md` (agent as single source of truth)
- `deep_worker_cli/task_manager/task_manager.py:345` (reference: `TaskManager:>` prefix)
- `deep_worker_cli/tools.py` (where the tool will live)
- `deep_worker_cli/client/agent_client.py` (RPyC transport)
- `deep_worker_cli/process/registry.py` (agent discovery by name)

---

## 1. Goal

Give any running deep-worker agent the ability to **send a message to another
running deep-worker agent** identified by its registry name.

The message received by the destination agent must automatically carry the
name of the sender in its body, using the same pattern already used by
`TaskManager` when it injects corrective prompts:

```
TaskManager:> <original text>            ← reference, existing behavior
agent:<sender_name>:> <original text>    ← new, introduced by this tool
```

So that the recipient (and all dwcli / web clients subscribed to the recipient)
clearly see who talked to them.

---

## 2. Scope

### In scope

- A new LangChain tool `send_message_to_agent` exposed to every deep-worker
  agent through the standard tools list in `agent_process.py`.
- Discovery of the destination agent by **name only** (must be a running agent
  present in `ProcessRegistry`).
- Automatic prefixing of the outgoing text with `agent:<sender>:> ` — the
  sender must **never** be trusted from tool arguments; it is injected by
  the tool at runtime from the agent's own identity.
- Broadcast semantics identical to those used for user / TaskManager
  messages: the prefixed text arrives on the destination agent as a regular
  **user-role** message and is therefore replayed to all its subscribers
  (dwcli, deep-worker-web, …) — obeying the *"agent is the single source of
  truth"* principle.
- Structured, machine-readable success / error responses so the sending
  agent can continue reasoning about the outcome.

### Out of scope (explicit non-goals)

- Request / response (RPC-style) semantics — this is **fire-and-forget**:
  the tool delivers the message and returns, it does not wait for the
  target agent's assistant reply. A separate tool (e.g. `ask_agent`) can
  be added later on top of the same transport.
- Broadcasting to multiple agents at once.
- Sending messages to agents on remote hosts (the registry is localhost-only).
- Persistence, queuing, or retry on delivery failure.
- Any change to the wire protocol or to `agent_service.py`'s exposed API —
  we **reuse** `exposed_invoke_agent` as-is.
- Authentication / ACLs between agents — the registry is a shared localhost
  resource, same trust model as dwcli.

---

## 3. Motivation & Constraints

### 3.1 Why this shape?

- **Symmetry with TaskManager.** TaskManager already injects a well-known
  prefix (`TaskManager:> …`) and the whole broadcast plumbing is in place.
  Following the same pattern keeps the rendering/parsing logic on clients
  unchanged — they already know how to display a user-role message whose
  content starts with `<something>:> `.
- **Agent is the single source of truth.** The recipient's
  `exposed_invoke_agent` already notifies all its subscribers with the
  incoming user message *before* processing it (see
  `agent_service.py:315-323` and `agent_process.py:1244-1252`). Reusing
  that path means we get the "everyone sees it immediately" property for
  free.
- **No new RPyC methods.** All transport exists already. One LangChain tool,
  one RPyC call — no server-side changes required for the baseline.

### 3.2 Hard constraints

- The **sender name MUST NOT** be a tool argument — the LLM could forge it
  and impersonate another agent. It must come from agent identity known
  server-side (i.e. from `AgentProcess.name`, injected into the tool via
  closure / factory).
- The tool **must not block the event loop**. It runs inside the sender's
  asyncio loop and makes an async RPyC call.
- The tool must tolerate:
  - target agent not running (→ structured error, no exception leak);
  - target agent not reachable (RPyC port closed, stale registry entry);
  - sending to self (must be explicitly allowed or rejected — see §5.5).
- `auto_approve=False` mode must gate this tool behind an HITL approval
  prompt (new entry in `_add_interrupt_on()` in `agent.py`).

---

## 4. Public Contract

### 4.1 Tool signature

```python
async def send_message_to_agent(
    agent_name: str,
    message: str,
) -> dict[str, Any]:
    """Send a message to another running deep-worker agent.

    The name of the CURRENT (sending) agent is automatically prepended to
    the message in the form:

        agent:<sender_name>:> <message>

    The sender name is obtained from the agent's own process identity and
    cannot be overridden from the LLM side.

    The message is delivered as a user-role message to the target agent.
    The target agent will see it and may generate a response autonomously.
    This call is fire-and-forget: it does NOT wait for a reply.

    Args:
        agent_name: Registry name of the target agent. The target agent
            MUST be running (registered in ProcessRegistry with a live
            PID and an RPyC port).
        message: The text to deliver. Must be non-empty. May be any length
            that fits a single RPyC call.

    Returns:
        Dict with one of the following shapes:

        Success:
            {
                "success": True,
                "target_agent": "<agent_name>",
                "delivered_content": "agent:<sender>:> <message>",
            }

        Failure (structured — do NOT raise to the LLM):
            {
                "success": False,
                "error": "<short machine-readable code>",
                "message": "<human-readable explanation>",
            }

        Possible error codes:
            - "agent_not_found"       target is not in the registry
            - "agent_not_running"     registry entry exists but process is dead
            - "rpyc_port_missing"     registry entry has no rpyc_port
            - "connection_failed"     could not connect / call timed out
            - "invalid_argument"      empty message or empty agent name
            - "self_send_forbidden"   agent_name == sender_name (see §5.5)
    """
```

### 4.2 Exact delivered format

Given `sender = "alpha"` and `message = "please review my report"`, the
target agent receives (as if a user had typed it):

```
agent:alpha:> please review my report
```

i.e. literally `"agent:" + sender + ":> " + message`. No trimming of the
user-provided text, no escaping, no newline normalization — we keep full
fidelity with the `TaskManager:>` convention (see
`task_manager.py:345`, `f"TaskManager:> {message}"`).

Rationale for `agent:<name>:>` rather than just `<name>:>`:
- Reserves a dedicated namespace so clients and the recipient can reliably
  distinguish agent-originated messages from `TaskManager:>` / `user:>` /
  future prefixes.
- `TaskManager:>` is effectively the reserved `agent:TaskManager:>`; we
  generalize the scheme rather than overload the bare `name:>` form.

---

## 5. Design

### 5.1 Component placement

```
deep_worker_cli/
├── tools.py
│   └── make_send_message_to_agent_tool(sender_name)   ← NEW factory
├── process/
│   └── agent_process.py
│       └── _init_agent():
│           ...
│           web_tools = [
│               http_request, fetch_url,
│               web_search_delegation_tool, read_image,
│               make_send_message_to_agent_tool(self.name),   ← wire here
│           ]
├── agent.py
│   └── _add_interrupt_on():
│       returns {..., "send_message_to_agent": <config>}      ← HITL gate
└── client/
    └── agent_client.py     ← reused unchanged
```

The tool is produced by a **factory** (not a module-level `@tool`) because
its behavior depends on the sender agent's name, which is only known once
`AgentProcess.__init__` has run. The factory closes over `sender_name`
and returns a LangChain tool with that name baked in. This is the only
safe way to prevent the LLM from forging the sender field.

### 5.2 Execution path

```
 ┌───────────────────────────────┐
 │ Sender agent (e.g. "alpha")   │
 │                               │
 │  LLM chooses tool:            │
 │    send_message_to_agent(     │
 │      agent_name="beta",       │
 │      message="please ...")    │
 └──────────────┬────────────────┘
                │
                ▼
 ┌───────────────────────────────┐
 │ send_message_to_agent tool    │
 │   (runs in alpha's loop)      │
 │                               │
 │  1. Validate args             │
 │  2. prefixed = f"agent:alpha  │
 │       :> {message}"           │
 │  3. Lookup beta via           │
 │     ProcessRegistry           │
 │  4. AgentClient("beta")       │
 │     .connect()                │
 │  5. client.invoke_agent(      │
 │       prefixed, stream=True)  │
 │     (fire-and-forget variant) │
 │  6. client.disconnect()       │
 │  7. Return structured result  │
 └──────────────┬────────────────┘
                │ RPyC (localhost)
                ▼
 ┌───────────────────────────────┐
 │ Target agent "beta"           │
 │                               │
 │ exposed_invoke_agent(         │
 │   "agent:alpha:> please...",  │
 │   stream=True)                │
 │                               │
 │   → _notify_subscribers({     │
 │       "type": "user",         │
 │       "content": "agent:alpha │
 │         :> please ...",       │
 │       "role":"user"})         │
 │   → agent processes it &      │
 │     replies normally          │
 └──────────────┬────────────────┘
                │
                ▼
 All dwcli / web clients subscribed to beta
 render the prefixed message in real time.
```

Point (5) uses the existing `AgentClient.invoke_agent(...)` /
`_send_command({"command": "invoke", ...})` path, which internally calls
`exposed_invoke_agent` through the low-level `HANDLE_ASYNC_CALLATTR`
optimization. No changes there.

### 5.3 Connection strategy

Two options considered; **chosen: fresh short-lived `AgentClient` per call.**

| Option | Pros | Cons |
|--------|------|------|
| **A. Fresh `AgentClient` each call** (chosen) | Minimal surface area, no lifecycle to manage, target list is discovered dynamically, robust to target restarts. | One extra RPyC handshake per message (≈ms on localhost — acceptable). |
| B. Persistent pool keyed by target name | Saves handshake cost on repeat calls. | New failure modes (stale connections, target restart detection), extra shutdown code path, more code to test. Not justified for the expected call frequency (low). |

If, during implementation, profiling shows the handshake dominates cost,
option B can be added **behind the same tool surface** as an optimization;
the public contract does not change.

### 5.4 Stream mode choice

`exposed_invoke_agent(message, stream=True)` is used — matches the way
dwcli and the TaskManager feed messages, and keeps per-chunk broadcasting
behavior on the target. We **do not await** the full assistant response
(see `interactive.py` fire-and-forget note) — we only need the message to
be accepted and broadcast on the target side.

Practical shape of step (5) in the tool:

```python
# Fire the invoke; don't await completion of target's reasoning.
# We do await the *acceptance* by awaiting the RPyC request object
# for a short deadline so we can surface transport errors to the LLM.
import asyncio
try:
    await asyncio.wait_for(
        client.invoke_agent(prefixed, stream=True),
        timeout=ACCEPT_TIMEOUT_SECONDS,  # see §5.6
    )
except asyncio.TimeoutError:
    # Message likely accepted; but we cannot confirm within deadline.
    # Decision: treat as success=True with a warning, OR return
    # "connection_failed". Chosen: success=True + `warning` field,
    # because the target may be slow but the delivery already happened.
    ...
```

Alternative considered: use `exposed_run_agent_task(message)` (which
internally calls `invoke_agent(..., stream=True)`). Both land at the same
method; `invoke_agent` is chosen because it's already first-class in
`AgentClient` and returns a JSON response useful for confirming delivery.

### 5.5 Self-send policy

Possible policies:

1. **Reject** with `self_send_forbidden` (safer, prevents trivial loops).
2. **Allow** (the agent can queue a message to itself — sometimes useful
   for self-reflection loops).

**Chosen: reject by default** with a clear error, because:

- The agent can just produce the text in its own reasoning; sending to
  itself only re-enters `exposed_invoke_agent` which can easily create
  runaway loops, especially combined with TaskManager.
- An opt-in `allow_self=True` kwarg can be added later if a real use case
  appears. Public contract does **not** include it in v1.

### 5.6 Timeouts & back-pressure

- Registry lookup: synchronous, sub-ms (file lock + JSON read).
- RPyC connect: uses `AgentClient.connect()`'s 10 s timeout already.
- Invoke accept timeout: **`ACCEPT_TIMEOUT_SECONDS = 5.0`**. Short, because
  `exposed_invoke_agent` is async and should *begin* quickly — it only
  needs to take the message and broadcast it before doing real work.
  (Tunable constant, defined in `tools.py` near the tool.)
- If the target is currently busy in its own agent loop, RPyC's async
  serving means the call is still accepted quickly; only the *completion*
  is delayed, which we don't wait for.

### 5.7 HITL approval

`send_message_to_agent` is added to `_add_interrupt_on()` in
`agent.py:387`, with a formatter mirroring the existing ones:

```python
def _format_send_message_to_agent_description(
    tool_call, _state, _runtime,
) -> str:
    args = tool_call["args"]
    target = args.get("agent_name", "unknown")
    text = args.get("message", "")
    preview = text if len(text) <= 500 else text[:500] + "..."
    sep = get_glyphs().box_horizontal * 40
    return (
        f"Target Agent: {target}\n\n"
        f"Message:\n{sep}\n{preview}\n{sep}\n\n"
        f"{get_glyphs().warning}  The message will be prefixed with "
        f"'agent:<your-name>:> ' and delivered to the target as a user message."
    )
```

And:

```python
"send_message_to_agent": {
    "allowed_decisions": ["approve", "reject"],
    "description": _format_send_message_to_agent_description,
}
```

With `auto_approve=True` (skip permissions) the tool runs without a prompt,
same as every other gated tool today.

### 5.8 System-prompt support (REQUIRED)

A dedicated section is added to `default_agent_prompt.md` describing:

1. **How to use** `send_message_to_agent` (outgoing direction).
2. **How to recognize** messages arriving from other agents (incoming
   direction) — i.e. when a user-role message starts with
   `agent:<name>:> `, it was produced by another agent, not by the human
   user.
3. **Strict usage policy:** the tool must NOT be used by default. It is
   only appropriate when:
   - the human user has explicitly asked this agent to pass a message or
     command to another agent, OR
   - another agent has previously sent a message to this agent (visible
     as `agent:<name>:> …`) and this agent wants to reply.

Exact draft (to be inserted into `default_agent_prompt.md`):

```
### Inter-Agent Messaging

You can communicate with other running deep-worker agents using the
`send_message_to_agent(agent_name, message)` tool.

**Outgoing messages.** When you call `send_message_to_agent`, the system
automatically prepends your own agent name to the message, so the
recipient will see:

    agent:<your-name>:> <your message>

**Incoming messages.** When you see a user-role message that starts with
`agent:<name>:> ` (for example `agent:test_1:> 2+2`), it was NOT sent by
the human user — it was sent by another running agent whose name is
`<name>`. Treat it as a peer request and, if a reply is appropriate,
respond by calling `send_message_to_agent("<name>", "<your reply>")`.

**Usage policy (IMPORTANT).** Do NOT use this tool by default. Use it
ONLY when one of the following is true:

1. The human user has explicitly asked you to pass a message or a command
   to another agent (e.g. "send this to agent beta", "ask alpha to ...").
2. You received an agent-to-agent message earlier in this conversation
   (content started with `agent:<name>:> `) and you want to reply to
   that sender.

Do not initiate agent-to-agent conversations on your own, do not broadcast,
and do not send status updates to other agents unprompted.

The tool is fire-and-forget: it returns once the message has been
accepted by the target agent; you will NOT automatically receive a reply
in the same turn. Any reply, if the other agent produces one, will
arrive later as a new `agent:<their-name>:> …` user-role message.
```

This change is in-scope for v1 and ships together with the tool.

---

## 6. Data & State

No new persistent state. No new files. No new RPyC service methods.

Reads (per tool call):
- `/tmp/deep_worker_registry.json` via `ProcessRegistry.lookup()` — already
  used by `AgentClient`.

Writes: none.

---

## 7. Concurrency / Thread-Safety

- The tool runs on the sender agent's event loop (LangGraph executes async
  tools on the current loop; sync tools go through `asyncio.to_thread`).
  The function is **async** end-to-end, matching `AgentClient`.
- `ProcessRegistry` uses `fcntl` file locks — safe for concurrent readers.
- `AgentClient` creates its own connection; multiple tool calls in flight
  are independent (the "no persistent pool" decision, §5.3).
- On the target side, `exposed_invoke_agent` serializes concurrent
  invocations the same way it does today for dwcli + TaskManager; nothing
  changes there.

---

## 8. Error Handling Matrix

| Situation | Detection | Returned `error` | `message` | Raised? |
|---|---|---|---|---|
| `agent_name` empty | arg validation | `invalid_argument` | "agent_name must be a non-empty string" | No |
| `message` empty | arg validation | `invalid_argument` | "message must be a non-empty string" | No |
| `agent_name == sender` | `self_send_forbidden` check | `self_send_forbidden` | "Agents cannot send messages to themselves" | No |
| Target not in registry | `registry.lookup()` returns None | `agent_not_found` | "Agent '<name>' is not running" | No |
| Registry entry exists but PID dead | `lookup()` auto-cleans, returns None | `agent_not_found` | same as above | No |
| Registry entry has no `rpyc_port` | `AgentClient.connect()` raises `IPCConnectionError` | `rpyc_port_missing` | reused from client error | No |
| TCP connect fails / times out | `IPCConnectionError` from `connect()` | `connection_failed` | exception text | No |
| RPyC call raises | caught around `invoke_agent` | `connection_failed` | exception text | No |
| Accept timeout exceeded | `asyncio.wait_for` | `success=True` + `warning` | "Delivery likely but unconfirmed within 5s" | No |
| Unknown unexpected exception | catch-all | `connection_failed` | `repr(e)` | No |

Nothing is raised out of the tool: LangChain tools should return values so
the LLM can reason about failures (same pattern as `http_request` and
`fetch_url`).

---

## 9. Security Considerations

- **No sender forgery.** `sender_name` comes from `AgentProcess.name` via
  factory closure, never from tool arguments. The prefix is built on the
  sender side, not the receiver.
- **Localhost only.** `ProcessRegistry` only lists localhost processes and
  RPyC is bound to 127.0.0.1 (see agent process setup). No cross-host
  messaging possible.
- **HITL gating** (§5.7) ensures the user sees every outgoing message in
  interactive mode.
- **No code execution.** The tool only sends text; the target agent
  decides what to do with it. Same trust boundary as any user input.
- **Denial-of-service.** A chatty agent could spam another agent. Mitigations
  (none mandatory in v1):
  - HITL prompt per call in non-skip-permissions mode.
  - Future: per-sender rate limit counter in `AgentProcess`.

---

## 10. Testing Plan

### 10.1 Unit tests (`tests/unit_tests/test_send_message_tool.py`)

- `test_tool_rejects_empty_agent_name`
- `test_tool_rejects_empty_message`
- `test_tool_rejects_self_send`
- `test_tool_prefixes_message_with_sender_name`
  → mock `AgentClient`, assert the argument passed to `invoke_agent` is
  exactly `"agent:alpha:> hello"` when sender is `"alpha"` and message is
  `"hello"`.
- `test_tool_returns_agent_not_found_when_registry_empty`
- `test_tool_returns_connection_failed_on_client_exception`
- `test_tool_factory_bakes_sender_name`
  → `make_send_message_to_agent_tool("alpha")` is called twice with
  different sender names; the messages produced carry the correct sender.
- `test_sender_name_cannot_be_overridden`
  → attempt to pass `sender=...` kwarg; it must be rejected by signature
  (tool has no such parameter).

### 10.2 Integration tests (`tests/integration_ai/test_inter_agent_messaging.py`)

Start two real agents via the existing clean-agent fixture (see
`tests/integration_ai/clean_agent_fixture.py` / `test_taskmanager_tui.py`
for patterns), then:

- `test_message_delivered_and_prefixed`
  1. Start agents `alpha` and `beta`.
  2. Attach dwcli-like RPyC subscriber to `beta`.
  3. Invoke `alpha` with a prompt that forces it to call
     `send_message_to_agent("beta", "ping")` (use `auto_approve=True`).
  4. Assert the subscriber receives a user-role message whose content is
     exactly `"agent:alpha:> ping"` within N seconds.
- `test_message_triggers_target_agent_response`
  Same as above + assert `beta` produces an assistant response afterwards
  (i.e. `exposed_invoke_agent` ran on `beta`, not only `_notify_subscribers`).
- `test_target_not_running_returns_error`
  Send to a name that isn't running; assert the tool returns
  `agent_not_found` (tool result message content contains the code).

### 10.3 Manual smoke

```bash
deep-worker alpha -sp &
deep-worker beta  -sp &
# From an alpha-connected dwcli, type:
#   "Send the message 'status please' to agent beta."
# Expect in the beta-connected dwcli:
#   agent:alpha:> status please
#   (followed by beta's reply)
```

---

## 11. Implementation Checklist

Do not start coding yet — this is the proposed order once approved:

1. `deep_worker_cli/tools.py`
   - Add `make_send_message_to_agent_tool(sender_name: str) -> BaseTool`.
   - Add `ACCEPT_TIMEOUT_SECONDS = 5.0` constant.
2. `deep_worker_cli/agent.py`
   - Add `_format_send_message_to_agent_description`.
   - Extend `_add_interrupt_on()` with the new entry.
3. `deep_worker_cli/process/agent_process.py`
   - Import factory and append it to `web_tools` inside `_init_agent()`.
4. `deep_worker_cli/default_agent_prompt.md` (REQUIRED, v1)
   - Add the "Inter-Agent Messaging" section from §5.8, covering
     outgoing usage, recognition of incoming `agent:<name>:> …` messages,
     and the strict usage policy.
5. Tests per §10.
6. Update `ARCHITECTURE.md` to mention agent-to-agent messages as another
   source feeding `_notify_subscribers` (same bullet list that already
   mentions dwcli / task manager / other clients).

---

## 12. Resolved Decisions (previously Open Questions)

All questions below are now resolved and frozen for v1 implementation.

- **Q1. Prefix format.** ✅ **Resolved:** `agent:<name>:> <text>` with a
  single space after `:>`. Example: `agent:test_1:> 2+2`. (Symmetric with
  the existing `TaskManager:>` convention but in a dedicated `agent:`
  namespace.)
- **Q2. Self-send.** ✅ **Resolved:** Rejected by default with
  `self_send_forbidden` error (see §5.5). No opt-in flag in v1.
- **Q3. HITL approval.** ✅ **Resolved:** Standard approval gate, identical
  to other side-effect tools (`write_file`, `edit_file`, `execute`, …).
  When the agent is launched with `-sp` / `--skip-permissions`
  (`auto_approve=True`), the tool runs without prompting, exactly like
  every other gated tool.
- **Q4. Fire-and-forget vs await-reply.** ✅ **Resolved:** Fire-and-forget
  only. The tool returns as soon as the message is delivered (or the
  acceptance deadline is reached). No reply is awaited in the same turn.
  A future `ask_agent` tool with request/response semantics is out of
  scope for v1.
- **Q5. System-prompt addition.** ✅ **Resolved:** REQUIRED in v1. The
  section in §5.8 ships together with the tool and explicitly covers
  (a) outgoing usage, (b) how to recognize incoming `agent:<name>:> …`
  messages, and (c) the strict usage policy (tool is not used by default;
  only when the user asks or when replying to a prior inter-agent message).

---

## 13. Summary

- **One new tool**, `send_message_to_agent(agent_name, message)`.
- Sender name is **injected by the tool factory** from `AgentProcess.name`;
  the LLM cannot forge it.
- Outgoing text becomes `agent:<sender>:> <message>`, mirroring the
  existing `TaskManager:>` convention.
- Transport reuses the existing `AgentClient` + `exposed_invoke_agent`
  path; **no server-side changes** are needed.
- Target agent receives the prefixed text as a user-role message and
  broadcasts it to all its subscribers in real time, preserving the
  "agent is the single source of truth" invariant.
- Failures are returned as structured tool results (never raised).
- HITL-gated when `auto_approve=False`.

**Design file path:**
`deep_worker/deep_worker_cli/docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md`
