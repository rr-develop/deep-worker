# Design: A2A protocol support for deep-worker agents

**Status:** Revised after review (RC-1…RC-7 incorporated) — pending re-review
**Date:** 2026-06-25
**Code location:** new package
`deep_worker/deepworker_a2a/` (at the `deep_worker` level,
NOT inside `deep_worker_cli` — A2A is not part of the CLI).
**Import name — `deepworker_a2a`, NOT `a2a`:** top-level `a2a` is taken by the
`a2a-sdk` library (`import a2a`), and the launcher puts `deep_worker` on PYTHONPATH —
an `a2a/` directory would shadow the SDK and break it (verified empirically). The name
`deepworker_a2a` mirrors the `grok_provider`/`codex_provider` pattern. The directory
stays at the `deep_worker` level (the user requirement is preserved). Agreed
with the user and the reviewer.

## User-approved decisions (locked)
- **Localhost only.** The A2A server listens exclusively on `127.0.0.1`. Nothing is
  exposed to the outside.
- But the **architecture must not hinder** a future switch to cross-host
  interaction: the bind address, the discovery scheme (URL in AgentCard) and security
  are in the config and are not hardcoded.
- **Server transport — Starlette** (extra `a2a-sdk[http-server]`). FastAPI
  works on top of Starlette; there is no reason to take it now. Switching to
  FastAPI/cross-host in the future does not require reworking the core.
- The remaining decisions (gRPC/REST, TaskStore, v0.3 compat) — made by the author below,
  aligned with the reviewer.

---

## 1. How A2A works (interaction model)

A2A is a **client-server protocol over HTTP**. There is no "bus", no P2P bypassing
HTTP. Any interaction:

```
  Agent A (A2A client)  ──HTTP (JSON-RPC)──►  Agent B (A2A server)
                        ◄──HTTP response/SSE──
```

- For agent B to be **discoverable and given a task**, it MUST
  raise an HTTP endpoint with A2A methods and publish an **AgentCard** at
  `/.well-known/agent-card.json`. This is a spec requirement (Agent Discovery), not
  our invention. No server → the agent is unreachable over A2A.
- For agent A to **initiate** a task, it acts as an A2A client: it resolves
  the target AgentCard by URL and sends `message/send` / `message/stream`.

For **full** protocol support each agent needs BOTH ends: a server
(inbound) and a client (outbound).

### End-to-end flow (agent A gives a task to agent B)
```
1. A (tool a2a_send_message): resolves B's AgentCard by URL,
   sends an HTTP POST JSON-RPC "message/send" { message: parts:[text] }.
2. B (A2A server, our code): DefaultRequestHandlerV2 creates Task(context_id,task_id)
   and calls our DeepWorkerAgentExecutor.execute(context, event_queue).
3. B (executor): text = context.get_user_input();
   runs an ORDINARY agent turn — the same _handle_invoke core as dwcli/web.
4. During the turn, events are translated into A2A via TaskUpdater
   (start_work → add_artifact(append) → complete | failed | canceled).
5. B → A: for message/send — the final Task(COMPLETED)+Message;
   for message/stream — a stream of SSE events (push, no polling).
6. A: receives the result as the return value of its tool.
```
**Server B does not reimplement the agent's logic** — it is an adapter: A2A input → our
turn → A2A output.

---

## 2. What we implement for agents

### 2.1 Inbound (receiving) — a process component, NOT a tool
The A2A server is not called by an LLM agent. It is raised automatically at the
agent process start, next to the existing RPyC server, on the same event loop.
Components (our code): `DeepWorkerAgentExecutor`, `AgentCard` assembly, ASGI startup.

### 2.2 Outbound (initiative) — LLM tools (required)
The LLM itself does not open HTTP sockets → tools-bridges are needed, 1:1 with the protocol methods:

| Tool | A2A method | Purpose |
|---|---|---|
| `a2a_discover(target)` | resolve AgentCard | business card/skills/capabilities of a remote agent |
| `a2a_send_message(target, text, context_id?)` | `message/send` | give a task, wait for the result |
| `a2a_send_message_streaming(target, text, context_id?)` | `message/stream` | the same with a stream of intermediate events |
| `a2a_get_task(target, task_id)` | `tasks/get` | state/result of a previously created task |
| `a2a_cancel_task(target, task_id)` | `tasks/cancel` | cancel a task |

- `target` — an external URL of an A2A agent OR the name of one of our agents (the URL is taken
  from the registry by `a2a_url`).
- `context_id` (opt.) — multi-turn dialog with the same remote agent.
- All tools return a structured dict and do not throw exceptions outward.
- Policy: do not initiate A2A calls without an explicit request (same as for inter-agent
  exchange).

---

## 3. Architecture inside the agent process

One process = one event loop. RPyC already runs on it. We add A2A next to it
(the same loop, WITHOUT new threads — `uvicorn.Server(Config(app)).serve()` as a task):

```
┌──────────────── agent B process (one asyncio loop) ──────────────────────────┐
│  RPyC AsyncioServer  (dwcli/web — internal management, as now)                 │
│                                                                                │
│  A2A ASGI (Starlette, uvicorn.Server.serve() as task, bind 127.0.0.1)         │
│    ├── GET /.well-known/agent-card.json  → AgentCard (discovery)                │
│    └── POST /  (JSON-RPC: message/send|stream, tasks/get|cancel|resubscribe)    │
│                  │                                                              │
│                  ▼                                                             │
│    DefaultRequestHandlerV2 ──execute(context,event_queue)──► DeepWorkerAgentExecutor │
│                                                                  │             │
│                                                                  ▼             │
│                                       _handle_invoke(...)  ← THE SAME TURN CORE │
│                                                  │  (LLM + tools, as dwcli)    │
│        turn events (notify_callback=bridge.on_event, in-process async)         │
│                                                  │                            │
│                                                  ▼                            │
│                     bridge → TaskUpdater → event_queue → Task/SSE response     │
│                                                                                │
│  Outbound: tools a2a_* ── A2A client (ClientFactory) ──HTTP──► AgentCard.url   │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Package structure
```
deep_worker/deepworker_a2a/   (import: `import deepworker_a2a`)
  __init__.py
  config.py     — A2AConfig: enabled, host(=127.0.0.1), port, transport, security (for future cross-host)
  card.py       — build_agent_card(agent_meta) -> AgentCard (+ AgentSkill)
  executor.py   — DeepWorkerAgentExecutor(AgentExecutor): execute()/cancel()
  bridge.py     — TurnEventBridge: turn subscriber → A2A events via TaskUpdater
  server.py     — start_a2a_server(agent_process, config): ASGI on the shared loop; writes a2a_port/a2a_url into the registry
  client.py     — A2AOutboundClient: wrappers over ClientFactory/create_client + AgentCard resolution
  tools.py      — LLM tools a2a_* (see §2.2)
  docs/DESIGN_A2A_PROTOCOL_SUPPORT.md  — this document
```

### Integration points with the existing code (minimal, without contract changes)
- **Server start (order matters, RC-3):** in
  `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py`
  RPyC is raised at line 1395 (`_start_rpyc_server`), and registry registration
  at line 1411 (`await self._register()`, "register ONLY after server ready").
  We start the A2A server **between** them: RPyC ready → `start_a2a_server(...)` (A2A
  ready) → `_register()` writes `rpyc_port` AND `a2a_port`/`a2a_url` in ONE
  atomic write. Thus the registry never has an agent without `a2a_url` when A2A
  is enabled. Gated by `A2AConfig.enabled`; when disabled — no-op.
- **Stream (RC-2, fixed):** `TurnEventBridge` is NOT registered in
  `AgentProcess._message_subscribers`. That mechanism calls `on_message(JSON-str)`
  on subscribers via RPyC `HANDLE_CALLATTR` (for web/dwcli netrefs) —
  the bridge does not fit there. The only way to receive turn events is to pass
  `notify_callback=bridge.on_event` into `_handle_invoke` (the same parameter that
  the RPyC path uses). `bridge.on_event` MUST be `async def` — the core does
  `await notify_callback({...})` (confirmed: agent_process.py, all calls of the form
  `await notify_callback(...)`).
- **Registry (RC-6):** extend `AgentProcessInfo` (types.py) and
  `ProcessRegistry.register()` (registry.py:71–78, currently writes only
  `rpyc_port`) with optional `a2a_port`/`a2a_url`. `lookup()` must return
  them (needed for `a2a_discover(target=<our agent name>)`). Backward-compatible:
  missing fields = A2A is not active for that agent.
- **Turn interruption (RC-1):** there is NO `interrupt_current_turn()` method.
  The real mechanism (agent_service.py:1072–1084): check
  `AgentProcess.is_running` + `current_worker_task.cancel()`. We add a thin
  wrapper without RPyC/netref:
  `AgentProcess.cancel_current_worker_task() -> bool` (True if there was an active
  turn and it was cancelled). The executor's `cancel()` calls it.
- **Mapping context_id → thread_id (RC-4):** `_handle_invoke_owned` currently
  hardcodes `config = {"configurable": {"thread_id": self.name}}`
  (agent_process.py:2737–2738). We add to `_handle_invoke`/`_handle_invoke_owned`
  an OPTIONAL parameter `thread_id_override: str | None = None` (backward-compatible —
  existing calls do not change, default None = current behavior). The executor
  passes `thread_id_override` (see §4.1). This is NOT a contract change: a new
  optional argument.
- **Config:** `A2AConfig` is read from `AgentConfig` (enable flag + port +
  host). By default `enabled=False` (RC-7).
- **Tools** `a2a_*` are registered as ordinary agent tools.

---

## 4. Inbound: DeepWorkerAgentExecutor (SDK contract confirmed against sources)

`AgentExecutor` (`a2a/server/agent_execution/agent_executor.py`):
`async execute(context: RequestContext, event_queue: EventQueue)` and
`async cancel(context, event_queue)`. Framework guarantees: one execute per
request; an unhandled exception → TASK_STATE_ERROR; after return you must not
touch context/event_queue; before finishing SHOULD publish a terminal/
suspended status; input-required = publish+return (the framework will repeat
execute); cancel = CancelledError in the execute task + an explicit cancel() call.

`RequestContext` provides: `message`, `task_id`, `context_id`, `current_task`,
`configuration`, `get_user_input(delimiter='\n')`.

`TaskUpdater` (`a2a/server/tasks/task_updater.py`) — the task driver:
`submit()`, `start_work()`, `add_artifact(parts, append, last_chunk, name, ...)`,
`complete(msg)`, `failed(msg)`, `cancel(msg)`, `requires_input(msg)`,
`requires_auth(msg)`, `update_status(state, message)`, `new_agent_message(parts)`.
Terminal states are protected by `asyncio.Lock`.

### execute() — algorithm (fixed per RC-1/RC-2/RC-4/RC-5)
```
async def execute(self, context, event_queue):
    text = context.get_user_input()
    updater = TaskUpdater(event_queue, context.task_id, context.context_id)
    await updater.submit()
    await updater.start_work()
    bridge = TurnEventBridge(updater)            # turn event translator → A2A
    # NOT registered in _message_subscribers (RC-2). The only path —
    # notify_callback. on_event — async (the core does await notify_callback(...)).
    result = await self.agent_process._handle_invoke(
        {"command": "invoke", "message": text, "stream": True},
        connection=None,
        notify_callback=bridge.on_event,          # the same parameter as in the RPyC path
        approval_callback=None,                   # HITL per the A2A model, see RC-5/§4.2
        thread_id_override=self._thread_id_for(context.context_id, context.task_id),  # RC-4
    )
    # terminal/suspension based on the turn result:
    if result.get("interrupted"):
        await updater.cancel()                    # CANCELED
    elif result.get("input_required"):            # see §4.2 (HITL)
        await updater.requires_input(updater.new_agent_message([text_part(result["prompt"])]))
        return                                     # the framework repeats execute with new input
    elif result.get("success"):
        await updater.complete(updater.new_agent_message([text_part(result["final_text"])]))
    else:
        await updater.failed(updater.new_agent_message([text_part(result.get("error",""))]))
```
> An unhandled exception in execute() is converted by the framework itself into TASK_STATE_ERROR.
> `add_artifact` chunks are emitted by the bridge DURING the turn (see below); the terminal
> status is set here from the final result.

### cancel() — algorithm (fixed per RC-1)
```
async def cancel(self, context, event_queue):
    # interrupt the current turn with the real mechanism (RC-1): is_running + worker_task.cancel()
    self.agent_process.cancel_current_worker_task()   # thin wrapper without RPyC/netref
    updater = TaskUpdater(event_queue, context.task_id, context.context_id)
    await updater.cancel()                            # CANCELED
```

### §4.1 Mapping context_id → thread_id (RC-4)
The LangGraph history is tied to `thread_id`. The core hardcodes `thread_id = self.name`
(agent_process.py:2738), so without mapping ALL A2A tasks with different
`context_id` would be mixed into one agent history and the dwcli history.
```
def _thread_id_for(self, context_id, task_id):
    cid = context_id or task_id          # the SDK almost always provides context_id
    return f"{self.agent_process.name}:a2a:{cid}"
```
- Stable multi-turn: one `context_id` → one thread_id → one A2A dialog history,
  isolated from the agent's main history (`self.name`).
- Passed through the optional `thread_id_override` into `_handle_invoke` (backward-compatible).
- An in-memory map is not required: `thread_id` is deterministic from `context_id`.

### §4.2 HITL / requires_input (RC-5) — NOT via the RPyC approval_callback
The RPyC `approval_callback` waits blocking on `exposed_request_approval` (Future +
broadcast) — wrong for A2A. The A2A model: when input is needed, the executor
publishes `updater.requires_input()` and **returns** from execute(); resume is
the NEXT inbound `message/send` with the same `context_id` (the framework calls
execute() again, new input comes via `context.get_user_input()`, the turn
continues on the SAME `thread_id`).
- Therefore in execute() above `approval_callback=None`, and the signal is
  `result["input_required"]` from the turn (stage 4 of the plan — mandatory for
  tool approval, not optional).
- The implementation of the "LangGraph interrupt → input_required → resume on the same
  thread" chain is detailed at stage 4 (may require `Command(resume=...)`
  on the repeated execute over the same thread_id; without auto-approve).

### TurnEventBridge — event mapping (fixed per RC-2)
The bridge receives events ONLY via `notify_callback=bridge.on_event`
(`async def on_event(self, msg: dict)`). The `msg` format — the same dicts that the core
sends to `notify_callback` (`{"type": ..., "role": ..., "content": ...}`). Translation:
- assistant text chunks / tool-call / tool-result → `await updater.add_artifact(
  parts=[text_part(...)], append=True, last_chunk=False)`; tool events
  are folded into a text part with the `[tool:<name>]` prefix;
- terminal statuses (`complete`/`failed`/`cancel`) are set by execute() from the
  final result — NOT by the bridge;
- an interruption event arrives as `{"type": "agent_interrupted", ...}` (and NOT
  `APPROVAL_REQUIRED`) — the bridge ignores it, since CANCELED will be set by cancel()/
  the `interrupted` branch in execute().
The canonical mechanism for streaming intermediate output in A2A is exactly artifacts
(`add_artifact(append=True)`), as confirmed by the reviewer.

---

## 5. Outbound: A2AOutboundClient (real API v1.0)

> WARNING: tutorials use the OLD API (`A2AClient`,
> `get_client_from_agent_card_url`). The real public API v1.0
> (`a2a/client/__init__.py`): `ClientFactory`, `create_client`, `ClientConfig`,
> `A2ACardResolver`, `minimal_agent_card`, `AuthInterceptor`/`CredentialService`
> (auth hooks — will be useful for future cross-host). Exact signatures
> are pinned during implementation against the installed package (stage 0 installs it).

`A2AOutboundClient`:
- `resolve_card(target) -> AgentCard` via `A2ACardResolver` (target = URL or
  our agent name → URL from the registry by `a2a_url`);
- `send(target, text, context_id=None) -> dict` via `create_client(...).send_message(...)`;
- `send_streaming(...)` via the streaming API (SSE, `async for`);
- `get_task` / `cancel_task`.
The `a2a_*` tools (§2.2) are thin wrappers over this client.

> **a2a_discover (per a reviewer note):** returns a STRUCTURED dict
> `{url, name, description, version, capabilities, skills:[{id,name,description}]}`,
> not a raw `AgentCard` — for consistency with the other tools (all return
> a dict and do not throw outward).

---

## 5.1 GATE RESULT (stage 0 DONE) — verified against the installed a2a-sdk 1.1.0

The agent environment: system Python 3.12, system site-packages (no venv;
PYTHONPATH adds the source directories). `pip install --break-system-packages
--user "a2a-sdk[http-server]"` → installed **a2a-sdk 1.1.0** + 3 leaf dependencies
(`aiologic`, `culsans`, `json-rpc`). **NO conflicts**: protobuf 6.33.2, pydantic
2.11.7, httpx, starlette 0.47.3, sse-starlette — already satisfied the pins, nothing
was upgraded. `import langgraph` / `import langchain_core` after the install —
OK (the core is not broken).

Verified REAL API 1.1.0 (important differences from reviews/`main`):
- **There is no `A2AStarletteApplication` class.** We assemble the ASGI app ourselves from
  route factories and mount it on `Starlette`:
  - `from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes`
  - `create_agent_card_routes(agent_card, card_modifier=None, card_url=AGENT_CARD_WELL_KNOWN_PATH) -> list[Route]`
  - `create_jsonrpc_routes(request_handler, rpc_url, context_builder=None, enable_v0_3_compat=False) -> list[Route]`
  - `app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))`
  - well-known path: `from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH`
    = `"/.well-known/agent-card.json"`.
- `DefaultRequestHandler = DefaultRequestHandlerV2(agent_executor, task_store,
  agent_card, ...)` — **`agent_card` is REQUIRED** (3rd positional).
  `from a2a.server.request_handlers import DefaultRequestHandler`.
- `from a2a.server.tasks import TaskUpdater, InMemoryTaskStore`.
  `TaskUpdater(event_queue, task_id, context_id)` — methods confirmed:
  `submit, start_work, add_artifact, complete, failed, cancel, requires_input,
  requires_auth, update_status, new_agent_message`.
- `from a2a.server.agent_execution import AgentExecutor, RequestContext`;
  `async execute(self, context, event_queue)`.
- Types: `from a2a.types import AgentCard, AgentSkill, Message, Part, Task,
  TaskState, Role` (a pydantic facade over the protobuf enum). `TaskState` is a
  protobuf enum; values: `TASK_STATE_{UNSPECIFIED, SUBMITTED, WORKING,
  COMPLETED, FAILED, CANCELED, INPUT_REQUIRED, REJECTED, AUTH_REQUIRED}`.
- Client: `from a2a.client import ClientFactory, create_client, ClientConfig,
  A2ACardResolver`.

→ Fixed in the design: instead of `A2AStarletteApplication` — assembling routes on
  `Starlette`; `DefaultRequestHandlerV2` receives `agent_card`.

## 6. Author decisions (aligned with the reviewer)

- **Request handler (RC-3):** use **`DefaultRequestHandlerV2`** (also the
  `DefaultRequestHandler` alias in `a2a.server.request_handlers`) + `InMemoryTaskStore`
  + the route factories `create_agent_card_routes`/`create_jsonrpc_routes` on `Starlette`
  (see §5.1 — verified against 1.1.0; `agent_card` is required in the handler constructor).
  NOT `LegacyRequestHandler`.
- **Protocol transport:** JSON-RPC (the SDK default). REST/gRPC — NOT now (we do not
  install extras). The architecture (Starlette ASGI) allows adding them later.
- **TaskStore:** start with `InMemoryTaskStore`. A durable store (SQL/adapter to our
  checkpointer) — a separate stage; for localhost-only single-process, InMemory
  is enough (a Task lives within the agent process).
- **Version:** we target clean **v1.0** (protobuf types `a2a.types.a2a_pb2`). The
  `compat v0.3` layer is not used until there are 0.3 clients.
- **Dependency (RC-7):** `pip install "a2a-sdk[http-server]"` into the agent
  environment; the version is pinned in the project dependencies (pre-commit). A2A is **disabled by
  default** (`A2AConfig.enabled=False`). With `enabled=True` and a failed `a2a`
  import — log + the agent keeps working WITHOUT A2A (not fatal for the process).
  Compatibility (sandbox): Python 3.12, pydantic 2.11.7, protobuf 6.33.2 —
  satisfy the SDK pins. The stage-0 gate re-checks this in the REAL environment
  of the agent process.

---

## 7. Event-driven and threads (project policies)

- A2A streaming = **SSE push** (the server proactively sends events). No
  status polling in a loop.
- The inbound bridge receives agent events by push via `notify_callback=bridge.on_event`
  (an in-process async callback, RC-2), it does not poll.
- `message/send` (without streaming) waits for the turn-completion future, not a polling loop.
- `tasks/get`/`resubscribe` — only ad-hoc on request.
- **No new threads:** the A2A server is an asyncio task on the existing loop.
- Permissions on created files (if a durable store/card cache appears) — 0o600/0o700,
  no 777/666.

---

## 8. Implementation plan (TDD; each stage → reviewer)

0. **Compatibility gate (RC-7):** install `a2a-sdk[http-server]` into the agent
   environment; check that the `a2a` import and protobuf/pydantic/langgraph do not
   conflict. Set `A2AConfig.enabled=False` by default and
   graceful degradation on a failed import (log + work without A2A).
1. **Inbound `message/send`:** `config.py`, `card.py`, `executor.py`
   (submit→working→completed), `server.py` (start on the loop BEFORE `_register`,
   `a2a_port`/`a2a_url` into the registry in one write). Core points: the wrapper
   `cancel_current_worker_task()` (RC-1), the `thread_id_override` parameter (RC-4),
   the registry fields (RC-6). Unit (card/executor with a mock `agent_process`) + integration without
   mocks (an external A2A client sends a message to a really running agent → response).
2. **Inbound streaming (`message/stream`/SSE):** `bridge.py` (`async on_event`,
   via `notify_callback`, RC-2). Integration: a stream of artifact chunks.
3. **Outbound `client.py` + tools `a2a_*`:** unit (mock client) + integration: two
   test agents talk over A2A (no mocks).
4. **Multi-turn (context_id, RC-4) + cancel (RC-1) + input-required/HITL (RC-5)
   — a mandatory stage** (not optional): the LangGraph interrupt →
   `requires_input` + return → resume on the same `thread_id` with the next inbound.
5. **Durable TaskStore / push-notifications / skills-mapping — by priority.**

Integration tests — WITHOUT mocks (we raise a real agent process and a real
HTTP call).

---

## 9. Review status
Review: **APPROVED with required changes**. RC-1…RC-7 are incorporated into the
document (cancel API, the bridge path via notify_callback, the
`DefaultRequestHandlerV2` choice, the `context_id→thread_id` mapping, HITL per the A2A model,
the registry fields, the gate+disabled-by-default). §6 (JSON-RPC-only, InMemoryTaskStore,
clean v1.0) and the `a2a_*` set — agreed. `a2a_discover` returns a dict.
After the reviewer's re-confirmation — move to stage 0 + TDD.
