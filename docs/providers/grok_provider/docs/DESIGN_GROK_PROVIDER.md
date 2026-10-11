# Design: Grok provider for deep-worker

> **⚠️ PARTLY SUPERSEDED — 2026-08-24: grok has been moved to the shared ACP layer.**
>
> This document describes grok's **own** ACP implementation inside `grok_provider`
> (its own `acp_client.py`, `proxy_mcp_server.py`, `client_state.py`, …). It was
> written BEFORE the shared `providers/acp/` layer existed and was a fork of it.
>
> `GrokChatModel` is now a thin subclass of `AcpChatModel` (~76 lines instead of
> ~1550), and everything grok-specific lives in `grok_provider/profile.py` as
> data (`AcpProfile`). The package shrank from 3535 to 478 lines.
>
> Sections §7-§11 (transport, client, proxy MCP, session state) should be read in
> [`acp/docs/DESIGN_ACP_PROVIDER_LAYER.md`](../../acp/docs/DESIGN_ACP_PROVIDER_LAYER.md);
> they are kept below as historical context.
>
> The fork was a **strict subset** of the shared layer (not a single method and
> not a single private attribute beyond `AcpChatModel`, while 29 methods were
> missing), so the move required no compromises. What grok gained with it:
> `SummarizerPort` (`asummarize`/`areset_after_summary`), the
> `session_new_validator` mechanism (a post-session gate from the 2026-08-15
> model-pin incident) on `session/new` **and** `session/load` — but **without a
> concrete validator**: grok takes its model from its own CLI configuration,
> there is no operator pin, there is nothing to check, and a validator would be
> dead code (only claude-code wires one up, because its pin travels via
> `ANTHROPIC_MODEL` and can be silently dropped by the adapter). Also gained:
> idempotent `initialize`/`authenticate`, a shared pool of CLI processes
> (previously an agent and its `:tm` each started their own `grok`), recovery
> after EOF (the `_acp_started` flag latched and wedged the model forever), and
> **fail-closed** permissions — the fork auto-approved ANY request, falling back
> to `offered[0]` (including a deny option) with no tie to `-sp`.
>
> Kept deliberately: the name `GrokBuiltin` and the key `_grok_original_title` —
> `bounded_tool_call` matches them literally to give image generation a 600s
> budget instead of 120s; they are also already recorded in checkpoints.

**Status:** 📐 DESIGN (rev 1, ready for review)
**Date:** 2026-06-17
**Companion:** [`RESEARCH_GROK_PROVIDER.md`](./RESEARCH_GROK_PROVIDER.md) — live findings
**Mirror of:** [`codex_provider/docs/DESIGN_CODEX_PROVIDER.md`](../../codex_provider/docs/DESIGN_CODEX_PROVIDER.md)

---

## 0. Hard constraints from the user (non-negotiable)

Taken directly from the codex design §0 — the same six invariants:

1. **We do not change deep-worker contracts.** `config["configurable"]["thread_id"]`
   = the deep-worker agent name; never mutated. [§3, §4]
2. **All messages — including every tool call and tool result — must
   pass through LangGraph** and end up in `result["messages"]`. [§5, §7]
3. **Maximally lightweight.** One long-lived `grok agent stdio`
   process per agent (NOT per-turn). In-process HTTP MCP server. [§7]
4. **History is read incrementally.** Tail-delta
   `messages[sent_count:]` per turn. No full rebuilds. [§6]
5. **No polling, no blocking operations on the event loop.** Fully
   event-driven through JSON-RPC notifications. [§7, §10]
6. **No leaks.** All long-lived state is bounded. [§11]

---

## 1. Motivation

The user wants to use **Grok** as an AI backend for deep-worker
agents, under the **supergrok** subscription (no separate API key).

```
$ deep-worker my-agent --provider grok
```

Authentication — via the cached login `~/.grok/auth.json`. The binary
`grok 0.2.54 (fee15ff8e) [stable]` is already installed on the host.

The Grok CLI implements **Agent Client Protocol (ACP) v1** through
`grok agent stdio` — a standardised JSON-RPC 2.0 protocol from Zed
Industries. Full specification:
https://agentclientprotocol.com/protocol/v1/overview

---

## 2. Background — current provider registry

Unchanged from `DESIGN_CODEX_PROVIDER.md` §2. The same 5 files need to be
extended with the new provider:

| # | File | What is added |
|---|---|---|
| 1 | `agent_config_pkg/agent_config.py:53` | `valid_providers = [..., "grok"]` |
| 1b | `agent_config_pkg/agent_config.py:60-65` | skip API-key load for grok |
| 2 | `cli/agent_command.py:653` | `choices=[..., "grok"]` |
| 3a | `config.py:1390` | `_PROVIDER_DEFAULT_MODELS["grok"] = ("GROK_MODEL", None)` |
| 3b | `config.py:1475` | skip API-key validation |
| 3c | `config.py:1739-` | factory branch `elif provider == "grok":` |
| 4 | `runtime/availability_tracker.py:11` | `"grok": null` |
| 5 | `runtime/provider_chain.py:4` | `["claude-code-sdk", "codex", "grok", "deepseek"]` |

`process/agent_process.py` and `task_manager/task_manager.py` — do NOT require
edits (grok has no non-resettable native tools analogous to codex's
`apply_patch`/`update_plan`). See §11A.2.

---

## 3. Terminology — three "thread/session" concepts

The same discipline as in the codex design:

| Name | Owned by | Lifetime | Source |
|---|---|---|---|
| **`agent_thread_id`** | deep-worker (LangGraph) | the whole agent process; equals `self.name` | Hard-coded in `agent_process.py:915, 1016, 1143` |
| **`grok_session_id`** | Grok agent process | Per-conversation; UUID assigned by Grok in the `session/new` response | ACP protocol; persisted by Grok itself in `~/.grok/sessions/<encoded-cwd>/<sid>/...` |
| **`mcp_call_id`** | our in-process proxy MCP server | one tool call (≤ 120 s) | Constructed by the handler: `f"{tool_name}_{counter}_{id(args)}"` |

**`agent_thread_id` IS NEVER MUTATED under any circumstances.** On
summarization — we discard `grok_session_id` (create a new
`session/new`), but the deep-worker `thread_id` remains.

`session/update` notifications from Grok carry their own `messageId` and
`toolCallId` — these are **Grok-internal identifiers that stay inside the
provider**. Only LangChain's `AIMessage.tool_calls[*].id` leave the provider
(we generate them ourselves during mapping, or
reuse Grok's `toolCallId` — discussed in §11A.2).

---

## 4. What the Grok CLI exposes via ACP (live-verified)

Full data — in `RESEARCH_GROK_PROVIDER.md` §§ 2–10. Here — what is
architecturally important.

### 4.1 Protocol handshake

```jsonrpc
→ initialize {protocolVersion:1, clientCapabilities, clientInfo}
← result    {protocolVersion:1, agentCapabilities, authMethods, _meta}

→ authenticate {methodId:"cached_token"}
← result      {_meta:{email, auth_mode, ...}}

→ session/new {cwd, mcpServers}
← result      {sessionId, models, _meta}

→ session/prompt {sessionId, prompt:[contentBlocks]}
   ↓ stream of session/update notifications
   ↓ (agent_thought_chunk, agent_message_chunk, tool_call, tool_call_update,
   ↓  plan, usage_update, available_commands_update, ...)
   ↓ possibly: session/request_permission (we reply)
← result    {stopReason: end_turn | max_tokens | refusal | cancelled | max_turn_requests, _meta}
```

### 4.2 Capabilities Grok 0.2.54 declares

```json
{
  "loadSession": true,
  "promptCapabilities": {"image": false, "audio": false, "embeddedContext": true},
  "mcpCapabilities": {"http": true, "sse": true},
  "_meta": {"x.ai/fs_notify": true}
}
```

- `loadSession:true` — `session/load <sid>` is supported (with replay).
- `sessionCapabilities` NOT declared → `session/resume` (without replay)
  NOT supported. Confirmed live: `Method not found`.
- `mcpCapabilities.http:true` → we can pass the HTTP URL of our in-process MCP.
- `embeddedContext:true` → we can send `ContentBlock::Resource`.

### 4.3 Auth — `~/.grok/auth.json`

`authenticate methodId:"cached_token"` uses the cached supergrok login.
Returns `{_meta:{email, auth_mode:"Oidc", ...}}`. **No API key needed.**

### 4.4 MCP servers are passed per-session (NOT via global config)

```json
{
  "type": "http",
  "name": "deepworker_<pid>",
  "url": "http://127.0.0.1:<port>/mcp",
  "headers": []
}
```

**We do NOT write `~/.grok/config.toml`** (policy, analogous to codex
§4.3.1 — a past config-pollution incident). Each `session/new` pulls
its OWN mcpServers; nothing global remains between agents.

### 4.5 System prompt — inline `<system>` block

`session/new` does NOT accept `systemPrompt`/`instructions`/
`_meta.system_prompt_override` (live-verified — all three are ignored).

**Working channel:** first `session/prompt` content text =
`<system>{ system prompt }</system>\n\n{ user query }`.

See §10 for implementation details.

### 4.6 Grok's built-in tools (see §11A)

Grok has 12+ built-in tools (`Shell`, `Read`, `Write`, `Edit`, `MultiEdit`, `Glob`, `Grep`, `WebSearch`, `WebFetch`, `Task`, `Plan`, `CallMcpTool`, ...), which **run locally inside the grok process** and are not delegated to the client.

**Established empirically (Q1 live-check 2026-06-17, see `RESOLUTION_Q1_DISALLOWED_TOOLS.md`):** in `agent stdio` mode they **cannot be disabled** by any CLI flag (`--disallowed-tools`, `--tools`, `--deny`, `--permission-mode`, `--sandbox` — all ignored), and grok **does not send** `session/request_permission` before executing them.

**Design approach (after agreement with the user 2026-06-17):** do not try to block — **fully record them in the LangGraph history** via the `tool_call`/`tool_call_update` notifications that ACP gives us with full args and result. Two equal tool-execution channels — deep-worker MCP (our single proxy MCP server) and grok built-ins (via `builtin_tool_passthrough.py`); both end up in `result["messages"]`. See §11A for details.

---

## 5. What the `claude-code-sdk`/`codex` providers do (slot we mirror)

Distilled lessons (unchanged from codex §5):

- **L1.** `agent_thread_id` is mandatory and unique per caller (main agent,
  TaskManager, subagent). No fallback.
- **L2.** We mirror only the tail delta `messages[sent_count:]`. Never
  the full history. `sent_count` cursor in a process-global singleton.
- **L3.** On summarization (detect `sent_count > len(messages)`) —
  discard the underlying session, **do not touch `agent_thread_id`**.
- **L4.** Tools execute via LangGraph ToolNode, **never** in the MCP
  handler. The handler parks an `asyncio.Future`; ToolNode executes the
  real tool; ToolMessage resolves the Future. Guarantees that every
  tool_call + tool_result ends up in `result["messages"]`.
- **L5.** Shared state — process-global singleton (LangGraph rebuilds
  model instances on pickling).

---

## 6. State model — choice B+: stateful-light with a long-lived ACP client

### 6.1 Options

- **A. Stateless.** Per-turn `grok agent stdio` subprocess; full
  history in `session/prompt`. ❌ Violates §0.4 (incremental cache).
  ❌ Wastes prompt cache. **Rejected.**
- **B. Stateful-light like codex.** Per-turn `grok exec` (if it existed)
  with `session/load <uuid>` on each turn. ❌ Grok has no
  `grok exec` — ACP mode is only `grok agent stdio` (interactive
  stream). And `session/load` does a full replay via notifications —
  per-turn that is expensive. **Rejected.**
- **B+. Long-lived ACP client.** **One `grok agent stdio` subprocess
  per agent, lives the whole life of the deep-worker process.** On each turn —
  one `session/prompt` request. This is the cleanest model for ACP.

  Per-agent state:
  - `grok_proc`: `asyncio.subprocess.Process` — long-lived process.
  - `grok_session_id: str | None` — UUID from `session/new`.
  - `sent_count: int` — how many LangGraph messages are already mirrored to Grok.
  - `pending_requests: dict[int, Future]` — JSON-RPC reply waiters.
  - `pending_tool_calls: dict[str, Future]` — for parking tool_calls.

  Per-turn flow:
  - **Turn 1** (`grok_session_id is None`): `→ session/new {cwd, mcpServers}` →
    captures `sessionId`. Then `→ session/prompt {sessionId, prompt}`.
  - **Turn N>1** (`grok_session_id` exists): simply
    `→ session/prompt {sessionId, prompt:messages[sent_count:]}`.
  - **Summarization** (`sent_count > len(messages)`): new `session/new`,
    discard the old `sessionId`. **Do NOT touch `agent_thread_id`.**

  `sent_count` is updated BEFORE waiting for the response (as with claude-code-sdk),
  so a retry does not duplicate.

  **Persistence across restarts.** `(grok_session_id, sent_count)`
  are written to disk in `<state_dir>/<agent>/grok_session_state.json`
  (mirroring `codex_session_state.json`).

  **Filesystem permissions (POLICIES.md §4 compliance, review 2026-06-17
  issue #5).** The file and its parent directory are created with
  explicit permissions:
  - `<state_dir>/<agent>/` — `0o700` (owner only).
  - `<state_dir>/<agent>/grok_session_state.json` — `0o600` (owner
    read/write only).

  Writing is atomic via a temp file + `os.replace`, with `os.chmod(tmp, 0o600)`
  before the replace (mirroring `codex_provider/session_state.py`'s perm contract).
  **No 777 / 666.** The explicit constraint is written into the spec; a regression
  unit test (`test_session_state.py::test_perms`) checks the stat-mode
  after `persist()`.

  On restart:
  1. Bring up a new `grok agent stdio` subprocess.
  2. If we have a `grok_session_id` — do `→ session/load <sid>`.
  3. Replay-notifications (arriving between request and response) —
     **discard** (LangGraph state already knows them via the checkpointer).
  4. After the response — continue from the same `sent_count`.

  If `session/load` errors (stale/missing) — fallback to a fresh
  `session/new` + full history (like the codex resume fallback).

**Accepted: B+.**

### 6.2 Why this satisfies all §0 constraints

| Constraint | How B+ satisfies it |
|---|---|
| We do not change deep-worker contracts | `agent_thread_id` read-only; checkpointer thread untouched. |
| All messages in the LangGraph history | Never bypass ToolNode; ToolMessages resolve Futures in proxy MCP (§7). tool_call/tool_call_update → `AIMessage.tool_calls`. |
| Lightweight, no extra processes | One long-lived subprocess (Grok itself is an external CLI binary). Proxy MCP — in-process asyncio task. |
| Incremental, no full rebuild | `sent_count` cursor; per-turn payload = `messages[sent_count:]`. |
| No polling, no blocking | JSON-RPC stdout: `await readline()` (asyncio non-blocking; epoll). MCP handler: `await asyncio.wait_for(future, ...)`. |
| No leaks | bounded state per agent_thread_id; one subprocess per agent_process; pending_tool_calls dropped on resolve. |

---

## 7. Module layout

Sibling of `codex_provider/`. The structure mirrors the codex one:

```
grok_provider/
├── __init__.py                   # re-exports GrokChatModel
├── chat_model.py                 # GrokChatModel(BaseChatModel)
├── client_state.py               # process-global singleton: agent_thread_id → _ThreadState
├── session_state.py              # disk-persistence of (grok_session_id, sent_count)
├── acp_client.py                 # GrokAcpClient — owns the long-lived stdio subprocess + JSON-RPC channel
├── acp_types.py                  # ACP event taxonomy (notifications + responses)
├── proxy_mcp_server.py           # in-agent HTTP MCP server (port + handler with Future-parking)
├── builtin_tool_passthrough.py   # records grok's built-in tool_call events into LangGraph history (mirrors codex native_tools.py — MANDATORY; main channel for built-ins, see §11A.4)
├── custom_types.py               # MCPResult, PendingCall, ToolMetadata, _ThreadState
├── pyproject.toml
├── py.typed
├── pyrightconfig.json
├── README.md
├── docs/
│   ├── RESEARCH_GROK_PROVIDER.md
│   └── DESIGN_GROK_PROVIDER.md    ← this file
└── tests/
    ├── test_acp_client.py
    ├── test_client_state.py
    ├── test_chat_model.py
    ├── test_proxy_mcp_server.py
    ├── test_session_state.py
    ├── test_builtin_passthrough.py
    └── test_integration.py        # CI-skipped, opt-in via GROK_INTEGRATION_TESTS=1
```

**Comparison with codex_provider:**

- ❌ No `exec_runner.py` (no per-turn subprocess).
- ❌ No `event_pump.py` (a single JSON-RPC reader does everything).
- ✅ Added `acp_client.py` (lifecycle of the long-lived subprocess + JSON-RPC).
- ✅ Added `acp_types.py` (ACP events instead of codex JSONL).
- ✅ `native_tools.py` → `builtin_tool_passthrough.py` — extended for the full set of grok built-ins (mandatory main channel, see §11A.4 and `RESOLUTION_Q1_DISALLOWED_TOOLS.md`).

Total line count — estimated 2400–2900 (codex provider ~3000). Slightly less because there is no:
- AGENTS.override.md write/cleanup machinery (~80 lines in exec_runner).
- per-feature `--disable` flag mapping (~150 lines in the config builder).
- event_pump (~250 lines; our reader_task does everything).

But an extended `builtin_tool_passthrough.py` is added (~250 LOC vs codex's ~200) and `acp_client.py` for the long-lived subprocess (~250 LOC).

---

## 8. `GrokChatModel` public surface

Mirror of `ClaudeCodeChatModel` / `CodexChatModel`:

```python
class GrokChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _client_state: Any = PrivateAttr(default=None)
    _acp_client: Any = PrivateAttr(default=None)      # GrokAcpClient (NOT per-thread; one per agent process)
    _proxy_mcp: Any = PrivateAttr(default=None)
    _tools_metadata: list[ToolMetadata] = PrivateAttr(default_factory=list)
    _working_dir: str | None = PrivateAttr(default=None)
    _model_name: str | None = PrivateAttr(default=None)
    _resolved_model_name: str | None = PrivateAttr(default=None)
    _resolved_model_callback: Callable | None = PrivateAttr(default=None)
    _disallowed_builtins: tuple[str, ...] = PrivateAttr(default=DEFAULT_DISALLOWED_BUILTINS)
    _proxy_started: bool = PrivateAttr(default=False)

    # Pydantic JSON-schema overrides — same surgery as codex (for checkpointer pickle).
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler): ...
    def model_dump(self, **kwargs): ...

    def __init__(self, model=None, max_tokens=None,
                 disallowed_builtins=None, **kwargs):
        kwargs["custom_get_token_ids"] = None
        super().__init__(**kwargs)
        self._client_state = get_global_client_state()
        self._tools_metadata = []
        self._model_name = model
        if disallowed_builtins is not None:
            self._disallowed_builtins = tuple(disallowed_builtins)

    @property
    def _llm_type(self) -> str:
        return "grok"

    def bind_tools(self, tools, **kwargs) -> "GrokChatModel":
        # 1. Extract MCP-compatible metadata from BaseTool list — same algorithm
        #    as claude_code_sdk_provider/chat_model.py:_extract_tool_schema
        #    (Pydantic schema first, fallback inspect.signature).
        # 2. Ensure proxy MCP HTTP server is initialized with this tool set.
        # 3. Server starts lazily on first ainvoke (NOT here — keep side-effect-free).
        ...
        return self

    async def ainvoke(self, input, config=None, **kwargs) -> AIMessage:
        # See §10.
        ...

    def _generate(self, *args, **kwargs):
        raise NotImplementedError("GrokChatModel supports async only. Use ainvoke().")
```

The interface is identical to `ClaudeCodeChatModel`/`CodexChatModel` so the factory
branch in `config.py` is a clean-copy with class renames.

---

## 9. ACP client architecture

### 9.1 Topology

```
┌──────────────────── deep-worker agent process ────────────────────┐
│                                                                    │
│  ┌─ LangGraph ─────┐     ┌─ GrokChatModel ─┐    ┌─ Proxy MCP ──┐  │
│  │ messages chan   │     │ ainvoke()       │    │ HTTP asyncio │  │
│  │ ToolNode        │◄───►│ client_state    │◄──►│ task on      │  │
│  │ checkpointer    │     └─────────────────┘    │ 127.0.0.1    │  │
│  │ broadcast subs  │           ▲                │ pending_calls│  │
│  └─────────────────┘           │                └──────────────┘  │
│                                ▼                       ▲          │
│              ┌─ GrokAcpClient ─────────┐               │          │
│              │ asyncio.subprocess.Proc │               │          │
│              │ reader_task (one ONE    │               │          │
│              │   reader of stdout)     │               │          │
│              │ pending_requests        │               │          │
│              │ pending_tool_calls      │               │          │
│              │ session_id              │               │          │
│              └─────────────────────────┘               │          │
│                       │                                │          │
└───────────────────────┼────────────────────────────────┼──────────┘
                       │ JSON-RPC                        │ HTTP MCP
                       │ stdio                           │ (in-process loopback)
                  ┌────▼────────────────────────────────▼────┐
                  │  grok agent stdio (long-lived child proc) │
                  └───────────────────────────────────────────┘
```

### 9.2 `GrokAcpClient` lifecycle

**One instance per agent process** (process-global singleton, as with the codex
proxy MCP). All `agent_thread_id`s (main + TaskManager + subagents)
SHARE one subprocess, but each has its own `grok_session_id` (via
separate `session/new`). Grok fully supports several active sessions
simultaneously — that is the ACP design.

**`async def start()`:**
- `proc = await asyncio.create_subprocess_exec("grok", "agent", "stdio",
   stdin=PIPE, stdout=PIPE, stderr=PIPE, limit=8*1024*1024)`
- `reader_task = asyncio.create_task(self._read_loop())`
- `await self._call("initialize", {"protocolVersion":1, "clientCapabilities":{
     "fs":{"readTextFile":False, "writeTextFile":False}, "terminal":False
  }, "clientInfo":{"name":"deep-worker","version":...}})`
- `await self._call("authenticate", {"methodId":"cached_token"})`

**`async def _read_loop()`** — the single reader of stdout:
- `line = await self.proc.stdout.readline()`
- `obj = json.loads(line)`
- If `id in obj` AND (`result in obj or error in obj`) → reply to our
  request: `self.pending_requests[obj["id"]].set_result(obj)`.
- If `id in obj` AND `method in obj` → request FROM grok to us (fs/*,
  terminal/*, session/request_permission). Routed through
  `_handle_agent_request()` (see §9.3).
- If only `method in obj` → notification. Routed through
  `_handle_notification()` (see §9.4).

**`async def _call(method, params)`** — JSON-RPC request with response wait:
- `next_id += 1; rid = next_id`
- `fut = asyncio.get_event_loop().create_future()`
- `self.pending_requests[rid] = fut`
- `self.proc.stdin.write((json.dumps(...) + "\n").encode())`
- `await self.proc.stdin.drain()`
- `try: result = await asyncio.wait_for(fut, timeout=...)`
- `finally: self.pending_requests.pop(rid, None)`

**`async def session_prompt(session_id, prompt_blocks, notification_sink)`**
— the main entry point for `ainvoke`:
- Registers the `notification_sink` (asyncio.Queue) to receive
  `session/update` notifications for this sessionId.
- Does `_call("session/prompt", {sessionId, prompt})`.
- When notifications with a matching `sessionId` arrive — pushes to the queue.
- When the response arrives — closes the queue, returns stopReason.

**`async def aclose()`:**
- Cancel `reader_task`.
- Cancel all pending_requests/pending_tool_calls futures.
- `proc.stdin.close(); await proc.wait()`.

### 9.3 Agent → Client requests (we handle them)

According to the ACP spec, the agent can send us JSON-RPC requests:

| Method | What we do |
|---|---|
| `session/request_permission` | See §9.5 |
| `fs/read_text_file` | `error -32601` (we declared `fs:false`) |
| `fs/write_text_file` | `error -32601` |
| `terminal/*` | `error -32601` (we declared `terminal:false`) |

If grok suddenly decides to call `fs/*` despite the `false` capability —
we return `-32601`; grok MUST gracefully fail its built-in tool and
move on. Live-test did not confirm this (`fs=false` → grok did
Read locally), but the protection is needed.

### 9.4 Notifications (`session/update`)

Routed to the notification_sink for the active sessionId.
We recognise the following `sessionUpdate` types:

| Type | Action in ainvoke |
|---|---|
| `user_message_chunk` | ignore (this is an echo of our prompt) |
| `agent_thought_chunk` | ignore by default; optionally save in `additional_kwargs["reasoning"]` |
| `agent_message_chunk` | accumulate into `final_text` |
| `tool_call` | **CASE A** (kind=mcp): we were waiting for this call via proxy MCP — ignore (resolution through handler/Future). **CASE B** (built-in tool — Shell/Read/etc.): see §11A.2 |
| `tool_call_update` | same; on `status="completed"` with content resolve the pending future for CASE B |
| `plan` | converted to AIMessage.tool_calls with name="update_plan" (similar to codex update_plan handling) |
| `usage_update` | broadcast via resolved_model_callback or metadata; does not go into the LangGraph history |
| `available_commands_update` | ignore (grok's slash commands) |

Vendor `_x.ai/*` notifications (mcp_initialized, git_head_changed,
queue/changed, settings/update, sessions/changed, models/update,
session/prompt_complete, announcements/update) — **fully ignored.**

### 9.5 Permission handling

**Established empirically (Q1 live-check 2026-06-17, see `RESOLUTION_Q1_DISALLOWED_TOOLS.md`):** grok 0.2.54 in `agent stdio` mode **never sends `session/request_permission`** for its built-in tools. It executes them locally without asking. This path is effectively dead in the current CLI version.

**Strategy — defense-in-depth in case of future grok CLI changes:** we implement a classifier, but do not rely on it for correctness. The user confirmed 2026-06-17 that HITL is not critical (agents work with auto-approve).

```python
def _classify_permission_request(req: dict) -> str:
    """Decide our response to session/request_permission.

    NOTE: not invoked on grok 0.2.54 — kept as forward-compatibility
    code in case future CLI versions start routing built-in tool calls
    through this channel. If/when that happens, this classifier ensures
    we don't accidentally auto-allow everything.

    Returns one of:
      - "allow"   : auto-grant (proxy MCP tools — they go through our
                    own LangGraph layer via the proxy MCP handler)
      - "deny"    : refuse outright (destructive built-ins where the
                    user has not opted into auto-approve)
    """
    tool_name = req["params"].get("toolCall", {}).get("title") or \
                req["params"].get("toolName") or ""
    kind = req["params"].get("toolCall", {}).get("kind") or ""

    # MCP tools — already routed through our proxy MCP handler and
    # LangGraph ToolNode. Safe to auto-allow at the ACP layer.
    if kind == "mcp" or tool_name.startswith("mcp__"):
        return "allow"

    # All built-ins — auto-allow under deep-worker's auto-approve default.
    # Recorded into LangGraph history via §11A.4 passthrough regardless.
    return "allow"
```

If HITL is ever needed for destructive built-ins, the `DESTRUCTIVE_BUILTINS` constant and the `deny` branch are already ready — just return `"deny"` for the required set. For now — auto-allow everything.

---

## 10. `ainvoke` turn algorithm

```python
async def ainvoke(self, input, config=None, **kwargs) -> AIMessage:
    # 1. Normalize input
    messages = _normalize(input)

    # 2. Mandatory agent_thread_id
    config = ensure_config(config)
    cfg = config.get("configurable") or {}
    if "thread_id" not in cfg:
        raise ValueError("GrokChatModel requires explicit thread_id ...")
    agent_thread_id = cfg["thread_id"]

    # 3. Per-thread state
    state = self._client_state.get_or_create(agent_thread_id)

    # 4. Summarization detection
    if state.sent_count > len(messages):
        # record_summarization also evicts the stale grok_session_id from
        # _session_index (L3) so the reader never routes a late update to a
        # dead session. agent_thread_id is NEVER mutated.
        self._client_state.record_summarization(agent_thread_id)
        state.grok_session_id = None
        state.sent_count = 0
        state = self._client_state.get_or_create(agent_thread_id)

    # 5. Extract SystemMessage as plain text
    system_prompt: str | None = None
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_prompt = _flatten_message_content_to_text(msg.content)
            break

    # 6. Ensure proxy MCP + acp_client started (lazy, idempotent)
    await self._ensure_proxy_started()
    await self._ensure_acp_client_started()

    # 7. Tail delta only
    new_messages = messages[state.sent_count:]
    user_text_parts: list[str] = []
    tool_resolved = False
    saw_builtin_resume = False
    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            is_builtin = bool(
                (getattr(msg, "additional_kwargs", None) or {}).get("grok_builtin", False)
            )
            if is_builtin:
                saw_builtin_resume = True
            else:
                self._proxy_mcp.resolve_call(
                    msg.tool_call_id,
                    {"content": _build_mcp_content(msg.content)},
                )
                tool_resolved = True
        elif isinstance(msg, HumanMessage):
            user_text_parts.append(
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
        # AIMessage / SystemMessage — already in grok session or delivered separately

    # 8. Persist sent_count BEFORE awaiting response (mirror codex/claude-code-sdk)
    state.sent_count = len(messages)
    self._client_state.persist(agent_thread_id)

    # 9. Mid-turn (resuming an in-flight grok turn) vs new turn.
    #    A mid-turn happens when LangGraph just gave us a ToolMessage —
    #    grok is still waiting on its proxy MCP tools/call to return.
    #    The Future resolution in step 7 will unblock grok inside the
    #    SAME session/prompt request we're already waiting on.
    if (tool_resolved or saw_builtin_resume) and state.pending_session_prompt_future is not None:
        # Same session/prompt request still pending. Just continue
        # waiting on its response + new notifications via the SAME
        # consumer. _consume_turn reads state.pending_session_prompt_future
        # and re-uses state.current_notif_queue WITHOUT creating a new
        # subscription — both are still valid here.
        #
        # NOTE: a built-in-only resume (saw_builtin_resume, no proxy-MCP
        # ToolMessage) MUST take this branch too. If we fell through to
        # step 10 we would fire a SECOND concurrent session/prompt while
        # the first is still in-flight — an ACP protocol violation
        # (one prompt turn per session at a time). Built-ins execute
        # locally inside grok, so there is nothing to resolve here; we
        # simply keep consuming notifications from the live turn.
        return await self._consume_turn(state, agent_thread_id)

    # 10. New turn → ensure session exists, build prompt blocks, fire session/prompt
    if state.grok_session_id is None:
        # First turn for this agent_thread_id (or post-summarization fresh start)
        proxy_url = await self._proxy_mcp.url_async()
        session_id = await self._acp_client.session_new(
            cwd=self._working_dir,
            mcp_servers=[{
                "type":"http", "name": self._proxy_mcp.server_name,
                "url": proxy_url, "headers":[],
            }],
        )
        state.grok_session_id = session_id
        # Register reverse index so reader_task can route session/update
        # notifications (which carry sessionId) back to this _ThreadState.
        self._client_state.register_session(session_id, agent_thread_id)
        self._client_state.persist(agent_thread_id)
        # System prompt is delivered inline in the first prompt of this session.
        is_first_prompt_in_session = True
    else:
        is_first_prompt_in_session = False

    # 11. Build prompt blocks — first prompt prepends <system> block
    blocks: list[dict] = []
    user_text = "\n\n".join(user_text_parts) if user_text_parts else ""
    if is_first_prompt_in_session and system_prompt:
        # Wrapping in <system>...</system> proved to make grok honor it
        # as a priority instruction (see RESEARCH §9.1, TEST 2).
        blocks.append({
            "type":"text",
            "text": (
                f"<system>\n{system_prompt}\n"
                f"{DEEP_WORKER_TOOL_CONTRACT}\n</system>\n\n"
                f"{user_text}"
            ),
        })
    else:
        blocks.append({"type":"text", "text": user_text})

    # 12. Fire session/prompt + consume notifications until response
    state.pending_session_prompt_future = self._acp_client.session_prompt(
        session_id=state.grok_session_id,
        prompt=blocks,
        notification_sink=state.current_notif_queue,
    )
    return await self._consume_turn(state, agent_thread_id)


async def _consume_turn(self, state, agent_thread_id) -> AIMessage:
    """Consume session/update notifications until either:
     - pending tool_call(s) need LangGraph routing → break, return AIMessage(tool_calls)
     - session/prompt response arrives → return AIMessage(content)

    NOTE: the early-exit (pending_tool_calls) path leaves
    state.pending_session_prompt_future POPULATED — that's the mid-turn
    invariant: the next ainvoke for this thread will see the populated
    future and re-enter _consume_turn to keep draining the SAME grok
    turn. The only paths that clear the future are:
      - response_fut resolved cleanly (the natural turn end)
      - any exception from this method, including asyncio.CancelledError
        (the try/finally below — required so a cancelled / crashed
        ainvoke does NOT leave a dangling future on _ThreadState).
    """
    final_text_parts: list[str] = []
    pending_tool_calls: list[dict] = []

    # Two concurrent things to await: notif_queue.get() and the
    # pending session/prompt response future. Use asyncio.wait().
    notif_q = state.current_notif_queue
    response_fut = state.pending_session_prompt_future

    cleared_normally = False  # set true on the two normal-exit paths
    try:
        while True:
            get_task = asyncio.create_task(notif_q.get())
            done, pending = await asyncio.wait(
                {get_task, response_fut},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if response_fut in done:
                get_task.cancel()
                response = response_fut.result()
                state.pending_session_prompt_future = None
                cleared_normally = True
                stop_reason = response.get("result", {}).get("stopReason")
                if stop_reason == "cancelled":
                    raise asyncio.CancelledError("grok turn cancelled")
                if stop_reason == "refusal":
                    raise GrokRefusalError("grok refused to continue")
                return AIMessage(content="".join(final_text_parts))

            notif = get_task.result()
            upd = notif["params"]["update"]
            kind = upd.get("sessionUpdate")

            if kind == "agent_message_chunk":
                txt = upd.get("content", {}).get("text", "")
                final_text_parts.append(txt)
            elif kind == "agent_thought_chunk":
                pass  # ignore reasoning chunks
            elif kind == "tool_call":
                # CASE A: this is a tool_call corresponding to a proxy MCP
                #   call grok just made — our handler is parking on it.
                #   We need to surface it to LangGraph as AIMessage.tool_calls.
                # CASE B: built-in tool (Shell/Read/Edit/...) — see §11A.2.
                tc = _map_tool_call_to_langchain(upd)
                if tc is not None:
                    pending_tool_calls.append(tc)
            elif kind == "tool_call_update":
                # If this is a built-in tool completion, resolve its
                # native-passthrough future and record the ToolMessage.
                self._resolve_builtin_future_if_any(state, upd)
            elif kind == "plan":
                # Surface plan as a synthetic AIMessage.tool_call (update_plan-style)
                tc = _plan_to_langchain_tool_call(upd)
                pending_tool_calls.append(tc)
            elif kind == "usage_update":
                await self._record_usage(upd)
            # other kinds — ignored

            if pending_tool_calls:
                # break out — give LangGraph the tool_calls, ToolNode will run them,
                # the resulting ToolMessages will come back into our next ainvoke
                # call where they resolve the proxy MCP futures and grok continues.
                # IMPORTANT: do NOT clear state.pending_session_prompt_future here —
                # this is the mid-turn pause; the future is still valid and the
                # NEXT _consume_turn re-enters this loop.
                cleared_normally = True
                return AIMessage(content="".join(final_text_parts),
                                 tool_calls=pending_tool_calls)
    finally:
        # If we exited abnormally (CancelledError, exception, KeyboardInterrupt,
        # GrokRefusalError, etc.) the pending_session_prompt_future is in an
        # indeterminate state. It MUST be cleared so the NEXT ainvoke for this
        # thread does not mistakenly think a turn is still mid-flight (which
        # would short-circuit step 9 into a dangling / cancelled future and
        # wedge the agent forever).
        # The two normal exit paths (response_fut resolved AND mid-turn
        # tool_calls pause) set cleared_normally=True. mid-turn pause keeps
        # the future on purpose; the response-resolved path nulled it
        # explicitly above. Both are already correct — we only act on the
        # abnormal exit.
        if not cleared_normally and state.pending_session_prompt_future is not None:
            fut = state.pending_session_prompt_future
            state.pending_session_prompt_future = None
            if not fut.done():
                fut.cancel()
            # Also send session/cancel to grok so it stops the turn cleanly,
            # otherwise the long-lived subprocess keeps streaming notifications
            # into a dead queue.
            with contextlib.suppress(Exception):
                await self._acp_client.session_cancel(state.grok_session_id)
```

Key principles:
- **Mid-turn detection** (step 9) — after a ToolMessage from LangGraph
  grok is still waiting for the reply to its tools/call; we do NOT create a new
  session/prompt, but continue consuming notifications from the current one.
- **One session/prompt = many rounds of tool_call ↔ ToolMessage**, until
  grok itself exits with `stopReason="end_turn"`.

### 10.1 Non-blocking invariants

| Invariant | How we satisfy it |
|---|---|
| No polling | `await readline()` (epoll). `await notif_q.get()`. `asyncio.wait(...)`. |
| No CPU steady-state | Between turns one subprocess in a wait-stdin state; HTTP MCP listener idle on epoll. |
| Event-driven end-to-end | One stdout reader; async JSON-RPC communication; Future resolution for tool calls. |
| Repeated client polls don't trigger heavy work | The provider is invoked by LangGraph only when there is new input. |

---

## 11. Tool execution — proxy MCP server

### 11.1 Why MCP

The Grok agent process is a separate OS process. It cannot call functions
in our address space directly. ACP dictates: tools = MCP servers,
passed via `session/new mcpServers`. HTTP transport is what
we need (in-process loopback).

### 11.2 In-agent-process HTTP MCP server

Identical to codex's `proxy_mcp_server.py`:
- `mcp.server.streamable_http` on `127.0.0.1:<random-free-port>`.
- `pending_calls: dict[call_id, {future, name, args}]`.
- `_jobs: dict[call_id, ...]` — job-handle for long-running tools.
- The handler parks an `asyncio.Future`, waits `await asyncio.wait_for(
    asyncio.shield(fut), timeout=wait_seconds)` (default 120s).
- On timeout — job-handle policy (see POLICIES.md §7.1), non-error
  `in_progress` notice with job_id.
- `resolve_call(call_id, content)` is called from `GrokChatModel.ainvoke`
  when a ToolMessage arrives.

### 11.3 Job-handle policy

Identical to codex — I do NOT repeat the details; I refer to
`codex_provider/proxy_mcp_server.py` as the reference implementation.

---

## 11A. Built-in tools and system prompt — detailed plan

(The section is numbered `11A` to avoid colliding with §7 (module layout)
— see the 2026-06-17 review. Logically it belongs to the tool
execution topic and follows right after §11.)

### 11A.1 Threat model — final (after Q1 live-check 2026-06-17)

**Established empirically on grok 0.2.54 (see `RESOLUTION_Q1_DISALLOWED_TOOLS.md`):**

In `grok agent stdio` mode **all CLI flags controlling tool availability are silently ignored** (`--disallowed-tools`, `--tools`, `--deny`, `--allow`, `--permission-mode`, `--sandbox`). Grok always gives the model the full set of built-ins and **executes them locally without `session/request_permission`**.

**CONCLUSION:** prevention of destructive built-ins at the protocol level is **impossible** in the current CLI version. This is by-design ACP — the client (us) sees the calls via `session/update`, but cannot block them.

**But this is OK for deep-worker:** agents almost always work with auto-approve (the user confirmed 2026-06-17), HITL is not critical. What IS critical — **full recording of all grok actions in the LangGraph history** (broadcast, checkpointer, TaskManager, dwcli).

Live-test confirmed that ACP gives us **everything needed** for full recording:

| Data | Where in the ACP payload |
|---|---|
| `toolCallId` (stable, unique) | every event |
| Tool name | `tool_call.title` (first event) |
| Args (command, path, etc.) | `tool_call.rawInput` |
| Streaming result chunks | `tool_call_update.content[].content.text` (status=in_progress) |
| Final result text | the final `tool_call_update status=completed` |
| Metadata (exit_code, cwd, duration_ms, log path) | `rawOutput` in the final update |

That is enough to reconstruct **exactly the same pair of `AIMessage(tool_calls=[...])` + `ToolMessage(content)`** that LangGraph produces via ToolNode for deep-worker MCP tools.

### 11A.2 Two-channel tool execution architecture

The final model — **two equal channels**, both end up in the LangGraph history:

**Channel A: deep-worker MCP tools (`execute`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `web_search_delegation_tool`, `fetch_url`, `send_message_to_agent`, `task`, project_memory, etc.)** — executed through our proxy MCP server (§11). Full cycle: grok calls via HTTP MCP → handler parks a Future → AIMessage(tool_calls) in LangGraph → ToolNode executes the real tool → ToolMessage resolves the Future → grok continues. **This is the main channel.** All basic operations (file I/O, shell, web, inter-agent communication) go through it.

**Channel B: grok built-in tools (`Shell`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `WebSearch`, `Task`, `Plan`, and any future ones)** — executed by grok locally. We **observe and record them post-factum** via `builtin_tool_passthrough.py`. **This is the observability/history channel.** Never blocks grok; the tool has already executed by the time we receive `status=completed`.

Channel B is needed for two reasons:
1. **History.** Otherwise grok's actions ("it edited foo.py") would not end up in `result["messages"]` → dwcli/web UI/TaskManager would not see them.
2. **Grok-specific tools without equivalents** (like `Plan` — the TodoWrite analog, or future grok-only ones). These tools have **no** alternative via channel A, and the user explicitly asked to support them.

### 11A.3 System prompt — guidance (NOT enforcement)

**Channel:** the first `session/prompt` content text. Format:
```
<system>
{deep_worker_system_prompt}

{DEEP_WORKER_TOOL_CONTRACT}
</system>

{user_query}
```

`DEEP_WORKER_TOOL_CONTRACT` — **guidance, not security**. The goal is to steer the model to use our MCP tools (channel A) when an equivalent exists, so that we:
- get our retry logic, unified timeouts, jobs-handle policy;
- use our project extensions (`project_memory`, `send_message_to_agent`, `task` middleware);
- have unified logging (deep-worker access logs vs grok local terminal logs).

Contract text:
```
Deep-worker tool contract:
- Prefer deepworker MCP tools (prefix `mcp__deepworker_*__`) over Grok's built-ins
  when an equivalent exists. They route through deep-worker's logging,
  retry, and HITL middleware:
  - Shell  → `execute`
  - Read   → `read_file`
  - Write  → `write_file`
  - Edit   → `edit_file`
  - Glob   → `glob`
  - Grep   → `grep`
  - WebSearch → `web_search_delegation_tool` / `fetch_url`
  - Task (subagent) → `task`
- Use Grok built-ins ONLY for grok-specific operations (e.g. `Plan`) that
  have no deep-worker equivalent.
- Both channels are recorded in conversation history; this is a preference,
  not a hard restriction.
```

If the model ignores it — this does **not break correctness**, because channel B still records the action in the history. The user simply sees `name="Shell"` instead of `name="execute"` in the Full Dialog. Semantically equivalent for most tasks.

### 11A.4 Built-in tool passthrough — critical module (`builtin_tool_passthrough.py`)

**PROMOTED IN STATUS:** from "optional safety net" (old revision) to **"mandatory main channel for built-ins"**.

Architecture (mirror of codex's `native_tools.py`, extended for the full set of grok built-ins):

```python
# Grok built-ins known at design time. The list is NOT fixed —
# any unknown name arriving in a tool_call is automatically surfaced
# via the _GenericBuiltinPassthrough fallback (see below).
KNOWN_GROK_BUILTINS: tuple[str, ...] = (
    "Shell",          # bash exec; rawInput={command, description}
    "Read",           # rawInput={file_path[, offset, limit]}
    "Write",          # rawInput={file_path, content}
    "Edit",           # rawInput={file_path, old_string, new_string[, replace_all]}
    "MultiEdit",      # rawInput={file_path, edits:[{old_string,new_string}]}
    "Glob",           # rawInput={glob_pattern, target_directory}
    "Grep",           # rawInput={pattern, [glob], [output_mode]}
    "WebSearch",      # rawInput={query}
    "WebFetch",       # rawInput={url}
    "Task",           # spawn subagent — rawInput={agent_type, prompt}
    "Plan",           # TodoWrite-like — rawInput={entries:[...]}
    "CallMcpTool",    # grok's MCP dispatcher (when our proxy MCP is invoked,
                      # grok wraps the call in CallMcpTool — we detect this and
                      # route through channel A instead, not passthrough)
)

class _BuiltinPassthroughTool(BaseTool):
    """LangChain tool that mirrors a grok built-in for history-recording.

    Lifecycle:
      1. bind_tools() registers one per name in KNOWN_GROK_BUILTINS
         (plus a wildcard via _GenericBuiltinPassthrough for unknown names).
      2. When grok emits session/update tool_call for this name, ainvoke
         surfaces it to LangGraph as AIMessage(tool_calls=[{id,name,args}]).
      3. LangGraph ToolNode calls this tool's _arun(**args, runtime).
      4. _arun parks an asyncio.Future on the per-thread state
         indexed by tool_call_id and AWAITS it (no timeout — bounded by
         grok turn lifetime which is itself bounded by job_handle policy).
      5. The reader_task in GrokAcpClient routes the matching
         tool_call_update status=completed → resolve_future(tool_call_id,
         content=final_text, metadata={exit_code, cwd, ...}).
      6. _arun returns the text → ToolMessage(content=text,
         tool_call_id=..., additional_kwargs={"grok_builtin": True,
         "grok_meta": {exit_code, cwd, duration_ms, log_path}}).
      7. Next ainvoke sees additional_kwargs["grok_builtin"] and SKIPS
         the proxy_mcp.resolve_call — grok ran the tool itself, the
         http MCP server was never invoked for this id.

    The args schema is OPEN (additional_properties: true) because we
    forward grok's rawInput verbatim. Validation happens on grok's side
    before the tool is even executed.
    """

class _GenericBuiltinPassthrough(_BuiltinPassthroughTool):
    """Wildcard fallback for tool_call names not in KNOWN_GROK_BUILTINS.

    Future-proofs against grok adding new built-ins between releases.
    Logged at WARNING level on first sighting per name + agent process,
    so we know to add explicit support in a follow-up.
    """
```

**Routing decision in `ainvoke` (extending §10):**

```python
# Inside _consume_turn, when we see session/update kind=tool_call:
upd = event["params"]["update"]
tcid = upd["toolCallId"]
title = upd.get("title")
raw_input = upd.get("rawInput") or {}

if title == "CallMcpTool" and raw_input.get("server") == self._proxy_mcp.server_name:
    # CASE A: this is grok's own dispatcher for our proxy MCP tool.
    #   Don't passthrough — proxy MCP handler is already parking on it.
    #   Surface the INNER tool name to LangGraph.
    inner_name = raw_input["toolName"]
    inner_args = raw_input.get("arguments") or {}
    pending_tool_calls.append({"id": tcid, "name": inner_name,
                               "args": inner_args, "type": "tool_call"})
elif title in KNOWN_GROK_BUILTINS or _looks_like_builtin(upd):
    # CASE B: grok built-in. Register a passthrough Future, surface as
    #   tool_call to LangGraph. Future resolves on tool_call_update
    #   status=completed in the reader_task (see _on_builtin_update).
    state.builtin_passthrough_futures[tcid] = asyncio.Future()
    pending_tool_calls.append({"id": tcid, "name": title,
                               "args": raw_input, "type": "tool_call"})
else:
    # Unknown — log warning and treat as built-in fallback.
    logger.warning("grok_provider: unknown tool_call %r — using generic passthrough", title)
    state.builtin_passthrough_futures[tcid] = asyncio.Future()
    pending_tool_calls.append({"id": tcid, "name": title or "UnknownBuiltin",
                               "args": raw_input, "type": "tool_call"})
```

**Reader → thread-state routing (sessionId → agent_thread_id):**

The single `reader_task` inside `GrokAcpClient` is shared across every
session (one session per `agent_thread_id`). Every ACP `session/update`
notification carries `params["sessionId"]`, so the reader resolves the
owning `_ThreadState` via a `sessionId → agent_thread_id` map maintained
by `ClientState` (populated in step 10 of §10 right after
`session_new` returns, removed on session teardown / summarization
reset). `builtin_passthrough_futures` therefore lives on that resolved
per-thread `_ThreadState`, NOT on the client — two agents sharing the
same grok CLI process never collide on a `toolCallId`. The `self.state`
shorthand in the snippet below denotes the `_ThreadState` resolved from
`params["sessionId"]`:

```python
def _on_session_update(self, params: dict) -> None:
    state = self._client_state.thread_for_session(params["sessionId"])
    if state is None:
        logger.warning("grok_provider: update for unknown session %r",
                       params.get("sessionId"))
        return
    upd = params["update"]
    k = upd.get("sessionUpdate")
    if k == "tool_call_update":
        tcid = upd["toolCallId"]
        status = upd.get("status")
        # streaming partial: just buffer.
        if status in (None, "in_progress"):
            self._buffer_partial_result(tcid, upd)
            return
        if status == "completed":
            final = self._render_completed_payload(upd)  # text + metadata
            fut = state.builtin_passthrough_futures.pop(tcid, None)
            if fut is not None and not fut.done():
                fut.set_result(final)
            return
        if status in ("failed", "cancelled"):
            fut = state.builtin_passthrough_futures.pop(tcid, None)
            if fut is not None and not fut.done():
                fut.set_result({"text": f"Tool {status}", "is_error": True,
                                "metadata": upd.get("rawOutput", {})})
            return
```

**`_render_completed_payload`** for `Shell` additionally pulls out `rawOutput.exit_code`, `current_dir`, `total_bytes`, `output_file` (path to the log file with the full output) — records them in `ToolMessage.additional_kwargs["grok_meta"]` for later inspection via Full Dialog / dwcli.

### 11A.5 Cleanup and leak analysis for channel B

`builtin_passthrough_futures` lives on the per-thread `_ThreadState`
(resolved by the reader via the `sessionId → agent_thread_id` map, see
§11A.4 routing note), never on the shared `GrokAcpClient` — so agents
sharing one grok CLI process never collide on a `toolCallId`.

`state.builtin_passthrough_futures: dict[str, asyncio.Future[dict]]`:
- **Entry is created** on `tool_call` (step CASE B above).
- **Entry is removed** on `tool_call_update status in {completed, failed, cancelled}` — Future resolved + popped.
- **Failsafe:** in the `_consume_turn` finally (change from review #3) and in `aclose()` any still-unresolved Future is cancelled. This handles the case where a grok session ends with a still-open built-in (theoretically possible on `session/cancel`).
- **Bounded:** max O(N concurrent tool_calls in one turn) ≈ a handful. In practice grok serializes built-ins.

### 11A.6 Test plan for channel B (TDD)

Units:
- `test_builtin_passthrough.py::test_shell_round_trip` — mock ACP events, verify that the Tool returns text and metadata.
- `test_builtin_passthrough.py::test_streaming_partial_buffered` — several in_progress events are stitched together.
- `test_builtin_passthrough.py::test_unknown_tool_via_generic` — unknown name → `_GenericBuiltinPassthrough`, WARNING log.
- `test_builtin_passthrough.py::test_call_mcp_tool_routes_to_channel_A` — CallMcpTool does NOT create a passthrough Future.
- `test_builtin_passthrough.py::test_failed_status_returns_error` — `is_error=True` in the result.
- `test_builtin_passthrough.py::test_cancel_clears_pending_futures` — aclose() cancels unresolved ones.

Integration (live grok):
- `test_integration.py::test_builtin_shell_recorded_in_history` — real grok, a prompt asking to run shell, verify that `state["messages"]` contains the pair AIMessage(tool_calls=[{name:"Shell"}]) + ToolMessage with the real stdout.



**Channel:** the first `session/prompt` content text. Format:
```
<system>
{deep_worker_system_prompt}

{DEEP_WORKER_TOOL_CONTRACT}
</system>

{user_query}
```

`DEEP_WORKER_TOOL_CONTRACT` — a copy of codex's same-named
constant from `codex_provider/chat_model.py:180`, adapted:
```
Deep-worker tool contract:
- Use only the deepworker_<pid> MCP server tools (prefix `mcp__deepworker_*__`)
  for file, shell, web, task, memory, and agent operations.
- DO NOT use Grok's built-in Shell — use deep-worker's `execute` tool.
- DO NOT use Grok's built-in Read/Write/Edit/Glob/Grep — use deep-worker's
  `read_file`/`write_file`/`edit_file`/`glob`/`grep`.
- DO NOT use Grok's built-in WebSearch — use deep-worker's
  `web_search_delegation_tool` / `fetch_url`.
- DO NOT use Grok's built-in Task (subagent spawn) — use deep-worker's `task` tool.
```

---

## 12. Lifecycle and leak analysis

### Long-lived objects (per agent process)

1. **`GrokAcpClient` singleton** — one per agent process. Owns:
   - `proc: asyncio.subprocess.Process` (one)
   - `reader_task: asyncio.Task` (one)
   - `pending_requests: dict` (ephemeral; entries dropped on response)
   - `pending_tool_calls: dict` (ephemeral)
2. **`GrokClientState` singleton** — process-global dict
   `agent_thread_id → _ThreadState`. Each `_ThreadState`: 3 scalars
   (grok_session_id, sent_count, pending_session_prompt_future?) +
   `current_notif_queue: asyncio.Queue` +
   `builtin_passthrough_futures: dict[str, asyncio.Future]`. Bounded by
   the number of `agent_thread_id`s in the process. Additionally holds the
   reverse index `_session_index: dict[str, str]`
   (`grok_session_id → agent_thread_id`), populated right after
   `session_new` (§10 step 10) and cleared on teardown / summarization
   reset (L3 — together with `grok_session_id`). The method
   `thread_for_session(session_id) -> _ThreadState | None` lets the
   reader_task route `session/update` to the right
   `_ThreadState`.
3. **`ProxyMCPServer` HTTP listener** — one per agent process.

### Short-lived objects (per turn)

- One `session/prompt` JSON-RPC request with a response future.
- Notification queue entries (drained as soon as consume_turn reads them).
- One `Future` in `pending_tool_calls` per tool call (drop on resolve).

### Cross-reference past incidents

| Incident | Class | Applicability to grok |
|---|---|---|
| dispatch-task leak | rpyc | N/A |
| traceback retention | exception on a long-lived obj | Risk: a `session/prompt` error persists on `state`. Mitigation: `state.pending_session_prompt_future = None` in finally. |
| AsyncResult leak | rpyc AsyncResult | N/A; asyncio.Future is local, dropped on completion. |
| **NEW: global config pollution analog** | global config pollution | Applicable if we wrote to `~/.grok/config.toml`. **We do NOT write** — mcpServers per session/new. |
| **NEW: subprocess pin** | long-lived subprocess | Applicable. Mitigation: `aclose()` cancels reader_task → proc.stdin.close() → proc.wait(); timeout → terminate → kill. |

---

## 13. Five lessons restated for Grok

| # | Codex/Claude-code rule | Grok restatement |
|---|---|---|
| L1 | thread_id mandatory, unique per caller | Same. Each agent_thread_id (main, :tm, :sub_*) gets own (grok_session_id, sent_count) slot. **agent_thread_id never mutated.** |
| L2 | Mirror only `messages[sent_count:]` | Same — grok server-side history via persistent session; we send only the delta in `session/prompt`. |
| L3 | Rotate session on summarization | Drop `grok_session_id`. The next ainvoke does a fresh `session/new`. **agent_thread_id NOT TOUCHED.** |
| L4 | Tools execute via ToolNode, never in MCP handler | Same. The proxy MCP handler parks a Future; ToolNode executes the real tool; ToolMessage in ainvoke → resolve_call. |
| L5 | Shared state is process-global singleton | Same. `GrokClientState` singleton; `GrokAcpClient` singleton; `ProxyMCPServer` singleton. |

---

## 14. Provider failover integration

Failover chains are defined by the operator in configuration and addressed by
name; hardcoding a chain in Python is forbidden (owner directive, 2026-09-03).
See `deep_worker_cli/docs/DESIGN_CONFIGURABLE_FAILOVER_CHAINS.md`.

The `default` chain:

```
claude-code → codex → grok → deepseek
```

(Order by decreasing quality/cost.) `grok` takes part in it as an ordinary step:
when its limit is exhausted the agent moves on to `deepseek`, and when the
preceding providers are exhausted it arrives at `grok`. `claude-code-sdk` is
absent from the chain deliberately: it is the second form of the same
`claude-code` provider with the same subscription, so switching between them
would land on an already exhausted account.

`deep_worker_cli/runtime/availability_tracker.py`:
```python
{
  "claude-code-sdk": {...},
  "codex": null,
  "grok": null,
  "deepseek": null,
}
```

### 14.1 Account-level exhaustion detection (RESOLVED)

The marker for usage-limit detection lives in `grok_provider/limit_errors.py`
(`_USAGE_LIMIT_MARKERS`). "Account-level exhaustion" covers TWO shapes,
both meaning "this account cannot serve the request, retry will not help —
failover now":

  * **Subscription / rate caps** — `usage limit`, `rate limit`,
    `quota exceeded`, `subscription limit`, `hit your … limit`.
  * **Billing / payment (HTTP 402)** — the xAI API answers `Payment Required`
    when the account has run out of credit. Markers: `payment required`, `\b402\b`,
    `insufficient credit/credits/balance/funds`, `out of credits`,
    `credit balance is too low`, `billing`.

Transient errors (5xx, timeout, connection reset) are **deliberately NOT matched** —
they are retryable and must not burn a failover step.

On a match, `GrokUsageLimitError` is raised; `agent_process._handle_invoke`
catches it by class name (`exc_name == "GrokUsageLimitError"`) → ERROR status
+ notify + `_schedule_failover("grok", …)` → deepseek.

### 14.2 Error-reply on the terminal session/prompt path (past incident)

There are TWO paths for getting a JSON-RPC reply, and previously only one checked the error:

  * `_call()` (handshake / session/new / session/load) — checked `"error"` and
    raised an exception.
  * `session/prompt` — resolves its future with the RAW reply (`_dispatch` →
    `fut.set_result(obj)`) and **bypasses `_call` entirely**. The terminal
    handling in `chat_model._finalize_response` took `response.get("result", {})`
    → `{}` on an error-reply → `stopReason=None` → returned an empty
    `AIMessage(content="")`. Result: HTTP 402 produced EMPTY messages in the history,
    with no error and no failover.

**Fix:** a single extractor `raise_for_error_reply(method, reply)` in
`acp_client.py` — single source of truth. Called from `_call` AND from
`_finalize_response` (before extracting `result`). Any error-reply now
raises `GrokUsageLimitError` (limit/billing) or `GrokAcpClientError`
(otherwise), instead of degrading into an empty message.

---

## 15. Test plan (TDD)

### 15.1 Unit tests

| File | Coverage |
|---|---|
| `test_acp_client.py` | initialize/authenticate/session/new/session/prompt handshake; reader_task notification routing; pending_requests resolution; reconnect after crash. |
| `test_client_state.py` | get_or_create; record_summarization; persist/restore disk state. |
| `test_chat_model.py` | bind_tools schema extraction; ainvoke flow (mocked acp_client + mocked proxy_mcp); mid-turn vs new-turn detection; summarization rotation; pydantic checkpoint pickle. |
| `test_proxy_mcp_server.py` | handler future-park; resolve_call; job-handle on timeout. |
| `test_session_state.py` | disk roundtrip; corruption tolerance. |
| `test_builtin_passthrough.py` | mapping tool_call → AIMessage.tool_calls; resolve via tool_call_update completed. |

### 15.2 Integration tests (opt-in)

`tests/test_integration.py` — skipped unless `GROK_INTEGRATION_TESTS=1`:
- Hello world turn.
- Multi-turn memory (resume via session_id).
- Tool call round-trip (via a real proxy MCP server with a real BaseTool).
- Thread isolation (main + TaskManager in one process with different
  agent_thread_ids).
- Restart resilience (kill grok subprocess, restart — session/load
  works).

### 15.3 deep-worker e2e

The history-integration e2e test module (test_deep_worker_history_integration.py) — launches a real
`deep-worker --provider grok` agent with a tempfile prompt, verifies
that `--history` shows tool_call + ToolMessage pairs. Mirror
of codex_provider.

---

## 16. Open issues (for review)

1. **Q-F resolution (CLOSED 2026-06-17).** Live verify showed: all CLI flags (`--disallowed-tools`, `--tools`, `--deny`, `--permission-mode`, `--sandbox`) are ignored in `agent stdio` mode, and `session/request_permission` for built-ins does not arrive. Decision in agreement with the user (2026-06-17): do not block, but **fully record** into the LangGraph history via `builtin_tool_passthrough.py` (§11A.4). Passthrough is promoted from "optional safety net" to "mandatory main channel for built-ins". See `RESOLUTION_Q1_DISALLOWED_TOOLS.md` for the full report.

2. **Resolved-model reporting.** Codex has `resolve_model_from_session_log`
   which reads the persisted `~/.codex/sessions/...jsonl` to learn
   which model grok actually chose. The analog for grok:
   `~/.grok/sessions/<encoded-cwd>/<sid>/...` contains `model_id`
   in `chat_history.jsonl` / `events.jsonl` and in `_meta.modelId` in the
   session/prompt response. **Use the response `_meta.modelId`** —
   it is directly in our `session/prompt` response, no need to touch the disk.

3. **Usage limit detection format.** Unknown for now. Need to wait for a
   live incident (or trigger one artificially). Mitigation: defensive
   exception handling in `_consume_turn`, classify by the content of the
   error message analogously to codex (see POLICIES §1.2).

4. **fs/* delegation alternative.** Currently we declare `fs:false`
   `terminal:false`. Alternative — `fs:true` + handlers that
   delegate to deep-worker's filesystem layer. Advantage:
   grok's `Read` falls under our HITL. Disadvantage: more code, and the
   protocol implies fs/* reads/writes real files (not
   virtual deep-worker dotfiles). **Defer** — we start with
   `false` and see how it behaves on a real task.

5. **`additionalDirectories` capability.** Grok did not declare it. If it
   declares it later — we can extend the session/new parameters.
   **Defer** — no use case right now.

---

## 17. Estimated work

- Design (this document): ✅ done.
- Implementation: ~2200–2700 lines (see §7).
- Unit tests: ~80–100 tests.
- Integration tests: ~5 live tests + 1 deep-worker e2e.
- Touch points in `deep_worker_cli`: 5 files (see §2), point edits.
- Failover wiring: 2 files (§14).

**Time:** 1–2 days of dedicated focus.

---

## 18. Review

The design goes through review before implementation.

After review sign-off — we move to TDD implementation.
