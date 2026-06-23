# Design: Codex provider for deep-worker

**Status:** ✅ IMPLEMENTED — first cut landed on branch `dev` on
2026-05-15. 88/88 unit tests pass; 4/4 live integration tests pass
(hello-world, multi-turn memory, tool-call round-trip, thread
isolation between main agent and TaskManager). See `tests/`.

**Date:** 2026-05-15 (rev 3, implemented)

## Implementation notes (added after the first PR landed)

Three things had to be adjusted versus the original design once we
exercised it live against `codex-cli 0.130.0`:

1. **`--dangerously-bypass-approvals-and-sandbox` is required** for MCP
   tool calls to go through unprompted. ``approval_policy="never"``
   alone only suppresses prompts for *shell commands* — Codex
   auto-cancels every MCP `tools/call` with "user cancelled MCP tool
   call" unless the bypass flag is set. This is the Codex CLI analog
   of claude-code's ``permission_mode="bypassPermissions"``. Safe in
   our setup because deep-worker owns sandboxing/approvals at the
   LangGraph layer (`interrupt_on`).
2. **The break point in `ainvoke` is `item.started`, not
   `item.completed`.** Codex emits `item.completed` for an
   `mcp_tool_call` only AFTER our HTTP handler has returned a result.
   We want to break out BEFORE handing the result back, so LangGraph's
   ToolNode runs first. The earliest signal Codex gives us is
   `item.started` with `status="in_progress"`. There's a small race
   window (stdout `item.started` and HTTP `tools/call` arrive on
   independent channels), so `ainvoke` parks on
   `ProxyMCPServer.wait_for_pending_calls(...)`. The HTTP handler
   resolves that waiter at the exact moment it registers the pending
   call. No sleep loop, no polling.
3. **The proxy MCP server is a process-global singleton.** Different
   `CodexChatModel` instances (LangGraph may rebuild the model during
   pickling; agent + TaskManager + subagents may share a process)
   share one in-process HTTP MCP server. `bind_tools()` hot-swaps its
   tool set; the server keeps the same port and the same registration
   in `~/.codex/config.toml`. To avoid Codex caching an empty
   `tools/list`, we defer registration until tools are actually
   bound — never register an empty server.
4. **Native Codex tools that overlap deep-worker tools are stripped
   when possible, recorded into LangGraph otherwise.** (Mechanism
   rewritten twice on 2026-05-18; see Implementation notes 6 and 7
   for the working channels.) The original `--config
   disabled_tools=[...]` override turned out to be silently ignored
   by Codex 0.130; we migrated to per-feature `--disable <feature>`
   flags + a top-level `web_search="disabled"` override
   (Implementation note 6). The seven native tools that survive
   every flag (`apply_patch`, `update_plan`, `view_image`,
   `request_user_input`, `list_mcp_resources*`) are now recorded
   into LangGraph as honest tool_call + ToolMessage pairs via the
   event pump + native_tools machinery (Implementation note 7). The
   legacy `_raise_if_blocked_native_item` defense-in-depth check
   that used to abort the turn on every native event was removed —
   it was harmful UX (turn became `status="error"` instead of
   answering the user).
5. **System prompt is delivered via `AGENTS.override.md`, not via
   `--config base_instructions=...`** (changed 2026-05-18). Codex CLI
   0.130.0 silently ignores the `base_instructions` config key —
   verified by comparing `session_meta.payload.base_instructions.text`
   in `~/.codex/sessions/.../*.jsonl` against the value we tried to
   set: Codex always keeps its built-in default GPT-5 prompt. Three
   delivery paths were tried and rejected: `-c base_instructions="..."`,
   `[profiles.<name>] base_instructions = "..."`, and
   `[profiles.<name>] instructions = "..."` — none of them overrode
   the built-in. The path that *does* work is the `AGENTS.override.md`
   file (recognised by `core/src/agents_md.rs` in the Codex binary);
   Codex prepends its contents as the first `user` message of every
   turn. `CodexExecRunner.spawn` now drops a marked override file into
   the working directory before each `codex exec` and
   `ExecHandle.aclose` removes it. The file is never overwritten if it
   already exists without our `<!-- deepworker:codex_override v1 -->`
   marker — a user-authored override is sacred.

   *Sub-fix:* the `SystemMessage.content` extraction in `ainvoke` was
   using `str(msg.content)`, which produced a Python `repr` of the
   `content_blocks` list (`"[{'type': 'text', 'text': '...'}]"`)
   whenever middleware wrapped the prompt — and deep-worker's
   middleware stack (memory, skills, project_memory, local_context,
   subagents) always wraps it. The provider now flattens content
   blocks to plain text via `_flatten_message_content_to_text`,
   skipping non-text blocks. See the test class
   `TestAinvokeToolCalls::test_system_prompt_*` for the contract.

6. **Native-tool deny-list channel: per-feature `--disable` flags, NOT
   `disabled_tools=[...]`.** (Changed 2026-05-18.) Codex CLI 0.130
   silently ignores the older `--config disabled_tools=[...]` channel —
   we verified live via `RUST_LOG=codex_api=trace` that the
   `response.create` websocket payload's `tools=[...]` list is
   identical with or without it. The working contract is one
   `--disable <feature>` per Codex feature flag whose tools we want
   stripped from the API request, plus top-level config overrides for
   the handful of keys that Codex moved out of the feature namespace
   in 0.130 (currently just `web_search`, which now lives at the
   top level as `"live" | "cached" | "disabled"`).

   Feature → native tool mapping (verified against `codex features
   list` on cli 0.130.0):

   | Feature flag         | Native tools it adds to the API request    |
   |----------------------|--------------------------------------------|
   | `shell_tool`         | `exec_command` (single-call variant)       |
   | `unified_exec`       | `exec_command` (with `write_stdin`)        |
   | `multi_agent`        | `spawn_agent`, `send_input`, `resume_agent`, `wait_agent`, `close_agent` |
   | `image_generation`   | `image_generation`                         |
   | `tool_search`        | `tool_search`                              |
   | `apps`               | `mcp__codex_apps__*` (e.g. GitHub)         |
   | `plugins`            | Plugin-contributed tools                   |

   Native tools that 0.130 has NO way to remove via config:
   `apply_patch` (always-injected `custom` tool), `update_plan`,
   `view_image`, `request_user_input`, `list_mcp_resources`,
   `list_mcp_resource_templates`, `read_mcp_resource`. These survive
   for every Codex 0.130 turn; their handling is described in
   Implementation note 7.

   See `codex_provider/chat_model.py:DEFAULT_DISABLED_FEATURES` and
   `DEFAULT_CONFIG_OVERRIDES` for the canonical defaults, and
   `codex_provider/tests/test_disallowed_tools.py` for the regression
   contract.

7. **Surviving native tools are recorded into LangGraph as honest
   tool_call + ToolMessage pairs.** (Added 2026-05-18, replaces the
   `_raise_if_blocked_native_item` defense-in-depth that aborted the
   turn.) When Codex emits `item.started` for a native tool
   (`command_execution`, `apply_patch`, `file_change`, `web_search`,
   `tool_search`, `custom_tool_call`, `function_call`,
   `function_call_output`), the codex_provider:

   - returns an `AIMessage(tool_calls=[{id, name, args, type}])`
     from `ainvoke` where `name` is exactly Codex's `item.type`
     (no synthetic prefix) and `id` is Codex's `item.id` verbatim,
   - LangGraph's ToolNode routes into one of the async BaseTool
     bodies registered by `codex_provider.native_tools`. The bound
     tool set is extended with `NATIVE_TOOLS` whenever a
     CodexChatModel-backed agent is created, i.e. in both
     the CLI's `process/agent_process.py` (main agent) and
     the CLI's `task_manager/task_manager.py`
     (`-tm` manager agent — symmetric injection so a native tool call
     during a manager turn does not crash it).

   - the tool body parks on an `asyncio.Future` that the background
     event-pump task created at the same `item.started` event,
   - when Codex emits `item.completed` the pump resolves the
     Future with the rendered result string and the tool body
     returns it. ToolNode wraps it in a `ToolMessage(name=…,
     tool_call_id=…, additional_kwargs={"codex_native": True})`.

   The outbound `ToolMessage` for a native call is *not* pushed
   back into Codex — Codex never blocked on it, it has already
   moved on inside the same `codex exec`. The discriminator is
   `additional_kwargs["codex_native"]`; the existing mid-turn
   reuse branch in `ainvoke` keeps reading the same handle's
   `ainvoke_queue` for the next event.

   The single reader of the JSONL stream is the background
   `EventPumpTask` (`codex_provider.event_pump`) created at
   `CodexExecRunner.spawn` time and torn down by
   `ExecHandle.aclose`. Two destinations only:

   - `ExecHandle.ainvoke_queue` (`asyncio.Queue(maxsize=64)`) — what
     `ainvoke` reads (thread.started, turn.started/completed/failed,
     item.started for both proxy MCP and native, item.completed for
     `AgentMessageItem`).
   - `ExecHandle.native_futures` (`dict[item_id → Future]`) —
     awaited by the matching tool body.

   All routing is pure event-driven: `async for`, `Queue.put`,
   `Future.set_result`. No `asyncio.sleep`, no polling, no
   while-True. See `codex_provider/docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md`
   for the full design (including 16 sections of rationale, concurrency
   invariants, and TDD plan), and
   `codex_provider/tests/test_event_pump.py` +
   `tests/test_native_tools.py` for the contract.

## Final decisions on open questions (user, 2026-05-15)

| # | Decision |
|---|---|
| Q1 | **D2** — own subprocess + JSONL parser; no runtime dep on Agents SDK. |
| Q2 | **Do not pin a model version.** Default = "let Codex CLI choose its own default" (do NOT pass `--model` to `codex exec` unless user explicitly set `CODEX_MODEL` env or `--model` flag). This way we automatically pick up Codex's evolving default. Concretely: `_PROVIDER_DEFAULT_MODELS["codex"] = ("CODEX_MODEL", None)` and `chat_model.py` omits `--model` when `model_name is None`. |
| Q3 | **`-sp` / `skip_permissions` is the default operating mode for deep-worker agents** and MUST work for Codex too. Mirror claude-code's pattern: claude-code unconditionally sets `permission_mode="bypassPermissions"` (`client_manager.py:259-275`) — i.e. the provider always tells the upstream CLI "do not prompt for approvals, ever; deep-worker's HITL via `interrupt_on` decides what to allow at the LangGraph layer". For Codex this translates to: provider **always** passes `--config approval_policy="never"` and `--sandbox workspace-write`. Approval / HITL stays at the LangGraph level (via `interrupt_on` middleware), exactly where it lives today. |
| Q4 | **HTTP** transport for the proxy MCP server (loopback, in-agent-process). |
| Q5 | Codex built-in deny-list MUST strip every native tool that overlaps a default deep-worker tool, so the model never picks a native bypass over the proxy MCP. Implementation (post-2026-05-18, see Implementation note 6): pass one `--disable <feature>` per relevant Codex feature flag (`shell_tool`, `unified_exec`, `multi_agent`, `image_generation`, `tool_search`, `apps`, `plugins`) plus a top-level `-c 'web_search="disabled"'` override. The older `--config disabled_tools=[...]` channel was empirically verified to be silently ignored by Codex 0.130 and is no longer used. In addition, reject native tool-like Codex item types at runtime if the CLI still emits them — `apply_patch` in particular cannot be stripped via config on 0.130. |

**Scope (in):** architecture, terminology discipline, state model,
lifecycle, leak surface, concurrency invariants, integration
touchpoints with the CLI.

**Scope (out, explicitly):** writing the provider package, tests,
packaging changes, modifications to `claude_code_provider` or to any
deep-worker contract.

---

## 0. Hard constraints from the user (non-negotiable)

This design must satisfy ALL of the following. Each is addressed
specifically in the sections below; the cross-reference is in
brackets.

1. **Do not change deep-worker contracts or protocols.** In
   particular `config["configurable"]["thread_id"]` is, in this
   project, **the agent's name** (`self.name`) — that mapping is
   fixed and we must not touch it. [§3, §4]
2. **All messages — including every tool call and every tool
   result — MUST go through LangGraph and end up in deep-worker's
   conversation history** (the `messages` channel in the LangGraph
   state, the single source from which dwcli, the web UI,
   checkpointer, and TaskManager all read). [§5, §7]
3. **Maximum lightweight, no external processes** unless strictly
   unavoidable. In particular, the proxy MCP server is
   **in-agent-process** (an asyncio task running an HTTP listener
   bound to `127.0.0.1`), not a separate OS process. [§7]
4. **Conversation history is read incrementally and cached.**
   The provider must NOT rebuild the message list from scratch on
   every turn; the same "send only the tail `messages[sent_count:]`"
   rule as `claude-code` applies. No periodic full reads, no
   "is anything new" polling. [§6]
5. **No polling, no blocking operations on the event loop, no
   high-CPU steady-state.** Architecture is event-driven end to
   end. [§7, §10]
6. **No memory leaks.** All long-lived state is bounded and
   accounted for. [§11]

---

## 1. Motivation

The user wants to use OpenAI's **Codex CLI** as an AI backend for
deep-worker agents, in the same slot as the existing `claude-code`
provider:

```
$ deep-worker my-agent --provider codex
```

Codex CLI is already installed on this machine (`codex-cli 0.130.0`)
and authenticates through a cached login at `~/.codex/auth.json`.
**No API key prompt is acceptable.**

This is **predesign**: the question to answer is "is this feasible at
all, and what's the right shape". The implementation is a follow-up.

---

## 2. Background — how providers plug in today

The CLI's `config.py:1404-1670`
holds the single provider registry — `create_model(provider,
model_name)`. The `claude-code` branch lives at `config.py:1647-1666`
and sets the precedent for a "local sibling package" provider.

The string `"claude-code"` appears in exactly five places — every one
needs a sibling entry for a new provider:

1. `agent_config_pkg/agent_config.py:52` — `valid_providers` list;
   and `:60-62` — skip API-key load.
2. `cli/agent_command.py:276` — `choices=[...]` for `--provider`.
3. `config.py:1344-1357` — `_detect_provider`.
4. `config.py:1379` — `_PROVIDER_DEFAULT_MODELS`.
5. `config.py:1453` — skip API-key validation; and `:1647-1666` —
   the factory branch.

Also: `process/agent_process.py:447-448` does
`setattr(model, '_working_dir', ...)`. The Codex provider must expose
the same `_working_dir` private attribute name — we follow the
contract, we do not change it.

---

## 3. Terminology — three distinct "thread/session" concepts

The previous revision blurred two of these. Explicit distinction; the
rest of the document uses ONLY these names.

| Name | Owned by | Lifetime | Source |
|---|---|---|---|
| **`agent_thread_id`** | deep-worker (LangGraph) | Whole agent process; equals the agent name | Hard-coded `self.name` at `agent_process.py:915, 1016, 1143` |
| **`codex_session_uuid`** | Codex CLI | Per Codex conversation; UUID assigned by Codex in its first `thread.started` event; can be `resume`d | Codex CLI internals; persisted by Codex itself in `~/.codex/sessions/<uuid>.jsonl` |
| **`mcp_call_id`** | the in-agent proxy MCP server (ours) | One tool call (≤ 120 s) | Constructed by our handler: `f"{tool_name}_{counter}_{id(args)}"` |

**The `agent_thread_id` NEVER changes.** It is the deep-worker
contract and we don't touch it. On summarization, **only the
`codex_session_uuid` is discarded**; the LangGraph `thread_id`
(= agent name) stays exactly the same, the LangGraph history stays
in the same checkpointer thread, dwcli and the web UI continue
subscribing to the same agent.

`claude_code_provider` already follows this discipline: it maps
`thread_id` → `effective_session_id`, and the `effective_session_id`
mutates (`<thread_id>__epoch_N`) without ever changing `thread_id`
itself. We mirror that, with the rename above to keep names
unambiguous.

Note that Codex's wire protocol uses the word "thread_id" for its
session UUID (see `{"type":"thread.started","thread_id":"019e2c0e-..."}`).
That is **a Codex name for a Codex concept** and stays inside the
provider's `exec_runner`. Nothing called `thread_id` ever crosses the
provider boundary toward LangGraph — outside the provider, only
`agent_thread_id` exists.

---

## 4. What Codex CLI actually exposes (verified live on this host)

`codex-cli 0.130.0` was probed live; the JSONL output below was
produced on this machine.

### 4.1 Non-interactive event stream

```bash
$ echo "say HELLO and nothing else" | \
    codex exec --experimental-json --skip-git-repo-check \
               -s read-only --cd /tmp -
{"type":"thread.started","thread_id":"019e2c0e-c1b2-74e0-..."}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"HELLO"}}
{"type":"turn.completed","usage":{"input_tokens":23808,"cached_input_tokens":21888,"output_tokens":6,"reasoning_output_tokens":0}}
```

Event types relevant to us:

| Event | Carries | Meaning to us |
|---|---|---|
| `thread.started` | `thread_id` (UUID) | Codex assigned a fresh session — we capture & cache as `codex_session_uuid` |
| `turn.started` | – | informational |
| `item.completed` (`item.type=agent_message`) | `.text` | the assistant's text reply |
| `item.completed` (`item.type=mcp_tool_call`) | `server`, `tool`, `arguments`, `status`, `result?`, `error?` | a tool call routed to our proxy MCP server |
| `item.completed` (other types) | – | reasoning is ignored; native tool-like items such as `command_execution`, `file_change`, `web_search`, `tool_search`, `custom_tool_call`, `function_call`, and `function_call_output` are rejected because they bypass LangGraph tool history |
| `turn.completed` | `usage` | end-of-turn; subprocess exits next |
| `turn.failed` | `error.message` | raise |

### 4.2 Resume across turns

```bash
$ codex exec resume <codex_session_uuid> --experimental-json -
```

Codex persists each session to `~/.codex/sessions/<uuid>.jsonl`. We
don't own that storage.

### 4.3 External MCP servers (THE integration lever)

```
$ codex mcp add <name> --url http://127.0.0.1:<port>   # streamable HTTP MCP
$ codex mcp add <name> -- <command...>                 # stdio MCP
$ codex mcp list
```

`codex mcp add` persists to `~/.codex/config.toml`. Every `codex exec`
re-reads this file and connects afresh to every declared MCP server.
This is the channel through which Codex calls our tools.

**⚠️ We do NOT use `codex mcp add` — see §4.3.1.**

### 4.3.1 MCP server isolation — per-agent, NEVER global

**`~/.codex/config.toml` is GLOBAL to every codex process on the host.**
A deep-worker host runs many codex agents over time, each with its own
proxy MCP server (`deepworker_<pid>`). If each agent registered itself
into that shared file via `codex mcp add`, the `[mcp_servers.*]` table
would grow without bound — `codex mcp add` writes, nothing ever cleans
the dead entries. Every subsequent `codex exec` (from any agent) then
loads the WHOLE list and tries to connect to every declared server at
startup. Dead ports (agent long gone) hang for the full
`ProxyMCPServer` timeout — **120 s per stale server** — before failover.

**Production incident (observed live):** the config had
accumulated 21 dead `deepworker_<pid>` entries. A turn that needed a
tool spent ~10 minutes timing out against dead servers one-by-one
before stumbling onto a live one and "suddenly working". Simple
prompts (`2+2`) that need no tool answered instantly — masking the
problem. The user had never sanctioned any global shared list; agents
are required to be fully independent.

**The contract (DO NOT REGRESS):**

1. The provider **never** runs `codex mcp add` / writes the global
   config. (`_register_proxy_with_codex` was deleted 2026-05-20.)
2. Every `codex exec` passes **`--ignore-user-config`** so it does NOT
   load `~/.codex/config.toml` at all — neutralising any junk other
   agents (or older builds) may have left. Auth still resolves via
   `CODEX_HOME` and `resume <uuid>` still works (sessions live in
   `CODEX_HOME/sessions`, not in config.toml).
3. The agent injects **only its own** proxy server, per-invocation:
   `--config mcp_servers.<name>.url="http://127.0.0.1:<port>/mcp"`,
   sourced from `ExecArgs.mcp_server_name` / `mcp_server_port`
   (set in `CodexChatModel.ainvoke`, rendered in
   `CodexExecRunner._build_argv`).

Result: a codex agent sees exactly one MCP server — its own — and
touches no global state. Regression guards:
`tests/test_exec_runner.py::TestMcpServerIsolation`.

### 4.4 Auth — `~/.codex/auth.json`

`CodexOptions.api_key` is optional. When `None`, the CLI is not
passed any `CODEX_API_KEY`, and it falls back to its cached login.
Verified: `~/.codex/auth.json` is present on this host; the live
test above used no env key.

### 4.5 The Python wrapper in the Agents SDK (reference, not necessarily a runtime dep)

The OpenAI Agents SDK's
`src/agents/extensions/experimental/codex/exec.py:61-228`
spawns `codex exec --experimental-json [resume <uuid>] -` via
`asyncio.create_subprocess_exec` and yields each stdout line.
Per-turn subprocess; the `Codex` class holds no long-lived process.
The wire protocol underneath is 4 event types and one prompt stream.

---

## 5. What `claude-code` provider actually does (the slot we mirror)

Distilled lessons:

**L1. `agent_thread_id` is mandatory and unique per caller.**
No fallback. Subagents and TaskManager use suffixed
agent_thread_ids (`<agent>:tm`, `<agent>:sub_<uuid8>`).

**L2. Mirror only the tail delta `messages[sent_count:]` per turn —
never the full history.**
Enforced by a `sent_count[agent_thread_id]` cursor in the
process-global singleton. After each turn `sent_count = len(messages)`.
**Incremental cache, not periodic rebuild.**

**L3. On LangGraph summarization, rotate the underlying provider
session — but never touch `agent_thread_id`.**
Detected by `sent_count > len(messages)`.

**L4. Tools execute via LangGraph ToolNode — never inside the MCP
handler.** The MCP handler blocks on an `asyncio.Future`; LangGraph's
ToolNode executes the real tool; the resulting ToolMessage resolves
the Future. **This is what guarantees every tool call AND every tool
result ends up in `result["messages"]`**, which is the single source
of truth for dwcli broadcast, checkpointer, and TaskManager (see
`agent_process.py:1542-1592` for the broadcast loop).

**L5. The shared state is a process-global singleton.**
LangGraph rebuilds model instances during pickling /
deserialization, so `sent_count` cannot live on the model.

---

## 6. State model — stateless, stateful, or hybrid?

### Option A — Stateless (mirror `deepseek` provider)

Every turn: spawn fresh `codex exec` (no `resume`), serialize the
**entire** LangGraph history into the prompt string.

- ❌ Violates §0.4 "no full rebuild" — every turn re-serializes the
  entire conversation as text.
- ❌ Wastes Codex's prompt cache; grows linearly with conversation
  length.
- ❌ Loses Codex-side reasoning continuity (Codex sees a flat
  transcript, not its own prior reasoning items).
- The proxy MCP server still has to persist across turns, so this
  option doesn't even remove all state.

**Rejected.**

### Option B — Stateful-heavy (mirror `claude-code` closely)

Long-lived Codex subprocess per `agent_thread_id` (e.g. via
`codex app-server`), mirror `sent_count`, rotate session on
summarization.

- ✅ Maximum prompt-cache reuse.
- ❌ Inherits every claude-code bug class (stale long-lived client,
  sent_count corruption between agent / TaskManager / subagents,
  deadlocks).
- ❌ `codex app-server` is marked experimental.

**Rejected.**

### Option C — Stateful-light. RECOMMENDED.

Per `agent_thread_id`, persist exactly **two scalars** in a
process-global singleton:

- `codex_session_uuid: str | None` — UUID captured from Codex's first
  `thread.started` event for this `agent_thread_id`.
- `sent_count: int` — how many messages we've already mirrored into
  that Codex session.

Per-turn flow:

- **Turn 1** (no cached `codex_session_uuid`):
  `codex exec --experimental-json - <<< "<prompt body>"`. On the
  first `thread.started` event, cache its UUID against
  `agent_thread_id`.
- **Turn N>1**:
  `codex exec resume <codex_session_uuid> --experimental-json -
  <<< "<serialized messages[sent_count:]>"`. Only the delta is in
  the prompt. ToolMessages are NOT in the prompt — they resolve
  futures in the proxy MCP server (§7).
- **Summarization** (`sent_count > len(messages)`): drop the cached
  `codex_session_uuid`. Next turn starts a fresh Codex session.
  **`agent_thread_id` is not touched, the LangGraph thread stays
  the same, dwcli/web UI subscriptions are not interrupted.**

After each successful turn: `sent_count = len(messages)`. Updated
**before** awaiting the response (mirroring claude-code at
`chat_model.py:598`), so a failed `recv` does not cause re-sending
on retry.

This is **the same incremental-mirror pattern** that claude-code
uses, with the only differences that the "session id" being rotated
is a UUID instead of a string, and the "client" is a short-lived
`codex exec` instead of a long-lived `ClaudeSDKClient`.

**Persistence across restarts.** The two scalars
(`sent_count` + `codex_session_uuid`) are ALSO persisted to disk so a
restarted agent does not "forget" — without it, a fresh process starts at
`sent_count=0` / `codex_session_uuid=None`, replays the whole history into an
empty Codex session, and drops every AIMessage / tool result. The on-disk
layer (`codex_provider/session_state.py`, stored in the agent's local
state directory as `codex_session_state.json`), the lazy-seed in
`CodexClientState.get_or_create`, the `persist()` call sites, and the
single resume fallback (stale/missing session → one fresh-session retry with
full-history replay) are specified in
[`DESIGN_HISTORY_RESTORE_ON_RESTART.md`](./DESIGN_HISTORY_RESTORE_ON_RESTART.md).

### Why this satisfies every hard constraint (§0 cross-check)

| Constraint | How Option C honours it |
|---|---|
| Don't change deep-worker contracts | `agent_thread_id` is read-only input; nothing we do mutates it. The deep-worker checkpointer thread is untouched. |
| All messages in LangGraph history | We never bypass ToolNode; ToolMessages arrive through LangGraph and we treat them as Future resolutions (§7.5). Native Codex tools that overlap deep-worker tools are disabled with `disabled_tools=[...]` and rejected if observed. Every accepted tool call + result is in `result["messages"]`. |
| Lightweight, no external processes | Per-turn `codex exec` is the ONLY subprocess; it exists because Codex itself is an external CLI binary, not because we add an extra process. The proxy MCP server is an in-agent-process asyncio task (§7.2). |
| Incremental history, no full rebuild | `sent_count` cursor; per-turn payload is `messages[sent_count:]` only. |
| No polling, no blocking | Subprocess stdout via `await stdout.readline()`; MCP handler awaits a Future; nothing in the steady state polls. (§10.1) |
| No memory leaks | Two scalars per `agent_thread_id`; one HTTP server per agent process; `codex exec` exits per turn. (§11) |

---

## 7. Tool execution — the proxy MCP server

### 7.1 Why an MCP server at all

Codex CLI is a separate OS process. **It cannot call functions inside
our address space directly** (this is unlike Claude SDK, which is a
Python library in the same process and lets us register handlers
in-process via `create_sdk_mcp_server`). The ONLY mechanism Codex
provides for calling host-defined tools is an external MCP server it
connects to over HTTP. We declare ours **per-invocation** via
`--config mcp_servers.<name>.url=...` (NOT via the global
`codex mcp add` / `~/.codex/config.toml` registry — see §4.3.1).

Therefore, exposing deep-worker tools to Codex **requires** us to
host an MCP server Codex CLI can connect to. There is no "in-process
tools" alternative — that path is not available with Codex. This is
the one hard constraint the Codex transport imposes; everything else
in our design honours §0.3 (no external processes).

### 7.2 In-agent-process server, NOT an external process

The MCP server is hosted **inside the deep-worker agent process** as
an asyncio task running a small HTTP listener bound to
`127.0.0.1:<random-free-port>`. Concretely: an
`mcp.server.streamable_http` instance (the Python MCP SDK already
ships this — `mcp/server/streamable_http.py`),
mounted on a small ASGI app and `asyncio.create_task`-ed during
`bind_tools()` or first `ainvoke()`. **No `subprocess.Popen`, no
`multiprocessing.Process`, no `os.fork`.** The handler functions are
plain `async def` in the agent's own event loop.

Topology:

```
                 ┌─────────────────────────── deep-worker agent process ───────────────────────────┐
                 │                                                                                  │
                 │   ┌─ LangGraph ────────┐    ┌─ CodexChatModel ──┐    ┌─ Proxy MCP server ─────┐  │
                 │   │ messages channel   │    │ ainvoke()         │    │ asyncio task on        │  │
                 │   │ ToolNode           │◄──►│ exec_runner       │    │ 127.0.0.1:<port>       │  │
                 │   │ checkpointer       │    │ client_state      │◄──►│ pending_calls          │  │
                 │   │ broadcast subs     │    └───────────────────┘    │ resolve_call()         │  │
                 │   └────────────────────┘                             └────────────────────────┘  │
                 │                                  ▲                              ▲                │
                 └──────────────────────────────────┼──────────────────────────────┼────────────────┘
                                                    │ stdin/stdout                 │ HTTP MCP
                                                    │ (per-turn)                   │ (per-turn connect)
                                       ┌────────────▼──────────────────────────────▼─────────┐
                                       │            codex exec subprocess (per turn)         │
                                       │            spawned by exec_runner; exits on         │
                                       │            turn.completed / turn.failed             │
                                       └─────────────────────────────────────────────────────┘
```

### 7.3 Why HTTP MCP and not stdio MCP for the proxy

Stdio MCP servers, in Codex's model, are **re-spawned by Codex CLI as
child processes on every `codex exec`**. That would mean: the proxy
MCP server is a separate OS process, with a new PID each turn, and
its `pending_calls` registry starts empty every time. That breaks our
model: between the ainvoke that returned `tool_calls=[...]` and the
ainvoke that resolves them, the proxy would have died and been
respawned.

It also literally adds an external process per turn — violates §0.3.

HTTP MCP, by contrast, is a server hosted by us; Codex CLI is a
client that connects fresh per `codex exec`. The server is long-lived
(lifetime = agent process); the connections are short-lived. The
`pending_calls` registry persists where it must (in our process);
Codex's per-turn lifecycle stays clean.

This is **not** "external process for the proxy" — it's "TCP loopback
between two things in the same agent process, plus a Codex subprocess
that connects to the listener". The listener itself is asyncio.

### 7.4 The Future-based handler — job-handle by default (identical to `claude_code_provider/proxy_mcp_server.py`)

**HARD POLICY (2026-06-16, common to ALL providers — no exceptions; see
`docs/POLICIES.md` §7.1.)** Every tool is waited on with a FINITE bound so
it can never hang the turn forever. On the wait elapsing the live Future is
**parked as a job** (work is NOT cancelled — `shield`) and the handler
returns a **non-error** `in_progress` job-handle. A timeout is **never**
`is_error`. The only opt-out is explicit (`long_running=False` + a required
`timeout`); a timeout-less opt-out is rejected at construction.

```python
async def handler(args: dict) -> list[dict]:
    call_id = f"{tool_name}_{counter}_{id(args)}"
    fut: asyncio.Future[MCPResult] = asyncio.get_event_loop().create_future()
    self.pending_calls[call_id] = {"future": fut, "name": tool_name, "args": args}
    mode, wait_seconds = self._wait_policy.get(
        tool_name, ("job_handle", self._long_running_wait_seconds))  # default 120s
    try:
        # shield: the timeout fires WITHOUT cancelling the real work.
        result = await asyncio.wait_for(asyncio.shield(fut), timeout=wait_seconds)
        return result.get("content", [])
    except asyncio.TimeoutError:
        # Park the live Future as a job — work continues, picked up by
        # wait_for_job / kill_job. NON-error in_progress notice.
        self._jobs[call_id] = {"future": fut, "tool_name": tool_name, "args": args}
        return [{"type": "text", "text":
            f"Tool '{tool_name}' is still running after {wait_seconds:g}s. "
            f"status: in_progress. job_id: {call_id}. "
            f"actions: wait_for_job(job_id=\"{call_id}\") or kill_job(job_id=\"{call_id}\")."}]
    finally:
        self.pending_calls.pop(call_id, None)  # parked Future stays in self._jobs
```

> ⚠️ The pre-2026-06-16 design returned an `is_error` block on timeout and
> dropped the call (work lost). That is now FORBIDDEN — it both lied to the
> agent (a still-running tool is not an error) and lost in-flight work. Do
> not reintroduce a bare `await future` or a timeout that cancels/erroring.

End-to-end timing of one full tool round-trip:

```
LangGraph ToolNode             CodexChatModel        codex exec        proxy MCP (in-process)
       │                              │                   │                      │
       │ ainvoke(messages)            │                   │                      │
       ├─────────────────────────────►│                   │                      │
       │                              │ exec_runner.run() │                      │
       │                              ├──────────────────►│                      │
       │                              │                   │ tools/call (HTTP)    │
       │                              │                   ├─────────────────────►│
       │                              │                   │                      │ Future created
       │                              │                   │                      │ pending_calls[call_id]
       │                              │  item.completed   │                      │
       │                              │  (mcp_tool_call)  │                      │
       │                              │◄──────────────────┤                      │
       │ AIMessage(tool_calls=[…])    │                   │                      │
       │◄─────────────────────────────┤                   │                      │
       │                              │                   │     …                │  awaiting Future
       │ ToolNode runs the tool       │                   │                      │
       │ ainvoke([…, ToolMessage])    │                   │                      │
       ├─────────────────────────────►│                   │                      │
       │                              │ resolve_call(     │                      │
       │                              │   call_id, result)├──────────────────────►│ Future.set_result
       │                              │                   │ tools/call response  │
       │                              │                   │◄─────────────────────┤
       │                              │  Codex continues  │                      │
       │                              │  item.completed   │                      │
       │                              │  (agent_message)  │                      │
       │                              │◄──────────────────┤                      │
       │                              │  turn.completed   │                      │
       │                              │◄──────────────────┤                      │
       │ AIMessage(content="…")       │                   │                      │
       │◄─────────────────────────────┤                   │                      │
```

### 7.5 Critical: every tool call AND every tool result lands in LangGraph history

The user's question — "will all tool calls go through LangGraph and
be stored in deep-worker's thread?" — is answered as follows.

The proxy MCP handler **never executes the tool itself**. It only
parks an awaitable Future. The only thing that can ever resolve that
Future is `resolve_call(call_id, ...)`, which is called from exactly
one place: `CodexChatModel.ainvoke` when it receives a ToolMessage
from LangGraph.

ToolMessages enter LangGraph **only as outputs of ToolNode**. ToolNode
runs in the LangGraph node graph; its output (a ToolMessage with
`tool_call_id`, `name`, `content`, `status`) is appended to the
`messages` channel in the LangGraph state.

So the chain is, for every single tool call:

1. Codex emits an MCP `tools/call` over HTTP → arrives at our
   in-process handler.
2. Handler parks Future, **returns nothing yet**.
3. Meanwhile the JSONL stream carries an `item.completed` event with
   `type=mcp_tool_call`, which `ainvoke` exposes to LangGraph as
   `AIMessage(tool_calls=[{id, name, args}])`. ← **this AIMessage IS
   in `result["messages"]`.**
4. LangGraph routes to ToolNode → ToolNode executes the real
   deep-worker tool → produces a ToolMessage. ← **this ToolMessage IS
   in `result["messages"]`.**
5. ToolMessage arrives at the next `ainvoke` → we call
   `resolve_call(tool_call_id, content)` → Future unblocks → MCP
   handler returns to Codex → Codex continues.

Both the `tool_calls` and the matching `ToolMessage` are in
`state["messages"]`. The broadcast code at
`agent_process.py:1542-1592` reads them and pushes them to dwcli /
web UI subscribers. The checkpointer persists them. TaskManager reads
them. Full history — **identical guarantee to `claude-code`**.

### 7.6 What deep-worker tools Codex sees

`CodexChatModel.bind_tools(tools)` extracts metadata (name,
description, args schema) from each `BaseTool` — exact same algorithm
as `claude_code_provider/chat_model.py:291-378` (Pydantic schema
first, fallback to `inspect.signature` for StructuredTool-via-
ToolRuntime). Each tool is registered as an MCP tool on the proxy.
Codex's view: a single MCP server named e.g. `deepworker_<pid>`
exposing N tools.

A Codex-side built-in deny-list (counterpart of
`claude_code_provider/client_manager.py:58-140` `BUILT_IN_TOOLS`) may
be needed once we see which Codex built-ins (its own shell, patch
tool, web_search, etc.) compete with the deep-worker toolset and
should be disabled. Open question Q5.

---

## 8. Module layout

Sibling of `claude_code_provider/`. The package is empty today —
only this `docs/` directory exists.

```
codex_provider/
├── __init__.py            # re-exports CodexChatModel
├── chat_model.py          # CodexChatModel(BaseChatModel)
├── client_state.py        # process-global singleton: agent_thread_id → _ThreadState
├── exec_runner.py         # asyncio.create_subprocess_exec + JSONL parser
├── proxy_mcp_server.py    # in-agent-process HTTP MCP server (asyncio task)
├── custom_types.py        # MCPResult, PendingCall, ToolMetadata, _ThreadState
├── pyproject.toml
├── py.typed
├── docs/
│   └── DESIGN_CODEX_PROVIDER.md   ← this file
└── tests/
    ├── test_exec_runner.py
    ├── test_client_state.py
    ├── test_chat_model.py
    ├── test_proxy_mcp_server.py
    └── test_integration.py
```

Mirrors `claude_code_provider/` one-for-one.

---

## 9. `CodexChatModel` public surface

```python
class CodexChatModel(BaseChatModel):
    # model_config and __get_pydantic_json_schema__ / model_dump
    # overrides are MANDATORY (LangGraph pickles models during
    # checkpointing; raw BaseChatModel fails on callable fields).
    # Mirror claude_code_provider/chat_model.py:132-246 verbatim,
    # changing only the class name and _llm_type.

    _client_state: Any = PrivateAttr(default=None)
    _proxy_mcp: Any = PrivateAttr(default=None)
    _tools_metadata: list[ToolMetadata] = PrivateAttr(default_factory=list)
    _working_dir: str | None = PrivateAttr(default=None)   # set via setattr by agent_process.py:447-448
    _model_name: str | None = PrivateAttr(default=None)
    _sandbox_mode: str = PrivateAttr(default="workspace-write")
    _approval_policy: str = PrivateAttr(default="never")

    def __init__(self, model=None, max_tokens=None,
                 sandbox_mode="workspace-write",
                 approval_policy="never", **kwargs): ...

    def bind_tools(self, tools, **kwargs) -> "CodexChatModel":
        # 1. Extract tool metadata, identical to claude_code_provider.
        # 2. Ensure the in-process proxy MCP server is initialised in
        #    the singleton with this tool set.
        # 3. Start the proxy server (bind a port). We do NOT register it
        #    globally via ``codex mcp add`` — see §4.3.1. The server is
        #    injected per-invocation into each ``codex exec`` via
        #    ExecArgs.mcp_server_name/mcp_server_port + --ignore-user-config.
        ...

    async def ainvoke(self, input, config=None, **kwargs) -> AIMessage:
        ...    # see §10

    def _generate(self, *args, **kwargs):
        raise NotImplementedError("CodexChatModel supports async only. Use ainvoke().")

    @property
    def _llm_type(self) -> str:
        return "codex"
```

Interface intentionally identical to `ClaudeCodeChatModel` so the
`config.py` factory branch is a clean copy with class renaming.

---

## 10. `ainvoke` turn algorithm — event-driven, non-blocking

```python
async def ainvoke(self, input, config=None, **kwargs) -> AIMessage:
    # 1. Normalize input to list[BaseMessage].
    messages = _normalize(input)

    # 2. agent_thread_id is MANDATORY. No fallback.
    config = ensure_config(config)
    cfg = config.get("configurable", {})
    if "thread_id" not in cfg:
        raise ValueError(
            "CodexChatModel requires explicit thread_id in config "
            "(this is deep-worker's agent name; each caller — main "
            "agent, TaskManager, subagent — must own its thread_id)."
        )
    agent_thread_id = cfg["thread_id"]

    # 3. Look up per-thread state (singleton; survives model rebuilds).
    state = self._client_state.get_or_create(agent_thread_id)

    # 4. Summarization detection — drop Codex session, restart fresh.
    #    agent_thread_id is NEVER touched.
    if state.sent_count > len(messages):
        state.codex_session_uuid = None
        state.sent_count = 0

    # 5. System prompt (first SystemMessage in messages); delivered
    #    via AGENTS.override.md (see Implementation note 5). Must be
    #    flattened from content_blocks to plain text — middleware
    #    routinely produces list-form `SystemMessage.content`.
    system_prompt = _flatten_message_content_to_text(_extract_system(messages))

    # 6. Tail delta — INCREMENTAL, NEVER full rebuild.
    new_messages = messages[state.sent_count:]

    # 7. Resolve ToolMessages in the delta against the proxy MCP.
    #    They DO NOT enter the Codex prompt — they unblock Futures.
    user_text_parts: list[str] = []
    tool_resolved = False
    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            self._proxy_mcp.resolve_call(msg.tool_call_id,
                                         _build_mcp_content(msg.content))
            tool_resolved = True
        elif isinstance(msg, HumanMessage):
            user_text_parts.append(_serialize_human(msg))
        elif isinstance(msg, (AIMessage, SystemMessage)):
            pass  # AIMessage already in Codex's session; SystemMessage via --config

    # 8. Update sent_count BEFORE awaiting (mirror claude-code).
    state.sent_count = len(messages)

    # 9. Mid-turn vs new-turn:
    #    If tool_resolved and state.current_exec is alive → keep
    #    consuming the SAME exec's events. Else → spawn fresh
    #    `codex exec [resume <codex_session_uuid>]`.
    if tool_resolved and state.current_exec is not None:
        events = state.current_exec.events
    else:
        prompt = "\n\n".join(user_text_parts)
        state.current_exec = await self._exec_runner.spawn(
            prompt=prompt,
            session_uuid=state.codex_session_uuid,
            base_instructions=system_prompt,
            working_directory=self._working_dir,
            sandbox_mode=self._sandbox_mode,
            approval_policy=self._approval_policy,
            model=self._model_name,
        )
        events = state.current_exec.events

    # 10. Event-driven consume — `await readline()`, NEVER poll.
    final_text = ""
    pending_tool_calls: list[dict] = []
    async for event in events:
        if event.type == "thread.started" and state.codex_session_uuid is None:
            state.codex_session_uuid = event.thread_id    # cache
        elif event.type == "item.started":
            it = event.item
            if it.type == "mcp_tool_call" and it.server == self._proxy_mcp.server_name:
                pending_tool_calls = await self._proxy_mcp.wait_for_pending_calls(
                    tool_name=it.tool,
                    arguments=it.arguments,
                    timeout=30.0,
                )
                break    # exec stays alive in state.current_exec
        elif event.type == "item.completed":
            it = event.item
            if it.type == "agent_message":
                final_text = it.text
            elif it.type in BLOCKED_NATIVE_ITEM_TYPES:
                raise RuntimeError("native Codex tool bypassed deep-worker proxy")
        elif event.type == "turn.completed":
            await self._exec_runner.aclose(state.current_exec)
            state.current_exec = None
            break
        elif event.type == "turn.failed":
            await self._exec_runner.aclose(state.current_exec)
            state.current_exec = None
            raise RuntimeError(f"Codex turn failed: {event.error.message}")

    if pending_tool_calls:
        return AIMessage(content=final_text, tool_calls=pending_tool_calls)
    return AIMessage(content=final_text)
```

**Where in this algorithm is any work proportional to history
length?** Step 4 reads `len(messages)` — O(1). Step 6 slices
`messages[sent_count:]` — O(delta). Step 7 iterates that slice — same.
**Nothing iterates the full history.** Step 9 spawns one subprocess
per turn (Codex behaviour, not ours). Step 10 awaits subprocess
stdout — no polling.

### 10.1 Non-blocking invariants — exhaustive checklist

| Invariant | How this design honours it |
|---|---|
| No polling | Subprocess stdout: `await stdout.readline()` (asyncio readline is non-blocking; the kernel wakes us via `epoll`). MCP handler: `await asyncio.wait_for(future, ...)`. No `while True: sleep(...)`. |
| No CPU steady-state | Between turns: no subprocess (Codex exited), no timers, no periodic tasks. The HTTP MCP listener is bound but quiescent; the underlying ASGI server idles on `epoll`. |
| No blocking ops on the loop | All subprocess I/O via asyncio streams. All MCP handlers are `async def`. No synchronous `time.sleep`, no `socket.recv` without await. |
| Event-driven end to end | One event source per turn: the Codex JSONL stream. Tool resolution is event-driven (`Future.set_result`). |
| Repeated client polls don't trigger heavy work | The provider is invoked **only** by LangGraph when the agent has new input — never by clients pulling status. `_handle_get_history` in `agent_process.py` already uses `_message_dict_cache` (memoized per message id) and we don't touch that path. |

There is **zero** code path here that does work proportional to
(`number_of_clients` × `poll_frequency`) or to (`total_history_length`
× `turn_count`). Both classes are forbidden by §0.4–§0.5 and the
design respects that.

---

## 11. Lifecycle and leak analysis

### Long-lived objects (per agent process)

1. **`CodexClientState` singleton** — process-global dict
   `agent_thread_id → _ThreadState`. Each `_ThreadState` is 3 scalars
   + (mid-turn only) one `ExecHandle`. Bounded by the number of
   distinct `agent_thread_id`s the agent ever uses (main agent +
   TaskManager + subagents). For a typical agent that's
   O(1)–O(N_subagents). Acceptable.
2. **`ProxyMCPServer` HTTP listener** — one per agent process. Bound
   on first `bind_tools`; torn down on agent shutdown. A pending Future's
   handler wait is bounded (default 120 s); on elapse the Future is moved
   out of `pending_calls` into `self._jobs` (the job-handle) — still
   one-shot and dropped once resolved or the job is killed. No leak from
   runaway tool calls (see §7.4 for the job-handle policy).

### Short-lived objects (per turn)

- One `codex exec` subprocess per Codex turn. Exits on
  `turn.completed` / `turn.failed` / our explicit kill on cancel.
- One MCP TCP connection per `codex exec`. Closed when the subprocess
  exits.

### Mid-turn objects (between two `ainvoke` calls inside one Codex turn)

- `state.current_exec` holds the live subprocess + its event-stream
  generator. Cleared on `turn.completed` / `turn.failed` / cancel. If
  the agent process dies between two ainvokes, the subprocess is
  reaped by the OS (parent death) — no zombie.

### Cross-reference to past incidents in this project

| Incident doc | Class of leak | Applicability here |
|---|---|---|
| a related internal incident analysis (not included here) | rpyc dispatch task pin | No rpyc here. N/A. |
| a related internal incident analysis (not included here) | exception tracebacks stored on long-lived objects | Risk: an exception from a failed Codex turn could be cached on `state.current_exec` if cleared improperly. Mitigation: `state.current_exec = None` in `finally`; traceback is not retained on `_ThreadState`. |
| a related internal incident analysis (not included here) | cleanup task GC | No periodic cleanup loop here. N/A. |
| related internal incident analyses (not included here, AsyncResult retention) | rpyc AsyncResult held past `await` | No rpyc-style awaitables here. The asyncio.Future in the MCP handler is local; it is dropped on resolution, or moved to the bounded `self._jobs` job registry on the wait elapsing (and dropped when that job resolves or is killed). Either way it is one-shot and never retained indefinitely. |

---

## 12. Five claude-code lessons restated for Codex

| # | claude-code rule | Codex restatement |
|---|---|---|
| L1 | `thread_id` mandatory, unique per caller | Same — we read `config["configurable"]["thread_id"]` (the agent name) and require it. Subagent / TaskManager get suffixed `agent_thread_id`s. Each maps to its own `(codex_session_uuid, sent_count)` slot. **The deep-worker `thread_id` itself is never mutated.** |
| L2 | Mirror only `messages[sent_count:]` | Same — Codex's `resume` keeps server-side history; we send only the delta into the prompt. **Incremental, cached, never rebuilt.** |
| L3 | Rotate session on summarization | Restated: drop the cached `codex_session_uuid`. Next turn opens a fresh Codex session. **`agent_thread_id` is NOT touched** — LangGraph thread, dwcli subscriptions, checkpointer, all keep going. |
| L4 | Tools execute via LangGraph ToolNode | Same — proxy MCP handler blocks on Future; ToolNode does the real work; result resolves the Future. **Every tool call AND every tool result is in `result["messages"]`.** §7.5. |
| L5 | Singleton state in process-global module | Same — `CodexClientState` is a process-global singleton. Survives LangGraph model rebuilds during pickling. |

---

## 13. Auth, env, and provider registration

- Provider registration in `config.py:create_model` skips API-key
  validation (mirroring `claude-code` at `:1453`).
- No `CODEX_API_KEY` is set. Codex CLI uses `~/.codex/auth.json`
  automatically.
- `_PROVIDER_DEFAULT_MODELS["codex"]` — open question Q2.
- `_detect_provider`: we do NOT auto-detect Codex from a model name,
  because the obvious patterns (`gpt-5`, `o3`, ...) already belong to
  the `openai` provider. Users opt in explicitly with `--provider codex`.
- The factory branch in `config.py` is a clean copy of the
  `claude-code` branch with class/import renames.

---

## 14. What this design EXPLICITLY does NOT do

- **No changes to `claude_code_provider`.**
- **No changes to LangGraph wiring** in `deepagents/`.
- **No changes to deep-worker contracts** —
  `config["configurable"]["thread_id"] == self.name` is fixed.
- **No external processes other than the Codex CLI itself.** The
  proxy MCP server is an asyncio task in the agent process; the
  Codex subprocess is per-turn and exists because Codex *is* an
  external CLI (unavoidable). No `subprocess.Popen` of the proxy,
  no `multiprocessing`, no daemons.
- **No new tests in this PR** — test plan in §16 lands with the
  implementation PR.
- **No packaging changes** — provider is a sibling folder imported
  by `config.py` exactly like `claude_code_provider`.
- **No auto-detection from model name.**
- **No web-search re-implementation** — Codex provider reuses
  `claude_code_provider.web_search.WebSearchDelegationTool`, as
  deepseek/openai/gemini already do.

---

## 15. Open questions for the user

### Q1. Dependency strategy — D1 vs D2

(These are project-internal acronyms; spelled out in full each time
they appear.)

- **D1 — "Depend on the upstream Agents SDK Codex extension."**
  Import `Codex`, `Thread`, `CodexExec` from
  the OpenAI Agents SDK's `src/agents/extensions/experimental/codex/`. We
  write only a `BaseChatModel` wrapper on top of their primitives.
  ~50–100 LOC of our code. Cost: runtime dep on the `agents` package
  and on an upstream-experimental submodule that OpenAI may change
  before GA.
- **D2 — "Do it ourselves."** Write `exec_runner.py` as ~200 LOC: one
  `asyncio.create_subprocess_exec`, one `while True: line = await
  stdout.readline()`, one `json.loads` per line. The Codex wire
  protocol is 4 event types and self-documenting from the live JSONL
  on stdout. Zero runtime dep on the Agents SDK.

**Recommendation: D2.** Codex JSONL is short, stable, and
live-verifiable; we don't need to track upstream `experimental.codex`
API churn.

### Q2. Default Codex model name

`_PROVIDER_DEFAULT_MODELS["codex"] = (env_var, fallback)`. Candidates:
`("CODEX_MODEL", "gpt-5.1")`, `("CODEX_MODEL", "gpt-5")`,
`("CODEX_MODEL", "o3")`, something else. User pick.

### Q3. Sandbox and approval defaults

- `sandbox_mode = "workspace-write"` + `approval_policy = "never"` —
  unattended agent, can write inside working directory. Matches
  deep-worker's `--skip-permissions` posture. **Recommended default.**
- `sandbox_mode = "read-only"` + `approval_policy = "on-request"` —
  safer; needs human approval per write. Not viable unattended.
- `sandbox_mode = "danger-full-access"` — no sandbox. Not recommended.

### Q4. Proxy MCP transport: stdio vs HTTP

The recommendation in §7.3 is HTTP. The user's "no external processes"
constraint also points to HTTP — stdio would force Codex to fork the
proxy server as a child process each turn, which IS an external
process. **Recommended: HTTP.**

### Q5. Codex built-in tool deny-list

`claude_code_provider/client_manager.py:58-140` maintains
`BUILT_IN_TOOLS` to disable Claude Code's own built-ins so all tool
execution flows through LangGraph. Codex has its own set of built-ins
(shell, patch tool, web_search, memory, todo_list, …). Which should
we disable so that all tool execution flows through LangGraph?
Preliminary list to be confirmed during implementation:
`shell`/`exec`, `patch`/`apply_patch`, `web_search`, `memory`,
`todo_list`.

---

## 16. Test plan (for the implementation PR, not this doc)

Unit tests (mocked subprocess):

- `test_exec_runner.py` — JSONL parser handles every event type;
  malformed JSON raises cleanly; subprocess termination on signal;
  `idle_timeout_seconds` cancellation.
- `test_client_state.py` — `sent_count` increments correctly;
  `sent_count > len(messages)` drops `codex_session_uuid` **without
  touching `agent_thread_id`**; thread isolation between
  `agent` / `agent:tm` / `agent:sub_*`; singleton survives
  `del state; state = get_global_client_state()`.
- `test_chat_model.py` — `ainvoke` requires `thread_id`;
  `bind_tools` builds metadata from `BaseTool` / `StructuredTool`;
  `_working_dir` setter works; Pydantic `model_dump` excludes
  callables.
- `test_proxy_mcp_server.py` — handler creates Future, `resolve_call`
  sets result, on the bounded wait elapsing the handler parks the live
  Future as a job and returns a **non-error** `in_progress` job-handle
  (never `is_error`), server starts on `127.0.0.1:0` with random free
  port, `codex mcp add` registration is idempotent.
- `test_job_handle_by_default.py` / `test_job_handle_integration.py` —
  pin the job-handle-by-default policy (§7.4); the integration test
  proves the production 120 s wait by mocking ONLY the clock.

Integration tests (live `codex` binary, cross-process per the
project's mandatory rule — proxy and exec are in separate processes
by design):

- `test_integration.py::test_hello_world` — single ainvoke, no tools.
- `test_integration.py::test_tool_call_e2e` — ainvoke emits
  `tool_calls`, second ainvoke resolves it, final answer arrives,
  **both `tool_calls` AIMessage and `ToolMessage` are present in
  `result["messages"]`** (this is the §0.2 check).
- `test_integration.py::test_multi_turn_resume` — second turn uses
  `codex exec resume <uuid>`; verify by process-list inspection.
- `test_integration.py::test_summarization_resets_session` — feed
  `messages` shorter than `sent_count`; verify `codex_session_uuid`
  is dropped, fresh Codex session starts, **`agent_thread_id`
  unchanged**.

---

## 17. Verification of THIS design (no code needed)

A reviewer can validate this document without running anything:

1. Re-run the live JSONL test from §4.1 on this host; confirm output
   matches.
2. `ls -l ~/.codex/auth.json` — cached login present → no API key
   needed.
3. `codex mcp list` — should be EMPTY. We never register globally;
   the proxy server is injected per-invocation (§4.3.1). A non-empty
   list means an old build polluted the global config — clean it.
4. Read `agent_process.py:915, 1016, 1143` and confirm
   `config["configurable"]["thread_id"] == self.name`. Design treats
   this as immutable input.
5. Read `agent_process.py:1542-1592` (the broadcast loop) and confirm
   it reads `tool_calls` and `ToolMessage`s from `result["messages"]`.
   Design routes all tool round-trips through LangGraph, so they
   appear there.
6. Read `claude_code_provider/proxy_mcp_server.py:103-159` and
   confirm the Future pattern. The Codex proxy implements the same
   pattern with HTTP transport instead of in-process SDK handlers.
7. Read `claude_code_provider/client_manager.py:319-355` for the
   `effective_session_id` / `rotate_session` precedent — Codex design
   renames these to `codex_session_uuid` and "drop on summarization",
   same effect, clearer names.
8. Read the Python MCP SDK's `server/streamable_http.py`
   — Python MCP SDK ships an in-process HTTP server we mount on
   asyncio. No external process is needed for the proxy.

If every check passes, the design is grounded.

---

## 18. Summary recommendation

Build the Codex provider as a sibling of `claude_code_provider/` with
the **Stateful-light** state model: per `agent_thread_id`, hold
`(codex_session_uuid, sent_count)` in a process-global singleton; one
short-lived `codex exec` per Codex turn; one **in-agent-process HTTP
MCP server** (asyncio task bound to `127.0.0.1:<random-port>`, **not**
a separate OS process) per agent process; ToolMessage → Future-resolve
bridge identical in spirit to claude-code's; **every tool call and
every tool result remains in `result["messages"]`**.

Use **D2** (own subprocess + JSONL parser) unless the user has a
reason to prefer **D1** (Agents SDK dependency).

Default to `sandbox_mode="workspace-write"` +
`approval_policy="never"` for unattended agents; override via flags.

Use HTTP transport for the proxy MCP server.

Pending the user's answers to Q1–Q5, this design is ready for
implementation. No code is written by this document.
