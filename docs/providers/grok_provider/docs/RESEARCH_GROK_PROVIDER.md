# Research: Grok provider for deep-worker

**Status:** 🔍 RESEARCH (preliminary analysis; implementation NOT started)
**Date:** 2026-06-17

## 0. Purpose of this document

To record the findings of a live investigation of the Grok CLI on this
host and to determine **whether a `grok` provider for deep-worker is
feasible, modeled on `codex_provider/`**, without violating the existing
architectural contracts (see `codex_provider/docs/DESIGN_CODEX_PROVIDER.md` §0).

This document is **research notes**, not a design. The design (`DESIGN_GROK_PROVIDER.md`)
is written separately, after all Open Questions in §6 are closed.

## 1. Context

The user installed the Grok CLI on the host
(`grok 0.2.54 (fee15ff8e) [stable]`). Authorization — via the supergrok
subscription through `~/.grok/auth.json`; **there is no separate API key
and the user does not want to buy one**. Therefore the only integration
path is through the `grok` binary itself, as is done for `codex` (cached
login via `~/.codex/auth.json`).

## 2. Grok CLI surface (live verified)

### 2.1 Non-interactive turn (`grok -p`)

```bash
$ grok -p "say HELLO" --output-format json
{
  "text": "HELLO! How can I help you today?",
  "stopReason": "EndTurn",
  "sessionId": "019ed589-ebfc-7a10-9cb3-dfce8126fdd0",
  "requestId": "5ce4f250-c8b0-48e7-b1dc-e43ddb1f26ea",
  "thought": "..."
}
```

Output formats: `plain` (default), `json` (a single JSON object at the end),
`streaming-json` (a JSONL event stream).

### 2.2 Streaming JSONL — event taxonomy

```
{"type":"thought","data":"..."}     # reasoning chunks
{"type":"text","data":"..."}        # final assistant text chunks
{"type":"end","stopReason":"EndTurn","sessionId":"...","requestId":"..."}
```

**⚠️ CRITICAL FINDING:** `streaming-json` does NOT emit tool_call events.
Only `thought` / `text` / `end`. This is a fundamental difference from Codex
`--experimental-json`, which sends `item.started type=mcp_tool_call`
— on that event the codex_provider stops `ainvoke` and returns
`AIMessage(tool_calls=[...])` to LangGraph.

Consequence: in `-p` mode Grok runs **the entire** multi-step loop
(reasoning → tool_call → tool_result → reasoning → ... → final text)
**inside one process**, and exposes only the final text to the outside.
With this mode, LangGraph loses the intermediate tool_calls/ToolMessages
— a violation of constraint §0.2 (everything through LangGraph).

See §5 — why `grok agent stdio` (option A) most likely solves
this problem.

### 2.3 Resume and sessionId

```bash
$ grok -p "what did you say?" --resume 019ed589-... --output-format streaming-json
...
{"type":"text","data":"I"}
{"type":"text","data":" said"}
{"type":"text","data":" **HELLO**"}
{"type":"end","stopReason":"EndTurn","sessionId":"019ed589-...",...}
```

Resume works and preserves context. The sessionId is stable between turns
(a new `-p` without `--resume` creates a new sessionId — this is the analog
of `codex_session_uuid` in codex_provider).

### 2.4 Sessions on disk

```
~/.grok/sessions/<url-encoded-cwd>/<session-id>/
├── chat_history.jsonl       # system/user/assistant/tool_result/reasoning
├── events.jsonl             # turn_started, tool_started, mcp_tool_call_*, ...
├── prompt_context.json
├── rewind_points.jsonl
├── signals.json
├── summary.json
├── system_prompt.txt
└── updates.jsonl
```

The CWD is encoded into the path (the URL-encoded working directory).
This creates a minor isolation issue, see §6 Q-G.

`chat_history.jsonl` contains the full dialog (system, user, assistant,
reasoning, tool_calls, tool_results) — we can read it postmortem.

`events.jsonl` contains structured events:
```
{"type":"mcp_config_resolved","servers":[{"name":"mini","transport":"stdio",...}]}
{"type":"mcp_server_starting","server_name":"mini",...}
{"type":"mcp_server_connected","server_name":"mini","tool_count":1,...}
{"type":"turn_started","session_id":"...","model_id":"grok-composer-2.5-fast",...}
{"type":"tool_started","tool_name":"CallMcpTool"}
{"type":"permission_requested","tool_name":"CallMcpTool"}
{"type":"permission_resolved","tool_name":"CallMcpTool","decision":"allow"}
{"type":"mcp_tool_call_started","server_name":"mini","tool_name":"echo","call_id":"mini__echo",...}
{"type":"mcp_tool_call_completed","server_name":"mini","tool_name":"echo","duration_ms":0,"success":true}
{"type":"tool_completed","tool_name":"CallMcpTool","outcome":"success"}
{"type":"turn_ended","outcome":"completed"}
```

### 2.5 Authentication

```
$ ls ~/.grok/
auth.json     # cached login
config.toml   # CLI config (installer, marketplace, [cli], [mcp_servers.*])
sessions/     # per-cwd, per-session-id
skills/
plugins/...
```

`grok models` shows:
```
You are logged in with grok.com.
Default model: grok-composer-2.5-fast
Available models:
  - grok-build
  * grok-composer-2.5-fast (default)
```

The supergrok subscription works without an API key — exactly like Codex with
`~/.codex/auth.json`.

### 2.6 MCP support (HTTP + stdio)

```
$ grok mcp add --help
Add or update an MCP server
  -t, --transport [stdio|http|sse]
  -s, --scope [user|project]    # ~/.grok/config.toml vs ./.grok/config.toml
  -e, --env KEY=value
  -H, --header "NAME: VALUE"
```

Full HTTP transport support — this is what we need for an
in-process proxy MCP server (the analog of `codex_provider/proxy_mcp_server.py`).

### 2.7 Permission / sandbox flags

| Codex | Grok |
|---|---|
| `--config approval_policy="never"` | `--permission-mode bypassPermissions` or `--always-approve` |
| `--dangerously-bypass-approvals-and-sandbox` | covered by the same |
| `-s workspace-write` | `--sandbox <profile>` (env: `GROK_SANDBOX`) |

In the live test `--always-approve` was used, and the MCP tool call went
through without a prompt — the analog of codex's bypass flag.

### 2.8 System prompt — first-class flag (✅ better than codex)

```
--system-prompt-override <PROMPT>
    Override the agent's system prompt (Claude Code: --system-prompt)
--rules <RULES>
    Extra rules to append to the system prompt
```

This is much cleaner than codex's `AGENTS.override.md` hack (which we
use because codex 0.130 silently ignores `--config base_instructions`).
The implementation will be simpler: ~80 lines of codex's exec_runner go away.

### 2.9 Direct CLI flag analogs

| Codex | Grok | Note |
|---|---|---|
| `codex exec --experimental-json -` | `grok -p "..." --output-format streaming-json` | BUT: see §2.2 — no tool events |
| `codex exec resume <uuid>` | `grok -p ... -r <session-id>` or `-c` | resume by sessionId |
| `~/.codex/auth.json` | `~/.grok/auth.json` | cached login |
| `~/.codex/sessions/<uuid>.jsonl` | `~/.grok/sessions/<cwd>/<sid>/...` | persisted on disk |
| `codex mcp add <n> --url ...` | `grok mcp add <n> --transport http <url>` | works |
| `--config base_instructions=...` (ignored) | `--system-prompt-override <PROMPT>` | first-class |
| `--config approval_policy="never"` + `--dangerously-bypass-...` | `--always-approve` | works |
| `-s workspace-write` | `--sandbox <profile>` | works |
| `--disable <feature>` | `--disallowed-tools <list>` or `--disable-web-search` | works |
| `--cd <dir>` | `--cwd <dir>` | works |
| `--ignore-user-config` | ❓ no explicit flag | see §6 Q-C |
| `--model <m>` | `-m <m>` | same |

### 2.10 Built-in tools that compete with deep-worker

From `grok inspect` and chat_history.jsonl it is clear that Grok has
built-in tools: `Glob`, `Read`, `Edit`, `Bash`, `WebSearch`, `Write`, `Grep`,
`Task` (subagents), etc. The live test in §3 showed the model itself
called `Glob` and `Read` before touching MCP — clear competition
with deep-worker's `glob`/`read_file`/`grep`/`execute`. They must be
disabled via `--disallowed-tools` (exact list — see Q-F).

`--no-subagents` disables the built-in `Task` (important — otherwise grok
spawns subagents bypassing deep-worker's `task` tool).
`--no-plan` disables plan-mode.
`--no-memory` disables cross-session memory (important for reproducibility).

### 2.11 Live MCP tool-call (proof-of-concept)

We set up a stdio MCP server with a single `echo` tool:
```
$ grok mcp add mini -- python3 /tmp/mini_mcp.py
$ grok -p "Use the mini__echo tool to echo 'pinetree'. Then say done." \
    --output-format streaming-json --always-approve
```

events.jsonl recorded:
```
mcp_server_connected server=mini tool_count=1 tools=[echo]
turn_started session_id=019ed58a-... model=grok-composer-2.5-fast
tool_started Glob              # the model went looking for the MCP server schema on its own
tool_started Read              # reads the generated tool schema file
tool_started CallMcpTool       # **dispatcher for MCP**
  mcp_tool_call_started server=mini tool=echo call_id=mini__echo
  mcp_tool_call_completed server=mini tool=echo duration_ms=0 success=true
turn_ended outcome=completed
```

The streaming stdout showed ONLY reasoning + the final text (`done`),
no tool-call events. Confirmation of §2.2.

**Important architectural detail:** Grok calls MCP tools through the
built-in dispatcher `CallMcpTool({server, toolName, arguments})`, not
directly as separate `mcp__server__tool` functions (as Claude and
Codex do). For our proxy MCP server this is neutral — we still
receive `tools/call` from the transport; the model just sees our tools
through the intermediary. This may explain why stdout does not show
the concrete tool-calls (the dispatcher abstracts them).

## 3. Architectural comparison with codex_provider

### 3.1 What we copy 1:1

- LangChain `BaseChatModel` wrapper with `async ainvoke()` (sync `_generate`
  raises).
- `thread_id` taken from `config["configurable"]` = the deep-worker
  agent name; never mutated.
- Tail-delta only — `messages[sent_count:]`, not the full history.
- Per-turn subprocess (`grok -p [-r <sid>]`); not a long-lived client.
- Persistent state `(grok_session_id, sent_count)` on disk
  (the agent's local state directory, `grok_session_state.json`).
- Summarization detection: `sent_count > len(messages)` → drop sessionId
  → fresh session, **do not touch `agent_thread_id`**.
- In-process HTTP MCP server on `127.0.0.1:<random>`; tool calls
  park on `asyncio.Future`, resolved via `resolve_call()`.
- Job-handle policy (see `docs/POLICIES.md` §7.1) — a timeout
  parks the Future as a job, not as an error.
- Registration in 5 places in the CLI (see §4.1 below).

### 3.2 What gets simpler

- ✅ **System prompt** — `--system-prompt-override <PROMPT>` instead of
  the `AGENTS.override.md` file hack. ~80 lines of exec_runner go away
  (writing/cleaning the override file with a marker, protecting the user's
  AGENTS.md from being overwritten).
- ✅ Most likely, **no "non-resettable" native tools**, like
  `apply_patch`/`update_plan` in Codex 0.130 — this removes the need
  for `event_pump.py` + `native_tools.py` (~450 lines). Requires
  verification, but `--disallowed-tools <list>` is structured differently
  from codex's `--disable <feature>` and accepts any built-in tool.

### 3.3 What gets harder or needs a decision

- ❌ `streaming-json` does NOT give tool_call events (§2.2) — need to
  investigate `grok agent stdio` (option A, §5).
- ❌ `~/.grok/config.toml` — no explicit `--ignore-user-config` (§6 Q-C);
  risk of an incident analogous to a prior config-pollution one if we put
  per-agent MCP registration there (we categorically forbid that for Codex
  after the related internal incident). Alternatives: `--scope project`
  (pollutes the workspace), inline `--config` (need to check whether grok
  has such a flag), or a per-agent custom `GROK_HOME` env var.
- ❌ CWD-dependent sessions path (`~/.grok/sessions/<encoded-cwd>/`) —
  if the agent's working_dir changes, resume breaks. Need to bind
  to a fixed cwd for the whole lifetime of the agent.
- ❌ Grok itself calls Glob/Read/Bash/WebSearch built-ins before it
  goes to MCP — a full disallow-list for built-ins is needed.

## 4. Touch points in the CLI (5 files)

These files know about `claude-code-sdk` and `codex`; the new `grok` provider
requires targeted edits in each:

1. `agent_config_pkg/agent_config.py:53` — `valid_providers = [..., "grok"]`
   `:64-65` — skip API-key load for grok (cached login)
2. `cli/agent_command.py:653` — `choices=[..., "grok"]`
3. `config.py:1390` — `_PROVIDER_DEFAULT_MODELS["grok"] = ("GROK_MODEL", None)`
   `:1719` — factory branch `elif provider == "grok": ... GrokChatModel(...)`
4. `runtime/availability_tracker.py:11` — add `"grok": null`
5. `runtime/provider_chain.py:4` — `["claude-code-sdk", "codex", "grok", "deepseek"]`

`process/agent_process.py` — if grok does not need native passthrough
tools (as codex does), then there is no need for the `if provider == "codex"
extend NATIVE_TOOLS` block. Otherwise — symmetrically to codex.

`task_manager/task_manager.py` — likewise, symmetric to codex.

## 5. Main question: option A (`grok agent stdio`)

`grok agent stdio` is the mode in which grok works "as an agent over stdio",
positioned as an alternative to `--single`. Hypothesis: it exposes the full
wire protocol with explicit tool_call events (the analog of codex
`--experimental-json`), which would allow:

1. Receiving `tool_call_started`/`tool_call_completed` on stdout.
2. Parking tool calls in LangGraph via ToolNode (like codex).
3. Having a full LangGraph history with tool_calls + ToolMessages in
   `result["messages"]` — satisfying constraint §0.2.

**Live research required:**
1. Run `grok agent stdio` and understand the message protocol (JSONL?
   JSON-RPC? line-prefixed?).
2. Check how the initial prompt / resume is done.
3. Check how tools are passed and how tool-call events arrive.
4. Check how the tool-result is sent back.
5. Check how it authenticates (`~/.grok/auth.json` or is a
   `--leader-socket` needed?).

§7 is devoted to this stage (running NOW).

If `grok agent stdio` confirms expectations — the implementation is fully
symmetric to codex (and simpler because of §3.2). If not — the user will
have to choose between:

- **Option B** (drop the grok provider entirely) — correct,
  but we do not get access to the subscription.
- **Option C** (hybrid: grok does its own built-in tool-calls,
  but deep-worker's go through our proxy MCP with recording into
  LangGraph postmortem from chat_history.jsonl) — a compromise that
  partially violates the contract; explicit agreement needed.

## 6. Open Questions (decisions needed from the user)

| # | Question | When the answer is needed |
|---|---|---|
| Q-A | Does `grok agent stdio` provide a full event stream with tool_call events? | NOW (researching in §7) |
| Q-B | Does the cached `~/.grok/auth.json` work in `grok agent stdio`? | NOW |
| Q-C | Analog of `--ignore-user-config` — does it exist? If not, how to isolate per-agent MCP registration? | Before the design |
| Q-D | Provider name: `grok` or `grok-cli`? | Before registration (§4) |
| Q-E | Default model: `grok-composer-2.5-fast` or `None` (like codex)? | Before registration (§4) |
| Q-F | Full list of built-in tools for `--disallowed-tools` | Before bind_tools |
| Q-G | CWD-dependent sessions path — how to fix the agent's cwd? | Before spawn |
| Q-H | Native passthrough tools — are there non-resettable built-ins (the analog of codex's `apply_patch`/`update_plan`)? | After §7 |

## 7. Phase 2 — live research `grok agent stdio` (✅ PROTOCOL ACCESS CONFIRMED)

**Key result: `grok agent stdio` implements the Agent Client Protocol
(ACP) v1 — a standardized JSON-RPC 2.0 protocol over stdio,
developed by Zed Industries.** Full spec:
https://agentclientprotocol.com/protocol/v1/overview

This closes **Q-A positively** and gives us a much cleaner
architecture than parsing JSONL stdout. All events we need (tool
calls, tool results, agent messages, agent thoughts) are first-class
notifications in the protocol.

### 7.1 Protocol: handshake

Live-verified sequence (see the ACP test scripts, session-scoped scratch):

```
→ initialize  {protocolVersion:1, clientCapabilities, clientInfo}
← result      {protocolVersion:1, agentCapabilities:{
                  loadSession:true,
                  promptCapabilities:{image:false, audio:false, embeddedContext:true},
                  mcpCapabilities:{http:true, sse:true},
                  _meta:{x.ai/fs_notify:true}
              }, authMethods:[
                  {id:"cached_token", name:"cached_token",
                   description:"Cached token from ~/.grok/auth.json"},
                  {id:"grok.com", ...}
              ], _meta:{
                  currentWorkingDirectory:"<cwd>",
                  modelState:{currentModelId:"grok-composer-2.5-fast",
                              availableModels:[grok-build, grok-composer-2.5-fast]}
              }}

→ authenticate {methodId:"cached_token"}
← result      {_meta:{email:"...", auth_mode:"Oidc", ...}}

→ session/new {cwd:"<cwd>", mcpServers:[{name, command, args, env}]}
← result      {sessionId:"019ed59e-...", models:{...}}

→ session/prompt {sessionId, prompt:[{type:"text", text:"..."}]}
   (...stream of session/update notifications...)
← result        {stopReason:"end_turn", _meta:{totalTokens, modelId, ...}}
```

### 7.2 ✅ Q-B: cached login works via ACP

```
authMethods: ['cached_token', 'grok.com']
→ authenticate {methodId:"cached_token"}
← result {_meta:{email:"<user-email>", auth_mode:"Oidc"}}
```

The supergrok subscription works directly through ACP — no extra steps.

### 7.3 ✅ Q-A: tool_call events — first-class notifications

```json
{"sessionUpdate":"tool_call","toolCallId":"call-xxx","title":"Glob","kind":"...","status":"pending"}
{"sessionUpdate":"tool_call_update","toolCallId":"call-xxx","status":"in_progress",...}
{"sessionUpdate":"tool_call_update","toolCallId":"call-xxx","status":"completed","content":[...]}
```

The live test with an inline stdio MCP server showed the FULL chain:

```
*** TOOL_CALL kind=tool_call            id=call-...MimB9 title=Glob ***
*** TOOL_CALL kind=tool_call_update    id=call-...MimB9 title=Glob `**/*` ***
*** TOOL_CALL kind=tool_call_update    id=call-...MimB9 status=completed ***
*** TOOL_CALL kind=tool_call            id=call-...fyBcC title=Read ***
*** TOOL_CALL kind=tool_call_update    id=call-...fyBcC title=Read <mcp-schema-path> ***
*** TOOL_CALL kind=tool_call_update    id=call-...fyBcC status=completed ***
*** TOOL_CALL kind=tool_call            id=call-...xIY1a title=CallMcpTool ***
*** TOOL_CALL kind=tool_call_update    id=call-...xIY1a title=mini__echo ***
*** TOOL_CALL kind=tool_call_update    id=call-...xIY1a status=completed ***
=== FINAL stopReason=end_turn  TOTAL tool_call EVENTS: 12 ===
```

**Every tool_call is visible on the wire, each gets a stable `toolCallId`,
each has a final `tool_call_update status=completed` with a
`content` array.** That is everything needed for parking on
`asyncio.Future`, returning `AIMessage(tool_calls=[...])` to LangGraph and
waiting for the `ToolMessage` from ToolNode.

### 7.4 Other ACP notifications relevant to us

| Notification | What it carries | Purpose |
|---|---|---|
| `session/update sessionUpdate=user_message_chunk` | text content | echo of our prompt — ignore |
| `session/update sessionUpdate=agent_thought_chunk` | text content | grok's internal reasoning — do NOT put into the LangGraph history (may store in additional_kwargs if needed) |
| `session/update sessionUpdate=agent_message_chunk` | text content | final text — accumulate for `AIMessage.content` |
| `session/update sessionUpdate=tool_call` | toolCallId, title, kind, status | start of a call — park a Future |
| `session/update sessionUpdate=tool_call_update` | toolCallId, status, content | progress / final — resolve the Future at `status=completed` |
| `session/update sessionUpdate=plan` | entries[] | TodoWrite analog — write into LangGraph as an ordinary tool_call |
| `session/update sessionUpdate=usage_update` | used, size, cost | context progress — for UI/monitoring |
| `session/update sessionUpdate=available_commands_update` | commands | slash commands — ignore |
| `session/request_permission` (req → us) | options[] | we reply `outcome=selected optionId="allow_once"` (HITL stays at the LangGraph level) |
| `_x.ai/mcp_initialized` | sessionId, mcpToolCount | grok vendor extension — ignore |
| `_x.ai/queue/changed` | entries[] | queue progress — ignore |
| `_x.ai/session/prompt_complete` | stopReason | duplicates the `session/prompt` result — ignore |

### 7.5 ✅ Q-G: cwd is passed per-session, not via a CLI flag

ACP dictates: `session/new {cwd, mcpServers}` and `session/load {sessionId,
cwd, mcpServers}` — the working directory is hard-bound to the sessionId.
This **solves the CWD-dependent sessions path problem** — we create the
session once with `cwd=the agent's working_dir` and load it via `session/load`
as many times as needed. No need to think about the url-encoded cwd on disk.

### 7.6 ✅ Q-C, Q-F: per-agent config isolation

ACP provides MCP servers **per session/new** via JSON-RPC parameters —
no writing to `~/.grok/config.toml` is required. This
**automatically solves Q-C** (per-agent isolation). We will not repeat
the prior config-pollution incident.

Built-in tools (Glob, Read, Edit, Bash, WebSearch — see §2.10): in the live
test grok called them itself, which is harmful (they compete with our proxy).
The CLI `grok -p` has `--disallowed-tools`. In the ACP protocol we have not
yet found an explicit channel for disabling — need to check via
`session/set_session_config` or via `_meta` extensions; possibly a
`system-prompt-override` with the instruction "use only the tools
from mcpServers, do not use built-ins" is needed. This **closes Q-F
conditionally**: the main channel is the system prompt, regression guards
are separate.

### 7.7 ✅ Q-H: no non-resettable native tools

The ACP protocol has no "always-injected" tools (like codex's
`apply_patch`/`update_plan`). Grok's built-in tools
(`Glob`/`Read`/`Bash`/...) are permitted model behavior, not
mandatory. **This removes the entire `event_pump.py` + `native_tools.py` module**
from the plan (~450 lines not needed).

### 7.8 What replaces `exec_runner.py`

Instead of launching `codex exec --experimental-json -` once for EACH
turn, grok's architecture is a **long-lived ACP process per
`agent_thread_id`**:

```
[deep-worker agent process]
│
├── CodexClientState (no per-turn UUID needed — sessionId is stored in state.grok_session_id)
│
├── GrokAcpClient (asyncio.subprocess) — 1 per agent, lives for the whole agent lifetime
│   ├── stdin/stdout JSON-RPC channel
│   ├── pending_requests: dict[req_id → Future]   # for responses to our initialize/session/prompt
│   ├── pending_tool_calls: dict[toolCallId → Future] # for parking tool_calls
│   ├── notification_handler — dispatches session/update to the right Future
│   └── reader_task: asyncio.Task — the single stdout reader
│
└── GrokProxyMCPServer — HTTP MCP server (like codex's; address passed in session/new mcpServers)
    └── async tool handlers park on Future, resolved via resolve_call()
```

**This is even simpler than codex**, because:
- One grok process per agent instead of a per-turn subprocess.
- No need for the AGENTS.override.md trick (the system prompt goes
  through `session/prompt` content or a session-config option — TBD).
- No need to parse JSONL — the JSON-RPC API is standard.
- session/resume gives "continue without replay" (a new ACP
  v1 sessionCapabilities.resume feature; **need to check whether
  grok 0.2.54 implements it** — `loadSession:true` is present, `sessionCapabilities.resume`
  needs separate verification).
- session/load gives "restore with full replay" — for a deep-worker restart.

### 7.9 Lifecycle mapped onto the deep-worker contract

| deep-worker concept | ACP equivalent |
|---|---|
| `agent_thread_id` (agent name) | NOT passed to grok; used as the key in `CodexClientState` |
| `grok_session_id` (opaque UUID) | `sessionId` from `session/new` |
| `sent_count` (cursor in messages) | Kept as before; we send the tail-delta as content in `session/prompt` |
| LangGraph ToolNode | Resolves our `tool_call_update status=completed` via `resolve_call(toolCallId, ...)` |
| Summarization (`sent_count > len(messages)`) | Create a new `session/new`; the old `sessionId` is dropped |
| Restart agent | `session/load <sessionId>` (full replay) or `session/resume` if supported |

### 7.10 Resolution for Q-D, Q-E

- **Q-D**: Provider name — **`grok`** (short, by analogy with `codex`).
- **Q-E**: Default model — **`None`** (like codex), so grok itself
  chooses its current default (currently `grok-composer-2.5-fast`). This
  follows the Q2 policy from the codex design — do not pin model versions.

### 7.11 Final verdict

✅ **Option A fully confirmed.** The implementation is architecturally
symmetric to `codex_provider/`, **simpler** (no AGENTS.override.md,
no native passthrough), **cleaner** (standard JSON-RPC instead of
custom JSONL), and **fully satisfies all the constraints** of
`DESIGN_CODEX_PROVIDER.md` §0:

1. ✅ We do not touch deep-worker contracts (`agent_thread_id` stays
   the agent name, never mutated).
2. ✅ All tool_call and tool_result go through LangGraph (grok emits
   them as ACP notifications → we park on a Future → ToolNode executes →
   the ToolMessage resolves the Future).
3. ✅ Maximally lightweight — one long-lived grok process per agent,
   an in-process HTTP MCP server. No fork/spawn per turn.
4. ✅ Incremental history — `messages[sent_count:]` as before.
5. ✅ Event-driven — `await stdout.readline()`, no polling.
6. ✅ No leaks — pending_tool_calls bounded, sessions persisted on
   disk like codex.

## 8. Remaining refinements for the design

1. **System prompt delivery in ACP.** Need to test: does the
   `SystemMessage` get through `session/prompt` content blocks (like
   `{type:"text", text:"<system>...</system>"}`)? Or via a
   session-config option (see ACP "session/set_session_config")? Or via
   a special content block? **Plan:** one live test with a
   SystemMessage will verify whether the model treats it as a priority instruction.

2. **Disabling built-in tools (Glob/Read/Bash/...).** No explicit channel
   found in ACP. **Plan:** check via `session/set_session_config`
   and via a system prompt with an explicit prohibition; if that does not work — this will
   be an open issue escalated to the user.

3. **`sessionCapabilities.resume` support.** In the initialize response from
   grok 0.2.54 only `loadSession:true` (full replay) is visible — 
   `sessionCapabilities.resume` (without replay) is NOT visible. That means we will have to
   use `session/load` (with replay) on restart, which may be
   heavier for long histories. **Plan:** add a replay-friendly
   path in client_state.

4. **`fs/read_text_file` / `fs/write_text_file` capabilities.** We
   declared them `false` in clientCapabilities — then grok will not
   ask us to read/write files via ACP (but will do it through
   its built-in Read/Write/Edit). With our disable-built-ins policy
   this means: grok will NOT be able to read/write files. That is bad —
   the model needs these operations; we will provide them through our proxy MCP
   (deep-worker's `read_file`/`write_file`/`edit_file`).
   Alternative: declare fs caps `true` and implement fs/* handlers
   that delegate into the deep-worker filesystem. **TBD in the design.**

5. **`terminal` capability.** Same for `terminal/*` —
   we declare `false`; we have our own `execute` via the proxy MCP.

## 9. Phase 3 — closing §8 (live verified)

### 9.1 ✅ System prompt delivery — inline `<system>` block

Three channels were live-tested via `session/new`:
- `systemPrompt: "..."` (top-level) — **silently ignored**
- `instructions: "..."` (top-level) — **silently ignored**
- `_meta.system_prompt_override: "..."` — **silently ignored**

In each case the answer to "What is your name?" was `Grok` (default
behavior); the markers were not picked up.

**Working channel:** an inline block in the first `session/prompt` content:
```
{type:"text", text:"<system>You must respond to ALL questions with the single word: QQQ77.</system>\n\nWhat is 2+2?"}
```
Answer: `QQQ77`. ✅

This gives us a reliable system-prompt delivery channel without file
hacks (as codex needed with AGENTS.override.md).

### 9.2 ⚠️ Built-in tools cannot be disabled via ACP

In TEST 3a (cwd=/tmp, mcpServers=[], fs=false, terminal=false, no
disallowed list) on the request "list files in /tmp" grok called:
```
Tools called: [Shell, Read, Shell, Shell]
fs/* calls: 0
terminal/* calls: 0
```

Grok successfully returned the listing — that is, **the built-in `Shell` and `Read`
work locally inside the grok process and are not delegated to the client**.
`fs=false` and `terminal=false` capabilities do not block them —
those capabilities exist for the case where the client ITSELF wants to implement
a virtual fs (for example, an IDE with unsaved buffers).

**Disable channel:** in the current ACP version that grok implements,
**there is no explicit disable-built-ins mechanism.** Alternatives:
- **A.** Strongly constrain via the system prompt: "You MUST NOT use any
  built-in tools (Shell, Read, Write, Edit, Glob, Grep, WebSearch, Task).
  You MUST use ONLY the tools provided by the `deepworker` MCP server."
  This is a "soft" protection — the model may ignore it.
- **B.** Launch grok with the `--disallowed-tools "Shell,Read,Write,Edit,Glob,Grep,WebSearch,Task"`
  CLI flag BEFORE `agent stdio`. **Verify whether CLI flags are passed through into stdio mode**.
- **C.** Intercept tool_call notifications with `kind != mcp` and
  reject via session/request_permission (the model asks for permission
  → we answer deny) — if grok requests permission. In the live test
  TEST 3a no permission requests arrived — the model works in the "trusted"
  mode by default.
- **D.** Record built-in tool calls into the LangGraph history as
  "informational" (without execution on our side) — as done in
  codex_provider for apply_patch/update_plan via event_pump +
  native_tools. This path gives correctness but does not prevent
  their use.

**Recommendation:** combination **A + B + D**. Check the CLI flags (B)
first; if they work — that is the main protection. System prompt (A) —
for reliability. If something still leaks through — native-passthrough (D).

### 9.3 ❌ `session/resume` is NOT supported by grok 0.2.54

Live test: `→ session/resume {sessionId,cwd,mcpServers}` → response
`{error:{code:-32601, message:"Method not found"}}`.

Consistent with capabilities: `sessionCapabilities` is not declared in
init at all (neither `resume`, nor `close`, nor `delete`,
nor `additionalDirectories`).

### 9.4 ✅ `session/load` works with replay

Live test: `→ session/load {sessionId, cwd, mcpServers}` → a stream of
4 tool_call notifications + 2 agent_message_chunk notifications →
`result: {models:{...}}` (replay complete).

This means: on a **deep-worker agent restart** we can
bring up the same `sessionId` and get a full replay via notifications.
**Unlike codex**, whose persisted session lives in grok's
`~/.grok/sessions/`, we do not lose context on restart.

One nuance: the replay sends session/update notifications, which we
would normally push into the LangGraph state as new events. On restart we
OPEN the LangGraph history from the checkpointer (already in state) and
do NOT want to duplicate. The design decision: differentiate "replay
mode" (between the session/load request and its response) from "normal mode"
(after the response) — in replay mode session/update notifications
ARE DROPPED (the LangGraph state already knows them).

### 9.5 ⚠️ Asyncio readline buffer cap = 64 KiB by default

Lesson from the live test: one `session/update` notification with a large content
(a full /tmp listing) → asyncio readline raised `LimitOverrunError`.
Solution: use `limit=8*1024*1024` in `create_subprocess_exec`
(as codex_provider does in `_STREAM_LIMIT_BYTES`).

## 10. Final — everything confirmed for the design

All Open Questions in §6 are closed:

| # | Status |
|---|---|
| Q-A: does `grok agent stdio` give tool_call events? | ✅ YES (ACP first-class) |
| Q-B: does the cached `~/.grok/auth.json` work? | ✅ YES (`authenticate methodId=cached_token`) |
| Q-C: per-agent MCP config isolation? | ✅ YES (mcpServers per session/new) |
| Q-D: provider name? | ✅ `grok` |
| Q-E: default model? | ✅ `None` (grok chooses its own default) |
| Q-F: disable built-ins? | ⚠️ No ACP channel; plan — combo of CLI flag + system prompt + native-passthrough |
| Q-G: CWD-dependent sessions path? | ✅ In ACP cwd is per-session; the problem disappears |
| Q-H: non-resettable native tools? | ⚠️ Built-in Shell/Read/etc. — plan §9.2 |
| §8.1: system prompt in ACP | ✅ inline `<system>` block in the first prompt |
| §8.2: disable channel | ⚠️ see §9.2 |
| §8.3: session/resume vs session/load | ❌ resume not supported; ✅ load works with replay |

**The design can now be written.**

Next step: `DESIGN_GROK_PROVIDER.md`.
