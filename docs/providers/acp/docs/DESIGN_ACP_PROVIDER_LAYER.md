# Design: deep-worker ACP provider layer (Agent Client Protocol)

**Status:** 📐 DESIGN (rev 4, incorporates the 3rd review)
**Date:** 2026-08-15
**Companion:** `docs/providers/grok_provider/docs/RESEARCH_GROK_PROVIDER.md` —
  a live ACP reference (grok), `docs/providers/grok_provider/docs/DESIGN_GROK_PROVIDER.md`
**Mirror of:** `docs/providers/grok_provider/docs/DESIGN_GROK_PROVIDER.md`,
  `docs/providers/codex_provider/docs/DESIGN_CODEX_PROVIDER.md`

---

## 0. Hard constraints from the user (non-negotiable)

The ideas and motivation come from the project owner; the invariants are carried
over from the codex/grok designs unchanged:

1. **A common (abstract) layer for connecting providers via ACP.**
   Providers that have no direct API (unlike deepseek) — Claude Code,
   Codex CLI, Gemini CLI, etc. — connect through a single ACP bridge.
2. **In particular — connect claude-code-sdk via ACP**, so as **NOT to pull
   a non-standard (forked) version of the Claude Agent SDK into the sources**
   (it is currently mandatory — see §3).
3. **Do not change deep-worker contracts.** `config["configurable"]["thread_id"]`
   = the deep-worker agent name; never mutated.
4. **All messages — including every tool call and tool result — must
   pass through LangGraph** and end up in `result["messages"]`. The adapter
   does NOT execute deep-worker tools itself and does NOT return the final text
   bypassing LangGraph (normative sequence — §4.4).
5. **As lightweight as possible.** One long-lived **ACP adapter process
   tree** per agent (NOT per turn), event-driven, no polling. "Process tree"
   — because the adapter (e.g. `claude-agent-acp`) may spawn
   child processes (Claude Code); terminate/wait/pipe-close of the
   whole group without orphan children is guaranteed (§5.3).
6. **No new threads.** All background dispatchers are asyncio tasks only
   (read-loop, drain pump), no `threading.Thread` without the user's
   explicit approval (project policy).
7. **Access permissions.** All files/directories created by the feature are free of
   777/666: session persist files `0600`, directories `0700`.
8. **No leaks.** All long-lived state is bounded; call-id maps/futures and
   notification buffers have explicit cleanup on cancel/error paths (§4.4,
   §4.5).
9. **Publishability.** The goal is a public version of deep-worker with no
   dependency on a private SDK fork. Reproducibility: the adapter is a
   pinned version, no mutable `latest` at startup (§5.2).

---

## 1. Scope v1 (the executable scope of this feature)

Fixed after review (rev 1 left a contradiction between §4.1 and Q7):

- **The common layer `deep_worker/providers/acp/` is created** (`AcpClient` +
  `AcpChatModel` + `acp_types` + `session_state` + `limit_errors` +
  in-process proxy MCP server) — as stated in §4.1.
- **Claude is the first new profile** on top of the layer: the provider
  `claude-code` (name — per the owner's decision, §8 Q1).
- **Grok stays on the current code** (`grok_provider/`) until a separate
  parity migration (out of scope for v1). The layer and the grok code do not share mutable
  state; grok serves as the behavioral reference.
- Out of scope for v1: codex on ACP, gemini, generalizing grok.

---

## 1. Motivation

### 1.1 Why ACP

The **Agent Client Protocol (ACP)** is a JSON-RPC 2.0 protocol (spec:
https://agentclientprotocol.com/protocol/v1/overview) for "agent →
agent" communication: a client process (our deep-worker) launches an agent subprocess
(Claude Code, Codex CLI, Grok CLI, Gemini CLI) and talks to it as to an
agent — not as to an API. ACP is becoming ever more popular (adapters from Zed,
the Agent Client Protocol org and the community for Claude Code, Codex, Gemini).

For deep-worker this means: **one common way to connect all
CLI providers** that have no (or no suitable) direct API:

```
deep-worker ──ACP (JSON-RPC over stdio)──► agent subprocess
                                             (claude-code-sdk / codex / grok / gemini …)
```

Bonus: many such tools bill **via a subscription** (Claude
Pro/Max, supergrok, Codex subscription) rather than via an API key.

### 1.2 Why now — claude-code-sdk

The current claude-code-sdk provider (`deep_worker/providers/claude_code_sdk_provider/`)
is **heavyweight and unpublishable**:

- `chat_model.py` (1787 lines) and `client_manager.py` (843 lines) insert the
  `claude_sdk/src` path into `sys.path` — a **local fork** of the Claude Agent SDK
  (a private fork of `anthropics/claude-agent-sdk-python`).
- The fork contains private fixes: `edfa02b` (ContextVar token-based reset),
  `2cb624a` (propagate the Claude tool-use ID via ContextVar into the MCP handler),
  `57a172b` (close the stdout pipe FD before `process.wait()`).
- The fork marker is the private module `claude_agent_sdk._mcp_context`
  (`tools.py::_sdk_fork_status()`); `WebSearchDelegationTool`
  and `backends.py::_current_tool_use_id()` (ContextVar tool-use ID) depend on the fork.
- The public version of deep-worker was recently released **practically without the
  claude-code-sdk provider** — because of this dependency.

**Goal:** connect claude-code-sdk via ACP, removing the mandatory dependency
on the forked SDK. The ACP adapter itself (`claude-agent-acp`, see §5) is built
on top of the official Claude Agent SDK and lives **outside** the deep-worker sources.

### 1.3 Key external fact: subscription usage is officially allowed

Research (June 2026): Anthropic officially allows using the
Claude Agent SDK and third-party applications via a Claude Pro/Max subscription —
the scheme `your agent → Claude Agent SDK → Claude Code → subscription`
officially works and consumes the subscription limit (support.claude.com article
"Use the Claude Agent SDK with your Claude plan"). The planned
split of the Agent SDK allowance ($20/$100/$200) was **suspended by
Anthropic on June 15, 2026** — there is no separate credit. Also `claude -p`
is officially counted as subscription usage.

This removes the main question "can claude-code-sdk be used at all without an
API key" — **it can, via a subscription**.

---

## 2. ACP ≠ MCP — an important distinction

| Protocol | Direction | Who is the client | Who is the server |
|---|---|---|---|
| **MCP** | tool integration | Claude Code (client) | your tools |
| **ACP** | agent integration | your agent (deep-worker) | Claude Code / Codex / Grok CLI |

The current deep-worker ↔ Claude Code integration is MCP in the reverse direction
from the usual: deep-worker, via `proxy_mcp_server.py`, **pretends to be an MCP
server** to Claude Code so that Claude Code can call deep-worker's own
tools (`CallMcpTool`). This works, but is tied to the SDK fork and to the
internals of ClaudeSDKClient.

ACP solves the same task "from the outside": deep-worker is the ACP client, Claude Code is the
ACP agent (through the `claude-agent-acp` adapter). No fork in the sources.

---

## 3. Current state of providers in deep-worker

| Provider | Transport | Billing | Notes |
|---|---|---|---|
| `deepseek` | HTTP API | API key | default, direct API |
| `gemini` / `openai` / `anthropic` / `vertexai` | HTTP API | API key | direct APIs |
| `codex` | `codex exec --experimental-json` (JSONL) | subscription/tokens | own client, NOT ACP |
| `grok` | **ACP v1** (`grok agent stdio`) | supergrok subscription | **already a full ACP client** |
| `claude-code-sdk` | ClaudeSDKClient (fork) | subscription | heavyweight, unpublishable |

### 3.1 Grok — a ready ACP reference

`deep_worker/providers/grok_provider/` is the best template for the abstract
layer:

- `acp_client.py` (558 lines) — `GrokAcpClient`: a long-lived stdio JSON-RPC
  channel (one per process), `_read_loop` dispatches replies/requests/
  notifications; `initialize → authenticate(cached_token) → session/new →
  session/prompt`; `session/load` (restart resume), `session_cancel`,
  `wait_session_ready`, `register_sink`; `_READ_LIMIT = 8*1024*1024`
  is mandatory (grok sends very large tool-result lines); `raise_for_error_reply`
  reads the **entire** error object, http_status 402 → `GrokUsageLimitError` →
  failover to deepseek.
- `acp_types.py` (144 lines) — a taxonomy of session/update kinds
  (agent_message_chunk, agent_thought_chunk, tool_call, tool_call_update,
  plan, usage_update, user_message_chunk) + vendor `_x.ai/*` noise is ignored.
- `chat_model.py` (1523 lines) — `GrokChatModel(BaseChatModel)`; **two tool
  channels**: Channel A (deep-worker MCP tools via an in-process proxy MCP
  server, `CallMcpTool`) and Channel B (grok built-ins via
  `builtin_tool_passthrough.py`); `_consume_turn`, `_finalize_response`,
  the background pump `_drain_pump`; persists `(grok_session_id, sent_count)` in
  `~/.deep_worker/<agent>/grok_session_state.json`; summarization resets the
  session but does NOT touch `agent_thread_id`.

**Conclusion:** the abstract ACP layer is a generalization of `GrokAcpClient` + the patterns of
`GrokChatModel`; grok remains the first (already existing) instance of the layer.

---

## 4. Proposed architecture

### 4.1 The common ACP layer

A new (lightweight) package `deep_worker/providers/acp/` — or a generalization of the
existing grok code with a common core extracted. Composition:

```
deep_worker/providers/acp/
├── __init__.py          # lazy root (PEP 562), like the others
├── acp_client.py        # AcpClient: JSON-RPC stdio channel, read-loop, session mgmt
│                        #   ← generalization of GrokAcpClient (method=path to binary + args)
├── acp_types.py         # session/update kinds + vendor noise (generalization of the grok version)
├── session_state.py     # persists (session_id, sent_count) to disk
├── limit_errors.py      # usage-limit detection → failover (generalization of the grok version)
└── chat_model.py        # AcpChatModel(BaseChatModel) — a common BaseChatModel over AcpClient
```

Key decisions:

1. **One long-lived ACP adapter process tree per agent** (as with grok).
   The launch command is a provider parameter: `grok agent stdio`, the
   claude-code-sdk adapter (installed, pinned — §5.2), etc. The lifecycle of the whole process
   group (adapter → possible child processes) — terminate/wait/pipe-close
   without orphan children (§5.3).
2. **Tools — via an in-process proxy MCP server** (the Channel A pattern
   from grok). The model dictates a `CallMcpTool` call, and the result is returned as a
   tool result into the ACP session. **The adapter does NOT execute deep-worker tools
   itself** — only the LangGraph ToolNode executes the bound tool (normative
   sequence — §4.4).
3. **`session/update` notifications** are the only source of events
   (agent chunks, tool calls, plan, usage). Vendor-specific noise
   (`_x.ai/*` for grok, `_anthropic/*` for claude-code-sdk) is filtered in
   `acp_types.py`.
4. **Session persist** — to disk, **provider-scoped, with a capability
   fingerprint**: path `~/.deep_worker/<agent>/acp_session_state.json`
   (agent-scoped), atomic replace (tmp+rename), file `0600`/directory
   `0700`, malformed/backward-compatible recovery (a corrupted file →
   fresh session). The file MUST contain: `provider`/profile id, `session_id`,
   `sent_count`, and a fingerprint `{adapter, adapter_version, session-capability}`
   (what the adapter declared in `initialize`/`session/new`). A fingerprint
   **mismatch** (different profile / updated pinned adapter / changed
   capabilities) → **fresh session** (session/new), and the file is rewritten
   atomically — this closes the stale-session risk (a test is required, §10).
   `session/load` only if the adapter declared support, otherwise a fresh
   session; on summarization the session is reset, and `agent_thread_id`
   (LangGraph) is NOT mutated (§6.3). `sent_count` is a monotonic increment
   (not a collection, "bounded" does not apply; reset only together with
   session_id — §4.5).
5. **Failover to deepseek** through the common `limit_errors.py` (usage-limit →
   `AcpUsageLimitError`), see §7 — including class-name/fatal registries,
   TaskManager propagation and restore-on-idle.
6. **Permission requests** (`session/request_permission`) — a **fail-closed
   policy** (§4.6): auto-approve ONLY under an explicit operator auto-approve semantic
   (`-sp` / an explicit flag); otherwise deny/passthrough via the
   existing approval channel; unknown permission kinds are NOT
   auto-approved.

### 4.4 Normative sequence: a tool call through LangGraph (the mandatory path)

The only permitted path for executing a deep-worker tool. It repeats the
real Grok Channel A flow — **the adapter is already inside an unfinished
`session/prompt`** and is waiting for a response from the in-process MCP handler it called:

```
1. An ACP session/update delivers a tool_call (vendor toolCallId, e.g. a uuid);
   the adapter calls the in-process MCP handler (CallMcpTool) — this happens
   INSIDE the current session/prompt, which has not yet finished.
2. The MCP handler parks a proxy Future (bounded map, TTL/timeout — §4.5) and
   ITSELF REMAINS await-pending: the adapter's original MCP request gets
   NO response (neither interim nor final) until
   proxy_mcp.resolve_call(...) is called at step 7. No "pending" response
   is sent to the adapter.
3. AcpClient correlates the ACP tool call (toolCallId → internal id); the parked
   future stays waiting for resolution.
4. AcpChatModel converts the tool_call into a LangChain
   AIMessage(tool_calls=[...]) and returns control to LangGraph
   (ainvoke finishes this step).
5. The LangGraph ToolNode (and only it) executes the surfaced, real bound
   deep-worker tool that corresponds to this tool_call — a real
   tool, a real result. The generic `CallMcpTool` is the adapter's
   transport call to the in-process proxy; it is NOT executed
   again by the ToolNode (unless the profile explicitly publishes it as a
   LangGraph tool — then a separate proven mapping is needed, §10).
6. The ToolMessage (result) lands in result["messages"] — a mandatory
   condition of the contract (hard constraint #4).
7. The next ainvoke, upon seeing the ToolMessage, calls the LOCAL
   proxy_mcp.resolve_call(...) — releasing the original MCP handler, which
   returns the result to the adapter (which receives it as the response to step 1/2).
   The existing prompt/drain pump CONTINUES; a NEW session/prompt is NOT
   created. This resolves the parked future from step 2.
```

**Critical:** do NOT create a second `session/prompt` to deliver the tool_result
— it contradicts the one-in-flight invariant (§4.5) and is not confirmed by the
ACP capability/schema. The tool result from deep-worker is returned to the adapter
ONLY via `proxy_mcp.resolve_call(...)` inside the same unfinished
prompt.

**Forbidden:** a path where the adapter itself executes a deep-worker tool and
AcpChatModel merely streams the final text (this would violate constraint #4).

**Correlation (ACP id ↔ proxy-call-id):**
- `vendor toolCallId` (from `session/update`) ↔ `proxy call id`
  (the identifier of the in-process MCP call) — an explicit bijection, stored in a
  bounded map (§4.5);
- duplicate tool_call: a second `session/update` with the same `toolCallId` while a
  future is already parked — is idempotently ignored (does not create a second future);
- lost: if the ToolNode did not call the tool (tool not found) or ainvoke
  was interrupted — the parked future is resolved with an error on TTL/cancel (§4.5);
- **a test that fails on a second `session/prompt`**: a fake subprocess
  emits a tool_call, deep-worker returns the result via resolve_call,
  and the test asserts that afterwards the adapter did NOT receive a second `session/prompt`
  (prompt counter == 1 per turn).

### 4.5 Bounded state and cleanup (no leaks — constraint #8)

ALL collections and buffers have a finite capacity; "bounded" means an explicit
capacity/overflow policy, not merely "small":

- **call-id maps / parked futures**: finite capacity (the stream of one
  active turn — no more than N concurrent tool_calls, N is a
  profile parameter, e.g. 32), with explicit cleanup:
  - success path: the future is resolved by step 7 (§4.4), the entry is removed;
  - cancel (`session_cancel` / ainvoke cancellation): all parked futures
    are cancelled, the map is cleared;
  - error path (`raise_for_error_reply`): the same parked futures
    are cancelled/removed;
  - timeout (the entry's TTL in the map, a profile parameter) — the entry is removed,
    the error is classified;
  - overflow (the map reached capacity on a new tool_call) — **deterministic
    default: reject-new** — the new tool_call is REJECTED with an explicit
    "over capacity" error (no silent drop), and the already running correlated call
    is NOT destroyed. Both the rejected and (in a profile's alternative mode)
    evicted requests get an explicit terminal error — no future
    hangs. (The evict-old alternative is allowed only by explicit profile
    configuration; it is not used in v1.)
- **notification queues / sinks**: finite capacity (e.g. 1000 messages);
  backpressure policy: when the queue fills, the turn ends with the error
  "notification backlog overflow" (fail-loud, not a silent drop); a sink lives
  only for the duration of the active turn; unregister_sink on completion/cancel/
  error; on subprocess EOF and on normal completion the queues are cleared.
- **partial chunk buffers**: aggregation of agent_message_chunk / tool_call_update
  — a byte limit (e.g. `_READ_LIMIT` per message) AND a chunk-count limit
  (e.g. 10_000 chunks) per message being assembled; exceeding →
  the error "partial buffer overflow", the buffer is reset, the turn ends
  with an error. Cleanup on EOF and normal completion.
- **always one in-flight prompt per session** (ACP invariant): a second
  session/prompt before the first completes is a programming error
  (assert/invariant), like grok's `_active_prompts`. This also rules out a
  "second prompt for the tool_result" (§4.4).
- **`sent_count`**: a monotonic increment (never rolled back) — this is NOT a
  collection and "bounded" does not apply to it; semantics: a counter of
  messages already sent into the ACP session, for tail-delta reads
  (`messages[sent_count:]`). Overflow (beyond int64 / meaningless
  accumulation) is not a problem in any practical horizon; normative rule:
  `sent_count` is reset on a new session (session/new, summarization,
  session/load fallback) — reset together with session_id, never separately.

### 4.6 Permission policy (fail-closed, security boundary)

`session/request_permission` from the claude-code-sdk adapter:

| Mode (provider config) | Behavior |
|---|---|
| `auto-approve` | approve ONLY known permission kinds from the profile's explicit allowlist; **only** when the operator has set auto-approve (explicit flag / `-sp` semantics). Unknown kinds → deny. |
| `deny` (default) | answer deny to any `request_permission`; do not fail over. |
| `passthrough` | forward to the operator via the existing approval channel (dwcli), with a timeout; on timeout — deny (fail-closed). |

The permission kind → approve/deny mapping is a table in the provider profile,
covered by unit tests. No permission kind is auto-approved by default.

### 4.2 Providers on top of the layer

| Provider | ACP adapter (subprocess) | Status |
|---|---|---|
| `grok` | native `grok agent stdio` (already exists) | ✅ works; **stays on the current code** until the parity migration (out of scope for v1) |
| `claude-code` | `@agentclientprotocol/claude-agent-acp` (Node, pinned) | 🎯 v1 goal |
| `codex` | `codex-acp` (Rust/Go adapters) — an **alternative** to the current JSONL client | ❓ out of scope for v1 |
| `gemini` | Gemini CLI / bridge | ❓ future |

Each provider is a thin profile on top of `AcpChatModel`:
`(binary_cmd, auth_method, session/update kinds, tool-call mapping,
limit detection, permission policy)`.

### 4.3 Diagram (claude-code-sdk via ACP)

```
┌────────────────────────── deep-worker agent process ──────────────────────────┐
│                                                                               │
│  LangGraph ──► AcpChatModel(claude-code)                                  │
│                     │                                                         │
│                     ▼                                                         │
│              AcpClient (common layer)                                         │
│              JSON-RPC over stdio, read-loop, sinks, persist (sid, sent_count) │
│              bounded maps: call-id ↔ parked future (cleanup §4.5)            │
│                     │                                                         │
│                     │ ACP v1 (initialize/auth/session/new/session/prompt,     │
│                     │        session/update notifications, request_permission)│
│                     ▼                                                         │
│        ┌─────────────────────────────┐                                        │
│        │  claude-agent-acp (Node)    │   ← pinned npm package, NOT in the     │
│        │  = ACP adapter on top of    │      deep-worker sources; lifecycle §5.3│
│        │    the official Agent SDK   │                                        │
│        └──────────────┬──────────────┘                                        │
│                       ▼                                                       │
│             Claude Code (Pro/Max subscription)                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

deep-worker tools (MCP) are connected as in grok: an in-process proxy MCP
server → `CallMcpTool` (Channel A). Tools are executed ONLY by the LangGraph
ToolNode (§4.4).

---

## 5. claude-code-sdk via ACP — details

### 5.1 The adapter: `claude-agent-acp`

The project `agentclientprotocol/claude-agent-acp` (2.4k★) is an ACP adapter on top of the
Claude Agent SDK; architecture `your agent → ACP (JSON-RPC) → claude-agent-acp
→ Claude Agent SDK → Claude Code → subscription`. It supports streaming,
tool calls, permission requests, sessions, terminals, slash-commands, MCP,
subagents. Package: `@agentclientprotocol/claude-agent-acp` (at the time of
research 0.68.0 (pinned, see §8 Q3); formerly `@zed-industries/claude-code`).

- Since the adapter is built on top of the Agent SDK, it falls into the category
  of "third-party apps built on the Agent SDK" that Anthropic allows with
  subscription quota (§1.3). (Historically there were OAuth/subscription problems —
  issues #517, #184 — to be verified during implementation.)

### 5.2 Installing and launching the adapter (reproducibility — constraint #9)

`npx -y @...` is **forbidden** as a runtime launch (mutable `latest`, network at
startup, non-reproducibility). Instead:

- **Install/preflight contract** (one-time, outside the hot path):
  - install into a known location: `npm install --save-exact --prefix
    ~/.deep_worker/acp/claude-agent-acp @agentclientprotocol/claude-agent-acp@<PINNED>`
    — **`--save-exact` in the normative command is mandatory** (pins the exact
    version in the installed tree's package.json); `package-lock.json` is
    kept as an artifact; trust boundary: package integrity rests on the
    npm registry + the lock file (documented);
  - preflight at provider start: the binary exists (`bin/…`), `node
    --version` ≥ the minimum, a `--version` / `initialize` smoke check is run;
  - if missing — **an actionable diagnostic**: a clear error with the
    install command, do NOT fail over (this is a config/provisioning error, see
    §7).
- **Offline**: after installation, launch does not use the network (local binary);
  verified in an integration test.
- The **pinned version** is fixed in the provider profile (a constant) and in
  the documentation; **drift detection (constraint #9, hard)**: preflight compares the
  installed version with the expected one — on **mismatch preflight FAILS** with
  an install/fix diagnostic (the exact reinstall command
  `<PINNED>`), and launch does NOT happen (an unpinned actual
  variant is not launched). A compatible version range — only if the owner
  explicitly allowed it in the provider config; by default exact match is required.

### 5.3 Process tree lifecycle (no orphan children — constraint #5)

The adapter may spawn child processes (Claude Code). The AcpClient
contract:

- **terminate**: on `aclose()`/cancel — first a soft `session_cancel`,
  then SIGTERM to the process, a short grace period, then SIGKILL;
- **wait**: `await proc.wait()` after closing the stdin/stdout/stderr pipes
  (in an order that excludes deadlock — the pattern from the grok fork fix `57a172b`);
- **process group**: launch in its own process group
  (`start_new_session=True`), terminate the whole group (os.killpg), so that no
  orphan children remain;
- **subprocess EOF/crash**: the read-loop detects EOF → the error is
  classified (subprocess crash/restart — see §7, §9); never a
  silently "dead" channel.

### 5.4 What we lose / what we gain

We gain:
- **No SDK fork in the sources** — publishability.
- A single transport shared with the other CLI providers.
- Subscription billing without an API key.

We lose / risk:
- ACP does not yet fully express Claude Code specifics: for example,
  **mid-turn message injection `/btw`** — issue #871 in claude-agent-acp.
- A dependency on Node.js on the host + an installed npm package (pinned).
- Slightly more layers than a direct SDK connection (but fewer than now: all
  the scaffolding around the fork goes away).

---

## 6. CLI integration — points (following the grok/codex pattern)

### 6.1 Mandatory integration table

| # | File | What is added |
|---|---|---|
| 1 | `deep_worker_cli/agent_config_pkg/agent_config.py:54` | `valid_providers` += `<name>` (per the owner's decision, §8 Q1) |
| 1b | `agent_config_pkg/agent_config.py:62` | skip API-key load for ACP providers |
| 2 | `cli/agent_command.py:1012` | `choices=[..., "<name>"]` |
| 3a | `config.py:1434` | `_PROVIDER_DEFAULT_MODELS["<name>"] = (..., None)` |
| 3b | `config.py:1803` | factory `elif provider == "<name>":` via `ensure_provider_importable` |
| 4 | `runtime/availability_tracker.py` | `_PROVIDER_FALLBACK_HOURS["<name>"]` (6h like claude-code-sdk) |
| 5 | `runtime/provider_chain.py` | `PROVIDER_FAILOVER_CHAIN` — insert/replace |
| 6 | `stats/token_stats.py:59` | `CUSTOM_PROVIDERS` += `<name>` |
| 7 | `hooks/stage_providers.py` + `providers_manifest.json` | the new package `deep_worker/providers/acp/` (SHA-256, `--update`) |
| 8 | `provider_paths.py` | needs no changes (common origin mechanism) |
| 9 | `model_predicate.py` (`model_can_belong_to_provider`) | a **mandatory** branch for `<name>` (otherwise the provider's models are considered foreign and are reset on restart) |
| 10 | `cli/import_claude_cli.py` / prompt hints | optional |

### 6.2 Persistence and lifecycle (normative requirements)

- **Launch/restart config preservation**: `--provider <name>` + `--model`
  are saved in the launch config; switching the provider resets the model
  (existing semantics); preferred-provider and restore-on-idle work
  as for grok/codex. Covered by unit tests (analogous to
  `test_pp_provider_switch_resets_model.py`).
- **AgentProcess contract**: working-dir, tool-metadata and prewarm are not
  broken (the provider is not imported until first use —
  capability guards as for grok/codex).
- **Session state (agent-scoped)**:
  - path `~/.deep_worker/<agent>/acp_session_state.json` (file `0600`,
    directory `0700`);
  - atomic replace (tmp + rename), malformed → fresh session
    (backward-compatible recovery), bounded `sent_count` (monotonic
    increment, never rolled back);
  - `session/load` — only if the adapter declared support; unsupported →
    fresh session (no error);
  - summarization: the session is reset (a new `session/new`), the LangGraph
    `thread_id` is NOT mutated.

### 6.3 Question: what to do with the old claude-code-sdk provider

Options:
- **(A) A new provider `claude-code` alongside the old `claude-code-sdk`**
  — a migration period, both available. The old one stays in the sources.
- **(B) Move the name `claude-code-sdk` to the ACP implementation** — remove/archive
  the old code. Cleaner for publication, but breaks existing agents
  (same name, different implementation — sessions are not compatible).
- **(C) The ACP variant under `claude-code-sdk`, the old one under `claude-code-sdk-legacy`**
  — symmetric to (A).

Recommendation: **(A)** for review and A/B comparison, with a subsequent decision
(an owner decision is needed — §8 Q1).

---

## 7. Errors, limits and failover

The common layer must preserve the grok semantics and **plug into the existing
runtime mechanisms**:

### 7.1 Error classification (mandatory integration points)

- `AcpUsageLimitError` — a **fatal/usage-limit class**: registered in the
  class-name/fatal registries (analogous to `ClaudeUsageLimitError` /
  `CodexUsageLimitError` / `GrokUsageLimitError`), so that:
  - `runtime/availability_tracker.py` records the provider as unavailable
    with a reset time (`~/.deep_worker/provider_availability.json`);
  - `runtime/provider_chain.py` performs failover along the chain;
  - TaskManager propagation delivers the error event (as for grok/codex —
    an event, not system);
  - restore-on-idle returns the agent to the preferred provider after the reset.
- **Auth / protocol / config errors do NOT fail over**: a missing/broken
  adapter, OAuth problems, protocol mismatch, a wrong config — these are
  action-required, not temporary unavailability (otherwise every launch would
  silently go to deepseek).
- **Provider-specific marker extraction**: for claude-code-sdk the marker
  `"You've hit your session limit · resets <TIME>"` (chat_model.py:60, 113)
  → `AcpUsageLimitError` + parsing of the reset time for the availability key;
  do NOT collapse different provider markers into one fallback window.
- **`_handle_invoke` / subprocess crash**: subprocess EOF/crash →
  `AcpSubprocessError` (recoverable), retry semantics as for grok; on
  repeated crashes — a notification to the operator.

### 7.2 Failover mechanics (without changing contracts)

The same mechanism as for grok/codex: `runtime/provider_chain.py` +
`availability_tracker.py` + persistence + restore-on-idle. Tests: unit
(classification, reset parsing, availability key, chain position) +
integration (a real failover scenario, see §10).

---

## 8. Open Questions (a user decision is needed)

- **Q1. Provider name:** **(Decided by the owner 2026-08-15: option A)**
  a new provider `claude-code` alongside the old `claude-code-sdk` (§6.3).
  Later, when the ACP implementation demonstrates stable operation, the name
  will be renamed to `claude-code-sdk` — BUT that is out of scope for the current work.
- **Q2. Python SDK vs own client:** use `agent-client-protocol`
  (PyPI, Pydantic models `acp.schema`, asyncio transports) or an own lightweight
  JSON-RPC client, as in grok? **(Decided: own — following the grok model, already
  proven in production, no new dependency.)**
- **Q3. The claude-code-sdk adapter:** **(Decided by the owner 2026-08-15:**
  `@agentclientprotocol/claude-agent-acp@0.68.0` pinned; Node ≥ 22 — on the host
  v22.22.2 ✓; Claude auth is present in `~/.claude.json`; live verification of
  OAuth/subscription — at the integration-test stage.)
- **Q4. Web-search tool:** `WebSearchDelegationTool` and
  `backends.py::_current_tool_use_id()` (ContextVar tool-use ID) depend on the
  SDK fork. **(Decided: the tool-use ID comes from the ACP tool_call id; web_search as
  an ordinary deep-worker MCP tool — independent of the fork.)**
- **Q5. Permission requests:** **(Decided by the owner 2026-08-15:
  auto-approve** — §4.6, the `auto-approve` mode with an allowlist of known kinds,
  ONLY with an explicit operator flag; default deny.)
- **Q6. Codex:** should the current codex JSONL client be moved to ACP
  (`codex-acp`)? (Out of scope for v1, but the layer allows it.) — **does not
  need a decision now.**
- **Q7. Generalization vs copy.** **(Decided: §1 Scope v1 — the common layer
  is created now, Claude is the first profile, grok stays until the
  parity migration.)**

---

## 9. Risks

1. **The claude-code-sdk ACP adapter is unstable** (OAuth, subscription edge cases) —
   mitigated: a live test before implementation, the old provider remains
   (option A).
2. **Loss of Claude Code specifics** via ACP (e.g. `/btw`, some
   slash commands) — document the discrepancies.
3. **Node.js runtime + npm package** as a host dependency — pinned version,
   preflight, documentation; supply-chain/runtime install risk.
4. **Change of Anthropic's billing scheme** (the Agent SDK allowance will
   perhaps return) — follow the official announcements.
5. **Huge tool-result lines** (as in grok) — keep `_READ_LIMIT` =
   `8*1024*1024`; BUT `_READ_LIMIT` by itself does not bound the queues or
   partial buffers — bounded state (§4.5) is mandatory.
6. **Protocol/capability negotiation drift**: the adapter may change its
   declared capabilities/versions between releases — a preflight version check
   (mismatch → **fail** with an install diagnostic, §5.2) and a check of the session
   capability fingerprint (mismatch → fresh session, §4.1.4);
   warn on other incompatibilities.
7. **Duplicate/lost tool-call correlation** — correlation by toolCallId with a
   bounded map and timeout (§4.5); tests for duplicate/lost.
8. **Subprocess crash/EOF/restart** — classification (§7.1), repeated
   crashes → operator.
9. **Stale/incompatible session/load** — session/load only when the
   adapter supports it, on incompatibility — a fresh session without an error (§6.2).
10. **Permission escalation** — fail-closed policy (§4.6), an allowlist of
    known kinds, tests.
11. **Unbounded notification/partial buffers** — bounded state (§4.5).

---

## 10. Testing

### 10.1 Unit tests (no external processes)

- `acp_types`: parsing of session/update kinds, filtering of vendor noise
  (`_x.ai/*`, `_anthropic/*`), partial chunks.
- `limit_errors`: extraction of provider-specific usage-limit markers
  (claude-code-sdk `"You've hit your session limit · resets <TIME>"`), parsing of the
  reset time, classification of auth/protocol/config (NOT failover) vs
  usage-limit (failover).
- `session_state`: atomic replace, `0600/0700` permissions, malformed file →
  fresh session, monotonic `sent_count`, session/load unsupported fallback.
- `permission policy` (§4.6): kind → approve/deny mapping, unknown kind →
  deny, auto-approve only with an explicit flag.
- Call-id correlation: duplicate/lost/timeout → bounded cleanup (§4.5).
- CLI integration: `valid_providers`, `choices`, the `create_model` factory,
  `model_can_belong_to_provider`, provider switch resets the model,
  preferred-provider/restore-on-idle (analogous to the grok/codex tests).
- capability guards: the provider is not imported until first use.

### 10.2 Integration tests (no mocks)

- **Fake ACP subprocess** (Python, speaks ACP v1 JSON-RPC over stdio):
  brought up in the test, runs the handshake `initialize → authenticate →
  session/new → session/prompt → session/update → tool_call`; verifies the
  normative sequence §4.4: `AIMessage(tool_calls)` → ToolNode executes →
  `ToolMessage` in `result["messages"]` → the next ainvoke resolves the
  correlation. This proves contract #4 without network and without real CLIs.
- **Fail-on-second-prompt** (§4.4): the fake subprocess counts the received
  `session/prompt`s; after the tool cycle (tool_call → resolve_call), assert that
  there was EXACTLY ONE `session/prompt` per turn — the test fails if the implementation
  sends a second prompt with the tool_result. Plus an invariant test of one-in-flight
  (§4.5): an attempt at a second prompt while the first is active → a programming
  error.
- **Bounded overflow** (§4.5): filling the notification queue / partial
  buffer (chunk-count and byte limits) → a fail-loud error, not a silent drop;
  call-id map overflow → **reject-new**: the new tool_call is rejected with an
  explicit "over capacity" error, the running correlated call is not
  destroyed; both affected futures get a terminal error and do not hang.
- **Session fingerprint mismatch** (§4.1.4): a session_state.json with a foreign
  `provider`/a different adapter version/different capabilities → fresh session
  (a new session/new), the file rewritten atomically; the stale session id is not
  reused.
- **Version drift fail** (§5.2): the installed adapter version != PINNED →
  preflight FAILS with an install diagnostic, the subprocess is NOT launched
  (an unpinned variant is not executed); a compatible range — only
  when explicitly allowed in the config.
- **Real CLI smoke** (if available on the host): `grok agent stdio` — a real
  ACP subprocess; the claude-code-sdk adapter — only if the pinned
  npm package is installed and there is a subscription (otherwise the test is skipped with a diagnostic).
- **Failover scenario**: the fake subprocess answers with a usage-limit marker →
  `AcpUsageLimitError` → the availability tracker records the reset →
  provider_chain switches → restore-on-idle.
- **Lifecycle**: launch/terminate of the process tree without orphan children (checking
  killpg and the pipe-close order), EOF/crash → classification.
