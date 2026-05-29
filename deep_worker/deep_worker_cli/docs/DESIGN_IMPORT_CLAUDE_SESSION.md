# Design: import a Claude CLI session into a deep_worker agent (`--import-claude`)

Status: proposed. Branch: `dev`. Author bootstrap: 2026-05-24.

## 1. Goal & guiding principle

Add `deep-worker <name> --import-claude <uuid|path>`: take a real Claude Code
CLI session transcript (a `.jsonl` produced by Claude run the *ordinary* way,
not by deep_worker) and write it as the **conversation history** (the
agent's memory) of a deep_worker agent named `<name>`, then start the agent.

This is the **inverse** of `claude_code_provider/session_synthesis.py`, which
builds a Claude transcript *from* a LangGraph history. Here we parse a Claude
transcript and build the LangGraph message history.

### Guiding principle (per the user)

History is **a list of messages the model looks at**, not checkpoints. It is
the agent's *memory of what it did before* — it is **NOT meant to be replayed**
(no tool re-execution). The import target is therefore singular and simple:

> **the agent's per-agent SQLite database must end up containing these
> messages** (so `deep-worker --history <name>` shows them and the agent sees
> them on its next turn).

Which provider later reads those messages is **secondary**. The messages are a
provider-neutral `list[BaseMessage]`. On its next start the agent will have
*its own* (possibly different) tool set — it learns the current tools from its
system prompt; the old tool calls in history are just a record of the past.
Tools partially differing from before is acceptable and expected.

So the importer's core job is exactly one thing:

> **Claude transcript → `list[BaseMessage]` → the agent's SQLite history.**

This happens **always, for every provider** — `--import-claude` produces the
agent's SQLite DB with the messages regardless of `--provider`.

### claude-code: no special-casing (and why a tempting "optimization" is wrong)

Import writes ONLY the SQLite history, for every provider including
claude-code. It is tempting to *also* hand claude-code the imported `.jsonl`
(via a `session_uuid` hint) so it `resume`s the real file — but that **loses the
agent's tools** and is forbidden. The imported transcript is an *external*
`claude` session whose restored tool/permission config has no deepworker MCP
tools; resuming it yields an agent with ~1 tool. Under claude-code we instead
start a fresh session and let the provider synthesize *our own* transcript from
the LangGraph history — the same path used for deepseek/codex continuation.
Full rationale and the related incident: **§6**.

## 2. The Claude transcript format (input)

Inspected against REAL sessions under `~/.claude/projects/<slug>/<uuid>.jsonl`
(CLI 2.1.150), not just our own writer — and real transcripts are richer than
`session_synthesis.py` output. Each line is one JSON object linked into a chain
by `parentUuid` (first entry `parentUuid: null`). A real main-session file
contains MANY entry types, e.g. observed:

```
user, assistant, system, attachment, last-prompt, permission-mode,
file-history-snapshot, queue-operation, ai-title, ...
```

⚠️ Therefore the parser uses a **whitelist, not a blacklist**: import ONLY
`type == "user"` and `type == "assistant"` entries (with a `message` object);
silently ignore every other type and any malformed line. Never crash on an
unknown shape. (A blacklist of "known junk" would rot as the CLI adds types.)

Real entry shapes (verified) and how we invert them — a superset of, and
compatible with, `synthesize_session_file`:

| transcript entry | LangGraph message |
|---|---|
| `type:"user"`, `message.content` = **string** | `HumanMessage(content=str)` |
| `type:"assistant"`, `content` = `[{type:"text",text}, {type:"tool_use",id,name,input,caller?}, …]` | `AIMessage(content=<text>, tool_calls=[{id,name,args:input}])` |
| `type:"user"`, `content` = `[{type:"tool_result",tool_use_id,content,is_error?}]` | `ToolMessage(content=<result>, tool_call_id=tool_use_id)` |

Real-field notes confirmed by inspection:
- assistant `message` also carries `id/model/usage/stop_reason/...` — ignore;
  only `content` blocks matter.
- `tool_use` block also has `caller` — ignore.
- `tool_result` block also has `is_error` (bool) — we MAY map it into the
  `ToolMessage` (`status="error"`) for fidelity; not required for correctness.
- `tool_result.content` is usually a string but can be a list of blocks — pass
  strings through, keep lists, stringify anything exotic (mirror of
  `_tool_result_content`).

Edge cases the parser handles:

- **Sidechains / subagents**: in the MAIN session file every entry has
  `isSidechain: false`; sub-agent (Task tool) turns live in SEPARATE files
  `~/.claude/projects/<slug>/<uuid>/subagents/agent-*.jsonl`. So importing the
  main `<uuid>.jsonl` already excludes sidechains by construction — see §2a for
  the answer to "what are sidechains" and why we skip them.
- **`tool_use` with no following `tool_result`** (session ended mid-turn, or the
  result entry was a non-imported type): a dangling tool call breaks the next
  real turn for Anthropic and others. Decision: **synthesize a placeholder
  `ToolMessage("<no result captured>")`** so the history is well-formed for any
  provider. Toggleable later.
- **Multiple `tool_use` in one assistant entry** → one `AIMessage` with
  multiple `tool_calls`; each subsequent `tool_result` user-entry → its own
  `ToolMessage` (symmetric to the writer).
- **Content list on a user entry that is NOT a tool_result** (multimodal input):
  join text blocks into a string; stringify the rest defensively.
- **Ordering**: follow the `parentUuid` chain, not raw file order, to be robust
  to out-of-order lines; fall back to file order if the chain is broken/missing.

### 2a. What "sidechains" are (answering the user's question)

A *sidechain* is the transcript of a **sub-agent** spawned by Claude's `Task`
tool — a separate, nested conversation that does a delegated sub-task and
reports a summary back to the main agent. In current CLI layout those live in
their **own files** (`.../<uuid>/subagents/agent-*.jsonl`, where every entry has
`isSidechain: true`); the main session file does not interleave them. The
ordinary Claude UI shows the *main* conversation, not the raw sub-agent
transcripts. **Conclusion: skip them.** A deep_worker agent's memory should be
its own main conversation; the sub-agents' internal back-and-forth is not part
of "what this agent did" at the top level (only the sub-agent's returned result,
which already appears in the main file as a tool_result, is). No
`--include-sidechains` for now; can be added if a need appears.

## 3. Resolving `<uuid|path>`

`--import-claude` accepts either:

1. An absolute/relative **path** to a `.jsonl` file → use as-is.
2. A bare **session uuid** → search `~/.claude/projects/*/<uuid>.jsonl`.
   - If the agent's working dir is known (`--dir`), prefer
     `session_project_dir(<dir>)/<uuid>.jsonl` (reusing the slug helper from
     `session_synthesis.py`) and fall back to a glob across all project dirs.
   - If multiple matches, error and list them (ambiguous).
3. Neither resolves → clear error listing where we looked.

Reuse `session_synthesis.session_project_dir()` / `_sanitize_cwd()` for the
slug so import and synthesis agree on the path rule (single source of truth).

## 4. Where the history is written (injection into the agent's SQLite DB)

Target (per §1): the agent's **per-agent SQLite database** —
`<agent-data-dir>/sessions_<agent>.db` (`sessions.get_db_path`). deep_worker stores
the message history there as a LangGraph checkpoint via `DeltaSqliteCheckpointer`
(`delta_checkpointer.py`). The main thread's `thread_id` is the agent name; the
offline READ path is `sessions.get_thread_history()` → `checkpointer.aget_tuple()`
→ `checkpoint["channel_values"]["messages"]`. Messages are
`langchain_core.messages` objects, serialized by `JsonPlusSerializer`. Our write
is simply the inverse of that read.

**Injection primitive — `graph.aupdate_state(config, {"messages": msgs})`.**
Compile the agent graph offline with the per-agent checkpointer and let
LangGraph write a valid checkpoint (it produces the correct
`channel_versions`/`versions_seen` itself). This is the same call
`textual_adapter.py` already uses to inject messages into a thread. We do NOT
hand-craft a raw checkpoint dict via `aput` — reproducing LangGraph's version
bookkeeping + every state channel by hand is brittle and a mistake reads back
empty.

```python
config = {"configurable": {"thread_id": <agent_name>, "checkpoint_ns": ""}}
await graph.aupdate_state(config, {"messages": parsed_messages})
```

After this, `deep-worker --history <agent>` shows the imported turns
immediately (validation without starting the agent), and the agent sees them on
its first turn.

### Open-Question 1 resolved (per the user): don't overthink the graph build

The user's guidance: history is just messages; the provider that later reads
them is secondary; they are never replayed. So we do **not** need a special
"trimmed" graph to avoid constructing a provider model. Use the **same
`create_cli_agent` wiring the agent itself uses** (with the requested
`--provider`/`--model`), call `aupdate_state`, dispose. This guarantees the
state schema matches exactly what the agent reads on first `aget_tuple` (no
schema-mismatch risk), and constructing the model once for a one-shot import is
an acceptable cost. The graph is built only to obtain a correctly-shaped
checkpoint writer — no model inference happens during import.

(If, in implementation, building the full graph offline proves heavy or has
side effects, a fallback is a minimal graph compiled with the SAME state schema
+ the per-agent checkpointer and no real model. But default to reusing
`create_cli_agent` for schema fidelity.)

### Interaction with `--clean` and existing history

- Fresh agent (no DB / empty thread): just write the messages.
- Agent that **already has** history: default is to **refuse** and tell the user
  to pass `--clean`, so we never silently splice two conversations together.
  `--import-claude … --clean` is the explicit "replace" path and reuses the
  existing `--clean` wipe (`sessions.clean_agent_history*`). The wipe runs
  before the import write.

## 5. CLI surface & control flow

New arguments in `create_parser()`:

```
--import-claude <uuid|path>   Import a Claude CLI session transcript as this
                              agent's conversation history (written to the
                              agent's SQLite DB), then start the agent.
                              Combine with --clean to replace existing history.
--import-only                 With --import-claude: import and EXIT without
                              starting the agent (validate / prepare only).
```

Control flow in `main()` (`agent_command.py`):

- The import is a **pre-daemonize, one-shot step** that runs *before*
  `run_agent`, so parse/resolve errors are visible on the user's terminal
  (mirrors how `--restart` does its pre-flight checks pre-daemonize). Sequence:
  1. Resolve `<uuid|path>` → transcript file (error out if not found/ambiguous).
  2. Parse → `list[BaseMessage]` (error out if nothing importable).
  3. Guard against clobbering existing history (unless `--clean`).
  4. If `--clean`: run the existing wipe.
  5. Write the parsed messages into the agent's SQLite DB via `aupdate_state`
     (this is the always-done core, independent of provider) — and the ONLY
     side effect. We do NOT seed any claude-code session hint (§6 explains why
     that would lose the agent's tools).
  6. **Default: fall through to the normal launch** (`daemonize` → `run_agent`)
     so the agent starts with the imported history. With **`--import-only`**:
     print a confirmation and exit 0 here (no daemonize, no launch).
- The import write is fully provider-agnostic; `--provider` affects nothing in
  the importer, only how the agent later reads the messages on first run.

`--import-claude` is **mutually incompatible** with `--restart` (restart
inherits a prior config and a prior history; importing into it is ambiguous) —
reject the combination with a clear message. `--import-only` without
`--import-claude` is also rejected.

## 6. claude-code: import writes ONLY history — never seed a resume hint

The SQLite history is written **always, for every provider** (§4 step 5). For
claude-code the import does **nothing else** — and that is deliberate.

### ⚠️ Why we must NOT hand claude-code the imported session file (live-agent bug)

An earlier version added a claude-code "optimization": when the source was a
real `<uuid>.jsonl` in the agent's project dir, the importer seeded
`<agent-data-dir>/<agent>/claude_session_state.json` with that `session_uuid`, so
the provider's restore would `resume` the real file (Approach A of the restore
design). **This broke tool availability and is forbidden.**

Root cause (confirmed live on an agent, 2026-05-24): the imported
transcript is an **ordinary `claude` session created OUTSIDE deep_worker**. When
the claude-code provider resumes it (`ClaudeAgentOptions(resume=<uuid>,
fork_session=True)`), the Claude CLI restores **that session's own
tool/permission configuration**, which contains none of the
`mcp__deepworker__*` tools. deep_worker grants its tools by setting only
`disallowed_tools=BUILT_IN_TOOLS` (blocking the native Bash/Read/Edit/…) and
**never** setting `allowed_tools` — it relies on a **fresh** session exposing
the deepworker MCP tools by default (`client_manager.get_or_create_client`).
On a resumed *external* session that default does not hold: native tools are
blocked by our `disallowed_tools`, and the deepworker MCP tools are not in the
restored allow-list — so the agent ends up with **~1 usable tool** instead of
its full set. `bind_tools` still reports all tools bound to the model; they
simply never reach the resumed CLI session.

The invariant from §1 — *an imported agent must have the same tools as a normal
start* — therefore **requires** that claude-code start a fresh session.

### Correct behaviour

Import writes only the SQLite history. With **no** session hint,
`_has_persisted_session(<agent>)` is false on first run, so the provider does
**not** take the resume path: it starts a **fresh** Claude CLI session (full
deepworker MCP tools, identical to a normal start) and carries the imported
memory the normal way — by synthesizing **our own** transcript from the
LangGraph history (`session_synthesis`), exactly like an ordinary cross-provider
continuation. Same code path as `--provider deepseek/codex`; nothing
claude-code-specific is needed.

Rule of thumb to prevent regression: **only ever `resume` a transcript that
deep_worker itself wrote** (via `session_synthesis`, which produces a session
configured with the deepworker MCP server). Never resume a transcript authored
by an external `claude` run — its tool config is not ours.

## 7. Module layout

- **Parser** (the inverse mapper) lives in `claude_code_provider`, next to its
  twin: `claude_code_provider/session_import.py` with
  `parse_claude_transcript(path: str | Path) -> list[BaseMessage]` and
  `resolve_transcript_path(uuid_or_path: str, cwd: str | None) -> Path`. Keep
  the slug/path helpers shared with `session_synthesis.py`. (Pyright-strict
  applies here — annotate fully.)
- **Injection + CLI glue** lives in `deep_worker_cli`:
  `sessions.import_history(agent_name, messages, *, replace)` (builds the
  offline graph with the per-agent checkpointer and `aupdate_state`s the
  messages into the agent's SQLite DB), plus the `--import-claude` /
  `--import-only` arguments and the `main()` branch in `cli/agent_command.py`.
  It does NOT touch `claude_code_provider.session_state` (no resume hint — §6).
- `deep_worker_cli` already depends on `claude_code_provider` (it imports
  `web_search`, `ClaudeCodeChatModel`), so importing the parser is consistent
  with existing layering.

## 8. TDD plan

Parser (`session_import.py`, provider-side, pyright-strict):

- user string entry → `HumanMessage`.
- assistant text+tool_use entry → `AIMessage` with `content` text and
  `tool_calls=[{id,name,args}]`.
- user tool_result entry → `ToolMessage(tool_call_id=…)`.
- multi `tool_use` in one entry → one `AIMessage`, multiple `tool_calls`;
  matching multiple `ToolMessage`s.
- `parentUuid` chain ordering (shuffled file lines still parse in chain order).
- **whitelist**: only `user`/`assistant` imported; `system`, `attachment`,
  `last-prompt`, `permission-mode`, `file-history-snapshot`, `queue-operation`,
  `ai-title`, unknown types, and malformed lines are ignored (no crash). Use a
  fixture built from a REAL transcript's type mix, not just our writer's output.
- dangling trailing `tool_use` → placeholder `ToolMessage` synthesized.
- assistant extra fields (`id/model/usage/...`), `tool_use.caller`,
  `tool_result.is_error` tolerated; `is_error:true` optionally → error status.
- **round-trip**: `synthesize_session_file(msgs)` then
  `parse_claude_transcript(file)` ≈ `msgs` (modulo cosmetic fields) — locks the
  inverse against the existing writer.
- `resolve_transcript_path`: path passthrough; uuid → project-dir hit; uuid →
  glob hit; not found; ambiguous (multiple) → error.

Injection (`sessions.import_history`, deep_worker_cli):

- after import, `get_thread_history(agent)` reads back the SAME messages from
  the agent's SQLite DB (mirror test — the read path is the existing offline
  one). This is the primary acceptance test: **messages land in the SQLite DB.**
- refuses when a non-empty history exists and `replace=False`; replaces when
  `replace=True`.
- fresh/cleaned agent → DB created with exactly the imported messages.

claude-code hint — REGRESSION GUARD (§6, incident observed 2026-05-24):

- import must NEVER seed a `session_uuid` hint, for ANY provider — including
  `claude-code` with the source `<uuid>.jsonl` sitting in the agent's project
  dir. Test asserts `session_state.save_session_state` is **not called** by
  `handle_import_claude`. (Resuming the external session loses the deepworker
  MCP tools — the agent would have ~1 tool.)

CLI (`agent_command.py`):

- `--import-claude PATH` resolves+parses+writes-to-SQLite then launches.
- `--import-claude UUID` resolves via project dir.
- `--import-claude --import-only` writes to SQLite and exits 0 without launch.
- `--import-only` without `--import-claude` → rejected.
- `--import-claude` + `--restart` → rejected.
- bad uuid/path → non-zero exit, message on stderr (pre-daemonize).
- claude-code + uuid-in-project-dir → still no hint written (regression guard).

End-to-end (integration, opt-in like existing `integration_ai`):

- import a real transcript, start under deepseek, `--history` shows the turns;
  agent's first reply demonstrably "remembers" an imported fact.
- import under claude-code: the agent has its FULL tool set (a fresh session is
  started, not a resume of the external file) and still "remembers" via the
  synthesized-transcript / send-loop path (§6).

## 9. Non-goals / explicitly out of scope

- No live import into a *running* agent (offline, pre-start only).
- No editing/filtering of the imported turns beyond the whitelist + skip rules.
- No attempt to replay tool side effects — tools are NOT re-executed. History is
  the agent's *memory*, not a script. On next start the agent gets its own
  (possibly changed) tool set from its system prompt; old tool calls in history
  are just a record of the past. This is identical in spirit to claude-code
  resume, which also does not re-execute (per E1).

## 10. Resolved decisions (was: open questions)

1. **Offline graph build** — RESOLVED: reuse `create_cli_agent` so the
   checkpoint's state schema exactly matches what the agent reads; constructing
   the model once for a one-shot import is fine; no inference runs during
   import. (§4 "Open-Question 1 resolved".)
2. **import+launch vs import-only** — RESOLVED: **import + launch by default**;
   add **`--import-only`** to import and exit. (§5.)
3. **claude-code continuity** — RESOLVED (revised after the related incident,
   2026-05-24): SQLite history is written **always**, for every provider, and
   that is **all** the importer does. We do NOT seed a `session_uuid` resume
   hint for claude-code — an earlier attempt did, which made the provider resume
   the external `claude` session and **lost the agent's tools** (~1 tool left,
   because the external session's restored config has no deepworker MCP tools).
   claude-code instead starts a fresh session (full tools) and synthesizes our
   own transcript from the imported history. Only ever resume transcripts that
   deep_worker itself wrote. (§6.)
4. **Sidechains** — RESOLVED: skip. They are sub-agent (`Task`) transcripts that
   live in separate `subagents/agent-*.jsonl` files and are not part of the main
   conversation the ordinary Claude UI shows; importing the main `<uuid>.jsonl`
   excludes them by construction. No `--include-sidechains` for now. (§2a.)
