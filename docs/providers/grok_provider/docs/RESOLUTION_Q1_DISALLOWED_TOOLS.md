# Resolution: Q1 — CLI tool-control flags ignored in `grok agent stdio`

**Status:** RESOLVED (2026-06-17). Implementation unblocked.
**Related:** `DESIGN_GROK_PROVIDER.md` §4.6, §9.5, §11A;
`RESEARCH_GROK_PROVIDER.md` §9 (Q-F).

---

## 1. Question (from §16 open issue Q-F)

Original design assumed a four-layer defense against grok built-in
tools (`Shell`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `WebSearch`,
`Task`) bypassing LangGraph:

1. CLI flag `--disallowed-tools` strips them from grok's tool list.
2. System prompt directs the model to prefer deep-worker MCP tools.
3. `builtin_tool_passthrough.py` records anything that slips through.
4. `session/request_permission` classifier denies destructive built-ins.

Layers 1 and 4 were the protocol-level enforcement. The implement_design
step opened with a live verification of Layer 1.

## 2. Live test (Q1)

Executed on a test host against `grok 0.2.54 (fee15ff8e)
[stable]`. Scripts: `q1.py`, `q1b.py`, `q1c.py`,
`q1d_capture.py` (session-scoped scratch probes).

### 2.1 Tested CLI flags (Layer 1)

For each flag, sent a `session/prompt` asking grok to run
`echo HELLO_SHELL` via the Shell tool. Observed whether grok emitted
`session/update sessionUpdate=tool_call title=Shell`.

| Invocation | Result |
|---|---|
| `grok agent stdio` (baseline) | Shell called |
| `grok --disallowed-tools Shell agent stdio` | Shell called |
| `grok --disallowed-tools "Shell,Bash,Read,Write,Edit,Glob,Grep,WebSearch" agent stdio` | Shell called |
| `grok --tools "" agent stdio` (empty allowlist) | Shell called |
| `grok --tools Read agent stdio` (allowlist excludes Shell) | Shell called |
| `grok --deny Shell agent stdio` | Shell called |
| `grok --deny "Shell(*)" agent stdio` | Shell called |
| `grok --permission-mode plan agent stdio` | Shell called |
| `grok --sandbox restricted agent stdio` | Shell called |

**Conclusion: ALL CLI tool-control flags are silently ignored in
`agent stdio` mode.** Layer 1 is non-functional on grok 0.2.54.

### 2.2 Permission-request path (Layer 4)

Without `--always-approve`, drained incoming JSON-RPC requests from grok
for the duration of one prompt and counted `session/request_permission`
arrivals. Result: **0 permission requests** received; Shell executed
unconditionally and `session/update tool_call` with full streaming output
followed.

**Conclusion: grok 0.2.54 `agent stdio` never invokes
`session/request_permission` for built-in tools.** Layer 4 is dead code.

### 2.3 What the ACP stream DOES give us (the unblocker)

Captured the full notification sequence for one Shell invocation
(`q1d_capture.py`). Every piece needed for a faithful LangGraph
record is present:

| Field needed | Present in ACP payload | Source |
|---|---|---|
| `toolCallId` (stable, unique) | yes | every `tool_call*` event |
| Tool name | yes — `"title": "Shell"` | initial `tool_call` |
| Arguments | yes — `"rawInput": {"command": "echo MAGIC_TOKEN_XYZ987", "description": "..."}` | initial `tool_call` |
| Streaming partial output | yes — `content[].content.text` (status=`in_progress`) | mid-flight `tool_call_update` |
| Final result | yes — same field with full text on completion | `tool_call_update status=completed` |
| Metadata: `exit_code`, `current_dir`, `total_bytes`, `duration_ms`, log path | yes — `rawOutput` on the final update | `tool_call_update status=completed` |

Sample of the final `tool_call_update` for `echo MAGIC_TOKEN_XYZ987`:

```json
{
  "sessionUpdate": "tool_call_update",
  "toolCallId": "call-d5446c00-...-composer_call_kLDxP",
  "status": "completed",
  "content": [{"type": "content",
               "content": {"type": "text", "text": "MAGIC_TOKEN_XYZ987\n"}}],
  "rawOutput": {
    "type": "Bash",
    "exit_code": 0,
    "command": "echo MAGIC_TOKEN_XYZ987",
    "current_dir": "/tmp",
    "total_bytes": 19,
    "output_for_prompt": "Exit code: 0\n\nCommand output:\n\n```\nMAGIC_TOKEN_XYZ987\n```\n\nCommand completed in 342 ms.\n...",
    "output_file": "<grok sessions dir>/terminal/call-d5446c00-...log"
  }
}
```

## 3. User decision (2026-06-17)

> "Can you intercept this call at all, to display it in the LangGraph
> history? I care little about HITL. Agents almost always run with
> auto-approve."

Accepted. Threat model reduced from "prevent destructive built-ins" to
"observe and record every built-in action in conversation history".

## 4. Design changes

§11A rewritten. Old §11A.2 "disable strategy" replaced by §11A.2
"two-channel architecture":

- **Channel A — deep-worker MCP tools.** Unchanged. proxy MCP server
  in-process, `tools/call` parks Future, LangGraph ToolNode resolves it.
- **Channel B — grok built-in tools.** Observed, not blocked. Surfaced
  as `AIMessage(tool_calls=[{id, name:"Shell", args:rawInput}])` from
  `ainvoke`; LangGraph ToolNode invokes `_BuiltinPassthroughTool._arun`
  which awaits a per-`toolCallId` Future. The reader_task resolves the
  Future on `tool_call_update status=completed` with the final text and
  metadata, ToolMessage flows into `state["messages"]` with
  `additional_kwargs={"grok_builtin": True, "grok_meta": {exit_code,
  cwd, duration_ms, log_path}}`.

§11A.3 "system prompt" downgraded from "soft defense" to "UX guidance
(prefer deep-worker MCP for unified logging/retry/HITL where an
equivalent exists)".

§11A.4 "passthrough" promoted from "optional safety net" to
"mandatory main channel for built-ins".

§9.5 "permission handling" kept as forward-compatibility code (with
auto-allow for everything; classifier scaffolding retained in case
grok adds permission routing in a later release).

§4.6 updated with the empirical finding and pointer to this resolution.

§16 Q-F closed with reference to this document.

## 5. Implementation impact

Channel B is implemented in `grok_provider/builtin_tool_passthrough.py`
(~250 LOC). It mirrors `codex_provider/native_tools.py` but covers the
full set of grok built-ins, not just `apply_patch`/`update_plan`. Plus
a wildcard `_GenericBuiltinPassthrough` fallback for forward
compatibility with grok-CLI additions.

Threat-model implications documented in `DEEP_WORKER_TOOL_CONTRACT`
inside system prompt: agents are told that both channels record into
history, and that channel A is preferred only for unified logging — not
for safety.

## 6. Repro

```
$ cat q1.py q1b.py q1c.py q1d_capture.py
```

(scripts are not part of the repo — they are quick probes; the
relevant payload sample is captured in §2.3 above.)
