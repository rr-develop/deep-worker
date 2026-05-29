# DESIGN: Restoring conversation history for claude-code agents on restart

**Status:** design only (analysis verified against the Claude Agent SDK source and the
agent code; no implementation yet). Branch: `dev`.

**Author context:** observed bug — when a `claude-code` agent is restarted, it
"forgets" what it was doing even though the full history is present in the
LangGraph checkpoint. It then tries to reconstruct context on its own with
varying success.

---

## 1. The bug — why claude-code agents lose memory on restart

### 1.1 How message mirroring works during a LIVE session

`ClaudeCodeChatModel.ainvoke` (`claude_code_provider/chat_model.py`) does NOT
send the whole history to Claude on every turn. It tracks how many messages
have already been delivered and sends only the new tail:

- `sent_count = ClientManager.get_sent_count(thread_id)` — count already sent.
- `new_messages = messages[sent_count:]` — only the unsent tail.
- For each message in `new_messages` (the send loop, ~chat_model.py:608):
  - **`HumanMessage`** → `client.query(content, session_id=...)` — sent as a
    user turn.
  - **`ToolMessage`** → `proxy_mcp.resolve_call(tool_call_id, result)` —
    resolves a *pending MCP Future* that exists only because Claude, in THIS
    live session, just emitted the matching `ToolUseBlock`.
  - **`AIMessage`** → `pass` ("already in Claude's history").
  - **`SystemMessage`** → `pass` (forwarded once via
    `ClaudeAgentOptions.system_prompt`).

This is correct **only while the session is live**: the Claude CLI subprocess
accumulates the full transcript internally, keyed by its `session_id`. We send
deltas; the CLI remembers everything else.

### 1.2 What breaks on restart

Two facts combine:

1. **`sent_count` is in-memory only.** `ClientManager._sent_count` is a plain
   dict (`client_manager.py:216`), never persisted. A fresh agent process
   starts at `sent_count = 0`, so `new_messages` becomes the *entire* history
   loaded from the LangGraph checkpoint.

2. **The new Claude CLI session is EMPTY.** `get_or_create_client`
   (`client_manager.py:281–298`) builds `ClaudeAgentOptions` **without**
   `resume`/`continue_conversation`, so the new `ClaudeSDKClient` starts a
   brand-new CLI session. The `session_id` we pass to `client.query(...)` is
   just the `thread_id` label — the new process has no transcript under it
   (the old transcript lived in the now-dead process / a different session
   file we never recorded).

Replaying the full history through the live-session send loop then drops most
of it:

- **`AIMessage` → `pass`** ⇒ Claude never sees its own prior responses.
- **`ToolMessage` → `resolve_call`** ⇒ resolves a Future that does NOT exist
  in the fresh session ⇒ the tool result is silently discarded.
- Only **`HumanMessage`s** actually reach Claude.

**Result:** after restart Claude receives only the raw user prompts — none of
its assistant turns, none of the tool results. Hence "doesn't remember, tries
to reconstruct." This is a correctness gap, not an LLM quirk.

---

## 2. What the Claude SDK/CLI actually supports (verified)

### 2.1 `client.query(...)` is a USER-turn channel — NOT a history loader

`ClaudeSDKClient.query(prompt, session_id)` (`claude_sdk/.../client.py`) is a
thin transport: for a `str` it wraps `{"type":"user", ...}`; for an
`AsyncIterable[dict]` it writes each dict to the CLI stdin verbatim
(`_internal/query.py:709 stream_input → transport.write`). **Every place the
SDK writes to input uses `type: "user"`** — there is no precedent or protocol
for injecting `type: "assistant"` history through `query`.

> ⚠️ **Verified pitfall (do NOT do this):** "replaying" history as a stream of
> `query` messages does NOT seed history — each user message triggers a REAL
> model generation, and the model would **re-call the tools** (re-running file
> edits, shell commands — real side effects). The CLI's stdin is a turn
> driver, not a transcript loader. An earlier draft of this analysis wrongly
> suggested this path; it is rejected.

### 2.2 The real history-load mechanism: session transcript file + `resume`

The CLI persists each session as a JSONL transcript at
`~/.claude/projects/<sanitized-cwd>/<session-uuid>.jsonl`. `ClaudeAgentOptions`
exposes the load path:

- **`resume: str | None`** → CLI `--resume <session_id>`: restore a prior
  session's full transcript (user + assistant + tool messages) WITHOUT
  re-running anything.
- **`continue_conversation: bool`** → `--continue`: resume the most recent
  session.
- **`fork_session: bool`** → on resume, branch to a new session UUID instead
  of appending to the old one.

`_internal/sessions.py` reads these transcripts and includes BOTH `user` and
`assistant` entries as history (`_to_session_message`, lines ~1000–1010);
`_internal/session_mutations.py::fork_session` copies a transcript into a new
session file (remapping UUIDs, preserving the `parentUuid` chain).

**Conclusion:** restoring history in Claude CLI = providing it a transcript
file and resuming it. That is the only side-effect-free path.

---

## 3. Two approaches (we likely need both)

### 3.1 Approach A — Resume an existing session, validated by TAIL MATCH

We do NOT rely on a reliably-persisted session UUID. The UUID is **not
stable**: with `fork_session=True` a resumed session forks to a NEW id
(verified — `ClaudeAgentOptions.fork_session` docstring: "resumed sessions
will fork to a new session ID rather than continuing the previous session"),
and a session may otherwise rotate its id (e.g. on summarization — see
`DESIGN_SESSION_RESET_ON_SUMMARIZATION.md`). Tracking the live UUID across
restarts and epochs is fragile.

**Decision (per owner):** instead of trusting a recorded UUID, on restart we
**find the candidate session file and VALIDATE it by content** before
resuming:

1. Locate candidate `.jsonl` session files under the project dir for the
   agent's `cwd` (optionally narrowed by a persisted UUID hint, see §3.4).
2. Read the candidate's **last visible message** (the transcript tail).
3. Compare it against the already-sent agent history tail — normally the
   message at index `sent_count - 1`, not a freshly appended user tail. Compare
   on a stable key: role + a content fingerprint (and, for assistant turns,
   tool_use ids). An exact-enough match means "this file represents our
   conversation."
4. **If the tail matches** → `resume` that file (Approach A). The CLI restores
   the already-sent transcript; any newly appended LangGraph tail is still sent
   through the normal send loop.
5. **If it does NOT match** (no file, stale file, divergent tail) →
   **synthesize a fresh transcript from the already-sent LangGraph prefix**
   (Approach B) and resume that. Do not include a newly appended user message in
   the synthesized transcript and then send it again.

A persisted session UUID, if we have one, is used only as a **hint** to find
the candidate file faster — the tail check is authoritative, so a wrong/stale
hint can never cause us to resume the wrong conversation.

- **Pros:** robust to UUID rotation/forking; no dependence on perfectly
  tracking a mutable identifier; the content check is self-validating.
- **Cons:** need a stable, cheap content fingerprint for the tail comparison
  (must survive JSON round-trip and the LangGraph↔CLI representation gap).

### 3.2 Approach B — Synthesize a transcript file from LangGraph history

When there is **no** recorded Claude session — notably **cross-provider
migration** (the agent ran on, say, deepseek/codex, and we now want it to
continue under claude-code) — there is no CLI transcript to resume. We must
**build a valid `.jsonl` transcript file** from the LangGraph message history
and then `resume` it.

This is feasible because the LangGraph message types map cleanly onto the CLI
transcript schema (verified against a real session file):

| LangGraph message | CLI transcript entry |
|---|---|
| `HumanMessage` | `{type:"user", message:{role:"user", content:<str>}}` |
| `AIMessage` (text) | `{type:"assistant", message:{role:"assistant", content:[{type:"text", text:...}]}}` |
| `AIMessage` (tool_calls) | assistant entry with `content:[{type:"tool_use", id, name, input}]` |
| `ToolMessage` | `{type:"user", message:{role:"user", content:[{type:"tool_result", tool_use_id, content}]}}` |
| `SystemMessage` | NOT a transcript entry — forwarded via `system_prompt` |

**Required fields per entry** (from a real
`~/.claude/projects/.../*.jsonl`): `uuid` (fresh uuid4 per entry),
`parentUuid` (previous entry's uuid → linked-list chain; first entry `null`),
`sessionId` (the new synthesized session UUID, identical on every line),
`type`, `message`, `timestamp` (ISO-8601 `...Z`), `cwd` (the agent's
working_dir), `version`, `gitBranch`, `isSidechain:false`. User entries also
carry `userType`, `entrypoint`, `permissionMode`; assistant entries carry
`requestId`.

**File placement** must match what `--resume` looks up:
`~/.claude/projects/<sanitize(cwd)>/<new-session-uuid>.jsonl`, where
`sanitize` replaces every non-alphanumeric char with `-`
(`_internal/sessions.py::_sanitize_path`). For paths > `MAX_SANITIZED_LENGTH`
the CLI appends a hash suffix — for short working dirs this
does not apply, but a robust implementation should reuse the SDK's own
`_get_project_dir` / `_find_project_dir` rather than re-deriving the slug.

- **Pros:** provider-agnostic — enables continuing ANY agent under claude-code
  regardless of its original provider; no dependency on a live CLI session
  surviving.
- **Cons / risks:**
  - We re-implement (or call into) a CLI-internal file format. The schema is
    not a public contract and can drift across CLI versions — pin behind a
    version check and a fallback (if resume of the synthesized file fails,
    fall back to today's behavior + a logged warning, never crash).
  - `tool_result.content` and `tool_use.input` must be serialized into the
    exact block shapes the CLI expects; mis-shaping risks a parse error on
    resume.
  - The synthesized session has no real API request history, so cache-prefix
    benefits start fresh (acceptable).

### 3.3 Unified selection logic (tail-match drives everything)

```
on agent start / first ainvoke with len(messages) > 0:
    sent_count = persisted_sent_count
    restore_prefix = messages[:sent_count] if sent_count > 0 else messages
    candidate = find_candidate_session_file(cwd, hint=persisted_uuid)
    if candidate and tail_matches(candidate, restore_prefix):
        options.resume = candidate.session_id          # Approach A (validated)
        options.fork_session = True                    # fork → our own writable id
    elif len(restore_prefix) > 0:
        new_uuid = synthesize_session_file(restore_prefix, cwd)  # Approach B
        options.resume = new_uuid
        options.fork_session = False  # we own the file already
    else:
        pass  # no history → start fresh (current behavior)

    after the client connects:
        persist the resulting session UUID (hint for next time)
        keep sent_count at the already-sent prefix length; send
        messages[sent_count:] normally so the first post-restart user message is
        delivered exactly once.
```

A is preferred when a real, matching transcript exists (faithful, includes the
CLI's own request history). B is the fallback and the cross-provider path. The
tail check is the single gate — it makes A safe (never resume a wrong/stale
file) and decides when B is required.

### 3.4 sent_count + session-UUID persistence

**Decision (per owner): persist both.** Store in `session_metadata.json` (or a
sibling file): the current `sent_count` and the last-known Claude session UUID.

- `sent_count` persisted ⇒ restart no longer silently resets to 0 and replays
  via the broken empty-session path; the restore logic runs deliberately.
- session UUID persisted ⇒ a cheap **hint** for locating the candidate file in
  §3.3. It is only a hint — the tail check is authoritative — so the fact that
  the UUID may rotate (summarization) or fork (resume) cannot cause a wrong
  resume; at worst the hint misses and we fall back to scan-by-cwd or to
  synthesis.
- Persist the UUID whenever the SDK reports a (possibly new) session id
  (`ResultMessage.session_id`), so the hint tracks the current epoch.

---

## 4. Decisions (resolved with owner) and experiments to run FIRST

### 4.1 Resolved decisions

1. **Session UUID is treated as unstable.** Do not depend on tracking it.
   Validate the candidate session file by **matching its last message against
   the agent's history tail** (`message at sent_count-1`). Match ⇒ resume that
   file; mismatch/absent ⇒ synthesize from history. (§3.1)
2. **Working assumption (to be experimentally confirmed):** `resume` only
   INITIALIZES Claude's history from the transcript — it does NOT replay/
   re-execute tool calls. Must verify before trusting in production. (§4.2)
3. **Synthesized-file fidelity is mandatory and must be well-tested** —
   tool_use/tool_result round-trip, ideally diffed against `fork_session()`
   output for the same logical history. (§4.2)
4. **Persist `sent_count` (and the session UUID hint).** (§3.4)

### 4.2 Pre-implementation experiments — RUN AND PASSED (2026-05-23)

All three were run on a disposable claude-code/opus agent (cwd
`/`, so the CLI wrote its transcript under `~/.claude/projects/-/`).
Results below; all passed, so the approach is validated and we may implement.

- **E1 — resume does NOT re-execute tools. ✅** The live agent ran
  `execute("echo … > /tmp/dw_resume_test/created.txt")`, then we resumed its
  recorded session in a **standalone** `ClaudeSDKClient(resume=<uuid>,
  fork_session=True)` and asked, "without running any tool, what file did you
  create?". Result: the model answered the exact path
  `/tmp/dw_resume_test/created.txt` (recall from the loaded transcript),
  `tool_calls == []`, and `created.txt` was **byte-identical with unchanged
  mtime** (md5 `db9be5c0…` before and after). Confirms: resume INITIALIZES
  history only; no replay, no side effects. (Owner assumption #2 confirmed.)
  Also observed: with `fork_session=True` the resumed session reported a NEW
  `session_id` — confirming the UUID is not stable and validating the
  tail-match design over UUID-tracking.

- **E2 — synthesized transcript resumes cleanly (incl. tools).** ⚠️
  **CORRECTION (2026-05-25): the original E2 ✅ was WRONG / not reproducible.**
  In production, resuming a synthesized transcript failed 100% of the time
  (`No conversation found` and `Control request timeout: initialize`), and
  there was NO committed no-mock test backing the ✅ — only a one-off manual
  claim. Two real defects were found and fixed:
  1. Synthesized **assistant** entries carried only `{role, content}`; a real
     CLI assistant entry's `message` needs `type:"message"`, `id`, `model`,
     `usage`, `stop_reason`, `stop_sequence`. Without them the CLI hangs at
     `initialize` on resume.
  2. The CLI's `--resume` only RECOGNIZES a transcript that BEGINS with the
     `queue-operation` (enqueue → dequeue) meta entries it writes itself. A
     file starting straight with a `user` entry → "No conversation found",
     even in the correct project-dir slug.
  Both are fixed in `session_synthesis.py` and now backed by a REAL no-mock
  integration test (`tests/test_synth_resume_real.py`) that synthesizes →
  resumes via a live `ClaudeSDKClient` → asserts recall of a secret present
  only in the synthesized history (text-only AND tool_use/tool_result shapes).
  **Lesson: a "resume works" claim is only valid if a no-mock test proves
  connect + recall — never a manual spot-check.**

- **E3 — tail content round-trips. ✅** The transcript tail read back as the
  exact assistant text we wrote; the last-message content is a usable
  fingerprint for the §3.1 tail-match. (Exact comparison key — role +
  normalized content (+ tool_use ids for assistant turns) — to be finalized in
  code.)

**Minimal repro recipe (for re-validation):** spawn a claude-code agent, run a
tool, find its session file by content under `~/.claude/projects/<slug>/`,
then `ClaudeSDKClient(ClaudeAgentOptions(resume=<uuid>, fork_session=True,
cwd=<cwd>))` + a recall question; watch the side-effect file + tool_calls.

### 4.3 Remaining items for implementation

- Finalize the tail-match comparison key (E3) in code; handle the
  representation gap (e.g. ToolMessage.content str vs tool_result block).
- ~~Decide the synthesized-file `cwd` to use for the project-dir slug — it must
  match the cwd the resumed `ClaudeSDKClient` is launched with (the agent
  daemonizes to `/`, so today the slug is `-`; pin this explicitly rather than
  relying on inherited cwd).~~ **RESOLVED 2026-05-25** — this was the actual
  restart-still-forgets bug observed on a live agent. The CLI subprocess `cwd`
  was never pinned, so live sessions were written under the shared `-` slug
  while restore/synthesis read from `slug(working_dir)`. Fixed by setting
  `ClaudeAgentOptions.cwd = working_dir` in `ClientManager.get_or_create_client`
  (both options branches), aligning the live-write, restore-read and
  synthesis-write/resumed-read slugs. See
  `DESIGN_FIX_CWD_SLUG_MISMATCH_ON_RESTART.md`.
- Version-guard the synthesis against CLI schema drift (`version` field was
  `2.1.150` in this run) + fallback to current behavior on any resume failure,
  never crash.

---

## 5. References

- `claude_code_provider/chat_model.py` — `ainvoke` send loop, `sent_count`.
- `claude_code_provider/client_manager.py` — `get_or_create_client`,
  `_sent_count`, `get_effective_session_id`, session epochs.
- `claude_code_provider/docs/DESIGN_SESSION_RESET_ON_SUMMARIZATION.md` — why
  session_id rotates on summarization (interacts with the persisted UUID).
- `claude_agent_sdk/types.py` — `ClaudeAgentOptions`
  (`resume`, `continue_conversation`, `fork_session`).
- `claude_agent_sdk/_internal/sessions.py` — transcript
  read + project-dir slug (`_sanitize_path`, `_get_project_dir`).
- `claude_agent_sdk/_internal/session_mutations.py` —
  `fork_session` (transcript write format, UUID remap, `parentUuid` chain).
