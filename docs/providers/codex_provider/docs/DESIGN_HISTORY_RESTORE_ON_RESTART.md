# DESIGN: Restoring conversation history for Codex agents on restart

**Status:** design — under review (branch `dev`). Mirrors the claude-code fix
in `claude_code_provider/docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md`.

**Author context:** the Codex provider was analysed (code-only) for
parity with the claude-code restart support we already shipped, and the
same "forgets everything on restart" class of bug was found, unfixed. This
document specifies the fix and is sent for review before implementation.

---

## 1. The bug — why Codex agents lose memory on restart

### 1.1 How message mirroring works during a LIVE session

`CodexChatModel.ainvoke` (`codex_provider/chat_model.py`) does NOT resend the
whole history each turn. It tracks how many LangGraph messages were already
mirrored into the live Codex session and sends only the new tail:

- `state = client_state.get_or_create(agent_thread_id)`
- `new_messages = messages[state.sent_count:]` — only the unsent tail
- send loop (chat_model.py:427):
  - **`HumanMessage`** → appended to `user_text_parts` → becomes the prompt
  - **`ToolMessage`** (proxy MCP) → `proxy_mcp.resolve_call(...)` — resolves a
    *pending MCP Future* that exists only because Codex, in THIS live session,
    just emitted the matching `McpToolCallItem`.
  - **`ToolMessage`** (`codex_native`) → marks `saw_native_resume` (Codex ran
    the native tool itself; we just keep reading the same exec handle)
  - **`AIMessage`** → `pass` ("already in Codex's session")
  - **`SystemMessage`** → `pass` (forwarded via `AGENTS.override.md`)
- spawn with `session_uuid=state.codex_session_uuid` →
  `codex exec resume <uuid>` when set, fresh session when `None`.

Correct **only while the session is live**: the Codex `exec` session
accumulates the full transcript internally under its `codex_session_uuid`. We
send deltas; Codex remembers the rest.

### 1.2 What breaks on restart

`codex_provider/client_state.py` has **no persistence layer at all** — there is
no `session_state.py` analog (that file exists only in `claude_code_provider`).
`ThreadState` (`sent_count`, `codex_session_uuid`, `current_exec`) is in-memory
only. So:

1. Process dies → `CodexClientState._states` (`{}`) is lost.
2. Fresh process: `sent_count = 0`, `codex_session_uuid = None`,
   `current_exec = None`.
3. First `ainvoke` after restart:
   - `new_messages = messages[0:]` — the *entire* checkpoint history.
   - send loop: `AIMessage → pass` (Codex never sees its prior answers);
     `ToolMessage → resolve_call` resolves a Future that does not exist in the
     fresh process (silently dropped); only `HumanMessage`s survive.
   - spawn with `session_uuid=None` → brand-new empty Codex session.

**Result:** after restart Codex receives only the raw user prompts — none of
its assistant turns, none of the tool results. Same correctness gap as
claude-code had. This is NOT an LLM quirk.

---

## 2. What Codex CLI actually supports (verified in code)

- `ExecArgs.session_uuid` already maps to `codex exec resume <uuid>`
  (`exec_runner.py:576-577`). The resume path is **already wired** in
  `ainvoke` (chat_model.py:484) — it just never has a uuid to pass after a
  restart because nothing persisted it.
- Codex stores full sessions in `CODEX_HOME/sessions/<uuid>.jsonl` and manages
  their integrity itself. `--ignore-user-config` does NOT affect resume —
  sessions live under `CODEX_HOME`, not in `config.toml` (exec_runner.py:509).

**Consequence — a key simplification vs claude-code.** claude-code needed a
*tail-match* / *synthesis* step (`session_tail_match`, `session_synthesis`)
because its CLI transcript files are fragile and the resume target must be
verified. Codex owns its session store and resumes by uuid natively, so for
Codex we can **trust the persisted uuid directly**: persist it, load it on the
first turn after restart, pass it on spawn. No tail-match needed (see §6 for
the safety argument and the fallback).

---

## 3. The fix — persist + lazily reload `sent_count` and `codex_session_uuid`

Mirror the claude-code mechanism, minus tail-match/synthesis.

### 3.1 New module `codex_provider/session_state.py`

A near-verbatim analog of `claude_code_provider/session_state.py`:

- Storage: the per-agent state file (`codex_session_state.json`) under the
  agent's data directory, a dict
  keyed by full `agent_thread_id` (so `agent` and `agent:tm` keep separate
  entries while sharing the agent data dir).
- `_thread_base(thread_id)` — strip `:tm` role suffix (Codex has no
  `__epoch_N` rotation suffix; claude-code's epoch handling is dropped). Keep
  the `__epoch_` strip too, defensively, in case a future suffix is added — it
  is a harmless no-op for current Codex thread_ids.
- `session_state_path(thread_id) -> Path`
- `save_session_state(thread_id, *, sent_count, session_uuid)` — best-effort,
  never raises, atomic temp-file replace, read-modify-write preserving other
  threads' entries.
- `load_session_state(thread_id) -> dict | None` — `{"sent_count", "session_uuid"}`
  or `None` on missing/corrupt file.

Filename `codex_session_state.json` (NOT `claude_session_state.json`) so the
two providers never collide in a shared agent data dir.

### 3.2 Lazy seed + persist in `CodexClientState`

Persistence lives in the singleton (the natural owner of `sent_count` /
`codex_session_uuid`), exactly where claude-code put it (`ClientManager`).

- `get_or_create(agent_thread_id)`: when creating a NEW slot (i.e. first access
  in a fresh process), lazily seed it from disk:
  ```
  st = ThreadState()
  loaded = load_session_state(agent_thread_id)   # best-effort
  if loaded is not None:
      st.sent_count = int(loaded["sent_count"])
      st.codex_session_uuid = loaded.get("session_uuid")
  self._states[agent_thread_id] = st
  ```
  This is the seam that makes restart deterministic: the first `ainvoke`
  after restart now sees the real `sent_count` (so `new_messages` is the
  correct small tail, not the whole history) AND a `codex_session_uuid` (so
  spawn resumes the existing Codex session instead of starting empty).

- New `persist(agent_thread_id)` helper on the singleton: read the slot, call
  `save_session_state(...)` with its current `sent_count` + `codex_session_uuid`.
  Best-effort, never raises.

- `record_summarization(agent_thread_id)`: after resetting `codex_session_uuid=None`
  and `sent_count=0` in memory, **also persist the reset** so a restart that
  happens right after a summarization does not reload a stale pre-summary uuid.

### 3.3 Call sites in `CodexChatModel.ainvoke`

Two persistence points, mirroring claude-code's `update_sent_count` /
`record_session_uuid`:

1. Right after `state.sent_count = len(messages)` (chat_model.py:453) →
   `self._client_state.persist(agent_thread_id)`. This captures the new cursor
   (and the current uuid, possibly still `None` if Codex has not reported it
   yet this turn).
2. When we learn the session uuid from `ThreadStartedEvent`
   (chat_model.py:537-538), after setting `state.codex_session_uuid`, →
   `self._client_state.persist(agent_thread_id)`. This captures the uuid as
   soon as Codex assigns it, so even a crash mid-turn leaves a resumable uuid
   on disk.

The summarization-reset branch (chat_model.py:377-386) already calls
`record_summarization`, which (per §3.2) will now also persist the reset — no
extra call site needed there.

`get_or_create` already seeds on creation (§3.2), so no explicit load call is
needed in `ainvoke`.

---

## 4. Why persist on the singleton, not the model

Same reason as claude-code: LangGraph rebuilds `CodexChatModel` instances
during checkpoint (de)serialization. Per-instance state would be lost on every
rebuild. The singleton lives at module scope and survives. Persistence belongs
next to the in-memory state it backs.

---

## 5. TDD plan (tests first)

New test file `codex_provider/tests/test_session_state_persistence.py`:

1. `session_state.py` unit tests (mirror claude-code's
   `test_session_state_persistence.py`):
   - save → load round-trips `sent_count` + `session_uuid`.
   - `load_session_state` returns `None` for a missing file.
   - corrupt JSON → `None` (never raises).
   - `_thread_base` strips `:tm`; `agent` and `agent:tm` are isolated keys in
     the same file.
   - other threads' entries preserved across a save (read-modify-write).
   - `Path.home()` monkeypatched to a tmp dir (per-test isolation).

2. `CodexClientState` seed/persist tests (extend or sit beside
   `test_client_state.py`):
   - fresh `get_or_create` with a persisted file seeds `sent_count` +
     `codex_session_uuid` from disk.
   - fresh `get_or_create` with no file → defaults (0 / None), no raise.
   - `persist()` writes the slot's current values; a NEW `CodexClientState`
     (simulating restart) then seeds them back via `get_or_create`.
   - `record_summarization()` persists the reset (a subsequent fresh
     `get_or_create` seeds `sent_count=0`, `codex_session_uuid=None`).

3. Restart-path test in `chat_model` (mirror claude-code
   `test_restore_trigger_in_ainvoke.py`), using the existing fake exec-runner
   pattern from `tests/test_chat_model.py`:
   - Simulate a "restart": persist `sent_count=N`, `session_uuid=U`; build a
     FRESH `CodexClientState` singleton; first `ainvoke` with a history of N
     messages must (a) compute `new_messages == messages[N:]` (not the whole
     history) and (b) spawn with `ExecArgs.session_uuid == U` (resume), not a
     fresh session.

All tests must keep the suite hermetic (no real `codex` binary, no network) —
reuse the fake `CodexExecRunner` / proxy already used in
`tests/test_chat_model.py`, and monkeypatch `Path.home()` for the on-disk part.

---

## 6. Safety of "trust the uuid" (no tail-match) + resume fallback

- The uuid is only ever read back from our own atomic write of the value Codex
  itself reported in `ThreadStartedEvent`. It is never user-controlled. Codex
  assigns the uuid once and it is stable for the session's life (unlike Claude
  CLI's uuid, which mutates on `fork_session` / epoch rotation) — so resuming
  by the persisted uuid is consistent.
- Persisting on summarization-reset (§3.2) prevents the one stale-uuid case we
  can cause ourselves (restart right after summarization).

### 6.1 Resume fallback (PROMOTED to in-scope per review)

If the Codex session file is gone (e.g. `CODEX_HOME` wiped, or the session
expired/was pruned by Codex), `codex exec resume <uuid>` fails. Unlike
claude-code — where resume failure surfaces synchronously at
`ClaudeSDKClient.connect()` and is retried right there
(`client_manager.py:344-355`) — for Codex the failure surfaces **inside the
ainvoke event-consumption loop**, asynchronously, as a `TurnFailedEvent` or an
`END_OF_STREAM` carrying `handle.pump_error` (the spawn itself returns a handle
successfully; the subprocess only errors once it tries to load the missing
session).

Without a fallback, a user who restarts an agent whose Codex session has
expired gets a hard `status=ERROR` instead of a working agent. This was
flagged in review; we implement the fallback now.

**Design — single retry without resume:**

Extract the "build ExecArgs → spawn → consume events" block of `ainvoke`
(steps 9–10) into a helper `_run_turn(...)` that returns the same
`(final_text, pending_tool_calls)` it does today and records the handle on
`state.current_exec`. Then, for the **fresh-turn spawn path only** (i.e. NOT
the mid-turn tool-resolution branch, which has no `session_uuid` decision to
retry) wrap it:

```
spawned_with_resume = state.codex_session_uuid is not None
try:
    result = await self._run_turn(state, args, ...)
except (RuntimeError, <CodexProtocol/turn-failed>) as exc:
    if not spawned_with_resume:
        raise                       # genuine failure, nothing to retry
    # Resume of a stale/missing Codex session failed. Fall back to a
    # fresh session and replay the full history once.
    log "resume of <uuid> failed (<exc>); starting fresh session"
    state.codex_session_uuid = None
    state.sent_count = 0            # force full-history replay next build
    self._client_state.persist(agent_thread_id)
    # rebuild new_messages == messages[0:], user_text_parts from the
    # full history, args2 with session_uuid=None, spawn again, consume.
    result = await self._run_turn(state, args_fresh, ...)
return result
```

Constraints / correctness:

- **Only retry when we actually attempted a resume** (`spawned_with_resume`).
  A first-ever fresh session that fails is a real error and must propagate
  unchanged — no behaviour change for the non-restart path.
- **Only the fresh-turn branch retries.** The mid-turn branch (resolving a
  proxy/native tool inside a live exec) never sets a new `session_uuid` and
  must keep its current error semantics.
- The fallback prompt is the same human-text replay we already build; AIMessage
  turns are still `pass`-ed (a truly fresh Codex session cannot be given the
  assistant turns back). This matches claude-code's accepted first-cut limit:
  the agent restarts working with the user prompts, rather than ERROR-ing.
- `current_exec` from the failed resume attempt is closed before re-spawn
  (reuse the existing `_suppress_errors()` + `aclose()` cleanup).
- The retry is **single**: if the fresh spawn also fails, the error propagates
  (genuine breakage, surfaced as today).

### 6.2 TDD additions for the fallback

Add to `test_session_state_persistence.py` (or `test_chat_model.py`) a test
using the fake exec-runner whose FIRST `spawn` (with `session_uuid=U`) yields a
turn-failure stream and whose SECOND `spawn` (with `session_uuid=None`) yields
a normal completion. Assert: (a) two spawns happened, (b) the second carried
`session_uuid is None`, (c) `state.codex_session_uuid` was cleared and the
reset persisted, (d) `ainvoke` returned the fresh session's `final_text`
(no exception). Plus a negative test: a fresh-session (no-resume) spawn failure
still raises (no spurious retry).

---

## 7. Files touched

- **new** `codex_provider/session_state.py`
- `codex_provider/client_state.py` — lazy seed in `get_or_create`, new
  `persist()`, persist-on-reset in `record_summarization`.
- `codex_provider/chat_model.py` — two `persist()` calls (after `sent_count`
  update; after `ThreadStartedEvent` sets the uuid) AND the single resume
  fallback on the fresh-turn path (§6.1).
- **new** `codex_provider/tests/test_session_state_persistence.py`
- (maybe) extend `codex_provider/tests/test_client_state.py`
- docs: this file; cross-reference from `DESIGN_CODEX_PROVIDER.md` if it has a
  state/persistence section.

Production behaviour for a LIVE (non-restarted) session is unchanged: the only
new runtime work is a small best-effort JSON write per turn and a one-time disk
read when a slot is first created in a process.
