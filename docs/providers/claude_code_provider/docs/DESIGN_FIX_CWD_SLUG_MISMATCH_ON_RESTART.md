# DESIGN: claude-code agents still forget on restart — CLI `cwd` is never pinned

**Status:** design (root cause verified on a live agent). Branch: `dev`.

**Author context:** the history-restore-on-restart feature (`e76bd99`) is in
place — `sent_count` + the session-uuid hint are persisted, `resolve_restore_session`
tail-matches a candidate transcript or synthesizes one. Yet restarting
an agent (a `claude-code`/opus agent launched with `--dir /path/to/working_dir`)
still loses all memory. This document isolates **why** and proposes a
one-line-class fix plus regression tests.

---

## 1. The bug — a `cwd` slug mismatch between WRITE and READ

Claude CLI stores each session transcript at:

```
~/.claude/projects/<sanitize(cwd)>/<session-uuid>.jsonl
```

where `sanitize` replaces every non-alphanumeric char with `-`
(so `/path/to/working_dir` → `-path-to-working-dir`, and `/` → `-`). The directory is
chosen by the **cwd of the Claude CLI subprocess** at the moment the session is
created.

`ClientManager.get_or_create_client` (`client_manager.py:224-355`) builds
`ClaudeAgentOptions` for every new client but **never sets `cwd`**. It passes
`working_dir` only to `ProxyMCPServer(...)` (line 278) — never to the options.
`ClaudeAgentOptions.cwd` therefore stays `None`, and the spawned CLI subprocess
**inherits the deep-worker daemon's cwd, which is `/`** (the daemon daemonizes
to `/`). So *live* sessions are written under `~/.claude/projects/-/`.

But the *restore* path reads from the agent's real working directory:

- `chat_model.py:621` → `resolve_restore_session(thread_id, restorable, cwd=self._working_dir or "/")`
- → `session_restore._candidate_tail_entry` → `get_session_messages(uuid, directory=cwd)`
  → looks under `~/.claude/projects/<sanitize(working_dir)>/`.

For such an agent, the restore read slug is `~/.claude/projects/-path-to-working-dir/` — which is
**empty**, because the live session was written under `-`. So:

1. `get_session_messages(uuid, directory="/path/to/working_dir")` returns nothing →
   **no candidate** → tail-match cannot succeed.
2. `resolve_restore_session` falls through to **synthesis** (Approach B):
   `synthesize_session_file(messages, cwd="/path/to/working_dir")` writes a brand-new
   transcript under `-path-to-working-dir`.
3. The resumed `ClaudeSDKClient` is then launched **still without `cwd`** (same
   bug), so the CLI looks for the resume file under `-` again — i.e. NOT where
   synthesis just wrote it. Resume of the synthesized file therefore also fails
   the slug lookup, and `get_or_create_client`'s best-effort fallback
   (`client_manager.py:344-355`) starts a **fresh empty session**.

Net effect on restart: Claude sees only the raw `HumanMessage`s replayed by the
send loop (AIMessages `pass`, ToolMessages resolve into non-existent Futures) —
exactly the original "forgets everything" symptom the feature was meant to fix.

### 1.1 Live evidence (captured 2026-05-25)

- Persisted state is correct: the per-agent state file (`claude_session_state.json`)
  → main `sent_count=183`, uuid `34927a7f-…`; `:tm` `sent_count=166`,
  uuid `09902912-…`. **Persistence is NOT the problem.**
- The two session files exist **only** under `~/.claude/projects/-/` (cwd `/`),
  NOT under `-path-to-working-dir`. Their `cwd` field inside the jsonl is `/`.
- The agent's launch_config `working_dir` is `/path/to/working_dir`.

This is the exact failure the prior design already *anticipated* but left open —
`DESIGN_HISTORY_RESTORE_ON_RESTART.md §4.3`:
> "the agent daemonizes to `/`, so today the slug is `-`; **pin this explicitly
> rather than relying on inherited cwd**."

The experiments E1/E2/E3 were run on a throwaway agent with cwd `/`,
so write-slug and read-slug both happened to be `-` and the
mismatch was masked. Any agent started with `--dir <somewhere>` hits it.

---

## 2. The fix — pin `ClaudeAgentOptions.cwd` to the agent's `working_dir`

Set `cwd=working_dir` on the options built in `get_or_create_client`, in BOTH
branches (the `base_options` `replace(...)` branch and the default
`ClaudeAgentOptions(...)` branch). One change aligns all three axes:

| Axis | Before (cwd `None` → `/`) | After (cwd = working_dir) |
|---|---|---|
| Live session WRITE slug | `-` | `-path-to-working-dir` |
| Restore candidate READ slug (`resolve_restore_session` cwd) | `-path-to-working-dir` | `-path-to-working-dir` ✅ match |
| Synthesis WRITE slug + resumed-client READ slug | write `-path-to-working-dir`, read `-` | both `-path-to-working-dir` ✅ match |

`cwd` is a documented field of `ClaudeAgentSDK.ClaudeAgentOptions`
(verified: `dataclasses.fields` includes `cwd` and `add_dirs`). The SDK passes
it straight to the CLI subprocess working directory.

### 2.1 Where exactly

```python
# base_options branch (client_manager.py ~291)
options = replace(
    self.base_options,
    mcp_servers={"deepworker": proxy_mcp.mcp_server_config},
    disallowed_tools=BUILT_IN_TOOLS,
    permission_mode="bypassPermissions",
    system_prompt=system_prompt,
    cwd=working_dir,            # NEW
)
# default branch (client_manager.py ~303)
options = ClaudeAgentOptions(
    mcp_servers={"deepworker": proxy_mcp.mcp_server_config},
    disallowed_tools=BUILT_IN_TOOLS,
    permission_mode="bypassPermissions",
    system_prompt=system_prompt,
    cwd=working_dir,            # NEW
)
```

`working_dir` is `str | None`. When `None` (no `--dir`), we leave `cwd=None`
(inherit) — identical to today; the slug is `-` and write/read still agree
because the restore path uses the same `self._working_dir or "/"` fallback that
resolves to `/` → slug `-`. So the `None` case stays consistent; only the
"real working_dir" case is fixed.

### 2.2 Why not change the restore-read side to look under `-` instead

Rejected. Writing every agent's transcripts under the shared `-` slug
co-mingles ALL agents' sessions in one directory (`~/.claude/projects/-/`),
which is what we observe today and is fragile (uuid collisions are unlikely but
the scan-by-cwd future enhancement in `session_restore.py:44` becomes
impossible). Pinning `cwd` to the real working dir is the correct, isolating
behavior and matches what synthesis already assumes.

### 2.3 Interaction with the existing resume fallback

`get_or_create_client` already retries once without `resume` if `connect()`
fails (lines 344-355). With the `cwd` fix the resume should now succeed
(file is where the CLI looks), so the fallback becomes the rare-error path it
was designed to be rather than the every-restart path it is today. No change to
the fallback itself.

---

## 3. Scope of change (production code)

- `claude_code_provider/client_manager.py` — add `cwd=working_dir` to both
  options-construction branches in `get_or_create_client`. That is the entire
  production fix.

No change needed in `chat_model.py`, `session_restore.py`, or
`session_synthesis.py`: they already use `self._working_dir` consistently; they
were just reading a directory the writer never populated.

### 3.1 Migration note (existing broken agents)

Agents whose live sessions are already stranded under `-` will not be
auto-healed by this fix alone (their persisted uuid points at a file under the
wrong slug). On their NEXT restart after the fix:

- restore-read looks under the correct slug, finds no candidate → synthesis
  writes a fresh transcript under the correct slug → resumed client (now with
  the right `cwd`) loads it → memory restored from LangGraph history.

So the synthesis fallback heals them on the first post-fix turn; no manual
file move required. (We may optionally document a one-time `mv` of the existing
`-`/<uuid>.jsonl into the correct slug dir, but it is not necessary — synthesis
covers it.)

---

## 4. TDD plan (tests first)

New/extended tests in `claude_code_provider/tests/`:

1. **`get_or_create_client` sets `cwd`** (new test, e.g. in
   `test_client_manager_cwd.py`): patch `ClaudeSDKClient` with a fake capturing
   the `options` it is constructed with; call `get_or_create_client(thread_id,
   tools_metadata=[], working_dir="/path/to/working_dir")`; assert
   `captured_options.cwd == "/path/to/working_dir"`.
   - Variant A: `base_options=None` (default branch).
   - Variant B: `base_options=ClaudeAgentOptions(model=...)` (`replace` branch)
     — assert `cwd` is set AND the other base fields survive.
   - Variant C: `working_dir=None` → `cwd is None` (inherit, unchanged).
2. **Resume path also carries `cwd`** (extend the above / new test): call with
   `resume_session_id="u"` and `working_dir="/path/to/working_dir"`; assert the final
   constructed options have BOTH `resume == "u"` (or fork_session True) AND
   `cwd == "/path/to/working_dir"`.
3. **(Optional, integration-ish, hermetic) write/read slug agreement**: assert
   `session_synthesis.session_project_dir(working_dir)` equals the dir the CLI
   would use for the same `cwd` — i.e. encode the invariant that the slug used
   to WRITE (synthesis) and the slug the resumed client READS (its `cwd`) are
   the same function of `working_dir`. This is a guard test, no real CLI.

All tests stay hermetic — fake `ClaudeSDKClient`, no real `claude` binary, no
network. Reuse existing fakes from the claude-code test suite. The autouse
`_isolate_claude_session_state` conftest fixture already isolates `Path.home()`.

---

## 5. Verification

- Full `claude_code_provider` suite green (new tests + no regressions).
- `pyright` clean for `claude_code_provider` (pre-commit hook runs strict
  pyright on it).
- Manual sanity (optional, gated): restart a `--dir`-launched agent and confirm
  its session now lands under `~/.claude/projects/<slug>/` and that it recalls
  prior context.

---

## 6. References

- `claude_code_provider/client_manager.py:224-355` — `get_or_create_client`.
- `claude_code_provider/chat_model.py:580-644` — restore trigger (read side).
- `claude_code_provider/session_restore.py:39-57` — `get_session_messages(directory=cwd)`.
- `claude_code_provider/session_synthesis.py:59-73` — `session_project_dir` slug.
- `DESIGN_HISTORY_RESTORE_ON_RESTART.md §4.3` — the open "pin cwd explicitly" item this resolves.
