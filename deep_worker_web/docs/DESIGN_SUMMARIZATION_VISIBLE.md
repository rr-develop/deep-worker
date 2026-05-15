# DESIGN: Summarization turns MUST be visible in Full Dialog

**Status:** 📋 PROPOSED → implementation PR
**Date:** 2026-04-23
**Supersedes (partially):** `DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md` §1
(the "summarization is the only thing we drop" rule is reversed here).

---

## 1. Problem

The web UI Full Dialog for an agent whose context has been compacted by
the model's summarization step hides critical behavioural signal:

- When the model's context overflows, a synthetic `HumanMessage` is
  injected whose body begins with one of two markers:
  - `"You are in the middle of a conversation that has been summarized…"`
  - (inline) `"Context Extraction Assistant"`
- This synthetic human message is usually followed by a synthetic
  `AIMessage` that contains the distilled recap of prior context — i.e.
  the *state* the agent carries forward.
- The current renderer
  (`deep_worker_web/app/services/dialog_formatter.py::render_dialog`)
  has an explicit `if kind == "summarization": continue` that drops the
  entire turn.

Consequence: after a compaction the reader sees the agent's subsequent
behaviour with no explanation — "why did the agent suddenly change
topic?" The summarization node IS that explanation. Hiding it breaks
post-mortem analysis, which is the whole point of Full Dialog.

### Scope of this document

**In scope:** making the summarization turn appear in Full Dialog with
an explicit, unambiguous label so the reader can tell synthetic
context-management events apart from real user input.

**Explicitly out of scope:**

- Any change to `AgentProcess._handle_get_history` / the RPyC payload
  contract. A separate analytical note observes that a CLI-side history
  test is currently RED (5 failures against `_message_dict_cache` after the
  delta-cache migration). That is a DIFFERENT root cause for "web sees
  less than CLI --history" and will be addressed in a separate PR — it
  MUST NOT be conflated with the summarization-visibility fix.
- Any structural cleanup of the summarization body (stripping the
  boilerplate preamble). The raw body is kept as-is because it carries
  useful post-mortem signal and because the promp-template text may
  evolve upstream in `deepagents` / `claude_code_provider`.

---

## 2. Goals

1. A summarization turn MUST render in every verbosity level
   (`final`, `text`, `full`) with its own header, distinct from
   `--- User ---` / `--- TaskManager ---` / `--- From agent:<name> ---`.
2. The synthetic `AIMessage`(s) that follow a summarization human
   message MUST render under the same turn — the reader needs to see
   *what the agent took away* from the compaction.
3. No other turn kind's behaviour changes. The existing demo-site
   regression MUST still pass.
4. No message is filtered out anywhere.
5. Purely local change inside one file; no new threads, no new polling,
   no blocking calls, no changes to async wiring, no cache
   invalidation.

### Non-goals

- Rewriting `split_into_turns`.
- Any body-cleanup / prefix-stripping for summarization (may come later
  behind user feedback).
- Any UI / CSS / frontend change. The renderer is plain text; the
  frontend already shows whatever text comes back.

---

## 3. Constraints (hard, from user)

The change must respect, without exception:

- **No sync regressions.** The change touches pure synchronous
  formatting helpers only. No `async def` becomes `def`, no `await` is
  removed. In particular: `dialog_formatter.py` is already fully sync
  (a pure function over plain dicts); we preserve that.
- **No polling.** There is no loop added; we only change a branch inside
  an existing loop.
- **No new threads.** The renderer runs on the existing FastAPI request
  thread (via `run_in_threadpool` from the caller; unchanged). We do
  not spawn, fork, or create executors.
- **Event-driven.** Dialog rendering is triggered by the existing
  `GET /api/agents/{name}/dialog-text` request; no push/broadcast
  surface is added or removed.
- **Non-blocking where it matters.** The only caller is already
  offloaded to a threadpool by FastAPI; our helpers stay O(n) over the
  message list with no I/O. No blocking sleep, no socket, no file.
- **Do NOT break the existing deep_worker message/history cache.** The
  renderer consumes an already-materialised `history: list[dict]` from
  `AgentProcess._handle_get_history`. We do not touch the cache, we do
  not re-fetch history, we do not re-classify more than once per turn.
  This is a smaller rule than the general directive — but load-bearing:
  the delta-cache work (`8c2e28d feat(context): delta-based context
  cache`) must remain untouched.

These constraints are satisfied trivially by the chosen design because
the change is confined to pure functions over already-materialised
dicts and touches no async path, no transport, no cache.

---

## 4. Design — Variant A (minimal)

Three surgical edits in `deep_worker_web/app/services/dialog_formatter.py`:

### 4.1 `_user_header`: add the `summarization` case

```python
def _user_header(kind: HumanKind, content: str) -> str:
    if kind == "user":
        return "--- User ---"
    if kind == "taskmanager":
        return "--- TaskManager ---"
    if kind == "agent_echo":
        return f"--- From agent:{_agent_echo_sender(content)} ---"
    if kind == "summarization":
        return "--- Summarization ---"
    # defensive — classify_human exhaustively covers HumanKind
    return "--- (unknown) ---"
```

### 4.2 `render_dialog`: drop the `continue`

```python
for turn in turns:
    raw = turn.human.get("content", "")
    if not isinstance(raw, str):
        raw = str(raw)
    kind = classify_human(raw)
    # NOTE: summarization is no longer filtered — it is labelled instead.
    blocks.append(_render_turn(turn, kind, verbosity))
```

### 4.3 Module docstring: reflect the new four-sources contract

Replace "Only `summarization` injections are filtered out." with a
paragraph explaining that summarization turns render with their own
`--- Summarization ---` header.

That is the complete implementation surface. Nothing else is touched.

### 4.4 Why this is safe

- `classify_human` already returns `"summarization"` for both marker
  variants — no detection change.
- `_render_turn` is kind-agnostic once it has a header; the summarization
  body flows through exactly like a user body (including "(no response
  yet)" placeholder if — unusually — the synthetic AIMessage is
  missing).
- `_strip_prefix` currently falls through to `return content` for
  summarization (unchanged). Body is rendered verbatim — deliberate,
  see §1 scope.

---

## 5. Expected output examples

### 5.1 `final` mode

```
--- User ---
help me with X

--- Assistant ---
ok, doing X.

--- Summarization ---
You are in the middle of a conversation that has been summarized
…distilled context body…

--- Assistant ---
<recap AIMessage body>

--- User ---
continue

--- Assistant ---
<post-compaction answer>
```

The reader now sees the compaction event between "ok, doing X." and
"continue" and understands why the agent's knowledge shifted.

### 5.2 `text` and `full` modes

Same turn structure; `text` adds intermediate AIMessages, `full` adds
tool-call lines. No summarization-specific divergence — it is just
another labelled turn.

---

## 6. Tests (TDD)

### 6.1 Rewrite `TestSummarizationFiltering` → `TestSummarizationVisibility`

File: `deep_worker_web/tests/unit/test_dialog_formatting.py`.

Replace the four pre-existing "summarization is filtered" tests with:

| # | Test name | What it pins |
|---|---|---|
| 1 | `test_summarization_human_is_rendered_with_summarization_header` | Header appears; synthetic AI body appears; not mis-labelled as User |
| 2 | `test_summarization_legacy_marker_is_rendered` | Both marker variants (`Context Extraction Assistant` and `You are in the middle of…`) route through the same header |
| 3 | `test_summarization_only_history_still_renders` | Lone summarization no longer collapses to NO_HISTORY |
| 4 | `test_summarization_and_following_turn_both_render` | Summarization does NOT swallow the next real turn |
| 5 | `test_summarization_body_preserved` | Body is kept verbatim (no silent truncation of the distilled summary) |
| 6 | `test_summarization_visible_in_text_mode` | Verbosity `text` shows header + all text AIs |
| 7 | `test_summarization_visible_in_full_mode` | Verbosity `full` shows header + tool-call lines if attached |

### 6.2 Regression tests preserved

- `test_dialog_regression_demo_site.py` — untouched; the demo agent's
  turn 374 is a TaskManager turn, not summarization, so its expectations
  are orthogonal.
- `TestClassifyHuman::test_classify_human_summarization_*` — untouched;
  classification rules do not change.

### 6.3 Live practical check (not mocks)

After unit tests are green, we exercise the full wire path:

1. If an existing agent with at least one summarization turn in its
   checkpoint is available (e.g. `my-agent`), point the running web
   server at it and call
   `GET /api/agents/my-agent/dialog-text?verbosity=full` over real
   RPyC. Verify `--- Summarization ---` appears and the post-compaction
   turns are present.
2. If no such agent exists in the sandbox, skip the live step and log
   it as a manual follow-up in the commit message rather than inventing
   a fake live agent.

This step MUST NOT spawn new threads, MUST NOT add polling, and MUST
use the existing async RPyC path already wired up in
`AgentService.get_agent_dialog_formatted`.

---

## 7. Rollout

Single commit. No migration, no data change, no backfill — all existing
checkpoints become readable as-is because only the renderer changes.

## 8. Risk analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| Header collides with user-typed "--- Summarization ---" | Extremely low | Headers are generated from `HumanKind`, not from message text |
| Body contains characters that break the plain-text transport | Low | No escaping change; same path as other kinds |
| Frontend relies on "summarization never appears" | Low | Search of `deep_worker_web/static/` for `Summarization` done before implementation |
| Delta-cache contract on CLI side changes | Unrelated | Scope §1: that bug is outside this PR |

---

### Related design

The path quoted in the summarization node (`/conversation_history/<thread>.md` in older agents, absolute `<data-root>/<agent>/summaries/<thread>.md` in current ones) is governed by the related design document `DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md`. That document explains why the path is now a real on-disk file rather than an ephemeral `/tmp` virtual path.
