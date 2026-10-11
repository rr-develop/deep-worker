# Recoverable failures must be visible

Status: design, revision 3 (A committed; B signed off; C revised again)
Date: 2026-09-04

## 1. What the owner reported

`test_lsp` hit `MaxIterationsExceeded`. Two complaints:

1. the turn failed at all;
2. the failure "was not shown in the agent status in deep_worker_web — it
   just went quietly to idle instead of error", and in the dialog it
   "simply looked like the model's final answer".

Both are legitimate. They are **three** separate defects, not one, and
they are independent: each can be fixed and verified alone.

## 2. Evidence

All facts below were established by execution against the live state of
`test_lsp` (`threads.db` in the agent data directory, 112 checkpoints,
21 `messages` writes), not by reading code.

The conversation, reconstructed from the `writes` table (LangGraph stores
delta checkpoints here, so `channel_values` is empty and the messages
live in `writes`):

```
19 [ai]   tc=['write_todos']  "<think> ... Let me plan this out ..."
20 [ai]   tc=[]               (same text, tool_call stripped)
21 [tool]/write_todos  "Tool 'write_todos' is disabled for this agent by its
                        tool selection and was NOT executed. Do not retry it..."
22 [system]  "[runtime] ... MaxIterationsExceeded ..."
```

Launch config: `"tools": ["none", "+serena_*"]`, provider `minimax`.

## 3. Defect A — the prompt advertises a tool the selection removed

### 3.1 Mechanism

`ToolSelectionMiddleware` removes the `write_todos` **schema**, but the
**prompt section** contributed by upstream `TodoListMiddleware` survives.
The model reads "You have access to the `write_todos` tool", calls it,
is refused, and on the next iteration reads the same instruction again.
Twenty iterations, then the cap.

### 3.2 Root cause

`middleware/tool_selection.py:90-102` — `_PROMPT_SECTION_OWNERS` lists six
sections. None of them owns `write_todos`, so `_section_owners()` returns
`None` ("unowned text, never touch") and the section is kept.

Proven by execution:

```
_section_owners(WRITE_TODOS_SYSTEM_PROMPT) = None   -> UNOWNED -> KEPT
kept tool schemas: ['serena__find_symbol']
PROMPT SURVIVES WHILE SCHEMA IS GONE: True
```

The file's own docstring (`:28-38`) already forbids exactly this:
"Removing a tool's schema while leaving that text in place would tell the
model it has tools it cannot call, which corrupts the very measurement
the feature exists to support."

So this is not a new rule — it is an unimplemented case of an existing one.

### 3.3 Fix

Add one entry to `_PROMPT_SECTION_OWNERS`:

```python
("## `write_todos`", frozenset({"write_todos"})),
```

The marker is the literal heading of upstream's `WRITE_TODOS_SYSTEM_PROMPT`
(verified: the constant starts with ``## `write_todos` ``).

`test_tool_selection_middleware` already pins every marker against the real
constant it comes from; the new marker is pinned against
`langchain.agents.middleware.todo.WRITE_TODOS_SYSTEM_PROMPT` the same way,
so an upstream heading drift fails the test instead of silently making the
section un-droppable again.

### 3.4 Why not raise the iteration cap

Raising `max_iterations` (`agent_process.py:5120`) would buy more loops of
the same futile cycle. The cap did its job — it stopped a loop. The loop is
the defect.

## 4. Defect B — a recoverable failure leaves no trace in the status

### 4.1 Mechanism

`agent_process.py:5357-5406`, on hitting the cap:

* `_log_error_context(..., recoverable=True)` -> error log
* `_append_recoverable_error_to_history(...)` -> SystemMessage in history
* `set_status(AgentStatus.IDLE)` -> status file

`session_metadata.json` afterwards:

```json
{"status": "idle", "error_message": null, ...}
```

`set_status` (`session_manager.py:182-198`) does not touch
`error_message`, and nothing else sets it on this path. The web UI reads
this file and the `status_change` event built from it
(`session_manager.py:170-174`), so "finished normally" and "hit the cap"
are indistinguishable.

### 4.2 This is NOT a violation of POLICIES.md §1

`docs/POLICIES.md` §1 (No-Agent-Error-State Policy) deliberately requires
IDLE here: the cap is recoverable, the user can type "continue". `ERROR`
is reserved for fatal conditions (`FATAL_EXCEPTION_TYPES`).

**The policy is not being changed.** §1 also requires: "persist a
structured record to the error log" — and a record was persisted. The gap
is that the record is invisible to the UI, which reads status, not logs.

### 4.3 Fix

Carry the last recoverable failure alongside IDLE, without ERROR.

Add an optional parameter to `SessionManager.set_status`:

```python
async def set_status(self, status, *, metadata=None, error_message=None)
```

* `error_message=None` (default) — **existing behaviour byte-for-byte**:
  the field is left exactly as it is. Every current caller is unaffected.
* `error_message="..."` — record it and include it in the persisted state
  and in the `status_change` payload, while the status stays IDLE.

The recoverable-error path passes the same text it already writes to
history.

`_notify_status` already puts `error_message` into the `status_change`
payload (`session_manager.py:170-174`), so the new value reaches the web
without touching the notifier.

### 4.4 The marker must be cleared, or it becomes permanent

An earlier revision of this design claimed "`clear_error()` already
exists and is called when a new turn starts". The first half is true, the
second is **false**: `clear_error` has **zero production callers** — the
only references are its own definition (`session_manager.py:213`), a
docstring, and two tests (`test_session_manager.py:132`,
`test_session_manager_notify.py:73`). A new turn calls
`set_status(AgentStatus.RUNNING)` (4 sites), which by this very design
does not touch `error_message`.

Verified by execution against a temporary `data_dir`:

```
after cap (today) : idle     | error_message = None
with B's marker   : idle     | error_message = [runtime] MaxIterationsExceeded
new turn RUNNING  : running  | error_message = [runtime] MaxIterationsExceeded   <-- not cleared
clean turn ends   : idle     | error_message = [runtime] MaxIterationsExceeded   <-- nor here
persisted: {'status': 'idle', 'error_message': '[runtime] MaxIterationsExceeded'}
```

So without an explicit reset, one cap hit would make the web show that
error **forever**, on top of every later successful turn. That is worse
than today's state: an invisible failure would become an unclearable one
— the same class of defect already seen at `agent_service.py:551`, where
a non-empty stale value permanently blocked recomputation.

**The reset is part of this fix, not an assumption.** The
recoverable-error path's counterpart is an explicit `clear_error()` call
when a turn starts.

**Only at two of the four `set_status(RUNNING)` sites.** They are not the
same kind of event:

| Site | Meaning |
|---|---|
| `agent_process.py:2677`, `:4597` | a turn genuinely **starts** |
| `status_callback.py:74`, `:121` | a **return** to RUNNING mid-turn, after `on_tool_end` / a recoverable `on_tool_error` |

Clearing at all four would erase the marker *inside the same turn*: the
cap sets `error_message`, the next `on_tool_end` in that very turn wipes
it, and nothing reaches the web again — defect B restored in the shape of
its own fix. So the reset goes in the two `agent_process` sites only, and
carries a comment saying why not the other two, or a later reader will
"unify" them.

The test must be able to tell those apart: cap → `error_message` non-empty
→ simulate `on_tool_end` (`set_status(RUNNING)` from `status_callback`) →
the marker **must survive** → only a new turn via `agent_process` clears
it. A test written merely as "cap → RUNNING → None" would pass on the
wrong placement.

Rejected: clearing `error_message` inside `set_status` when
`status == RUNNING`. It would make `set_status` implicitly mutating,
contradicting the "byte-for-byte unchanged when `error_message=None`"
property this design sells it on.

Rejected: a new `AgentStatus.RECOVERABLE_ERROR`. It would be a new status
value every consumer (web, CLI, registry, A2A) must learn, to express
something the existing `error_message` field already expresses, and it
would sit uncomfortably next to §1's two-state model.

## 5. Defect C — the default dialog rendering hides the failure entirely

This is the one that matches the owner's words most directly: "it just
looked like the model's final answer".

### 5.1 Mechanism

`deep_worker_web/app/services/dialog_formatter.py`:

* `render_dialog(..., verbosity="final")` — **`final` is the default**
  (`:475`).
* `_render_assistant_block_final` (`:332-340`) returns only
  `turn.ai_text_messages[-1]`. It never looks at `turn.stream` and never
  at `turn.system_messages`.
* `_render_system_preamble` returns `None` at `final` (`:390-391`).

Reproduced by execution on the real message shape from `test_lsp`:

```
verbosity=final   ERROR VISIBLE = False
verbosity=text    ERROR VISIBLE = True
verbosity=full    ERROR VISIBLE = True
```

At `final` the rendering shows the model's `<think>` musing as the
assistant's answer and drops the runtime error completely. That is
precisely what the owner saw.

### 5.2 Root cause

The `Turn` docstring (`:215-219`) states the policy:

> ⚠️ POLICY: history rendered at `text` or higher MUST surface system
> messages — a "full" rendering without them is misleading and is
> forbidden.

The policy was written about `text` and `full`, and says nothing about
`final`. The owner's complaint is that `final` — the default, the view
most people actually look at — is misleading in exactly the way the
policy exists to prevent.

### 5.3 Fix

At `final`, render system messages that carry a runtime failure. Not all
system messages: the base prompt and routine middleware hints stay hidden
at `final`, which is the point of that verbosity.

The failure marker is already present and unambiguous — the runtime stamps
`additional_kwargs={"recoverable_error": True, "exc_name": ...}` when it
appends the message (`agent_process.py:4482-4489`).

#### 5.3.1 The flag does not currently reach the web at all

An earlier revision observed that `recoverable_error` appears nowhere in
`deep_worker_web` and concluded the flag "is set and never consumed".
The observation is right, the inferred cause was wrong: it is not
unconsumed by oversight — it **never arrives**.

The web reads history via `_handle_get_history`
(`agent_process.py:3973`), which converts each message through
`MessageDictCache.serialize` → `serialize_message`
(`process/message_dict_cache.py:46`). That function reads
`additional_kwargs` for exactly one key, `created_at` (`:60-64`); the
dict itself is never emitted. Verified by running the real serializer on
the real message shape:

```
serialized keys: ['content', 'created_at', 'id', 'type']
recoverable_error survives: False
additional_kwargs emitted at all: False
```

`split_into_turns` therefore *cannot* see the flag — its input dict does
not contain it. A parallel `system_error_flags` list would collect
nothing but `None`.

#### 5.3.2 Radius correction

`deep_worker_cli/process/message_dict_cache.py` joins the radius: the
flag must be forwarded into the serialized dict, whitelisted key by key
in the manner of `created_at`. Emitting `additional_kwargs` wholesale is
rejected — it carries `input_message_id` and other internals, and would
add bytes to every polling `get_history` call.

**Cache staleness — the stale dict is real, and it would sink the fix.**

Revision 2 of this design claimed "no stale dict can hide the new field",
on the strength of this measurement:

```
id of runtime SystemMessage: None
cached entries after serialize: 0
```

Both numbers were real and the conclusion was wrong, for two independent
reasons — a useful reminder that "verified by execution" only certifies
the state actually fed to the probe:

1. The probe read `cache._cache`, an attribute that does not exist
   (`MessageDictCache` declares `__slots__ = ("_by_id", "_hits",
   "_misses")`), so it reported `0` no matter what the cache held.
2. The message was measured **as constructed**. Production reads it back
   **after** `aupdate_state` (`agent_process.py:4493`), whose `messages`
   reducer is `add_messages` — and `add_messages` assigns an `id`,
   mutating the message in place:

```
plain construct: id=None
after add_messages: id='7dcbd9ec-…'   same object mutated: True
```

`_handle_get_history` serialises the persisted message, which therefore
*does* carry an `id` and *is* cached (`serialize` caches iff `id` is a
non-empty string, `:146-160`).

The stale dict then follows, reproduced on a live cache:

```
poll #1 (before fix) keys: ['content','id','type']          misses=1 entries=1
<fix C lands: serialize_message starts emitting the flag>
poll #2 (after fix)  keys: ['content','id','type']          hits=1
SAME OBJECT FROM CACHE: True
STALE DICT SERVED WITHOUT THE FLAG: True
fresh cache (after restart) keys: ['content','id','recoverable_error','type']
```

The practical consequence is exactly the acceptance scenario: an agent
that has already hit the cap and whose history the web has already
polled would keep serving flagless dicts after the fix, until its process
restarts — and restart is itself currently broken (`restart_self` reports
success without replacing the process). The first real check ("here is
`test_lsp`, show me the error") would fail for a reason unrelated to the
fix.

**Chosen strategy: store the version beside the entry**, i.e.
`_by_id: dict[str, tuple[int, dict]]`, with a module-level
`_SERIALIZATION_VERSION` bumped whenever the shape produced by
`serialize_message` changes. A read whose stored version differs counts
as a miss and re-serialises. A shape change therefore misses every
pre-existing entry by construction.

**Not** `(version, message_id)` as the key, although that was the first
proposal. It would silently break `prune_to`: the owner calls
`prune_to(live_ids)` on **every** `get_history` (`agent_process.py:4070`)
with a set of *plain string* ids, and `prune_to` evicts every key not in
that set (`:162-172`). With tuple keys nothing ever matches:

```
keys: [(2, 'id-a'), (2, 'id-b')]   keep_ids: {'id-a', 'id-b'}
evicted 2 of 2  -> cache wiped on EVERY get_history: True
```

No error, no failing test — every answer stays correct and only the hit
rate collapses to zero, re-serialising the whole history on each poll.
That is precisely what this cache exists to prevent
(`DESIGN_CONTEXT_DELTA_CACHE.md` §4.7). Keeping the key a plain id leaves
`prune_to` byte-for-byte unchanged and keeps the defect out of reach.

Rejected: not caching flagged messages (narrow — it fixes this field and
leaves the next shape change to rediscover the same trap); and accepting
"visible only after a restart" (it would make §8 depend on a restart path
we know to be broken).

#### 5.3.3 Then, in the web

* `split_into_turns` keeps the flag when building the `Turn` (a parallel
  `system_error_flags` list, matching the existing `system_timestamps`
  pattern — payload shapes unchanged);
* `_render_assistant_block_final` appends the flagged system messages
  after the assistant text.

Until the flag actually arrives, the only signal available to the web is
the `[runtime]` text prefix. Matching on it as an interim measure is
rejected for the reason given above: it would entrench exactly what this
section refuses.

Matching on the `[runtime]` text prefix instead was rejected: the flag is
structured data put there for this purpose, the prefix is a human string
that can be reworded without anyone noticing the rendering broke.

### 5.4 Consequence for the "no response" case

When a turn has no assistant text at all and only a flagged system
message, `final` currently prints `NO_RESPONSE_TEXT`. After the fix it
prints the failure — strictly more informative, and the case is covered
by a test below.

## 6. Radius

| File | Change | Defect |
|---|---|---|
| `deep_worker_cli/middleware/tool_selection.py` | one entry in `_PROMPT_SECTION_OWNERS` | A |
| `deep_worker_cli/session/session_manager.py` | optional `error_message` kwarg on `set_status` | B |
| `deep_worker_cli/process/agent_process.py` | pass that kwarg on the recoverable-error paths; `clear_error()` when a turn starts | B |
| `deep_worker_cli/process/message_dict_cache.py` | forward `recoverable_error` (whitelisted) | C |
| `deep_worker_web/app/services/dialog_formatter.py` | carry the flag; show flagged systems at `final` | C |
| tests (§7) | new cases | A/B/C |

Not touched: `docs/POLICIES.md` §1, `AgentStatus`, the iteration cap,
`FATAL_EXCEPTION_TYPES`.

## 7. Tests

Each defect gets a test that fails before its fix and passes after.

**A** — `tests/unit_tests/test_tool_selection_middleware.py`
* the `write_todos` marker is pinned to upstream's real constant
  (drift fails the test);
* with selection `none,+serena_*`, the filtered prompt does not contain
  the `write_todos` section while a kept tool's section survives;
* the negative control: with `write_todos` **kept**, its section survives
  — proving the fix binds section to schema rather than deleting
  unconditionally;
* guard: every prompt section contributed by a middleware **in our own
  set** has an owner in `_PROMPT_SECTION_OWNERS`. Deliberately not
  phrased as "every section has an owner" — unowned text (the agent's
  own instructions, user additions) is legitimate, and that phrasing
  would make the test fail on user content.

**B** — `tests/unit_tests/session/` (existing set_status tests)
* `set_status(IDLE)` without the kwarg leaves `error_message` untouched
  (pins the no-op default);
* `set_status(IDLE, error_message=...)` persists it AND keeps the status
  IDLE (not ERROR);
* the `status_change` payload carries it;
* **the reset**: cap → marker set → new turn → `error_message is None`.
  Without this test §4.4's defect returns silently;
* **the placement**: a mid-turn return to RUNNING (the `status_callback`
  path) leaves the marker intact. Without this case the reset test passes
  even when the reset is wired into all four sites, which would wipe the
  marker inside the failing turn.

**C** — `tests/unit_tests/` (cli) + `deep_worker_web/tests/`
* `serialize_message` emits `recoverable_error` for a flagged message and
  omits the key entirely for an ordinary one (no new bytes on the common
  path);
* **cache invalidation**: a message serialised into the cache under the
  old shape is not served after the shape version changes — the probe
  must use a message carrying an `id` (i.e. one that went through
  `add_messages`), because an id-less message is never cached and would
  make the test vacuous;
* **`prune_to` still prunes by plain id**: fill the cache, call
  `prune_to` with the ids of the live messages, and assert the live
  entries **survived**. A regression here breaks no behaviour and fails
  no existing test — it only silences the cache — so nothing else would
  catch it;
* a turn with assistant text + a flagged system message renders the
  failure at `final`;
* a turn with an UNflagged system message (base prompt) still hides it at
  `final` — otherwise the fix would just be "show everything";
* `text` / `full` unchanged.

**Integration** — the cap path end to end: hitting the cap leaves status
IDLE, a non-null `error_message`, and a flagged SystemMessage in history.

## 8. Acceptance

1. `pytest tests/unit_tests/` (deep_worker_cli) green.
2. `pytest deep_worker_web/tests/` green.
3. Replay of the real `test_lsp` message shape at `verbosity="final"`
   shows the error (it currently does not — §5.1).
4. A fresh agent launched with `--tools "none,+serena_*"` asked a
   planning question does not loop to the cap.

(4) requires a process started after the change; per the restart defect
observed earlier today, `restart_self` reports success without replacing
the process, so this must be checked on a genuinely new agent — and the
PID/start-time must be shown, not asserted.

## 9. Order

Defects A, B, C are independent. Commit as three separate commits in one
branch so each can be reverted alone, then FF to master.
