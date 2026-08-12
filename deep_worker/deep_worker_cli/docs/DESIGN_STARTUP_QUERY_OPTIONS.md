# Design: `--sum-query`, `--sum-query-file`, `-q/--query`, `--query-file` startup options

Status: Draft
Date: 2026-07-02
Owner: (internal agent)
Reviewer: (internal review agent)
Worktree: a local development checkout (branch `dev`)

## 1. Problem

Today `deep-worker <name> [--restart]` starts an agent and leaves it in
`IDLE` waiting for the first user message. The only way to inject a
starting user query is to launch the agent, then send it a message via
the CLI / `send_message_to_agent` / web UI. Two use cases are painful:

1. **Scripted startup with a first task.** Running a headless agent that
   should immediately start executing a specific instruction with no
   human in the loop.
2. **Fresh-context prompt after work on a different topic.** The user
   wants the agent to *forget* the current thread and start a new task
   from a clean slate — but keep the evicted history on disk for later
   inspection or for the agent's `history_memory` tool.

Both are already partially expressible: (1) is close to what
`_auto_continue_after_restart` does for the restart-notice path, and (2)
is close to what `SummarizationMiddleware` does when it hits a token
threshold. This design ties them into first-class CLI options.

## 2. Goals

1. Add four new CLI options to `deep-worker`:
   - `--sum-query <text>` — starting user query treated as if it were the
     summary message emitted by `SummarizationMiddleware`.
   - `--sum-query-file <path>` — same, content read from a UTF-8 file.
   - `-q, --query <text>` — plain starting user query, no offload.
   - `--query-file <path>` — same, content read from a UTF-8 file.
2. All four work on `deep-worker <name>` (new agent), `deep-worker
   <name> --restart` (restart existing), and `deep-worker <name> -q ...`
   for an already-running agent (send-through path).
3. The agent starts executing the query immediately after
   initialisation — no polling, no external nudge.
4. When `--sum-query*` is given, the *pre-existing* thread history is
   appended to `summaries/<agent>.jsonl` in the per-agent data directory
   using the exact same lossless JSONL format written by
   `SummarizationMiddleware`.
   The active context is then wiped and replaced with a summary-shape
   `HumanMessage` carrying the sum-query text.
5. Both flags may be used together in the same invocation. The
   combined semantics is TWO SEQUENTIAL agent turns: the agent first
   EXECUTES the sum-query (a full turn on the freshly-installed
   summary-shape state) and then EXECUTES the plain query as the next
   user message — "first comes the sum-query request, then immediately
   the query request". See §4.6. (Corrected 2026-07-04: the earlier
   "single turn, two messages" reading executed only the query and
   never ran the sum-query as its own request — a contract violation.)
6. Fail loud on any error condition. Missing file, empty text, offload
   failure — all surface as user-visible errors before / instead of
   silent no-op.

## 3. Non-goals

1. Do NOT persist `startup_query` / `startup_sum_query` into
   `launch_config`. They are single-shot per invocation. A subsequent
   `deep-worker <name> --restart` MUST NOT replay them. If the user
   wants the same startup query again, they pass it again explicitly.
2. Do NOT retry on transient failures inside this feature — errors are
   surfaced to the user or set `AgentStatus.ERROR` per the existing
   fatal-error contract.
3. Do NOT chain more than two queries per invocation. The design
   accepts at most one sum-query and one plain query per launch.
4. Do NOT change JSONL history format, summary message shape,
   `SummarizationMiddleware` public API contract for existing callers,
   or `--restart` semantics.
5. Do NOT expose separate endpoints for these queries; extend the
   existing invoke command dict (see §4.5).

## 4. Design

### 4.1 CLI parsing

Four new options on the `deep-worker` parser (`agent_command.py::
create_parser`), all `default=None`:

```
--sum-query <TEXT>
--sum-query-file <PATH>
-q, --query <TEXT>
--query-file <PATH>
```

**Mutual exclusion**: `--sum-query` xor `--sum-query-file`; `--query` xor
`--query-file`. `argparse` `add_mutually_exclusive_group()` enforces
each pair. The two pairs are independently combinable.

**File reading and validation** happen in a new helper
`resolve_startup_queries(args) -> tuple[str | None, str | None]`
called BEFORE any daemonization / heavy setup so failures surface on
stderr with a non-zero exit:

- File not found / not readable → `Error: cannot read <path>: <cause>` → exit 2.
- File contents empty or whitespace-only → `Error: <path> is empty` →
  exit 2.
- Text value directly given and empty / whitespace-only → same hard-fail
  (`--query ""`).
- Returns `(sum_query_or_None, query_or_None)`.

Persistence policy: neither value is written to `launch_config` — see
§3.1. `build_launch_config()` is not modified.

### 4.2 AgentProcess plumbing

`AgentProcess.__init__` gains two kwargs, both `default=None`:

```python
startup_query: str | None = None,
startup_sum_query: str | None = None,
```

Stored on `self._pending_startup_query` and
`self._pending_startup_sum_query`. Neither is ever persisted to disk.
Neither survives a process restart (fresh CLI invocation is required to
re-arm).

### 4.3 Startup hook (new agent / --restart path)

Inside `AgentProcess.start()`, after `create_cli_agent` returns and
after `_prewarm_client_manager` (so tool inventory is authoritative), a
new hook fires — symmetric to `_auto_continue_after_restart`:

```python
if self._pending_startup_query is not None or self._pending_startup_sum_query is not None:
    asyncio.create_task(self._auto_run_startup_query())
```

The task is created after `self.running = True` and after any pending
restart-notice task, so ordering is: restart-notice invoke (if any)
completes on its own path; then the startup-query task runs.

`_auto_run_startup_query()` mirrors `_auto_continue_after_restart`:
wraps a single `_handle_invoke` in a `try/except`, logs failures, never
crashes the agent process — but see §4.6 for how errors during the
sum-query offload become `HistoryOffloadError` (fatal, agent → ERROR
via the existing catch-all in `_handle_invoke`).

### 4.4 Semantics: sum-query offload

The pre-summarization offload is a NEW public helper on
`SummarizationMiddleware`. It is a **pure producer**: it evicts the
supplied messages to JSONL and returns the replacement message list +
path. It does NOT mutate any graph/checkpointer state itself — the
single wipe of the active context is performed by the caller via the
compiled graph's `aupdate_state` (see §4.6, Blocker-2 resolution).

```python
async def apply_pseudo_summary(
    self,
    thread_id: str,
    summary_text: str,
    current_messages: list[AnyMessage],
) -> tuple[list[AnyMessage], str | None]:
    """Evict ``current_messages`` to JSONL history and produce the
    replacement summary-shape HumanMessage list.

    Pure producer: returns the new messages + path. Does NOT write to
    the checkpointer and does NOT emit RemoveMessage — the caller owns
    the state wipe (see §4.6).

    Args:
      thread_id: The thread whose JSONL file the offload appends to.
        The caller passes ``self.name`` (matching the real invoke's
        checkpointer thread_id, agent_process.py:2901) so the history
        lands in ``<agent>.jsonl``. Threaded explicitly to the offload
        helpers instead of relying on langgraph's config contextvar,
        which is not populated outside a graph run (Blocker-1).
      summary_text: Text for the summary-shape HumanMessage.
      current_messages: The active-context messages to evict.

    Returns:
      new_messages: exactly one HumanMessage carrying summary_text
        with additional_kwargs={"lc_source": "summarization"}. Format
        matches _build_new_messages_with_path — with JSONL reference
        line when the offload actually wrote, without it when the
        message list was empty (see below).
      file_path: JSONL path when a write occurred, else None.

    Delegates to existing private helpers, with one additive change:
      _filter_summary_messages (unchanged), _aoffload_to_backend
      (gains an additive ``thread_id`` param — see Blocker-1),
      _build_new_messages_with_path (unchanged).

    Raises HistoryOffloadError on write failure — same fail-loud
    contract as SummarizationMiddleware.before_model / abefore_model
    (ae16f7c).
    """
```

**Delegation invariant**: this method MUST NOT reimplement serialize,
write, edit, or summary-message building. It orchestrates the existing
private helpers. The ONE additive change to those helpers is an explicit
`thread_id` override (Blocker-1 below); everything else — serialization,
read-existing, append, fail-loud, summary-message building — stays in the
helpers untouched. This is "delegates WITH an additive `thread_id`
override", chosen deliberately over the earlier contextvar approach (see
Blocker-1 for why the contextvar path was rejected).

**Blocker-1 resolution — offload writes to the correct `<agent>.jsonl`
(explicit `thread_id` override, reviewer option B).** `_offload_to_backend`
/ `_aoffload_to_backend` derive the JSONL path from `_get_history_path()`
→ `_get_thread_id()` → langgraph's `get_config()` contextvar
(`summarization.py:388,396,364-386`). That contextvar (`langgraph.config`
reads `var_child_runnable_config`) is populated ONLY while a LangGraph
runnable is executing. `apply_pseudo_summary` is invoked from
`_auto_run_startup_query`, OUTSIDE any `astream` run, so `_get_thread_id()`
falls back to a random `session_<uuid8>` id and writes to the wrong file.

The earlier draft tried to fix this by wrapping the call in
`langchain_core.runnables.config.set_config_context(...)`. **That does
NOT work** — verified empirically: `set_config_context` sets
`var_child_runnable_config` inside a *copied* context (`ctx =
copy_context(); ctx.run(_set_config_context, config); yield ctx`) that
the `with`-body never enters, so `langgraph.config.get_config()` inside
the body still raises `RuntimeError: Called get_config outside of a
runnable context` and the fallback id is used. Rejected.

Instead, the offload helpers accept an **additive, optional `thread_id`
override** (default `None` → existing contextvar behaviour, so all
current callers — `before_model` / `abefore_model` — are byte-for-byte
unaffected):

```python
def _get_history_path(self, thread_id: str | None = None) -> str:
    tid = thread_id if thread_id is not None else self._get_thread_id()
    return f"{self._history_path_prefix}/{tid}.jsonl"

# _offload_to_backend / _aoffload_to_backend gain the same optional
# param and forward it:  path = self._get_history_path(thread_id)
```

`apply_pseudo_summary(thread_id=self.name, ...)` then forwards
`thread_id` explicitly to `_aoffload_to_backend`, so the offload appends
to `summaries/<agent>.jsonl` in the per-agent data directory — exactly
where `history_memory` / `dw-hist` look — with NO dependence on the
langgraph
contextvar and NO cross-package coupling to
`var_child_runnable_config`. The caller does not manipulate any
contextvar; the path is a plain argument, so this is fully testable
without a graph. Existing summarization offloads (which DO run inside a
graph and pass no `thread_id`) keep resolving via `_get_thread_id()`
exactly as before.

**Backend resolution invariant (Gap-4).** `apply_pseudo_summary` needs a
`BackendProtocol` to hand to `_aoffload_to_backend`. Outside a graph run
there is no `state`/`Runtime`, so `_get_backend(state, runtime)` cannot
be used. The helper therefore requires that the middleware was
constructed with a **concrete backend instance**, not a factory:

```python
if not isinstance(self._backend, BackendProtocol):
    raise RuntimeError(
        "apply_pseudo_summary requires SummarizationMiddleware to hold a "
        "concrete backend instance; factory backends need a ToolRuntime "
        "that is unavailable outside a graph run."
    )
backend = self._backend
```

This is a hard-fail (not a silent fallback). In the deep-worker build
the middleware is always constructed with a concrete filesystem backend
instance (graph.py), so the factory branch is unreachable in production;
the guard exists to make the out-of-graph contract explicit and to fail
loud if a future build wires a factory backend.

**Empty-history invariant** (per reviewer feedback #2): after
`_filter_summary_messages(current_messages)` returns an empty list, the
helper MUST skip `_aoffload_to_backend` entirely and MUST build the
summary-shape HumanMessage WITHOUT a JSONL reference in its content
(no dangling file link), returning `file_path=None`. No empty JSONL
section is written. This is also the correct behaviour for `--clean` +
`--sum-query`: after clean the checkpointer has no messages, so
`current_messages` is empty, nothing goes to disk, and the summary-shape
HumanMessage is the only content of the new state. See §4.6 Blocker-3
for the ordering guarantee that makes "after clean the checkpointer has
no messages" actually true at read time.

`AgentProcess._auto_run_startup_query()` needs access to the
`SummarizationMiddleware` instance embedded in the compiled deep-agent
graph. It walks `self.agent`'s middleware list to find the instance
(`isinstance(m, SummarizationMiddleware)`), same pattern used by
existing code that pokes at graph middleware. If not found, sum-query
fallback is a hard error at startup (this is a broken deep-agent build,
not a runtime edge — the caller can only recover by fixing the graph).

### 4.5 Idle-agent send-through

For an already-running agent, `deep-worker <name> -q ...` / `--sum-query
...` (without `--restart`) detects the running process via
`ProcessRegistry` and, instead of spawning, dispatches an RPyC call.

Per reviewer feedback #4, the existing `exposed_invoke_agent` command
dict is extended with two new optional keys, `sum_query` and `query`.
No new endpoint. When either key is present, the handler runs the SAME
in-process sequence documented in §4.4/§4.6 — it does not fork a second
copy of the offload/wipe logic:

- If `sum_query` present: read `current_messages` via
  `aget_state({"configurable":{"thread_id": self.name}})`, run
  `apply_pseudo_summary(self.name, sum_query, current_messages)` — the
  explicit `thread_id` argument targets the offload at `<agent>.jsonl`
  (Blocker-1, option B), then wipe via `aupdate_state` with
  `RemoveMessage(REMOVE_ALL_MESSAGES)` + summary-shape (Blocker-2),
  then feed the single-turn input described in §4.6.
- If only `query` present: pass through as a normal HumanMessage (no
  wipe, no offload).

Concretely, the extended `exposed_invoke_agent` normalises `sum_query`/
`query` into the same `_auto_run_startup_query`-style path (a shared
private helper, e.g. `_run_startup_or_send_query`, invoked both from the
startup hook and from the RPyC handler) so there is ONE implementation
of the offload→wipe→invoke sequence. This keeps auth / streaming /
status transitions on the one existing `_handle_invoke_owned` path and
guarantees the idle and startup routes cannot drift.

### 4.6 Combined sum-query + query — TWO SEQUENTIAL invokes

**Corrected 2026-07-04.** The user contract is explicit: "first comes the
sum-query request, then immediately the query request" — the sum-query and
the query are TWO SEQUENTIAL user requests, each EXECUTED as its own turn.
An
earlier revision of this section (then-"reviewer feedback #1") collapsed
them into ONE turn whose input was `[summary-shape, query]`. That was
wrong: in a single turn the agent responds to the *query* with the
summary as prior context, so the **sum-query is never executed as its own
request**. Downstream consumers (e.g. my_job) that expect the sum-query to
run first and produce its own result silently got nothing. This is the bug
being fixed.

The correct sequence, all under the one `_invoke_lock` critical section:

1. Offload the pre-existing history to `<agent>.jsonl` (pure producer).
2. Wipe + install exactly `[summary-shape HumanMessage]` via
   `aupdate_state` (RemoveMessage(REMOVE_ALL_MESSAGES) + summary-shape).
3. **Turn 1 — execute the sum-query.** `_handle_invoke_owned` with an
   EMPTY `message_list` re-drives the graph on the installed
   `[summary-shape]` state: the agent responds to the summary-shape
   message as its first user request, producing an AI response.
4. **Turn 2 — execute the plain query.** `_handle_invoke_owned` with
   `[plain HumanMessage]` appends the query on top of turn 1's result
   (summary-shape + AI response) and runs it as the next user request.

Each of steps 3 and 4 is a distinct `_handle_invoke_owned` call. When
only ONE of the two is supplied, exactly one turn runs (sum-query-only →
step 3 with empty list over the installed state; query-only → step 4,
no offload/wipe). Properties:

- Failure of the offload aborts the whole thing atomically (via
  `HistoryOffloadError` → runtime catch-all → `AgentStatus.ERROR`),
  BEFORE any wipe is committed (offload precedes wipe — see below) and
  BEFORE either turn runs.
- Both turns hold the same `_invoke_lock` we acquired for the sequence,
  so no concurrent normal turn can interleave between them.
- Ordering (wipe → turn 1 → turn 2) is guaranteed structurally by the
  await sequence, not by timing.

**Blocker-2 resolution — one explicit wipe mechanism, owned by the
caller.** There is exactly one way the active context is emptied, and it
is NOT inside `apply_pseudo_summary`. The `before_model` summarization
path wipes by *returning* `{"messages":[RemoveMessage(REMOVE_ALL_MESSAGES),
...]}` and letting the graph apply it; that mechanism only works while
the graph is running a node. `_auto_run_startup_query` runs OUTSIDE the
graph, so it uses the compiled graph's public `aupdate_state` API to
apply the identical reducer op against the same `thread_id`:

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage
await self.agent.aupdate_state(
    config,                                    # {"configurable": {"thread_id": self.name}}
    {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_msgs]},
)
```

`aupdate_state` writes a new checkpoint whose message channel, after the
`add_messages` reducer processes `REMOVE_ALL_MESSAGES`, contains exactly
`new_msgs` (the single summary-shape HumanMessage). The subsequent
invoke's `add_messages` then appends the plain query on top. This is the
standard langgraph state-update API — no new mutation path is invented;
it reuses the same `REMOVE_ALL_MESSAGES` sentinel the summarization
middleware already relies on, just applied from outside the graph. The
plain-query-only path (`sum_query is None`) performs NO `aupdate_state`
and NO wipe — it simply appends a HumanMessage over the existing
checkpointed history, matching today's `_build_invoke_messages`
behaviour.

**current_messages source.** The messages handed to
`apply_pseudo_summary` are read from the checkpointer for
`thread_id == self.name` via the compiled graph's `aget_state(config)`
(`state.values["messages"]`, empty list if the thread has no
checkpoint). This reads the SAME state the wipe will replace, so the
offload and the wipe are consistent by construction.

**Blocker-3 resolution — deterministic `--clean` + `--sum-query`
ordering (no race).** `--clean` (DB-only variant) must complete
**synchronously before** `current_messages` is read. The startup path
therefore awaits `clean_agent_history_db_only(self.name)` to completion
(not as a background task) prior to scheduling / entering
`_auto_run_startup_query` whenever both `--clean` and `--sum-query` are
present. Sequence:

1. `await clean_agent_history_db_only(self.name)` — SQLite rows for the
   thread physically removed; the compiled graph's `aget_state` for
   `self.name` now returns no messages.
2. `_auto_run_startup_query` reads `current_messages` → empty →
   `apply_pseudo_summary` hits the empty-history invariant (§4.4): skip
   offload, no JSONL reference, `file_path=None`.
3. `aupdate_state` installs `[summary-shape]`; invoke appends the query.

This removes the previously-documented race where a backgrounded clean
could run after the read. If implementing the synchronous ordering
inside `start()` is not feasible for the `--clean` code path, the
fallback is an argparse-level mutual exclusion (`--clean` +
`--sum-query` rejected with exit 2); the design PREFERS the synchronous
ordering and only falls back to rejection if the clean sequencing cannot
be made deterministic. Either way the outcome is specified — never
"rely on after clean it's empty" without a guarantee.

Implementation shape:

```python
async def _auto_run_startup_query(self) -> None:
    sum_query = self._pending_startup_sum_query
    query = self._pending_startup_query
    self._pending_startup_sum_query = None
    self._pending_startup_query = None

    config: RunnableConfig = {"configurable": {"thread_id": self.name}}

    if sum_query is not None:
        sm = self._find_summarization_middleware()   # hard-fail if absent (§4.4)
        state = await self.agent.aget_state(config)
        current_messages = state.values.get("messages", []) if state else []
        # Offload targets thread_id==self.name via the explicit thread_id
        # override so the JSONL lands in <agent>.jsonl (§4.4 Blocker-1,
        # option B — no contextvar manipulation). apply_pseudo_summary is a
        # pure producer — it does NOT wipe.
        new_msgs, _ = await sm.apply_pseudo_summary(self.name, sum_query, current_messages)
        # Caller owns the single wipe (§4.6 Blocker-2): replace the whole
        # message channel with exactly the summary-shape message.
        await self.agent.aupdate_state(
            config,
            {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_msgs]},
        )

    # TWO SEQUENTIAL turns (§4.6). Turn 1 executes the sum-query (empty
    # message_list re-drives the graph on the installed [summary-shape]).
    # Turn 2 executes the plain query as the next user message. When only
    # one is set, exactly one turn runs.
    if sum_query is not None:
        await self._handle_invoke(
            {"message_list": [], "stream": True, "reply_back": False}, ...
        )
    if query is not None:
        await self._handle_invoke(
            {
                "message_list": [HumanMessage(content=query, id=f"user-{uuid4().hex}")],
                "stream": True,
                "reply_back": False,
            },
            ...
        )
```

(The real implementation lives in the shared `_run_startup_or_send_query`
and calls `_handle_invoke_owned` because the whole sequence already holds
`_invoke_lock` — see §4.5 / the code. The shape above elides that for
readability.)

`_handle_invoke` currently takes a single `message: str`; a new
`message_list: list[AnyMessage]` path is added and used only by this
helper (`message` remains the default). Semantics of the new path:

- Non-empty `message_list` → those messages are the invoke input
  (`astream({"messages": message_list}, ...)`); `add_messages` appends
  them onto whatever the checkpointer already holds.
- Empty `message_list` **with a sum-query having just installed the
  summary-shape message via `aupdate_state`** → the invoke drives the
  graph from the checkpointed state directly. Because `astream` requires
  at least a state entry, this case passes `{"messages": []}` (a no-op
  for `add_messages`) so the model runs against the already-installed
  `[summary-shape]` context. This preserves the "sum-query alone still
  starts the agent working" goal (§2.3).

This is a private-shape change inside `agent_process.py`, not a public
RPyC contract change. The RUNNING status transition, notify_callback
plumbing, and catch-all error handling of `_handle_invoke_owned` are
reused unchanged — the summary-shape message's first-message
`content`/`role` for `notify_callback` is derived from
`message_list[0]` when present, else from a short synthetic
"(starting from summarized context)" notice so subscribers still see a
user-turn marker.

### 4.7 Error handling

| Failure mode                                                | Behaviour                                                                 |
|-------------------------------------------------------------|---------------------------------------------------------------------------|
| `--sum-query-file`/`--query-file` unreadable                | Exit 2 on stderr BEFORE daemonize.                                        |
| File content empty / whitespace-only                        | Exit 2 on stderr BEFORE daemonize.                                        |
| Both `--sum-query` and `--sum-query-file` given             | argparse mutually-exclusive error, exit 2.                                |
| Both `-q/--query` and `--query-file` given                  | argparse mutually-exclusive error, exit 2.                                |
| `apply_pseudo_summary` backend write fails                  | `HistoryOffloadError` → runtime catch-all → `AgentStatus.ERROR`. Raised BEFORE `aupdate_state`, so no wipe is committed on failure. |
| Deep-agent graph has no `SummarizationMiddleware` instance  | `RuntimeError` at `_auto_run_startup_query` startup — user-visible error. |
| Middleware holds a factory backend, not a concrete instance | `RuntimeError` from `apply_pseudo_summary` (Gap-4 guard) → catch-all → `AgentStatus.ERROR`. Unreachable in the production build. |
| `aupdate_state` wipe fails (checkpointer write error)       | Exception propagates through `_auto_run_startup_query` into the runtime catch-all → `AgentStatus.ERROR`. Offload already succeeded, so history is preserved on disk; no partial invoke runs. |
| `--clean` + `--sum-query` both given                        | Deterministic ordering: `clean_agent_history_db_only(self.name)` awaited to completion before `current_messages` read (§4.6 Blocker-3). Empty-history invariant then applies. Fallback: argparse exit 2 if synchronous ordering is infeasible. |
| Idle-agent send-through when agent is not actually idle     | The ENTIRE offload→wipe→invoke sequence runs under `_invoke_lock` as one critical section (`_run_startup_or_send_query` acquires the lock and calls `_handle_invoke_owned` directly, since `_handle_invoke` would re-acquire the non-reentrant lock → deadlock). So it is serialised against any concurrent normal turn — no stale `aget_state` read, no wipe clobbering a concurrent append. |

## 5. Alternatives considered

- **Re-use `--import-claude` flow.** Rejected: import-claude replaces
  the entire history with a transcript, does not carry a starting
  query, and does not fit the "pseudo-summary" semantics.
- **Offload logic inlined in `agent_process.py`.** Rejected — forks
  serialize/write logic away from `SummarizationMiddleware`,
  duplicates the fail-loud contract, drifts on the next JSONL format
  change.
- **Two sequential `_handle_invoke` calls for sum-query + query.**
  Rejected — see §4.6.
- **New RPyC endpoint `exposed_start_with_query`.** Rejected — see
  §4.5, redundant surface area.
- **Persist startup queries into `launch_config`.** Rejected — see
  §3.1, would replay on every `--restart` and clobber human control.

## 6. TDD plan

Failing tests written BEFORE the implementation:

### 6.1 CLI parsing (unit)
- `test_startup_query_parses_sum_query_text`
- `test_startup_query_parses_sum_query_file` (tmp_path fixture)
- `test_startup_query_parses_plain_query_text` and `-q` short form
- `test_startup_query_parses_query_file`
- `test_startup_query_mutex_sum_query_pair_rejected`
- `test_startup_query_mutex_query_pair_rejected`
- `test_startup_query_pair_combination_allowed`
- `test_startup_query_file_missing_fails`
- `test_startup_query_file_empty_fails`
- `test_startup_query_text_whitespace_only_fails`
- `test_startup_query_not_persisted_to_launch_config`

### 6.2 SummarizationMiddleware.apply_pseudo_summary (unit)
- `test_apply_pseudo_summary_offloads_and_returns_summary_shape_message`
- `test_apply_pseudo_summary_delegates_to_offload_to_backend` (via spy)
- `test_apply_pseudo_summary_empty_history_skips_offload_no_reference_in_summary`
- `test_apply_pseudo_summary_raises_history_offload_error_on_backend_failure`
- `test_apply_pseudo_summary_filters_previous_summary_messages`
- `test_apply_pseudo_summary_offload_path_uses_explicit_thread_id`
  (Blocker-1 option B: call `apply_pseudo_summary("agent7", ...)` with NO
  graph / NO config-context, assert JSONL written to `.../agent7.jsonl`,
  NOT `session_<uuid>.jsonl`)
- `test_offload_helpers_default_thread_id_none_preserves_contextvar_behaviour`
  (Blocker-1 regression guard: `_get_history_path()` / `_offload_to_backend`
  / `_aoffload_to_backend` with `thread_id=None` still resolve via
  `_get_thread_id()` — existing `before_model` callers unaffected)
- `test_apply_pseudo_summary_hard_fails_when_backend_is_factory` (Gap-4:
  middleware built with a factory backend → `RuntimeError`)
- `test_apply_pseudo_summary_does_not_emit_remove_message` (Blocker-2:
  returned `new_messages` contains NO `RemoveMessage` — the helper is a
  pure producer; wipe is the caller's job)
- async analogues for the above where relevant

### 6.3 AgentProcess startup hook (unit / focused)
- `test_auto_run_startup_query_runs_when_query_only`
- `test_auto_run_startup_query_runs_when_sum_query_only`
- `test_auto_run_startup_query_runs_when_both`
- `test_auto_run_startup_query_single_invoke_two_messages`
- `test_auto_run_startup_query_first_message_is_summary_shape_second_is_plain`
  (Gap-5: assert `invoke_input[0].additional_kwargs["lc_source"] ==
  "summarization"` AND `"lc_source" not in invoke_input[1].additional_kwargs`,
  so a later real summarization does not filter the query via
  `_is_summary_message`)
- `test_auto_run_startup_query_wipes_via_aupdate_state_before_invoke`
  (Blocker-2: `aupdate_state` called with
  `RemoveMessage(id=REMOVE_ALL_MESSAGES)` + summary-shape, against
  `thread_id == self.name`, BEFORE `astream`)
- `test_auto_run_startup_query_reads_current_messages_from_aget_state`
  (current_messages source is the checkpointer for `self.name`)
- `test_auto_run_startup_query_offload_failure_skips_wipe`
  (Blocker-2 atomicity: `HistoryOffloadError` from offload → no
  `aupdate_state` call, agent → ERROR)
- `test_auto_run_startup_query_clean_then_sumquery_reads_empty_history`
  (Blocker-3: clean awaited to completion before read → offload skipped,
  no JSONL, summary-shape has no file reference)
- `test_auto_run_startup_query_clears_pending_fields`
- `test_auto_run_startup_query_no_op_when_neither_set`
- `test_auto_run_startup_query_error_from_offload_transitions_agent_to_error`

### 6.4 Integration (no mocks, real filesystem backend)
- `test_startup_sum_query_appends_to_jsonl_and_wipes_active_context`
  (assert the offload file is exactly
  `summaries/<agent>.jsonl` in the per-agent data directory — Blocker-1 —
  NOT a `session_<uuid>.jsonl`; assert post-invoke `aget_state` messages
  start
  with the summary-shape message and do NOT contain any pre-existing
  message — Blocker-2 wipe verified end-to-end)
- `test_startup_query_only_adds_single_human_message_to_history`
- `test_startup_both_produces_single_turn_with_two_human_messages`
- `test_startup_clean_plus_sum_query_writes_no_jsonl_and_no_reference`
  (Blocker-3: seed a thread, run `--clean` + `--sum-query`, assert no
  JSONL section written and the summary-shape message has no file link)
- `test_idle_agent_send_through_invoke_dict_extension` (running agent,
  send `sum_query`+`query`, assert JSONL is `<agent>.jsonl` and state
  wiped-then-replaced exactly as the startup path)

## 7. Compatibility

- Existing callers of `SummarizationMiddleware` untouched. New public
  method is additive.
- Existing `_handle_invoke` contract untouched. `message_list` is an
  additive alternative to `message`; when both provided, `message_list`
  wins.
- Existing `exposed_invoke_agent` callers untouched. Extra dict keys
  are ignored by older command dicts.
- `--restart` behaviour unchanged: startup query fields are one-shot and
  not stored in `launch_config`.
- All existing tests must keep passing.

## 8. Roll-out

1. Land RED unit tests (§6.1 – §6.3).
2. Implement CLI parsing + helper + AgentProcess plumbing.
3. Implement `apply_pseudo_summary` + `message_list` in `_handle_invoke`.
4. Add integration tests (§6.4).
5. Full unit + integration suite green, internal review, then
   commit on `dev`, FF into `master`.

## 9. Open questions

None outstanding after the pre-review feedback round.
