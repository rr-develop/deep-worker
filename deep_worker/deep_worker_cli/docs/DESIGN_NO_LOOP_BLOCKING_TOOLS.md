# DESIGN: No tool/middleware may block the agent's event loop

**Status:** implemented (on `master`).
**Scope:** the deep-worker agent process — every tool execution path and
every middleware hook that runs on the agent's single asyncio event loop.

---

## 1. The hard invariant (project policy)

The agent process is **single-thread, single event loop** (see
`deep_worker/CLAUDE.md`, `REFACTORING_SUMMARY.md`). On top of that:

> **No tool and no middleware hook may perform a blocking call on the
> event-loop thread.** Any heavy or potentially-blocking synchronous work
> (model load, vector DB I/O, subprocess, network, large file I/O) MUST be
> offloaded off the loop via `asyncio.to_thread(...)` / `run_in_executor`,
> or be natively `async`. The loop must keep ticking at all times.

"The agent may `await` a tool" — yes. But the work behind that await must
not run **inline on the loop thread holding the GIL**.

## 2. Why blocking the loop is catastrophic here (the GIL deadlock)

Several of our dependencies run their **own** background event loop in a
**separate thread** and bridge to it synchronously. The clearest example
is **LanceDB** (`lancedb/background_loop.py`):

```python
class BackgroundEventLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever,
                                        name="LanceDBBackgroundEventLoop", daemon=True)
        self.thread.start()
    def run(self, future):
        concurrent_future = asyncio.run_coroutine_threadsafe(future, self.loop)
        return concurrent_future.result()   # ← blocks the CALLING thread
```

If a LanceDB call runs **on our agent's event-loop thread**, that thread
blocks in `.result()` **while holding the GIL**. LanceDB's background loop
needs the GIL to execute the scheduled coroutine — but it can never get it,
because the agent loop thread holds it and is blocked. **Mutual GIL
starvation → permanent deadlock.** The agent's loop never ticks again:
RPyC stops being served, status stays `running` forever, the agent is dead
to the user.

This was proven live (agent_c, 2026-06-15): two threads — the agent
MainThread and `LanceDBBackgroundEventLoop` — were both parked on the
**GIL futex** (`_PyRuntime.ceval.gil.mutex`, address inside the
`_PyRuntime` symbol range), neither able to proceed.

Offloading the blocking call to a worker thread breaks the deadlock: the
worker thread's `.result()` wait **releases the GIL**, so LanceDB's
background loop acquires it, runs the coroutine, and returns. The agent's
own loop keeps ticking the entire time. Verified empirically: the same
`search_history` call run via `run_in_executor` completed in ~10 s while
the event loop kept ticking continuously; run inline it deadlocks.

## 3. The regression

This class of bug has been hit and "fixed" before, but only ever at the
**trigger** level, never at the **architectural** level — so it keeps
coming back through new code paths:

* `70e2515` — "Replace blocking subprocess.run() with async execution to
  prevent event loop freezing" (one specific blocking call).
* `semantic_search/docs/DESIGN_FIX_PROJECT_MEMORY_OFFLINE_MODEL_LOAD.md`
  (2026-05-28) — `project_memory`/`test_v3` hung on a **network** model
  load. That fix forced `local_files_only=True` (removing the network
  trigger) and **explicitly documented the deeper latent vulnerability**:
  "the model load runs **synchronously inside the agent turn**
  (`project_memory` is a sync `@tool`; the turn coroutine awaits it) … no
  timeout." It fixed the network trigger, not the inline-on-loop root.

**Current regression (agent_c):** summarization indexing.
`SummarizationMiddleware.abefore_model` (async, runs on the loop) →
`_aoffload_to_backend` (async) → `self._on_summarized(path)` →
`index_all_history_files(...)` — a **fully synchronous LanceDB write run
inline on the event-loop thread**. New blocking path, same root cause,
same GIL deadlock. The `_on_summarized` callback was wired in without the
offload discipline.

The lesson: removing one trigger (network, this-subprocess) does not
prevent the next inline-blocking path. We need (a) the offload applied at
every such site, with (b) a guard test that fails when a new one appears.

## 4. Violators found (full sweep)

| # | Site | Blocking work | How reached | Severity |
|---|------|---------------|-------------|----------|
| 1 | `deepagents/middleware/summarization.py` `_aoffload_to_backend` → `_on_summarized(path)` | `index_all_history_files` — LanceDB **write** + (cold) embed model | inline in **async** `abefore_model` hook on the loop | **CRITICAL — the live deadlock** |
| 2 | `deep_worker_cli/history_memory_tool.py` `history_memory` | semantic_search → LanceDB/torch | sync `@tool`; relies on LangChain's implicit `run_in_executor` | hardening |
| 3 | `deepagents/tools/project_memory.py` `project_memory` | semantic_search → LanceDB/torch | sync `@tool`; relies on implicit offload | hardening |
| 4 | `deep_worker_cli/history_access_tool.py` `history_range`/`history_at`/`history_slice` | markdown file read | sync `@tool`; relies on implicit offload | low |

Note on #2–#4: LangChain's `BaseTool._arun` already wraps a sync-only
`@tool` in `run_in_executor` when the graph is driven with
`ainvoke`/`astream` (which our agent does), so these are offloaded **today**.
But that protection is **implicit and fragile** — any code path that calls
`tool.invoke()` (sync), or any future refactor, silently reintroduces the
inline execution. Per the hard invariant we make the offload **explicit
and self-documenting** for the heavy ones, and guard it with a test.

#1 has **no** such implicit protection — `_on_summarized` is a plain
callback invoked directly inside an async method — so it is the one that
actually deadlocked and is fixed first.

## 5. Fix

### 5.1 Critical — offload the summarization index write (#1)

In `_aoffload_to_backend`, invoke the `_on_summarized` callback via
`await asyncio.to_thread(self._on_summarized, path)` instead of calling it
inline. This moves `index_all_history_files` (LanceDB write) off the loop
thread; the loop keeps ticking and the LanceDB background loop can acquire
the GIL. The sync sibling `_offload_to_backend` (used only by the
sync-graph `before_model` path, which is not the agent's loop) keeps the
direct call.

### 5.2 Hardening — explicit offload for the heavy semantic-search tools (#2, #3)

Give `history_memory` and `project_memory` an explicit async path
(`coroutine=`) that runs the heavy sync body via `asyncio.to_thread`, so
the offload is guaranteed and visible, not dependent on LangChain
internals. Light file-read tools (#4) are left as sync `@tool` (their work
is small and they carry the standard banner).

### 5.3 Strict anti-regression comments

Every offload site gets a banner stating: this call hits LanceDB / a
background-event-loop library; running it inline on the agent loop
deadlocks the GIL; DO NOT remove the `to_thread`/`run_in_executor`; see
this document and the regression test.

## 6. Regression guard (the protective test)

`tests/test_no_loop_blocking_summarization.py` (no mocks of the loop):

* Drives a real event loop that ticks a counter every 50 ms.
* Triggers the exact production path — the `_on_summarized` callback doing
  a real `index_all_history_files` over a real (temp) summaries dir with a
  real LanceDB write — the way `_aoffload_to_backend` does.
* Asserts the loop kept ticking (no gap > a threshold) **during** the
  indexing — i.e. the work was offloaded, not run inline.
* A regression (callback called inline again) makes the loop stall for the
  whole index duration → the max-gap assertion fails.

Plus a fast **static guard** `tests/test_tools_do_not_block_loop.py`:
asserts the heavy tools expose an async/offloaded path (a `coroutine`) and
that `_aoffload_to_backend` does not call `_on_summarized` inline (source
check), so the guard fires even without the heavy deps installed.

## 7. Files

* `deepagents/middleware/summarization.py` — offload `_on_summarized` in
  `_aoffload_to_backend` (#1) + banner.
* `deep_worker_cli/history_memory_tool.py`,
  `deepagents/tools/project_memory.py` — explicit offloaded async path +
  banner (#2, #3).
* `deep_worker_cli/tests/test_no_loop_blocking_summarization.py` — runtime
  loop-liveness regression test.
* `deep_worker_cli/tests/test_tools_do_not_block_loop.py` — static guard.
* this document.
