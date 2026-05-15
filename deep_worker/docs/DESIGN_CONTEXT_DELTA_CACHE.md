# Design: delta-based context cache for deep_worker agents

**Status:** Phases 1, 2, 3 implemented (2026-04-22); Phase 4 deferred.
**Author:** context-analysis session, 2026-04-22
**Scope:** `DeltaSqliteCheckpointer`, `_handle_get_*` in `agent_process.py`, all
`before_agent` / `before_model` middleware in `deepagents/middleware/*`, RPyC
serialization layer in `rpyc_service/agent_service.py`.
**Does NOT touch:** LangGraph internals, the web UI's polling cadence.

## Implementation status

| Phase | What | Status | Tests |
|-------|------|--------|-------|
| 1 | Checkpointer cache (cache-aware `aget_tuple`, `aput`-populates-cache, listeners, feature flag, stats) | **Done** | `tests/unit_tests/test_delta_checkpointer_cache.py` — 11 tests |
| 2 | Middleware caches: `PatchToolCallsMiddleware` returns `None` when no patching needed + high-water mark; `SummarizationMiddleware._cached_total_tokens`; `MemoryMiddleware` / `SkillsMiddleware` already short-circuit via state presence (no change needed) | **Done** | `deepagents/tests/unit_tests/middleware/test_patch_tool_calls_cache.py` — 4 tests; `test_summarization_cache.py` — 5 tests |
| 3 | Message-dict cache in `_handle_get_history` via `MessageDictCache` (by-id memoization, prune on summarization) | **Done** | `tests/unit_tests/test_message_dict_cache.py` — 10 tests |
| 4 | `alist` batching (single SQL query + single-pass delta application) | **Deferred** — not on the critical path for the observed CPU hotspot; revisit only if benchmarks show `alist` is a problem |

---

## 1. Problem statement

A zombie agent was observed consuming ~20 % CPU and
`rchar = 37 GB` over 39 min while sitting idle — no new messages, no disk
writes, only the two `SELECT *` rebuilds triggered by the web UI's 5-second status
polls and one-off history requests. Empirical amplification was
**≈ 150 × rchar / TCP-out**: every kilobyte sent over RPyC cost ~150 kB of
internal SQLite re-reads.

Root cause is a chain of places that **rebuild state from scratch on every
request** when the state has not changed at all, or has changed by exactly one
appended message. The existing delta-checkpointer stores deltas on the write
path correctly, but **reads** still walk the entire parent chain and
deserialize every ancestor, merge `channel_values`, and hand the fully
materialized state back to LangGraph — which also does not cache it.

This burns CPU, blocks the asyncio event loop, and produces log storms. In the
failure mode from 2026-04-22 (an EOF-spam incident in the RPyC protocol
layer) it also
amplified a single disconnect into hundreds of thousands of log lines.

The codebase **must** behave as follows:

> If nobody wrote to the thread since the last read, the read MUST hit an
> in-memory cache and issue **zero** SQLite queries. If a single delta
> checkpoint was appended, the read MUST apply that one delta on top of the
> cached state, not rebuild from the chain root.

Everything in this document follows from that rule.

---

## 2. Inventory of rebuild sites

Numbers below come from the live investigation (see accompanying research
notes). Every "rebuild" here means an operation whose cost scales with the
size of the conversation rather than with the size of the change.

### 2.1 `DeltaSqliteCheckpointer.aget_tuple`

`deep_worker_cli/delta_checkpointer.py:124-203`

- Called by LangGraph **before every graph node** during
  `astream` / `ainvoke` (via `aget_state`), and by `_handle_get_history`
  / `_handle_get_last_response` on every RPyC request.
- Fetches one row, then unconditionally calls `_reconstruct_checkpoint`
  which walks the parent chain.
- The already-existing `self._state_cache` (line 102) is **not consulted on
  this path** — it is only written here, and only by `_get_prev_channel_values`
  and `aput`.

### 2.2 `DeltaSqliteCheckpointer._reconstruct_checkpoint`

`deep_worker_cli/delta_checkpointer.py:205-289`

- Walks parent chain one `SELECT` at a time until it hits a v1 checkpoint or
  an explicit full snapshot.
- Deserializes every ancestor blob with `serde.loads_typed`.
- Builds a fresh `full_channel_values` dict by re-applying every delta from
  root to leaf.
- Chain depth is unbounded in practice — the only snapshot trigger is
  summarization or `snapshot_interval` (disabled by default).
- **Every call** does O(depth) work regardless of whether we just did it a
  moment ago for the same `checkpoint_id`.

### 2.3 `DeltaSqliteCheckpointer.alist`

`deep_worker_cli/delta_checkpointer.py:535-579`

- Yields `CheckpointTuple` for every checkpoint id in the thread by calling
  `aget_tuple` per row.
- Cost: `N_checkpoints × O(depth)`. For a 12-checkpoint thread with depth ~5
  this is **~60 full chain walks** per `alist()` call.

### 2.4 `_handle_get_history` and `_handle_get_last_response`

`deep_worker_cli/process/agent_process.py:741-985`

- Both call `self.agent.aget_state(config)` on line 805 / 933 — which hits
  the checkpointer end of the pipeline above.
- Then iterate every selected message and rebuild a dict representation
  (type, id, content, tool_calls, etc.) — `O(messages)` per call, allocated
  fresh every time.
- No per-message memoization — even immutable, long-ago AIMessages are
  re-serialized every call.

### 2.5 `_handle_invoke` / `astream`

`deep_worker_cli/process/agent_process.py:987-1500`

- `self.agent.astream(...)` triggers LangGraph's per-node
  `checkpointer.aget_tuple(...)` and `checkpointer.aput(...)`.
- For a typical turn with 5–10 nodes this is 5–10 full reconstructions,
  each of which walks a parent chain whose depth has just increased by one.

### 2.6 Middleware that rereads state on every turn

| Middleware | File | What it rereads | Should be cached by |
|------------|------|-----------------|---------------------|
| `MemoryMiddleware` | `deepagents/middleware/memory.py` | AGENTS.md files from backend on every `before_agent` | (source_path, mtime) → text |
| `SkillsMiddleware` | `deepagents/middleware/skills.py` | Full scan of skill directories + YAML-frontmatter parse on every `abefore_agent` | (source_path, mtime) → list[SkillMetadata] |
| `PatchToolCallsMiddleware` | `deepagents/middleware/patch_tool_calls.py:14` | Iterates every message in state to look for dangling tool_calls | last-seen-message-id high-water mark |
| `SummarizationMiddleware` | `deepagents/middleware/summarization.py` | Token counts over all messages on every `before_model` | (tuple of message ids) → token count |
| `FilesystemMiddleware` | `deepagents/middleware/filesystem.py` | Reads backend on every tool call, patches `files` state | state-reducer already there; tighten |

### 2.7 RPyC JSON serialization

`deep_worker_cli/rpyc_service/agent_service.py` — every `exposed_get_*`
round-trips the dict through `json.dumps`. For `get_history` the payload is
the entire formatted history, rebuilt and re-encoded on every call.

### 2.8 Web-UI polling

The web UI's `agent_service.py` — every 5 s per
agent. Only calls `get_status()` (cheap) on the monitor path. History
endpoints are on-demand, but the combined effect of status polls + an
occasional history fetch is enough to sustain the observed 20 % CPU idle load.

This is the **caller side** of the problem. The document assumes we do NOT
ask the web UI to back off — we fix the server instead, because a legitimate
rate-limited polling consumer should cost ~nothing when nothing changed.

---

## 3. Design principles

1. **"Nothing changed" must be free.** A read that follows another read, with
   no intervening `aput`, must serve from RAM. Zero SQLite queries, zero
   deserializations, zero full-state copies.
2. **"One delta appended" must be cheap.** Applying one appended delta on
   top of a cached, already-reconstructed state is `O(delta_size)`, not
   `O(history_size)`. The chain-walk path is a fallback for cold starts and
   summarization, not the common case.
3. **One authoritative cache per (thread_id, checkpoint_ns).** Every
   rebuild site consults the same cache. No duplicated, independently-
   invalidated caches scattered across middleware.
4. **Writes are the only invalidation trigger.** Reads never mutate cache
   validity. `aput` and `aput_writes` fire a single "version bumped" event;
   everything else is derivative.
5. **Explicit staleness, not TTL.** The cache is not time-based. Entries
   are either valid (no write since their `cached_version`) or stale
   (write happened; they are discarded or upgraded via delta). No
   wall-clock timers.
6. **Bounded memory.** One entry per (thread_id, checkpoint_ns) per
   process. Entry size is dominated by `channel_values` — same as a
   fully-reconstructed state. No multiplicative growth; no per-checkpoint-id
   cache that grows unbounded.
7. **No polling.** The cache does not poll SQLite for changes; it is
   authoritative inside the process because `aput` goes through this
   same checkpointer instance. A second process writing to the same
   SQLite file is explicitly out of scope — we have one writer per
   agent.

---

## 4. Architecture

### 4.1 Data model

```python
@dataclass
class ThreadCacheEntry:
    thread_id: str
    checkpoint_ns: str
    # Latest materialized state. Same shape as Checkpoint["channel_values"],
    # kept as a live reference (NOT deepcopied on read — callers must not
    # mutate; we rely on the LangGraph contract that channel_values is
    # treated as immutable outside the checkpointer).
    channel_values: dict[str, Any]
    # Identity of the checkpoint this entry represents. Every read that
    # requests this specific checkpoint_id is a direct hit.
    checkpoint_id: str
    # Full materialized Checkpoint object — what aget_tuple returns
    # downstream. Cached so we do not rebuild the TypedDict wrapper every
    # call.
    checkpoint: Checkpoint
    # Metadata and parent_id of the cached checkpoint — part of the
    # CheckpointTuple contract.
    parent_checkpoint_id: str | None
    metadata: dict[str, Any]
    # Monotonic version bumped on every aput / aput_writes. Used to
    # detect "nothing changed since I last looked" in derivative caches
    # (middleware, message-dict cache). Not exposed outside the process.
    version: int
```

### 4.2 Single cache location

The cache lives on `DeltaSqliteCheckpointer`. The checkpointer instance is
already created once per agent process and passed into LangGraph, so its
lifetime is the agent's lifetime — exactly the right scope.

```python
class DeltaSqliteCheckpointer(BaseCheckpointSaver):
    ...
    # Replaces the existing _state_cache, which is currently
    # dict[(thread_id, checkpoint_ns)] -> channel_values_dict.
    _cache: dict[tuple[str, str], ThreadCacheEntry]
    # Broadcast hook. Every time aput / aput_writes succeeds, bumps
    # version on the entry and fires callbacks. Middleware subscribes.
    _listeners: list[Callable[[str, str, int], None]]
```

### 4.3 Read path — `aget_tuple`

New logic, ordered:

1. Parse `thread_id`, `checkpoint_ns`, `checkpoint_id` from `config`.
2. Look up `entry = self._cache.get((thread_id, checkpoint_ns))`.
3. **Hit, exact match:** `entry is not None and (checkpoint_id is None or
   entry.checkpoint_id == checkpoint_id)`. Return the cached
   `CheckpointTuple` immediately. **Zero SQLite queries.**
4. **Hit, stale:** entry exists but caller asked for a different
   `checkpoint_id` (historical browse, rare). Fall through to full
   reconstruction for that id **without evicting** the cached latest;
   historical tuples are built on demand and not cached (§5.1 justifies).
5. **Miss:** no entry. Cold start. Do the full chain walk exactly as
   today (this is the **only** path that still does it), populate the
   cache with the reconstructed entry, return.

Pending-writes fetch (currently lines 165-177) stays on every call — writes
are per-checkpoint and we do not try to cache them across reads.

### 4.4 Write path — `aput`

Current `aput` already builds the full new state (it has `checkpoint` as an
argument, which is the new LangGraph checkpoint with full `channel_values`).
New logic:

1. Compute delta and serialize exactly as today (lines 309-363 unchanged).
2. Commit.
3. Build / update `ThreadCacheEntry` from the new checkpoint — this is
   free because `checkpoint["channel_values"]` is already in memory (we
   are writing it to SQLite anyway):
   - `entry.channel_values = dict(checkpoint["channel_values"])`
   - `entry.checkpoint_id = new cp_id`
   - `entry.parent_checkpoint_id = parent_checkpoint_id`
   - `entry.checkpoint = <rebuilt from the new cp in memory, without
     touching SQLite>`
   - `entry.version += 1`
4. Invoke `self._listeners` with `(thread_id, checkpoint_ns, new_version)`.

Result: the very next `aget_tuple` after an `aput` is a cache hit by
construction — we never re-deserialize what we literally just serialized.

### 4.5 Write path — `aput_writes`

Currently `aput_writes` writes to the `writes` table without touching
`_state_cache`. This is correct: writes are per-checkpoint side-data, they
do not change `channel_values`. We still bump `entry.version` so the
middleware-layer caches (which may legitimately depend on pending writes)
can invalidate themselves. An alternate, tighter option is a separate
`writes_version` counter — mentioned in §8.

### 4.6 Middleware cache hooks

`MemoryMiddleware`, `SkillsMiddleware`, `SummarizationMiddleware`,
`PatchToolCallsMiddleware` each keep a **per-middleware** dict keyed by
the same `(thread_id, checkpoint_ns)`, with a stored `version` and the
cached derivative:

- `MemoryMiddleware`: `{(thread_id, ns): (stat_mtimes_of_sources, rendered_text)}`.
  On `before_agent` — stat the sources; if all mtimes unchanged and we
  have a rendered text, use it. No version dependency, because AGENTS.md
  is not part of `channel_values`. (State-independent — simplest case.)
- `SkillsMiddleware`: same pattern, keyed by source_path.
- `PatchToolCallsMiddleware`: cache
  `{(thread_id, ns): (last_scanned_message_id, patched_message_ids)}`.
  Iterate only messages **after** `last_scanned_message_id`. If no new
  messages (look at `entry.version` from the checkpointer — if
  unchanged, skip entirely).
- `SummarizationMiddleware`: cache
  `{(thread_id, ns): (last_version, token_count, first_uncounted_msg_idx)}`.
  Each turn appends 1–3 messages — add their tokens to the running
  count instead of recounting.

These do NOT need a cross-process mechanism. They subscribe to the
checkpointer's version bump via the `_listeners` hook defined in §4.2
and invalidate their own entry when a write happens to their thread.

### 4.7 Message-dict serialization cache

`_handle_get_history` currently rebuilds a dict for every message on
every call. Add a per-agent cache:

- `{message_id: {"type": ..., "content": ..., "tool_calls": ...}}`
- `message_id` is stable (LangChain assigns one per message).
- Entries are immutable once written — a message never changes after
  being appended. We do not invalidate, only add.
- Cache is bounded by current history length; no eviction policy needed
  at agent-lifetime scale (messages are trimmed by summarization, at
  which point we drop entries whose id is no longer in the live
  history).

On every `get_history` call: for each message in the slice, either hit
the dict cache or build + insert. The serialization loop becomes
`O(new_messages_since_last_call)` instead of `O(all_messages_in_slice)`.

### 4.8 RPyC JSON serialization

The history payload is a list of per-message dicts. §4.7 already caches
those. The outer `json.dumps` is cheap compared to the rebuild and is
left alone. (It could be further optimized with an
incremental-JSON-by-appending scheme, but that is a premature
optimization — benchmark first.)

---

## 5. Edge cases and correctness

### 5.1 Historical browse (`checkpoint_id` specified)

When a caller asks for a specific non-latest checkpoint (time travel,
replay), we fall through to full reconstruction **without evicting** the
cached latest. These calls are rare; caching them would push the cache
size to `O(N_checkpoints)`. Document the asymmetry in the code comment
and in README.

### 5.2 `alist` and per-tuple reconstructions

`alist` calls `aget_tuple` in a loop. Two options:

- **A (preferred):** keep current behavior but skip per-call rebuild for
  the latest entry (cache hit). Other entries still cost O(depth), same
  as today. No regression, partial win.
- **B:** build `alist` on a single SQL query that returns the whole
  chain and apply all deltas in one pass in Python. O(chain) instead of
  O(N_checkpoints × depth). This is a nice-to-have for debug / admin
  tools; not on the critical path for the CPU hotspot in question.

Ship A now, file B as a follow-up.

### 5.3 Summarization

Summarization is the one event that legitimately invalidates the append
assumption (`SummarizationMiddleware` shrinks the message list). When
`aput` sees `_is_summarization(...)` (already detected at line 322), it:

1. Persists a full snapshot (already today).
2. **Replaces** the cache entry rather than appending. `version` still
   bumps, all derivative caches invalidate, everything rebuilds once.

This is the one place full rebuild is unavoidable, and it is rare by
construction.

### 5.4 Agent restart

Cache is in-process. Restarting the agent loses the cache → first
read is a cold chain walk, subsequent reads are hits. Acceptable:
restart is a manual / exceptional event, not part of steady state.

### 5.5 Concurrent reads

`DeltaSqliteCheckpointer` runs in a single asyncio event loop, so there
is no true concurrency between `aput` and `aget_tuple`. Ordering within
the loop is preserved by `await` points. No locks needed.

If this changes (e.g. migration to a thread pool), upgrade the cache to
a versioned-CAS scheme: reads snapshot `(entry, version)` at entry,
writes bump `version`. A read that notices version changed during its
work retries. Out of scope for this document.

### 5.6 Two processes sharing one SQLite file

Not supported. One writer per agent is a hard invariant of the current
architecture (each agent has its own `sessions_<agent>.db`). If a second
process ever needs to read, it will see stale data — document this
limitation. SQLite-level change-notification is a much larger redesign
and not what we are solving here.

---

## 6. Performance expectations

Baseline measurement (zombie agent, 39 min idle): `rchar = 37 GB`,
`syscr = 10 M`, steady 20 % CPU, no user activity, no agent activity,
~12 rpyc round-trips / 5 s from the web UI (two connections, status polls +
occasional history).

With the proposed cache:

| Scenario | Today | After Phase 1 | After Phase 2 |
|----------|-------|---------------|---------------|
| `get_status` (web UI, 5 s interval) | 0 history reads — already cheap | unchanged | unchanged |
| `get_history` with no new message since last call | full chain walk + per-msg dict rebuild | single dict lookup | single dict lookup |
| `get_history` after 1 new message | full chain walk + rebuild of N dicts | chain walk avoided (cache hit); 1 dict built, N-1 reused | same |
| `get_history` cold / different checkpoint_id | full chain walk (unchanged — rare) | unchanged | unchanged |
| One invoke turn (5 nodes) | 5 full chain walks + 5 `aput`s | 5 cache hits + 5 `aput`s (each updates cache) | same |
| `before_agent` / `before_model` middleware stack | rereads sources / recounts tokens every turn | unchanged | O(1) amortized per turn |

Expected steady-state `rchar` in the observed idle scenario: **under 1 MB
per 5 s** (status poll JSON round-trip only), down from ~50 MB / s.
Expected CPU: **under 1 %** idle, down from ~20 %.

The numbers are not speculative: today's web-UI status poll is
already cheap (only reads `session_manager.status`); the cost comes
entirely from the paths we are caching.

---

## 7. Implementation plan

All phases are independently valuable and ship-able.

**Phase 1 — checkpointer cache on the read path.** §4.3 + §4.4 + §4.5.
Expected impact: eliminates the 150× rchar amplification and the
20 % idle CPU. Smallest change, biggest win. ~150 lines, tests in
`tests/unit_tests/test_delta_checkpointer.py` (we own this file).

**Phase 2 — middleware caches.** §4.6. Separate PRs per middleware
— each is independent. Each middleware grows ~30 lines + a test. Biggest
gains during an active invoke, smaller gains idle.

**Phase 3 — message-dict cache.** §4.7. ~40 lines in `agent_process.py`.
Biggest gain is on long-history `get_history` calls (the web UI's "view full
dialog" button).

**Phase 4 (optional) — `alist` batching.** §5.2 option B. Only if
benchmarks show `alist` is a real hotspot; unlikely given current
call graph.

### 7.1 Testing strategy

- Unit tests on `DeltaSqliteCheckpointer`: cache hit after `aput` — zero
  SQLite queries (assert via a counting wrapper on `conn.execute`).
- Unit tests on cache invalidation: `aput` bumps version; derivative
  caches rebuild.
- Unit tests on summarization: full snapshot forces cache replacement,
  not patching.
- Property test: for a random sequence of `aput` / `aget_tuple` / historical
  `aget_tuple` calls, the cache-on and cache-off paths return
  byte-identical results.
- Benchmark harness: replay the observed web-UI poll pattern (12 polls
  / 5 s, 1 `get_history` / min) against a 100-checkpoint thread and
  measure `rchar`, `syscr`, CPU.

### 7.2 Observability

Add debug counters on `DeltaSqliteCheckpointer`:

- `cache_hits_total`
- `cache_misses_total`
- `chain_walks_total`
- `cache_size_entries`

Exposed via a lightweight `get_cache_stats()` method. Not wired into
metrics infra in this patch — just printable on demand.

### 7.3 Rollback

Feature-flag via constructor arg: `DeltaSqliteCheckpointer(conn,
snapshot_interval=0, cache_enabled=True)`. Defaults to on. Set to
`False` to restore current behavior without redeploying.

---

## 8. Open questions

1. **Per-checkpoint_id caching?** §5.1 says no; revisit if time-travel /
   replay tools become common.
2. **Separate `writes_version` vs. unified `version`?** Current proposal
   bumps a single `version` on both `aput` and `aput_writes`.
   `aput_writes` is per-step and does not change `channel_values`; a
   separate counter would let middleware that does not care about
   pending writes skip an invalidation. Defer until we have a
   middleware that actually benefits.
3. **Shared cache across sibling agents in one process?** Currently each
   agent is its own OS process. If this changes (merged-process model),
   one cache per process with `(agent_name, thread_id, checkpoint_ns)`
   keying is trivial — document but do not implement speculatively.
4. **Integration with LLM prompt caching (Anthropic).** Prompt caching
   also benefits from stable prefixes. Our delta cache preserves
   prefix stability as a side effect, but we should confirm the
   `AnthropicPromptCachingMiddleware` breakpoint placement does not
   change between turns in a way that defeats its cache. Out of scope
   here, track separately.
5. **Thread-local cache during `astream`.** LangGraph may call
   `aget_tuple` and `aput` many times inside one `astream`. Our cache
   handles this correctly by construction (every `aput` updates,
   every `aget_tuple` reads). Worth confirming with a targeted trace.

---

## 9. Non-goals

- Cross-process cache sharing.
- SQLite-level change notification / triggers.
- Redesigning the checkpoint schema (we keep the existing v1/v2 delta
  format).
- The web UI's polling frequency or UI-side caching.
- Rewriting middleware base classes.
- Any change to the RPyC protocol or serialization format.
