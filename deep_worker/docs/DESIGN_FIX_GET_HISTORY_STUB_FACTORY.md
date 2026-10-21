# DESIGN: Fix `test_get_history_no_limit.py` stub factory after delta-cache

**Status:** 📋 implementation PR
**Author:** DeepWorker team
**Date:** 2026-04-23
**Related:** `8c2e28d feat(context): delta-based context cache (phases 1+2+3)`,
`1eee827 fix(summarization): restore _truncate_args_* init after cache
insertion`, `DESIGN_CONTEXT_DELTA_CACHE.md §4.7`.

---

## 1. Problem

`tests/unit_tests/process/test_get_history_no_limit.py` has 5 tests RED:

```
AttributeError: 'AgentProcess' object has no attribute '_message_dict_cache'
```

Root cause (diagnosed in the prior analytical report):

- Commit `8c2e28d` added a new attribute to `AgentProcess.__init__`:

  ```python
  self._message_dict_cache = MessageDictCache()
  ```

- `_handle_get_history` now uses that attribute unconditionally
  (`agent_process.py:844`, `:855`).
- The test factory `_make_agent_process_with_messages` constructs the
  object via `AgentProcess.__new__(AgentProcess)` (i.e. WITHOUT calling
  `__init__`) and hand-sets only the fields the OLD `_handle_get_history`
  read (`name`, `agent`). It never got updated to mirror the new
  attribute, so the new code-path hits `AttributeError`.

`_handle_get_history` catches the exception and returns
`{"success": False, "error": ...}` — no `history` key — and the tests
fall over on `resp["history"]`.

**Production is NOT affected.** Real `AgentProcess(...)` always runs
`__init__`; the cache is always present. This is entirely a test-stub
bug.

## 2. Goals

1. Restore the 5 tests to GREEN without touching production code.
2. Preserve — and ACTIVELY VERIFY — the incremental-cache contract:
   - calling `_handle_get_history` twice on an unchanged history MUST
     produce cache hits, not re-serialise;
   - calling `_handle_get_history` after appending a NEW message MUST
     only serialise the new message (one extra miss, no blanket rebuild);
   - `prune_to` is the only path that evicts, and it only evicts ids
     not in the live set.
3. Keep the fix small, local, and robust to *future* `__init__`
   extensions (so the next attribute added to `AgentProcess.__init__`
   does not silently break the same factory again).

### Non-goals

- Refactoring `AgentProcess.__init__` or the cache implementation.
- Introducing a fat test fixture that fully mocks RPyC/TaskManager —
  out of scope and high-risk.
- Merging the factory into a shared `conftest.py` helper — three
  separate copies exist (grep evidence), consolidating them is a
  separate hygiene PR.

## 3. Constraints (from user)

Strict rules for this change:

- **No sync regressions.** `_handle_get_history` is already async; the
  tests are `@pytest.mark.asyncio`. No sync/async inversion.
- **No polling.** The added incremental-cache test reads `cache.stats()`
  directly — one call, not a loop.
- **No new threads.** Everything runs on the existing asyncio event
  loop; the cache is intentionally thread-unaware.
- **Event-driven, non-blocking.** `_handle_get_history` is awaited once
  per test case; no sleeps, no busy-wait.
- **Client and agent (server) MUST run in separate processes when
  tested together.** For this ticket we do NOT add such an integration
  test — it is covered by the existing
  `tests/integration/test_get_history_real_rpyc.py` / sibling integration
  suite that *already* spawns agents out-of-process. Our unit tests
  legitimately stay in-process because they target `AgentProcess`
  internals (cache semantics, limit handling) where spawning a full
  process would be both slower and less precise. The "separate process"
  rule applies to the live-verification step (§6).
- **Do NOT break the message/history delta cache.** The fix explicitly
  constructs the very same `MessageDictCache()` that production
  `__init__` does. We additionally pin the incremental contract in a
  new test so future drift is caught immediately.

## 4. Design — Variant A (minimal factory patch) + incremental-cache pin

Two edits in `tests/unit_tests/process/test_get_history_no_limit.py`:

### 4.1 Factory: replicate the `__init__` field the new code needs

```python
def _make_agent_process_with_messages(messages: list[Any]):
    from deep_worker_cli.process.agent_process import AgentProcess
    from deep_worker_cli.process.message_dict_cache import MessageDictCache

    proc = AgentProcess.__new__(AgentProcess)
    proc.name = "test-agent"

    fake_state = MagicMock()
    fake_state.values = {"messages": messages}

    fake_agent = MagicMock()
    fake_agent.aget_state = AsyncMock(return_value=fake_state)
    proc.agent = fake_agent  # type: ignore[assignment]

    # ── NEW: mirror AgentProcess.__init__ attrs that _handle_get_history
    # touches. Must match production wiring (DESIGN_CONTEXT_DELTA_CACHE §4.7).
    # If __init__ grows more attrs that _handle_get_history reads, mirror
    # them here and pin them with a dedicated contract test.
    proc._message_dict_cache = MessageDictCache()

    return proc
```

No production code touched. No other test factory is altered (the two
sibling factories do not call `_handle_get_history`, as verified by the
baseline run: they are GREEN without the cache attribute).

### 4.2 New test class: `TestIncrementalDictCache`

Three tests pinning the incremental contract that the user explicitly
protected:

| # | Test name | What it pins |
|---|---|---|
| 1 | `test_second_call_on_unchanged_history_is_all_hits` | Call `_handle_get_history` twice on the same N-message list; assert `cache.stats()` shows `misses==N` after the first, `misses==N, hits>=N` after the second (no extra misses). |
| 2 | `test_appending_a_new_message_serialises_only_that_message` | Call once on N messages, append one new message with a fresh id, call again. `misses` must become exactly `N+1` — incremental, not blanket. |
| 3 | `test_prune_to_is_the_only_eviction_path` | After a call populating the cache, invoke `proc._message_dict_cache.prune_to({subset})`; assert evicted-count matches, and a follow-up call with a history restricted to `subset` produces hits only. |

These tests double as documentation: they are the executable spec for
the cache's incremental promise.

## 5. Implementation order (TDD)

1. Add the three `TestIncrementalDictCache` tests first. They FAIL with
   the same `AttributeError` — confirming RED.
2. Apply the factory patch. All 5 pre-existing failures plus the 3 new
   tests go GREEN in one move.
3. Run the entire `deep_worker_cli` unit suite to ensure nothing else
   regresses. Target: 1313 + 3 = 1316 passed, 1 skipped.

## 6. Live verification (not mocks)

Constraint reminder: when client + agent are exercised together, they
MUST run in separate processes.

Procedure:

1. An agent process is already running on the sandbox (`lc6_fix`,
   pid 698655, rpyc_port 47893). It was started via the normal
   `deep_worker_cli` entry-point, so its `AgentProcess.__init__` ran
   completely and the cache is fully initialised in production shape.
2. From THIS process (separate from the agent), connect via
   `rpyc.async_connect` (already used in the prior summarization live
   check) and call `root.get_history(limit=None, offset=0)` **twice
   in succession** with no intervening agent activity.
3. Assertion: second call's server-side cache stats show `hits >=
   first_call_misses` (every message hit the cache the second time).
   There is no direct RPyC method to read `cache.stats()` — so the
   indirect proxy is timing + byte-length: second call returns the same
   payload (byte-identical `history` list shape) and demonstrably
   faster because serialisation is memoised. We report both numbers.
4. Append step: send a no-op inter-agent message from this process so
   the agent writes one new AIMessage, then call `get_history` a third
   time. History length grows by ≥1. No payload corruption, no
   duplicate ids — which would be the symptom of blanket rebuild
   mishandling ids.

This live check uses:
- `async_connect` — native non-blocking (rpyc_async AsyncioServer);
- one round-trip per observation — no polling;
- no thread spawn;
- the agent process was independently started — separate-process rule
  satisfied.

## 7. Risk analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| Silent factory drift again next `__init__` change | Medium | `TestIncrementalDictCache` pins the cache contract; any future drift in `_message_dict_cache` semantics trips the new tests |
| Cache evicts too aggressively | Low | `test_prune_to_is_the_only_eviction_path` verifies |
| Incremental-only serialisation broken | Low | `test_appending_a_new_message_serialises_only_that_message` verifies |
| Live-verification step touches production process | Out-of-process by construction (`async_connect` to a separately-spawned PID), no mutation beyond a benign test-tagged invoke |
