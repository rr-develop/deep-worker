# DESIGN — Multi-agent validation under load + refresh↔tool race

Series C, commit 11.

Branch `dev` (dev tree) → ff into `master`.

## §1. Goal

Formally, the "two agents through one service" criterion is already covered
by the smoke test `test_two_agents_share_one_serena_and_see_symbols`
(commit 8) — but that is a _single_ `get_symbols_overview` call
through two channels. Real operation implies:

1. Both agents send **batches** of tool calls back-to-back and
   **simultaneously**.
2. At any moment **either** of them may trigger `RefreshProject`
   (or the watcher does it automatically).

We must prove that under such conditions:

- **Every** tool call from **both** agents gets a valid response
  (no empty results, no `is_error=True`).
- **All** running `RefreshProject` calls finish successfully; on the
  AttachSession stream of one agent `REFRESH_STARTED` and
  `REFRESH_FINISHED` are visible in the right order; no "FINISHED without
  STARTED" or vice versa.
- The backend's `_lock` really serializes `tool ↔ refresh` pairs rather
  than letting them overlap (this is already verified by the unit test
  `test_refresh_shares_lock_with_call_tool` on a fake session; here we run
  the same thing on a **real** Serena under concurrent load).

## §2. What is NOT in scope

- Changing the gRPC protocol / supervisor code / policy guard.
- Stress on more than two agents (a next step, if ever needed).
- Microbenchmarks of latency — not this task.

## §3. Invariants the tests will pin

1. **Zero errors in responses.** N parallel `get_symbols_overview` calls
   from each of the two agents return a non-empty symbol list;
   no call raises an exception or returns `is_error`.
2. **Refresh is not lost.** If `RefreshProject(force=True)` is inserted
   between batches of tool calls, the RPC returns `triggered=True`, and
   the AttachSession stream of one of the agents receives
   `SESSION_EVENT_REFRESH_STARTED` followed by
   `SESSION_EVENT_REFRESH_FINISHED` (pair order — preserved).
3. **One Serena for all.** After the load, the service process tree has
   exactly one live Serena-child (regression: commit 10 must not have
   restarted it accidentally on a race).
4. **Server admin RPCs stay available.** Simultaneously with the tool load,
   `Health` and `ProjectStatus` respond within 1 s.

## §4. Test design

One new file `tests/integration_tests/lsp/test_lsp_multi_agent_stress.py`,
marker `integration`. It uses the same helpers that already exist in
`test_serena_backend_e2e` / `test_serena_auto_restart`:

- `_py_project(tmp_path)` — a fixture with `pkg/core.py::Widget`;
- `_supervisor` + `_dev_serve_argv` + `_dev_pythonpath` autouse
  (the service from the dev tree, not the PATH-based master);
- `_serena_pids_under(service_pid)` for the "one Serena" invariant;
- `_wait_backend_ready(stub, budget_s)` for an event-driven Health-wait
  at startup.

**Hard-coded budgets (A4)**: `_START_S = 40.0` (matches e2e),
`_TOOL_S = 30.0` (per-call `asyncio.wait_for`), `_WHOLE_TEST_S = 120.0`
(the overall wall-budget; if exceeded — `pytest.fail` instead of a hang),
`_REFRESH_EVENT_S = 20.0` (wait for a STARTED→FINISHED pair on the stream).

### Attach-stream observation (A1)

`BridgeSession` is **not suitable** — its `_hold_attach_stream` reads the
stream and **throws away** the events. Production code is NOT changed.

Instead, the test opens a **third** gRPC channel with a raw
`pbg.LspServiceStub` and calls `AttachSession(pb.AttachRequest(
client_id="observer"))` — reading each `SessionEvent` into an
`asyncio.Queue`. The agent_a/agent_b clients use `BridgeSession` as usual
(this is what real agents use in production).

Pseudocode of the observer:

```python
async def _observe_events(stub, out_queue: asyncio.Queue, stop_event):
    stream = stub.AttachSession(pb.AttachRequest(client_id="observer"))
    try:
        async for ev in stream:
            await out_queue.put(ev)
            if stop_event.is_set():
                return
    except grpc.aio.AioRpcError:
        return
```

The observer runs as a separate `asyncio.Task` and is cancelled correctly at
the end of the test (`task.cancel()` + `await` in try/except).

### RefreshProject semantics (A2)

`RefreshProject` returns `triggered=True` **before** `REFRESH_STARTED`/
`FINISHED` appear on the stream (internally — `create_task`).
So "refresh is not lost" is checked by two conditions:

1. The RPC returned `triggered=True` — otherwise fail immediately.
2. An event-driven wait on the observer queue: first STARTED, then
   FINISHED in the **correct** order, both within `_REFRESH_EVENT_S`.
   If `FINISHED.message` starts with `"error:"` — the test **fails**
   (that reflects a backend failure, not a successful refresh).

The order is fixed by indices in the queue: index(FINISHED) must come
strictly after index(STARTED), and both must belong to the same trigger
(compared via `timestamp_ns` timestamps).

### The main stress scenario

`test_two_agents_concurrent_tool_calls_stay_correct`:

```python
_ROUNDS = 3
_CALLS_PER_ROUND = 6   # from each agent

for _ in range(_ROUNDS):
    calls = []
    calls += [
        asyncio.wait_for(
            agent_a.call_tool("get_symbols_overview", {"relative_path": "pkg/core.py"}),
            timeout=_TOOL_S,
        )
        for _ in range(_CALLS_PER_ROUND)
    ]
    calls += [
        asyncio.wait_for(
            agent_b.call_tool("find_symbol", {"name_path": "Widget"}),
            timeout=_TOOL_S,
        )
        for _ in range(_CALLS_PER_ROUND)
    ]
    results = await asyncio.gather(*calls, return_exceptions=True)
    for r in results:
        assert not isinstance(r, Exception), f"tool call raised: {r!r}"
        assert r, f"tool returned empty result: {r!r}"
```

Checks after the storm (A5, A6):

- the `get_symbols_overview` response must contain `"Widget"` as a symbol
  name — otherwise a non-empty payload could be garbage.
- the `find_symbol` response must return at least one element with the
  field `name == "Widget"`.
- `len(_serena_pids_under(entry.pid)) == 1` before and after the test;
  preferably the same pid (otherwise Health should have dipped, which we
  would already have caught — see the next point).

Admin-availability check under load (A4):

```python
# runs in the same gather as the tool-storm:
admin_probe = asyncio.wait_for(_probe_admin_rpcs(stub, iters=10), timeout=_TOOL_S)
```

where `_probe_admin_rpcs` in a loop over iters calls `Health` and
`ProjectStatus` with `asyncio.wait_for(..., timeout=1.0)` and fails if any
call does not fit within one second. This proves invariant (d) under
**peak** load, not "afterwards".

### The second scenario (refresh↔tool race)

`test_refresh_between_tool_batches_is_never_lost`:

```python
# 1. The observer is up before the agents start.
# 2. A tool_storm from B (see above);
# 3. in parallel, A calls RefreshProject(force=True).
# 4. asyncio.gather(tool_storm_task, refresh_task) with per-task
#    timeouts and a common _WHOLE_TEST_S.
# 5. After the gather: refresh_task.result().triggered == True;
#    the observer.queue contains STARTED and FINISHED in the correct
#    order (see A2); tool_storm == all ok.
```

### A single test wall-budget (A4)

Each test is wrapped in:

```python
async with asyncio.timeout(_WHOLE_TEST_S):
    ...
```

If we go over 120 s — we fail loudly rather than hang on CI.

## §5. What already exists and is NOT repeated

- `test_two_agents_share_one_serena_and_see_symbols` (backend_e2e):
  one call, one Serena. Stays as a smoke test.
- `test_refresh_shares_lock_with_call_tool` (unit): a fake session,
  proves locking. Stays as a fast unit.
- The auto-restart integ (commit 10): Serena death. Orthogonal.

The new tests are concentrated stress load + a race.

## §6. Code invariants — no changes planned (in fact — a minimal fix, see §9)

The planned hypothesis was "production invariants are already implemented in
commits 4–10". Stress test item 2 **refuted** it live — see §9
"Found defect (Serena 1.6 refresh tool)": `SerenaBackend.refresh`
failed in production on real Serena 1.6.x. Per §7 the fix is made in this
same commit, minimal and local (two files + a shipped resource).

## §7. What to do if the stress shows a defect

Three possible outcomes of the stress tests:

- All green → a clean commit with tests only (+agent memory update). The
  design does not change.
- A real bug found → we fix it in this same commit, add a regression test,
  and re-review with the reviewer. The design gets a § describing the found
  defect.
- The test itself is flaky → this is not "skip it", but rewriting it to an
  honest event-driven wait for the effect being checked. No retries /
  sleep-polls.

## §8. Delta (as built)

Tests:

- NEW `tests/integration_tests/lsp/test_lsp_multi_agent_stress.py`
- MOD `tests/unit_tests/lsp/test_serena_config.py` — two
  regression tests for the §9 fix:
  `test_entry_layers_the_deep_worker_serena_mode`,
  `test_refresh_tool_candidates_prefer_restart_language_server`.

Production code (the minimal fix found by the stress — §9):

- NEW `deep_worker_cli/lsp/resources/__init__.py` — re-exports
  `SERENA_MODE_YML` (the absolute path to the YAML).
- NEW `deep_worker_cli/lsp/resources/deep_worker_serena_mode.yml` —
  the Serena mode YAML with `included_optional_tools:
  [restart_language_server]`.
- MOD `deep_worker_cli/lsp/serena_config.py::build_serena_mcp_entry` —
  adds `--add-mode <SERENA_MODE_YML>` to the spawn argv.
- MOD `deep_worker_cli/lsp/service/serena_backend.py::_REFRESH_TOOL_CANDIDATES` —
  `restart_language_server` first in the list, the old candidates kept as
  a legacy fallback.

Documentation:

- NEW `docs/DESIGN_LSP_MULTI_AGENT_STRESS.md` (this file).

The root `AGENTS.md` is **not touched** (reviewer's note — "a tests-only
commit shouldn't touch the internal service's AGENTS.md"); the agent's
memory note is updated outside the commit.

## §9. Found defect (Serena 1.6 refresh tool)

**Symptom (from a live run of the stress test).**
`test_refresh_between_tool_batches_is_never_lost` caught:

```
REFRESH_FINISHED reports backend error:
"error: no serena refresh tool advertised
 (tried ['activate_project', 'refresh_project', 'serena__refresh'];
  got ['replace_content', 'replace_in_files', 'replace_symbol_body',
       'insert_after_symbol', 'insert_before_symbol',
       'search_for_pattern', 'get_symbols_overview', 'find_symbol',
       'find_referencing_symbols', 'find_implementations',
       'find_declaration', 'get_diagnostics_for_file', 'rename_symbol',
       'safe_delete_symbol', ... , 'onboarding', 'initial_instructions'])"
```

At the same time `RefreshProject.triggered=True` was returned, but in fact
the refresh never fired: `backend.refresh` threw
`RuntimeError("no serena refresh tool advertised")`, `_run_refresh`
caught it and wrote `error: ...` into `note_refresh`/the session event.
The watcher-triggered refresh (commit 8) silently degraded into a **no-op**
with an error in the stream, and nobody noticed until the concurrent stress
with a stream check started.

**Root cause.** Serena 1.6.x moved the reload functionality into
`RestartLanguageServerTool`, marked `ToolMarkerOptional` —
by default NOT advertised to the MCP client. The old candidates of our
`_REFRESH_TOOL_CANDIDATES` in 1.6.x are either removed
(`refresh_project`, `serena__refresh`) or also optional
(`activate_project`).

**Fix — three components, local:**

1. **The `deep_worker_serena_mode.yml` resource** — a minimal mode-YAML:
   ```yaml
   included_optional_tools:
     - restart_language_server
   ```
   No prompt/exclude, only the opt-in of one tool.
2. **`build_serena_mcp_entry`** adds `--add-mode
   <abs path to shipped YAML>` to the spawn argv. Serena accepts an
   absolute path to a YAML as an alternative to a mode-name; verified
   in the `serena.cli` sources. Isolation preserved: we do NOT touch the
   user's `~/.serena/serena_config.yml` and do NOT write into the
   `.serena/project.yml` of open-source projects.
3. **`_REFRESH_TOOL_CANDIDATES`** reordered: `restart_language_server`
   first, the old names kept as a legacy fallback (in case of a pin on an
   old Serena somewhere in CI).

**Live confirmation of the fix.** With `--add-mode
deep_worker_serena_mode.yml`, Serena 1.6.1 exposes 23 tools instead of 22
(see the log: `restart_language_server` appeared between
`replace_symbol_body` and `safe_delete_symbol`); `pick_refresh_tool`
returns `"restart_language_server"`; the stress test
`test_refresh_between_tool_batches_is_never_lost` passes in 3.81 s with
ordered STARTED→FINISHED without `error:`.

**Regression points:**
- `test_entry_layers_the_deep_worker_serena_mode` (unit) — if
  `build_serena_mcp_entry` stops passing `--add-mode` or the YAML
  disappears from disk / loses `restart_language_server`, it fails
  immediately.
- `test_refresh_tool_candidates_prefer_restart_language_server`
  (unit) — if someone reverts the candidate order, it fails.
- `test_refresh_between_tool_batches_is_never_lost` (integ) —
  an end-to-end real check, proof that the stream-side refresh is no
  longer degraded.
