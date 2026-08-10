# DESIGN — LSP service: observability

Series B, commit 12.

Branch `dev` → ff `master` (standard process).

## §1. Goal

Give the operator and the tests an **honest live status** of the service:

1. **Health.serena_version** — currently empty. Fill it with the real
   version of the backend brought up by the service (from the
   `mcp initialize` response or a separate Serena tool).
2. **Backend liveness metrics**:
   - `backend_uptime_seconds` — integer seconds since the last
     **successful** `attach_backend` for the current live backend.
     `0` while the backend is not ready; reset to `0` on
     `mark_backend_not_ready`, and starts again with the next successful
     attach.
   - `restarts_total` — counter of **successful** Serena auto-restarts
     in this service process. Incremented only from
     `SerenaSupervisor._restart_once` with `ok=True`. The first attach
     via `supervisor.start()` does NOT count as a restart (the counter
     stays 0).
   - `last_tool_call_unix` — `int(time.time())` of the last successful
     tool call (not is_error). `0` while no successful call has happened.
   - `tool_calls_total` — total counter of successfully completed
     tool calls (only `not is_error`). RefreshProject and other
     administrative RPCs do NOT increment, because they do not go
     through `_dispatch_tool`.
3. **`deep-worker lsp service-status`** — extend: besides the registry
   info (pid/slug/transport/socket), call Health + ProjectStatus and
   display all metrics in human-readable form. It works through the
   same UDS socket as the agents; no new transport.

## §2. What is NOT in scope

- Export to Prometheus / OpenTelemetry (next piece of work if needed;
  no requirement right now).
- Structured JSON output for `service-status` — human-readable output
  only. Tests parse it with regex against specific keys; if JSON is
  ever needed, we add a `--json` flag.
- Long-term metric storage across service restarts. Everything is
  in-memory and reset when a new service is spawned.

## §3. Invariants

1. **Zero polling.** Metrics update event-driven (on restart — from the
   supervisor; on a tool call — from the tool handler).
2. **Fail-quiet only for cosmetic.** If Serena does not answer the
   version request, `serena_version` stays empty (does not break the
   Health `alive` contract). If Health/ProjectStatus do not answer in
   `service-status`, we print `unreachable: <reason>` instead of
   raising. Registry info is always printed.
3. **Backward wire compatibility.** We extend `HealthReply` and
   `StatusReply` with **new** fields (stable numbers), we do not reuse
   numbers. All old behavior is preserved.
4. **`serena_version` is read ONCE** — at the moment `session.initialize()`
   is called inside `_OwnedStdioSession.__aenter__`
   (see §4/A1 — that single call site; no second `initialize`).
   It is stored on the transport (`_OwnedStdioSession.
   server_version`) and from there copied into `SerenaBackend._serena_version`.
   Health serves the cache.
5. **`_backend_started_ns` is set** to `time.time_ns()` on a
   **successful** `attach_backend`. On `mark_backend_not_ready` it is
   reset to `None` → Health reports
   `backend_uptime_seconds = 0`. Rate-limit exhaustion in the
   supervisor (after which `mark_backend_not_ready` has in fact already
   run via a transport error or `on_lost`) does NOT invent a synthetic
   uptime: until a successful restart, the field stays `0`. A new start
   time comes only from the next successful `attach_backend`.

## §4. API and proto

### Proto — extension (new numbers, wire-compat)

```
message HealthReply {
  bool alive = 1;
  string service_version = 2;
  string serena_version = 3;
  string project_root = 4;
  int64 uptime_seconds = 5;
  int32 attached_clients = 6;
  // === NEW (commit 12) ===
  int64 backend_uptime_seconds = 7;   // 0 when the backend is not ready
  int64 restarts_total = 8;           // counter of successful restarts
}

message StatusReply {
  // ... existing 1–11 fields unchanged ...
  // === NEW (commit 12) ===
  int64 backend_uptime_seconds = 12;
  int64 restarts_total = 13;
  int64 last_tool_call_unix = 14;    // 0 if no call has happened
  int64 tool_calls_total = 15;
  string serena_version = 16;         // copy from Health for convenience
}
```

The pb stubs are regenerated with the standard command from the proto file.

### `LSPServiceServer` — new fields

```
self._backend_started_ns: int | None = None   # None = backend not ready
self._restarts_total: int = 0
self._last_tool_call_unix: int = 0
self._tool_calls_total: int = 0
self._serena_version: str = ""
```

`attach_backend`: on success — sets `_backend_started_ns` and
`_serena_version` (received from the backend — see below). On failure —
does not touch them.

`mark_backend_not_ready`: resets `_backend_started_ns = None`
and `_serena_version = ""` (the backend is no longer alive).

`_dispatch_tool` (around the tool handler): on a **successful** response
(`not is_error`) increments `_tool_calls_total` and updates
`_last_tool_call_unix`.

New public method `note_restart(success: bool)` (for SerenaSupervisor
only): if success — `_restarts_total += 1`; it does not touch
`_backend_started_ns` (the next `attach_backend` will do that).

### `SerenaBackend` — version (A1)

**The only `initialize()` call site** is inside
`_OwnedStdioSession.__aenter__`. `_run_session` does not call
`initialize` again; commit 12 uses the existing call. Scheme:

```
class _OwnedStdioSession:
    ...
    def __init__(self, argv):
        ...
        self.server_version: str = ""  # populated in __aenter__

    async def __aenter__(self):
        ...
        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        try:
            init_result = await session.initialize()
        except BaseException:
            await self._teardown()
            raise
        # Fail-quiet: bad serverInfo shape → empty string, not crash.
        server_info = getattr(init_result, "serverInfo", None)
        version = getattr(server_info, "version", "") if server_info else ""
        self.server_version = str(version) if isinstance(version, str) else ""
        self._session = session
        return session
```

`SerenaBackend._run_session` copies the value right after the session
is ready:

```
async with _capture_serena_process(...):
    session_cm = await _open_mcp_session(...)
    session = await self._stack.enter_async_context(session_cm)
    self._serena_version = getattr(session_cm, "server_version", "")
    listed = await session.list_tools()
    ...
```

Public:

```
class SerenaBackend:
    @property
    def serena_version(self) -> str:
        return self._serena_version
```

`_serena_version` is cleared in `stop()` / `_mark_lost` (the backend
is dead — the version is stale).

### `SerenaSupervisor` — `note_restart` calls

`_restart_once` with `ok=True` (at the end) calls
`server.note_restart(success=True)`. On failure — it does not call it
(that is an attempt; the attempts counter lives in the rate limit, while
`restarts_total` is a counter of successes).

### `service_cli.cmd_service_status` — extension

After the existing registry info:
1. Open a gRPC channel to `entry.socket_path` (timeout 2 s).
2. Call Health and ProjectStatus (1 s each).
3. Print a "backend:" and "metrics:" block with the new fields.
4. On RPC failure — print `unreachable: <error>` and return 1
   (not 0, so `deep-worker lsp service-status` can be used as a
   liveness probe).

## §5. Tests

### Unit `test_grpc_server_metrics.py`

- `test_health_includes_backend_uptime_after_ready` — manually
  set `_backend_started_ns` via `attach_backend`
  (we use the existing `_StubBackend` from test_serena_supervisor
  or a local stub); backend_uptime grows monotonically.
- `test_health_backend_uptime_zero_when_not_ready` — before
  `attach_backend` = 0.
- `test_health_backend_uptime_resets_on_not_ready` — after
  `mark_backend_not_ready` = 0.
- `test_restarts_total_increments_on_note_restart` — calling
  `note_restart(True)` increments; `note_restart(False)` — does not.
- `test_tool_call_metrics_advance_on_success` — dispatch a tool
  through a fake handler returning `is_error=False`;
  `tool_calls_total` +=1, `last_tool_call_unix` != 0.
- `test_tool_call_metrics_ignore_errors` — the handler returns
  `is_error=True` — the counters do NOT grow.
- `test_status_reply_carries_new_metrics` — all new fields
  in `StatusReply` are filled correctly.

### Unit `test_serena_backend_version.py`

The tests run a REAL `_OwnedStdioSession` against a tiny
python child speaking a minimal MCP handshake — the same trick
as in `test_serena_backend_pid.py`. This also proves that the version
comes from the real `initialize()`, not from a mock.

- `test_owned_session_captures_server_version` — the child returns
  `serverInfo.version = "1.6.1"`; `session.server_version == "1.6.1"`.
- `test_backend_exposes_and_clears_version` —
  `backend.serena_version` = the version after `start()` and **empty again**
  after `stop()` (a stopped backend must not advertise
  a version).
- `test_blank_version_degrades_to_empty` — fail-quiet. Clarification to
  the original plan: `serverInfo` in the MCP schema **is required**
  (pydantic rejects a response without it already in `ClientSession`),
  so the only reachable degradation is an empty version string. Attach
  must still succeed in that case.

### Unit — extension of the existing `test_serena_supervisor.py`

(no separate file: the tests use the already available
`_StubServer`/`_StubBackend`/`_FakePidfd`; duplicating them would be
worse than adding three tests next to them)

- `test_first_attach_does_not_count_as_restart` (A5) — after
  `supervisor.start()` **without** a pidfd fire, `note_restart` was not
  called. A guard against the mutation "the first attach also increments".
- `test_successful_restart_notes_restart_true` — on a pidfd fire
  the server sees `note_restart(True)` **exactly once**.
- `test_failed_restart_does_not_note_restart` — if
  `attach_backend` returned False, `note_restart` is not called.

### Integ `test_lsp_service_status_metrics.py`

- Bring up the service from the dev tree, wait for Health.alive.
- Run a tool call through the bridge (so that `tool_calls` is
  non-zero: a metrics block showing only zeros would pass even if it
  were completely disconnected).
- Call `cmd_service_status` directly from Python (A6 —
  a simple and reliable path, no PATH-master race; capsys captures
  stdout). We run the synchronous command via `asyncio.to_thread`
  because it opens its own loop.
- Check the registry block (`pid`, `transport`) and the live block:
  `alive=True`, `restarts=0`, `tool_calls=1`, non-empty `serena=`,
  non-zero `last_tool_call_unix`.
- A second test: the registry points at a live pid, but the socket is
  an ordinary file nobody listens on (exactly the state the registry
  itself cannot distinguish). `cmd_service_status` prints
  `unreachable:` and returns 1.

## §6. Mutations

- **MUT-X**: do not increment `restarts_total` in `note_restart` →
  `test_restarts_total_increments_on_note_restart` fails.
- **MUT-Y**: count errored tool calls → `test_tool_call_metrics_ignore_errors`
  fails.
- **MUT-Z**: do not reset `_backend_started_ns` in
  `mark_backend_not_ready` → `test_health_backend_uptime_resets_on_not_ready`
  fails.
- **MUT-V2**: do not clear `_serena_version` when the backend stops →
  `test_backend_exposes_and_clears_version` fails.
- **MUT-U**: `cmd_service_status` returns 0 on an unreachable
  socket → `test_service_status_reports_unreachable_socket` fails.

All six were run, all six were killed.

## §7. Delta

- MOD `deep_worker_cli/lsp/service/proto/lsp_service.proto` —
  add the new fields (7, 8 in Health; 12–16 in StatusReply).
- MOD `deep_worker_cli/lsp/service/proto/lsp_service_pb2.py`,
  `..._pb2_grpc.py` — regenerated stubs.
- MOD `grpc_server.py` — new state fields, note_restart, dispatch
  increment, Health/Status population.
- MOD `serena_backend.py` — `_serena_version` + reading from
  `initialize`; `serena_version` property.
- MOD `serena_supervisor.py` — `note_restart(True)` on a successful
  restart.
- MOD `service_cli.py::cmd_service_status` — RPC enrichment of the output.
- NEW tests (see §5).

What does NOT change (clarification A4):

- `ServiceSupervisor` (in `supervisor.py` — the service process spawner).
- Bridge (`bridge_loop.py`, `bridge_stdio.py`, `bridge.py`).
- Watcher / change_detector.
- policy_guard, the Serena config YAML (except what is already in the
  project).

What changes (see the §7 files): `SerenaSupervisor` is the only
supervisor touched here, and only to call `note_restart(True)` after a
successful `_restart_once`.

Wire compat is preserved (new fields only).

## §8. Compatibility

- Old clients keep working: read-only access to the new fields in
  protobuf3 yields defaults (0 / ""), no breaking changes.
- The (our) bridge does not look at Health/Status today — the
  extension is transparent to it.
