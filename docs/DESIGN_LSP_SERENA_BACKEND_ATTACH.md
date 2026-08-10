# Design: attaching a real Serena backend to the shared service (commit 9)

Status: DESIGN SIGNED OFF, amendments 1–8 included.
Context: `docs/DESIGN_LSP_SHARED_SERVICE.md` §4 (the contract),
`docs/POLICY_NO_INPROCESS_HEAVY_MCP.md` §8, `docs/DESIGN_LSP_WATCHER_WIRING.md`
§7/§8 (single-flight refresh, backend hook).

## 1. What this commit solves

After commits 1–8 the service is up and watches the filesystem, but the 11
tool RPCs (`FindSymbol`, `FindReferencingSymbols`, `GetDiagnosticsForFile`, …)
honestly return `UNIMPLEMENTED`:

```
CallToolResult(
    isError=True,
    content=[TextContent(text='FindSymbol: no tool backend attached
             to this service')],
)
```

That is as it should be while no backend is attached (fail-loud instead of
an empty result, so the model does not mistake "no backend" for "no
symbol"). Here the backend appears: the service brings up **Serena** as a
child stdio process, talks to it over MCP, and wraps each tool call from
gRPC in `session.call_tool(...)`. After that, agents see real symbols
through the bridge.

Completion criterion (set by the user):
`test_repo_root_pyrightconfig_kills_false_missing_imports` **passes**
on a real backend, and two agents over one project see symbols through
ONE shared service.

## 2. Where the spawn lives

A new module **`deep_worker_cli/lsp/service/serena_backend.py`**.
Everything related to starting/shutting down Serena lives only here.

It falls under the already existing allowlist in `policy_guard.py`
(`_ALLOWED_SPAWNER = ("deep_worker_cli", "lsp", "service")`), and
`serena`/`serena-agent` are already in `HEAVY_BINARIES` — so any
accidental spawn attempt in the bridge/agent/subprojects will be caught
by the hook guard without changing the guard itself.

The bridge (`deep_worker_cli/lsp/bridge*.py`) spawns nothing and knows
nothing about Serena. Neither does the agent. The only route: agent →
bridge → gRPC → service → Serena. One Serena process per project, shared
by all agents (the main policy invariant).

## 3. API shape

```python
class SerenaBackend:
    def __init__(self, project_root: Path, *, command: str = "serena",
                 startup_timeout_s: float = 30.0,
                 call_timeout_s: float = 120.0): ...

    async def start(self) -> None:
        """Spawn serena start-mcp-server --transport stdio --project ROOT,
        MCP initialize, list_tools for a sanity check."""

    async def stop(self) -> None:
        """Softly close stdio (EOF on stdin), wait for exit;
        SIGTERM → SIGKILL on timeout. Idempotent."""

    async def call_tool(self, rpc_name: str, args_json: str) -> pb.ToolResult:
        """RPC name (FindSymbol) → snake (find_symbol) → call_tool on the
        live MCP session; result → gRPC ToolResult. Fail-loud."""

    async def refresh(self, reason: str) -> None:
        """Backend hook for set_backend_refresh_handler:
        calls serena.activate_project (or the refresh equivalent)."""
```

`LSPServiceServer` **owns** the `SerenaBackend` instance (creates it in
`start()`, stops it in `stop()`). Handlers are set once:

```python
backend = SerenaBackend(config.project_root)
await backend.start()
server.set_tool_handler(backend.call_tool)
server.set_backend_refresh_handler(backend.refresh)
server.mark_backend_ready()
```

## 4. Call serialization (answer to Q6)

**One `asyncio.Lock` across the whole backend surface.** Serena stdio is
one pair of pipes; MCP JSON-RPC requires request/response to match by id,
but parallel calls through one session provoke races inside the mcp client
itself and make error recovery extremely hard. Plus the language server
under Serena is often single-threaded.

Therefore:

- `SerenaBackend._lock: asyncio.Lock` is taken for the whole `call_tool`.
- `refresh` takes the same `_lock`. This gives **correct composition with
  the single-flight `_refresh_lock` from commit 8**: the two lock levels
  are never taken in a different order (the server first takes its own
  `_refresh_lock`, then calls `backend.refresh`, which takes
  `backend._lock`; `call_tool` takes only `backend._lock`, never
  `_refresh_lock`), so a deadlock is impossible.
- Readers of the `AttachSession` stream do not care: they take no part in
  these locks; `SessionEvent` publication is synchronous.

What we explicitly do **not** do: a pool of several Serenas. The policy is
one backend process per project; sharding across instances is the exact
opposite. If serial processing becomes a bottleneck, that is a separate
decision with its own design — not now.

## 5. Fail-loud contract

No empty results. Every failure is a `ToolResult` with `is_error=True`
and a non-empty `error_message`, or `grpc.abort(INTERNAL)`.

Matrix:

| Situation | Response |
|---|---|
| Backend not started / dead | `ToolResult(is_error=True, error_message="serena backend not running")` |
| MCP `initialize` fails at startup | `SerenaBackend.start()` raises; the service comes up, but `mark_backend_ready` is NOT called → Health.alive=False, tool RPCs return UNIMPLEMENTED as before (no worse than today) |
| `call_tool` timeout | `ToolResult(is_error=True, error_message="serena {tool} timed out after {N}s")` |
| Serena returned `isError=True` | Passed through as is: `ToolResult(is_error=True, error_message=<flatten content>)` |
| MCP transport error (broken pipe) | `ToolResult(is_error=True, error_message="serena transport failed: …")`, mark the backend as "dead" → the supervisor restarts it (see §7) |
| Unknown RPC name | `pb.ToolResult(is_error=True, error_message="unknown RPC …")` — unified with the other fail-loud paths; the client must not distinguish "server-side isError" from "client-side invalid". The previous plan was `grpc.abort(INTERNAL)` — replaced with ToolResult for uniformity of bridge responses. |

**No stdio fallback**: bridge → service → Serena, period. If Serena does
not come up, tool RPCs stay `UNIMPLEMENTED` and `Health.alive=False` —
the agent sees an explicit refusal, not "the search found no symbols".

## 6. Non-blocker #3 from the previous review

Contract: `_run_refresh` publishes
`SESSION_EVENT_REFRESH_FINISHED` with `message=f"error: {e}"` on error.
The client (agent) must **not** fail neighboring tool calls because of
such an event — the agent asked for `FindSymbol`, not "wait for refresh".

Amendment to the previous design (review of commit 9): **three lines in
`bridge_loop`** (log WARNING + a `last_refresh_error` flag) are **not
added**. The responsibility for interpreting the `error:` message lies
with the consumer of the AttachSession stream on the agent side, not with
the bridge (the bridge is a dumb gRPC↔MCP-stdio translator). The message
itself is already visible to the client through the regular stream flow.
If explicit bridge behavior is ever needed, we add it as a separate
micro-commit with a test, not "write it into the design and forget".

## 7. Lifecycle: crash (commit 9) and restart (next commit)

Serena is a subprocess. Its death is **not** the death of the service.

**In this commit (commit 9): react to the fact of death, but do NOT
auto-restart.**

- A transport exception inside `SerenaBackend.call_tool` /
  `refresh` (broken pipe, closed connection, etc.) counts as the backend
  dying: `_mark_lost()` idempotently clears `self._session = None` and
  calls the `on_lost` hook once.
- `attach_backend` on the server side sets `on_lost =
  server.mark_backend_not_ready`, so the very first error on the session
  pipe flips `Health.alive` back to False (A3 is no longer dead code —
  regression point `test_transport_error_flips_health_alive_off` +
  mutation MUT-M).
- The client sees `is_error=True` with a non-empty message on a failed
  tool call and `Health.alive=False` on the next Health request — an
  explicit "backend died", not a silent `alive=True` over a corpse.

**Deferred to the next commit** (does not affect the series completion
criterion): a pidfd-monitor on the subprocess and automatic restart (up
to 3 attempts in 60 s). Reasons to defer:

1. Here fail-loud is sufficient: the series criterion from the user —
   "two agents through one service see symbols" + pyrightconfig — both
   hold in the absence of crashes.
2. A "crash → restart" test requires a stable rebuild of the language
   server state under Serena, which is unstable in CI without separate
   work on its seeding.
3. Separating one responsibility (fail-loud on death) from another
   (auto-recovery) is cleaner and better aligned with the future
   pool/BACKEND_LOST event design.

The mutation minimum shifts accordingly: MUT-M "remove
`self._mark_lost()` in the transport-error branch" — the pin for commit 9;
the "remove monitor+restart" mutation moves to the design of the next
commit.

## 8. What the tests cover

Unit (mock stdio, no real serena; we check the contract):
- `test_call_tool_serializes_on_single_lock` — two concurrent call_tool
  calls never run in parallel; the "remove the lock" mutation fails.
- `test_call_tool_reports_backend_not_running_as_is_error` — before start().
- `test_call_tool_wraps_transport_error_as_is_error` — read raises.
- `test_call_tool_propagates_serena_is_error` — the session returned an error.
- `test_call_tool_times_out_as_is_error` — the call did not finish in time.
- `test_stop_is_idempotent`.
- `test_bridge_logs_refresh_error_without_failing_next_tool` — under §6.

Extension of `test_policy_guard`:
- `test_spawn_of_serena_outside_service_package_is_flagged` —
  `subprocess.Popen(["serena", …])` from `deep_worker_cli/agent.py`
  fails; from `deep_worker_cli/lsp/service/serena_backend.py` — does not.

Integration (real serena, `-m integration`):
- `test_two_agents_share_one_serena_and_see_symbols` — the series
  completion criterion from the spec.
- `test_serena_lsp_integration::test_repo_root_pyrightconfig_kills_false_missing_imports`
  — must pass on a real backend (currently fails — this is the criterion
  set by the user).
- `test_backend_crash_is_survived_and_recovered` — kill the child
  serena; the next call_tool after a short pause works.

Mutation minimum:
- remove `_lock` around call_tool → test_call_tool_serializes fails.
- return an empty `ToolResult()` in the error paths instead of fail-loud →
  the is_error tests fail.
- remove monitor/restart → test_backend_crash_is_survived fails.
- remove the `set_tool_handler(backend.call_tool)` call →
  test_repo_root_pyrightconfig and test_two_agents fail.
- remove `session.initialize()` → any call_tool fails; caught by
  test_call_tool_reports_backend_not_running_as_is_error (the backend is
  not considered ready, `set_tool_handler` is not set).

## 9. Deliberately out of scope for this commit

- **A separate BACKEND_LOST event.** I use the existing
  DIAGNOSTICS_UPDATED — a proto extension — as a separate commit.
- **A pool of N Serenas.** See §4.
- **A learned restart policy** (jitter, capped exponential). Three
  attempts, then stop — enough; tuning — only if it becomes a problem.
- **Prometheus/OTLP metrics.** The service does not have metrics yet;
  they should be introduced separately and wholesale.

## 10. Review amendments (accepted, DESIGN SIGNED OFF)

All eight reviewer amendments are part of this commit's implementation.
Recorded here in full so that the code reads against this spec, not
against "what I originally sent".

### A1. Remove the unconditional `mark_backend_ready()` in `cmd_serve_impl`

Today `service_cli.cmd_serve_impl` calls `server.mark_backend_ready()`
right after `server.start()` with a "commit 4" comment. As soon as the
readiness criterion is a live Serena, this becomes a lie:
`Health.alive=True` with no backend present.

We remove it. Readiness is set only by `SerenaBackend.start()`
through the server: after a successful `initialize` + `list_tools` +
handler installation, the service calls `mark_backend_ready()`.

Pin: the unit test `test_serve_without_backend_is_not_ready` — the
service is up with `watch_enabled=False` and no backend → `Health.alive is
False`.

### A2. A Serena startup failure does NOT bring down the gRPC surface

The order stays: bind UDS → start gRPC → start watcher → start
backend. `backend.start()` is wrapped in try/except:

- success → `set_tool_handler(backend.call_tool)` +
  `set_backend_refresh_handler(backend.refresh)` +
  `mark_backend_ready()`;
- failure → log error, `_tool_handler` stays `None`,
  `mark_backend_ready` is NOT called, the service keeps running and
  serving administrative RPCs (Health/Status/Attach) — now with
  `alive=False`.

We never let an unhandled exception from `backend.start()`
bring down `__serve_impl__` — otherwise the registry would show the pid
of a live service that does not actually exist.

### A3. `mark_backend_not_ready()` on death and on restart exhaustion

Today there is only `mark_backend_ready() → True`. We add
`mark_backend_not_ready()` (or a parameterized setter). On
detecting Serena's exit → `False`; on a successful restart → `True`.
Three failures in a row within 60 s → `False` forever until an external
`lsp stop && lsp serve`.

### A4. Prod argv from `serena_config`

We build the Serena child process arguments with the same contract as
`build_serena_mcp_entry` (`--context ide`, `--transport stdio`,
`--project <abs root>`, command from `SERENA_COMMAND`). No second
CLI dialect.

Pin: a unit test checks the structural shape of the argv (without
`command`, since our launch wrappers differ).

### A5. Spawn only via `asyncio.create_subprocess_exec` / `Popen`

`os.fork` is forbidden in this module (the same thread-safety rule as
for the supervisor, although here it is the service process itself).

### A6. Shutdown order

`LSPServiceServer.stop()`: first the watcher → then
`SerenaBackend.stop()` (EOF stdin → SIGTERM → SIGKILL on timeout) →
and only then gRPC. Guarantee: idle-shutdown does not leave an orphaned
language server.

### A7. Refresh-tool identification

In `SerenaBackend.start()`, after `initialize`, take `list_tools`,
look for a suitable refresh tool by name (variants found across
serena-agent versions: `activate_project`, `refresh_project`,
`serena__refresh`). Use the first one found; if none is found —
`start()` does NOT fail (call_tool keeps working), but
`refresh` returns `is_error=True` with an explicit message; a test
pins this. Hardcoding a single name that drifts between Serena versions
without checking the list is forbidden.

### A8. `args_json` → MCP args: skip `null`

We parse the JSON into a dict; keys with value `None` (== JSON `null`)
are **dropped**, so strict optional-typed server schemas do not break,
as already decided in `external_mcp` (the `_UNSET` pattern).

### Reviewer non-blockers

- A temporary overload of `DIAGNOSTICS_UPDATED` for Serena's death: the
  message must contain the substrings `serena` and `died` / `rc=` so
  clients/logs can filter. A separate `BACKEND_LOST` — later.
- Units with `watch_enabled=False` + explicit `mark_backend_ready()` for
  a clean gRPC surface remain valid (only for the gRPC-protocol stand,
  not for a real production scenario).
- The crash-integ "short pause" is an event-driven readiness wait
  through AttachSession, no `sleep`.

### Mutations (§8) — added on the reviewer's advice

- **MUT-J**: return the unconditional `mark_backend_ready()` to
  `cmd_serve_impl` → `test_serve_without_backend_is_not_ready` fails
  (Health.alive=True without Serena).

Everything else — per §8 unchanged.

## A9 — readiness handshake strictly event-driven (post-review edit)

**Problem.** The first revision waited for backend readiness in the bridge
via a `while True: Health(...); await asyncio.sleep(0.25)` loop. This is a
direct violation of the project's no-polling rule: a full unit run gave
**two regressions** — `test_no_polling_policy.py::test_no_polling_in_deep_worker_cli`
and `test_no_polling_policy_all.py::test_no_polling_anywhere`. I fix the
CODE, not the test: the AST guard fired correctly.

**Solution.** Polling is removed entirely; readiness travels over the
already existing `AttachSession` stream:

1. The service withholds the `ATTACHED` event until `_backend_ready_event`
   is set (it is set by `mark_backend_ready()` from `attach_backend`).
   Ceiling — the knob `_backend_ready_wait_s` (default
   `_DEFAULT_BACKEND_READY_WAIT_S = 60.0`).
2. When the ceiling expires, `ATTACHED` is still sent, but with the
   `backend-not-ready` marker in `message`. The stream never "sticks":
   the bridge unblocks, and tool RPCs fail loudly (UNIMPLEMENTED) with
   the real reason — instead of a mysterious `connect()` hang.
3. The bridge simply blocks on `self._attached` (set by the stream
   reader) with the ceiling `_ATTACH_READY_S = 75.0` — one wait covers
   both session registration and Serena startup.

Why the bridge ceiling (75 s) is larger than the server one (60 s): the
server must fire first and send a tagged `ATTACHED`; if the bridge
expired earlier, it would tear down the connection and lose the
diagnostic of the cause.

**Side effect (positive).** Previously `AttachSession` in unit tests
waited the full 60 s (the test `test_attach_session_yields_attached_then_keepalive`
took 60.11 s). With the knob, the `tests/unit_tests/lsp/` run dropped
from ~83 s to ~25 s.

**Regression points:**

- `test_attached_is_withheld_until_backend_becomes_ready` — `ATTACHED`
  does not leak while the backend is not ready.
- `test_attached_is_emitted_tagged_when_backend_never_readies` —
  on timeout the event arrives and is tagged `backend-not-ready`.
- `test_connect_blocks_until_the_backend_is_ready` — `connect()` does
  not return before the backend is ready.

**Mutations:**

- **MUT-N**: remove the readiness gate in `AttachSession` (send `ATTACHED`
  immediately) → both `test_attached_is_*` tests fail.
- **MUT-O**: `timeout=None` in the gate (wait forever) →
  `test_attached_is_emitted_tagged_when_backend_never_readies` fails.
- **MUT-P**: the bridge does not wait for `_attached` in `connect()` →
  `test_connect_blocks_until_the_backend_is_ready` fails.

MUT-P **survived the first attempt**: the `running_service` fixture
called `mark_backend_ready()` before all `connect()` calls, so the wait
was not exercised at all. That was a test weakness, not a code one — a
separate test was added that brings up the server without a ready backend.
