# DESIGN: LSP-agnostic shared per-project service

**Status:** DESIGN SIGNED OFF (2026-07-31) — ready for TDD
implementation per §15 commit order
**Date:** 2026-07-31
**Branch:** `dev`
**Author:** (internal)
**Parent policy:** `docs/POLICY_NO_INPROCESS_HEAVY_MCP.md` (the first
application of this policy; it closes the "Serena LSP over stdio" section
of the grandfathered list)
**Related:** `f127f68` (Kotlin/Android Gradle init script — a patch
inside Kotlin LSP, stays as is).

## 1. Problem

Today `deep-worker lsp enable` writes a Serena launch over stdio into each
agent's `mcp.json`:

```json
"serena": { "command": "serena", "args": ["start-mcp-server",
             "--context", "ide", "--transport", "stdio",
             "--project", "<path>"] }
```

Each agent **forks its own Serena**. Inside, Serena brings up **its own**
language server (Kotlin LSP / pyright / tsserver / rust-analyzer / …), and
that one brings up **its own** workspace with a cold start of seconds to
minutes (Kotlin+Gradle is the worst case). Five agents over one project =
5 × (JVM + Gradle configuration + LSP index). This is a direct violation
of `POLICY_NO_INPROCESS_HEAVY_MCP` and the root of the user's original
complaint.

The defect is **LSP-agnostic**: it concerns all LSPs, not only Kotlin.

## 2. A single (LSP-agnostic) service

**One long-lived process** `deep-worker lsp serve --project <root>` per
`<project root>`, regardless of language. Inside — Serena, wrapped in a
**gRPC facade**. Agents connect as gRPC clients through a thin in-process
MCP↔gRPC bridge (falls under §5 of the policy: no indexes, no JVM, no
watchers).

Structural diagram:

```
┌──── agent A ────┐    ┌──── agent B ────┐
│  MCP client     │    │  MCP client     │
│  mcp↔grpc bridge│    │  mcp↔grpc bridge│
└──────┬──────────┘    └──────┬──────────┘
       │ gRPC (UDS 600)       │
       └─────────┬────────────┘
                 ▼
     ┌──────────────────────────┐
     │  deep-worker lsp serve   │
     │  (per <project root>)    │
     │                          │
     │  ┌── grpc server ─────┐  │
     │  │  facade over Serena│  │
     │  └────────┬───────────┘  │
     │           ▼              │
     │   Serena (SSE 127.0.0.1) │
     │   ├── kotlin-lsp (JVM)   │
     │   ├── pyright            │
     │   ├── tsserver           │
     │   └── …                  │
     │                          │
     │   change-detector        │
     │   (inotify + debounce)   │
     │                          │
     │   supervisor + idle      │
     │   shutdown (30 min)      │
     └──────────────────────────┘
```

## 3. What it gives

- One JVM/tsserver/rust-analyzer per project, not per agent.
- One Gradle configuration cycle, one LSP model, one Serena index.
- A shared cache: an edit in one agent is visible to the others without
  a per-agent refresh.
- Automatic model invalidation on structural project changes — the
  service knows when the module composition changed (event-driven); the
  user and the agents do not need to restart anything.

## 4. Transport: gRPC

**Choice:** a user requirement. UDS (`SOCK_STREAM`) at
`~/.deep_worker/lsp/<slug>/grpc.sock` with permissions 600 (the socket
itself) and 700 (the directory). TCP loopback + PSK token in
`metadata['authorization']` — a fallback for platforms without UDS (we
run Linux, so UDS by default; the TCP fallback is described for
miscOS/test stands).

**Serena's internal path.** Serena itself does not support gRPC — only
stdio/HTTP/SSE. Our service brings up Serena on **loopback SSE
`127.0.0.1:<internal>` (localhost only, only for its own process)** and
wraps its calls in gRPC. The internal SSE port is not published outside
and is not written to the registry — it exists only inside the service.

**Alternatives considered and rejected:**

1. **Direct agent↔Serena SSE transport** — does not allow adding our
   invariants (auth, health, metrics) and violates the single transport
   declared by the policy.
2. **A fork of Serena with a gRPC transport** — a large scope; the
   upstream custodian changes the API; raises maintenance.
3. **Our own MCP-over-gRPC implementation for Serena without Serena
   inside** — we lose all of Serena's logic (project roles, prompts,
   tools).

**Result:** a thin gRPC facade over a local Serena SSE. If the review
produces strong arguments for a fork, we discuss it separately; **for now
the scope is limited to the facade**.

**Version pin (per the reviewer's note).** Stack: `grpcio` + `grpc.aio` +
`grpcio-tools` (Q1). Exact versions are fixed in the `pyproject.toml` of
the service package at commit 3; in this design only ranges:
`grpcio>=1.60,<2`, `grpcio-tools>=1.60,<2` (matches `grpcio` on MAJOR —
a mandatory requirement of `grpc_tools.protoc`). `protoc` ships with
`grpcio-tools`; no separate install needed. The exact pinned version is
recorded in the service package's README.

**Responsibility for launching Serena lies only with the service.** The
only module allowed to spawn `serena` (including the internal SSE
process) is the package `deep_worker_cli.lsp.service*`. The `bridge`, the
MCP client, and any other project code have **no right** to launch
Serena; this goes into the allowlist of the AST invariant (§14, the
reviewer's note).

### 4.1. `.proto` sketch

```
service LspService {
  rpc Health(HealthRequest) returns (HealthReply);

  // Session (server-stream) — attach + push events (diagnostics,
  // workspace changes, refresh notifications).
  rpc AttachSession(AttachRequest) returns (stream SessionEvent);

  // Unary tools — 1:1 to the Serena MCP tools.
  rpc FindSymbol(FindSymbolRequest) returns (FindSymbolReply);
  rpc FindReferencingSymbols(RefRequest) returns (RefReply);
  rpc FindDeclaration(DeclRequest) returns (DeclReply);
  rpc FindImplementations(ImplRequest) returns (ImplReply);
  rpc GetSymbolsOverview(OverviewRequest) returns (OverviewReply);
  rpc GetDiagnosticsForFile(DiagRequest) returns (DiagReply);
  rpc RenameSymbol(RenameRequest) returns (RenameReply);
  rpc ReplaceSymbolBody(ReplaceRequest) returns (ReplaceReply);
  rpc InsertBeforeSymbol(InsertRequest) returns (InsertReply);
  rpc InsertAfterSymbol(InsertRequest) returns (InsertReply);
  rpc SafeDeleteSymbol(DeleteRequest) returns (DeleteReply);

  // Administrative.
  rpc RefreshProject(RefreshRequest) returns (RefreshReply);
  rpc ProjectStatus(StatusRequest) returns (StatusReply);
}
```

The schema is narrow — only the tools the agent actually uses plus the
administrative `Health` / `AttachSession` / `RefreshProject` /
`ProjectStatus`. It is extended as needed while preserving backward
compatibility.

## 5. Project identification: slug

**Slug** = `sha256(os.path.realpath(project_root))[:16]`.

- Resistant to worktree moves and symlinks.
- Usable as a directory name (no special characters).
- Stored in `~/.deep_worker/lsp/<slug>/`.

Different worktrees of one repository, if they point at the same
`realpath`, share the service. If they point at different `realpath`
(a git worktree with a twin working directory), they get different
services, which is correct: they may have different structures and
different uncommitted changes.

## 6. Registry and disk layout

```
~/.deep_worker/lsp/<slug>/            (700)
├── pid                                (600)
├── grpc.sock                          (600)  # UDS
├── port                               (600)  # only for TCP fallback
├── psk                                (600)  # only for TCP fallback
├── project_root                       (600)  # realpath, for verification
├── started_at                         (600)
├── state.json                         (600)  # structural snapshot
└── log                                (600)  # service stdout+stderr
```

The service's shared log is duplicated to `~/.deep_worker/logs/lsp-<slug>.log`
for uniformity with the other agent logs.

## 7. Auto-start and bridge

The agent **never** launches Serena directly. `mcp.json`
gets a thin wrapper:

```json
"serena": { "command": "deep-worker", "args": [
    "lsp", "bridge", "--project", "<realpath>"
]}
```

`deep-worker lsp bridge`:

1. Reads `<slug>` = `sha256(realpath(--project))[:16]`.
2. Checks `~/.deep_worker/lsp/<slug>/pid` (is the process alive via
   `kill -0`) and `grpc.sock` (a quick `Health` probe with timeout=1s).
3. If the service is alive — opens MCP stdio with the agent, translates
   MCP tool calls into gRPC requests to the service; returns the
   responses as MCP tool results. No local state.
4. If the service is not alive — calls `deep-worker lsp serve --project
   <realpath> --detach` (double-fork + `setsid`, detached from the agent),
   waits **event-driven** for `pid` and `grpc.sock` to appear via
   `inotify` on the `~/.deep_worker/lsp/<slug>/` directory with a total
   timeout (default 30 s). No polling.
5. If the service is still unavailable after the timeout — the bridge
   returns the MCP error `service unavailable for project X` to the
   agent. **stdio fallback is forbidden** by policy.

The bridge is stateless and light (imports: `grpcio` + the MCP SDK). It
may be kept per-agent (§5 of the policy).

## 8. Service lifecycle

CLI:

- `deep-worker lsp serve --project <root> [--detach] [--foreground]` —
  start (idempotent; if the service is already alive — no-op, exit 0).
- `deep-worker lsp stop --project <root>` — SIGTERM → SIGKILL fallback
  (the 2-stage scheme, as in `kill_job`).
- `deep-worker lsp status [--project <root>|--all]` —
  `pid/uds/uptime/attached-agents/watched-modules/last-refresh`.
- `deep-worker lsp refresh --project <root>` — forced refresh (for
  diagnostics; normally not required).

`serve` internally:

- `setsid`, `os.chdir("/")`, close inherited fds (except the log file),
  write `pid`, start Serena SSE on loopback, start gRPC on UDS/TCP,
  start the change detector, start the supervisor with an idle-shutdown
  timer.
- `SIGTERM`/`SIGINT` → graceful shutdown of Serena SSE → close gRPC →
  clean up the registry files.

**Idle shutdown:** 30 minutes without active client connections and
without incoming requests. The value is configurable (`--idle 1800`),
minimum 60 s, 0 forbidden. A "keep-alive" is supported through a separate
`Attach` stream — as long as at least one client holds the stream, the
service does not shut down.

## 9. Change detection (event-driven, LSP-agnostic)

The `change_detector` module lives in the service. Invalidation rules are
given by a **table of providers** (Provider Rules), selected by the
presence of marker files in `<project root>`:

| Provider | Markers | Triggers (watch) | Structure snapshot | Refresh action |
|---|---|---|---|---|
| Kotlin/Android/JVM (Gradle) | `settings.gradle(.kts)` | `settings.gradle(.kts)`, `**/build.gradle(.kts)`, `gradle.properties`, `gradle/libs.versions.toml`, `gradle/wrapper/gradle-wrapper.properties`, `~/.gradle/init.d/deep-worker-*`, create/delete in `src/*/{kotlin,java}` | `{includes[], source_sets[], toml_hash, init_hash}` | Serena `refresh_project` + LSP `didChangeWorkspaceFolders` |
| Java (Maven) | `pom.xml` | `pom.xml`, `**/pom.xml`, `.mvn/**` | `{modules[], deps_hash}` | the same |
| Python (uv/poetry/pip) | `pyproject.toml`/`setup.py` | `pyproject.toml`, `uv.lock`, `poetry.lock`, `requirements*.txt`, `setup.py`, `setup.cfg`, create/delete `src/**/__init__.py` | `{packages[], entry_points[], deps_hash}` | reindex pyright/basedpyright |
| TypeScript/JavaScript | `package.json` | `package.json`, `pnpm-lock.yaml`, `yarn.lock`, `package-lock.json`, `tsconfig*.json`, `pnpm-workspace.yaml`, `nx.json`, `turbo.json` | `{workspaces[], tsconfig_refs[], deps_hash}` | tsserver `project/reload` |
| Rust | `Cargo.toml` | `Cargo.toml`, `Cargo.lock`, `**/Cargo.toml` | `{members[], features[]}` | rust-analyzer `rust-analyzer/reloadWorkspace` |
| Go | `go.mod` | `go.mod`, `go.sum`, `go.work` | `{modules[], require_hash}` | gopls `workspace/didChangeConfiguration` |
| C/C++ | `CMakeLists.txt`/`compile_commands.json` | `compile_commands.json`, `CMakeLists.txt`, `**/CMakeLists.txt`, `meson.build`, `Makefile` | `{targets[], flags_hash}` | clangd re-read compile db |
| Scala | `build.sbt` | `build.sbt`, `project/**.scala`, `project/**.sbt`, `.bloop/**`, `.metals/**` | `{modules[], sbt_hash}` | metals import build |
| **default** (no marker matched) | — | `.git/HEAD`, root `*.toml|*.json|*.yml` | `{git_head, root_manifest_hash}` | Serena `refresh_project` |

Default rules apply as a fallback (never run simultaneously with a
specific provider — otherwise a double refresh).

**Future providers:** Ruby, Haskell, Swift, PHP, Elixir — added to the
table without changing the `Provider` interface. The list is open, but
each addition is a separate commit with a test.

**Mechanics:**

- Watchers: `watchfiles` (a unified wrapper over inotify/kqueue/FSEvents;
  used in the current codebase via the `pyinotify` style per the
  "event-driven" policy). Backend — `inotify` on Linux, at most one
  kernel watcher per directory.
- **Debounce 300 ms** for "save bursts".
- Event handling → recompute the **structure snapshot** of the active
  provider → compare with the previous snapshot:
  - snapshot changed → `refresh_action` of the corresponding provider,
    the new snapshot becomes current;
  - snapshot unchanged → **no-op** with a log entry (an important
    frugality rule).

  **Amendment (commit 8).** The draft proposed storing the snapshot in
  `state.json`. `state.json` belongs to the registry and is written by the
  supervisor from ANOTHER process; a shared writer without a lock is a
  direct path to lost writes (atomic tmp+rename protects against
  half-written files, not against a writer race). So the snapshot lives
  only in the detector's memory, and after a service restart it is
  recomputed from disk
  (`docs/DESIGN_LSP_WATCHER_WIRING.md` §5/§7).
- Edits inside existing sources (`*.kt`, `*.py`, `*.ts`, …) do not
  trigger a refresh. The LSP catches this itself via
  `didChangeWatchedFiles`.
- Snapshot computation is cheap: the trigger files are small, parsing is
  regex-based (`include(":app")`), sha256 of the content. A **full**
  recompute runs at most once per minute — the upper frugality bound.

**Polyglot projects.** If several markers match (Android + a TS web
module in one repo), the rules merge: each provider watches its own paths
and triggers its own refresh independently. No duplicates: an event fires
on one watcher, regardless of how many providers are active.

## 10. Authentication and security

- **UDS + permissions 600 in a 700 directory** for a one-user host — a
  sufficient isolation (`os.stat(sock).st_uid == getuid()` check on the
  client).
- **TCP fallback:** PSK token from `~/.deep_worker/lsp/<slug>/psk` (600);
  the client passes it in `metadata['authorization'] = 'Bearer <token>'`;
  the server rejects without the token. The token is generated
  cryptographically (32 urlsafe bytes) at `serve` and lives until `stop`.
- No network listeners outside loopback. `bind` strictly `127.0.0.1`.
- Permissions inherit the daemon umask policy (2026-07-04).

## 11. Failure modes

| Situation | Behavior |
|---|---|
| TCP port busy (fallback mode) | try the next free one, update `port` |
| pid in the registry but no process | clean up the files, start again |
| Serena/LSP crash | the supervisor restarts **only** the internal Serena SSE (keeping the gRPC connections); N restarts per window — escalation to the log |
| gRPC server crash | the service exits, the bridge gets `service unavailable`, the next agent/attempt auto-starts it |
| UDS socket busy incompatibly (stale) | unlink and bind again |
| Client drops off (RST) | the server closes the corresponding Serena requests, frees resources |
| A refresh is running at a tool-call moment | the tool call is served on the **old** snapshot; refresh is atomic |
| `stdio` fallback requested externally | **refused**; we log it and return an error — policy §7 |

## 12. `mcp.json` migration

- `deep-worker lsp enable` detects the old format
  (`command: serena`) and **rewrites** the entry to the new one
  (`command: deep-worker`, `args: [lsp, bridge, --project, <root>]`),
  keeping a backup `mcp.json.bak-<ts>`.
- Idempotent: a repeated `enable` does not rewrite if the entry is
  already in the new format.
- The Kotlin/Android init script (`f127f68`) stays exactly as is — it
  lives inside Kotlin LSP and does not depend on the service.
- Compatibility: old agents (not updated) keep working over stdio until
  the first `enable`; they see full functional coverage — just the old
  expensive way, as before. New agents go through the service.

## 13. TDD

Unit:

- `test_slug_stable_per_realpath` — the slug does not change on a
  worktree move.
- `test_slug_differs_for_different_realpath` — different projects —
  different slugs.
- `test_registry_stale_pid_is_reclaimed` — a dead pid → restart on
  auto-start.
- `test_uds_file_perms_600_dir_700` — permissions match the policy.
- `test_change_detector_ignores_intra_module_edits` — an edit in `.kt`/
  `.py`/`.ts` → no-op.
- `test_change_detector_triggers_on_settings_include_change` → refresh.
- `test_change_detector_debounces_burst` — 50 events/s → 1 refresh.
- `test_snapshot_diff_empty_is_noop` — an event without a structural
  diff → 0 refreshes.
- `test_provider_dispatcher_picks_kotlin_for_gradle_root`.
- `test_provider_dispatcher_picks_python_for_pyproject_root`.
- `test_polyglot_project_merges_rule_sets` — Android + TS → both
  providers active, refreshes independent.
- `test_default_rules_only_fires_without_specific_provider`.
- `test_bridge_returns_error_when_service_unavailable` — the stdio
  fallback ban: the bridge returns an MCP error, it does not bring up
  Serena itself.
- `test_no_stdio_command_in_written_mcp_json` — a regression invariant
  of the policy (see §14).
- `test_service_setsid_ppid_differs_from_agent` — detach is correct.
- `test_idle_shutdown_after_no_clients` — with not a single client the
  service shuts itself down.
- `test_active_attach_stream_prevents_idle_shutdown`.

  **Amendment (commit 7).** The draft proposed `--idle 2` and waiting
  2 s. This is incompatible with the prod threshold
  `_MIN_IDLE_SHUTDOWN_S = 60 s`, which is validated in the
  `LSPServiceServer` constructor: a service that died after a second
  would die between two agent calls. Relaxing the threshold for the sake
  of the test would mean testing behavior that does not exist in prod, so
  both tests live in `tests/integration_tests/lsp/test_shared_service_lifecycle.py`
  and take a real server with real 60 s, shifting the idle-start
  timestamp (`_idle_since_ns`) into the past. The real event-driven
  watchdog is tested, not its imitation, and the test fits within a
  ≤60 s budget.

Integration (real binaries, isolated `HOME` and `GRADLE_USER_HOME`,
real Serena):

- `test_two_agents_share_one_serena_process` — start two agents over
  one project → **one** Serena process in the system, both see symbols.
- `test_service_survives_agent_restart` — an agent restart does not
  bring down the service.
- `test_service_autostart_from_second_agent_when_first_absent` — the
  first agent is gone, the second brings the service up again.
- `test_add_module_refreshes_index` — `mkdir new-module && add an
  include to settings.gradle` → cross-file references from the new module
  work within debounce + refresh.
- `test_stop_removes_registry_files`.

**State (commit 7).** The *lifecycle* half of the list above is
implemented — `tests/integration_tests/lsp/test_shared_service_lifecycle.py`,
10 tests on a real service process (a real `build_serve_argv` →
`python -m deep_worker_cli.lsp.service`, a real UDS, real detach):

| §13 item | Test |
|---|---|
| `two_agents_share_one_serena_process` | `test_two_agents_share_one_service_process` (+ `test_different_projects_get_different_services` — sharing is per project) |
| `service_survives_agent_restart` | `test_service_survives_the_death_of_its_spawning_process` (+ the structural `test_service_detaches_from_the_agent_that_started_it`) |
| `service_autostart_from_second_agent_when_first_absent` | `test_service_autostart_after_previous_one_is_gone` (graceful exit) and `test_autostart_after_a_crash_that_left_the_socket_behind` (SIGKILL/OOM: the socket outlived the pid — the only case where `is_stale` carries the load) |
| `stop_removes_registry_files` | `test_stop_kills_the_process_and_clears_the_registry` |
| `idle_shutdown_after_no_clients` | `test_idle_shutdown_stops_the_server_with_no_clients` |
| `active_attach_stream_prevents_idle_shutdown` | `test_attached_client_prevents_idle_shutdown` |
| access rights (§5) | `test_socket_and_slug_dir_permissions_are_private` |

Deliberately NOT closed by commit 7 and stays open:

- `test_add_module_refreshes_index` — an end-to-end change-detector e2e.
  Unit coverage of the detector exists (`test_change_detector.py`), but
  there is no end-to-end run of "added a module → symbols from it
  resolve".
- Attaching a real Serena backend through
  `LSPServiceServer.set_tool_handler`. While no handler is installed,
  tool RPCs honestly return `UNIMPLEMENTED` (fail-loud, §4) — the model
  cannot get "found nothing". Because of this,
  `test_serena_lsp_integration.py::test_repo_root_pyrightconfig_kills_false_missing_imports`
  fails; this is the expected state, not a regression.

## 14. AST/CI invariant in pre-commit

A regression lock on the policy (§8 of the policy):

- The checks are implemented in `deep_worker_cli/lsp/policy_guard.py`
  (unit-tested separately); the git plumbing is in
  `scripts/pre_commit_no_inprocess_heavy_mcp.py`, called from
  `.githooks/pre-commit`. There are three checks — one per §8 policy
  item:
  1. `mcp.json` entries (in any repo directory and in the generator)
     with a `command` from the blocklist of resource-heavy binaries on a
     stdio transport — **explicit or default** (a missing `type` must not
     allow bypassing the check). The case where a heavy binary is hidden
     in `args` (`node …/tsserver`) is also caught. Remote transports
     (`sse`, `http`, `ws`) are not flagged: the ban is on in-process
     stdio, not on the mere existence of a binary.
  2. AST of Python sources: `subprocess.Popen/run/call/check_*`,
     `asyncio.create_subprocess_exec`, `stdio_client(...)` with a
     resource-heavy binary — including the nested form
     `stdio_client(StdioServerParameters(command="serena"))` — outside
     the allowlist. **The allowlist is only the package
     `deep_worker_cli.lsp.service*`** (the service itself, whose
     responsibility is to bring up the backend). Bridge / client /
     agent paths are **not** in the allowlist.
  3. Changing the policy file (including the Existing exemptions list)
     without a `DESIGN_*.md` in the same delta.
- The check is deliberately conservative: ordinary `subprocess.run(["git",
  …])` / `Popen(["ls", …])` are not flagged. A lock that "cries wolf"
  would be disabled within a week, and a disabled lock protects nothing.

> **Amendment (commit 6).** The draft of this section named
> `deep_worker_cli.lsp.bridge` as the allowed module. That contradicted
> §8 of the policy, which allows **only** `deep_worker_cli.lsp.service*`
> and explicitly excludes bridge/client/agent paths. The policy is
> authoritative; the implementation follows it, and the bridge by
> construction cannot spawn anything (pinned by AST tests over `bridge`
> and `bridge_loop`). The section text is aligned accordingly.

## 15. Commit order

1. `policy: forbid in-process resource-heavy MCP servers` — only
   `docs/POLICY_NO_INPROCESS_HEAVY_MCP.md`.
2. `docs(lsp): shared-service design` — only this document.
3. `feat(lsp): grpc proto + generated stubs (dev-only)` —
   `.proto` + generated `_pb2*.py`, committed to the repository.
4. `feat(lsp): per-project shared service + auto-start + bridge` —
   supervisor, `serve|stop|status|refresh`, `bridge`, `mcp.json`
   migration.
5. `feat(lsp): event-driven change detection with provider rules` —
   watcher, dispatcher, all providers from §9.

   > **Historical note.** In commit (4), all 11 tool RPCs were annotated
   > `-> pb.ToolResult`, while `.proto` declares a wrapper return
   > (`FindSymbolReply` etc. with a `result` field). The defect did not
   > surface because every stub called `context.abort()` and in fact
   > returned nothing — the first real return would have failed on
   > serialization. Fixed in (5): each RPC returns its own
   > `*Reply(result=...)`; `_unimpl` was replaced with `_dispatch_tool` +
   > `set_tool_handler`. Without a handler installed, the RPC returns
   > `UNIMPLEMENTED`, not an empty result. Takeaway: a stub that always
   > `abort`s does not check the type contract — the surface stays
   > untested on the wire until the first real return.

   5c. `feat(lsp): cmd_bridge serves MCP stdio over BridgeSession` —
   transport. Only after it can `lsp enable` write the
   bridge entry (reviewer constraint).
6. `feat(hook): AST invariant against stdio-heavy MCP regressions` —
   the pre-commit script from §14.
7. `test(lsp): integration — shared service, autostart, refresh on
   module add` — integration tests as in §13.

Each is separate, TDD; design review after (2) and code review before
each `feat`. ff `dev → master` at the end of the whole series.

## 16. Questions for the reviewer

Explicit points where external opinion is wanted before implementation:

1. **gRPC stack:** `grpcio` (official, C-extension) vs `grpclib`
   (pure-Python, asyncio-first). Our agent is pure asyncio, but `grpcio`
   supports an asyncio server (`grpc.aio`). Which do you prefer:
   `grpcio` (more widespread, heavier) or `grpclib` (lighter, fewer
   dependencies, asyncio-native)?
2. **Stub generation and versioning:** commit the generated `_pb2*.py`
   to the repository and keep the pre-commit invariant "`.proto`
   changed → `_pb2*.py` regenerated" (our standard for generated
   artifacts) — or regenerate at build time and not commit (then a build
   step is needed in pre-commit)?
3. **The form of the AST invariant** (§14): is a static `ast.parse`
   enough for the three checks, or is a full CST needed (e.g. `libcst`)?
   In my experience `ast` suffices; I want confirmation.
4. **The policy's grandfathered list** (see
   `POLICY_NO_INPROCESS_HEAVY_MCP.md §6`) — I listed `semantic_search`,
   `dw_hist`, `external_mcp.pool`, "Serena stdio". Anything forgotten?
   A grep over the codebase showed no other heavy in-process workers; I
   want an independent re-check.
5. **gRPC facade vs fork of Serena.** I chose the facade (§4,
   alternative 2). Do you agree the scope is limited to the facade and a
   fork only if the facade hits a Serena API limitation?
6. **Multiplexing the internal Serena SSE.** Serena is single-threaded
   (per the MCP contract, client↔server 1:1). If two agents through the
   bridge call different tools simultaneously — do we serialize at the
   gRPC facade level (mutex + queue) or multiplex with several Serena
   sessions inside one process? The former is simpler, the latter faster
   under concurrent load. Your opinion?
7. **Idle shutdown of 30 min** — an acceptable default, or should it be
   shorter/longer? It affects the "cold start vs RAM" balance.

### Answers (accepted, 2026-07-31)

1. **`grpcio` + `grpc.aio`** — official, mature, a wide ecosystem, an
   asyncio server/client. `grpclib` — only if `grpc.aio` turns out to be
   incompatible with our loop constraints (not the default).
2. **Commit `_pb2*.py` / `_pb2_grpc.py` to the repository** + the
   pre-commit invariant "`.proto` changed → stubs regenerated and
   staged". The exact `protoc` / `grpc_tools` versions are pinned in the
   service package's README.
3. **`ast.parse` is enough** for the three §14 checks. `libcst` is
   overkill. The residual risk (dynamic f-string assembly of binary
   names) should be documented; static analysis cannot catch it anyway.
4. **The grandfathered list is complete for v1:** `semantic_search`,
   `dw_hist`, `external_mcp.pool`, Serena-stdio. Extend only via DESIGN.
5. **Facade over Serena SSE; fork only as a last resort** (a hard API
   limitation not solvable in the supervisor). Scope fixed.
6. **Serialization at the facade (mutex + queue)** for v1. Multi-session —
   a separate design, only if the measured queue latency becomes a real
   problem.
7. **Idle shutdown of 30 min — accepted.** Configurable (`--idle`),
   minimum 60 s, 0 forbidden. Do not go below 10 min by default without
   data.

`DESIGN SIGNED OFF` — proceed to commits 1–2 (documents only), then the
TDD implementation of commits 3–7 with code review per feature.
