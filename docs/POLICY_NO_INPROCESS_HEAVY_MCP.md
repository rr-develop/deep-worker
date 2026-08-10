# POLICY: no in-process resource-heavy MCP servers or shared workers

**Status:** ACTIVE (2026-07-31)
**Scope:** all `deep_worker` agents on all providers; all code in this
project and in the main repository. Violations are forbidden; a workaround
requires **explicit direct** user approval, recorded in the design document
of the specific feature.

## 1. Rule

Any resource-heavy MCP server (or other worker) whose state/work can be
**shared between multiple agents** working on one project or on one host
**must** run as a **separate process-service**, and agents must connect to
it as **clients**.

**Forbidden** to start such a worker inside the agent process (in-process,
stdio-child, subprocess-per-agent, thread-in-agent).

## 2. Definitions

**"Resource-heavy"** — one or more of the following traits:

- JVM or a large language runtime (Kotlin LSP, Gradle Tooling API, `jdtls`,
  `metals`, Node with a large footprint like `tsserver` on a monorepo,
  `rust-analyzer`, `clangd`, `pyright --watch`).
- Index/cache > **100 MB** (Serena/LSP workspace, embeddings, ANN index,
  semantic codebase index).
- Persistent resident **RSS > 200 MB**.
- Cold start > **3 s** (Gradle configuration, JVM warmup, tsserver full
  project load).
- Requires an exclusive lock on a shared resource (`~/.gradle/daemon`,
  sqlite writer, exclusive TCP port).
- Holds **open watchers** / file descriptors on the project filesystem that
  are expensive to multiplex.

**"Shareable"** — more than one agent of a single user can realistically
work in one project / on one host. By default treat as **yes** unless the
opposite is explicitly proven in a design document.

## 3. Mandatory service properties

Every service covered by this policy must provide:

1. **One process per sharing unit.** Usually per project (unit-of-sharing =
   `<project root>`), sometimes per host.
2. **Auto-start on demand.** The first agent that needs the service starts
   it if it is not running. **No "the user must start it beforehand"**.
3. **Lifecycle tool** in `deep-worker` (or in the corresponding CLI):
   `serve|status|stop|refresh` — idempotently, with a pid file under
   `~/.deep_worker/<service>/<slug>/` (or `/tmp` where unavoidable), log in
   the standard `~/.deep_worker/logs/`.
4. **Client transport — gRPC** (preferably over a Unix domain socket with
   600 permissions on a one-user host; or TCP loopback + PSK token in
   metadata if UDS is not possible on the platform).
   Forbidden: stdio-child inside the agent, unnamed unix pipes,
   file-polling-based transports.
5. **Event-driven state maintenance.** Watchers / notifications / gRPC
   streaming. Polling is forbidden by a separate policy; it is not revoked
   here.
6. **Economy.** Update only what actually changed (a diff of the project
   structure or its segment, not a full reindex). "Event exists, but the
   diff is empty" → no-op with a log.
7. **Health & handshake.** Liveness probe with a bounded event-driven
   timeout. When the service is unavailable — a clear error to the agent.
   **`stdio`-fallback / in-process fallback is forbidden**.
8. **Permissions.** Directories 700, files 600 (matches `daemon umask fix
   2026-07-04`).
9. **Multi-project.** One service — one `<project root>`. Different projects
   — different instances. Registry `project → pid/port/socket` in
   `~/.deep_worker/<service>/`.
10. **No new threads** inside the agent for "local scheduling" of such a
    service. Everything is asynchronous on the client event loop.
11. **Setsid / detachment from the agent.** The service is NOT a child
    process of the agent. `SIGTERM` of any agent must not bring the service
    down. Double `fork` + `setsid` with `--detach`.
12. **Idle-shutdown.** By default — 30 minutes without connected clients
    → graceful shutdown, so as not to eat RAM forever. The value is
    configurable, but must be > 0.

## 4. What falls under the ban

Not exhaustive:

- Serena / Kotlin LSP / any LSP via `stdio` in each agent's `mcp.json` —
  **a violation**. Allowed only through a per-project service (see
  `DESIGN_LSP_SHARED_SERVICE.md`).
- Any embedding server / semantic-search worker per agent.
- Any Gradle/Maven tooling API started from the agent process.
- Any headless browser (Chromium/Playwright) per agent.
- ANN indexes (Faiss/hnswlib) in the agent's memory, if they can be shared.

## 5. What is NOT under the ban

- Light stateless MCP wrappers (thin CLI wrappers, synchronous functions
  without indexes, without persistent state, cold start < 100 ms, RSS
  < 50 MB).
- Single short-lived subprocesses (`git`, `ripgrep`, a compiler in one-shot
  mode, `curl`).
- Private per-agent workers that **fundamentally cannot be shared** (e.g. a
  private `TaskManager` of a specific agent).
- Thin client **adapters** to the service (MCP↔gRPC bridge — in-process at
  the agent, but without indexes/JVM/watchers; falls under §5, not §4).

## 6. Existing exemptions (grandfathered), migrated as capacity allows

The following implementations violate the policy but were built **before it
was introduced** and are moved to the new model **as capacity allows** (does
not block merging new code). The list is fixed; extending it is forbidden.

| Component | Location | What it violates | Migration plan |
|-----------|----------|------------------|----------------|
| **`semantic_search`** (codebase index, `SentenceTransformer` + tokenizer, embedding model in the agent's RSS) | `semantic_search/`, `deep_worker_cli/semantic_search_path.py`, `deepagents/tools/project_memory.py`, `tools/semantic_search.py` | in-process per agent; embedding-model RSS > 200 MB; cold start > 3 s | move into a per-project service (same pattern as LSP: gRPC + auto-start + slug) |
| **`history_memory` / `dw_hist`** (agent history index, same embedding stack) | `dw_hist/`, `deep_worker_cli/middleware/history_memory.py`, `deep_worker_cli/history_memory_tool.py` | in-process per agent | move into a per-user (or per-agent) service with a common facade with `semantic_search` |
| **`external_mcp.pool`** — existing stdio wrappers of MCP servers | `deep_worker_cli/external_mcp/pool.py` | stdio-child per agent to third-party MCP servers | keep for light third-party MCP servers; move resource-heavy ones to a shared service as they are discovered |
| **Serena LSP via stdio** (current `mcp.json`, `command: serena` field) | `deep_worker_cli/lsp/serena_config.py`, `deep_worker_cli/cli/lsp_command.py` | stdio-child per agent to a heavy LSP | **closed by the first application of this policy** — see `DESIGN_LSP_SHARED_SERVICE.md` |

**Transition-period restrictions:**

- Extend the functionality of grandfathered components **per the policy**
  (new features are designed as a service) even before the full core
  migration.
- Do not add **new** launch points for them (per-worktree, per-thread,
  per-subagent).
- Do not extend their state beyond what already exists — otherwise the
  migration will become more expensive.
- Changes to the list above require a **separate design document** (either
  adding to the list — with a justification of why it cannot be done
  otherwise; or removing from the list — together with the actual
  migration).

Migration design documents will appear separately.

## 7. Designing new services

A new service is accepted for review **only** if a design document exists
with the mandatory sections:

- **Shared scope** — what exactly is shared and why in-process is
  impossible.
- **Lifecycle & auto-start** — start/stop/idle-shutdown, registry,
  permissions.
- **Transport** — gRPC contract (`.proto`), UDS/loopback, authentication.
- **Change detection (event-driven)** — what triggers the update, diff
  snapshot, economy.
- **Failure modes** — worker crash, duplicate pid, stale registry, occupied
  port, network partition (if TCP).
- **Migration** — how existing code and old client configs move over.

Without these sections the reviewer **must reject**.

## 8. Regression lock

The pre-commit hook (`.githooks/pre-commit`) must contain an AST/CI
invariant forbidding:

- adding to `mcp.json` records (in any repository directory and its
  generator) `command:` fields with values from the block-list of
  resource-heavy binaries (`serena`, `kotlin-language-server`, `kotlin-lsp`,
  `pyright-langserver`, `typescript-language-server`, `rust-analyzer`,
  `clangd`, `gopls`, `jdtls`, `metals`, `gradle`, `mvn`, `java`,
  `node ... tsserver`) combined with `transport: stdio` (explicit or
  default);
- writing `subprocess.Popen(['serena', ...])` /
  `asyncio.create_subprocess_exec('serena', ...)` /
  `stdio_client(...)` with a resource-heavy binary in the arguments —
  outside the explicitly allowed allowlist: **only** the modules of the
  `deep_worker_cli.lsp.service*` package (the service itself, which is
  responsible for starting Serena). Bridge / client modules / any agent
  paths are **not** in the allowlist;
- changing the list of existing exemptions without a corresponding design
  document in the commit delta (`docs/DESIGN_*` file in `git diff --stat`).

The exact implementation of the invariant — in the LSP service design (the
first application of the policy).

## 9. Sign-off

Changes to this policy require explicit user approval. Reviews of design
documents applying the policy are conducted by the review lead.
