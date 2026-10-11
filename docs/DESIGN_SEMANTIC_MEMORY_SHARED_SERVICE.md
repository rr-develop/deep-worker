# DESIGN: shared semantic-memory service (`project_memory` + `history_memory`)

**Status:** DRAFT (2026-08-30), under review
**Date:** 2026-08-30
**Branch:** `dev`
**Parent policy:** `docs/POLICY_NO_INPROCESS_HEAVY_MCP.md` — applies §6
(migration of grandfathered components, the `semantic_search` and
`history_memory`/`dw_hist` rows) and §7 (how new services are structured).
**Reference pattern:** `docs/DESIGN_LSP_SHARED_SERVICE.md`,
`deep_worker_cli/lsp/service/*`; individual techniques are borrowed from a
sibling project's shared service.

### Out of scope

* **Access control** between agents and repositories — explicitly taken out of
  scope by the user (§1.2 (5)); the environment is trusted. The extension
  point is recorded in §10.
* **Auto-indexing of repositories** (`.vector_index`) — policy §6 forbids
  expanding the state of grandfathered components until the migration is
  complete (§16 Q6).
* **Moving the `dw_hist` CLI** onto the service — it is a separate short-lived
  process, not an agent; policy §5 does not forbid it (§16 Q4).
* **Changing the tool output format** — user requirement §1.2 (6), pinned by
  existing tests (§12).
* **Moving indexes to new paths** — the on-disk layout does not change (§6).
* **Streaming indexing progress** — v1 uses a unary RPC (§16 Q3).

---

## 1. Problem

`project_memory` and `history_memory` are the only agent tools that pull the
entire embedding stack into **every agent's process**:
`llama-index` + `LanceDB` + `HuggingFaceEmbedding` → `transformers` → `torch`.

Measured on this host (2026-08-30):

| Metric | Value | Policy §2 threshold |
|--------|-------|---------------------|
| Cold start (stack import) | **7.13 s** | > 3 s |
| RSS after import | **884 MB** | > 200 MB |
| Largest `.history_index` | **1.1 GB** (one agent) | > 100 MB |
| Largest `.vector_index` | **873 MB** (one repository) | > 100 MB |

Sharing scale: **10** repositories with a `.vector_index`, **26** agents with a
`.history_index`, 132 agent directories. The embedding model is the same
(`intfloat/multilingual-e5-small`, `config.py:114`) — so today the same
~884 MB resident footprint is duplicated in every agent that has `-m` enabled
or that works in an indexed repository.

These are exactly the two rows that policy §6 lists as grandfathered
violations with a plan to "move into a service".

### 1.1. Important: the agent does not only read, it also **writes** the index

A common mistake in framing the problem is to assume that moving out search is
enough. It is not. With `enable_history_memory`, the agent imports the indexer
**at construction time**, before any search:

* `agent.py:1413` — `from ...history_indexer import index_all_history_files`
  (unconditional eager import under `-m`);
* `agent.py:1414` — `_on_summarized` re-indexes the history on **every**
  summarization;
* `agent.py:1466`/`:1472` — the startup catch-up may run a full indexing pass
  at startup;
* `agent_process.py:2648` — `--reindex-history`.

Therefore the service must own **both search and indexing**. A search-only
service would leave the heavy stack in the agent and would not solve the
problem.

### 1.2. User requirements (2026-08-30)

1. The tools remain **native deep-worker tools**, not MCP.
2. `project_memory` is extended to work with **any** repository; by default
   (argument omitted) — the agent's working directory.
3. `history_memory` is extended to work with the memory of **any** agent; by
   default — its own memory.
4. If there is no default index, the agent gets an **informative error**
   explaining that there is no index in its directory, but that the tool can be
   used with an explicitly specified directory/agent.
5. Security questions (who may access whose memory) are **out of scope**; the
   environment is trusted (see §10).
6. Otherwise, using the tools looks the same to the agent as it does now.

---

## 2. Shared scope: what is shared and why it cannot be in-process

**The unit of sharing is the host (one process per OS user), not the project.**

This is a deliberate divergence from the LSP service, where the unit of
sharing = `<project root>`. Rationale:

* The dominant resource here is **not** the index but the **embedding model**:
  it is the same (`multilingual-e5-small`) for all repositories and all
  agents, and costs ~884 MB RSS + 7.1 s of loading.
* Per-project sharding would load the model N times and reproduce exactly the
  problem we are eliminating. With 10 repositories that is 8.8 GB versus
  0.9 GB.
* Indexes (LanceDB tables, docstore) are files on disk and are already
  addressed by path; keeping them "next to the project" does not require a
  separate process.
* User requirements §1.2 (2) and (3) directly imply cross-repo / cross-agent
  access from a single agent. With a per-project model, that would mean the
  agent spawning a service for every foreign repository.

For LSP the situation is the opposite: there the resident (the language
server) is **bound** to a specific project tree and is not reused across
projects, so per-project is correct there. The different optima follow from
the different nature of the resident.

**This is an application of policy §3.1, not a workaround.** §3.1 explicitly
allows the unit of sharing to be "usually per project, **sometimes per host**".
§3.9 ("one service — one project root") describes the typical LSP case and is
not a veto on host-wide services; the `history_memory`/`dw_hist` row in the
grandfathered table of §6 already planned a **per-user** service. This is
stated explicitly so that §3.9 is not read as a prohibition.

Caches inside the service stay per-tenant, as they are today in the agent
process: `retriever._index_cache` is keyed by `(repo_path, embedding_model)`
(`retriever.py:76`), `history_retriever._index_cache` similarly. Only **whose**
address space they live in changes.

**Why in-process is not possible:** see §1 — four of the six "resource
intensity" criteria of policy §2 are exceeded by a wide margin.

---

## 3. What this gives

* The embedding model is loaded **once per host**, not in every agent.
* An agent with `-m` starts without the 7.1 s delay and without the 884 MB
  resident footprint.
* Indexing (summarization, catch-up, `--reindex-history`) no longer occupies
  the agent's event loop — the current `asyncio.to_thread` workaround around
  the LanceDB deadlock goes away as well (`history_memory_tool.py:176`, an
  earlier deadlock incident of 2026-06-15).
* Cross-repo / cross-agent access appears (requirements §1.2).
* Two rows of the grandfathered list in policy §6 are closed.

---

## 4. Transport: gRPC over UDS

As in the LSP service and in the sibling project's service: `grpc.aio` + Unix
domain socket `unix:<state_dir>/grpc.sock`, permissions `0600` in a `0700`
directory. `insecure_channel` is correct: authentication is the filesystem
permissions, which are enforced by the kernel (trusted single-user
environment, §10).

Pin the toolchain exactly as in `lsp/service/README.md`: `grpcio==1.76.0`,
`grpcio-tools==1.76.0`, `protobuf 6.33.6`. **Do not** install `grpcio-tools`
from a different MINOR series: `>=1.83` pulls in `protobuf 7.x` and breaks
`grpcio-status` / `google-api-core` on this host.

Stubs are generated with `--pyi_out` and are **committed** (policy §7, LSP
design §16 Q2).

### 4.1. `.proto` schema (sketch)

The package is versioned: `deep_worker.semantic_memory.v1`.

```proto
syntax = "proto3";
package deep_worker.semantic_memory.v1;

// ---------- Common ----------

// The result is opaque to the transport — the same technique as in
// lsp_service.proto: the ready-made agent text travels as is, so the
// output format can be changed without touching the wire schema. This is
// what gives byte-identical output (§12).
message ToolResult {
  string content       = 1;  // ready-made text for the agent, verbatim
  bool   is_error      = 2;
  string error_message = 3;  // set only when is_error
}

// ---------- Administrative ----------

message HealthRequest {}
message HealthReply {
  bool   ready            = 1;
  int32  protocol_version = 2;
  string service_version  = 3;
  int64  uptime_seconds   = 4;
  int32  attached_clients = 5;
  bool   model_loaded     = 6;   // embedding model is already in RSS
}

message StatusRequest {}
message StatusReply {
  int64  pid                 = 1;
  string transport_address   = 2;
  int64  uptime_seconds      = 3;
  int32  attached_clients    = 4;
  bool   model_loaded        = 5;
  string embedding_model     = 6;
  int64  searches_total      = 7;
  int64  index_ops_total     = 8;
  int64  last_activity_unix  = 9;
  repeated string warm_tenants = 10;  // keys of warmed caches
}

// ---------- Search ----------

message ProjectSearchRequest {
  string repo_path = 1;   // realpath; mandatory on the wire (see §5)
  string query     = 2;
  int32  top_k     = 3;
}
message ProjectSearchReply { ToolResult result = 1; }

message HistorySearchRequest {
  string summaries_dir = 1;  // realpath of the target agent's summaries directory
  string query         = 2;
  int32  top_k         = 3;
}
message HistorySearchReply { ToolResult result = 1; }

// ---------- Indexing ----------

message IndexHistoryRequest {
  string summaries_dir = 1;
  bool   clear_first   = 2;   // --reindex-history
}
message IndexHistoryReply {
  ToolResult result = 1;
  int64 files_indexed = 2;
  int64 chunks_added  = 3;
}

// Index kind, so that one RPC serves both families of tools.
enum IndexKind {
  INDEX_KIND_UNKNOWN = 0;
  INDEX_KIND_PROJECT = 1;   // <repo>/.vector_index
  INDEX_KIND_HISTORY = 2;   // <summaries>/.history_index
}

message IndexStatusRequest {
  IndexKind kind = 1;
  string    path = 2;       // repo root or summaries dir
}
message IndexStatusReply {
  bool   exists          = 1;
  bool   complete        = 2;  // all components are in place (see §11.3)
  int64  pending_lines   = 3;  // for history: delta against line_hashes.json
  string problem         = 4;  // human-readable, empty if everything is fine
}

// Tenants that HAVE an index — for the error text of §5.1.
message ListTenantsRequest {
  IndexKind kind  = 1;
  int32     limit = 2;      // 0 = server default (10)
}
message ListTenantsReply {
  repeated string tenants = 1;   // repository paths or agent names
  int64           total   = 2;   // may exceed len(tenants) when truncated
}

// ---------- Session ----------

// Server-stream. Held by the client for the whole lifetime of the agent and
// serves as a keep-alive: while there are attached clients, the idle watchdog
// does not fire.
enum SessionEventType {
  SESSION_EVENT_UNKNOWN   = 0;
  SESSION_EVENT_ATTACHED  = 1;
  SESSION_EVENT_KEEPALIVE = 2;
  SESSION_EVENT_SHUTDOWN  = 3;
}

message AttachRequest { string client_id = 1; }  // agent name + pid
message SessionEvent {
  SessionEventType event_type = 1;
  string message      = 2;
  int64  timestamp_ns = 3;
}

service SemanticMemory {
  rpc Health(HealthRequest) returns (HealthReply);
  rpc Status(StatusRequest) returns (StatusReply);
  rpc AttachSession(AttachRequest) returns (stream SessionEvent);
  rpc SearchProject(ProjectSearchRequest) returns (ProjectSearchReply);
  rpc SearchHistory(HistorySearchRequest) returns (HistorySearchReply);
  rpc IndexHistory(IndexHistoryRequest) returns (IndexHistoryReply);
  rpc IndexStatus(IndexStatusRequest) returns (IndexStatusReply);
  rpc ListTenants(ListTenantsRequest) returns (ListTenantsReply);
}
```

#### Contract extension after sign-off (2026-08-30)

Two items were added **after** the design was signed off; they are recorded
here so that the document does not diverge from the code:

1. **`ListTenants`** — a new RPC. Without it, the accepted answer **A2** ("list
   of available targets in the error text, truncated to 10 + `… (+N more)`")
   cannot be carried out: the client inside the agent has no right to scan
   other people's directories and does not know who has an index at all.
   Enumeration is the job of the service, which already holds the tenant
   registry.
2. **`IndexKind` in `IndexStatus`** — instead of a bare `path`. One RPC serves
   both families (`.vector_index` and `.history_index`); without a
   discriminator the service would have to guess the index kind from the shape
   of the path, which breaks on non-standard layouts.

Both are additive (a new RPC + a new field with a new number) and do not break
wire compatibility.

`AttachSession` is mandatory and must be a `stream`: it is the keep-alive for
idle-shutdown (§7). The server sends keepalive events at an interval of
**< 30 s** so as not to run into gRPC's default http/2 idle timeout (the LSP
service uses 25 s, `grpc_server.py:19`). There is no client-side polling — the
client simply reads the stream.

An important difference from LSP: there, `AttachSession` **holds back** the
`ATTACHED` event until the backend is ready. Here the embedding model is loaded
**lazily, on the first search**, not at service start, so there is nothing to
hold back: attach replies immediately, and the "cold" first request simply
takes longer (those same ~7 s). This is a deliberate choice: an agent that
never called the tool should not pay for loading the model.

### 4.2. All LanceDB/embedding work — on an executor, never on the loop (blocker B2)

Today the agent escapes the deadlock through a forced offload:
`history_memory_tool.py:164-178` ("MANDATORY OFFLOAD — DO NOT run the search
inline") and `summarization.py:1186-1209` ("NEVER revert to an inline call").
The cause is an earlier deadlock incident (2026-06-15): LanceDB bridges into
its **own** background event loop through a blocking
`run_coroutine_threadsafe(...).result()`; executed on the loop thread, it
holds the GIL, LanceDB's loop cannot acquire it → a permanent deadlock.

Moving the code into the service does **not** cancel this physics — it only
moves it into the gRPC process. Therefore the following invariant is fixed:

* **any** call into `llama_index` / LanceDB / the embedding model in the
  service is executed on a **dedicated executor**, not on the `grpc.aio` loop.
  This applies to `SearchProject`, `SearchHistory` **and** `IndexHistory` —
  not just indexing;
* serialization at the tenant level: writing and searching **one** index are
  not executed concurrently (shared index file, one process). The
  implementation is a per-tenant `asyncio.Lock` before dispatching to the
  executor; different tenants run in parallel;
* the unit test `test_no_lancedb_on_loop.py` pins that RPC handlers do not
  call the engine inline.

Side effect: `asyncio.to_thread` disappears from the agent's tools (the client
is natively async), but the offload itself does not disappear — it **moves**
into the service.

### 4.3. Deadlines

| RPC | Client deadline | Rationale |
|-----|-----------------|-----------|
| `Health` | 5 s | A correctness boundary, not tuning: a shorter one would declare a live service dead and cause a duplicate spawn (§7). |
| `SearchProject` / `SearchHistory` | **30 s** | A cold first search includes loading the model, ~7.1 s. The deadline must be **≫ 7 s**, otherwise the very first request after service start would fail. |
| `IndexHistory` | 120 s | A long RPC; if exceeded — `DEADLINE_EXCEEDED`, indexing continues in the service (idempotent, incremental via `line_hashes.json`), and a repeated call finishes the job. |
| `Status` / `IndexStatus` | 5 s | Cheap, do not touch the model. |

We do not implement streaming of indexing progress in v1 — §16 Q3.

---

## 5. Addressing and defaults

**On the wire the address is always explicit** (a technique from the sibling
project: `default_repository` lives only on the client, the server always
requires an explicit id). The "default" is resolved **in the client, inside the
agent**, where its working directory and its name are known.

* `project_memory(query, top_k=5, repo_path=None)`
  * `repo_path is None` → `runtime.state["working_dir"]` → `working_dir`
    kwarg → `Path.cwd()` (exactly the current chain at
    `project_memory.py:139-146`).
  * otherwise — the given path, `realpath`.
* `history_memory(query, top_k=5, agent_name=None)`
  * `agent_name is None` → `runtime.state["summaries_dir"]` (set by
    `HistoryMemoryMiddleware.before_agent`, `history_memory.py:123-138`).
  * otherwise — `deep_worker_home()/agents/<agent>/summaries`.

The parameter names are `repo_path` and `agent_name`, accepted (§16 A1).
Specifically `agent_name` rather than `agent`: in the tool schema `agent` is
confused with `runtime`.

The tenant identifier for caches and logs is `sha256(realpath)[:16]`, like
`project_slug` (`lsp/service/registry.py:120`) and the sibling project's
`repository_id`.

### 5.0. Tool registration does NOT depend on a local index (blocker B1)

Today both middlewares **hide the tool** if there is no default index:

* `middleware/history_memory.py:118-121` — `if enabled and self._has_index:
  self.tools = [history_memory]`, otherwise `self.tools = []`;
* `ProjectMemoryMiddleware.__init__` — the same via
  `has_semantic_index(working_dir)`.

If this gate is kept, requirements §1.2 (2)(3)(4) are **unfulfillable**: an
agent without a local index does not see the tool at all and physically cannot
pass `repo_path` / `agent_name`, and the "informative error" would never be
shown — there would be nobody to show it to.

**Decision.** Tool visibility is decoupled from the presence of a local index:

1. The tool is registered **always** when the `semantic-search` extra and the
   service are available (for `history_memory` — still when `-m` is enabled;
   that is a user flag, not a property of the index).
2. `IndexStatus` determines **not visibility**, but (a) the error text of §5.1
   and (b) the wording of the cognitive hint.
3. The cognitive hint when there is no local index is **different**. The
   current text (`COGNITIVE_HINT`, "ALWAYS check `project_memory` FIRST") is
   harmful in this situation: the agent would keep hammering an empty default.
   Instead — a hint saying that there is no local index and that the tool is
   applicable only with an explicitly specified target, with a list of
   available targets.

This is not a change of the user's requirements but the only way to fulfill
them. Access control remains out of scope (§10).

### 5.1. Informative error when there is no index (requirement §1.2 (4))

If there is no default index, the agent must get not "empty" but an
explanation with a way forward. Text (final wording — §16 Q2):

```
Project memory: no semantic index for '<resolved_dir>'.

This is the agent's working directory and it has not been indexed.
You can still query any repository that HAS an index by passing it
explicitly, e.g. project_memory(query=..., repo_path="/path/to/<name>").

Indexed repositories on this host: /path/to/a, /path/to/b, ...
```

The list of indexed tenants is provided by the service (it already scans the
registry) — this is a cheap `IndexStatus`-like call; without it the advice
"specify explicitly" remains unfulfillable. Likewise for `history_memory`, with
a list of agents that have a `.history_index`.

---

## 6. Registry and on-disk layout

One service per OS user → one directory, no slug sharding:

```
<deep_worker_home>/semantic_memory/      # 0700
├── grpc.sock                            # 0600, created by the service
├── state.json                           # 0600, atomic write
├── spawn.lock                           # 0600, flock
└── service.log                          # 0600 (see §11.6)
```

`state.json` — the same schema as LSP + `start_ticks` from the sibling
project:

```json
{ "pid": 12345, "socket_path": "...", "started_at_ns": 0,
  "protocol_version": 1, "start_ticks": 8202245, "schema_version": 1 }
```

`start_ticks` is field 22 of `/proc/<pid>/stat` (0-indexed 21). Verified on
this host: it is the process start time counter in ticks; when a PID is reused
the value differs. This is **protection against PID reuse**, which the LSP
registry lacks (`registry.py:237-253` relies only on `os.kill(pid, 0)`). We
take the hardened variant.

**Deviation from the LSP original (stated explicitly).** The LSP registry
considers an entry live if `os.kill(pid, 0)` did not raise `ESRCH`. That is not
enough: pids are reused, and a foreign process that inherited the pid of a dead
service makes the supervisor consider the dead service alive forever — the
agent would keep going to a non-existent socket. Our rule:

* `pid <= 0` → stale;
* `ESRCH` → stale (no such process);
* **`EPERM` → also stale.** The process exists but belongs to a different uid —
  so it is **not our** service: the service always runs under the same user
  that started it, so a foreign owner means a reused pid. Considering such an
  entry live is a hole, not a theoretical concern: the service died, the
  socket file remained, the pid was taken over by a process of a foreign uid →
  `ensure_started` returns a dead socket; since the liveness check deliberately
  does not make a Health RPC (§7), nobody lower in the stack can catch this,
  and the client's single retry (§11.2) returns the same dead socket — the
  agent loops;
* pid alive, but `start_ticks` is recorded and **does not match** the current
  one → stale (pid was reused);
* `start_ticks == 0` ("not recorded", old schema) → the check degenerates to a
  pid check rather than declaring the entry dead.

Parsing is robust to spaces and parentheses in the `comm` field: the tail is
taken after the **last** `)`. The lock is `test_registry.py::TestLiveness`.

The write is atomic: `O_CREAT|O_EXCL` into a temporary file in the same
directory → `fsync(file)` → `os.replace` → **`fsync` of the parent directory**
(the last step is from the sibling project's registry; it is what makes the
rename survive a power failure).

Indexes **do not move**: they remain `<repo>/.vector_index` and
`<deep_worker_home>/agents/<name>/summaries/.history_index`. No data migration
is required (§13).

---

## 7. Auto-start and lifecycle

`ensure_started()` is a port of the sibling project's supervisor with
corrections from the LSP supervisor:

1. `mkdir(state_dir, mode=0o700)` + an **explicit `os.chmod(0o700)`** (the
   sibling project lacks this, and its live state directory turned out to be
   0755 — `mkdir(mode=)` has no effect on an existing directory and is masked
   by the ambient umask).
2. `flock(spawn.lock, LOCK_EX)` — serializes the race of N agents.
3. If `_live_state()` → return the socket (the service is already up).
4. **Arm inotify before `Popen`** — closes the race "the service became ready
   before the watch was installed".
5. Launch **without `fork()`**: the agent is multithreaded (asyncio + provider
   stacks), and `fork()` from such a parent can deadlock the child. We use the
   LSP technique — `Popen` of a one-shot launcher that itself `Popen`s the
   service and exits immediately; the service is reparented to init.
   `start_new_session=True` — the service is not in the agent's group, so the
   agent's `SIGTERM` does not take it down (policy §3.11).
6. Waiting for readiness — `select([inotify_fd, pidfd])`: three outcomes
   (ready / died / deadline) are multiplexed by a single blocking kernel call.
   **No polling and no `sleep`** (the polling policy + §3.5).
7. Readiness = a **live `Health` RPC**, not the existence of the socket, plus a
   matching `protocol_version` (a stale binary → considered "not live" and
   replaced).

`_live_state()` is a five-step check: `os.kill(pid,0)` → `start_ticks` match →
`ready_at_ns > 0` → `socket_path.is_socket()` → `Health` RPC with a bounded
deadline (5 s; this is a correctness boundary, not tuning: a deadline that is
too short would declare a live service dead and cause a duplicate spawn onto
the same socket).

**Idle-shutdown** (policy §3.12): 30 minutes without attached clients →
graceful shutdown. Implementation — a watchdog task on the event loop, like
`grpc_server.py:868`, the value is configurable, > 0.

**Lifecycle CLI** (policy §3.3): `deep-worker semantic-memory
serve|status|stop|reindex`, idempotent, log in the deep-worker logs directory.

---

## 8. Change detection (event-driven)

Here is an important difference from LSP: **filesystem watchers are not needed
and are not created**. Indexing is already incremental and is triggered by
domain events, not by a timer:

* `history` — the event "a summarization happened" (`_on_summarized`,
  `agent.py:1414`) and "agent start with a non-empty delta"
  (`_should_catchup_history_index`, `agent.py:788-817`). Both become an
  `IndexHistory` call to the service.
* Economy (policy §3.6) is already ensured: the delta is computed from
  `line_hashes.json` (`_count_pending_history_lines`, `agent.py:712-785`) —
  microseconds, and with a zero delta the embedding model is not loaded at all.
  "Event happened, diff is empty" → no-op with a log entry.
* `project` (`.vector_index`) is indexed by a separate CLI
  (`scripts/index_repo.py`) manually; there is currently no auto-indexing of
  repositories and we **do not add it** (that would expand the state of a
  grandfathered component, which policy §6 forbids).

The invariant "`count_jsonl_lines` splits only on `"\n"`"
(`history_indexer.py:104-116`) is preserved verbatim: `str.splitlines()` breaks
on `U+2028/U+2029` inside JSON strings and once caused endless re-indexing.

### 8.1. `_on_summarized` must not block the agent's loop (blocker B3)

A clarification based on the code (verified): the callback has **two** call
paths.

* `summarization.py:1209` (async path `_aoffload_to_backend`) —
  already `await asyncio.to_thread(self._on_summarized, path)`, with a large
  comment about the earlier deadlock incident;
* `summarization.py:1094` (sync path `_offload_to_backend`) — the call is
  **inline**, `self._on_summarized(path)`. Today this is safe only because the
  sync path itself does not run on the loop thread.

The callback contract is **synchronous** (`Callable[[str], None]`,
`summarization.py:454`). Therefore it is forbidden to make `_on_summarized` a
coroutine or to call a blocking gRPC with a deadline from it on the agent's
loop (policy §3.10, `DESIGN_NO_LOOP_BLOCKING_TOOLS.md`).

Fixed: `_on_summarized` remains a **synchronous** function and performs a
**non-blocking** send of `IndexHistory` (fire-and-forget). Indexing is
idempotent and incremental, so losing a notification does not lose data: the
next agent start finishes the delta through the startup catch-up
(`_should_catchup_history_index`). Waiting for indexing to finish inside the
callback is forbidden on both paths.

**Exactly how a synchronous callback sends gRPC without waiting and without
starting a thread.** The formulation "just don't block" is too coarse and
easily breaks in implementation, so the mechanism is fixed explicitly:

1. When the agent is created, its event loop is remembered
   (`asyncio.get_running_loop()`).
2. `_on_summarized` is an ordinary synchronous function. Inside, it does
   **only** the handoff of a task to that loop:

```python
def _on_summarized(path_str: str) -> None:
    # Synchronous contract (summarization.py:454). We wait for nothing.
    loop.call_soon_threadsafe(
        lambda: asyncio.create_task(_client.index_history(summaries_dir))
    )
```

3. `call_soon_threadsafe` is correct on **both** call paths: both when the
   callback runs on a worker thread (`to_thread`, `summarization.py:1209`) and
   when it runs on the loop thread — in both cases it merely enqueues a
   callback on the loop and returns immediately.

**Forbidden** in the callback: `await` (the contract is synchronous), a
blocking gRPC call, `asyncio.run`, `run_coroutine_threadsafe(...).result()`,
`subprocess`/`Popen`, creating new threads (policy "no new threads without
explicit user approval").

Task errors are logged inside the task itself and do **not** bubble up into
summarization: an indexing failure must not break saving the history. Losing a
notification is safe — the startup catch-up finishes the delta.

The existing `tests/test_no_loop_blocking_summarization.py` stays green and
serves as the lock for this item; `test_on_summarized_non_blocking.py`
additionally pins that the callback creates no threads and does not wait for a
result.

---

## 9. Thin client: how lightness is guaranteed

Rule: **no heavy import may be reachable from the agent process**. The client
imports only `grpc`, the stdlib and the generated stubs.

Location: `deep_worker_cli/semantic_memory/` with an **empty**
`service/__init__.py`. The supervisor lives **inside** `service/` (as in LSP) —
the rationale and the analysis of the alternative are in §9.1. An empty
`service/__init__.py` is a mandatory condition: it is what prevents importing
the supervisor from dragging in `service.grpc_server` and the engine. This is
also the spot of a defect in the sibling project, where the design doc §3.3
claims that the client does not import `service`, while the code imports the
supervisor from its `service` package; we must not have a doc/code divergence,
so the emptiness of `__init__.py` is pinned by a test.

**Regression lock** — an AST test, but **transitive**, unlike the sibling
project's architectural boundary test (it reads only `mcp/*.py`,
non-recursively, and checks a single string `"neo4j"`). Our test walks the
actual import graph from the tools' entry points and forbids `llama_index`,
`lancedb`, `torch`, `transformers`, `sentence_transformers`,
`deep_worker.providers.semantic_search`.

### 9.1. Where the supervisor lives and whether a `policy_guard` change is needed

The previous revision of this document recorded **both** at once (the
supervisor outside `service/` + an allowlist change). The reviewer is right:
this is a contradiction. The choice is fixed below.

**Accepted: the supervisor lives in `semantic_memory/service/supervisor.py`,
i.e. INSIDE `service/`,** as in LSP (`lsp/service/supervisor.py`).

Rationale:

* The client's lightness is ensured not by the file's location but by the fact
  that `service/__init__.py` is **empty** and the supervisor imports only
  stdlib + `grpc` + the registry. This is how LSP is organized; this is what
  the sibling project lacked (its supervisor is also inside `service/`, but
  their design doc claims the opposite — a doc/code divergence).
* The transitive import lock of §9 checks the **fact** of the absence of heavy
  imports along the graph, not the module's location. It is the real
  guarantee; the directory is only a convention.
* Keeping the spawner inside `service/` is a direct requirement of policy §8
  ("only modules of the `*.service*` package may bring up a backend").

Consequence: a change to `policy_guard._ALLOWED_SPAWNER`
(**`deep_worker_cli/lsp/policy_guard.py:61`** — the module lives inside the
`lsp/` package, not at the root of `deep_worker_cli/`, although it applies to
the whole repository) is **needed**. Currently it is a tuple of exactly one
package:

```python
_ALLOWED_SPAWNER: Final = ("deep_worker_cli", "lsp", "service")
```

It becomes a **closed set** of path tuples:

```python
_ALLOWED_SPAWNERS: Final = frozenset({
    ("deep_worker_cli", "lsp", "service"),
    ("deep_worker_cli", "semantic_memory", "service"),
})
```

The semantics do not change: spawning is allowed only from `service/*`. The
list stays closed and grows by one entry — this is a change to the lock's
infrastructure, not a weakening of the policy.

**Discovered during implementation: the allowlist itself was not covered by
tests.** The mutation "remove `semantic_memory` from the list" changed
behavior (`_in_allowlisted_package` started returning False for our
supervisor) but did not fail a single test in the whole repository —
`_ALLOWED_SPAWNER` was not mentioned in `tests/` even once. That is, a change
to the protective mechanism would itself have been unprotected, and the next
revision could have quietly weakened it to the whole package.

Therefore `TestSpawnerAllowlist` is introduced
(`tests/unit_tests/semantic_memory/test_package_boundary.py`), pinning
**both** sides: (1) our `service/` is allowed, (2) `lsp/service/` did not drop
out in the transition from a tuple to a set, (3) modules OUTSIDE `service/`
(`semantic_memory/registry.py`, `agent.py`) are still forbidden. The third
point is not a formality: it catches a weakening of the list to
`("deep_worker_cli", "semantic_memory")`, which would otherwise look like a
harmless simplification.

Note: unlike LSP, our heaviness is not a separate binary but the **import of
Python libraries inside the service itself**, and the spawn goes as
`python -m ...service` (`sys.executable`), which `HEAVY_BINARIES` does not
catch. So the primary mechanism is our transitive import lock of §9, and the
change to `_ALLOWED_SPAWNERS` is needed for consistency of the lock and so that
a future spawn of a binary from our service does not end up outside the
allowlist.

Additionally — an integration check (§14): bring up a real agent with `-m` and
verify that `torch` is **absent** from `sys.modules` and that there is no
`libtorch` in `/proc/<pid>/maps`.

---

## 10. Authentication and security

The user has explicitly taken access-control questions out of scope (§1.2
(5)): the environment is trusted, a single OS user. Therefore:

* Authentication = filesystem permissions: socket `0600` in a `0700`
  directory, `insecure_channel`.
* Checks of the kind "this agent may read that agent's memory" are **not
  performed**.
* No `0777`/`0666`; `os.umask(0o077)` as the first line of `serve()` (a
  technique from the sibling project's server) as insurance for everything
  created afterwards without an explicit mode.
* Files are created via `os.open(..., mode)`, not `open()`/`write_text()`, so
  that there is no window in which a file exists with broad permissions.

If access control is needed later, there is a single extension point: a check
in `_require_tenant_allowed()` on the server. For now it is a no-op.

---

## 11. Failure modes

| Scenario | Behavior |
|----------|----------|
| 11.1 Service does not start | `ServiceStartError` → the tool returns the agent a clear text with the path to `service.log`. **In-process fallback is forbidden** (policy §3.7) — silently degrading into a heavy import is not allowed. |
| 11.2 Service died mid-session | The RPC returns `UNAVAILABLE` → the client re-runs `ensure_started()` **once** and repeats the request (search is idempotent). The sibling project has no reconnect at all and its shim stays broken until restart — we fix this in ours. |
| 11.3 Index missing / incomplete | The informative error of §5.1. We also eliminate a discrepancy: `RepoIndex.exists()` (`config.py:170`) checks only the directory, while `has_semantic_index()` (`project_index.py:28-35`) checks three components. A partially deleted index today passes the `search()` gate but hides the tool. The service uses the **strict** three-component check in both places. |
| 11.4 Stale `state.json` (PID reused) | `start_ticks` mismatch → "not live" → re-spawn (§6). |
| 11.5 Duplicate spawn | `flock` + a repeated `_live_state()` check under the lock. |
| 11.6 Corrupt `state.json` | Treated as absent (like `registry.read`, `registry.py:207-224`), the service is brought up again. |
| 11.7 `semantic-search` extra missing | The error is **in the service**, not in the agent: a `CapabilityError` equivalent with remedy `pip install deep-worker[semantic-search]`. The text reaches the agent via `ToolResult.error_message`. |
| 11.8 Long indexing | `DEADLINE_EXCEEDED` to the client; work in the service continues, a repeated call incrementally finishes the job. |
| 11.9 Service went into idle-shutdown between calls | The ordinary case 11.2: `UNAVAILABLE` → auto-start → retry. |

---

## 12. Tool contract compatibility

Requirement §1.2 (6) — "to agents everything looks as it does now".

Precise wording (corrected after review; the earlier revision claimed that
existing tests would stay green **without changes** — that was wrong):

* **The format of SUCCESSFUL output does not change byte for byte.** This is
  what the model sees, and it remains an invariant.
* **The test files will still have to be edited** — because what changes is
  not the format but the way the dependency is substituted.

Breakdown by test:

| Test | Fate | Why |
|------|------|-----|
| `test_project_memory.py:73` (`Result 1 (score=0.892) [python]`, `Type: parent`) | the mock is **rewritten**, the format assertions remain | it patches `deepagents.tools.project_memory._ensure_semantic_search`; §13.1 removes this symbol, so the mock would no longer refer to anything. Verified: the symbol is patched in 4 places (lines 91, 133, 154, 200). |
| `test_project_memory.py:114` (`Error: Project memory not available.` / `does not have a semantic search index`) | the text is **updated** | §5.1 introduces a different error text. "Byte for byte" and requirement §1.2 (4) are simultaneously unfulfillable; the user's requirement wins. |
| `test_project_memory.py:167` (`Error: top_k must be between 1 and 20`) | **no changes** | `top_k` validation remains in the tool, on the client, before the RPC. |
| `dw_hist/tests/unit_tests/test_search.py:21` | **no changes** | it tests `dw_hist.search.format_text`, not the agent tool; with Q4 = "keep the CLI one-shot" it is unaffected. |

`top_k` validation (1..20) stays **on the client**, before contacting the
service: it is a cheap check, and its text is pinned by the test at `:167`.

The only extension is the new **optional** parameters (§5). Existing calls
without them work as before.

---

## 13. Migration

No data is migrated: index paths do not change (§6).

Order of replacement (see §15):

1. `project_memory.py` — the body of `_run_project_search` is replaced with a
   gRPC call; the lazily imported `_ensure_semantic_search` is removed.
   `asyncio.to_thread` disappears too (the client is natively async).
2. `history_memory_tool.py` — likewise.
3. `agent.py:1400-1483` — instead of the eager import of the indexer, only a
   call to `IndexStatus`/`IndexHistory` on the service remains. This very item
   takes 884 MB off every agent.
4. `agent_process.py:177-207, 2648` (`--reindex-history`) → `IndexHistory`.
5. `middleware/*` — detection of index presence goes through `IndexStatus`,
   without importing `history_indexer`/`project_index`.
6. `dw_hist/cli.py` — a separate CLI process, **not** an agent. It may keep
   loading the stack directly (policy §5: short-lived one-shot). Not touched in
   v1; §16 Q4.
7. Policy §6: the two rows of the grandfathered-exceptions table are **struck
   out** together with the actual migration (§6 "Transition period
   restrictions" requires this). The change to
   `POLICY_NO_INPROCESS_HEAVY_MCP.md` is part of the commit delta — as the AST
   invariant of policy §8 also requires.

Rollback: restore the previous bodies of the two tools; data is untouched.

---

## 14. TDD

Unit tests (no network, no model), `tests/unit_tests/semantic_memory/`:

| File | What it pins |
|------|--------------|
| `test_registry.py` | write atomicity, permissions 0600/0700, corrupt file → `None`, `start_ticks` mismatch → stale, PID reuse |
| `test_supervisor.py` | `flock` serializes two `ensure_started` calls → **one** spawn; "service died before readiness" → `ServiceStartError`; `chmod 0o700` on an existing directory |
| `test_no_polling.py` | AST: neither `time.sleep(` nor `asyncio.sleep(` in `semantic_memory/**` (analogous to the sibling project's boundary test) |
| `test_client_defaults.py` | resolution of `repo_path`/`agent_name` per §5, `top_k` 1..20 validation before the RPC |
| `test_client_retry.py` | `UNAVAILABLE` → exactly one `ensure_started` + retry; a second `UNAVAILABLE` → error to the agent |
| `test_no_heavy_imports.py` | the **transitive** import lock of §9 |
| `test_proto_stubs.py` | proto↔stub consistency, `AttachSession` declared as `stream`, stubs not in `.gitignore` |
| `test_grpc_server.py` | `Health`/`Status`/`AttachSession`, idle watchdog, `idle_shutdown_s` floor |
| `test_index_status.py` | strict three-component index check (§11.3) |
| `test_tool_always_registered.py` | **B1**: the tool is registered even without a local index; the hint in that case is "no local index", not "ALWAYS check FIRST" |
| `test_no_lancedb_on_loop.py` | **B2**: RPC handlers do not call the engine inline; the per-tenant lock serializes write vs search |
| `test_on_summarized_non_blocking.py` | **B3**: the callback stays synchronous and does not block the loop (in addition to the existing `test_no_loop_blocking_summarization.py`) |

Existing tests — per the table in §12: `test_project_memory.py:167` and
`dw_hist test_search.py:21` without changes; `:73` changes its mock (format
assertions remain), `:114` changes its error text. The format of **successful**
output is an invariant.

Integration tests, `tests/integration_tests/semantic_memory/`:

| File | Scenario |
|------|----------|
| `test_shared_service_lifecycle.py` | two `ensure_started` calls → one process; detach (survives the death of the spawner); auto-restart after a crash; `stop` cleans the registry; idle-shutdown; socket and directory permissions |
| `test_search_parity.py` | the real `.vector_index` of this repo and the real `.history_index` of a live agent: result via the service == result of a direct `search()` call |
| `test_agent_has_no_torch.py` | **the key migration test**, see §14.1: an absolute ban on `llama_index`/`lancedb`/`sentence_transformers` + a **differential** delta for `torch`/`transformers` |
| `test_rig_is_event_driven.py` | **B7.2**: AST lock — no file of the suite waits via `sleep`; waits only via `pidfd_open` + `select` |

### 14.1. How "the agent does not pull the heavy stack" is measured

The naive check "`torch` is not in the agent's `sys.modules`" is **wrong and
turns the test green for the wrong reason**: `langchain.agents` itself imports
`torch` and `transformers` before our code does anything. A test written as an
absolute ban on `torch` would fail both before and after the migration,
measuring nothing.

Therefore the check is split in two.

**Absolute ban** — on what must never appear in the agent process, from
anyone: `llama_index`, `lancedb`, `sentence_transformers`. This is our stack;
any appearance of it means that a tool is again dragging the engine into the
process.

**Differential delta** — for `torch` and `transformers`, which foreign code
brings in. Measured as follows: a baseline is taken (`sys.modules` and
`/proc/self/maps`) right after `from langchain.agents import create_agent`,
then an agent with both memory tools is built, then a second snapshot is
taken. The delta must be **empty** — no new module, no new
`libtorch`/`libc10` mapping entry. The test's assertion is "our code adds
nothing", not "there is no torch in the process"; this is exactly the subject
of the migration.

Integration tests (no mocks, policy "integration tests must not hang": budget
≤ 60 s, event-driven, guaranteed `SIGKILL` in `finally`):

* a real service + the real `.vector_index` of this repository:
  `SearchProject` returns a non-empty result matching a direct `search()`
  call;
* the real `.history_index` of an existing agent: `SearchHistory`;
* **the key migration test** (§14.1): build an agent with both memory tools and
  check that `llama_index`/`lancedb`/`sentence_transformers` are absolutely
  absent, and the delta for `torch`/`transformers` and for the
  `libtorch`/`libc10` maps is empty;
* two parallel `ensure_started` calls → **one** service process;
* killing the service between two requests → the second request succeeds
  (auto-restart, §11.2);
* idle-shutdown with a shortened timeout → the process exited on its own.

---

## 14-bis. File delta

**NEW**

```
docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md          (this document)
deep_worker_cli/semantic_memory/__init__.py
deep_worker_cli/semantic_memory/registry.py
deep_worker_cli/semantic_memory/client.py              (thin client)
deep_worker_cli/semantic_memory/service/__init__.py    (EMPTY, §9)
deep_worker_cli/semantic_memory/service/supervisor.py  (inside service/, §9.1)
deep_worker_cli/semantic_memory/service/__main__.py
deep_worker_cli/semantic_memory/service/grpc_server.py
deep_worker_cli/semantic_memory/service/engine.py      (the only place with heavy imports)
deep_worker_cli/semantic_memory/service/service_cli.py
deep_worker_cli/semantic_memory/service/README.md
deep_worker_cli/semantic_memory/service/proto/semantic_memory.proto
deep_worker_cli/semantic_memory/service/proto/*_pb2.py|.pyi|_pb2_grpc.py   (COMMITTED)
tests/unit_tests/semantic_memory/*
tests/integration_tests/semantic_memory/*
```

**MOD**

```
deepagents/tools/project_memory.py          — body → gRPC client
deepagents/middleware/project_memory.py     — IndexStatus ONLY for hint/error;
                                              tool registration per §5.0 (not a gate)
deep_worker_cli/history_memory_tool.py      — body → gRPC client
deep_worker_cli/middleware/history_memory.py— IndexStatus ONLY for hint/error;
                                              tool registration per §5.0 (not a gate)
deep_worker_cli/agent.py:1400-1491          — remove the eager import of the indexer
deep_worker_cli/process/agent_process.py    — --reindex-history → IndexHistory
deep_worker_cli/cli/lsp_command.py          — (none) / new cli/semantic_memory_command.py
deep_worker_cli/lsp/policy_guard.py:61      — _ALLOWED_SPAWNER → set (§9.1)
docs/POLICY_NO_INPROCESS_HEAVY_MCP.md §6    — strike out 2 rows (§13.7)
```

**What does NOT change**

```
deep_worker/providers/semantic_search/**    — the engine stays as is,
                                              only the address space changes
dw_hist/**                                  — §16 Q4
on-disk index layout                        — §6
tool output format                          — §12
```

## 15. Commit order

1. `.proto` + generated stubs + README with the toolchain pin.
2. Registry + supervisor (+ their unit tests).
3. Service: gRPC server, lazy model bootstrap, idle watchdog.
4. Thin client + default resolution + AST lock.
5. Switch `project_memory`.
6. Switch `history_memory` (search).
7. Move indexing out of `agent.py` / `agent_process.py`.
8. Lifecycle CLI (`serve|status|stop|reindex`).
9. Integration tests.
10. Strike the two rows out of policy §6.

---

## 16. Questions for the reviewer

**Q1. Names of the new parameters.** I propose `project_memory(..., repo_path=...)`
and `history_memory(..., agent=...)`. Alternatives: `repo=` / `directory=` and
`agent_name=`. It matters because the name ends up in the args schema and is
visible to the model.

**Q2. Error text of §5.1.** Does it need a list of available
repositories/agents? It makes the advice actionable, but adds an RPC and may be
long with 10 repositories and 26 agents. I propose including it, truncated to
10 entries with the suffix `… (+N more)`.

**Q3. Indexing progress.** In v1 `IndexHistory` is a unary RPC with a deadline,
without progress streaming. Currently, with ≥3 files, the agent draws a rich
progress bar (`agent.py:1439-1469`). After the migration it disappears (a
single "Indexing…" line remains). Acceptable for v1, or should we make it a
server-stream right away?

**Q4. The `dw_hist` CLI.** Do we leave it with a direct heavy import (policy
§5, one-shot process) or also move it onto the service for the sake of a single
model-loading path? I lean towards leaving it in v1.

**Q5. Unit of sharing.** Confirm the divergence from LSP: host-wide instead of
per-project (§2). Rationale — the embedding model is common to all tenants.

**Q6. Auto-indexing `.vector_index`.** I confirm that we do **not** add it in
this feature (policy §6 forbids expanding the state of grandfathered
components). Repository indexing remains manual via `scripts/index_repo.py`.

### Answers (accepted, 2026-08-30)

**A1.** `repo_path` (matches the name of the `search()` argument in the engine)
and `agent_name` — specifically `agent_name`, not `agent`: in the tool schema
`agent` is confused with `runtime`. Both optional, defaults per §5. **Accepted,
applied.**

**A2.** A list of available targets in the error text is **needed**, otherwise
the advice "specify explicitly" is unfulfillable. Truncation to 10 +
`… (+N more)`. Only tenants **with an index** are listed (not all 132 agent
directories). **Accepted, applied in §5.1.**

**A3.** v1 — a unary RPC without progress streaming. Losing the progress bar at
agent start is acceptable; a stream is scope bloat that the user did not ask
for. **Accepted.**

**A4.** The `dw_hist` CLI stays one-shot with a direct heavy import in v1
(policy §5). Pointing the CLI at the service is a separate future item.
**Accepted.**

**A5.** Host-wide is confirmed as an application of §3.1, not a workaround of
§3.9. **Accepted, caveat added in §2.**

**A6.** No auto-indexing of `.vector_index`. **Accepted.**

### Review corrections (round 1, 2026-08-30)

The reviewer confirmed the following key design findings as **correct**:
host-wide (§2), writer+reader (§1.1), lazy model loading with an immediate
ATTACHED (§4.1), strict three-component index check (§11.3).

Blockers closed:

* **B1** → new §5.0. The tool is registered **always**; `IndexStatus` controls
  the error text and the hint wording, not tool visibility. Without this, user
  requirements §1.2 (2)(3)(4) would be unfulfillable: an agent without a local
  index would not see the tool and could not specify the target explicitly.
  Confirmed against the code: `middleware/history_memory.py:118-121`.
* **B2** → new §4.2. All `llama_index`/LanceDB/embedding work — on a dedicated
  executor, **including Search**, not just indexing; a per-tenant lock
  serializes write and search. Moving the code into the service does not cancel
  the physics of the GIL deadlock from the earlier incident, it only moves it
  into the gRPC process.
* **B3** → new §8.1. Clarified from the facts of the code: the async path
  (`summarization.py:1209`) already does `to_thread`, the sync path (`:1094`)
  calls inline. The callback contract is synchronous (`:454`), so
  `_on_summarized` stays synchronous and sends `IndexHistory` non-blockingly
  (fire-and-forget); losing a notification is safe — the startup catch-up
  finishes the delta.

Other:

* **§4.3** — a deadlines table was added. Key point: `Search` = **30 s**,
  because a cold first search includes ~7.1 s of model loading; a short
  deadline would fail the very first request.
* **§12** — the earlier wording "existing tests stay green without changes" was
  **wrong** and has been corrected. Correct: the **successful output format**
  does not change; mocks and error text in tests are edited. Verified:
  `_ensure_semantic_search` is patched in `test_project_memory.py` in four
  places (91, 133, 154, 200) and is removed by migration §13.1.
* **§9.1** — the contradiction "supervisor outside `service/` + allowlist
  change" is eliminated. Accepted: the supervisor **inside** `service/` (as in
  LSP), `_ALLOWED_SPAWNER` → a closed set.
* **§2** — a reference to §3.1 was added so that §3.9 is not read as a veto.
