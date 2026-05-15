# DESIGN: Persistent, human-inspectable summarization storage

**Status:** ✅ IMPLEMENTED
**Date:** 2026-04-23
**Related bugs & prior work:**
- `feat(web UI): render summarization turns in Full Dialog` — shows the summarization node so the reader can see the path claimed in it.
- A companion design document — why the summarization node is a first-class turn in Full Dialog.
- This document: fixes what that node POINTS AT on disk.

---

## 1. Problem

When `SummarizationMiddleware` (`deepagents/middleware/summarization.py`) compacts an
agent's context, it:

1. Offloads the evicted messages as markdown to a backend-relative path
   `{history_path_prefix}/{thread_id}.md` (line 373 of `summarization.py`).
2. Injects a synthetic `HumanMessage` into the graph state whose body reads:
   > "The full conversation history has been saved to
   > `/conversation_history/my-agent.md` should you need to refer back to
   > it for details."

Three things are broken about *where* that file ends up in the current
`deep_worker_cli` wiring.

### 1.1 The path in the message is virtual and misleading

`deep_worker_cli/agent.py` (lines 775–791) constructs the
`conversation_history` backend as:

```python
conversation_history_backend = FilesystemBackend(
    root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
    virtual_mode=True,
)
composite_backend = CompositeBackend(
    default=backend,
    routes={
        "/large_tool_results/": large_results_backend,
        "/conversation_history/": conversation_history_backend,
    },
)
```

The path the summarization node quotes (`/conversation_history/my-agent.md`)
is NOT a filesystem path — it is a *virtual* path that exists only inside
the `CompositeBackend` router. The real file lives at
`/tmp/deepagents_conversation_history_<8-random-chars>/my-agent.md`, with
that random suffix only known to the agent process itself. For the
`my-agent` agent today, that is
`/tmp/deepagents_conversation_history_<random-suffix>/my-agent.md` (755 KB,
created Apr 22 17:40 — confirmed on disk during analysis).

A human reading the summarization node sees a path they cannot `cat`.

### 1.2 `/tmp` is ephemeral

`tempfile.mkdtemp` under `/tmp` is wiped on reboot on most Linux
distros, and on many systems it is tmpfs (RAM-backed) — meaning even a
shutdown without an explicit cleanup clears every offloaded
conversation.

Agents that ran for days, accumulated hundreds of megabytes of evicted
context, and whose surviving summarization nodes STILL quote that path
would be silently pointing at nothing after the next reboot. The
checkpoint state would reference a file that no longer exists anywhere
in the universe.

### 1.3 Every process restart creates a fresh empty tempdir

`tempfile.mkdtemp` returns a brand-new directory on each call. Every
time `deep-worker --restart my-agent` runs, a new empty
`deepagents_conversation_history_<different-suffix>/` is created. The
agent's own checkpoint-state still references the OLD path it minted
before the restart, but the NEW process's `FilesystemBackend` is rooted
at the NEW directory — so if the agent does try to read back its own
offloaded history (e.g. by asking its `read_file` tool to open
`/conversation_history/my-agent.md`), it gets an empty file or
`FileNotFoundError`, not the 755 KB that used to be there.

Observed today in `/tmp`: **~190 empty
`deepagents_conversation_history_*` directories**, only **4 with
actual content**. The vast majority are orphaned tempdirs from prior
restarts.

### 1.4 No way for the user to browse / manage offloaded history

There is no CLI command to list, show, or prune offloaded
conversation histories. They are invisible from outside the agent.

---

## 2. Goals

1. **The path quoted in the summarization node must match a real file on a
   real, non-random, persistent filesystem path** that the user can
   `cat`, `grep`, `less`, `wc -l` directly.
2. **The path must be stable across process restarts** — the same agent
   restarting must keep seeing (and writing to) the same offload
   directory, so the summarization message from a prior run still
   points at something real.
3. **The path must survive reboots** — no `/tmp` or any other `tmpfs`
   location.
4. **Multiple summarizations per agent are preserved as a history, not
   overwritten.** The user wants to see *all* past summarizations for an
   agent and easily identify the most recent one.
5. **CLI management**: extend the existing `deep-worker --remove-before
   DATE` command so it also evicts old summarization files; add
   `deep-worker --list-summaries [--agent NAME]` to inventory them.
6. **Backward-compatibility & incrementalness are preserved** —
   `SummarizationMiddleware` already appends per-event sections to a
   single thread file via `backend.edit(...) / backend.write(...)`. We
   MUST NOT break that append-only semantics.
7. **No sync regression, no polling, no new threads, event-driven,
   fire-and-forget where applicable** — per the project-wide rules. The
   offload path inside `SummarizationMiddleware` already has a sync
   `_offload_to_backend` and an async `_aoffload_to_backend`; this
   design keeps both paths intact.

### Non-goals

- **Not** changing `SummarizationMiddleware` itself. It already accepts
  `history_path_prefix` as a constructor parameter. The fix is at the
  **wiring** site in `deep_worker_cli/agent.py`, plus the CLI
  `--remove-before` / `--list-summaries` extensions.
- **Not** migrating existing `/tmp/deepagents_conversation_history_*/*`
  files. They will be left where they are; a one-time cleanup can be
  done by the user manually or via a helper script in a follow-up. The
  concern of this design is "from now on, where does it go". (That said,
  we WILL document the one-liner to move existing data if the user
  wants.)
- **Not** changing the Full Dialog renderer or the web UI. The
  summarization node in Full Dialog will continue to display whatever
  path the middleware injected; the fix is that the injected path now
  actually works.

---

## 3. Design

### 3.1 New on-disk layout

```
<data-root>/<agent_name>/
├── session_metadata.json     (existing)
├── ...                       (existing — logs dir, pid files, etc.)
└── summaries/                (NEW — per-agent summary directory)
    ├── 2026-04-22T15-40-18Z.md
    ├── 2026-04-23T09-15-02Z.md
    └── ...
```

**Rationale for each choice:**

- `<data-root>/<agent_name>/` is the already-blessed per-agent data
  directory (`agent_process.py:205`, `agent_stop.py:62`,
  `cli/agent_command.py:36/892`). Putting summaries there co-locates
  them with `session_metadata.json` and the per-agent logs. No new
  top-level directory convention to invent.
- `summaries/` subdirectory rather than a flat `{agent}_summary.md`:
  addresses goal #4 (one file per summarization event; easy to list,
  sort by timestamp, delete individually, or keep-only-latest).
- Filename `<ISO-8601-timestamp>.md` where colons are replaced with
  dashes and the `+00:00`/microsecond tail is dropped in favour of a
  trailing `Z`. The `Z`-suffixed minute-precision ISO form is sortable
  lexicographically, filesystem-safe (no colons — matters on Windows,
  though we target Linux), and matches what
  `_offload_to_backend` already puts inside the file body as its
  `## Summarized at <iso>` section header — so `ls` and grep of
  in-file headers stay in lock-step.
- **NOT** `summaries/<agent_name>.md` with append-only sections. That
  was the old shape. It makes "find the most recent summary" require
  parsing file contents. The new shape `summaries/<ts>.md` lets the
  user (and `--list-summaries`) answer "which was last?" with
  `ls -1 | tail -1`.

### 3.2 Virtual path the agent sees in its summarization node

The message body generated by `_build_new_messages_with_path` is:

> "The full conversation history has been saved to `{file_path}`..."

With this design:

```
{file_path} = "<data-root>/<agent>/summaries/<ts>.md"
```

i.e. **an absolute, real, unambiguous filesystem path** the user can
`cat` directly from a shell. `{file_path}` is the same string the
backend writes to (no virtualization).

This is achieved by constructing `SummarizationMiddleware` with
`history_path_prefix` set to the agent's real summaries directory, and
by NOT putting a `CompositeBackend` route in front of it. The offload
goes straight through the default backend (the agent's working-dir
`FilesystemBackend`), which resolves absolute paths without rewriting.

### 3.3 Filename granularity — one file per event

Current middleware writes one file per thread and appends a
`## Summarized at <ts>` section on every summarization event:

```python
# summarization.py:617 (sync) / :688 (async)
path = self._get_history_path()   # = {prefix}/{thread_id}.md
...
new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(...)}\n\n"
existing = backend.download_files([path])[0].content or ""
backend.edit(path, existing, existing + new_section) if existing else backend.write(path, new_section)
```

This append-to-one-file behaviour is ALREADY a middleware-internal
implementation detail. We change it at the wiring-point level only:
`_get_history_path` uses `{prefix}/{thread_id}.md`, so if `prefix` is
**dynamic per-event** (e.g. includes the current timestamp), we get
one file per event for free.

**Chosen mechanism**: set `history_path_prefix` per agent to
`<data-root>/<agent>/summaries` and leave `{thread_id}.md`
untouched. That still gives one file per (agent, thread) which **appends
sections**, which is the middleware's native mode.

**But goal #4 says "multiple summarizations per agent are preserved as a
HISTORY, not overwritten".** Append-mode DOES preserve the history —
just inside one file. That's what the middleware offers out of the
box; no patching of middleware is needed.

**DECISION:** keep append-mode (one file per `thread_id`, sections
within) — because:
1. It is the middleware's native, tested, async-safe behaviour.
2. "Find the latest summarization event" is already trivial: last
   `## Summarized at` header in the file (one grep).
3. Changing the middleware to "one file per event" forces us to
   subclass / patch it, violating the constraint "do not touch the
   middleware". Wiring-only fix is sufficient.
4. The file sizes observed (755 KB for `my-agent`, 660 KB for
   `other-agent`) are tiny vs. the checkpoint DBs (1.7 GB). No storage
   pressure.

So the **actual** disk layout becomes:

```
<data-root>/<agent>/summaries/
├── <thread_id>.md                 # e.g. my-agent.md
│     ↳ contains:
│          ## Summarized at 2026-04-22T15:40:18.122420+00:00
│          <messages>
│          ## Summarized at 2026-04-23T11:15:04.501100+00:00
│          <messages>
│          ...
├── my-agent:tm.md                 # TaskManager sub-thread, if it summarizes
└── <uuid>.md                      # subagent threads (keep UUID filename,
                                   #  since that IS their thread_id)
```

### 3.4 Why a path with `:` inside filename (e.g. `my-agent:tm.md`) is OK

The `:tm` convention is already used in this repo for TaskManager
sub-threads on Linux filesystems. ext4/btrfs/xfs handle `:` in
filenames fine. Moving to Windows would force a rename; not a concern
today.

### 3.5 CompositeBackend route: removed

In `deep_worker_cli/agent.py` we currently have

```python
composite_backend = CompositeBackend(
    default=backend,
    routes={
        "/large_tool_results/": large_results_backend,
        "/conversation_history/": conversation_history_backend,  # ← REMOVE
    },
)
```

With the new design, the summarization middleware writes to
**absolute real filesystem paths** under `<data-root>/<agent>/summaries/`.
The default `backend` (the local `FilesystemBackend` rooted at the
agent's working dir with `virtual_mode=False`) already handles absolute
paths correctly; no routing required.

**`/large_tool_results/` route stays** — that is a separate concern not
in scope. (Though the exact same argument applies; a follow-up could
move those out of `/tmp` too. Out of scope here.)

### 3.6 Wiring change (proposed, not yet written)

```python
# deep_worker_cli/agent.py — replace the conversation_history_backend block with:

if sandbox is None:
    # Large tool results: unchanged, still go to a tempdir (not this design's concern)
    large_results_backend = FilesystemBackend(
        root_dir=tempfile.mkdtemp(prefix="deepagents_large_results_"),
        virtual_mode=True,
    )
    composite_backend = CompositeBackend(
        default=backend,
        routes={
            "/large_tool_results/": large_results_backend,
            # /conversation_history/ route REMOVED — summarization writes
            # to absolute <data-root>/<agent>/summaries/ paths via the
            # default backend.
        },
    )
else:
    composite_backend = CompositeBackend(default=backend, routes={})
```

And at the site where `SummarizationMiddleware` is constructed (search
`grep -rn 'SummarizationMiddleware(' deep_worker_cli/` — only one
place, in `agent.py`'s middleware assembly around line 660):

```python
# Compute a per-agent summaries directory and ensure it exists:
summaries_dir = Path("~/.deep_worker").expanduser() / agent_name / "summaries"
summaries_dir.mkdir(parents=True, exist_ok=True)

summarization_middleware = SummarizationMiddleware(
    model=summarization_model,
    backend=composite_backend,
    history_path_prefix=str(summaries_dir),   # absolute, real path
    # ...other params unchanged
)
```

Exactly one new line (`history_path_prefix=...`) in the middleware
constructor, plus the `mkdir` call right above it. The default backend
already writes to absolute paths without rewriting, so
`_get_history_path()` returning
`<data-root>/<agent>/summaries/<thread_id>.md` will be
written exactly there.

### 3.7 `--list-summaries` CLI command

New subcommand — parallel to the existing `--list` (threads) and
`--history` (messages):

```
deep-worker --list-summaries [--agent NAME]
```

Output (text form):

```
Agent           Thread                      Events  Latest                    Size
--------------  --------------------------  ------  ------------------------  --------
my-agent         my-agent                         7  2026-04-22T15:40:18Z      755 KB
my-agent         my-agent:tm                      2  2026-04-19T08:22:01Z      112 KB
other-agent      other-agent                      1  2026-04-23T17:55:56Z      660 KB
```

Where:
- **Agent** — from directory name.
- **Thread** — filename stem (before `.md`).
- **Events** — number of `## Summarized at ` headers in the file.
- **Latest** — the last such header.
- **Size** — bytes of the file.

`--format json` support, mirror of `--list --format json`, for
tooling.

Implementation lives next to `list_threads_command` in
`cli/agent_command.py`. Reads only `<data-root>/*/summaries/*.md`.
No DB, no network, no running-agent dependency — works fully offline.

### 3.8 `--remove-before DATE` semantic extension

Today `--remove-before DATE` evicts checkpoint rows from per-agent
`sessions_<name>.db` files whose `updated_at < DATE`. We extend it so it
also:

1. For each `<data-root>/<agent>/summaries/<thread_id>.md`, parses
   the file for `## Summarized at <iso>` headers.
2. If **all** headers are older than `DATE` → delete the file (atomic).
3. If some are older and some are newer → **rewrite the file** keeping
   only the sections with header ≥ DATE. The first-section-header
   timestamp of the rewritten file becomes its new "earliest". Use an
   atomic rename (write to `<file>.tmp` then `os.replace`).
4. If **no** headers are older than DATE → leave the file untouched.

`--dry-run` prints what it would do without writing.

This mirrors how `--remove-before` already handles per-thread DB pruning.

### 3.9 Preservation of incremental cache (deep-worker principle)

The middleware's offload path does not touch `MessageDictCache` —
offload happens on the agent side reading `state.values["messages"]`,
serializes via `get_buffer_string`, and writes via the backend. The
delta-checkpointer cache is orthogonal and stays intact. No cache
invalidation is added or removed by this design.

Confirmed by reading `_offload_to_backend` (lines 597–666 of
`summarization.py`): it uses `get_buffer_string` and backend I/O only;
no `MessageDictCache` coupling anywhere in that method.

### 3.10 Async / event-driven compliance

- The middleware already exposes both `_offload_to_backend` (sync) and
  `_aoffload_to_backend` (async). No change to either. The `a`-variant
  stays `await`-friendly.
- No polling introduced. The offload is driven by the
  summarization-trigger event inside the graph, same as today.
- No new threads created. `FilesystemBackend` is sync over `pathlib` —
  that is NOT a new thread, it is a blocking I/O call inside the
  existing agent event loop. The file is tiny (700–800 KB typical),
  write is milliseconds, so not a concern. Matches the middleware's
  existing sync write path.
- Fire-and-forget: doesn't apply here because the summarization result
  must be awaited before the graph returns the new `HumanMessage`. The
  existing flow (`await self._aoffload_to_backend(...)`) is the correct
  boundary. We don't change it.

---

## 4. Migration & rollback

### 4.1 Existing offloads in `/tmp` — untouched

The ~190 orphan tempdirs under `/tmp/deepagents_conversation_history_*`
are NOT migrated by this change. Two that currently hold real data
(`.../<random-suffix>/my-agent.md`, `.../<random-suffix>/other-agent.md`) are simply
stranded — their existing HumanMessage in each agent's state still
points at the old path, which no longer works once the route is
removed.

We will document a **one-shot manual migration** for the user:

```bash
# One-time: move stranded offloads into the new layout.
for d in /tmp/deepagents_conversation_history_*; do
  for f in "$d"/*.md; do
    [ -f "$f" ] || continue
    thread=$(basename "$f" .md)
    # heuristic: if thread name is a UUID-4, we don't know the agent;
    # otherwise assume thread == agent (the common case).
    case "$thread" in
      ????????-????-????-????-????????????) echo "skip $f (unknown agent for UUID thread)"; continue ;;
    esac
    dst="$HOME/.deep_worker/$thread/summaries"
    mkdir -p "$dst"
    mv -vi "$f" "$dst/$thread.md"
  done
done
```

Running agents should be **stopped** before migration, otherwise a
fresh summarization mid-migration might collide. Documented in the
user guide update (§5 below).

### 4.2 Rollback

Rollback is trivial: revert the three-line patch to `agent.py`. The
`<data-root>/<agent>/summaries/` directories left behind are
harmless.

---

## 5. Tests (TDD)

### 5.1 Unit tests (wiring)

`deep_worker_cli/tests/unit_tests/test_summarization_storage.py` (NEW):

1. `test_history_path_prefix_is_per_agent_absolute_path` — construct
   agent with name `"foo"`, assert
   `middleware._history_path_prefix == f"{Path('~/.deep_worker').expanduser()}/foo/summaries"`.
2. `test_summaries_directory_is_created_before_middleware_runs` —
   assert `.exists() and .is_dir()` after agent construction.
3. `test_composite_backend_has_no_conversation_history_route` —
   assert `"/conversation_history/" not in composite_backend.routes`.
4. `test_large_results_route_still_present` — negative control,
   verifying we did not nuke too much.
5. `test_no_tempdir_created_for_conversation_history` — monkey-patch
   `tempfile.mkdtemp`; after constructing the agent, its call count
   for any prefix starting with `"deepagents_conversation_history_"`
   must be 0. (Allows 1 call for `deepagents_large_results_`, which
   stays.)

### 5.2 Integration test (real write, real filesystem, no mocks)

`deep_worker_cli/tests/integration/test_summarization_writes_to_home.py`
(NEW):

1. Spawn a deep-worker agent with a unique name in a temporary home directory.
2. Send it enough user messages to trigger summarization at a low
   threshold (use `trigger=("messages", 5)` or similar).
3. Assert file exists at
   `<data-root>/<name>/summaries/<name>.md` with non-zero size.
4. Assert file contains a `## Summarized at ` header.
5. Via RPyC, fetch the agent's history and assert msg[0].content
   contains the SAME absolute path that the file actually lives at.
6. **Restart** the agent, send more messages, trigger another
   summarization. Assert the SAME file now has ≥ 2
   `## Summarized at ` headers — proving persistence across restart.

### 5.3 `--list-summaries` unit tests

`deep_worker_cli/tests/unit_tests/cli/test_list_summaries.py` (NEW):

1. `test_empty_home_returns_no_summaries` — clean home, empty output.
2. `test_lists_per_agent_summaries` — 2 agents × 2 threads each,
   output has 4 rows.
3. `test_latest_timestamp_matches_last_section_header` — fixture
   file with 3 headers; "Latest" column = last header's ts.
4. `test_events_count_matches_section_header_count`.
5. `test_filter_by_agent` — `--agent foo` excludes `bar`'s summaries.
6. `test_format_json_shape` — JSON mode returns a list of dicts with
   the 5 fields.

### 5.4 `--remove-before` extension tests

`deep_worker_cli/tests/unit_tests/cli/test_remove_before_summaries.py`
(NEW):

1. `test_deletes_file_when_all_sections_older_than_date`.
2. `test_rewrites_file_keeping_only_newer_sections`.
3. `test_no_op_when_all_sections_newer_than_date`.
4. `test_dry_run_does_not_write`.
5. `test_atomic_rewrite_uses_rename` — monkey-patch `os.replace`,
   assert it is called with `<file>.tmp → <file>`.
6. `test_checkpoint_pruning_still_works` — negative control: existing
   behavior is NOT broken by the extension.

All tests are pure-sync file I/O over `tmp_path`; no asyncio, no RPyC,
no real spawned agent.

---

## 6. Documentation updates

1. `deep_worker_cli/README.md` — add a "Summarization storage" section
   pointing at `<data-root>/<agent>/summaries/` and explaining the
   file format.
2. `deep_worker_cli/README.md` — document the new `--list-summaries`
   and the extended `--remove-before` semantics in the existing
   "Thread management commands" block.
3. The companion design document — add a one-line pointer to
   this new design since the summarization node now references a real
   file.

No changes to the web UI docs are needed — Full Dialog already
renders the path; now it just works when clicked on.

---

## 7. Open questions for the user

None. Design is fully specified from user requirements. Ready for
approval → TDD implementation.

---

## 8. Acceptance checklist (post-implementation)

- [ ] `<data-root>/<agent>/summaries/<agent>.md` is created on the
      first summarization event and persists across process restarts.
- [ ] The path in the summarization `HumanMessage` is an absolute
      filesystem path that `cat` opens.
- [ ] `ls -la <data-root>/<agent>/summaries/` shows the file with
      size equal to accumulated summarization output.
- [ ] `deep-worker --list-summaries` prints the new table.
- [ ] `deep-worker --list-summaries --agent my-agent --format json`
      returns a valid JSON array.
- [ ] `deep-worker --remove-before 30d --dry-run` previews summary
      prunes without writing.
- [ ] `deep-worker --remove-before 30d` prunes summaries and still
      prunes threads.
- [ ] All new unit + integration tests pass.
- [ ] Full test suite stays green (1321 + new; web-suite 169 + unchanged).
- [ ] No new tempdir under `/tmp/deepagents_conversation_history_*`
      is created by a fresh `deep-worker <name>` run.

---

**End of design.**
