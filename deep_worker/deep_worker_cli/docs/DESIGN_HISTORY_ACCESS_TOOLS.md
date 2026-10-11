# DESIGN: History Access Tools (`history_range`, `history_at`, `history_slice`)

## Goal

Provide three new tools that give any agent fast, read-only access to its own
pre-summarization history stored in `~/.deep_worker/<name>/summaries/<thread_id>.md`.
These tools work for ALL agents — no `--history-memory` flag required.

## Data format

Summaries are append-only **markdown** files with sections::

    ## Summarized at <iso-8601-ts>

    <evicted messages as langchain get_buffer_string output>

Each `## Summarized at ...` section is one "entry". The body is the
free-text get_buffer_string representation of the evicted messages.

There is also a JSONL format (`<thread_id>.jsonl`) written by the
`deepagents.middleware.summarization` module — it stores individual messages
with their `message_id`, `message_index`, and `message_type`. Both formats
coexist in the same directory. These tools work with the markdown format,
which coarser (section-level) but simpler. Each section yields:

## Tools

### 1. `history_range(runtime, working_dir) -> json`

Returns the range of sections in the agent's markdown summary file.

**Parameters (injected by framework):**
- `runtime`: LangGraph ToolRuntime — provides `runtime.state` (set by
  `HistoryAccessMiddleware.before_agent()`) with `history_access_assistant_id`
  and `history_access_home`; and `runtime.config` with `configurable.thread_id`.
- `working_dir`: optional override for MCP direct execution.

**Output:**
```json
{
  "earliest": "2026-01-15T10:30:00+00:00",
  "latest": "2026-04-22T15:40:18+00:00",
  "count": 142
}
```

- `earliest`: `summarized_at` of the first section.
- `latest`: `summarized_at` of the last section.
- `count`: number of `## Summarized at` sections in the file.
- Missing/empty file → `{"earliest": null, "latest": null, "count": 0}`.

**Implementation:** Read file, split sections via regex, return first/last
timestamps and count. O(n) where n = file size.

### 2. `history_at(timestamp, before=1, at_or_after=2, runtime, working_dir) -> json`

Finds sections around a given ISO-8601 timestamp using **datetime** comparison
(not string comparison, to handle `+00:00` vs `Z` vs microsecond variants).

**Parameters:**
- `timestamp`: ISO-8601 string to search around (e.g. `"2026-04-22T15:40:18+00:00"`).
- `before`: number of sections to return BEFORE the timestamp (default 1).
- `at_or_after`: number of sections AT or AFTER the timestamp (default 2).
- `runtime`, `working_dir`: injected by framework (see above).

**Output:**
```json
{
  "messages": [
    {"summarized_at": "...", "content_snippet": "..."},
    ...
  ],
  "indices": {"start": 5, "end": 8}
}
```

- `messages`: list of `{summarized_at, content_snippet}`.
- `indices.start`: index of first returned section (0-based).
- `indices.end`: index AFTER last returned section (exclusive, Python slice style).
- If fewer sections exist than requested → return what's available.
- If timestamp is unparseable → empty result (no crash).

**Implementation:** Read file, split sections, find pivot with `datetime >=`
comparison, collect surrounding sections. O(n) where n = file size.

### 3. `history_slice(start, end, runtime, working_dir) -> json`

Returns a slice of sections by index (0-based, end-exclusive, Python slice
convention). Negative indices are clamped to 0 (documented limitation).

**Parameters:**
- `start`: 0-based start index (clamped to 0 if negative).
- `end`: exclusive end index (clamped to section count if beyond file).
- `runtime`, `working_dir`: injected by framework.

**Output:**
```json
{
  "messages": [{"summarized_at": "...", "content_snippet": "..."}],
  "indices": {"start": 3, "end": 7}
}
```

**Rules:**
- `start >= count` → empty `messages` list.
- `end > count` → clamped to count.
- `end <= start` → empty list.
- No exceptions on out-of-range — always returns what's available.

**Implementation:** Read file, split sections, slice list. O(n) where
n = file size. No sidecar index in v1.

## Middleware: `HistoryAccessMiddleware`

- Always registers all three tools (no `--history-memory` guard).
- `before_agent()` sets `history_access_assistant_id` and `history_access_home`
  in `runtime.state`.
- Tools read these from `runtime.state.get(...)`.
- Thread ID is read from `runtime.config.get("configurable", {}).get("thread_id")`.

### Registration location

In `agent.py`, added AFTER `HistoryMemoryMiddleware` but BEFORE the
`create_deep_agent()` call. No injection hint is needed since the tools
use the standard `runtime.state` / `runtime.config` pattern (same as
`history_memory_tool.py`).

## Where to implement

| File | Description |
|------|-------------|
| `deep_worker_cli/history_access_tool.py` | Three `@tool` functions |
| `deep_worker_cli/middleware/history_access.py` | `HistoryAccessMiddleware` |
| `deep_worker_cli/middleware/__init__.py` | Export `HistoryAccessMiddleware` |
| `deep_worker_cli/agent.py` | Append middleware to `agent_middleware` list |
| `tests/unit_tests/test_history_access_tools.py` | Unit tests |
| `docs/DESIGN_HISTORY_ACCESS_TOOLS.md` | This document |

## Acceptance criteria

- `history_range`: returns `{earliest: null, latest: null, count: 0}` for
  missing/empty file.
- `history_at`: datetime comparison handles `+00:00`, `Z`, microsecond variants.
- `history_slice`: clamps out-of-range gracefully, no exceptions.
- Tools appear for every fresh agent (no flag required).
- All three tools return JSON strings (for consistency with other tools).

## Out of scope

- Changing the `.md` summary format.
- Deleting or mutating history entries.
- Cross-agent access to another agent's history.
- Sidecar offset-index in v1.

## Where to implement

### 1. `deep_worker_cli/history_access_tool.py`
~ 120 lines. Three `@tool`-decorated functions plus a shared helper
`_read_jsonl_range()` and `_parse_entry()`.

### 2. `deep_worker_cli/middleware/history_access.py`
~ 60 lines. `HistoryAccessMiddleware` that:
- Stores `assistant_id` and `home` path.
- Always exposes all 3 tools via `self.tools`.
- `before_agent()`: extracts `thread_id` from runtime configurable and
  passes it (along with `assistant_id` and `home`) into state so the
  tools can construct the JSONL path.

### 3. `deep_worker_cli/agent.py`
One new line: append `HistoryAccessMiddleware(assistant_id=assistant_id)`
to the middleware pipeline (after `DanglingToolCallMiddleware`, around
line 854).

### 4. `deep_worker_cli/middleware/__init__.py`
Export `HistoryAccessMiddleware`.

### 5. `deep_worker_cli/summaries.py` (or `storage_paths.py`)
Shared helper `get_thread_history_path(assistant_id, thread_id, home=None)`
→ `Path(home)/.deep_worker/assistant_id/summaries/thread_id.jsonl`.

### 6. `docs/DESIGN_HISTORY_ACCESS_TOOLS.md`
This document.

## Tool input injection pattern

Tools receive runtime context via a `tool_input: dict | None` parameter that
the middleware fills with:
```python
{
    "assistant_id": "agent_name",
    "thread_id": "thread_abc",
    "home": "/home/user"  # or Path.home() for production
}
```

The `@tool`-decorated functions use `StructuredTool.from_function()` with
`inject_defaults=True` so the middleware can set defaults on the tool before
adding it to the agent.

**Alternative approach considered and rejected:** reading `runtime.state` in
the tool body. This would work but requires the middleware to set state, and
the tool to know the state keys — same coupling but more indirect.

## Edge cases

| Condition | Behaviour |
|-----------|-----------|
| JSONL file missing | `count: 0`, earliest/latest null, empty messages list |
| JSONL file empty (0 bytes) | Same as missing |
| `history_at` with no matching timestamp | Return entries closest to the boundary (before: last N, at_or_after: 0) |
| `history_slice` start >= count | Empty messages |
| `history_slice` end > count | Clamped to count |
| `history_slice` negative start | Clamped to 0 |
| `history_slice` negative end | Clamped to 0, then if end <= start → empty |
| `history_range` on multi-thread file | Returns data for that thread only |

## Out of scope

- Modifying JSONL format.
- Deleting/mutating history entries.
- Cross-agent history access.
- Semantic search (already covered by `history_memory`).
- Line-offset sidecar index (reserved for v2 if profiling shows need).
