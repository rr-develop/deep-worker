# Data Format Versions

This document describes all versioned data formats used by the deep-worker agent
system. Every on-disk format that carries an explicit version number is listed here,
along with its scope, file location, and the code that defines it.

---

## Storage Version 1 — Global sessions.db

| Field | Value |
|---|---|
| **Version constant** | `STORAGE_VERSION_GLOBAL_DB = 1` |
| **Source** | `storage_paths.py:10` |
| **Status** | Deprecated |
| **Location** | `~/.deepagents/sessions.db` |
| **Format** | SQLite database shared by all agents |

Single database file that stores thread/session state for every agent. Replaced
by v2 because concurrent agents on the same host serialised writes through one
SQLite connection.

---

## Storage Version 2 — Per-Agent sessions_<name>.db

| Field | Value |
|---|---|
| **Version constant** | `STORAGE_VERSION_DEEPAGENTS_PER_AGENT = 2` |
| **Source** | `storage_paths.py:11` |
| **Status** | Legacy (still in use by some agents) |
| **Location** | `~/.deepagents/sessions_<name>.db` |
| **Format** | SQLite database per agent |

One SQLite database per agent, still in the legacy `~/.deepagents/` directory.
Version 2 exists when `session_metadata.json` is missing, has a bad format, or
explicitly carries `"storage_version": 2`.

The `session_metadata.json` record, when present, pins the version on restart
because `detect_storage_version()` reads the persisted `storage_version` field
and returns it verbatim, never upgrading automatically.

**Upgrade path:** `deep-worker migrate <name>` re-reads old DB and writes v3.

---

## Storage Version 3 — Agent-Root threads.db

| Field | Value |
|---|---|
| **Version constant** | `STORAGE_VERSION_AGENT_ROOT = 3` |
| **Source** | `storage_paths.py:12` |
| **Status** | Current (`CURRENT_STORAGE_VERSION = 3`) |
| **Location** | `~/.deep_worker/<name>/threads.db` |
| **Format** | SQLite database per agent under `~/.deep_worker/<name>/` |

Current storage layout. Each agent has its own directory under
`~/.deep_worker/<name>/` containing:

- `threads.db` — thread/session state
- `data/` — deepagents data directory
- `summaries/` — summarization history files (see v4)
- `session_metadata.json` — top-level metadata with `"storage_version": 3`
- `claude_session_state.json` / `codex_session_state.json` — provider session data
- `tm/` — task-manager state (optional, when `--task-manager` is active)

Detection logic (`detect_storage_version()`, `storage_paths.py:102-121`):
1. If `agent_name` is `None` → v1.
2. If `session_metadata.json` exists and has a valid `storage_version` int → return it.
3. If legacy `sessions_<name>.db` exists → v2.
4. Otherwise → `CURRENT_STORAGE_VERSION` (v3).

---

## Format Version 4 — Summarization History JSONL

| Field | Value |
|---|---|
| **Version constant** | `SCHEMA_VERSION = 4` (in `summarization_history.py`) |
| **Source** | `summarization_history.py:16` |
| **Status** | Current |
| **Location** | `~/.deep_worker/<name>/summaries/<name>.jsonl` |
| **Format** | Append-only JSONL, one message per line |

One JSON object per line. Each line:

```json
{
  "schema_version": 4,
  "kind": "deepagents.summarization_history.message",
  "summarized_at": "2026-05-29T10:15:30.123456+00:00",
  "message_index": 0,
  "message_type": "ai",
  "message_id": "run-...",
  "message": {
    "type": "ai",
    "data": {
      "content": "...",
      "additional_kwargs": {},
      "response_metadata": {},
      "id": "run-...",
      "tool_calls": []
    }
  }
}
```

- `message` is the canonical LangChain dict from `messages_to_dict()`, serialised
  with `json.dumps(..., default=str, ensure_ascii=False)` — human-readable plain JSON.
- Non-JSON-native types (`datetime`, `Decimal`) are converted to ISO strings via `default=str`.
- Non-empty lines are authoritative for thread order.

**Deprecated sub-format** (original v4 implementation): `message` contained an
`{"encoding": "langgraph_jsonplus", "type": "msgpack", "data": "base64..."}`
wrapper. The current `deserialize_summarized_messages_jsonl()` handles both
formats transparently.

**Related design doc:** `DESIGN_SUMMARIZATION_HISTORY_JSONL_V4.md`

---

## Delta Checkpointer Version 2

| Field | Value |
|---|---|
| **Version constant** | `_DELTA_VERSION = 2` |
| **Source** | `delta_checkpointer.py:46` |
| **Status** | Current |
| **Location** | Inside `threads.db` / checkpoint store |
| **Format** | Key-value snapshots with delta chains |

Internal checkpointer format. Each snapshot record carries a `"v"` field
equal to `_DELTA_VERSION`. Readers check `data.get("v", 1) != _DELTA_VERSION`
to detect incompatible snapshots.

---

## Summary Table

| # | Name | Location | Status | Defined In |
|---|---|---|---|---|
| 1 | Global sessions.db | `~/.deepagents/sessions.db` | Deprecated | `storage_paths.py` |
| 2 | Per-agent sessions.db | `~/.deepagents/sessions_<name>.db` | Legacy | `storage_paths.py` |
| 3 | Agent-root threads.db | `~/.deep_worker/<name>/threads.db` | **Current** | `storage_paths.py` |
| 4 | Summarization history JSONL | `~/.deep_worker/<name>/summaries/<name>.jsonl` | **Current** | `summarization_history.py` |
| Δ2 | Delta checkpointer | Inside checkpoint store | **Current** | `delta_checkpointer.py` |
