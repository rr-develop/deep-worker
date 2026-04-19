# Design: Delta Checkpointer — storing only changes

## Problem

LangGraph's `AsyncSqliteSaver` saves a **full snapshot** of the state (including ALL messages)
in every checkpoint. A single user→AI exchange creates ~12 checkpoints (node transitions
in middleware). Every checkpoint contains a complete copy of the entire history.

**Real scale of the problem:**

| Agent | Checkpoints | File size | Sum of checkpoint BLOBs | Messages in the last one |
|-------|-------------|-----------|-------------------------|--------------------------|
| agent A | 9,732 | 14 GB | 11.3 GB | 866 |
| agent B | 13,607 | 6.2 GB | — | — |
| agent C | 0 (writes only) | 4.3 GB | — | — |

Growth is **O(N²)**: N checkpoints × the (growing) size of the full history.

An additional problem: `deep-worker -ls` runs `GROUP BY` + `json_extract` over
ALL SQLite files (~27 GB in total) → **31 seconds** for a simple listing.

## Goal

Store only the delta (changed channel values) in every checkpoint.
The full history is restored by walking the chain of deltas.

## Research: how LangGraph stores checkpoints

### Tables

```sql
-- Full state snapshots
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,        -- monotonically increasing (sorts as a string)
    parent_checkpoint_id TEXT,          -- reference to the previous checkpoint
    type TEXT,                          -- serializer name
    checkpoint BLOB,                    -- FULL serialized Checkpoint dict
    metadata BLOB,                      -- JSON metadata (agent_name, step, source, etc.)
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Intermediate writes between checkpoints (pending computation)
CREATE TABLE writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,              -- channel name (messages, todos, etc.)
    type TEXT,
    value BLOB,                         -- serialized value
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

### Checkpoint BLOB (full snapshot)

```python
{
    "v": 1,                              # format version
    "id": "1ef...",                       # unique ID
    "ts": "2026-04-19T...",              # timestamp
    "channel_values": {                  # ← THIS IS DUPLICATED IN FULL
        "messages": [msg1, msg2, ...],   #    FULL history — the main space hog
        "todos": [...],
        "working_dir": "...",
        "skills_metadata": {...},
        ...
    },
    "channel_versions": {                # which channels were updated
        "messages": "00032.123456",
        "todos": "00001.654321",
        ...
    },
    "versions_seen": {                   # node → channel → version
        "model": {"messages": "00031"},
        "tools": {"messages": "00032"},
        ...
    },
    "updated_channels": ["messages"],    # ← KEY: which channels changed
}
```

### Key observation

The `updated_channels` field in every checkpoint already contains the **list of changed channels**.
And `channel_versions` makes it possible to determine whether a channel changed since the previous checkpoint.

Out of the 12 checkpoints per exchange, only 2 actually add messages:
- CP 1: `updated_channels: ["messages"]` — +HumanMessage
- CP 9: `updated_channels: ["messages"]` — +AIMessage
- CP 2-8, 10-11: messages did NOT change, yet a FULL copy is stored

## Solution design: DeltaCheckpointer

### Principle

1. **A checkpoint stores a delta** — only the channel_values of the channels listed in `updated_channels`
2. **Full reconstruction** — walk the chain parent_checkpoint_id → ... → root
3. **Periodic full snapshots** — every N checkpoints or when the depth limit is exceeded
4. **Compatibility** — implements the `BaseCheckpointSaver` interface, a drop-in replacement

### Delta BLOB format

```python
# Instead of a full snapshot:
{
    "v": 2,                              # format version (2 = delta)
    "id": "...",
    "ts": "...",
    "delta_channel_values": {            # ONLY the changed channels
        "messages": [new_msg],           # delta for messages = only the NEW ones
    },
    "channel_versions": {...},           # as before
    "versions_seen": {...},              # as before
    "updated_channels": ["messages"],    # as before
    "is_full_snapshot": false,           # marker: delta or full
    "delta_type": "append",             # delta type for messages (append-only channel)
}
```

### Delta types per channel

| Channel | Delta type | Description |
|-------|-----------|----------|
| `messages` | **append** | Only new messages (messages is an append-only list) |
| `todos` | **replace** | Full replacement (todos are overwritten entirely) |
| `working_dir` | **replace** | Scalar — store on change |
| `structured_response` | **replace** | Overwritten entirely |
| `skills_metadata` | **replace** | Overwritten entirely |
| `memory_contents` | **replace** | Overwritten entirely |
| Unchanged | **skip** | Not stored (delta = null) |

For `messages` — an **append delta**: store only `messages[prev_count:]`.
For the rest — a **replace delta**: store the full new value (they are small).

### Full state reconstruction

```python
async def _reconstruct_state(self, checkpoint_id, thread_id, checkpoint_ns):
    """Walk parent chain and accumulate deltas."""
    # Collect chain: current → parent → ... → full_snapshot
    chain = []
    current_id = checkpoint_id
    
    while current_id:
        row = await self._fetch_checkpoint_row(thread_id, checkpoint_ns, current_id)
        chain.append(row)
        
        cp_data = self.serde.loads_typed((row.type, row.blob))
        if cp_data.get("is_full_snapshot", True):  # v1 checkpoints = full snapshots
            break
        current_id = row.parent_checkpoint_id
    
    # Apply deltas from oldest to newest
    chain.reverse()
    full_state = chain[0].channel_values  # base = full snapshot
    
    for delta_cp in chain[1:]:
        if delta_cp.delta_type == "append" and "messages" in delta_cp.delta_channel_values:
            full_state["messages"].extend(delta_cp.delta_channel_values["messages"])
        for channel, value in delta_cp.delta_channel_values.items():
            if channel != "messages":  # messages handled above
                full_state[channel] = value
    
    return full_state
```

### Full snapshots: on demand only, not by default

By default **all** checkpoints are deltas. Full snapshots are created ONLY:

1. **The first checkpoint** of a thread (the chain root — a full snapshot is mandatory)
2. **On summarization** — when SummarizationMiddleware substantially restructures the
   messages (removes old ones, adds a summary). A delta is not applicable to a restructured
   list — a new baseline is needed. Detected via the metadata `source == "summarize"`
   or by a length mismatch: `len(new_messages) < len(prev_messages)`.
3. **The optional flag** `--snapshot-interval N` when starting the agent. If specified,
   a full snapshot is created every N checkpoints. **Disabled by default** (0).

```python
class DeltaSqliteCheckpointer(BaseCheckpointSaver):
    def __init__(self, conn, snapshot_interval: int = 0):
        """
        Args:
            snapshot_interval: Create full snapshot every N checkpoints.
                0 = disabled (default). Only root and summarization
                trigger full snapshots.
        """
        self._snapshot_interval = snapshot_interval

    async def aput(self, config, checkpoint, metadata, new_versions):
        is_root = parent_checkpoint_id is None
        is_summarization = self._is_summarization(checkpoint, previous_state)
        is_interval = (
            self._snapshot_interval > 0
            and delta_depth >= self._snapshot_interval
        )
        
        need_full_snapshot = is_root or is_summarization or is_interval
        
        if need_full_snapshot:
            blob = self._serialize_full(checkpoint)
        else:
            blob = self._serialize_delta(checkpoint, previous_state)
        ...

    def _is_summarization(self, checkpoint, previous_state):
        """Detect summarization: messages were restructured, not just appended."""
        prev_msgs = previous_state.get("messages", [])
        new_msgs = checkpoint["channel_values"].get("messages", [])
        # Messages REDUCED or first messages changed = summarization
        if len(new_msgs) < len(prev_msgs):
            return True
        if prev_msgs and new_msgs and prev_msgs[0] != new_msgs[0]:
            return True  # First message replaced with summary
        return False
```

**Result**: for a typical agent (no summarization, no --snapshot-interval):
- 1 full snapshot (the root)
- All other checkpoints are deltas
- Reconstruction: a walk from the current checkpoint to the root (the whole chain)

### Backward compatibility: existing v1 checkpoints

Existing checkpoints (`"v": 1`) are full snapshots. DeltaCheckpointer
treats them as full snapshots:

```python
if data.get("v", 1) == 1 or data.get("is_full_snapshot", True):
    # Legacy v1 or explicit full snapshot — use as-is
    return data["channel_values"]
else:
    # v2 delta — reconstruct
    return self._reconstruct_state(...)
```

**No migration.** Old data is read as-is. New checkpoints are written as deltas.

### Savings estimate

For an agent with 9,732 checkpoints and 11.3 GB:

| Metric | Now (full) | With delta |
|---------|---------------|---------|
| Checkpoints per exchange without tool calls | ~12, all full | 12: 1 full + 11 delta (0 bytes of messages) |
| Checkpoints per exchange with 1 tool call | ~20 | 20: 1 full + 19 delta |
| Typical delta size (without messages) | 800 KB (full) | ~2-5 KB (only channel_versions + metadata) |
| Messages delta | 800 KB (ALL) | ~0.5-2 KB (1 new message) |
| 9,732 checkpoints in total | 11.3 GB | ~50-100 MB (estimate) |

**Expected reduction: ~100x** for long conversations.

### Optimizing `deep-worker -ls`

**A separate table for thread metadata:**

```sql
CREATE TABLE thread_metadata (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    agent_name TEXT,
    updated_at TEXT,
    message_count INTEGER DEFAULT 0,
    PRIMARY KEY (thread_id, checkpoint_ns)
);
```

Updated on every `aput`. `deep-worker -ls` reads only this table —
an instant query without `json_extract` and `GROUP BY` over huge BLOBs.

## Code changes

### New file: `delta_checkpointer.py` (in the CLI package)

Implements `BaseCheckpointSaver` with delta logic:

```python
class DeltaSqliteCheckpointer(BaseCheckpointSaver):
    """SQLite checkpointer that stores only deltas between checkpoints.
    
    Key differences from AsyncSqliteSaver:
    - aput: computes delta, stores only changed channel values
    - aget_tuple: reconstructs full state from delta chain
    - Periodic full snapshots every N checkpoints
    - thread_metadata table for fast listing
    - Backward compatible with v1 (full snapshot) checkpoints
    """
    
    async def aput(self, config, checkpoint, metadata, new_versions):
        ...
    
    async def aget_tuple(self, config):
        ...
    
    async def alist(self, config, *, filter=None, before=None, limit=None):
        ...
    
    async def aput_writes(self, config, writes, task_id, task_path=""):
        # Writes table — unchanged (intermediate data, already efficient)
        ...
```

### Changes in existing files

1. **`sessions.py`**: `get_checkpointer()` returns `DeltaSqliteCheckpointer`
   instead of `AsyncSqliteSaver`

2. **`sessions.py`**: `list_all_threads()` reads from the `thread_metadata` table

3. **`sessions.py`**: `get_thread_history()` uses delta-based reconstruction

4. **`agent_process.py`**: unchanged — the checkpointer is injected via DI

### SQL schema for the delta checkpointer

```sql
-- The same checkpoints table, but the checkpoint BLOB contains a delta
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,                    -- delta or full snapshot (distinguished by the v=2 marker)
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Unchanged — writes store intermediate data
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- NEW: fast index for listing
CREATE TABLE IF NOT EXISTS thread_metadata (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    agent_name TEXT,
    updated_at TEXT,
    message_count INTEGER DEFAULT 0,
    checkpoint_count INTEGER DEFAULT 0,
    PRIMARY KEY (thread_id, checkpoint_ns)
);
```

## Implementation order

1. **`DeltaSqliteCheckpointer`** — a new class implementing `BaseCheckpointSaver`
2. **Tests**: unit tests for delta computation, reconstruction, backward compat
3. **`sessions.py`**: switch `get_checkpointer()` to the new class
4. **`list_all_threads`**: read from `thread_metadata`
5. **`get_thread_history`**: use delta-aware reconstruction
6. **Integration tests**: a real agent with the delta checkpointer
7. **Verification**: existing agents read the old data correctly

## Migrating existing data

**No migration is required.** DeltaCheckpointer:
- Reads v1 checkpoints (full snapshots) as full snapshots
- Writes new checkpoints as v2 deltas
- The `thread_metadata` table is created automatically on first run
- It is populated lazily: on the first access to a thread or on `aput`

Optionally: an offline `VACUUM` to compact bloated SQLite files (304K free pages
in one existing agent database = 1.2 GB of garbage).

## Risks

1. **Corrupted chain**: if a delta in the middle of the chain is lost or damaged →
   the state cannot be restored. Mitigation: SQLite WAL mode + integrity checks.
   On summarization a new baseline is created automatically.
   Optionally: `--snapshot-interval N` for extra protection.

2. **Reconstruction latency**: the first read of a long delta chain
   may require walking all the deltas down to the root.
   Mitigation: caching the reconstructed state in memory (one walk on
   `aget_tuple`, after which deltas are applied incrementally).
   Summarization automatically creates a new baseline, limiting the depth.

3. **Concurrent writes**: SQLite WAL mode + per-agent DB files already solve this.

4. **LangGraph upgrades**: if LangGraph changes the Checkpoint format, the delta logic
   must be updated. Mitigation: versioning (v=2), backward compat.
