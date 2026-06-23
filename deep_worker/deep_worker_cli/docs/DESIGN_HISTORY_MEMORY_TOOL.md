# Design: `history_memory` Tool — Semantic Search over Summarized Conversation History

**Status:** ✅ Approved — ready for implementation
**Author:** AI Assistant
**Date:** 2026-07-17
**Related:**
- `deep_worker/deepagents/deepagents/tools/project_memory.py` (reference tool)
- `deep_worker/deepagents/deepagents/middleware/project_memory.py` (reference middleware)
- `deep_worker/deepagents/deepagents/middleware/summarization.py` (trigger point)
- `deep_worker/deepagents/deepagents/middleware/summarization_history.py` (JSONL format)
- `semantic_search/semantic_search/` (indexing/retrieval engine)
- `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py` (CLI flags)
- `deep_worker/deep_worker_cli/deep_worker_cli/agent.py` (agent creation)

---

## 1. Problem

When an agent's conversation is summarised (history offloaded to keep within context window), the agent can no longer access the original messages. The summarization middleware saves them to JSONL files at `~/.deep_worker/<agent>/summaries/<thread_id>.jsonl`, but the agent has no tool to search them. It only sees a short summary text injected as a HumanMessage.

The `project_memory` tool solves the analogous problem for *project code* — it provides semantic search over the indexed codebase. We need the same capability for *conversation history*: a tool that lets the agent recall specifics from its own past, before summarization truncated them.

## 2. Goal

Provide a `history_memory()` tool that the agent can call to semantically search its own summarised conversation history. The tool must:

1. Search over messages stored in v4+ JSONL files (`~/.deep_worker/<agent>/summaries/*.jsonl`).
2. Return relevant messages with file path, line range (position in JSONL), and timestamp metadata.
3. Use the same semantic search engine (LlamaIndex + LanceDB) as `project_memory`.
4. Be enabled only when explicitly requested via CLI flag (`--history-memory`), because indexing is expensive.
5. Be indexed/updated automatically after each summarization event.

## 3. JSONL Format (already implemented)

Per `summarization_history.py`, each line in the JSONL is:

```json
{
  "schema_version": 4,
  "kind": "deepagents.summarization_history.message",
  "summarized_at": "2026-04-22T15:40:18.122420+00:00",
  "message_index": 0,
  "message_type": "human",
  "message_id": "some-uuid",
  "message": { "id": ["langgraph", ...], "type": "human", "data": {"content": "hello", ...} }
}
```

Files are stored at `~/.deep_worker/<agent>/summaries/<thread_id>.jsonl`, one per thread, appended to over time.

## 4. Indexing Strategy

### 4.1 Index Location
`~/.deep_worker/<agent>/summaries/.history_index/` — next to the JSONL files, following the same pattern as `.vector_index/` for project code.

### 4.2 Unit of Indexing
**One message** per indexed entry. Small messages (fewer than ~200 chars) are grouped with adjacent messages from the same summarization event (same `summarized_at` and thread) into a single chunk, up to `child_chunk_size` (1024 tokens). This mirrors `_merge_small_chunks` in `indexer.py`.

### 4.3 Chunk Size
- `child_chunk_size`: **1024 tokens** (same as `project_memory`)
- `parent_chunk_size`: **None** (flat structure initially — no hierarchical retrieval for history)
- Messages are split on token boundaries if a single message exceeds 1024 tokens.

### 4.4 Metadata per Chunk

| Field | Source | Description |
|---|---|---|
| `file_path` | Absolute path to `.jsonl` file | Identifies which thread file |
| `start_line` | Line number in JSONL | 1-based, first line of the message(s) in this chunk |
| `end_line` | Line number in JSONL | 1-based, last line of the message(s) in this chunk |
| `timestamp` | `summarized_at` from first message in chunk | ISO-8601 timestamp |
| `message_id` | `message_id` from first message | For traceability |
| `message_type` | `message_type` of first message | `human`, `ai`, `tool`, etc. |

**Excluded from embedding** (to not bias similarity): `file_path`, `start_line`, `end_line`, `message_id`.

**Excluded from LLM** (metadata not needed by agent): `start_line`, `end_line`, `message_id`.

### 4.5 Trigger: Index After Summarization

The `SummarizationMiddleware._offload_to_backend()` / `_aoffload_to_backend()` methods already write the JSONL. We add a hook there: after a successful write, if `history_memory` is enabled, call the indexer to process the new lines.

The indexer is **incremental**: it tracks which lines have been indexed per file via a hash file (same pattern as `file_hashes.json` in semantic_search). Only new/appended lines are processed.

## 5. Tool API

```python
@tool
def history_memory(
    query: str,
    top_k: int = 5,
    thread_id: str | None = None,
) -> str:
    """Search the agent's own past conversation history.

    Use this when you need to recall details from earlier in the conversation
    that have been summarised and removed from context.

    Args:
        query: Natural language description of what you're looking for.
        top_k: Number of results (1-20, default 5).
        thread_id: Optional filter to a specific thread.
    """
```

## 6. Middleware: `HistoryMemoryMiddleware`

Follows the exact pattern of `ProjectMemoryMiddleware`:

```python
class HistoryMemoryMiddleware(AgentMiddleware):
    """Registers history_memory tool if --history-memory is enabled and index exists."""

    def __init__(self, summaries_dir: Path, enabled: bool):
        self._enabled = enabled
        self._has_index = has_history_index(summaries_dir) if enabled else False
        self.tools = [history_memory] if self._has_index else []
        self._cognitive_hint = COGNITIVE_HINT if self._has_index else None

    def before_agent(self, state, runtime):
        """Set summaries_dir in state for the tool."""
        ...
```

When the tool is registered, a **cognitive hint** is injected into the system prompt (same pattern as `ProjectMemoryMiddleware.COGNITIVE_HINT`), telling the agent:

> **CRITICAL**: You have a `history_memory` tool. ALWAYS use it FIRST when you need to recall details from earlier parts of the conversation that happened before the current context window.
>
> Examples:
> - "What did the user say about X 10 messages ago?"
> - "Find the error message that occurred earlier"
> - "What was the previous task status update?"

## 7. CLI Flag: `--history-memory`

### 7.1 Name
- Long form: `--history-memory`
- Short form: **None** (to avoid confusion with existing `-m`/`--message` in the interactive CLI)

### 7.2 Where to add
In `agent_command.py::create_parser()`:

```python
parser.add_argument(
    "--history-memory",
    action="store_true",
    dest="history_memory",
    help="Enable semantic search over summarised conversation history "
         "(indexes ~/.deep_worker/<agent>/summaries/*.jsonl)",
)
```

### 7.3 Propagation
1. Parsed in `create_parser()` → passed to `run_agent()`.
2. Stored in `launch_config` dict.
3. Loaded on `--restart` via `merge_restart_config()`.
4. Passed to `AgentProcess.__init__()` as `history_memory: bool`.

### 7.4 AgentProcess to Middleware
`AgentProcess.start()` passes `history_memory` to `create_cli_agent()` → `HistoryMemoryMiddleware(summaries_dir, enabled=history_memory)` is added to middleware stack.

## 8. Cognitive Hint

Injected into system prompt when `--history-memory` is enabled AND a valid index exists:

```
### History Memory Tool

**CRITICAL**: You have a `history_memory` tool that can search your own
past conversation history. Use this BEFORE asking the user to repeat
information that was discussed earlier.

**When to use:**
- User refers to something from earlier in the conversation
- You need to recall error messages, code, or instructions from summarised turns
- You're unsure about details the user mentioned previously
- Before asking the user to clarify or repeat themselves

**Example queries:**
- "What did the user ask me to implement?"
- "Find the error trace from earlier"
- "What was the last command result?"
```

## 9. Integration Points

### 9.1 `agent.py::create_cli_agent()`
```python
# History memory middleware (semantic search over conversation history)
agent_middleware.append(
    HistoryMemoryMiddleware(
        summaries_dir=summaries_dir,
        enabled=history_memory,
    )
)
```

### 9.2 `summarization.py` — Post-write callback

To avoid coupling `deepagents/` (shared) with `deep_worker_cli/` (specific), `SummarizationMiddleware` accepts an **optional callable** in its constructor:

```python
class SummarizationMiddleware(BaseSummarizationMiddleware):
    def __init__(
        self,
        ...,
        on_summarized: Callable[[str], None] | None = None,  # NEW
    ):
        ...
        self._on_summarized = on_summarized
```

After `_offload_to_backend()` successfully writes JSONL (line ~672), it calls:
```python
if self._on_summarized and path:
    self._on_summarized(path)
```

In `agent.py::create_cli_agent()`, the caller wires them together:
```python
summarization_mw = SummarizationMiddleware(
    model=model,
    backend=...,
    on_summarized=history_indexer.index_new_lines if history_memory else None,
)
```

This is pure Dependency Injection — no coupling between packages.

### 9.3 `summaries.py` — Index-aware prune
When `prune_summaries_before()` removes sections from summary files, the history index must also be updated. This is a low-priority enhancement — initial implementation can require full re-index after prune.

## 10. Files to Create/Modify

**New files:**

| File | Purpose |
|---|---|
| `semantic_search/semantic_search/history_indexer.py` | Index JSONL messages into LanceDB |
| `semantic_search/semantic_search/history_retriever.py` | Search over history index |
| `deep_worker/deep_worker_cli/deep_worker_cli/utils/history_memory.py` | Tool implementation |
| `deep_worker/deep_worker_cli/deep_worker_cli/middleware/history_memory.py` | Middleware |

**Modified files:**

| File | Change |
|---|---|
| `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py` | Add `--history-memory` flag, persist in launch_config |
| `deep_worker/deep_worker_cli/deep_worker_cli/agent.py` | Wire up `HistoryMemoryMiddleware` in `create_cli_agent()` |
| `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py` | Accept and pass `history_memory` flag |
| `deep_worker/deepagents/deepagents/middleware/summarization.py` | Post-write hook to trigger indexing |

## 11. Resolved Decisions

1. **Tool placement**: **`deep_worker_cli/`** — not `deepagents/`. Rationale:
   - The tool is tightly coupled to `deep_worker`'s specific JSONL format and `summaries/` directory layout.
   - The `deepagents/` package is shared across projects; adding `deep_worker`-specific tooling there would create unwanted coupling.
   - `project_memory` lives in `deepagents/` because it indexes *any* codebase — it's provider-agnostic. `history_memory` indexes only `deep_worker`'s summarization JSONL files.
   - Follow the precedent of `SendMessageCompletionMiddleware` which lives in `deep_worker_cli/middleware/`.
   - **Concrete locations**:
     - `deep_worker_cli/deep_worker_cli/tools/history_memory.py` — tool function
     - `deep_worker_cli/deep_worker_cli/middleware/history_memory.py` — middleware
     - `semantic_search/semantic_search/history_indexer.py` — indexing engine (reuses LanceDB, embedding model from semantic_search)

2. **Message grouping**: Adjacent messages from the same `summarized_at` event are grouped into one chunk up to 1024 tokens. This avoids splitting tool_call + tool_result across separate chunks.

3. **Re-indexing after prune**: v1 requires `--clean` (full re-index from scratch) after prune. Can be optimized later with incremental line-tracking.

4. **Resource usage**: No limit on threads indexed in v1. Storage per message is ~1 LanceDB vector (384 floats for multilingual-e5-small) + metadata.

## 12. Future Enhancements (explicitly out of scope for v1)

- Cross-thread search across ALL history (not just current agent's)
- Automatic pruning of history index when summary files are pruned
- Hierarchical retrieval (AutoMergingRetriever) for history chunks
- Streaming re-index during summarization
