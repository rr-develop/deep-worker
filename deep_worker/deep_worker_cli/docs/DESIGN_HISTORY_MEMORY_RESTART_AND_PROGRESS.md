# Design: Enable `--history-memory` on Existing Agents & Startup Indexing Progress

**Status:** Draft
**Author:** AI Assistant
**Date:** 2026-05-31
**Related:**
- `DESIGN_HISTORY_MEMORY_TOOL.md` (original feature design)
- `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py`
- `deep_worker/deep_worker_cli/deep_worker_cli/agent.py`
- `semantic_search/semantic_search/history_indexer.py`

---

## 1. Problem

The `history_memory` tool exists but has three gaps:

1. **No CLI shorthand**: `--history-memory` is the only form. Users asking about `-m` get an error. The original design rejected `-m` citing a possible conflict with `--message` in the interactive CLI, but such a flag does not exist.

2. **No startup indexing**: When `--history-memory` is first enabled on an agent with existing summarization history, the index is EMPTY until the next summarization event fires the `_on_summarized` callback. The agent can't search its past until summarization happens — which may be hours into the session or never if the context window never fills up.

3. **No progress display**: `index_all_history_files()` runs synchronously and can take 10-60+ seconds (loading the embedding model, embedding chunks). The agent startup appears frozen with no feedback.

---

## 2. Goal

1. Add `-m` shorthand for `--history-memory` in all CLI entry points (parser, detector, tests).
2. On agent startup, when `--history-memory` is enabled, index all existing history JSONL files synchronously with a Rich progress bar so the user sees what's happening.
3. The `_on_summarized` callback continues to handle incremental indexing after each summarization — startup indexing is a one-time initial pass.

---

## 3. Design

### 3.1 CLI Shorthand `-m`

Add `-m` to `--history-memory` in three places in `agent_command.py`:

```python
# create_parser()
parser.add_argument("-m", "--history-memory", ...)

# _explicit_cli_flags()
detector.add_argument("-m", "--history-memory", ...)
```

**Conflict check:** No existing flag uses `-m`. The flags are: `-sp`, `-tm`, `-asm`, `--clean`, `--dev`, `--config`, `--dir`, `--model`, `--provider`, `--history-memory`, `--use-sqlite`, `--memory-only`, `--import-claude`.

Effect: `deep-worker --restart some_agent -m` now works identically to `deep-worker --restart some_agent --history-memory`.

### 3.2 Startup Indexing with Progress

**Where**: `agent.py::create_cli_agent()` — BEFORE `HistoryMemoryMiddleware` creation, but AFTER all other middleware setup.

**Critical ordering**: Middleware that depends on the index MUST be created AFTER indexing completes. Otherwise `_has_index` would be `False` and the tool wouldn't be registered.

```python
# In create_cli_agent(), after all other middleware but BEFORE HistoryMemoryMiddleware:

if enable_history_memory:
    from semantic_search.history_indexer import (
        history_index_exists,
        index_all_history_files,
    )
    from deep_worker_cli.config import console

    summaries_path = Path(summaries_dir)
    jsonl_files = list(summaries_path.glob("*.jsonl"))
    
    history_index_ready = False
    if jsonl_files and not history_index_exists(summaries_path):
        # Startup indexing with progress
        n_files = len(jsonl_files)
        if n_files >= 3:
            # Full progress bar for many files
            from rich.progress import (
                Progress, SpinnerColumn, TextColumn, BarColumn,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "Indexing conversation history...", total=n_files
                )
                def _on_file_progress(filename: str) -> None:
                    progress.update(task, description=f"Indexing {filename}...", advance=1)
                try:
                    index_all_history_files(summaries_dir=summaries_path, on_progress=_on_file_progress)
                    history_index_ready = True
                except Exception as exc:
                    logger.warning("History indexing failed: %s", exc)
        else:
            # Brief spinner for few files
            from rich.progress import Progress, SpinnerColumn, TextColumn
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Indexing conversation history...", total=None)
                try:
                    index_all_history_files(summaries_dir=summaries_path)
                    history_index_ready = True
                except Exception as exc:
                    logger.warning("History indexing failed: %s", exc)
    elif history_index_exists(summaries_path):
        history_index_ready = True

# THEN create HistoryMemoryMiddleware — it checks the index and registers the tool
agent_middleware.append(HistoryMemoryMiddleware(
    assistant_id=assistant_id,
    enabled=enable_history_memory and history_index_ready,
))
```

On indexing failure: `history_index_ready = False` → middleware gets `enabled=False` → tool is NOT registered for this session. This prevents the silent-empty-result problem.

On success: middleware gets `enabled=True` → `__init__` checks `history_index_exists()` → tool IS registered.

### 3.3 Progress Callback in `index_all_history_files()`

Add an optional `on_progress: Callable[[str], None] | None = None` parameter to `index_all_history_files()`. Called once per file before processing it, with the filename.

```python
def index_all_history_files(
    summaries_dir: str | Path,
    config: IndexConfig | None = None,
    embed_model: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> HistoryIndexStats:
    ...
    for jsonl_file in sorted(summaries_dir.glob("*.jsonl")):
        if on_progress:
            on_progress(jsonl_file.name)
        ...
```

### 3.4 Pre-Indexing Check

To avoid re-indexing on every restart after the first startup, the progress logic checks `history_index_exists()` and line_hashes:

- If no history index exists at all: show progress bar, index all files.
- If index exists AND all files are fully indexed (no new lines): skip progress, just proceed.
- If index exists but some files have new lines: the `_on_summarized` callback handles this incrementally during the session. We could optionally show a minimal progress bar here too, but the first summarization is expected to happen soon, so it's less critical.

For v1: Only show startup progress when the index doesn't exist at all. Incremental indexing at summarization time updates `_on_summarized` is fast (only new lines from one file) and runs inside the existing summarization middleware flow.

### 3.5 Edge Cases

- **No JSONL files**: Skip indexing silently (no progress shown).
- **Summaries directory doesn't exist**: Skip silently (common for brand-new agents).
- **Index already exists**: Skip progress (already indexed from a previous session).
- **Error during indexing**: Log the error, continue with agent startup. The index is best-effort — missing a few messages doesn't block the agent.
- **`-m` already used by another flag**: No existing conflict confirmed. If a future flag adds `-m`, the argparse error will surface at parse time.

---

## 4. Files to Modify

| File | Change |
|---|---|
| `semantic_search/semantic_search/history_indexer.py` | Add `on_progress` parameter to `index_all_history_files()` |
| `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py` | Add `-m` shorthand in `create_parser()` and `_explicit_cli_flags()` |
| `deep_worker/deep_worker_cli/deep_worker_cli/agent.py` | Add startup indexing with Rich progress in `create_cli_agent()` |

## 5. Files New

None.

---

## 6. Resolved Decisions

1. **`on_progress` callback**: Added to both `index_all_history_files()` and `index_history_file()` for consistency. `index_all_history_files` wraps `index_history_file` and passes the callback through.

2. **Progress display threshold**: If >= 3 JSONL files exist, show a full Rich progress bar (file name + percentage). If < 3 files, show a brief spinner "Indexing conversation history...".

3. **Error handling during startup indexing**: Catch `Exception`, log a warning via `logger.warning()`, and continue agent startup without history memory. The agent will retry on the next summarization event. This is best-effort — missing a few messages doesn't block the agent.

---
