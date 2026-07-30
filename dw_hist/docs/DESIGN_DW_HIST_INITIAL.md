# dw-hist — agent-history exploration tool (initial design)

Status: Proposed
Branch: dev
Date: 2026-06-30

## 1. Goal

Provide a standalone host-level CLI `dw-hist` for **inspecting** agent
conversation history without having to attach to a running agent or use
`deep-worker`'s already-overloaded option matrix.

This first iteration ships two commands; the tool is designed so additional
commands can be added later without breaking the CLI surface.

## 2. User-facing CLI

### 2.1 `dw-hist --list`

List every agent on this host whose `~/.deep_worker/<agent>/summaries/`
directory contains a built history semantic index (`.history_index/lancedb/`
present). Agents need **not** be running.

Output (text, default): one agent per line, with a stable sort, e.g.:

```
agent1
agent2
agent3
...
```

`--format json` emits `{"agents": [{"name": "...", "summaries_dir": "..."}]}`
for scripting.

### 2.2 `dw-hist <agent_name> -q <query>`

Run a semantic search over `<agent_name>`'s history. Equivalent to what
`history_memory` tool returns from within the agent, **plus** the explicit
similarity score per result.

Default output mirrors `history_memory_tool`'s human-readable formatting and
adds `score=` already present there. Options:

- `-q, --query TEXT` (required) — natural-language query.
- `-k, --top-k INT` (default 5, max 20) — same bounds as the tool.
- `--format {text,json}` — `text` default; `json` prints raw
  `HistorySearchResult` records (`{content, score, thread_id, line_range,
  timestamp, message_types, file_path}`) so downstream tools can parse.

The agent need not be running. Read-only operation against the on-disk
LanceDB index.

### 2.3 Error behaviour

| Exit | Trigger |
|------|---------|
| 0 | success / `--list` with zero agents (prints nothing — `for a in $(dw-hist --list)` safe) |
| 2 | **usage error**: conflicting flags (`--list` with `agent_name`, `agent_name` without `-q`, `-q` without `agent_name`), invalid `--top-k` (outside 1..20), **or unknown agent** (`~/.deep_worker/<agent>/` does not exist as a directory) |
| 3 | agent exists but no history index (`~/.deep_worker/<agent>/summaries/.history_index/lancedb/` missing). Also raised when `search_history` raises `FileNotFoundError`. Stderr: `"No history index for agent '<name>'. Start the agent with --history-memory and let a summarization cycle complete before searching."` |
| 4 | search runtime error (any other exception raised inside `search_history`). Stderr: `"Search error: <type>: <message>"` |

Notes:
- "unknown agent" check: we look at the **agent root** `~/.deep_worker/<agent>/`, not at the `summaries/` subdir. An agent may legitimately exist without `summaries/` (never summarised) — that is exit 3 with the "no index" message, NOT exit 2 "unknown agent".
- `--top-k` is validated against 1..20 (the same bounds `history_memory_tool` enforces) **before** any heavy import or search call.

## 3. Storage layout we read

This is a **read-only consumer** of the existing semantic_search artefacts.
The contracts (paths, function signatures) we depend on:

- `~/.deep_worker/<agent>/summaries/` —
  `deep_worker_cli.summaries.get_summaries_dir(agent_name)`.
- `~/.deep_worker/<agent>/summaries/.history_index/lancedb/` — presence
  detected by `semantic_search.history_indexer.history_index_exists(dir)`.
- Search:
  `semantic_search.history_retriever.search_history(query, summaries_dir,
  top_k) -> list[HistorySearchResult]`.
- Result type:
  `semantic_search.history_retriever.HistorySearchResult` with fields
  `content, score, thread_id, line_range, timestamp, message_types,
  file_path`.

We do **not** call into `history_memory` (the agent-side tool) — that path
is async and tied to a `ToolRuntime`. We call `search_history` directly,
which is the same function `history_memory` ends up invoking.

No deadlock concern: the LanceDB GIL/loop-blocking problem applies only when
search runs on an agent's asyncio event-loop thread. `dw-hist` is a plain
synchronous CLI on the main thread; no event loop, no `to_thread`.

## 4. Package layout

New top-level package `dw_hist/` mirroring the `deep_worker/deep_worker_cli/`
shape (separate launcher + python module). Placed at repo root, sibling to
`semantic_search/` and `deep_worker/`, so it has no special relationship to
the agent CLI but can still source-import from it.

```
dw_hist/
├── dw-hist                  # bash launcher (sets PYTHONPATH, execs python)
├── docs/
│   └── DESIGN_DW_HIST_INITIAL.md   # this document
├── dw_hist/                 # python package
│   ├── __init__.py
│   ├── cli.py               # argparse + dispatch
│   ├── lister.py            # --list implementation
│   └── search.py            # <agent> -q <query> implementation
├── tests/
│   └── unit_tests/
│       ├── __init__.py
│       ├── test_cli.py
│       ├── test_lister.py
│       └── test_search.py
└── README.md                # short user-facing usage
```

Source-import only. No `pip install`. No `pyproject.toml` needed for the
initial drop (we already require `semantic_search` and `deep_worker_cli` to
be on PYTHONPATH; the launcher arranges that). A pyproject can be added
later if/when distribution is wanted; not in scope here.

### 4.1 Installation (host setup)

Documented in `dw_hist/README.md`:

```bash
ln -sf /path/to/production-tree/dw_hist/dw-hist ~/.local/bin/dw-hist
# (production tree)
# or
ln -sf /path/to/dev-tree/dw_hist/dw-hist ~/.local/bin/dw-hist
# (dev tree)
```

Same convention `deep-worker` and `dwcli` already follow.

### 4.2 Launcher contract

```bash
#!/usr/bin/env bash
# dw-hist launcher — adds repo modules to PYTHONPATH and execs python.

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$SCRIPT_DIR:$REPO_ROOT/deep_worker/deep_worker_cli:$REPO_ROOT/semantic_search:${PYTHONPATH:-}"

exec python3 -m dw_hist.cli "$@"
```

This guarantees `dw_hist`, `deep_worker_cli`, and `semantic_search` all
resolve from the same tree as the launcher (production or dev), matching
the `deep-worker` dispatcher's tree-locality property.

The launcher file is committed with **mode 0755**, matching the
existing `dwcli` / `deep-worker` launchers and the project's permission
policy (no 777/666).

## 5. CLI argument design

argparse with subcommand-less surface (one positional `agent_name` OR a
flag). The minimal cases we must keep distinguishable:

| invocation | meaning |
|---|---|
| `dw-hist --list` | list agents with an index |
| `dw-hist <agent> -q <query>` | search that agent's history |
| `dw-hist --help` | argparse help |

Implementation: a single `argparse.ArgumentParser` where:
- `agent_name` is positional and `nargs='?'`.
- `--list` is a flag.
- `-q/--query`, `-k/--top-k`, `--format` are options.

`dw_hist/cli.py::main(argv=None)` validates the combination:
- `--list` AND `agent_name` → error.
- `agent_name` without `-q` → error.
- `-q` without `agent_name` → error.

### 5.1 Exception mapping in `search.py`

```python
def run_search(agent_name, query, top_k) -> tuple[int, str]:
    # 1. Pre-check unknown agent  -> exit 2
    agent_root = Path.home() / Path("~/.deep_worker").name / agent_name
    if not agent_root.is_dir():
        return 2, f"Unknown agent: '{agent_name}'"

    # 2. Pre-check index presence  -> exit 3
    summaries_dir = get_summaries_dir(agent_name)
    if not history_index_exists(summaries_dir):
        return 3, f"No history index for agent '{agent_name}'. ..."

    # 3. Run search; FileNotFoundError -> exit 3, other -> exit 4
    try:
        results = search_history(query, summaries_dir, top_k=top_k)
    except FileNotFoundError as exc:   # race: index removed between checks
        return 3, f"History index disappeared mid-search: {exc}"
    except Exception as exc:           # everything else
        return 4, f"Search error: {type(exc).__name__}: {exc}"

    return 0, format_results(results, ...)
```

This split is what test_cli.py asserts.

### 5.2 Third-party noise isolation

The first time the embedder/index is loaded the underlying stack
(`transformers`, `huggingface_hub`, `tqdm`, `llama_index`) splashes a fair
amount of noise on **both** streams. None of it is a real error — the model
is already on disk and `semantic_search` already passes
`local_files_only=True` — but on a CLI tool meant for operators the noise is
useless and, for `--format json`, actively harmful (it corrupts the
JSON document).

Observed (real cold-cache run, 2026-06-30):

```
stderr:
  Warning: You are sending unauthenticated requests to the HF Hub. ...
  Loading weights: 100%|████| 199/199 [00:00<00:00, 2323.13it/s, ...]
  BertModel LOAD REPORT from: intfloat/multilingual-e5-small
  Key                     | Status     |  |
  embeddings.position_ids | UNEXPECTED |  |
stdout:
  LLM is explicitly disabled. Using MockLLM.
```

Two-layer fix:

1.  **Pre-import** — both the bash launcher and `cli.py` set a small
    set of env vars BEFORE any import of the heavy stack:
    `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
    `HF_HUB_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_PROGRESS_BARS=1`,
    `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, `TRANSFORMERS_VERBOSITY=error`,
    `TRANSFORMERS_NO_ADVISORY_WARNINGS=1`, `TOKENIZERS_PARALLELISM=false`,
    `TQDM_DISABLE=1`. The model is already cached locally so offline is the
    correct mode anyway.

2.  **Around the search call** — even with the env vars, libraries may
    still log to `stdout`/`stderr` (e.g. the bare `print()` in
    `llama_index/core/llms/utils.py`). We open `/dev/null` and redirect both
    streams around the `search_func` call only. dw-hist's own formatted
    output is printed *after* the context manager closes, on the clean
    streams.

Operator opt-out: `--verbose` (`-v`). The launcher detects the flag and
skips step (1); `cli.py` sets `DW_HIST_VERBOSE=1` and skips step (2). All
third-party noise then surfaces normally — useful when debugging actual
cold-cache or model-load issues.

Covered by `test_cli.py::test_search_silences_both_streams_by_default` and
`test_cli.py::test_search_verbose_lets_noise_through`.

## 6. Future extensions (out of scope here, but shape preserved)

Anticipated next features (so the CLI surface does not need a rewrite):

- `dw-hist <agent> --info` — index stats (size, message count, latest
  summary).
- `dw-hist <agent> --reindex` / `--clear` — rebuild / delete the index.
- `dw-hist <agent> --tail [N]` — last N messages directly from the JSONL.
- `dw-hist <agent> --thread <thread_id>` — print a single thread.

All of these are mode flags on the same agent positional. Today we ship only
`-q` and `--list`; the others are placeholders for the design's shape, not
implemented.

## 7. Tests

Unit:
- `test_lister.py` — `discover_agents(home)` against a fake `~/.deep_worker`
  tree with three agents (one with index, one without, one with an empty
  `.history_index/`). Assert returned list is exactly the one with
  `.history_index/lancedb/`.
- `test_cli.py` — argparse combinations: `--list` alone parses; agent + `-q`
  parses; conflicting combos exit non-zero; unknown agent exits 2; missing
  index exits 3.
- `test_search.py` — `run_search(...)` against a stubbed `search_history`
  that returns a fixed `HistorySearchResult` list; assert text and JSON
  formatting both contain the score.

No integration tests against a real LanceDB index — that path is exercised
by `semantic_search`'s own test suite and would require a heavy embedding
model in CI.

### 7.1 top_k test cases

`test_cli.py::test_top_k_zero_exits_2`, `::test_top_k_above_20_exits_2`,
`::test_top_k_negative_exits_2`. Validation happens **before** any heavy
import or call to `search_history` so a bogus `-k -1` doesn't pay the
embedding cold-start cost just to fail.

## 7.2 Future refactor (NON-blocking for v1)

The text formatter currently duplicates the per-result formatting block
from `history_memory_tool._run_history_search` (the `--- Result N
(score=X.XXX) --- / Thread: ... / Lines: ...` shape). To prevent drift,
a follow-up task should extract this to a shared helper. v1 ships a
parity test (`test_search.py::test_text_format_matches_history_memory_tool`)
that compares dw-hist's output against the byte-exact format that
`history_memory_tool` would produce for the same `HistorySearchResult`
list. If the production tool changes its formatting, our parity test
fails loudly and we update both at once.

## 8. What this design does NOT do

- Does not modify `deep-worker` argparse. (Explicit requirement.)
- Does not add anything to `pyproject.toml` of any existing package.
- Does not introduce new runtime dependencies — only re-uses
  `semantic_search` (already on PYTHONPATH for the agent process).
- Does not change any existing file in `semantic_search/` or
  `deep_worker_cli/`.

## 9. Non-goals / Open questions deliberately deferred

- Distribution via pip: not in scope.
- Wrapping into MCP / exposing over RPyC: not in scope (this is a host
  operator's tool, not an agent tool).
- Web UI: separate concern.
