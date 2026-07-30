# DESIGN — Fix: `deep-worker --reindex-history` fails offline with
# `No module named 'semantic_search.history_indexer'`

## Status
Proposed — awaiting review.

## Symptom (user repro)
```
$ deep-worker --reindex-history my_job_analyst
Error: failed to clear history index: No module named 'semantic_search.history_indexer'
```
Agent `my_job_analyst` is **not running** (offline CLI path).

## Root cause
The `semantic_search` package physically lives "wrapped":
```
<tree>/semantic_search/                 <- wrapper folder, WITHOUT __init__.py
<tree>/semantic_search/semantic_search/ <- the real package (history_indexer.py etc.)
```
For `import semantic_search.history_indexer` to work, the **wrapper folder**
`<tree>/semantic_search` must be on `sys.path`. The working runtime modules do
this with an explicit injection (single-runtime):
- `middleware/history_memory.py:26-30`
- `history_memory_tool.py:20-25`
- `deepagents/tools/project_memory.py:20`
  each one: `Path(__file__).parent×5 / "semantic_search"` → `sys.path.insert(0, …)`.

The `--reindex-history` feature (commit `72c55bb`) added TWO new call sites
that import `semantic_search.history_indexer` **without** this injection:
- `cli/agent_command.py:2273 _history_indexer_funcs()` — **offline path** (broken)
- `process/agent_process.py:77 _history_indexer_funcs()` — running path (usually
  "saved" by the fact that the middleware is already imported at agent startup
  and the path is already injected; but relying on that side effect is fragile).

The launcher (`deep-worker`, prod without `--dev`) sets
`PYTHONPATH=$SCRIPT_DIR:$PARENT_DIR/deepagents:$PARENT_DIR` — `semantic_search`
is NOT there at all. On the offline path nothing injects the path →
`semantic_search` resolves as a namespace package from a foreign root (in the
repro, a `semantic_search` wrapper folder in a different local checkout) → the
`history_indexer` submodule is not there → `ModuleNotFoundError`.

A repro in a clean environment (`env -i`) confirms both the cause and the fix:
injecting `<tree>/semantic_search` before the import → `history_indexer`
resolves, `clear_history_index` / `index_all_history_files` become available.

## Why the tests did not catch it
The feature's integration test ran in an environment where `semantic_search`
was already on `sys.path` (via conftest / an already-imported middleware), so
the offline-path hole in a clean environment did not surface. The test did NOT
isolate `sys.path`.

## Fix design

### Core: a single helper (single source of truth)
Introduce a common helper that removes the duplicated `parent×5` pattern and
provides a single point of truth:

Location: `deep_worker_cli/semantic_search_path.py` (a new lightweight module,
no heavy imports — only `pathlib`/`sys`).

```python
from __future__ import annotations
import sys
from pathlib import Path

def ensure_semantic_search_importable() -> bool:
    """Idempotently put the semantic_search WRAPPER dir on sys.path.

    Returns True if the wrapper dir exists (now importable), else False.
    Safe to call many times; no-op when already present.
    """
    # this file: <tree>/deep_worker/deep_worker_cli/deep_worker_cli/semantic_search_path.py
    #   parent(1) deep_worker_cli(pkg) parent(2) deep_worker_cli parent(3)
    #   deep_worker parent(4) <tree>  -> /semantic_search wrapper
    wrapper = Path(__file__).resolve().parent.parent.parent.parent / "semantic_search"
    if not wrapper.exists():
        return False
    s = str(wrapper)
    if s not in sys.path:
        sys.path.insert(0, s)
    return True
```
(The exact number of `parent`s is verified by computing from the file's real
path; the implementation pins it with a unit test, not "by eye".)

### Application
1. `cli/agent_command.py::_history_indexer_funcs()` — call
   `ensure_semantic_search_importable()` BEFORE the first import; if it returns
   `False` (no wrapper) — fail loud with a clear message, not a bare
   ModuleNotFoundError.
2. `process/agent_process.py::_history_indexer_funcs()` — the same (remove the
   hidden dependency on middleware import order).

### Optional (NOT in this fix, to keep the surface small)
Migrate the 3 existing call sites to the common helper — as a separate task;
here we only add the helper and fix the 2 broken call sites. The existing
behavior of the working modules is untouched (risk of regression in the hot
path).

## Contract / architecture change
None. The feature's behavior stays the same; only the import resolution on the
offline path is fixed, plus a fail-loud is added. No new threads, polling, or
777/666 permissions.

## Testing plan (TDD)
- **Unit 1**: `ensure_semantic_search_importable()` returns True and adds the
  existing wrapper to `sys.path`; it is idempotent (a repeated call does not
  duplicate).
- **Unit 2**: the path computed by the helper really points at
  `<tree>/semantic_search` and contains the nested `semantic_search/history_indexer.py`
  (protection against a wrong number of `parent`s — a regression guard for the
  future).
- **Unit 3**: `_history_indexer_funcs()` (offline) imports both functions after
  the helper runs (no mocking needed — the real package is in place).
- **Integration (no mocks)**: in a SUB-PROCESS with a clean environment
  (`env -i`, PYTHONPATH as the launcher sets it, WITHOUT semantic_search) run
  `python -m deep_worker_cli.cli.agent_command --reindex-history <tmp_agent>`
  on a temporary summaries directory with 1 valid `*.jsonl` → expect success
  (reindex stats) and exit 0. This test would have caught the original bug
  (isolating sys.path via a separate process — the key thing the previous test
  lacked).

## Backward compatibility / rollback
Additive. Rollback is a trivial revert of the fix commit. No data is touched.
