# Design: `--dev` flag for `deep-worker` (run the agent on the dev codebase)

**Status:** implemented in the dev tree (branch `dev`). Not yet in the master tree (`<prod-tree>`) — will be merged after the dev branch is confirmed stable.
**Date:** 2026-05-13
**Related documents:**
- `deep_worker_web/app/config.py` — the already-implemented per-checkout port split (7887/7888)
- `<dev-tree>/docs/AGENT_SEPARATION_DESIGN.md` (if present — the general two-tree concept)

---

## 1. Problem statement

### 1.1 Current state

`deep-worker` is installed as a single symbolic link:

```
$ readlink -f $(which deep-worker)
<prod-tree>/deep_worker/deep_worker_cli/deep-worker
```

This bash launcher (`<prod-tree>/deep_worker/deep_worker_cli/deep-worker`) does literally the following:

```bash
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$SCRIPT_DIR:$PARENT_DIR/deepagents:${PYTHONPATH:-}"
cd "$SCRIPT_DIR" && DEEP_WORKER_ORIGINAL_CWD="$ORIGINAL_CWD" \
  exec python3 -m deep_worker_cli.cli.agent_command "$@"
```

`SCRIPT_DIR` always points at the directory where the script itself physically lives — i.e. `<prod-tree>/deep_worker/deep_worker_cli`. `PYTHONPATH` is built from the parent directories of *the same* copy. Consequently, no matter which directory the user passes to `--dir`, **the agent code always executes from `<prod-tree>`** (including `deep_worker_cli`, `deepagents`, `claude_code_sdk_provider`). `--dir` only changes the process working directory (what `os.getcwd()` sees), not the codebase.

### 1.2 What the user wants

> To test agents against the code in `<dev-tree>`, I propose adding a `--dev` option to the deep-worker command. If present, the agent is launched on the `<dev-tree>` codebase.

Goal: the same `deep-worker` launcher must be able to run agents either on the stable production codebase (`<prod-tree>`, branch `master`) or on the development codebase (`<dev-tree>`, branch `dev`) — chosen by `--dev`. Default behaviour (no flag) is **unchanged**: the production codebase, as today.

### 1.3 Why this is needed

What is already established in `deep_worker_web/app/config.py`:

* `<prod-tree>` — the "live" copy. User agents and the web server (port 7887) run on it. It must not be broken.
* `<dev-tree>` — the development copy. Code is edited, experiments are run, risky refactors are made on it. Its web server runs on 7888 so as not to conflict with prod.

Today only the web-server paths and ports are split, but the *executed agent code* is still shared — because the launcher references only the prod tree. `--dev` closes this gap: it switches the executed agent code to the dev tree.

---

## 2. What exactly "run the agent on codebase X" means

For the design to be complete, we must explicitly list **all** the points where the agent code is "bound" to a particular checkout. I found them by grepping both copies; the list is **exhaustive** within the current repository.

### 2.1 Bootstrap launcher (main point)

`<TREE>/deep_worker/deep_worker_cli/deep-worker` — the bash wrapper. Execution starts here and `PYTHONPATH` is defined here too.

### 2.2 Relative `sys.path.insert` (correct; work on their own)

These locations use `Path(__file__).parent...` and **automatically pick up the checkout they were imported from**:

| File (relative to tree root) | What it does |
| --- | --- |
| `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:20–22` | Adds `rpyc_async` (external repository, shared) |
| `deep_worker/deep_worker_cli/deep_worker_cli/agent.py:15–17` | Walks up to `deep_worker/deepagents` via `__file__.parent.parent.parent.parent` |
| `deep_worker/deep_worker_cli/deep_worker_cli/config.py:1654–1656` | Walks up to `<TREE>/` and adds it so that `claude_code_sdk_provider` can be imported |
| `deep_worker/deep_worker_cli/deep_worker_cli/tools.py:31–41` | Similarly — walks up to `<TREE>/` to import `claude_code_sdk_provider.web_search` |

These locations are resilient to a checkout switch: it is enough for the bash launcher to start Python from the right tree — all of them will automatically find "their own" `deepagents` and "their own" `claude_code_sdk_provider`.

### 2.3 Hardcoded absolute paths on `<prod-tree>` (do NOT work for `--dev`)

This is the "dirt" that `--dev` either bypasses or requires fixing. Full list (identical in both copies):

**In `deep_worker_cli` (affects the agent):**

| File | Line | What is hardcoded |
| --- | --- | --- |
| `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py` | 403 | `PathLib("deep_worker/deepagents")` — added to `sys.path` *inside* the agent process when the graph is raised. This means that even if the launcher set PYTHONPATH correctly, this line still **forces the prod tree into `sys.path` with first priority**. |

**In `deep_worker_web` (affects only the web server, not the agent itself):**

| File | Line | What is hardcoded |
| --- | --- | --- |
| `deep_worker_web/app/routers/agents.py` | 691–692 | `<prod-tree>/deep_worker/deep_worker_cli` and `/deepagents` are added to the sys.path of the worker handling HTTP requests |
| `deep_worker_web/app/services/agent_service.py` | 126 | `Path("<prod-tree>/deep_worker/deep_worker_cli")` — for spawning agent sub-processes from the web |
| `deep_worker_web/tests/...` (a dozen files) | various | Hardcoded paths for tests — relevant to `--dev` only if you want to run dev-tree tests |

**Note:** for the `--dev` task itself (running the *agent* on dev code), only `agent_process.py:403` is fatally critical. The rest in `deep_worker_web` is a separate story; the dev-tree web server already starts via its own `run.py` from `deep_worker_web/` and resolves its own paths differently.

### 2.4 External artifacts shared by both checkouts

* `/tmp/deep_worker_registry.json` — single process registry (see `process/registry.py:27`). Agent names are global. This is **intentional**: `dwcli` must see both kinds of agents in one list.
* `~/.deep_worker/<name>/` — a given agent's data; does not depend on the tree.
* `~/.deep_worker/agents/<name>/skills/` — the user's skills.
* `rpyc_async`, `claude_sdk` — external dependencies, mounted identically by both trees.

### 2.5 What must NOT be substituted

* `--dir` (the agent's working directory) is a separate axis. You can do `deep-worker foo --dev --dir /tmp` and get an agent *executing dev code* while working in `/tmp`. Useful for tests.
* The web server (`deep_worker_web`). It already has its own per-checkout split (port 7887/7888 — see `deep_worker_web/app/config.py`). `--dev` does not touch it.
* The process registry (`/tmp/deep_worker_registry.json`). Intentionally shared.

---

## 3. Solution architecture

### 3.1 Main idea

`--dev` must take effect **as early as possible** in the bash launcher — before Python even starts. If the launcher manages to set the correct `PYTHONPATH` and `exec` the correct Python module from the dev tree, everything else follows automatically (thanks to the relative `__file__.parent...` imports in § 2.2).

This means:

* `--dev` is parsed **in bash**, before `python3 -m`.
* The bash launcher `exec`s *the very same `deep_worker_cli.cli.agent_command`*, but from the dev tree.

### 3.2 Where the `--dev` logic lives

Logically — in **two** places:

1. **`<prod-tree>/deep_worker/deep_worker_cli/deep-worker`** (the bash launcher). This is "the one true launcher" that `~/.local/bin/deep-worker` points to. It intercepts `--dev`, switches `PYTHONPATH`, and `exec`s Python from the dev tree.

2. **`<dev-tree>/deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py`** (argparse). The `--dev` flag must also be **added to `create_parser()`** here, for two purposes:
   * So that `argparse` does not fail with an "unrecognized argument" error inside the dev tree (after switching we land in the dev variant of `agent_command.py`, and it too must know about this flag).
   * So that `--dev` is preserved in `launch_config` (for `--restart` — see § 3.6).

No other Python code **requires** the flag, because after the `PYTHONPATH` switch all dev code works "automatically".

### 3.3 Execution flow

#### 3.3.1 Without `--dev` (current behaviour, unchanged)

```
user $ deep-worker foo --provider claude-code-sdk --model opus
        │
        ▼
~/.local/bin/deep-worker  →  symlink  →
<prod-tree>/deep_worker/deep_worker_cli/deep-worker
        │
        ▼
bash: SCRIPT_DIR=<prod-tree>/deep_worker/deep_worker_cli
      PYTHONPATH="$SCRIPT_DIR:$PARENT_DIR/deepagents:..."
      exec python3 -m deep_worker_cli.cli.agent_command foo --provider ...
        │
        ▼
The Python agent runs on prod code from <prod-tree>
```

#### 3.3.2 With `--dev` (new behaviour)

```
user $ deep-worker foo --dev --provider claude-code-sdk --model opus
        │
        ▼
~/.local/bin/deep-worker  →  symlink  →
<prod-tree>/deep_worker/deep_worker_cli/deep-worker      ← launcher from the master tree
        │
        ▼
bash:  notices --dev in "$@", computes
       DEV_TREE=<dev-tree>
       DEV_LAUNCHER="$DEV_TREE/deep_worker/deep_worker_cli/deep-worker"
       Checks:   does DEV_LAUNCHER exist and is it executable?
                 is --dev used correctly (see § 3.4)?
       If OK: exec "$DEV_LAUNCHER" "$@" without --dev
        │
        ▼ (this is now the dev launcher from the dev tree)
bash: SCRIPT_DIR=<dev-tree>/deep_worker/deep_worker_cli
      PYTHONPATH="$SCRIPT_DIR:$PARENT_DIR/deepagents:..."
      DEEP_WORKER_DEV_TREE=1  (internal variable — see § 3.5)
      exec python3 -m deep_worker_cli.cli.agent_command foo --provider ...
        │
        ▼
The Python agent runs on dev code from <dev-tree>
```

Key observations:

* On the second pass of the bash launcher the `--dev` flag is already **removed** from the arguments. This is needed so the dev launcher does not loop by re-intercepting `--dev`. (Alternative — check `DEEP_WORKER_DEV_TREE=1` and ignore a secondary `--dev`; both work, removal is cleaner.)
* If `--dev` is passed but `DEV_LAUNCHER` is missing/not executable — the launcher fails with a clear message, **not** silently falling back to prod.
* If the variable `DEEP_WORKER_DEV_TREE` is already set when entering the master launcher — we are either in recursion (a bug) or someone set it manually for debugging; in both cases we continue as-is and do not try to handle `--dev`.

### 3.4 `--dev` contract

* A purely boolean flag, no value. A parallel to the existing `-sp`, `-tm`, `--clean`.
* Can be placed anywhere on the command line (`deep-worker --dev foo`, `deep-worker foo --dev`, etc.) — because the bash parser simply looks for an exact `--dev` occurrence in `"$@"` and removes the first such occurrence.
* No incompatibilities: `--dev` is orthogonal to all other flags.
* `--dev` does **not** affect:
  * the agent name (`foo` stays `foo`, the registry is shared);
  * `--dir` (working directory, a separate axis);
  * the web-server/port choice (the web is already split).

### 3.5 Environment variables (contract between bash and Python)

The bash launcher and Python must agree on markers. Proposed:

* `DEEP_WORKER_DEV_TREE` — `"1"` if the active codebase is dev. Otherwise not set (or `"0"`). Python can read it for:
  * Logging (show the user "Running from <dev-tree> (dev tree)").
  * Passing into `launch_config` (for `--restart`).
  * Reacting to edge cases (e.g. excluding the hardcoded prod path in `agent_process.py:403` — see § 3.7).

* `DEEP_WORKER_TREE_ROOT` — the absolute path of the active codebase (`<prod-tree>` or `<dev-tree>`). Useful for logs and for a future refactor of the hardcoded paths.

These variables are **inherited** by child processes (including the `claude` CLI that the agent later forks). Convenient: if it ever becomes necessary for `claude.exe` to also know which tree it lives in, the information will already be there.

### 3.6 `--restart` behaviour

`--restart` reads `launch_config` from `~/.deep_worker/<name>/session_metadata.json` and re-raises the agent with the same parameters (see `merge_restart_config` in `agent_command.py:58`). The correct behaviour for `--dev`:

* On the first run with `--dev`, Python (in the dev tree) saves a new field in `launch_config`:
  ```json
  "dev_tree": true
  ```
* On `deep-worker foo --restart` (WITHOUT `--dev`) the wrapper must notice the flag in the saved `launch_config` and automatically switch to the dev launcher.

Since `--restart` is parsed by Python, while the tree decision is made by bash *before* Python starts, there is a chicken-and-egg problem. The clean solution:

1. The bash launcher, on seeing `--restart` (and WITHOUT an explicit `--dev`), reads `~/.deep_worker/<name>/session_metadata.json` at its own risk (the name is taken from the first positional argument — that is `name`), looks at `.launch_config.dev_tree`, and if it is `true` — behaves as if it had seen `--dev`.
2. If the file does not exist or fails to parse — bash continues down the prod path (this is a safe fallback; Python will then emit a meaningful "previous launch not found" error anyway).

To avoid pulling in `jq` as a dependency, parse via `python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('launch_config',{}).get('dev_tree',False))"`. Python is always present — it is the interpreter of the main code.

Alternative (simpler, at the cost of UX): just require `deep-worker foo --restart --dev` if the launch was dev. I consider this **worse** — `--restart` promises "as it was", and breaking that for `--dev` is unpleasant. But the first implementation can start with the alternative and add auto-detection in a second iteration.

### 3.7 What to do with the hardcoded `deep_worker/deepagents` in `agent_process.py:403`

This line does:

```python
deepagents_path = PathLib("deep_worker/deepagents")
if deepagents_path.exists() and str(deepagents_path) not in sys.path:
    sys.path.insert(0, str(deepagents_path))
```

It is placed as the **first** element of `sys.path`, i.e. it has priority even over the launcher's `PYTHONPATH`. This means: if left as-is, then *inside a dev agent* `import deepagents` will still pull in the prod tree.

**In this design document** I do not propose changing this line (the user explicitly asked "don't change the code yet"), but I flag it as a **mandatory part of the `--dev` implementation**. Options:

1. **(recommended)** Replace the hardcode with a computation from `__file__`:
   ```python
   deepagents_path = Path(__file__).parent.parent.parent.parent / "deepagents"
   ```
   This is symmetric to what is already done in `agent.py:15`, `config.py:1654`, `tools.py:31`. No flags or variables are required — `__file__` itself reflects the tree we were imported from.

2. **(fallback)** Read `DEEP_WORKER_TREE_ROOT` from the environment and build the path from it:
   ```python
   tree_root = os.environ.get("DEEP_WORKER_TREE_ROOT", "<prod-tree>")
   deepagents_path = PathLib(tree_root) / "deep_worker" / "deepagents"
   ```

   Option 1 is strictly better — it requires no infrastructure and works even when `deep-worker` is run not through the bash launcher (e.g. under a debugger).

Both alternatives are a **separate code change** that must be made **in the same PR** as `--dev`. Without it, `--dev` will "leak" back into prod.

In `deep_worker_web` (`agents.py:691`, `services/agent_service.py:126`) there is an **analogous problem**, but it only manifests when working with the dev-tree web interface. The dev-tree web server is started separately (`deep_worker_web/run.py`), and those lines can already pull the prod `deep_worker_cli` into the dev web. This is a **separate task**; within `--dev` I only record it as technical debt, not fix it.

---

## 4. Code changes (checklist, without execution)

### 4.1 `<prod-tree>/deep_worker/deep_worker_cli/deep-worker` (bash)

Refinement of the master launcher:

```bash
#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
ORIGINAL_CWD="$(pwd)"

# --- NEW: dev-tree dispatch -------------------------------------------------
# Look for an explicit --dev anywhere in the args. If present, drop it and
# delegate to the dev-tree launcher, which will set up PYTHONPATH from
# the dev tree. This must run BEFORE we touch PYTHONPATH so the
# dev launcher gets a clean environment.
DEV_TREE="<dev-tree>"
DEV_LAUNCHER="$DEV_TREE/deep_worker/deep_worker_cli/deep-worker"

dev_requested=0
filtered_args=()
for arg in "$@"; do
  if [[ "$arg" == "--dev" && $dev_requested -eq 0 ]]; then
    dev_requested=1
  else
    filtered_args+=("$arg")
  fi
done

# Auto-detect --restart with previously dev launch_config.
if [[ $dev_requested -eq 0 ]]; then
  for arg in "${filtered_args[@]}"; do
    if [[ "$arg" == "--restart" ]]; then
      # First positional arg = agent name. Try to peek session_metadata.json.
      name="${filtered_args[0]:-}"
      meta="$HOME/.deep_worker/$name/session_metadata.json"
      if [[ -n "$name" && -f "$meta" ]]; then
        if python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('launch_config',{}).get('dev_tree',False))" "$meta" 2>/dev/null | grep -q '^True$'; then
          dev_requested=1
        fi
      fi
      break
    fi
  done
fi

if [[ $dev_requested -eq 1 && -z "${DEEP_WORKER_DEV_TREE:-}" ]]; then
  if [[ ! -x "$DEV_LAUNCHER" ]]; then
    echo "deep-worker: --dev requested but $DEV_LAUNCHER is missing or not executable" >&2
    exit 2
  fi
  export DEEP_WORKER_DEV_TREE=1
  export DEEP_WORKER_TREE_ROOT="$DEV_TREE"
  exec "$DEV_LAUNCHER" "${filtered_args[@]}"
fi
# --- END NEW ----------------------------------------------------------------
```

```bash
export PYTHONPATH="$SCRIPT_DIR:$PARENT_DIR/deepagents:${PYTHONPATH:-}"
export DEEP_WORKER_TREE_ROOT="${DEEP_WORKER_TREE_ROOT:-<prod-tree>}"

cd "$SCRIPT_DIR" && DEEP_WORKER_ORIGINAL_CWD="$ORIGINAL_CWD" \
  exec python3 -m deep_worker_cli.cli.agent_command "$@"
```

(This is **pseudocode** for the design doc; the final implementation will be drafted separately. What matters is the structure, not the exact quoting.)

### 4.2 `<dev-tree>/deep_worker/deep_worker_cli/deep-worker` (bash)

Must be identical to the master launcher with one difference: if we are already inside the dev tree (i.e. `SCRIPT_DIR` points to `<dev-tree>/...`), a repeated re-exec is NOT needed, even if `--dev` remains in the arguments (this should not happen — the filter in the master strips it — but it is worth guarding against).

In practice this is the same logic as in § 4.1, but with a guard: `if [[ "$SCRIPT_DIR" == "$DEV_TREE/deep_worker/deep_worker_cli" && $dev_requested -eq 1 ]] ; then # already in dev tree — just drop the flag ; fi`.

### 4.3 `deep_worker_cli/cli/agent_command.py` (Python, both copies)

In `create_parser()` (roughly after `--clean`):

```python
parser.add_argument(
    "--dev",
    action="store_true",
    dest="dev_tree",
    help=(
        "Run the agent using the development codebase at <dev-tree> "
        "instead of the production codebase at <prod-tree>. The flag "
        "is consumed by the bash launcher (deep-worker) before Python starts; it "
        "is also declared here so argparse does not reject it inside the dev tree "
        "and so it can be persisted to launch_config for --restart."
    ),
)
```

In `_PARSER_DEFAULTS` (on the same page):

```python
_PARSER_DEFAULTS = {
    ...,
    "dev_tree": False,
}
```

In `build_launch_config(args)`:

```python
launch_config = {
    ...,
    "dev_tree": getattr(args, "dev_tree", False)
                or os.environ.get("DEEP_WORKER_DEV_TREE") == "1",
}
```

(The `or` combination is needed because at the time of the re-exec the flag is physically removed from `sys.argv`, and its value arrives through the environment variable.)

In `merge_restart_config(saved, args)`:

```python
effective.dev_tree = saved.get("dev_tree", False)
if args.dev_tree != _PARSER_DEFAULTS["dev_tree"]:
    effective.dev_tree = args.dev_tree
```

No other Python edits are **required** for `--dev`. The "which code to use" logic is already handled by bash.

### 4.4 `process/agent_process.py:403` (Python, both copies)

Mandatory fix of the hardcode — see § 3.7. Without it `--dev` is NOT FULLY functional (the agent will execute `agent_process.py` from the dev tree, but `deepagents` will be pulled from prod). Change to:

```python
deepagents_path = Path(__file__).parent.parent.parent.parent / "deepagents"
```

(Here `Path(__file__).parent.parent.parent.parent` == `deep_worker`, and after `+ "deepagents"` we get `deep_worker/deepagents`.)

### 4.5 Documentation

* Update `<dev-tree>/README.md` and `<prod-tree>/README.md` (or the corresponding file with commands): add a section about `--dev`.
* The `--dev` line in `deep-worker`'s `--help` will appear automatically thanks to argparse.

### 4.6 Tests (minimal set for the PR)

1. **Unit**: parsing `--dev` in argparse, persisting to `launch_config`, restoring via `merge_restart_config`.
2. **Integration (bash)**:
   * `deep-worker dummy --dev --dir /tmp --help` — must show help (or some meaningful reaction), using the dev launcher. Can be checked via `DEEP_WORKER_TREE_ROOT` (exported into the shell after exec).
   * `deep-worker dummy_dev --dev --provider claude-code-sdk -sp --dir /tmp` (kill quickly with `pkill`): verify that `~/.deep_worker/dummy_dev/session_metadata.json` records `"dev_tree": true`.
   * `deep-worker dummy_dev --restart` (without `--dev`) — the launcher must switch to the dev tree on its own.
3. **Smoke**: both variants (`--dev` and without) live in parallel, names do not conflict — the registry in `/tmp/deep_worker_registry.json` contains both.

---

## 5. Edge-case behaviour

| Scenario | Behaviour |
| --- | --- |
| `deep-worker foo` (as now) | prod tree, unchanged |
| `deep-worker foo --dev` | dev tree |
| `deep-worker foo --dev --dev` | dev tree (the second `--dev` is filtered out by the filter, or does nothing again) |
| `deep-worker --dev foo` | dev tree (flag position does not matter) |
| `--dev` given, but `<dev-tree>/...` is missing | launcher fails with exit code 2, message to stderr |
| `--dev` given, but the dev tree is broken (e.g. a syntax error in `agent_command.py`) | fails with a normal Python traceback, prod is NOT affected |
| Running `deep-worker foo --restart` after a previous `--dev` | launcher automatically switches to the dev tree (reads `dev_tree: true` from session_metadata) |
| Running `deep-worker foo --restart --dev` after a previous run WITHOUT `--dev` | dev tree (the explicit flag wins) |
| Running `deep-worker foo --restart` after a run WITHOUT `--dev` | prod tree (default) |
| Direct invocation of the dev launcher (`<dev-tree>/deep_worker/deep_worker_cli/deep-worker foo`) | dev tree (native behaviour of the dev launcher — it starts its own tree) |
| Direct run of `python3 -m deep_worker_cli.cli.agent_command foo --dev` WITHOUT the bash wrapper | argparse accepts the flag (we declared it), but there will be no re-exec — the agent starts with whatever `sys.path` the user assembled manually. This is **legal**: whoever runs python directly knows what they are doing. For UX one could add a warning "`--dev only takes effect when invoked via the deep-worker shell launcher`". |
| Parallel run of `deep-worker prod_agent` and `deep-worker dev_agent --dev` | both live in the same `/tmp/deep_worker_registry.json`, each with its own Python process from its own tree, no conflicts |
| `dwcli prod_agent` to the prod agent, `dwcli dev_agent` to the dev agent | work over RPyC on the ports from the registry, the tree does not matter |

---

## 6. What this design does **not** do (and why)

* The web-port selector is **not changed**. It is already split correctly (`deep_worker_web/app/config.py`). `--dev` concerns only the agent process.
* The process registry is **not unified** between trees. Intentionally: the user must see both prod and dev agents in one list in `dwcli`.
* A "hybrid" is **not implemented**: an agent in the prod tree suddenly using the dev copy of `deepagents`. If that is ever needed — it is a separate, much more complex scenario (requiring not a bash launcher but a full `sys.path` overlay mechanism).
* The hardcoded `<prod-tree>` path debris in `deep_worker_web` is **not fixed** (see § 2.3). That is separate technical debt, orthogonal to the task.

---

## 7. Summary of changes on one screen

| File | Type | Complexity | Mandatory |
| --- | --- | --- | --- |
| `<prod-tree>/deep_worker/deep_worker_cli/deep-worker` | bash | ~30 lines | yes |
| `<dev-tree>/deep_worker/deep_worker_cli/deep-worker` | bash | ~30 lines (identical) | yes |
| `deep_worker_cli/cli/agent_command.py` (both copies) | Python, argparse + launch_config | ~10 lines | yes |
| `deep_worker_cli/process/agent_process.py:403` (both copies) | Python, replace the hardcoded `<prod-tree>/...` with `Path(__file__).parent...` | 1 line | **yes, otherwise `--dev` breaks** |
| README, --help, tests | doc/test | as desired | recommended |

No new Python module is needed. No new dependency is needed. The solution is **strictly local in both repositories** (no global settings, no user env vars, no systemd).

---

## 8. Agreed decisions

1. **Flag name** — `--dev`.

2. **Environment variable names** — `DEEP_WORKER_DEV_TREE` (value `"1"`, otherwise not set) and `DEEP_WORKER_TREE_ROOT` (the absolute path of the active tree).

3. **`--restart` auto-detect** — implement right away. `--restart` must behave the same as for the other options (`-sp`, `-tm`, `--model`, …): the value is inherited from `session_metadata.json` if there is no explicit override.

4. **Hardcodes of `<prod-tree>` in `deep_worker_web`** — fix in this same PR, do not postpone.

5. **When the dev tree is missing** — `exit 2` + message to stderr. No fallbacks to prod.

---

**Full path to this document:** `<dev-tree>/docs/DESIGN_DEV_FLAG.md`

---

## 9. Implementation status (at the time of the commit)

All changes were made **only in the dev tree** (branch `dev`).
Nothing was changed in the prod tree (branch `master`) — the pre-existing prod launcher there continues to work as before. After the dev branch is confirmed stable, the changes will be carried into master with a separate merge.

### Changed files (dev tree)

| File | What was done |
| --- | --- |
| `deep_worker/deep_worker_cli/deep-worker` | Fully rewritten. Parses `--dev`, auto-detects `--restart` via `session_metadata.json`, re-execs into the dev-tree launcher, exit 2 when the dev tree is missing. Supports `DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE` (testing hook). |
| `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py` | `--dev` added to `create_parser()`, `dev_tree` to `_PARSER_DEFAULTS`, to `build_launch_config()` (taking the env var `DEEP_WORKER_DEV_TREE` into account) and to `merge_restart_config()`. |
| `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py` | The hardcoded `deep_worker/deepagents` replaced with `Path(__file__).resolve().parent.parent.parent.parent / "deepagents"` (lines ~403). |
| `deep_worker_web/app/routers/agents.py` | `_spawn_subprocess`: hardcoded paths in `PYTHONPATH` replaced with `__file__`-relative ones. Added env `DEEP_WORKER_DEV_TREE=1` (if tree root = `<dev-tree>`) and `DEEP_WORKER_TREE_ROOT` for spawned agents. |
| `deep_worker_web/app/services/agent_service.py` | `AgentService.__init__`: the hardcoded path to `deep_worker_cli` replaced with `__file__`-relative. |
| `deep_worker_web/tests/conftest.py` | Hardcoded `sys.path` entries for tests replaced with `__file__`-relative ones — dev-tree tests now import dev-tree code. |

### Added tests (dev tree)

| File | Number of tests | What it checks |
| --- | --- | --- |
| `deep_worker/deep_worker_cli/tests/unit_tests/cli/test_dev_flag.py` | 12 | argparse accepts `--dev`; `build_launch_config()` persists `dev_tree` from the argument OR from the env var; `merge_restart_config()` inherits `dev_tree` from session_metadata. |
| `deep_worker/deep_worker_cli/tests/integration_tests/test_dev_launcher.py` | 12 | The real bash launcher: dispatch on `--dev`, filtering of `--dev` out of argv, positional independence of the flag, exit 2 when the dev tree is missing, `--restart` auto-detect via `session_metadata.json`, explicit `--dev` beats saved `dev_tree=false`. |
| `deep_worker/deep_worker_cli/tests/unit_tests/process/test_agent_process_deepagents_path.py` | 4 | Regression test: `agent_process.py` must not contain a hardcoded prod or dev path to `deepagents`; must use `__file__`-relative resolution. |
| `deep_worker_web/tests/unit/test_tree_path_resolution.py` | 6 | Regression test: `agents.py` and `agent_service.py` must not contain hardcoded tree paths; must use `__file__`-relative resolution. |

### Principle "tests validate the same codebase they live in"

All paths in the tests are computed via `Path(__file__)`-relative resolution — therefore the copy of the tests in the dev tree validates only dev-tree code, and the copy in the master tree (when the merge happens) will validate master-tree code. No cross-tree dependencies.

### Smoke verification

```
$ <dev-tree>/deep_worker/deep_worker_cli/deep-worker --help | grep -A 2 dev
                   [-tm] [-asm] [--config CONFIG] [--clean] [--dev]
                   ...
  --dev                 Run the agent using the development codebase at
                        <dev-tree> instead of the production
                        codebase at <prod-tree>. ...
```
