# Startup Diagnostic Log — Design

**Issue:** #86 — Log commit SHA + launch config on every agent start/restart.

## Goal

Every time an agent starts or restarts (fresh `deep-worker <name>`, `--restart <name>`, web-spawn), a single-line structured diagnostic entry is written to the agent's log file containing:

1. Git commit SHA of the running codebase
2. Full launch configuration (provider, model, flags etc.)

## Where to write

After daemonization, the process's stdout is redirected to `~/.deep_worker/logs/<name>.out.log` (set in `agent_command.py::daemonize()`). Agent messages and `print()` calls flow there. The diagnostic line will be a `print()` at the start of `AgentProcess.start()` — the earliest point after daemonization where all config is available.

## Format

```
[STARTUP] agent=<name> provider=<provider> model=<model> tree=<tree_root> commit=<sha> describe=<describe> pid=<pid> <flag>=<value> ...
```

Key=value pairs, space-separated, easily grep-able. All critical config fields are included. `commit=unknown` if git SHA cannot be obtained. `describe=<sha>-dirty` if the tree has uncommitted changes.

## Implementation plan

### 1. Add `tree_root` to launch_config

In `agent_command.py`:
- `run_agent()`: add `tree_root` key to the inline launch config dict (~line 964)
- `build_launch_config()`: add `tree_root` key (~line 368)

The tree root is the repo root: `Path(__file__).resolve().parents[4]` (same pattern already used at line 22).

### 2. Add git SHA helper in `agent_process.py`

A module-level async function `_get_git_commit_info(tree_root: str) -> tuple[str, str]` that runs:
- `git -C <tree_root> rev-parse HEAD` → commit SHA
- `git -C <tree_root> describe --always --dirty` → describe string

On any failure (no git, no .git, etc.) returns `("unknown", "unknown")` — never crashes.

### 3. Write diagnostic at start of `AgentProcess.start()`

At the top of `start()`, after basic initialisation (timestamp, event loop), build the key=value string from:
- `self.name`, `self.launch_config` (provider, model, working_dir, flags...)
- `_get_git_commit_info(self.launch_config.get("tree_root", ""))`
- `os.getpid()`

And `print()` it as a single line.

### 4. Coverage

All three start paths converge on `AgentProcess.start()`:
- Fresh start: `main()` → daemonize → `run_agent()` → `AgentProcess(...)` → `agent.start()`
- Restart: `main()` → daemonize → `restart_agent()` → `run_agent()` → `agent.start()`
- Web-spawn: `agent_service.py` → `AgentProcess(...)` → `agent.start()`

One injection point, complete coverage.

## Out of scope

- No changes to session_metadata.json format
- No changes to deep_worker_web server behaviour
