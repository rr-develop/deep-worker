# deep-worker --info <name>

## Goal

Add `deep-worker --info <agent_name>` to display the launch configuration
and runtime status of an agent, read-only, without starting or restarting
the agent.

## CLI design

- `--info` is a boolean flag (mutually exclusive with `--restart`, `--list`, etc.)
- Agent name comes from the positional `name` argument
- `deep-worker --info <name>` → show launch config + status
- `deep-worker --info` (no name) → exit code 2, usage message
- No conflict with existing flags (verified via `create_parser()`)

## Output format

```
agent: <agent_name>
provider: claude-code
model: opus
working_dir: the development codebase
tree_root: the development codebase (dev_tree=true)
commit: <sha> (dirty)
started_at: 2026-06-05T20:14:29
pid: 1842836 (running)
flags:
  skip_permissions: true
  task_manager: true
  history_memory: true
  memory_only: false
  use_sqlite: false
  auto_approve_send_message: false
  clean: false
  dev_tree: true
cli_args: --restart <agent_name> -m
status: waiting_input
last_activity: 2026-06-05T20:18:42
```

## Data sources

| Field | Source |
|-------|--------|
| agent, provider, model | `load_launch_config(name)` (from `session_metadata.json`) |
| tree_root, commit | `load_launch_config(name)` + `get_git_commit_info()` |
| started_at, pid, cli_args | `load_launch_config(name)` |
| running/NOT RUNNING | `ProcessRegistry().lookup(name)` (check if pid is alive) |
| status, last_activity | `session_metadata.json` top-level keys |
| dev_tree | `launch_config.dev_tree` |

## Error handling

- No `session_metadata.json` → exit code 1 with clear error
- Unknown agent name → same error
- Git SHA unavailable → show `commit: unknown`

## Files to modify

1. `deep_worker_cli/cli/agent_command.py`
   - `create_parser()`: add `--info` flag
   - `main()`: add dispatch for `--info` after name check
   - New function: `show_agent_info(name: str) -> int`
2. Tests in `tests/unit_tests/cli/test_info_flag.py`
3. Design doc (this file)
