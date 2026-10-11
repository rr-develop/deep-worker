# Launch Config Persistence

## Overview

Agent launch parameters (provider, model, working directory, CLI flags) are now
persisted to disk in `~/.deep_worker/{agent_name}/session_metadata.json`.
This allows inspecting how an agent was launched **even after the process has
terminated**.

## Problem

Previously, launch parameters (e.g. `--provider claude-code-sdk --model opus -sp -tm`)
were only available at runtime:

- In-memory via `AgentConfig` and `AgentProcess` attributes
- Via live RPyC `get_metadata()` call (requires running process)
- Via `/proc/{pid}/cmdline` (lost when process exits)

After the agent was killed or crashed, there was **no way** to determine what
parameters it was started with.

## Solution

A `launch_config` dict is built from CLI arguments in `agent_command.py` and
passed through `AgentProcess` → `SessionManager`, which persists it to
`session_metadata.json`.

### Data flow

```
agent_command.py::run_agent()
    │
    ├─ builds launch_config dict from argparse args + AgentConfig
    │
    └─ AgentProcess(launch_config=launch_config)
           │
           └─ agent.start()
                  │
                  └─ SessionManager(data_dir, launch_config=launch_config)
                         │
                         └─ writes to ~/.deep_worker/{name}/session_metadata.json
```

### File format

`~/.deep_worker/{agent_name}/session_metadata.json`:

```json
{
  "status": "idle",
  "tasks_completed": 0,
  "message_count": 0,
  "error_message": null,
  "last_activity": "2026-04-11T22:00:55.651807",
  "launch_config": {
    "provider": "claude-code-sdk",
    "model": "opus",
    "working_dir": "/path/to/project",
    "skip_permissions": true,
    "task_manager": true,
    "memory_only": false,
    "use_sqlite": false,
    "clean": true,
    "started_at": "2026-04-11T16:59:41.692129",
    "pid": 3726921,
    "cli_args": "myagent -sp -tm --dir /path/to/project --provider claude-code-sdk --model opus --clean"
  }
}
```

### launch_config fields

| Field | Type | Description |
|-------|------|-------------|
| `provider` | string | AI provider (`deepseek`, `claude-code-sdk`, `gemini`, `openai`) |
| `model` | string | Resolved model name (e.g. `opus`, `deepseek-chat`) |
| `working_dir` | string | Agent working directory (absolute path) |
| `skip_permissions` | bool | Whether `-sp` / `--skip-permissions` was passed |
| `task_manager` | bool | Whether `-tm` / `--task-manager` was enabled |
| `memory_only` | bool | Whether `--memory-only` was passed |
| `use_sqlite` | bool | Whether `--use-sqlite` was passed |
| `clean` | bool | Whether `--clean` was passed |
| `started_at` | string | ISO 8601 timestamp of launch |
| `pid` | int | Process ID at launch time |
| `cli_args` | string | Raw CLI arguments as passed to the command |

## Behavior

- **On fresh start**: `launch_config` is written to disk during `SessionManager.__init__()`.
- **On status updates**: `launch_config` is preserved — `set_status()`, `set_error()`,
  `increment_tasks_completed()` etc. do not modify it.
- **On reload**: A new `SessionManager` instance without `launch_config` parameter
  loads the previously saved config from disk.
- **On restart with new params**: The new `launch_config` overwrites the old one.
- **Backward compatibility**: Old `session_metadata.json` files without `launch_config`
  are loaded without error; `manager.launch_config` will be `None`.

## Files changed

| File | Change |
|------|--------|
| `session/session_manager.py` | Added `launch_config` parameter, `_build_metadata_dict()`, updated `_load_metadata()` |
| `process/agent_process.py` | Added `launch_config` parameter, passes it to `SessionManager` |
| `cli/agent_command.py` | Builds `launch_config` dict from CLI args and `AgentConfig` |
| `tests/unit_tests/session/test_session_manager_launch_config.py` | 11 TDD tests |

## Tests

```bash
python3 -m pytest tests/unit_tests/session/test_session_manager_launch_config.py -v
```

11 tests covering:
- `launch_config` saved on init
- Stored as attribute
- No launch_config is OK (backward compat)
- CLI args preserved
- Preserved across status updates (set_status, set_error, increment)
- Survives reload from disk
- New config overwrites old
- Old metadata without launch_config loads fine
- Status updates on old metadata don't inject launch_config
