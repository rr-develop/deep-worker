# Design: `--restart` Option

**Status: ✅ IMPLEMENTED**

## Overview

The `--restart` option allows re-launching an agent using the same parameters
from its previous run, without having to re-type them. It reads saved
`launch_config` from `~/.deep_worker/{name}/session_metadata.json` and uses
those values as defaults, with the ability to override any of them via
additional CLI flags.

## Usage

```bash
# Restart with exact same parameters as last time
deep-worker --restart myagent

# Restart but change model
deep-worker --restart myagent --model sonnet

# Restart with clean history (must be explicit)
deep-worker --restart myagent --clean

# Restart with a different working directory
deep-worker --restart myagent --dir /path/to/other-project
```

## Behavior

### 1. Loading saved parameters

When `--restart` is specified, the command:

1. Reads `~/.deep_worker/{name}/session_metadata.json`
2. Extracts the `launch_config` section
3. Uses its fields as **defaults** for this launch

The following fields are restored from `launch_config`:

| Field | Restored as | Notes |
|-------|-------------|-------|
| `provider` | `--provider` | e.g. `claude-code`, `deepseek` |
| `model` | `--model` | e.g. `opus`, `deepseek-chat` |
| `working_dir` | `--dir` | Absolute path |
| `skip_permissions` | `-sp` | Boolean flag |
| `task_manager` | `-tm` | Boolean flag |
| `memory_only` | `--memory-only` | Boolean flag |
| `use_sqlite` | `--use-sqlite` | Boolean flag |

**NOT restored:**
- `clean` — always defaults to `false` on restart (history is preserved).
  Must be explicitly passed as `--clean` to take effect.
- `pid` — new PID assigned automatically.
- `started_at` — new timestamp generated.
- `cli_args` — rebuilt from the effective parameters.

### 2. Override behavior

Any CLI flag explicitly provided alongside `--restart` **overrides** the
corresponding saved value:

```bash
# Saved: --provider claude-code --model opus -sp -tm --dir /path/to/project
# Effective: --provider deepseek --model opus -sp -tm --dir /path/to/project
deep-worker --restart myagent --provider deepseek
```

Override detection uses argparse defaults: if a flag differs from the argparse
default, it was explicitly provided and takes priority. For boolean flags
(`-sp`, `-tm`, `--clean`, `--memory-only`, `--use-sqlite`), the user can
explicitly set them to override the saved value.

### 3. Handling a running agent

Before starting, `--restart` checks the `ProcessRegistry`:

| Agent state | Behavior |
|---|---|
| **Not running** | Launch normally with restored parameters |
| **Running, idle/waiting_input** | Stop the running process (SIGTERM), wait for exit, then launch |
| **Running, actively working (status=running)** | Return error, do not stop |

"Actively working" means `session_metadata.json` has `status: "running"`.
Statuses `idle`, `waiting_input`, `error`, `shutdown` are considered safe
to stop.

Stopping procedure:
1. Look up PID from `ProcessRegistry`
2. Send `SIGTERM` to the process
3. Wait up to 10 seconds for process to exit (poll `/proc/{pid}`)
4. If still alive after 10s, return error (do not SIGKILL)
5. On successful stop, clean up stale registry entry and proceed with launch

### 4. No previous launch_config

If `--restart` is used but `~/.deep_worker/{name}/session_metadata.json` does
not exist or does not contain `launch_config`, the command returns an error:

```
Error: Cannot restart agent 'myagent' — no previous launch configuration found.
Start the agent normally first:  deep-worker myagent --provider ... --model ...
```

### 5. Persistence of launch_config on restart

When `--restart` launches the agent, the **effective** parameters (saved +
overrides) are written to `session_metadata.json` as the new `launch_config`.
This means:

- Subsequent `--restart` calls will use the most recent effective parameters
- If you override `--model sonnet` once, future `--restart` calls will
  default to `sonnet` (not the original `opus`)

The `clean` field in the saved `launch_config` is always set based on
whether `--clean` was explicitly provided in this invocation (not inherited
from previous runs).

## Implementation plan

### Files to change

#### 1. `cli/agent_command.py`

- Add `--restart` argument to `create_parser()`:
  ```python
  parser.add_argument(
      "--restart",
      action="store_true",
      help="Restart agent with parameters from previous launch. "
           "Additional options override saved values.",
  )
  ```

- Add `restart_agent()` async function:
  ```
  async def restart_agent(args) -> int:
      1. Validate: args.name is required
      2. Load launch_config from ~/.deep_worker/{name}/session_metadata.json
      3. If no launch_config: return error
      4. Check ProcessRegistry for running agent:
         a. If running + status=="running": return error
         b. If running + idle/waiting/error: stop it (SIGTERM + wait)
      5. Merge saved config with CLI overrides
      6. Build AgentConfig from merged values
      7. Build new launch_config dict from effective values
      8. Call run_agent() with merged args
  ```

- Update `main()` to handle `--restart` before the normal flow.

#### 2. `process/agent_stop.py` (new file)

Utility function to gracefully stop a running agent:

```python
async def stop_agent(name: str, registry: ProcessRegistry, timeout: int = 10) -> bool:
    """Stop a running agent by name.

    Returns True if agent was stopped successfully.
    Raises RuntimeError if agent is actively working.
    Raises TimeoutError if agent didn't stop within timeout.
    """
```

This is extracted as a separate utility because it may be reused by
future commands (e.g. `deep-worker --stop <name>`).

#### 3. `session/session_manager.py`

No changes needed — `launch_config` persistence already works.

#### 4. `process/agent_process.py`

No changes needed — already accepts `launch_config` parameter.

### Merge logic (pseudocode)

```python
def merge_restart_config(saved: dict, args: argparse.Namespace) -> argparse.Namespace:
    """Apply saved launch_config as defaults, then override with explicit CLI args."""

    # Start with saved values
    effective = argparse.Namespace()
    effective.provider = saved.get("provider", "deepseek")
    effective.model = saved.get("model")
    effective.dir = Path(saved["working_dir"])  # required in saved
    effective.skip_permissions = saved.get("skip_permissions", False)
    effective.task_manager = saved.get("task_manager", False)
    effective.memory_only = saved.get("memory_only", False)
    effective.use_sqlite = saved.get("use_sqlite", False)
    effective.clean = False  # NEVER inherit --clean

    # Override with explicitly provided CLI args
    # (detect via comparison with argparse defaults)
    if args.provider != "deepseek":  # parser default
        effective.provider = args.provider
    if args.model is not None:  # parser default is None
        effective.model = args.model
    if args.dir is not None:  # parser default is None
        effective.dir = args.dir
    if args.skip_permissions:  # parser default is False
        effective.skip_permissions = True
    if args.task_manager:  # parser default is False
        effective.task_manager = True
    if args.clean:  # parser default is False
        effective.clean = True
    if args.memory_only:  # parser default is False
        effective.memory_only = True
    if args.use_sqlite:  # parser default is False
        effective.use_sqlite = True

    return effective
```

**Important caveat for boolean flags:**

With `store_true` flags, argparse cannot distinguish "not passed" from
"explicitly passed as False". The merge logic defaults to the saved value.
If user passes the boolean flag, it overrides to `True`. There is no way
to override a saved `True` back to `False` via `--restart` — user must do
a full launch without `--restart` for that.

This is acceptable because:
- The common case is restarting with same flags
- Users who want to change flags can do a normal launch
- Adding `--no-sp`, `--no-tm` negation flags is possible as a follow-up

### Error messages

```
# No previous config
Error: Cannot restart agent 'myagent' — no previous launch configuration found.
Start the agent normally first:  deep-worker myagent --provider ... --model ...

# Agent is actively working
Error: Cannot restart agent 'myagent' — it is actively running (status: running).
Wait for it to finish or interrupt it first.

# Agent didn't stop in time
Error: Agent 'myagent' (PID 12345) did not stop within 10 seconds.
Check the process manually: kill -9 12345

# Missing agent name
Error: Agent name is required with --restart
```

### Test plan

#### Unit tests (`tests/unit_tests/cli/test_restart_option.py`)

1. **Merge logic tests:**
   - Saved config used when no CLI overrides
   - CLI override replaces saved value (provider, model, dir)
   - Boolean flags override when explicitly passed
   - `--clean` never inherited from saved config
   - `--clean` applied when explicitly passed with `--restart`

2. **Validation tests:**
   - Error when no session_metadata.json exists
   - Error when session_metadata.json has no launch_config
   - Error when agent name is missing

3. **Agent stop tests:**
   - Stop idle agent (SIGTERM sent, wait, success)
   - Error on actively running agent
   - Timeout handling

4. **Integration-like tests (mocked ProcessRegistry + filesystem):**
   - Full restart flow: load config → stop agent → merge → launch
   - Restart with overrides persists new launch_config
   - Multiple sequential restarts use latest config each time

## Sequence diagram

```
User: deep-worker --restart myagent --model sonnet

  ┌─ Read ~/.deep_worker/myagent/session_metadata.json
  │   └─ launch_config: {provider: claude-code, model: opus, dir: /path/to/project, ...}
  │
  ├─ Check ProcessRegistry
  │   └─ myagent is running (PID 12345)
  │       └─ Read status from session_metadata.json → "idle" ✓
  │
  ├─ Stop agent: SIGTERM → PID 12345
  │   └─ Wait up to 10s → process exits ✓
  │
  ├─ Merge: saved {model: opus} + CLI {model: sonnet} → effective {model: sonnet}
  │
  ├─ Build launch_config from effective params
  │   └─ {provider: claude-code, model: sonnet, dir: /path/to/project, clean: false, ...}
  │
  ├─ Build AgentConfig(provider=claude-code, model=sonnet, working_dir=/path/to/project)
  │
  └─ AgentProcess(name=myagent, config=..., launch_config=...).start()
      └─ SessionManager writes new launch_config to session_metadata.json
```

## Out of scope (future work)

- `--stop <name>` — standalone stop command (shares stop logic)
- `--no-sp`, `--no-tm` — negation flags for boolean overrides
- `--restart --all` — restart all known agents
- Restart from YAML config file + launch_config merge

## Implementation

### Files added

| File | Purpose |
|------|---------|
| `process/agent_stop.py` | `stop_agent()` — graceful SIGTERM + wait |
| `tests/unit_tests/cli/test_restart_option.py` | 24 TDD tests |

### Files modified

| File | Changes |
|------|---------|
| `cli/agent_command.py` | `--restart` parser arg; `merge_restart_config()`, `load_launch_config()`, `build_launch_config()`, `restart_agent()` functions; pre-flight validation before `daemonize()` |

### Tests

```bash
python3 -m pytest tests/unit_tests/cli/test_restart_option.py -v
# 24 tests: merge logic (12), stop_agent (6), validation (3), persistence (3)
```

### Live verification

Tested with `test_restart_verify` agent:
1. Start: `deep-worker test_restart_verify -sp --dir /path/to/project --provider claude-code --model opus --memory-only`
2. Kill, then restart: `deep-worker --restart test_restart_verify` → restored all params
3. Restart with override: `deep-worker --restart test_restart_verify --model sonnet` → model changed, rest preserved
4. Error case: `deep-worker --restart test_nonexistent_agent_xyz` → clear error in terminal
