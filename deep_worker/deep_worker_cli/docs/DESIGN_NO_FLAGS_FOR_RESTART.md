# Design: `--no-*` Flags for `deep-worker --restart`

**Status: Draft**

## Problem

When `deep-worker --restart <name>` is run, all saved boolean flags are restored
(`-sp`, `-tm`, `--dev`, `--memory-only`, `--use-sqlite`, `-m`/`--history-memory`,
`-asm`/`--auto-approve-send-message`, `--clean`).

If an agent was started with `-tm`, the task manager turns back on at restart.
To turn it off, you have to **explicitly edit the metadata file before launching** —
there is no CLI way to disable a saved flag.

## Solution

Add a paired `--no-*` flag for each boolean flag, which on `--restart`
sets the corresponding `dest` to `False`. The existing mechanism
`_explicit_cli_flags()` + `_is_override()` already handles this case:
once in the `explicit` set, the flag overrides the saved value.

## Design Decisions

### 1. Which flags get a `--no-*` version

All boolean flags that are saved in `launch_config` and applied on restart:

| Flag | `dest` | `--no-*` flag |
|---|---|---|
| `-sp` / `--skip-permissions` | `skip_permissions` | `--no-skip-permissions` |
| `-tm` / `--task-manager` | `task_manager` | `--no-task-manager`, `--no-tm` |
| `-asm` / `--auto-approve-send-message` | `auto_approve_send_message` | `--no-auto-approve-send-message` |
| `--memory-only` | `memory_only` | `--no-memory-only` |
| `-m` / `--history-memory` | `history_memory` | `--no-history-memory`, `--no-m` |
| `--use-sqlite` | `use_sqlite` | `--no-use-sqlite` |
| `--dev` | `dev_tree` | `--no-dev` |
| `--clean` | `clean` | `--no-clean` |

### 2. Why `store_const` instead of `BooleanOptionalAction`

- `BooleanOptionalAction` (Python 3.9+) does not work with short aliases (`-sp`, `-tm`)
- `store_const` is explicit and version-independent
- Each `--no-*` is added as a separate `add_argument()` line

### 3. Override mechanism

```python
parser.add_argument("--no-skip-permissions", dest="skip_permissions",
                    action="store_const", const=False)
```

In `_explicit_cli_flags()` the same flags are registered with `default=argparse.SUPPRESS`.
If the user passes `--no-skip-permissions`, then `dest="skip_permissions"` lands
in the `explicit` set. `_is_override()` sees `dest in explicit` → returns `True` → the
saved value is overridden.

### 4. `_PARSER_DEFAULTS` and `build_launch_config`

Unchanged. `_PARSER_DEFAULTS["skip_permissions"]` stays `False`.
`build_launch_config()` reads `args.skip_permissions` — the flag has already taken
effect; the value is `False`.

## Usage Examples

```bash
# Started with the task manager
deep-worker myagent -tm

# Restart without the task manager
deep-worker --restart myagent --no-tm

# Restart without the dev tree (was --dev)
deep-worker --restart myagent --no-dev

# Restart with several flags disabled
deep-worker --restart myagent --no-skip-permissions --no-task-manager
```

## Files to Change

1. `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py`
   - `create_parser()`: +8 `--no-*` arguments
   - `_explicit_cli_flags()`: +8 arguments in the detector parser

2. `deep_worker/deep_worker_cli/tests/unit_tests/cli/test_no_flags.py`
   - New test file

## Out of Scope

- Changing `merge_restart_config()` — not required
- Changing `build_launch_config()` — not required
- Changing `_PARSER_DEFAULTS` — not required
- `--clean` is not inherited on restart (already implemented)
