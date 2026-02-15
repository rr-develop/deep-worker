# Working Directory Fix for Deep-Worker

## Problem

When running `deep-worker agent_name` from the `/tmp` directory, the agent started with the working directory `/` (root) instead of `/tmp`.

**Example of the problem:**
```bash
$ cd /tmp
$ deep-worker test_22
Agent 'test_22' starting in background...

$ cco -ls
Active Claude sessions:
test_22              [dw]         /        # ← Wrong! Should be /tmp
```

## Cause

The `daemonize()` function in `agent_command.py` calls `os.chdir("/")` to detach from the parent process (standard daemonization practice). After that the current working directory is lost.

**Sequence:**
1. The user runs `deep-worker test_22` from `/tmp`
2. `main()` calls `daemonize()`
3. `daemonize()` performs `os.chdir("/")`
4. `run_agent()` calls `Path.cwd()`, which returns `/` instead of `/tmp`

## Solution (TDD approach)

### Step 1: Tests written (RED phase)

5 unit tests were created in `tests/unit_tests/cli/test_agent_command_working_dir.py`:
- `test_uses_current_directory_when_no_dir_arg` - using the current directory
- `test_uses_env_var_when_cwd_is_root` - using the environment variable after daemonization
- `test_dir_arg_takes_precedence` - priority of the --dir argument
- `test_main_sets_env_var_from_cwd` - setting the env var from cwd
- `test_main_sets_env_var_from_dir_arg` - setting the env var from --dir

### Step 2: Code fixed (GREEN phase)

The `main()` function in `agent_command.py` was changed (lines 203-209):

```python
# Save current working directory BEFORE daemonize changes it to /
# This allows run_agent() to use the original directory after daemonization
# Priority: --dir arg > current directory
if args.dir:
    os.environ["DEEP_WORKER_ORIGINAL_CWD"] = str(args.dir.resolve())
else:
    os.environ["DEEP_WORKER_ORIGINAL_CWD"] = os.getcwd()

# Daemonize before starting agent
daemonize(args.name)
```

The `run_agent()` function already had the logic for reading this environment variable (lines 84-93):

```python
# Determine working directory
# Priority: --dir arg > DEEP_WORKER_ORIGINAL_CWD env > current dir
working_dir = args.dir
if working_dir is None:
    # Get from environment variable set by main()
    original_cwd = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
    if original_cwd:
        working_dir = Path(original_cwd)
    else:
        working_dir = Path.cwd()
```

### Step 3: Verification (GREEN phase)

```bash
$ cd deep_worker/deep_worker_cli
$ uv run --all-groups pytest tests/unit_tests/cli/test_agent_command_working_dir.py -v

5 passed ✓
```

```bash
$ uv run --all-groups pytest tests/unit_tests/cli/ -v

32 passed ✓
```

## Working directory resolution priority

1. **--dir argument** (if given explicitly)
2. **DEEP_WORKER_ORIGINAL_CWD** (set by main() before daemonization)
3. **Path.cwd()** (current directory, fallback)

## Result

The agent now correctly uses the directory it was started from:

```bash
$ cd /tmp
$ deep-worker test_22
Agent 'test_22' starting in background...

$ cco -ls
Active Claude sessions:
test_22              [dw]         /tmp     # ✓ Correct!
```

## Commit

```
commit 9a73f65
fix: Preserve working directory when launching deep-worker agent

- Added 5 unit tests for working directory handling
- Modified main() to save cwd to DEEP_WORKER_ORIGINAL_CWD before daemonize()
- All 32 CLI tests passing
```

## Related files

- `deep_worker_cli/cli/agent_command.py` - main() fixed
- `tests/unit_tests/cli/test_agent_command_working_dir.py` - new tests
