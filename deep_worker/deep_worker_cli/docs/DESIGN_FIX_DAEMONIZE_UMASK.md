# DESIGN: fix agent file/dir permissions — `daemonize()` must not zero the umask

## Problem

Files and directories that the agent creates "the ordinary way" (via the
`execute` tool: `touch`, `mkdir`, shell redirects; via `write_file`; any
child processes) get permissions **666** (files) and **777** (directories)
instead of the normal user-level **644 / 755**.

### Root cause

`deep_worker_cli/cli/agent_command.py::daemonize()` performs the classic
textbook daemonization recipe, which includes the line:

```python
os.umask(0)
```

`os.umask(0)` sets the daemon process's umask to `0000`. The umask is
inherited by all descendants (child processes of the `execute` tool) and is
applied by the kernel on any `open(O_CREAT)` / `mkdir` without an explicit
mode:

* file: `0666 & ~umask` → with umask `0000` = **0666**;
* directory: `0777 & ~umask` → with umask `0000` = **0777**.

Empirically confirmed in production: the agent process's
`/proc/<pid>/status` shows `Umask: 0000`; `touch` yields 666, `mkdir`
yields 777.

The agent's private internal paths (session, checkpoints, registry) do NOT
suffer from this — they are protected by explicit `os.chmod(..., 0o600/0o700)`
in `storage_paths.py` (`ensure_private_dir`/`ensure_private_file`) and in
`registry.py`. The problem concerns two categories:

1. **User working artifacts** created by the agent via the `execute` tool /
   `write_file` — their mode is determined precisely by the umask.
2. **Internal non-private mkdir** calls that do not go through
   `ensure_private_dir` and have no explicit `mode`. Example from a
   production state, independently confirmed by the reviewer: sub-agent
   workdirs (named `<agent>:sub_*`) under the per-agent data directory
   currently have mode `drwxrwxrwx` (777) — they are created with a plain
   `Path.mkdir(parents=True, exist_ok=True)` without an explicit mode, so
   they inherit the unmasked `0o777` from umask=0. The fix automatically
   repairs this too (777→755) — that is a bonus, not a side effect.

### Why `os.umask(0)` was there in the first place

It is a quote from the canonical "Unix double-fork daemon" recipe (Stevens).
In that recipe `umask(0)` is set so that a daemon creating files with
explicit modes (for example daemons writing pid files with an exact `0644`)
does not depend on the unpredictable umask of the inherited environment. For
our case this is harmful: our daemon runs an LLM agent that creates
user-facing files **without** an explicit mode and must behave exactly like
the user's interactive shell.

## User requirement

> The default behavior when creating folders and files must be the same as
> for the user in the shell. No 777, no 666.

Target behavior = the system default: **umask `022`** (`/etc/login.defs
UMASK 022`, the host's `login.defs` confirms `UMASK 022`) → files **644**,
directories **755**.

## Solution

Two related edits in `daemonize()`:

1. **Daemon umask**. Replace `os.umask(0)` with:

   ```python
   os.umask(DAEMON_UMASK)   # DAEMON_UMASK = 0o022
   ```

   Where `DAEMON_UMASK` is a module-level constant in `agent_command.py`
   with the value `0o022` (files 644, directories 755 — the system default
   from `login.defs`).

2. **log_dir explicit chmod**. `log_dir.mkdir(parents=True, exist_ok=True)`
   runs in the parent process **before** the daemon umask is set; it
   inherits the umask of the CALLER (`deep-worker --restart`, a systemd
   unit, the parent daemon), which in production is often `0` → the log
   directory becomes `0o777`. After the `mkdir` we explicitly do
   `log_dir.chmod(0o755)` (OSError is ignored — for the case of a foreign
   owner / read-only fs). This covers review nit #2 about "internal
   non-private mkdir" and fixes the real production symptom (the agent's
   log directory at 777).

### Why `022` specifically, rather than "read the user's current umask"

* `022` — the system default (`/etc/login.defs`) and the conventional value
  for interactive sessions; it is exactly what yields the requested 644/755.
* Reading the "live" umask at launch time is not reliable: the daemon often
  starts from an environment (systemd/tmux/CI) where the umask may already
  be non-standard; by the time `daemonize()` runs, the process has already
  gone through code that may itself have touched the umask. A fixed explicit
  constant is deterministic and matches the system default.
* This is NOT a contract/architecture change: private paths still force
  600/700 with explicit chmod regardless of the umask, so privacy does not
  weaken.

### Scope

* Exactly one line changes, plus one named constant is added.
* Private files (session/checkpoints/registry) — unchanged (they have
  explicit chmod).
* The process registry keeps its 0o666 (a deliberate multi-user contract,
  set by an explicit `os.chmod` in `registry.py`) — the fix does not touch it.

## TDD

### Unit (RED→GREEN)

`tests/unit_tests/test_daemonize_umask.py`:

1. **Behavior-based regression guard (not AST-substring)**: mock `os.umask`
   (intercept the actually-passed value), `os.fork` (the parent branch
   returns `>0` → `sys.exit` in the parent), `os.setsid`, `os.chdir`,
   `os._exit`, `sys.exit` and the minimal necessary set; call `daemonize()`
   exactly up to the first fork; assert that the first (and only) argument
   to `os.umask` is `0o022`. Catches `0`, `0o0`, and `000` alike — any
   regression to a zero umask fails.
2. The constant value `DAEMON_UMASK == 0o022`.

### Integration (no mocks)

`tests/integration_tests/test_agent_file_permissions.py`:

* Actually launch the agent on top of the existing integration fixture
  (`agent_fixture.py`), have it create a file (`touch f`) and a directory
  (`mkdir d`) in tmp via `execute` (in the daemonized process); then
  `os.stat` checks the mode is `0644` / `0755`.
* **Direct root guard**: before the file-permission checks, read
  `/proc/<agent_pid>/status` and assert `Umask: 0022`. This guarantees that
  a permission failure is not an FS flake but a consequence of the umask
  actually being set.
* **internal-mkdir coverage**: assert that a non-private directory created
  by the agent itself (e.g. its `<agent>:sub_*` sub-agent workdirs under
  the per-agent data directory on the first spawn of a sub-agent; or, more
  simply, the agent's own log directory created by `daemonize()`) has mode
  `0755`, not `0777`.
* The test is skipped if the host's `login.defs`/environment gives a
  non-standard umask (portability), but on our host (dev) it genuinely
  passes.

## Documentation update

* `IMPLEMENTATION_NOTES.md` (provider docs) — a short note about
  the daemon's umask policy.
* A comment in the code itself at the `os.umask(...)` call site explaining
  why `022` and not `0`.
