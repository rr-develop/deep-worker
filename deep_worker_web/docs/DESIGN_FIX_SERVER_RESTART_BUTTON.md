# DESIGN: Fix web UI Restart Server button

## Problem

The `Restart Server` button calls `POST /api/server/restart`. The endpoint returns success, but the web server on `127.0.0.1:7887` can be left down.

Live reproduction on master (production tree, commit `7a9b046`):

1. `127.0.0.1:7887` was served by `python3 run.py` from the web server's checkout directory.
2. `curl -X POST http://127.0.0.1:7887/api/server/restart` returned `{"success": true, ...}`.
3. Afterwards `curl /health` failed to connect and no process listened on `:7887`.

Root cause in `app/routers/server.py::do_restart()`:

* It starts the replacement process immediately, while the current server still owns the port.
* It then kills `os.getppid()` (the parent shell/supervisor wrapper), not just the current server process.
* If the replacement loses the bind race, the old process exits and the parent wrapper is killed; nothing remains to recover the service.

Existing unit tests encoded the wrong behavior (`test_do_restart_kills_parent_process`). They must be changed to guard the real contract.

## Required behavior

Clicking the UI button must restart the same checkout's web server without manual host access:

* HTTP endpoint returns success quickly.
* The endpoint protects HTTP response delivery by scheduling `do_restart()` in a background thread; it must not call `do_restart()` inline.
* The background restart path preserves the short response-send gap before `os._exit(0)` (the current implementation's `time.sleep(0.1)` is acceptable), so the success response can flush before the process exits.
* Current web process exits only after that background-thread delay.
* Parent shell/supervisor is not killed.
* A detached helper starts the replacement only after the old process has exited / the port can be rebound.
* Replacement runs `run.py` from the same checkout directory as the current server (the production checkout for prod/master, the development checkout for dev).

## Proposed implementation

In `app/routers/server.py`:

1. Add small pure helpers:
   * `_server_run_py() -> Path`: resolve the current checkout's `deep_worker_web/run.py` from `__file__` (not from `os.getcwd()` alone).
   * `_restart_helper_code() -> str`: Python snippet executed in a detached helper.

2. `do_restart()`:
   * capture `current_pid = os.getpid()`;
   * resolve `run_py` and `cwd = run_py.parent`;
   * spawn detached helper:

     ```python
     subprocess.Popen(
         [sys.executable, "-c", helper_code, str(current_pid), str(run_py)],
         cwd=str(cwd),
         start_new_session=True,
         stdin=subprocess.DEVNULL,
         stdout=subprocess.DEVNULL,
         stderr=subprocess.DEVNULL,
     )
     ```

   * call `os._exit(0)` for the current server process.
   * Do not call `os.kill(os.getppid(), ...)`.

3. Detached helper logic:
   * wait for `current_pid` to disappear using `os.kill(pid, 0)` with a bounded timeout;
   * start `[sys.executable, str(run_py)]` in `cwd=run_py.parent`, `start_new_session=True`, stdio to DEVNULL;
   * exit.

This helper is a one-shot restart handoff, not a runtime polling loop. It exists only after the user explicitly requested restart and dies immediately after launching the replacement (or timing out). It introduces no periodic monitoring to the application.

## TDD plan

Unit tests (`tests/unit/test_server_restart.py`):

* RED: `do_restart()` must not call `os.kill(parent_pid, ...)`.
* RED: `do_restart()` must spawn a detached helper with `[sys.executable, "-c", helper_code, current_pid, run_py]`, cwd `run_py.parent`, and then `os._exit(0)`.
* Keep endpoint tests: `POST /api/server/restart` returns success and schedules `do_restart` in a background thread.

Integration test:

* Start a real dev web server process from the development checkout's `run.py` (port `7888` by config).
* Read `/health` and record PID from `/proc` / `ss`.
* POST `/api/server/restart`.
* Wait boundedly for `/health` to return again and verify the PID changed.
* Verify `127.0.0.1:7888` is still listening.

The integration test uses a bounded wait/retry around a user-triggered restart. It is not application polling and does not add runtime polling behavior.

## Manual/live verification

After commit + FF to master:

1. Restart the master service on `127.0.0.1:7887` with the new code.
2. In the live UI, click `Restart Server`.
3. Verify `/health` returns again and PID changed.
4. Verify the UI reload/reconnects without manual host restart.
